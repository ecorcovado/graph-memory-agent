"""
Zeek Adapter for GraphMemory Agent.
Extracts normalized network security events from Zeek (Bro) logs.
Supports conn.log, dns.log, notice.log, and http.log in both JSONL and native TSV formats,
with automatic conversion of numeric Unix epoch timestamps to ISO-8601 UTC.
"""

import json
import re
from typing import Any, Dict, List, Optional
from core.adapters.base import BaseLogAdapter, NormalizedEvent
from core.graph_builder import normalize_entity


class ZeekAdapter(BaseLogAdapter):
    """Adapter for Zeek / Bro network security monitoring telemetry."""

    @staticmethod
    def parse_tsv_separator(line_str: str) -> str:
        """Extracts declared delimiter from a Zeek #separator line, e.g. '#separator \\x09' -> '\\t'."""
        line = line_str.strip()
        if line.startswith("#separator"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                raw_sep = parts[1].strip()
                if raw_sep.startswith(r"\x"):
                    try:
                        return chr(int(raw_sep[2:], 16))
                    except ValueError:
                        return "\t"
                return raw_sep
        return "\t"

    @staticmethod
    def parse_tsv_set_separator(line_str: str) -> str:
        """Extracts declared collection delimiter from a Zeek #set_separator line, e.g. '#set_separator \\x2c' or '#set_separator ,'."""
        line = line_str.strip()
        if line.startswith("#set_separator"):
            parts = line.split(maxsplit=1)
            if len(parts) > 1:
                raw_sep = parts[1].strip()
                if raw_sep.startswith(r"\x"):
                    try:
                        return chr(int(raw_sep[2:], 16))
                    except ValueError:
                        return ","
                return raw_sep
        return ","

    @staticmethod
    def parse_tsv_headers(line_str: str, separator: str = "\t") -> List[str]:
        """Extracts field names from a Zeek #fields metadata line respecting declared separator."""
        parts = line_str.strip().split(separator)
        if parts and parts[0].startswith("#fields"):
            parts = parts[1:]
        elif parts and parts[0].startswith("#"):
            parts = line_str.strip().split()
            if len(parts) > 1 and parts[0] == "#fields":
                parts = parts[1:]
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def parse_tsv_types(line_str: str, separator: str = "\t") -> List[str]:
        """Extracts data types from a Zeek #types metadata line respecting declared separator."""
        parts = line_str.strip().split(separator)
        if parts and parts[0].startswith("#types"):
            parts = parts[1:]
        elif parts and parts[0].startswith("#"):
            parts = line_str.strip().split()
            if len(parts) > 1 and parts[0] == "#types":
                parts = parts[1:]
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def unescape_zeek_string(val: str) -> str:
        """Unescapes Zeek hex escape sequences (e.g. \\x20 -> ' ', \\x09 -> '\\t')."""
        if not val or r"\x" not in val:
            return val
        return re.sub(r"\\x([0-9a-fA-F]{2})", lambda m: chr(int(m.group(1), 16)), val)

    @classmethod
    def parse_tsv_line(
        cls,
        fields: List[str],
        line_str: str,
        types: Optional[List[str]] = None,
        separator: str = "\t",
        set_separator: str = ",",
        unset_field: str = "-",
        empty_field: str = "(empty)",
    ) -> Dict[str, Any]:
        """Converts a separated Zeek log line into a normalized dictionary using field names and type metadata."""
        parts = line_str.split(separator)
        record: Dict[str, Any] = {}
        for idx, field_name in enumerate(fields):
            field_type = types[idx].lower() if types and idx < len(types) else ""
            if idx < len(parts):
                val = parts[idx].strip()
                if val == unset_field:
                    record[field_name] = None
                elif val == empty_field:
                    record[field_name] = [] if field_type.startswith("set") or field_type.startswith("vector") else ""
                else:
                    # Unescape string values
                    unescaped_val = cls.unescape_zeek_string(val)
                    # 1. Cast by Zeek #types metadata if available
                    if field_type in ["time", "interval", "double"]:
                        try:
                            record[field_name] = float(unescaped_val)
                        except ValueError:
                            record[field_name] = unescaped_val
                    elif field_type in ["port", "count", "int"]:
                        try:
                            record[field_name] = int(unescaped_val)
                        except ValueError:
                            record[field_name] = unescaped_val
                    elif field_type == "bool":
                        record[field_name] = unescaped_val.lower() in ("t", "true", "1")
                    elif field_type.startswith("set") or field_type.startswith("vector"):
                        # Split by declared set separator first, then unescape each element
                        # Prevents an escaped literal separator (e.g. \x2c) from acting as a collection delimiter
                        items = [cls.unescape_zeek_string(v.strip()) for v in val.split(set_separator) if v.strip()]
                        record[field_name] = items
                    else:
                        # 2. Heuristic fallback when #types is omitted
                        if field_name in ["id.orig_p", "id.resp_p", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
                            try:
                                record[field_name] = int(unescaped_val)
                            except ValueError:
                                record[field_name] = unescaped_val
                        elif field_name in ["ts", "duration"]:
                            try:
                                record[field_name] = float(unescaped_val)
                            except ValueError:
                                record[field_name] = unescaped_val
                        else:
                            record[field_name] = unescaped_val
            else:
                record[field_name] = None
        return record

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type in ["ZEEK_FLOW", "ZEEK_LOG", "ZEEK_CONN", "ZEEK_DNS", "ZEEK_NOTICE"]:
            return True
        f_lower = file_rel.lower()
        if "zeek" in f_lower or "conn.log" in f_lower or "dns.log" in f_lower or "notice.log" in f_lower or "http.log" in f_lower:
            return True
        if any(k in sample_record for k in ["id.orig_h", "id_orig_h", "conn_state", "history"]):
            return True
        if isinstance(sample_record.get("id"), dict) and "orig_h" in sample_record["id"]:
            return True
        return False

    def parse_record(
        self,
        record: Dict[str, Any],
        source_id: str,
        file_rel: str,
        line_idx: int,
        evidence_basis: str,
        event_time: str,
        ingested_at: str,
    ) -> List[NormalizedEvent]:
        events = []
        raw = json.dumps(record, default=str)
        purdue_level = record.get("purdue_level", 3.5)

        # Extract orig/resp addresses across flat, dotted, or nested Zeek schemas
        id_obj = record.get("id") if isinstance(record.get("id"), dict) else {}
        orig_h = record.get("id.orig_h") or record.get("id_orig_h") or record.get("src_ip") or id_obj.get("orig_h", "")
        orig_p = record.get("id.orig_p") or record.get("id_orig_p") or record.get("src_port") or id_obj.get("orig_p", "")
        resp_h = record.get("id.resp_h") or record.get("id_resp_h") or record.get("dst_ip") or id_obj.get("resp_h", "")
        resp_p = record.get("id.resp_p") or record.get("id_resp_p") or record.get("dst_port") or id_obj.get("resp_p", "")
        uid = record.get("uid") or ""
        proto = (str(record.get("proto") or "TCP")).upper()
        service = record.get("service") or ""

        # 1. Zeek DNS Log
        query = record.get("query")
        if query:
            answers = record.get("answers") or []
            if isinstance(answers, str):
                answers = [a.strip() for a in answers.split(",") if a.strip()]
            qtype = record.get("qtype_name") or record.get("qtype") or ""
            rcode = record.get("rcode_name") or record.get("rcode") or ""
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"zeek_dns_{line_idx}",
                    source_type="ZEEK_DNS",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="ZEEK_DNS",
                    subject=normalize_entity(str(orig_h or "unknown_host")),
                    predicate="queried_dns",
                    object_entity=normalize_entity(str(query)),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "uid": uid,
                        "query": query,
                        "qtype": qtype,
                        "answers": answers,
                        "rcode": rcode,
                        "server_ip": resp_h,
                    },
                )
            )
            return events

        # 2. Zeek Notice Log
        note = record.get("note") or record.get("notice")
        if note:
            msg = record.get("msg") or ""
            src = record.get("src") or orig_h or "unknown_source"
            dst = record.get("dst") or resp_h or ""
            actions = record.get("actions") or []
            if isinstance(actions, str):
                actions = [a.strip() for a in actions.split(",") if a.strip()]
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"zeek_notice_{line_idx}",
                    source_type="ZEEK_NOTICE",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="ZEEK_NOTICE",
                    subject=normalize_entity(str(src)),
                    predicate="triggered_notice",
                    object_entity=normalize_entity(str(note)),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "uid": uid,
                        "message": msg,
                        "target": dst,
                        "actions": actions,
                    },
                )
            )
            return events

        # 3. Zeek Conn Log (Flow Telemetry)
        if orig_h and resp_h:
            conn_state = record.get("conn_state") or ""
            duration = record.get("duration")
            orig_bytes = record.get("orig_bytes")
            resp_bytes = record.get("resp_bytes")
            history = record.get("history") or ""

            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"zeek_conn_{line_idx}",
                    source_type="ZEEK_CONN",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="NETWORK_FLOW",
                    subject=normalize_entity(str(orig_h)),
                    predicate="communicated_with",
                    object_entity=normalize_entity(str(resp_h)),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "uid": uid,
                        "source_port": orig_p,
                        "destination_port": resp_p,
                        "protocol": proto,
                        "service": service,
                        "conn_state": conn_state,
                        "duration": duration,
                        "orig_bytes": orig_bytes,
                        "resp_bytes": resp_bytes,
                        "history": history,
                    },
                )
            )

        return events
