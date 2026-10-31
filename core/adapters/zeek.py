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
    def parse_tsv_headers(line_str: str) -> List[str]:
        """Extracts field names from a Zeek #fields metadata line."""
        parts = line_str.strip().split("\t")
        if parts and parts[0].startswith("#fields"):
            parts = parts[1:]
        elif parts and parts[0].startswith("#"):
            parts = line_str.strip().split()
            if len(parts) > 1 and parts[0] == "#fields":
                parts = parts[1:]
        return [p.strip() for p in parts if p.strip()]

    @staticmethod
    def parse_tsv_line(fields: List[str], line_str: str) -> Dict[str, Any]:
        """Converts a tab-separated Zeek log line into a normalized dictionary."""
        parts = line_str.split("\t")
        record: Dict[str, Any] = {}
        for idx, field_name in enumerate(fields):
            if idx < len(parts):
                val = parts[idx].strip()
                if val == "-" or val == "(empty)":
                    record[field_name] = None
                else:
                    # Attempt numeric cast for ports and counts
                    if field_name in ["id.orig_p", "id.resp_p", "orig_bytes", "resp_bytes", "orig_pkts", "resp_pkts"]:
                        try:
                            record[field_name] = int(val)
                        except ValueError:
                            record[field_name] = val
                    elif field_name in ["ts", "duration"]:
                        try:
                            record[field_name] = float(val)
                        except ValueError:
                            record[field_name] = val
                    else:
                        record[field_name] = val
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
