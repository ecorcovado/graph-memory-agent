"""
Sysmon Adapter for GraphMemory Agent.
Extracts normalized endpoint security events from Microsoft Sysmon logs (JSONL/EVTX).
Supports Event IDs 1 (Process Create), 3 (Network Connect), 11 (File Create),
12-14 (Registry Activity), and 22 (DNS Query).
"""

import json
from typing import Any, Dict, List
from core.adapters.base import BaseLogAdapter, NormalizedEvent
from core.graph_builder import normalize_entity


class SysmonAdapter(BaseLogAdapter):
    """Adapter for Microsoft Sysmon event telemetry."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type == "SYSMON_EVENT":
            return True
        if "sysmon" in file_rel.lower():
            return True
        eid = sample_record.get("event_id") or sample_record.get("EventID")
        if eid in [1, 3, 11, 12, 13, 14, 22] and ("process" in sample_record or "Image" in sample_record):
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
        raw = json.dumps(record)
        eid = record.get("event_id") or record.get("EventID")
        host = normalize_entity(record.get("host") or record.get("Computer") or "")
        user = record.get("user") or record.get("User") or ""
        purdue_level = record.get("purdue_level", 4.0)

        # 1. Event ID 1: Process Creation
        if eid == 1:
            full_proc = record.get("process") or record.get("Image") or ""
            proc = full_proc.split("\\")[-1] if full_proc else "unknown_process"
            cmd = record.get("command_line") or record.get("CommandLine") or ""
            parent_full = record.get("parent_process") or record.get("ParentImage") or ""
            parent = parent_full.split("\\")[-1] if parent_full else ""
            sha256 = record.get("sha256") or record.get("Hashes", {}).get("SHA256", "") if isinstance(record.get("Hashes"), dict) else (record.get("sha256") or "")
            guid = record.get("process_guid") or record.get("ProcessGuid") or ""

            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"sysmon_eid1_{line_idx}",
                    source_type="SYSMON_EVENT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="PROCESS_CREATE",
                    subject=host,
                    predicate="executed",
                    object_entity=normalize_entity(proc),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "command_line": cmd,
                        "parent_process": parent,
                        "sha256": sha256,
                        "full_process_path": full_proc,
                        "process_guid": guid,
                    },
                )
            )

        # 2. Event ID 3: Network Connection
        elif eid == 3:
            dst_ip = record.get("destination_ip") or record.get("DestinationIp") or ""
            dst_port = record.get("destination_port") or record.get("DestinationPort") or ""
            proto = record.get("protocol") or record.get("Protocol") or "TCP"
            src_ip = record.get("source_ip") or record.get("SourceIp") or ""

            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"sysmon_eid3_{line_idx}",
                    source_type="SYSMON_EVENT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="NETWORK_CONNECT",
                    subject=host,
                    predicate="communicated_with",
                    object_entity=normalize_entity(dst_ip),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "source_ip": src_ip,
                        "destination_port": dst_port,
                        "protocol": proto,
                    },
                )
            )

        # 3. Event ID 11: File Creation
        elif eid == 11:
            target_file = record.get("target_filename") or record.get("TargetFilename") or ""
            target_name = target_file.split("\\")[-1] if target_file else "unknown_file"
            proc_full = record.get("process") or record.get("Image") or ""

            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"sysmon_eid11_{line_idx}",
                    source_type="SYSMON_EVENT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="FILE_CREATE",
                    subject=host,
                    predicate="created",
                    object_entity=normalize_entity(target_name),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "target_filename": target_file,
                        "process": proc_full,
                    },
                )
            )

        # 4. Event ID 12/13/14: Registry Activity
        elif eid in [12, 13, 14]:
            target_obj = record.get("target_object") or record.get("TargetObject") or ""
            details = record.get("details") or record.get("Details") or ""
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"sysmon_reg_{line_idx}",
                    source_type="SYSMON_EVENT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="REGISTRY_MODIFY",
                    subject=host,
                    predicate="modified_registry",
                    object_entity=normalize_entity(target_obj),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "details": details,
                        "event_id": eid,
                    },
                )
            )

        # 5. Event ID 22: DNS Query
        elif eid == 22:
            query = record.get("query_name") or record.get("QueryName") or ""
            results = record.get("query_results") or record.get("QueryResults") or ""
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"sysmon_dns_{line_idx}",
                    source_type="SYSMON_EVENT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="DNS_QUERY",
                    subject=host,
                    predicate="queried_dns",
                    object_entity=normalize_entity(query),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "query_name": query,
                        "query_results": results,
                    },
                )
            )

        return events
