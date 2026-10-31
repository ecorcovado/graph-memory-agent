"""
Windows Security Adapter for GraphMemory Agent.
Extracts normalized authentication, logon, and process execution events from
Windows Security Event logs (Event IDs 4624, 4672, and Workstation process audits).
"""

import json
from typing import Any, Dict, List
from core.adapters.base import BaseLogAdapter, NormalizedEvent
from core.graph_builder import normalize_entity


class WindowsSecurityAdapter(BaseLogAdapter):
    """Adapter for Windows Security event telemetry."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type == "WINDOWS_SECURITY":
            return True
        f_lower = file_rel.lower()
        if "windows_security" in f_lower or "engineering_workstation" in f_lower:
            return True
        eid = sample_record.get("event_id") or sample_record.get("EventID")
        if eid in [4624, 4625, 4672, 4688]:
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
        src_host = normalize_entity(record.get("source_host") or record.get("workstation_name") or record.get("WorkstationName") or "")
        user = record.get("target_user_name") or record.get("subject_user_name") or record.get("user") or record.get("TargetUserName") or ""
        logon_type = record.get("logon_type") or record.get("LogonType") or ""
        proc = record.get("process")
        purdue_level = record.get("purdue_level")

        # 1. Remote Network Logon (EID 4624 with distinct source host)
        if eid == 4624 and src_host and src_host != host:
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"winsec_4624_{line_idx}",
                    source_type="WINDOWS_SECURITY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="LOGON_SUCCESS",
                    subject=src_host,
                    predicate="connected_to",
                    object_entity=host,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "logon_type": logon_type,
                        "source_ip": record.get("source_ip", ""),
                        "status": record.get("status", "0x0"),
                    },
                )
            )

        # 2. Local Interactive Logon
        elif eid == 4624 and user:
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"winsec_logon_{line_idx}",
                    source_type="WINDOWS_SECURITY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="LOGON_LOCAL",
                    subject=normalize_entity(user),
                    predicate="logged_into",
                    object_entity=host,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={"user": user, "logon_type": logon_type},
                )
            )

        # 3. Privilege Assignment (EID 4672)
        elif eid == 4672 and user:
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"winsec_priv_{line_idx}",
                    source_type="WINDOWS_SECURITY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="PRIVILEGE_ASSIGNMENT",
                    subject=normalize_entity(user),
                    predicate="authenticated_to",
                    object_entity=host,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={"privileges": record.get("privilege_list", [])},
                )
            )

        # 4. Workstation Process Execution
        elif proc:
            proc_name = proc.split("\\")[-1]
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"ws_exec_{line_idx}",
                    source_type="WINDOWS_SECURITY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="PROCESS_EXECUTION",
                    subject=host,
                    predicate="executed",
                    object_entity=normalize_entity(proc_name),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=purdue_level,
                    attributes={
                        "user": user,
                        "opened_project": record.get("opened_project", ""),
                        "full_path": proc,
                    },
                )
            )

        return events
