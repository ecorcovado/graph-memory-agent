"""
Case Loader & Deterministic Structured Log Ingestion for Project Aegis (v0.4.1).
Extracts normalized events with strict source trust validation, path traversal defense,
per-source stable fact IDs, rich forensic attributes, and complete ingestion auditing.
"""

import hashlib
import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from core.graph_builder import (
    AtomicFact, KnowledgeGraph, normalize_entity, classify_relation_forensics,
    EVIDENCE_BASIS_TELEMETRY, EVIDENCE_BASIS_UNCLASSIFIED, TRUSTED_TELEMETRY_SOURCES,
)


@dataclass(frozen=True)
class NormalizedEvent:
    """Canonical event model for structured telemetry records."""
    source_id: str
    record_id: str
    source_type: str
    event_time: str
    ingested_at: str
    event_type: str
    subject: str
    predicate: str
    object_entity: str
    raw_record: str
    source_location: str
    evidence_basis: str
    purdue_level: Optional[float] = None
    attributes: Dict[str, Any] = field(default_factory=dict)
    parser_version: str = "1.0"


@dataclass
class IngestionReport:
    """Forensic audit summary of the case ingestion process."""
    case_id: str
    files_discovered: int
    files_processed: int
    source_records_read: int
    source_records_parsed: int
    source_records_rejected: int
    normalized_events_generated: int
    atomic_facts_generated: int
    graph_edges_generated: int
    missing_sources: List[str]
    source_hashes: Dict[str, str]
    is_complete: bool
    empty_sources: List[str] = field(default_factory=list)
    degraded_reason: Optional[str] = None

    @property
    def facts_produced(self) -> int:
        return self.atomic_facts_generated

    @property
    def edges_produced(self) -> int:
        return self.graph_edges_generated

    @property
    def records_read(self) -> int:
        return self.source_records_read

    @property
    def records_parsed(self) -> int:
        return self.source_records_parsed

    @property
    def records_rejected(self) -> int:
        return self.source_records_rejected


class StructuredLogParser:
    """Parses JSONL log lines across Purdue levels into NormalizedEvents."""

    @staticmethod
    def parse_line(
        data: dict,
        source_id: str,
        source_type: str,
        file_rel: str,
        line_idx: int,
    ) -> List[NormalizedEvent]:
        events = []
        ingested_at = datetime.now(timezone.utc).isoformat()
        raw = json.dumps(data)
        # Missing timestamps must be explicit "UNKNOWN" to prevent false timeline creation
        event_time = data.get("timestamp", "UNKNOWN")
        purdue_level = data.get("purdue_level")

        # Source trust validation: Only verified telemetry sources may assign DIRECT_TELEMETRY
        if source_type in TRUSTED_TELEMETRY_SOURCES:
            assigned_basis = EVIDENCE_BASIS_TELEMETRY
        else:
            assigned_basis = EVIDENCE_BASIS_UNCLASSIFIED

        # 1. Sysmon Event Logs (Level 4)
        if source_type == "SYSMON_EVENT":
            eid = data.get("event_id")
            host = normalize_entity(data.get("host", ""))
            user = data.get("user", "")
            if eid == 1:
                # Process creation -> host executed proc (retains user, parent, sha256, cmd)
                full_proc = data.get("process", "")
                proc = full_proc.split("\\")[-1] if full_proc else "unknown_process"
                cmd = data.get("command_line", "")
                parent = data.get("parent_process", "").split("\\")[-1]
                sha256 = data.get("sha256", "")
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"sysmon_eid1_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="PROCESS_CREATE",
                        subject=host,
                        predicate="executed",
                        object_entity=normalize_entity(proc),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={
                            "user": user,
                            "command_line": cmd,
                            "parent_process": parent,
                            "sha256": sha256,
                            "full_process_path": full_proc,
                        },
                    )
                )
            elif eid == 3:
                # Network connection -> host communicated_with dst_ip (retains port, protocol, user)
                dst_ip = data.get("destination_ip", "")
                dst_port = data.get("destination_port", "")
                proto = data.get("protocol", "TCP")
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"sysmon_eid3_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="NETWORK_CONNECT",
                        subject=host,
                        predicate="communicated_with",
                        object_entity=normalize_entity(dst_ip),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={
                            "user": user,
                            "source_ip": data.get("source_ip", ""),
                            "destination_port": dst_port,
                            "protocol": proto,
                        },
                    )
                )

        # 2. Windows Security Event Logs (Level 4 & Level 2)
        elif source_type == "WINDOWS_SECURITY":
            eid = data.get("event_id")
            host = normalize_entity(data.get("host", ""))
            src_host = normalize_entity(data.get("source_host", "") or data.get("workstation_name", ""))
            user = data.get("target_user_name", "") or data.get("subject_user_name", "") or data.get("user", "")
            logon_type = data.get("logon_type", "")
            proc = data.get("process")

            # Remote network logon
            if eid == 4624 and src_host and src_host != host:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"winsec_4624_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="LOGON_SUCCESS",
                        subject=src_host,
                        predicate="connected_to",
                        object_entity=host,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={
                            "user": user,
                            "logon_type": logon_type,
                            "source_ip": data.get("source_ip", ""),
                            "status": data.get("status", "0x0"),
                        },
                    )
                )
            # Local interactive logon
            elif eid == 4624 and user:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"winsec_logon_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="LOGON_LOCAL",
                        subject=normalize_entity(user),
                        predicate="logged_into",
                        object_entity=host,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={"user": user, "logon_type": logon_type},
                    )
                )
            # Privilege assignment
            elif eid == 4672 and user:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"winsec_priv_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="PRIVILEGE_ASSIGNMENT",
                        subject=normalize_entity(user),
                        predicate="authenticated_to",
                        object_entity=host,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={"privileges": data.get("privilege_list", [])},
                    )
                )
            # Process execution (Engineering workstation)
            elif proc:
                proc_name = proc.split("\\")[-1]
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"ws_exec_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="PROCESS_EXECUTION",
                        subject=host,
                        predicate="executed",
                        object_entity=normalize_entity(proc_name),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=purdue_level,
                        attributes={
                            "user": user,
                            "opened_project": data.get("opened_project", ""),
                            "full_path": proc,
                        },
                    )
                )

        # 3. Industrial DMZ Firewall & Remote Access Gateway (Level 3.5)
        elif source_type in ["FIREWALL_LOG", "REMOTE_ACCESS"]:
            is_allow = data.get("action") == "ALLOW"
            is_auth_success = data.get("authentication_result") == "SUCCESS"
            if is_allow or is_auth_success:
                src = normalize_entity(data.get("src_host", "") or data.get("src_ip", "") or data.get("client_ip", ""))
                dst = normalize_entity(data.get("dst_host", "") or data.get("dst_ip", "") or data.get("target_host", ""))
                rule_id = data.get("rule_id", "") or data.get("session_id", "")
                dst_port = data.get("dst_port", "") or ("3389" if data.get("protocol") == "RDP" else "")
                proto = data.get("protocol", "TCP")
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"gw_conn_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="FIREWALL_ALLOW",
                        subject=src,
                        predicate="connected_to",
                        object_entity=dst,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=3.5,
                        attributes={
                            "rule_id": rule_id,
                            "destination_port": dst_port,
                            "protocol": proto,
                            "account": data.get("account", ""),
                            "source_zone": data.get("src_zone", ""),
                            "destination_zone": data.get("dst_zone", ""),
                        },
                    )
                )

        # 4. Historian Mirror (Level 3.5 SQL tag query export)
        elif source_type == "HISTORIAN_MIRROR":
            host = normalize_entity(data.get("host", "dmz-hist-mirror"))
            account = data.get("account", "")
            query = data.get("query", "")
            recs = data.get("records_returned", 0)
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"hist_mirror_{line_idx}",
                    source_type=source_type,
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="SQL_QUERY_EXPORT",
                    subject=host,
                    predicate="connected_to",
                    object_entity="plant_tags",
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=assigned_basis,
                    purdue_level=3.5,
                    attributes={
                        "account": account,
                        "query": query,
                        "records_returned": recs,
                        "client_ip": data.get("client_ip", ""),
                    },
                )
            )

        # 5. Plant Historian Audit (Level 3 Operations replication sync)
        elif source_type == "HISTORIAN_AUDIT":
            peer = normalize_entity(data.get("peer_node", "dmz-hist-mirror"))
            host = normalize_entity(data.get("host", "l3-hist-srv01"))
            tags = data.get("transferred_tags", [])
            # Connects DMZ mirror to Level 3 historian along replication conduit
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"hist_audit_{line_idx}",
                    source_type=source_type,
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="TAG_REPLICATION_SYNC",
                    subject=peer,
                    predicate="connected_to",
                    object_entity=host,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=assigned_basis,
                    purdue_level=3.0,
                    attributes={"service": data.get("service", ""), "transferred_tags": tags},
                )
            )

        # 6. SCADA Ignition Gateway (Level 2 Supervisory)
        elif source_type == "SCADA_AUDIT":
            src = normalize_entity(data.get("source_host", "") or data.get("source_ip", ""))
            gw = normalize_entity(data.get("gateway", "scada-ignition-gw01"))
            account = data.get("account", "")
            roles = data.get("roles", [])
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"scada_session_{line_idx}",
                    source_type=source_type,
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="GATEWAY_SESSION",
                    subject=src,
                    predicate="authenticated_to",
                    object_entity=gw,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=assigned_basis,
                    purdue_level=2.0,
                    attributes={"account": account, "roles": roles, "status": data.get("status", "")},
                )
            )

        # 7. Controller CIP Audits (Level 1)
        elif source_type == "CONTROLLER_AUDIT":
            src = normalize_entity(data.get("client_host", "") or data.get("client_ip", ""))
            controller = normalize_entity(data.get("controller", "plc-ballast-01"))
            routine = data.get("routine_modified", "")
            prior_sig = data.get("prior_signature", "")
            new_sig = data.get("new_signature", "")
            service = data.get("cip_service", "")
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"controller_cip_{line_idx}",
                    source_type=source_type,
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="CIP_PROGRAM_DOWNLOAD",
                    subject=src,
                    predicate="modified_logic",
                    object_entity=controller,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=assigned_basis,
                    purdue_level=1.0,
                    attributes={
                        "cip_service": service,
                        "routine_modified": routine,
                        "prior_signature": prior_sig,
                        "new_signature": new_sig,
                        "model": data.get("model", ""),
                        "ip_address": data.get("ip_address", ""),
                    },
                )
            )

        # 8. Physical Process Telemetry (Level 0)
        elif source_type == "PROCESS_TELEMETRY":
            tag = normalize_entity(data.get("tag_id", ""))
            desc = data.get("description", "")
            pressure = data.get("value_bar")
            interlock = data.get("interlock_status")
            cmd_state = data.get("commanded_state")
            act_state = data.get("actual_state")

            if pressure is not None:
                obs_id = f"obs-{tag}-001" if tag else f"obs-pressure-{line_idx:03d}"
                # 1. tag -> has_observation -> obs_id
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_obs_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="PROCESS_OBSERVATION",
                        subject=tag,
                        predicate="has_observation",
                        object_entity=obs_id,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=0.0,
                        attributes={
                            "description": desc,
                            "normal_range": data.get("normal_range", ""),
                        },
                    )
                )
                # 2. obs_id -> measured_value -> f"{pressure}_bar"
                pressure_val = f"{pressure}_bar"
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_measurement_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="PRESSURE_MEASUREMENT",
                        subject=obs_id,
                        predicate="measured_value",
                        object_entity=normalize_entity(pressure_val),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=0.0,
                        attributes={
                            "value_bar": pressure,
                            "high_alarm_threshold": data.get("high_alarm_threshold"),
                            "trip_state": data.get("trip_state"),
                        },
                    )
                )

            if interlock or cmd_state or act_state:
                obs_valve_id = f"obs-{tag}-001" if tag else f"obs-valve-{line_idx:03d}"
                # 1. tag -> has_observation -> obs_valve_id
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_valve_obs_{line_idx}",
                        source_type=source_type,
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="VALVE_OBSERVATION",
                        subject=tag,
                        predicate="has_observation",
                        object_entity=obs_valve_id,
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=assigned_basis,
                        purdue_level=0.0,
                        attributes={"description": desc},
                    )
                )
                if cmd_state:
                    events.append(
                        NormalizedEvent(
                            source_id=source_id,
                            record_id=f"process_cmd_{line_idx}",
                            source_type=source_type,
                            event_time=event_time,
                            ingested_at=ingested_at,
                            event_type="COMMANDED_STATE",
                            subject=obs_valve_id,
                            predicate="commanded_state",
                            object_entity=normalize_entity(cmd_state),
                            raw_record=raw,
                            source_location=f"{file_rel}:L{line_idx}",
                            evidence_basis=assigned_basis,
                            purdue_level=0.0,
                            attributes={"commanded_state": cmd_state},
                        )
                    )
                if act_state:
                    events.append(
                        NormalizedEvent(
                            source_id=source_id,
                            record_id=f"process_act_{line_idx}",
                            source_type=source_type,
                            event_time=event_time,
                            ingested_at=ingested_at,
                            event_type="ACTUAL_STATE",
                            subject=obs_valve_id,
                            predicate="actual_state",
                            object_entity=normalize_entity(act_state),
                            raw_record=raw,
                            source_location=f"{file_rel}:L{line_idx}",
                            evidence_basis=assigned_basis,
                            purdue_level=0.0,
                            attributes={"actual_state": act_state},
                        )
                    )
                if interlock:
                    events.append(
                        NormalizedEvent(
                            source_id=source_id,
                            record_id=f"process_int_{line_idx}",
                            source_type=source_type,
                            event_time=event_time,
                            ingested_at=ingested_at,
                            event_type="INTERLOCK_STATUS",
                            subject=obs_valve_id,
                            predicate="interlock_status",
                            object_entity=normalize_entity(interlock),
                            raw_record=raw,
                            source_location=f"{file_rel}:L{line_idx}",
                            evidence_basis=assigned_basis,
                            purdue_level=0.0,
                            attributes={"interlock_status": interlock},
                        )
                    )

        return events


class CaseLoader:
    """
    Ingests an entire Project Aegis structured case bundle, hashes source documents,
    protects against path traversal, extracts normalized events, and builds a verified MultiDiGraph.
    """

    SOURCE_TYPE_MAPPING = {
        "sysmon.jsonl": "SYSMON_EVENT",
        "windows_security.jsonl": "WINDOWS_SECURITY",
        "firewall.jsonl": "FIREWALL_LOG",
        "remote_access.jsonl": "FIREWALL_LOG",
        "historian_mirror.jsonl": "HISTORIAN_MIRROR",
        "historian_audit.jsonl": "HISTORIAN_AUDIT",
        "scada_audit.jsonl": "SCADA_AUDIT",
        "engineering_workstation.jsonl": "WINDOWS_SECURITY",
        "controller_audit.jsonl": "CONTROLLER_AUDIT",
        "process_events.jsonl": "PROCESS_TELEMETRY",
    }

    def __init__(self, case_dir: str):
        self.case_dir = os.path.realpath(case_dir)
        self.manifest_path = os.path.join(self.case_dir, "case_manifest.json")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Case manifest missing at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

    def load_case(self, exclude_sources: Optional[List[str]] = None) -> KnowledgeGraph:
        """
        Executes full deterministic ingestion of all case files in the manifest.
        Validates path confinement, tracks complete ingestion audit metrics, and builds MultiDiGraph.
        """
        exclude_sources = set(exclude_sources or [])
        kg = KnowledgeGraph()
        all_events: List[NormalizedEvent] = []

        files_discovered = 0
        files_processed = 0
        source_records_read = 0
        source_records_parsed = 0
        source_records_rejected = 0
        missing_sources: List[str] = []
        empty_sources: List[str] = []

        print(f"[*] Ingesting Case: {self.manifest.get('title')} ({self.manifest.get('case_id')})")
        evidence_sources = self.manifest.get("evidence_sources", {})

        for category, file_list in evidence_sources.items():
            for rel_file in file_list:
                files_discovered += 1
                file_name = os.path.basename(rel_file)

                # Path Traversal Defense: Ensure resolved path strictly stays within case directory
                canonical_case = self.case_dir
                abs_file = os.path.realpath(os.path.join(self.case_dir, rel_file))
                if not (abs_file.startswith(canonical_case + os.sep) or abs_file == canonical_case):
                    raise ValueError(f"Security Violation: Path traversal detected in '{rel_file}' outside case root")

                if file_name in exclude_sources:
                    print(f"    [SKIP] Excluded for fail-closed benchmark: {rel_file}")
                    missing_sources.append(rel_file)
                    continue

                if not os.path.exists(abs_file):
                    print(f"    [WARN] File missing: {abs_file}")
                    missing_sources.append(rel_file)
                    continue

                # 1. Compute full SHA-256 digest from byte snapshot
                with open(abs_file, "rb") as bf:
                    file_bytes = bf.read()
                full_sha = hashlib.sha256(file_bytes).hexdigest()
                source_id = full_sha[:32]
                kg.source_hashes[source_id] = full_sha
                files_processed += 1

                # 2. Determine source type
                source_type = self.SOURCE_TYPE_MAPPING.get(file_name, "UNCLASSIFIED")

                # 3. Parse directly from the hashed byte snapshot (Byte-Preserving Ingestion)
                file_text = file_bytes.decode("utf-8", errors="replace")
                raw_lines = [l.strip() for l in file_text.splitlines() if l.strip()]

                if not raw_lines:
                    print(f"    [WARN] Empty source file detected: {rel_file}")
                    empty_sources.append(rel_file)

                for line_idx, line_str in enumerate(raw_lines, 1):
                    source_records_read += 1
                    try:
                        record = json.loads(line_str)
                        events = StructuredLogParser.parse_line(
                            record,
                            source_id=source_id,
                            source_type=source_type,
                            file_rel=rel_file,
                            line_idx=line_idx,
                        )
                        if events:
                            source_records_parsed += 1
                            all_events.extend(events)
                            # Store raw record in vault
                            for ev in events:
                                kg.raw_record_vault[(ev.source_id, ev.record_id)] = record
                        else:
                            source_records_rejected += 1
                    except json.JSONDecodeError:
                        source_records_rejected += 1
                        continue

        print(f"    [+] Normalized {len(all_events)} structured telemetry events across Purdue zones.")

        # 4. Generate deterministic, stable AtomicFacts (per-source line numbering independent of manifest order)
        for ev in all_events:
            line_str = ev.source_location.split(":L")[-1]
            fact_id = f"{ev.source_id}_R{line_str}_{ev.record_id}"

            # Assemble rich forensic text summary preserving specific attributes
            attr_parts = []
            for k, v in ev.attributes.items():
                if v:
                    attr_parts.append(f"{k}: {v}")
            attr_text = f" ({', '.join(attr_parts)})" if attr_parts else ""
            fact_text = f"[{ev.event_time}] {ev.subject} {ev.predicate} {ev.object_entity}{attr_text}"

            tier, basis = classify_relation_forensics(
                ev.predicate,
                source_type=ev.source_type,
                evidence_basis=ev.evidence_basis,
            )

            raw_dict = kg.raw_record_vault.get((ev.source_id, ev.record_id), {})

            atomic_fact = AtomicFact(
                fact_id=fact_id,
                chunk_id=0,
                text=fact_text,
                source_id=ev.source_id,
                source_record_id=ev.record_id,
                source_location=ev.source_location,
                event_time=ev.event_time,
                purdue_level=ev.purdue_level,
                subject=ev.subject,
                relation=ev.predicate,
                object_entity=ev.object_entity,
                key_elements=[ev.subject, ev.object_entity],
                evidence_basis=basis,
                source_type=ev.source_type,
                confidence="HIGH",
                attributes=ev.attributes,
                raw_record=raw_dict,
            )
            kg.atomic_facts[fact_id] = atomic_fact

            # Index nodes and store asset purdue level metadata
            for elem in [ev.subject, ev.object_entity]:
                if not elem:
                    continue
                if elem not in kg.node_to_facts:
                    kg.node_to_facts[elem] = []
                kg.node_to_facts[elem].append(fact_id)
                if not kg.graph.has_node(elem):
                    kg.graph.add_node(elem, label=elem, purdue_level=ev.purdue_level)

            # Add directed edge with strict telemetry classification
            if ev.subject and ev.object_entity and ev.subject != ev.object_entity:
                kg.graph.add_edge(
                    ev.subject,
                    ev.object_entity,
                    relation=ev.predicate,
                    classification=tier,
                    evidence_basis=basis,
                    source_type=ev.source_type,
                    fact_id=fact_id,
                )

        # 5. Compile Ingestion Completeness Report
        is_complete = (len(missing_sources) == 0 and len(empty_sources) == 0 and source_records_rejected == 0)
        degraded_reason = None
        if not is_complete:
            reasons = []
            if missing_sources:
                reasons.append(f"Missing sources: {missing_sources}")
            if empty_sources:
                reasons.append(f"Empty sources: {empty_sources}")
            if source_records_rejected > 0:
                reasons.append(f"Rejected records: {source_records_rejected}")
            degraded_reason = "; ".join(reasons)

        report = IngestionReport(
            case_id=self.manifest.get("case_id", "UNKNOWN"),
            files_discovered=files_discovered,
            files_processed=files_processed,
            source_records_read=source_records_read,
            source_records_parsed=source_records_parsed,
            source_records_rejected=source_records_rejected,
            normalized_events_generated=len(all_events),
            atomic_facts_generated=len(kg.atomic_facts),
            graph_edges_generated=kg.graph.number_of_edges(),
            missing_sources=missing_sources,
            source_hashes=dict(kg.source_hashes),
            is_complete=is_complete,
            empty_sources=empty_sources,
            degraded_reason=degraded_reason,
        )
        kg.ingestion_report = report

        print(f"[+] KnowledgeGraph populated from Case: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges, {len(kg.atomic_facts)} atomic facts.")
        print(f"    -> Ingestion Status: {'COMPLETE (All sources verified)' if is_complete else f'INCOMPLETE (Missing: {missing_sources})'}")
        return kg
