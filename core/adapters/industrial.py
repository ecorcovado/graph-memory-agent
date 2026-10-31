"""
Industrial OT/ICS Adapters for GraphMemory Agent.
Specialized adapters across Purdue zones:
- FirewallAdapter: Level 3.5 perimeter and remote access gateway telemetry
- HistorianAdapter: Level 3.5 tag store queries and Level 3 replication sync
- ControllerAuditAdapter: Level 1 CIP controller program downloads (PLC logic)
- ProcessTelemetryAdapter: Level 0 sensor observations and actuator interlocks
"""

import json
from typing import Any, Dict, List
from core.adapters.base import BaseLogAdapter, NormalizedEvent
from core.graph_builder import normalize_entity


class FirewallAdapter(BaseLogAdapter):
    """Adapter for Industrial DMZ Firewall and Remote Access Gateway telemetry."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type in ["FIREWALL_LOG", "REMOTE_ACCESS"]:
            return True
        f_lower = file_rel.lower()
        if "firewall" in f_lower or "remote_access" in f_lower:
            return True
        if "rule_id" in sample_record or "session_id" in sample_record or "src_zone" in sample_record:
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
        is_allow = record.get("action") == "ALLOW"
        is_auth_success = record.get("authentication_result") == "SUCCESS"

        if is_allow or is_auth_success:
            src = normalize_entity(record.get("src_host", "") or record.get("src_ip", "") or record.get("client_ip", ""))
            dst = normalize_entity(record.get("dst_host", "") or record.get("dst_ip", "") or record.get("target_host", ""))
            rule_id = record.get("rule_id", "") or record.get("session_id", "")
            dst_port = record.get("dst_port", "") or ("3389" if record.get("protocol") == "RDP" else "")
            proto = record.get("protocol", "TCP")

            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"gw_conn_{line_idx}",
                    source_type="FIREWALL_LOG",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="FIREWALL_ALLOW",
                    subject=src,
                    predicate="connected_to",
                    object_entity=dst,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=3.5,
                    attributes={
                        "rule_id": rule_id,
                        "destination_port": dst_port,
                        "protocol": proto,
                        "account": record.get("account", ""),
                        "source_zone": record.get("src_zone", ""),
                        "destination_zone": record.get("dst_zone", ""),
                    },
                )
            )

        return events


class HistorianAdapter(BaseLogAdapter):
    """Adapter for Plant Historian and SCADA Gateway audits (DMZ Mirror queries, Level 3 Replication, SCADA sessions)."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type in ["HISTORIAN_MIRROR", "HISTORIAN_AUDIT", "SCADA_AUDIT"]:
            return True
        f_lower = file_rel.lower()
        if "historian" in f_lower or "scada" in f_lower:
            return True
        if "transferred_tags" in sample_record or "queried_tag_store" in sample_record or "query" in sample_record or "gateway" in sample_record:
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

        # 1. DMZ Historian Mirror (Level 3.5 SQL tag query export)
        if "query" in record or "records_returned" in record:
            host = normalize_entity(record.get("host", "dmz-hist-mirror"))
            account = record.get("account", "")
            query = record.get("query", "")
            recs = record.get("records_returned", 0)
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"hist_mirror_{line_idx}",
                    source_type="HISTORIAN_MIRROR",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="SQL_QUERY_EXPORT",
                    subject=host,
                    predicate="queried_tag_store",
                    object_entity="plant_tags",
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=3.5,
                    attributes={
                        "account": account,
                        "query": query,
                        "records_returned": recs,
                        "client_ip": record.get("client_ip", ""),
                    },
                )
            )

        # 2. Plant Historian Replication Audit (Level 3 Operations)
        elif "peer_node" in record or "transferred_tags" in record:
            peer = normalize_entity(record.get("peer_node", "dmz-hist-mirror"))
            host = normalize_entity(record.get("host", "l3-hist-srv01"))
            tags = record.get("transferred_tags", [])
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"hist_audit_{line_idx}",
                    source_type="HISTORIAN_AUDIT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="TAG_REPLICATION_SYNC",
                    subject=peer,
                    predicate="replicated_tags_to",
                    object_entity=host,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=3.0,
                    attributes={"service": record.get("service", ""), "transferred_tags": tags},
                )
            )

        # 3. SCADA Gateway Session (Level 2)
        elif "gateway" in record or "roles" in record:
            src = normalize_entity(record.get("source_host", "") or record.get("source_ip", ""))
            gw = normalize_entity(record.get("gateway", "scada-ignition-gw01"))
            account = record.get("account", "")
            roles = record.get("roles", [])
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"scada_session_{line_idx}",
                    source_type="SCADA_AUDIT",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="GATEWAY_SESSION",
                    subject=src,
                    predicate="authenticated_to",
                    object_entity=gw,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=2.0,
                    attributes={"account": account, "roles": roles, "status": record.get("status", "")},
                )
            )

        return events


class ControllerAuditAdapter(BaseLogAdapter):
    """Adapter for Level 1 CIP Controller audit logs (modified_logic on PLC)."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type == "CONTROLLER_AUDIT":
            return True
        f_lower = file_rel.lower()
        if "controller" in f_lower or "plc" in f_lower:
            return True
        if "cip_service" in sample_record or "routine_modified" in sample_record:
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
        src = normalize_entity(record.get("client_host", "") or record.get("client_ip", ""))
        controller = normalize_entity(record.get("controller", "plc-ballast-01"))
        routine = record.get("routine_modified", "")
        prior_sig = record.get("prior_signature", "")
        new_sig = record.get("new_signature", "")
        service = record.get("cip_service", "")

        events.append(
            NormalizedEvent(
                source_id=source_id,
                record_id=f"controller_cip_{line_idx}",
                source_type="CONTROLLER_AUDIT",
                event_time=event_time,
                ingested_at=ingested_at,
                event_type="CIP_PROGRAM_DOWNLOAD",
                subject=src,
                predicate="modified_logic",
                object_entity=controller,
                raw_record=raw,
                source_location=f"{file_rel}:L{line_idx}",
                evidence_basis=evidence_basis,
                purdue_level=1.0,
                attributes={
                    "cip_service": service,
                    "routine_modified": routine,
                    "prior_signature": prior_sig,
                    "new_signature": new_sig,
                    "model": record.get("model", ""),
                    "ip_address": record.get("ip_address", ""),
                },
            )
        )

        return events


class ProcessTelemetryAdapter(BaseLogAdapter):
    """Adapter for Level 0 Physical Process telemetry (sensors and actuators)."""

    def can_handle(self, source_type: str, file_rel: str, sample_record: Dict[str, Any]) -> bool:
        if source_type == "PROCESS_TELEMETRY":
            return True
        f_lower = file_rel.lower()
        if "process_events" in f_lower or "level0" in f_lower:
            return True
        if any(k in sample_record for k in ["tag_id", "value_bar", "interlock_status", "commanded_state"]):
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
        tag = normalize_entity(record.get("tag_id", ""))
        desc = record.get("description", "")
        pressure = record.get("value_bar")
        interlock = record.get("interlock_status")
        cmd_state = record.get("commanded_state")
        act_state = record.get("actual_state")

        # 1. Analog Measurement Observations (Pressure)
        if pressure is not None:
            obs_id = f"obs-{tag}-{source_id[:8]}-r{line_idx:04d}" if tag else f"obs-pressure-{source_id[:8]}-r{line_idx:04d}"
            # tag -> has_observation -> obs_id
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"process_obs_{line_idx}",
                    source_type="PROCESS_TELEMETRY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="PROCESS_OBSERVATION",
                    subject=tag,
                    predicate="has_observation",
                    object_entity=obs_id,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=0.0,
                    attributes={
                        "description": desc,
                        "normal_range": record.get("normal_range", ""),
                    },
                )
            )
            # obs_id -> measured_value -> {pressure}_bar
            pressure_val = f"{pressure}_bar"
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"process_measurement_{line_idx}",
                    source_type="PROCESS_TELEMETRY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="PRESSURE_MEASUREMENT",
                    subject=obs_id,
                    predicate="measured_value",
                    object_entity=normalize_entity(pressure_val),
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=0.0,
                    attributes={
                        "value_bar": pressure,
                        "high_alarm_threshold": record.get("high_alarm_threshold"),
                        "trip_state": record.get("trip_state"),
                    },
                )
            )

        # 2. Discrete Valve / Actuator Observations
        if interlock or cmd_state or act_state:
            obs_valve_id = f"obs-{tag}-{source_id[:8]}-r{line_idx:04d}" if tag else f"obs-valve-{source_id[:8]}-r{line_idx:04d}"
            events.append(
                NormalizedEvent(
                    source_id=source_id,
                    record_id=f"process_valve_obs_{line_idx}",
                    source_type="PROCESS_TELEMETRY",
                    event_time=event_time,
                    ingested_at=ingested_at,
                    event_type="VALVE_OBSERVATION",
                    subject=tag,
                    predicate="has_observation",
                    object_entity=obs_valve_id,
                    raw_record=raw,
                    source_location=f"{file_rel}:L{line_idx}",
                    evidence_basis=evidence_basis,
                    purdue_level=0.0,
                    attributes={"description": desc},
                )
            )
            if cmd_state:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_cmd_{line_idx}",
                        source_type="PROCESS_TELEMETRY",
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="COMMANDED_STATE",
                        subject=obs_valve_id,
                        predicate="commanded_state",
                        object_entity=normalize_entity(cmd_state),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=evidence_basis,
                        purdue_level=0.0,
                        attributes={"commanded_state": cmd_state},
                    )
                )
            if act_state:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_act_{line_idx}",
                        source_type="PROCESS_TELEMETRY",
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="ACTUAL_STATE",
                        subject=obs_valve_id,
                        predicate="actual_state",
                        object_entity=normalize_entity(act_state),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=evidence_basis,
                        purdue_level=0.0,
                        attributes={"actual_state": act_state},
                    )
                )
            if interlock:
                events.append(
                    NormalizedEvent(
                        source_id=source_id,
                        record_id=f"process_int_{line_idx}",
                        source_type="PROCESS_TELEMETRY",
                        event_time=event_time,
                        ingested_at=ingested_at,
                        event_type="INTERLOCK_STATUS",
                        subject=obs_valve_id,
                        predicate="interlock_status",
                        object_entity=normalize_entity(interlock),
                        raw_record=raw,
                        source_location=f"{file_rel}:L{line_idx}",
                        evidence_basis=evidence_basis,
                        purdue_level=0.0,
                        attributes={"interlock_status": interlock},
                    )
                )
                if "PLC" in str(interlock).upper():
                    # Physical control link connecting Level 1 controller to Level 0 actuator
                    events.append(
                        NormalizedEvent(
                            source_id=source_id,
                            record_id=f"process_plc_link_{line_idx}",
                            source_type="PROCESS_TELEMETRY",
                            event_time=event_time,
                            ingested_at=ingested_at,
                            event_type="CONTROLLER_ACTUATOR_LINK",
                            subject="plc-ballast-01",
                            predicate="sent_command",
                            object_entity=tag,
                            raw_record=raw,
                            source_location=f"{file_rel}:L{line_idx}",
                            evidence_basis=evidence_basis,
                            purdue_level=0.0,
                            attributes={"interlock_status": interlock, "commanded_state": cmd_state},
                        )
                    )

        return events
