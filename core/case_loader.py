"""
Case Loader & Deterministic Structured Log Ingestion for Project Aegis (v0.5.1).
Implements the 9-stage forensic ingestion architecture:
Case Manifest -> Case Loader -> Source SHA-256 & Record Identity -> Format Detection ->
Sysmon & Zeek Adapters -> Normalized Event Model -> Atomic Fact Generator ->
Evidence Validation -> MultiDiGraph -> GraphMemory Agent.
"""

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from core.graph_builder import (
    AtomicFact, KnowledgeGraph, normalize_entity, classify_relation_forensics,
    EVIDENCE_BASIS_TELEMETRY, EVIDENCE_BASIS_UNCLASSIFIED, TRUSTED_TELEMETRY_SOURCES,
)
from core.adapters import (
    BaseLogAdapter, NormalizedEvent, SysmonAdapter, ZeekAdapter,
    WindowsSecurityAdapter, FirewallAdapter, HistorianAdapter,
    ControllerAuditAdapter, ProcessTelemetryAdapter, FormatDetector
)


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


class SourceIdentifier:
    """
    Stage 3: Source SHA-256 and Record Identity.
    Computes cryptographic SHA-256 digests over raw byte snapshots and constructs
    deterministic, collision-free source and record identifiers.
    """

    @staticmethod
    def compute_source_identity(file_bytes: bytes) -> Tuple[str, str]:
        """Calculates full 64-hex-character SHA-256 digest and 32-character source_id."""
        full_sha = hashlib.sha256(file_bytes).hexdigest()
        source_id = full_sha[:32]
        return full_sha, source_id


def normalize_timestamp(raw_ts: Any) -> str:
    """Normalizes ISO-8601 strings, Unix epoch floats/integers, or numeric strings to UTC ISO-8601."""
    if raw_ts is None:
        return "UNKNOWN"
    try:
        # 1. Direct numeric timestamp (e.g. Zeek Unix epoch float/int)
        if isinstance(raw_ts, (int, float)):
            dt = datetime.fromtimestamp(float(raw_ts), tz=timezone.utc)
            return dt.isoformat()
        ts_str = str(raw_ts).strip()
        if not ts_str:
            return "UNKNOWN"
        # 2. Numeric string timestamp (e.g. "1730112310.120000" or "1730112310")
        if re.match(r"^\d+(\.\d+)?$", ts_str):
            dt = datetime.fromtimestamp(float(ts_str), tz=timezone.utc)
            return dt.isoformat()
        # 3. ISO-8601 string
        dt = datetime.fromisoformat(ts_str.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc).isoformat()
    except Exception:
        return "INVALID_TIMESTAMP"


class StructuredLogParser:
    """
    Unified dispatcher connecting format detection to specialized log adapters:
    - SysmonAdapter (Sysmon EID 1, 3, 11, 12-14, 22)
    - ZeekAdapter (Zeek conn.log, dns.log, notice.log, http.log in JSON & TSV)
    - WindowsSecurityAdapter (Windows Security EID 4624, 4672, Workstation)
    - FirewallAdapter (DMZ Firewall & Remote Access Gateway)
    - HistorianAdapter (Historian Mirror SQL query export & Replication Sync)
    - ControllerAuditAdapter (Level 1 CIP Controller logic modification)
    - ProcessTelemetryAdapter (Level 0 physical sensors & valve interlocks)
    """

    _detector = FormatDetector()

    @classmethod
    def parse_line(
        cls,
        data: dict,
        source_id: str,
        source_type: str,
        file_rel: str,
        line_idx: int,
    ) -> List[NormalizedEvent]:
        """
        Parses a single log record dictionary into canonical NormalizedEvents
        with timestamp normalization and strict source trust validation.
        """
        ingested_at = datetime.now(timezone.utc).isoformat()

        # 1. Timestamp validation & normalization (handles ISO and Unix epoch floats)
        raw_ts = data.get("timestamp") if "timestamp" in data else data.get("ts")
        event_time = normalize_timestamp(raw_ts)

        # 2. Source trust validation: Only verified telemetry sources may assign DIRECT_TELEMETRY
        if source_type in TRUSTED_TELEMETRY_SOURCES:
            assigned_basis = EVIDENCE_BASIS_TELEMETRY
        else:
            assigned_basis = EVIDENCE_BASIS_UNCLASSIFIED

        # 3. Format Detection & Adapter Routing
        canonical_type, adapter = cls._detector.detect_and_get_adapter(
            file_rel=file_rel,
            sample_records=[data],
            manifest_source_type=source_type,
        )

        if adapter:
            return adapter.parse_record(
                record=data,
                source_id=source_id,
                file_rel=file_rel,
                line_idx=line_idx,
                evidence_basis=assigned_basis,
                event_time=event_time,
                ingested_at=ingested_at,
            )

        return []


class AtomicFactGenerator:
    """
    Stage 7: Atomic Fact Generator.
    Transforms canonical NormalizedEvents into immutable, deterministic AtomicFact instances
    with per-source stable fact identifiers, rich attribute text summaries, and provenance ties.
    """

    @staticmethod
    def generate_fact(ev: NormalizedEvent, line_idx: int, raw_record: dict) -> Tuple[AtomicFact, str, str]:
        """Generates an AtomicFact, edge relation tier, and evidence basis."""
        fact_id = f"{ev.source_id}_R{line_idx}_{ev.record_id}"

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
            raw_record=raw_record,
        )
        return atomic_fact, tier, basis


class EvidenceValidator:
    """
    Stage 8: Evidence Validation.
    Enforces the source trust firewall, path traversal verification, and audit completeness checks.
    """

    @staticmethod
    def validate_source_trust(source_type: str, requested_basis: str) -> str:
        """Untrusted source types strictly fail closed to UNCLASSIFIED."""
        if source_type in TRUSTED_TELEMETRY_SOURCES and requested_basis == EVIDENCE_BASIS_TELEMETRY:
            return EVIDENCE_BASIS_TELEMETRY
        return EVIDENCE_BASIS_UNCLASSIFIED


class CaseLoader:
    """
    Coordinates the 9-stage deterministic structured log ingestion pipeline:
    Case Manifest -> Case Loader -> Source SHA-256 and Record Identity ->
    Format Detection -> Sysmon Adapter / Zeek Adapter -> Normalised Event Model ->
    Atomic Fact Generator -> Evidence Validation -> MultiDiGraph.
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
        "zeek_conn.jsonl": "ZEEK_CONN",
        "zeek_dns.jsonl": "ZEEK_DNS",
        "zeek_notice.jsonl": "ZEEK_NOTICE",
        "conn.log": "ZEEK_CONN",
        "dns.log": "ZEEK_DNS",
        "notice.log": "ZEEK_NOTICE",
    }

    def __init__(self, case_dir: str):
        # Stage 1: Case Manifest & Stage 2: Case Loader
        self.case_dir = os.path.realpath(case_dir)
        self.manifest_path = os.path.join(self.case_dir, "case_manifest.json")
        if not os.path.exists(self.manifest_path):
            raise FileNotFoundError(f"Case manifest missing at: {self.manifest_path}")

        with open(self.manifest_path, "r", encoding="utf-8") as f:
            self.manifest = json.load(f)

        self.format_detector = FormatDetector()

    def load_case(self, exclude_sources: Optional[List[str]] = None) -> KnowledgeGraph:
        """
        Executes full deterministic ingestion of all case files listed in the manifest.
        Validates path confinement, tracks complete ingestion audit metrics, and builds MultiDiGraph.
        """
        exclude_sources = set(exclude_sources or [])
        kg = KnowledgeGraph()
        kg.investigation_targets = self.manifest.get("investigation_targets", {})
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

                # Stage 3: Source SHA-256 and Record Identity
                with open(abs_file, "rb") as bf:
                    file_bytes = bf.read()
                full_sha, source_id = SourceIdentifier.compute_source_identity(file_bytes)
                kg.source_hashes[source_id] = full_sha
                files_processed += 1

                # Stage 4: Format Detection
                manifest_type = self.SOURCE_TYPE_MAPPING.get(file_name, "UNCLASSIFIED")

                # Parse byte snapshot (Byte-Preserving Ingestion)
                try:
                    file_text = file_bytes.decode("utf-8")
                except UnicodeDecodeError:
                    print(f"    [WARN] File contains invalid UTF-8: {rel_file}")
                    source_records_rejected += 1
                    empty_sources.append(rel_file)
                    continue

                raw_lines = [l.strip() for l in file_text.splitlines() if l.strip()]

                if not raw_lines:
                    print(f"    [WARN] Empty source file detected: {rel_file}")
                    empty_sources.append(rel_file)

                is_zeek_tsv = any(l.startswith("#fields") or l.startswith("#separator") for l in raw_lines[:20])

                if is_zeek_tsv:
                    separator = "\t"
                    set_separator = ","
                    tsv_fields: List[str] = []
                    tsv_types: List[str] = []
                    unset_field = "-"
                    empty_field = "(empty)"

                    for l in raw_lines:
                        if l.startswith("#separator"):
                            separator = ZeekAdapter.parse_tsv_separator(l)
                        elif l.startswith("#set_separator"):
                            set_separator = ZeekAdapter.parse_tsv_set_separator(l)
                        elif l.startswith("#empty_field"):
                            parts = l.split(maxsplit=1)
                            if len(parts) > 1:
                                empty_field = parts[1].strip()
                        elif l.startswith("#unset_field"):
                            parts = l.split(maxsplit=1)
                            if len(parts) > 1:
                                unset_field = parts[1].strip()
                        elif l.startswith("#fields"):
                            tsv_fields = ZeekAdapter.parse_tsv_headers(l, separator=separator)
                        elif l.startswith("#types"):
                            tsv_types = ZeekAdapter.parse_tsv_types(l, separator=separator)

                    for line_idx, line_str in enumerate(raw_lines, 1):
                        if line_str.startswith("#"):
                            continue
                        source_records_read += 1
                        record = ZeekAdapter.parse_tsv_line(
                            fields=tsv_fields,
                            line_str=line_str,
                            types=tsv_types,
                            separator=separator,
                            set_separator=set_separator,
                            unset_field=unset_field,
                            empty_field=empty_field,
                        )
                        stype = manifest_type if manifest_type != "UNCLASSIFIED" else "ZEEK_FLOW"
                        events = StructuredLogParser.parse_line(
                            data=record,
                            source_id=source_id,
                            source_type=stype,
                            file_rel=rel_file,
                            line_idx=line_idx,
                        )
                        if events:
                            source_records_parsed += 1
                            all_events.extend(events)
                            record["_raw_line"] = line_str
                            for ev in events:
                                kg.raw_record_vault[(ev.source_id, ev.record_id)] = record
                        else:
                            source_records_rejected += 1
                else:
                    for line_idx, line_str in enumerate(raw_lines, 1):
                        source_records_read += 1
                        try:
                            record = json.loads(line_str)
                            if not isinstance(record, dict):
                                source_records_rejected += 1
                                continue

                            # Stage 5: Sysmon & Zeek Adapters -> Stage 6: Normalized Event Model
                            events = StructuredLogParser.parse_line(
                                data=record,
                                source_id=source_id,
                                source_type=manifest_type,
                                file_rel=rel_file,
                                line_idx=line_idx,
                            )
                            if events:
                                source_records_parsed += 1
                                all_events.extend(events)
                                # Store raw record and original line text in vault
                                record["_raw_line"] = line_str
                                for ev in events:
                                    kg.raw_record_vault[(ev.source_id, ev.record_id)] = record
                            else:
                                source_records_rejected += 1
                        except json.JSONDecodeError:
                            source_records_rejected += 1
                            continue

        print(f"    [+] Normalized {len(all_events)} structured telemetry events across Purdue zones.")

        # Stage 7: Atomic Fact Generator -> Stage 8: Evidence Validation -> Stage 9: MultiDiGraph
        for ev in all_events:
            line_str = ev.source_location.split(":L")[-1]
            try:
                line_num = int(line_str)
            except ValueError:
                line_num = 1

            raw_dict = kg.raw_record_vault.get((ev.source_id, ev.record_id), {})

            # Generate Atomic Fact
            atomic_fact, tier, basis = AtomicFactGenerator.generate_fact(
                ev=ev,
                line_idx=line_num,
                raw_record=raw_dict,
            )
            kg.atomic_facts[atomic_fact.fact_id] = atomic_fact

            # Index nodes with Purdue level metadata
            for elem in [ev.subject, ev.object_entity]:
                if not elem:
                    continue
                if elem not in kg.node_to_facts:
                    kg.node_to_facts[elem] = []
                kg.node_to_facts[elem].append(atomic_fact.fact_id)
                if not kg.graph.has_node(elem):
                    kg.graph.add_node(elem, label=elem, purdue_level=ev.purdue_level)

            # Stage 9: MultiDiGraph Edge Construction
            if ev.subject and ev.object_entity and ev.subject != ev.object_entity:
                is_blocked_edge = (
                    ev.predicate in ["blocked_connection_to", "connection_denied", "access_denied", "process_terminated"]
                    or ev.attributes.get("action") in ["BLOCK", "DENY"]
                    or "block" in str(ev.attributes.get("rule_id", "")).lower()
                )
                kg.graph.add_edge(
                    ev.subject,
                    ev.object_entity,
                    relation=ev.predicate,
                    classification=tier,
                    evidence_basis=basis,
                    source_type=ev.source_type,
                    fact_id=atomic_fact.fact_id,
                    is_blocked=is_blocked_edge,
                    traversable=(not is_blocked_edge),
                )

        # Ingestion Audit Completeness Report
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
