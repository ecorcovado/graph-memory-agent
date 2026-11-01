"""
Graph Builder module for Graph Memory Agent.
Extracts atomic facts from text and constructs a directed MultiDiGraph in NetworkX
with typed relationships, provenance tracking, and explicit 4K context bounds.
"""

import hashlib
import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import networkx as nx
from core.budget_manager import PromptBudgetManager


def normalize_entity(name: str) -> str:
    """Canonical entity normalization to prevent duplicate nodes."""
    cleaned = re.sub(r'["\']', '', name.strip().lower())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


# Strict forensic classification allowlists
OBSERVED_PREDICATES = {
    "communicated_with", "connected_to", "executed", "created", "dropped_file",
    "modified_registry", "logged_into", "downloaded", "established_beacon",
    "modified_logic", "sent_command", "authenticated_to", "scanned",
    "has_observation", "measured_value", "commanded_state", "actual_state", "interlock_status",
    "queried_tag_store", "replicated_tags_to",
    "queried_dns", "triggered_notice", "http_request",
}

INTELLIGENCE_PREDICATES = {
    "attributed_to", "known_technique", "associated_with", "suspected_actor",
    "maintains_persistence", "uses_malware", "targets_sector", "aliases",
    "founded_inside", "architected", "worked_at", "previously_worked_at", "affiliated_with",
}

# Forensic Evidence Basis constants (separates what happened from how it is known)
EVIDENCE_BASIS_TELEMETRY = "DIRECT_TELEMETRY"
EVIDENCE_BASIS_REPORTED = "ANALYST_REPORTED"
EVIDENCE_BASIS_INTEL = "THREAT_INTELLIGENCE"
EVIDENCE_BASIS_INFERRED = "INFERRED"
EVIDENCE_BASIS_CO_OCCURRENCE = "CO_OCCURRENCE"
EVIDENCE_BASIS_UNCLASSIFIED = "UNCLASSIFIED"

VALID_EVIDENCE_BASES = {
    EVIDENCE_BASIS_TELEMETRY,
    EVIDENCE_BASIS_REPORTED,
    EVIDENCE_BASIS_INTEL,
    EVIDENCE_BASIS_INFERRED,
    EVIDENCE_BASIS_CO_OCCURRENCE,
    EVIDENCE_BASIS_UNCLASSIFIED,
}

TRUSTED_TELEMETRY_SOURCES = {
    "SYSMON_EVENT", "WINDOWS_SECURITY", "FIREWALL_LOG",
    "HISTORIAN_MIRROR", "HISTORIAN_AUDIT", "SCADA_AUDIT",
    "CONTROLLER_AUDIT", "PROCESS_TELEMETRY", "ZEEK_FLOW",
    "ZEEK_LOG", "ZEEK_CONN", "ZEEK_DNS", "ZEEK_NOTICE"
}


def classify_relation(relation: str) -> str:
    """
    Classifies edge relations into strict forensic tiers via exact allowlist matching:
    - OBSERVED: Direct physical/endpoint/network events on strict allowlist.
    - INTELLIGENCE_REPORTED: CTI attributions and TTP mappings on strict allowlist.
    - CO_OCCURRENCE: Explicit secondary entity mentions.
    - UNCLASSIFIED: Any unrecognized or ambiguous predicate (strictly never assumed OBSERVED).
    """
    rel = relation.lower().strip().replace(" ", "_")
    if rel in ["co_occurs_with", "co_occurs", "related_to"]:
        return "CO_OCCURRENCE"
    if rel in OBSERVED_PREDICATES:
        return "OBSERVED"
    if rel in INTELLIGENCE_PREDICATES:
        return "INTELLIGENCE_REPORTED"
    return "UNCLASSIFIED"


def classify_relation_forensics(
    relation: str,
    source_type: str = "INCIDENT_REPORT",
    evidence_basis: Optional[str] = None,
) -> Tuple[str, str]:
    """
    Separates predicate semantics (what happened) from evidence basis (how it is known).
    Enforces strict source trust validation BEFORE honoring explicit evidence bases:
    - Untrusted source types strictly fail closed to UNCLASSIFIED and CANNOT assert DIRECT_TELEMETRY.
    - Explicit evidence basis cannot elevate beyond the verified capabilities of its source_type.
    """
    tier = classify_relation(relation)

    # 1. Strict Source-Type Validation First (Source Trust Firewall)
    if source_type not in TRUSTED_TELEMETRY_SOURCES and source_type not in {
        "INCIDENT_REPORT", "THREAT_INTELLIGENCE", "INFERRED", "CO_OCCURRENCE"
    }:
        # Untrusted source: strictly fail closed to UNCLASSIFIED
        return "UNCLASSIFIED", EVIDENCE_BASIS_UNCLASSIFIED

    # 2. Prevent untrusted sources from self-asserting DIRECT_TELEMETRY
    if evidence_basis == EVIDENCE_BASIS_TELEMETRY and source_type not in TRUSTED_TELEMETRY_SOURCES:
        return "UNCLASSIFIED", EVIDENCE_BASIS_UNCLASSIFIED

    # 3. Validate explicit evidence basis if provided
    if evidence_basis:
        if evidence_basis in VALID_EVIDENCE_BASES:
            basis = evidence_basis
        else:
            basis = EVIDENCE_BASIS_UNCLASSIFIED
            tier = "UNCLASSIFIED"
    elif tier == "CO_OCCURRENCE":
        basis = EVIDENCE_BASIS_CO_OCCURRENCE
    elif source_type in TRUSTED_TELEMETRY_SOURCES:
        basis = EVIDENCE_BASIS_TELEMETRY if tier == "OBSERVED" else EVIDENCE_BASIS_REPORTED
    elif source_type == "INCIDENT_REPORT":
        basis = EVIDENCE_BASIS_REPORTED if tier == "OBSERVED" else (
            EVIDENCE_BASIS_INTEL if tier == "INTELLIGENCE_REPORTED" else EVIDENCE_BASIS_UNCLASSIFIED
        )
    elif source_type == "THREAT_INTELLIGENCE":
        basis = EVIDENCE_BASIS_INTEL
    elif source_type == "INFERRED":
        basis = EVIDENCE_BASIS_INFERRED
    else:
        basis = EVIDENCE_BASIS_UNCLASSIFIED
        tier = "UNCLASSIFIED"

    return tier, basis


class ForensicRelation(tuple):
    """
    4-tuple (direction, relation, classification, target) with extended metadata attributes
    (.evidence_basis, .source_type, .fact_id) ensuring 100% backwards compatibility with 4-tuple unpacking.
    """
    def __new__(
        cls,
        direction: str,
        relation: str,
        classification: str,
        target: str,
        evidence_basis: str = "UNCLASSIFIED",
        source_type: str = "INCIDENT_REPORT",
        fact_id: str = "",
    ):
        return super().__new__(cls, (direction, relation, classification, target))

    def __init__(
        self,
        direction: str,
        relation: str,
        classification: str,
        target: str,
        evidence_basis: str = "UNCLASSIFIED",
        source_type: str = "INCIDENT_REPORT",
        fact_id: str = "",
    ):
        self.direction = direction
        self.relation = relation
        self.classification = classification
        self.target = target
        self.evidence_basis = evidence_basis
        self.source_type = source_type
        self.fact_id = fact_id


@dataclass
class AtomicFact:
    fact_id: str
    chunk_id: int
    text: str
    source_id: str = ""
    source_record_id: str = ""
    source_location: str = ""
    event_time: str = ""
    purdue_level: Optional[float] = None
    subject: str = ""
    relation: str = "related_to"
    object_entity: str = ""
    key_elements: List[str] = field(default_factory=list)
    evidence_basis: str = "UNCLASSIFIED"
    source_type: str = "INCIDENT_REPORT"
    confidence: str = "HIGH"
    attributes: Dict[str, Any] = field(default_factory=dict)
    raw_record: Optional[Dict[str, Any]] = None


@dataclass
class KnowledgeGraph:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    chunks: Dict[int, str] = field(default_factory=dict)
    atomic_facts: Dict[str, AtomicFact] = field(default_factory=dict)
    # Map key element (normalized) -> list of fact_ids
    node_to_facts: Dict[str, List[str]] = field(default_factory=dict)
    # Map source_id -> full SHA-256 digest for document verification
    source_hashes: Dict[str, str] = field(default_factory=dict)
    # Source Record Vault: (source_id, record_id) -> original raw dictionary
    raw_record_vault: Dict[Tuple[str, str], dict] = field(default_factory=dict)
    ingestion_report: Optional[Any] = None
    investigation_targets: Dict[str, Any] = field(default_factory=dict)

    def get_neighbors_with_relations(self, node: str) -> List[ForensicRelation]:
        """
        Returns list of ForensicRelation items unpackable as (direction, relation, classification, target)
        with .evidence_basis and .source_type attributes.
        """
        results = []
        if not self.graph.has_node(node):
            return results

        # Outgoing edges: node -> target
        for _, target, data in self.graph.out_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            cls = data.get("classification", "UNCLASSIFIED")
            basis = data.get("evidence_basis", "UNCLASSIFIED")
            src_type = data.get("source_type", "INCIDENT_REPORT")
            fid = data.get("fact_id", "")
            results.append(
                ForensicRelation(
                    "outgoing", rel, cls, target,
                    evidence_basis=basis, source_type=src_type, fact_id=fid
                )
            )

        # Incoming edges: source -> node
        for source, _, data in self.graph.in_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            cls = data.get("classification", "UNCLASSIFIED")
            basis = data.get("evidence_basis", "UNCLASSIFIED")
            src_type = data.get("source_type", "INCIDENT_REPORT")
            fid = data.get("fact_id", "")
            results.append(
                ForensicRelation(
                    "incoming", rel, cls, source,
                    evidence_basis=basis, source_type=src_type, fact_id=fid
                )
            )

        return results

    def is_transition_blocked(self, source: str, target: str) -> bool:
        """
        Determines whether the transition from source to target was actively blocked by boundary defenses.
        Returns True if a blocked edge exists AND no separate allowed connection exists between source and target.
        If an allowed connection exists (e.g. lateral movement succeeded despite an earlier blocked attempt),
        the transition is NOT blocked.
        """
        if not self.graph.has_edge(source, target):
            return False
        edges = self.graph.get_edge_data(source, target) or {}

        has_blocked = False
        has_allowed = False

        for attr in edges.values():
            is_blk = (
                attr.get("is_blocked", False)
                or attr.get("relation") in ["blocked_connection_to", "connection_denied", "access_denied", "process_terminated"]
                or attr.get("action") in ["BLOCK", "DENY"]
            )
            if is_blk:
                has_blocked = True
            else:
                # Any non-blocked directional connectivity or operational event represents an allowed transition
                if attr.get("relation") in [
                    "connected_to", "communicated_with", "authenticated_to", "executed", "modified_logic",
                    "sent_command", "logged_into", "downloaded", "established_beacon"
                ]:
                    has_allowed = True

        return has_blocked and not has_allowed

    def get_traversable_neighbors(self, node: str) -> List[str]:
        """
        Returns only adjacent nodes that can be reached via valid, unblocked attack-path transitions.
        Distinguishes evidence exploration from attack-path compromise progression.
        """
        if not self.graph.has_node(node):
            return []
        traversable = []
        for target in self.graph.successors(node):
            if target != node and not self.is_transition_blocked(node, target):
                if target not in traversable:
                    traversable.append(target)
        return traversable

    def get_adjacent_nodes(self, node: str, traversable_only: bool = False) -> List[str]:
        """
        Returns adjacent nodes. If traversable_only is True, excludes nodes connected only
        via blocked conduits (e.g. firewall blocked_connection_to).
        """
        if not self.graph.has_node(node):
            return []
        if traversable_only:
            return self.get_traversable_neighbors(node)
        successors = list(self.graph.successors(node))
        predecessors = list(self.graph.predecessors(node))
        seen = set()
        adj = []
        for n in successors + predecessors:
            if n not in seen and n != node:
                seen.add(n)
                adj.append(n)
        return adj


class LocalOllamaClient:
    """Client for local Ollama instance with explicit context limits."""

    def __init__(
        self,
        model: str = "qwen2.5-coder:7b",
        base_url: str = "http://localhost:11434",
        num_ctx: int = 4096,
        num_predict: int = 500,
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx
        self.num_predict = num_predict

    def generate(self, prompt: str, temperature: float = 0.1, system: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,  # Enforces strict 4K context budget
                "num_predict": self.num_predict,  # Enforces generation output budget
            },
        }
        if system:
            payload["system"] = system

        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, headers={"Content-Type": "application/json"}
        )

        with urllib.request.urlopen(req, timeout=120) as resp:
            res_json = json.loads(resp.read().decode("utf-8"))
            return res_json.get("response", "").strip()


class GraphBuilder:
    """
    Constructs a directed, typed Knowledge Graph from unstructured text or incident logs.
    """

    def __init__(self, llm_client: LocalOllamaClient, chunk_size: int = 1500):
        self.llm = llm_client
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """Splits raw text into manageable chunks respecting paragraphs."""
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0

        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
            if current_len + len(para) > self.chunk_size and current_chunk:
                chunks.append("\n\n".join(current_chunk))
                current_chunk = [para]
                current_len = len(para)
            else:
                current_chunk.append(para)
                current_len += len(para)

        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

    def extract_atomic_facts(
        self,
        chunk: str,
        chunk_id: int,
        source_id: str = "src_doc",
        source_type: str = "INCIDENT_REPORT",
    ) -> List[AtomicFact]:
        """
        Prompts LLM to extract atomic facts and typed triples (Subject | Relation | Object).
        Uses stable document-level source_id and zero-padded chunk and fact ordinals.
        """
        system_prompt = (
            "You are an expert DFIR and Knowledge Graph analyst. Deconstruct the given text into standalone atomic facts.\n"
            "For each atomic fact, identify the directed relationship:\n\n"
            "FORMAT FOR EVERY LINE:\n"
            "<Full Complete Sentence Fact> | <Subject Entity> | <Relation> | <Object Entity>\n\n"
            "EXAMPLES:\n"
            "WORKSTATION-402 made an outbound connection to 198.51.100.44 | WORKSTATION-402 | communicated_with | 198.51.100.44\n"
            "Forensic triage identified an obfuscated powershell execution | WORKSTATION-402 | executed | powershell.exe\n"
            "Threat intelligence attributes IP 198.51.100.44 to APT-SilverFox | 198.51.100.44 | attributed_to | APT-SilverFox\n"
            "APT-SilverFox maintains persistence using SystemHealthCheck task | APT-SilverFox | maintains_persistence | SystemHealthCheck\n"
            "SystemHealthCheck executes a backdoor DLL | SystemHealthCheck | executes | backdoor DLL\n"
            "Marcus Thorne architected the ingestion engine for Project Aegis | Marcus Thorne | architected | Project Aegis\n"
            "Marcus Thorne previously worked at Sentinel Core Labs | Marcus Thorne | previously_worked_at | Sentinel Core Labs\n"
            "Sentinel Core Labs was founded inside Old Treasury Building | Sentinel Core Labs | founded_inside | Old Treasury Building\n\n"
            "RULES:\n"
            "- Text before the first '|' must be a full factual sentence.\n"
            "- Relation must be a snake_case verb/predicate (e.g. communicated_with, executed, attributed_to, founded_inside).\n"
            "- Output only pipe-separated lines. No markdown headers."
        )

        sys_prompt, user_prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text=system_prompt,
            context_text=f"Text to Deconstruct:\n\"\"\"\n{chunk}\n\"\"\"",
            instructions="Extracted Facts (pipe-separated lines only):",
        )
        raw_output = self.llm.generate(prompt=user_prompt, system=sys_prompt, temperature=0.1)

        extracted_facts = []
        lines = raw_output.split("\n")
        fact_counter = 1

        for line in lines:
            line = line.strip()
            line = re.sub(r"^\d+[\.\)]\s*", "", line)
            line = re.sub(r"^-\s*", "", line)

            if "|" not in line:
                continue

            parts = [p.strip() for p in line.split("|") if p.strip()]
            if len(parts) >= 2:
                fact_text = parts[0]
                subject = normalize_entity(parts[1]) if len(parts) >= 2 else ""
                relation = normalize_entity(parts[2]).replace(" ", "_") if len(parts) >= 3 else "related_to"
                obj = normalize_entity(parts[3]) if len(parts) >= 4 else ""

                key_elements = [subject]
                if obj:
                    key_elements.append(obj)
                # Additional entities if any
                for extra in parts[4:]:
                    norm_extra = normalize_entity(extra)
                    if norm_extra and norm_extra not in key_elements:
                        key_elements.append(norm_extra)

                fact_id = f"{source_id}_C{chunk_id:04d}_F{fact_counter:04d}"
                fact_counter += 1

                tier, basis = classify_relation_forensics(relation, source_type=source_type)

                extracted_facts.append(
                    AtomicFact(
                        fact_id=fact_id,
                        chunk_id=chunk_id,
                        text=fact_text,
                        source_id=source_id,
                        subject=subject,
                        relation=relation,
                        object_entity=obj,
                        key_elements=key_elements,
                        evidence_basis=basis,
                        source_type=source_type,
                    )
                )

        return extracted_facts

    def build_graph(
        self,
        text: str,
        source_id: Optional[str] = None,
        source_type: str = "INCIDENT_REPORT",
    ) -> KnowledgeGraph:
        """
        End-to-end pipeline: Chunks text, extracts facts, links typed directed edges in NetworkX.
        Stores full document SHA-256 for independent verification.
        """
        full_doc_sha = hashlib.sha256(text.encode("utf-8")).hexdigest()
        resolved_source_id = source_id or full_doc_sha[:32]

        chunks = self.chunk_text(text)
        kg = KnowledgeGraph()
        kg.source_hashes[resolved_source_id] = full_doc_sha
        all_facts: List[AtomicFact] = []

        model_name = getattr(self.llm, "model", "local-llm")
        print(f"[*] Processing {len(chunks)} chunks through local Ollama ({model_name})...")
        print(f"    -> Document Source ID: {resolved_source_id}")
        print(f"    -> SHA-256 Digest: {full_doc_sha}")

        for idx, chunk in enumerate(chunks):
            kg.chunks[idx] = chunk
            print(f"    -> Extracting atomic facts from Chunk {idx+1}/{len(chunks)}...")
            facts = self.extract_atomic_facts(chunk, idx, source_id=resolved_source_id, source_type=source_type)
            all_facts.extend(facts)

        # Index facts & nodes
        for fact in all_facts:
            kg.atomic_facts[fact.fact_id] = fact
            for elem in fact.key_elements:
                if not elem:
                    continue
                if elem not in kg.node_to_facts:
                    kg.node_to_facts[elem] = []
                kg.node_to_facts[elem].append(fact.fact_id)

                if not kg.graph.has_node(elem):
                    kg.graph.add_node(elem, label=elem)

        # Create directed edges with typed relationships and classifications
        for fact in all_facts:
            u = fact.subject
            v = fact.object_entity
            rel = fact.relation

            if u and v and u != v:
                tier, basis = classify_relation_forensics(
                    rel,
                    source_type=fact.source_type,
                    evidence_basis=fact.evidence_basis,
                )
                kg.graph.add_edge(
                    u, v,
                    relation=rel,
                    classification=tier,
                    evidence_basis=basis,
                    source_type=fact.source_type,
                    fact_id=fact.fact_id,
                )

            # Secondary entity co-occurrence edges (explicitly tagged CO_OCCURRENCE)
            elems = [e for e in fact.key_elements if e]
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    e1, e2 = elems[i], elems[j]
                    if e1 != e2 and not kg.graph.has_edge(e1, e2):
                        kg.graph.add_edge(
                            e1, e2,
                            relation="co_occurs_with",
                            classification="CO_OCCURRENCE",
                            evidence_basis=EVIDENCE_BASIS_CO_OCCURRENCE,
                            source_type=fact.source_type,
                            fact_id=fact.fact_id,
                        )

        print(f"[+] Directed Knowledge Graph built: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges, {len(kg.atomic_facts)} atomic facts.")
        return kg
