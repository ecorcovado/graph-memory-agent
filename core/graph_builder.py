"""
Graph Builder module for Graph Memory Agent.
Extracts atomic facts from text and constructs a directed MultiDiGraph in NetworkX
with typed relationships, provenance tracking, and explicit 4K context bounds.
"""

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import networkx as nx


def normalize_entity(name: str) -> str:
    """Canonical entity normalization to prevent duplicate nodes."""
    cleaned = re.sub(r'["\']', '', name.strip().lower())
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return cleaned


def classify_relation(relation: str) -> str:
    """Classifies edge relations into forensic tiers: OBSERVED, INTELLIGENCE_REPORTED, or CO_OCCURRENCE."""
    rel = relation.lower().strip()
    if rel in ["co_occurs_with", "co_occurs", "related_to"]:
        return "CO_OCCURRENCE"
    if any(k in rel for k in ["attributed", "intel", "known", "associated", "suspected", "reported"]):
        return "INTELLIGENCE_REPORTED"
    return "OBSERVED"


@dataclass
class AtomicFact:
    fact_id: str
    chunk_id: int
    text: str
    subject: str = ""
    relation: str = "related_to"
    object_entity: str = ""
    key_elements: List[str] = field(default_factory=list)


@dataclass
class KnowledgeGraph:
    graph: nx.MultiDiGraph = field(default_factory=nx.MultiDiGraph)
    chunks: Dict[int, str] = field(default_factory=dict)
    atomic_facts: Dict[str, AtomicFact] = field(default_factory=dict)
    # Map key element (normalized) -> list of fact_ids
    node_to_facts: Dict[str, List[str]] = field(default_factory=dict)

    def get_neighbors_with_relations(self, node: str) -> List[Tuple[str, str, str, str]]:
        """
        Returns list of (direction, relation, classification, connected_node).
        Distinguishes OBSERVED events from INTELLIGENCE_REPORTED or CO_OCCURRENCE.
        """
        results = []
        if not self.graph.has_node(node):
            return results

        # Outgoing edges: node -> target
        for _, target, data in self.graph.out_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            cls = data.get("classification", "OBSERVED")
            results.append(("outgoing", rel, cls, target))

        # Incoming edges: source -> node
        for source, _, data in self.graph.in_edges(node, data=True):
            rel = data.get("relation", "connected_to")
            cls = data.get("classification", "OBSERVED")
            results.append(("incoming", rel, cls, source))

        return results

    def get_adjacent_nodes(self, node: str) -> List[str]:
        """Returns all unique adjacent nodes (both incoming and outgoing)."""
        if not self.graph.has_node(node):
            return []
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
    ):
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.num_ctx = num_ctx

    def generate(self, prompt: str, temperature: float = 0.1, system: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_ctx": self.num_ctx,  # Enforces strict 4K context budget
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

    def extract_atomic_facts(self, chunk: str, chunk_id: int) -> List[AtomicFact]:
        """
        Prompts LLM to extract atomic facts and typed triples (Subject | Relation | Object).
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

        prompt = f"Text:\n\"\"\"\n{chunk}\n\"\"\"\n\nExtracted Facts:"
        raw_output = self.llm.generate(prompt=prompt, system=system_prompt, temperature=0.1)

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

                fact_id = f"C{chunk_id}_F{fact_counter}"
                fact_counter += 1

                extracted_facts.append(
                    AtomicFact(
                        fact_id=fact_id,
                        chunk_id=chunk_id,
                        text=fact_text,
                        subject=subject,
                        relation=relation,
                        object_entity=obj,
                        key_elements=key_elements,
                    )
                )

        return extracted_facts

    def build_graph(self, text: str) -> KnowledgeGraph:
        """
        End-to-end pipeline: Chunks text, extracts facts, links typed directed edges in NetworkX.
        """
        chunks = self.chunk_text(text)
        kg = KnowledgeGraph()
        all_facts: List[AtomicFact] = []

        print(f"[*] Processing {len(chunks)} chunks through local Ollama ({self.llm.model})...")

        for idx, chunk in enumerate(chunks):
            kg.chunks[idx] = chunk
            print(f"    -> Extracting atomic facts from Chunk {idx+1}/{len(chunks)}...")
            facts = self.extract_atomic_facts(chunk, idx)
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
                cls = classify_relation(rel)
                kg.graph.add_edge(u, v, relation=rel, classification=cls, fact_id=fact.fact_id)

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
                            fact_id=fact.fact_id,
                        )

        print(f"[+] Directed Knowledge Graph built: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges, {len(kg.atomic_facts)} atomic facts.")
        return kg
