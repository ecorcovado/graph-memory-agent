"""
Graph Construction Engine (Inspired by GraphReader - arXiv:2406.14550)
Extracts atomic facts and key elements using a local Ollama LLM,
then constructs a multi-hop knowledge graph via NetworkX.
"""

import json
import re
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple
import networkx as nx


@dataclass
class AtomicFact:
    fact_id: str
    chunk_id: int
    text: str
    key_elements: List[str]


@dataclass
class KnowledgeGraph:
    graph: nx.Graph = field(default_factory=nx.Graph)
    chunks: Dict[int, str] = field(default_factory=dict)
    atomic_facts: Dict[str, AtomicFact] = field(default_factory=dict)
    # Map key element (normalized) -> list of fact_ids
    node_to_facts: Dict[str, List[str]] = field(default_factory=dict)


class LocalOllamaClient:
    """Client for local Ollama instance (zero external pip dependencies)."""

    def __init__(self, model: str = "qwen2.5-coder:7b", base_url: str = "http://localhost:11434"):
        self.model = model
        self.base_url = base_url.rstrip("/")

    def generate(self, prompt: str, temperature: float = 0.1, system: str = "") -> str:
        url = f"{self.base_url}/api/generate"
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": temperature},
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
    Builds a structured knowledge graph from raw text using atomic fact extraction.
    """

    def __init__(self, llm_client: LocalOllamaClient, chunk_size: int = 1500):
        self.llm = llm_client
        self.chunk_size = chunk_size

    def chunk_text(self, text: str) -> List[str]:
        """Splits long text into paragraph-aware chunks."""
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
        Prompts LLM to extract atomic facts and key elements following GraphReader Figure 6.
        """
        system_prompt = (
            "You are an expert knowledge-graph extractor. Deconstruct the given text into standalone atomic facts.\n"
            "For each atomic fact, identify all key named entities, concepts, and nouns.\n\n"
            "FORMAT FOR EVERY LINE:\n"
            "<Full Complete Sentence Fact> | <entity1> | <entity2> | <entity3>\n\n"
            "EXAMPLES:\n"
            "The lead engineer who architected Project Aegis is Marcus Thorne. | Project Aegis | Marcus Thorne | lead engineer\n"
            "Marcus Thorne previously worked at Sentinel Core Labs in Melbourne. | Marcus Thorne | Sentinel Core Labs | Melbourne\n"
            "Sentinel Core Labs was founded inside the Old Treasury Building. | Sentinel Core Labs | Old Treasury Building | Melbourne\n\n"
            "RULES:\n"
            "- The text BEFORE the first '|' must be a full, meaningful sentence.\n"
            "- Include the primary subject and object as key entities after '|'.\n"
            "- Output only the pipe-separated lines. No markdown formatting, no headers."
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
                key_elements = [k.lower() for k in parts[1:]]

                fact_id = f"C{chunk_id}_F{fact_counter}"
                fact_counter += 1

                extracted_facts.append(
                    AtomicFact(
                        fact_id=fact_id,
                        chunk_id=chunk_id,
                        text=fact_text,
                        key_elements=key_elements,
                    )
                )

        return extracted_facts

    def build_graph(self, text: str) -> KnowledgeGraph:
        """
        End-to-end pipeline: Chunks text, extracts facts, links nodes in NetworkX.
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
                elem_norm = elem.strip().lower()
                if not elem_norm:
                    continue
                if elem_norm not in kg.node_to_facts:
                    kg.node_to_facts[elem_norm] = []
                kg.node_to_facts[elem_norm].append(fact.fact_id)

                if not kg.graph.has_node(elem_norm):
                    kg.graph.add_node(elem_norm, label=elem_norm)

        # Create edges based on co-occurrence in atomic facts
        for fact in all_facts:
            elems = [e.strip().lower() for e in fact.key_elements if e.strip()]
            for i in range(len(elems)):
                for j in range(i + 1, len(elems)):
                    u, v = elems[i], elems[j]
                    if u == v:
                        continue
                    if kg.graph.has_edge(u, v):
                        kg.graph[u][v]["weight"] += 1
                        if fact.fact_id not in kg.graph[u][v]["fact_ids"]:
                            kg.graph[u][v]["fact_ids"].append(fact.fact_id)
                    else:
                        kg.graph.add_edge(u, v, weight=1, fact_ids=[fact.fact_id])

        print(f"[+] Knowledge Graph built: {kg.graph.number_of_nodes()} nodes, {kg.graph.number_of_edges()} edges, {len(kg.atomic_facts)} atomic facts.")
        return kg
