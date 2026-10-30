# GraphMemory-Agent 🧠🕸️

> **Autonomous Graph-Augmented Agentic Memory for Local LLMs within a Strict 4K Context Window.**  
> *Inspired by Alibaba & CUHK's research paper: "GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models" (arXiv:2406.14550).*

---

## 🚀 Overview

Standard Vector RAG (embedding similarity) often fails when answering **multi-hop questions** because semantic similarity cannot traverse logical links across distinct document segments. Conversely, feeding hundreds of thousands of tokens into 128k context windows leads to severe **"lost-in-the-middle" attention degradation** and prohibitive inference costs.

**GraphMemory-Agent** implements an autonomous, coarse-to-fine graph exploration engine that allows lightweight local models (e.g. `Qwen 2.5 Coder 7B`, `Llama 3.2 3B`) running on **Ollama** to solve complex multi-hop queries across long texts while staying strictly within a **4k context window**.

---

## 🏗️ Architecture

```
                 [ Long Document / Incident Logs ]
                                 │
                                 ▼
                    [ 1. Graph Construction ]
                - Text Chunking (Paragraph-aware)
                - Atomic Fact Extraction (LLM)
                - Entity / Key Element Normalization
                                 │
                                 ▼
                    [ 2. Knowledge Graph (G) ]
                 - Nodes: Entities / Key Concepts
                 - Edges: Co-occurrence in Atomic Facts
                                 │
                                 ▼
                     [ 3. Autonomous Agent ]
           ┌───────────────────────────────────────────┐
           │ • Rational Plan Decomposition             │
           │ • Initial Node Selection                  │
           │ • Coarse-to-Fine Fact Inspection          │
           │ • Hop-to-Neighbor or Read-Chunk           │
           │ • Dynamic Notebook Scratchpad             │
           └───────────────────────────────────────────┘
                                 │
                                 ▼
                     [ 4. Answer Synthesis ]
```

---

## ✨ Key Features

* **100% Local & Privacy-Preserving:** Runs entirely on local Ollama endpoints (`localhost:11434`). Zero cloud data leakage, zero API subscription costs.
* **4K Context Strictness:** The agent never ingests raw full documents. It navigates local subgraphs and records fine-grained evidence in a compact notebook.
* **Multi-Hop Traversal:** Capable of resolving 3-hop to 5-hop relationship queries that conventional cosine vector search completely misses.
* **Zero Heavy Dependencies:** Engineered cleanly with Python standard libraries (`urllib`, `json`, `dataclasses`) and `networkx`.

---

## ⚡ Quickstart

### Prerequisites
1. Ensure Python 3.10+ is installed.
2. Install [Ollama](https://ollama.com) and pull your preferred local model:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

### Run the Demo
```bash
git clone https://github.com/ecorcovado/graph-memory-agent.git
cd graph-memory-agent

# Run the 3-hop multi-hop reasoning demo
python run_agent.py --model qwen2.5-coder:7b

# Run unit tests offline
python -m unittest tests/test_components.py
```

---

## 🔬 Live Reasoning Trajectory Example

Below is an actual execution trace on a 3-hop reasoning benchmark using `qwen2.5-coder:7b`:

```text
Target Question:
"In which historic building was the lab founded where the lead architect of Project Aegis previously worked?"

[1. RATIONAL PLAN]
Decompose query: Identify lead architect of Project Aegis -> Identify previous lab/company -> Locate founding historic building.

[2. INITIAL NODE SELECTED]
Target: 'project aegis'

[STEP 1: NOTEBOOK UPDATED]
Node: 'project aegis'
Discovered Fact: Marcus Thorne is the lead engineer who architected the ingestion engine for Project Aegis.
Action: ACTION: HOP | marcus thorne

[STEP 2: NOTEBOOK UPDATED]
Node: 'marcus thorne'
Discovered Fact: Marcus Thorne previously led the threat research division at Sentinel Core Labs based in Melbourne.
Action: ACTION: HOP | sentinel core labs

[STEP 3: NOTEBOOK UPDATED]
Node: 'sentinel core labs'
Discovered Fact: Sentinel Core Labs was founded in 2018 inside the historic Old Treasury Building in Melbourne, Victoria.
Action: ACTION: HOP | old treasury building

[FINAL ANSWER]
"The historic building where the lab (Sentinel Core Labs) was founded, where the lead architect of Project Aegis (Marcus Thorne) previously worked, is the Old Treasury Building in Melbourne, Victoria."
```

---

## 📊 Research Lineage & Benchmark Context

Based on the findings in *GraphReader (Li et al., 2024)*:
* **LV-Eval & LongBench:** An agent equipped with a 4k context window exploring an atomic fact graph outperformed GPT-4-128k across context lengths from 16k to 256k tokens.
* **Mitigating "Lost in the Middle":** By decomposing text into atomic facts $A_i$ and maintaining an active notebook scratchpad, factual recall remained above 60% even at 256k token distances.

---

## 📄 License
MIT License. Created by Edgard Corcovado (2026).
