# graph-memory-agent

A lightweight Python tool that lets small, local LLMs (like Qwen 2.5 Coder 7B on Ollama) navigate complex, multi-hop documents without blowing past a 4k token window.

Built from the core concepts in the *GraphReader* paper ([arXiv:2406.14550](https://arxiv.org/abs/2406.14550)), with a specific focus on **local cybersecurity incident analysis and threat hunting**.

---

## Why this exists

### 1. Vector RAG fails at multi-hop links
Vector search (cosine similarity on text embeddings) works when the answer is in one chunk. It breaks down when answering multi-stage questions like:
> *"What persistence registry key was dropped by the threat group behind the C2 beacon detected on WKSTN-809?"*

The endpoint alert on `WKSTN-809` mentions an IP, the threat intel report links that IP to an APT group, and the group profile mentions the registry key. An embedding search for "registry key on WKSTN-809" will not find the connection.

### 2. Massive context windows degrade and cost money
Dumping 50,000 lines of SIEM logs into a 128k context window triggers "lost-in-the-middle" attention degradation. More importantly, in security operations (SOC/DFIR), you cannot leak client network logs, internal IPs, or credential paths to third-party cloud APIs.

This project runs **100% locally** via Ollama. It extracts an entity co-occurrence graph and uses an autonomous agent loop that hops from entity to entity, logging intermediate clues into a scratchpad memory until it has enough evidence to answer.

---

## How it works

```
 Raw Document / Incident Log
            │
            ▼
 1. Atomic Fact Extraction
    - Chunks text
    - Prompts local LLM for standalone atomic facts + key entities
            │
            ▼
 2. NetworkX Knowledge Graph
    - Nodes: Hostnames, Users, IPs, Threat Actors, Hashes, Labs
    - Edges: Shared occurrence across atomic facts
            │
            ▼
 3. Autonomous Traversal Loop (4k Context Budget)
    ┌────────────────────────────────────────────────────────┐
    │ a. Formulate a rational step-by-step plan              │
    │ b. Pick the best entry-point node                       │
    │ c. Inspect atomic facts at current node                 │
    │ d. Record discovered clues into the Notebook            │
    │ e. Decide next action: HOP to neighbor OR TERMINATE     │
    └────────────────────────────────────────────────────────┘
            │
            ▼
 4. Direct Answer Synthesis from Notebook Evidence
```

---

## Quickstart

### Prerequisites
1. Python 3.10+
2. [Ollama](https://ollama.com) installed and running locally:
   ```bash
   ollama pull qwen2.5-coder:7b
   ```

### Setup & Run
```bash
git clone https://github.com/ecorcovado/graph-memory-agent.git
cd graph-memory-agent

# Install dependencies (only networkx required)
pip install networkx

# Run the multi-hop test
python run_agent.py --model qwen2.5-coder:7b

# Run offline unit tests
python -m unittest tests/test_components.py
```

---

## Example Trajectory

Tested locally on `qwen2.5-coder:7b` (4K context constraint):

```text
Target Question:
"In which historic building was the lab founded where the lead architect of Project Aegis previously worked?"

[1. RATIONAL PLAN]
Decompose query: Identify lead architect of Project Aegis -> Identify previous lab/company -> Locate founding historic building.

[2. INITIAL NODE SELECTED]
Target: 'project aegis'

[STEP 1: NOTEBOOK UPDATED]
Node: 'project aegis'
Discovered: Marcus Thorne is the lead engineer who architected the ingestion engine for Project Aegis.
Action: ACTION: HOP | marcus thorne

[STEP 2: NOTEBOOK UPDATED]
Node: 'marcus thorne'
Discovered: Marcus Thorne previously led the threat research division at Sentinel Core Labs based in Melbourne.
Action: ACTION: HOP | sentinel core labs

[STEP 3: NOTEBOOK UPDATED]
Node: 'sentinel core labs'
Discovered: Sentinel Core Labs was founded in 2018 inside the historic Old Treasury Building in Melbourne, Victoria.
Action: ACTION: HOP | old treasury building

[FINAL ANSWER]
"The historic building where the lab (Sentinel Core Labs) was founded, where the lead architect of Project Aegis (Marcus Thorne) previously worked, is the Old Treasury Building in Melbourne, Victoria."
```

---

## Project Structure

```
graph-memory-agent/
├── core/
│   ├── graph_builder.py   # Ollama client, fact extractor, and NetworkX builder
│   ├── notebook.py        # Append-only structured scratchpad memory
│   └── agent.py           # Rational planner, node explorer, and traversal logic
├── tests/
│   └── test_components.py # Offline deterministic unit tests
├── run_agent.py           # CLI entry point
└── README.md
```

## Reference
- Li et al., *"GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models"*, arXiv:2406.14550 (2024).
