# graph-memory-agent

A lightweight Python tool that lets small, local LLMs (like Qwen 2.5 Coder 7B on Ollama) navigate complex, multi-hop documents without blowing past a 4k token window.

Built from the core concepts in the *GraphReader* paper ([arXiv:2406.14550](https://arxiv.org/abs/2406.14550)), with a specific focus on **offline-first, evidence-driven cybersecurity incident analysis and threat hunting**.

---

## Why this exists

### 1. Vector RAG fails at multi-hop links
Vector search (cosine similarity on text embeddings) works when the answer is in one chunk. It breaks down when answering multi-stage questions like:
> *"What persistence scheduled task is associated with the threat group communicating with WORKSTATION-402?"*

The endpoint alert on `WORKSTATION-402` mentions an IP, the threat intel report links that IP to an APT group, and the group profile mentions the scheduled task. An embedding search for "scheduled task on WORKSTATION-402" will not find the connection.

### 2. Massive context windows degrade and cost money
Dumping 50,000 lines of SIEM logs into a 128k context window triggers "lost-in-the-middle" attention degradation. More importantly, in security operations (SOC/DFIR), you cannot leak client network logs, internal IPs, or credential paths to third-party cloud APIs.

This project runs **100% locally** via Ollama. It extracts a directed `MultiDiGraph` with typed relationships (`Subject -> Relation -> Object`), uses an autonomous agent loop that hops across entities, and logs intermediate clues into a bounded scratchpad memory with strict token budgets.

---

## How it works

```
 Raw Document / Incident Log
            │
            ▼
 1. Atomic Fact Extraction
    - Chunks text into paragraph-aware blocks
    - Prompts local LLM for standalone atomic facts + typed triples
            │
            ▼
 2. Directed MultiDiGraph (NetworkX)
    - Nodes: Canonicalized Hostnames, Users, IPs, Threat Actors, Tasks
    - Edges: Directed, typed predicates (communicated_with, executed, attributed_to, maintains_persistence)
            │
            ▼
 3. Autonomous Traversal Loop (Strict 4k Context Budget)
    ┌────────────────────────────────────────────────────────┐
    │ a. Formulate a rational step-by-step plan              │
    │ b. Pick the best entry-point node (heuristics fallback)│
    │ c. Inspect atomic facts & directional relations        │
    │ d. Record discovered clues into Bounded Notebook       │
    │ e. Decide next action: HOP to neighbor OR TERMINATE    │
    └────────────────────────────────────────────────────────┘
            │
            ▼
 4. Evidence-Verified Answer Synthesis (Step-by-step Chain)
```

---

## Key Technical Guardrails

* **Directed MultiDiGraph with Forensic Tiers:** Distinguishes relationship directions (`Host -> communicated_with -> IP` vs reverse) and tags edges with forensic classifications: `OBSERVED` (endpoint events), `INTELLIGENCE_REPORTED` (threat intelligence attribution), or `CO_OCCURRENCE`.
* **Centralized Prompt Budget Manager:** Strictly allocates context budgets (system instructions: 400t, query/plan: 300t, notebook: 1,000t, context: 1,500t, output reserve: 500t) and clamps candidate node lists to guarantee requests never exceed 4,096 tokens.
* **Bounded Provenance Memory (`EvidenceNote`):** Enforces a 1,000-token budget with per-note clamping, deduplication, and FIFO eviction. Each note preserves immutable `fact_ids` and source excerpts.
* **Explicit Context Limit:** Passes `"options": {"num_ctx": 4096}` directly to the Ollama API.
* **Resilient Node Selection & Action Validation:** Avoids blind fallbacks using query keyword overlap and hub-degree centrality. Rejects invalid hop targets before execution.
* **Deterministic Evidence Verification:** Requires an unbroken evidence chain citing verified `fact_ids` before synthesizing a final answer, preventing unsupported hallucinations.

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

# 1. Run against the built-in benchmark
python run_agent.py --model qwen2.5-coder:7b

# 2. Run against the DFIR incident report
python run_agent.py --file examples/cyber_incident_report.txt --question "What persistence scheduled task is associated with the threat group communicating with WORKSTATION-402?"

# 3. Run offline unit tests
python -m unittest tests/test_components.py
```

---

## Live Reasoning Trajectory (Cyber DFIR Triage)

Tested locally on `qwen2.5-coder:7b` (4K context constraint):

```text
Target Question:
"What persistence scheduled task is associated with the threat group communicating with WORKSTATION-402?"

[1. RATIONAL PLAN]
Discover threat group communicating with WORKSTATION-402 -> Identify scheduled tasks -> Establish persistence link.

[2. INITIAL NODE SELECTED]
Target: 'workstation-402'

[STEP 1: NOTEBOOK UPDATED]
Node: 'workstation-402'
Discovered: WORKSTATION-402 made an outbound connection to 198.51.100.44; obfuscated powershell.exe execution identified.
Action: ACTION: HOP | 198.51.100.44

[STEP 2: NOTEBOOK UPDATED]
Node: '198.51.100.44'
Discovered: Threat intelligence attributes IP 198.51.100.44 to APT-SilverFox.
Action: ACTION: HOP | apt-silverfox

[STEP 3: NOTEBOOK UPDATED]
Node: 'apt-silverfox'
Discovered: APT-SilverFox maintains persistence using the SystemHealthCheck task.
Action: ACTION: HOP | systemhealthcheck

[STEP 4: NOTEBOOK UPDATED]
Node: 'systemhealthcheck'
Discovered: SystemHealthCheck executes a backdoor DLL.
Action: ACTION: HOP | backdoor dll

[FINAL ANSWER]
Evidence Chain:
1. WORKSTATION-402 made an outbound connection to 198.51.100.44.
2. Threat intelligence attributes IP 198.51.100.44 to APT-SilverFox.
3. APT-SilverFox maintains persistence using the SystemHealthCheck task.
4. SystemHealthCheck executes a backdoor DLL.

Final Direct Answer:
The persistence scheduled task associated with the threat group communicating with WORKSTATION-402 is SystemHealthCheck.
```

---

## Project Structure

```
graph-memory-agent/
├── core/
│   ├── budget_manager.py  # Central prompt budget manager (enforces 4K token ceiling)
│   ├── graph_builder.py   # Ollama client, triple extractor, and NetworkX MultiDiGraph
│   ├── notebook.py        # Bounded provenance memory with EvidenceNotes & FIFO eviction
│   └── agent.py           # Rational planner, node explorer, and evidence verifier
├── examples/
│   └── cyber_incident_report.txt # DFIR incident triage scenario
├── tests/
│   └── test_components.py # 8 offline unit tests (budgeting, graph direction, fallbacks)
├── run_agent.py           # CLI entry point
└── README.md
```

---

## Roadmap

* [x] Directed `MultiDiGraph` with typed relationship predicates
* [x] Strict 4K token budget enforcement in Notebook memory
* [x] Evidence verification chain on final answers
* [ ] **OT/ICS Security (Purdue Model):** Ingest industrial control logs correlating Level 2 SCADA/HMI commands, Level 3.5 DMZ jump-hosts, and Level 1 PLC logic modifications.
* [ ] **Structured Log Parsers:** Ingestion for Sysmon JSON and Windows Event Logs (EVTX).
* [ ] **MITRE ATT&CK & ATT&CK for ICS:** Direct mapping of observed graph edges to ATT&CK techniques.

---

## Reference
- Li et al., *"GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models"*, arXiv:2406.14550 (2024).
