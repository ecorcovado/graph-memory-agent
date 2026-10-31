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

* **Directed MultiDiGraph with Strict Forensic Tiers & Separated Evidence Basis:** Distinguishes relationship directions (`Host -> communicated_with -> IP` vs reverse) and enforces strict allowlists for forensic classifications (`OBSERVED`, `INTELLIGENCE_REPORTED`, `CO_OCCURRENCE`, `UNCLASSIFIED`). Crucially separates **predicate semantics** (what happened, e.g. `executed`) from **evidence basis** (how it is known, e.g. `DIRECT_TELEMETRY` vs `ANALYST_REPORTED`), ensuring telemetry events are distinguished from report narratives.
* **Document-Level Provenance & Collision-Resistant Fact IDs:** Generates stable fact IDs derived from the SHA-256 hash of the source document (`f"{source_id}_C{chunk_id:04d}_F{fact_counter:04d}"`), and stores the full SHA-256 digest in `kg.source_hashes` for post-investigation independent auditing.
* **Fully Linked Provenance Memory (`SourceReference`):** Working memory entries (`EvidenceNote`) contain immutable `SourceReference` records explicitly linking `source_id`, `source_record_id`, `fact_ids`, `source_location`, and exact textual `excerpt`. Corroborating duplicate summaries merges references without losing individual record provenance.
* **Bounded Prompt Construction & Explicit Context Ceilings:** Prompt assembly routes through `PromptBudgetManager` using conservative character-based heuristic budgeting (4 chars/token) alongside explicit Ollama API constraints (`"options": {"num_ctx": 4096, "num_predict": 500}`).
* **Strict Action Parsing & Resilient Forensic Routing:** Traversal decisions require exact full-line action matching (`re.fullmatch(r"ACTION:\s*TERMINATE", line)`), preventing false terminations from conversational context (e.g. *"Do not use ACTION: TERMINATE"*). Fallback hops route based on edge forensic tier priority (`OBSERVED` > `INTELLIGENCE_REPORTED` > `CO_OCCURRENCE`).
* **Deterministic Fail-Closed Evidence Verification:** Synthesized answers are audited against gathered fact IDs. If the model cites fabricated fact IDs or provides uncited claims, the system **fails closed** and returns `INSUFFICIENT EVIDENCE`.

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

# 3. Run against Project Aegis (Deterministic Structured Case Ingestion)
python run_agent.py --case examples/purdue_ot_incident --question "Did the enterprise compromise reach the PLC?"

# 4. Run against Project Aegis (Unstructured Incident Narrative)
python run_agent.py --file examples/purdue_ot_incident/incident_overview.txt --question "Which PLC was modified and what safety interlock was bypassed following the enterprise compromise?"

# 5. Run offline unit tests (40 passing tests)
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
│   ├── case_loader.py     # Deterministic structured log ingestion (Sysmon, FW, SCADA, CIP)
│   ├── budget_manager.py  # Prompt budget manager (4K token context bounds)
│   ├── graph_builder.py   # Document SHA-256 hashes, typed triples, and NetworkX MultiDiGraph
│   ├── notebook.py        # Linked SourceReference memory with provenance-preserving merging
│   └── agent.py           # Rational planner, full-line action parser, and fail-closed verifier
├── examples/
│   ├── cyber_incident_report.txt  # DFIR incident triage scenario
│   └── purdue_ot_incident/        # Project Aegis synthetic OT/ICS intrusion bundle
│       ├── case_manifest.json     # Case metadata (NIST SP 800-82 Rev. 3 & MITRE ATT&CK for ICS v20.0)
│       ├── incident_overview.txt  # Narrative incident report across Purdue Levels 4 -> 0
│       ├── enterprise/            # Level 4 Sysmon & Windows Security JSONL logs
│       ├── industrial_dmz/        # Level 3.5 DMZ Firewall, RDP, and Historian Mirror logs
│       ├── level3_operations/     # Level 3 Plant Historian audit logs
│       ├── level2_scada/          # Level 2 SCADA Gateway & Engineering Workstation logs
│       ├── level1_plc/            # Level 1 Controller CIP program download audit logs
│       ├── level0_process/        # Level 0 Sensor telemetry & safety valve trip events
│       └── ground_truth/          # Benchmark expected paths, atomic facts, and fail-closed answers
├── tests/
│   └── test_components.py # 40 offline unit tests (fail-closed auditing, complete vs degraded benchmark, claim integrity, temporal observations)
├── run_agent.py           # CLI entry point (supports --file and --case with degraded ingestion warning policy)
└── README.md
```

---

## Roadmap

* [x] Directed `MultiDiGraph` with typed relationship predicates
* [x] Strict 4K token budget enforcement in Notebook memory
* [x] Evidence verification chain on final answers
* [x] **Project Aegis:** Synthetic Purdue Model OT/ICS intrusion bundle (Levels 4 through 0) aligned to NIST SP 800-82 Rev. 3 & MITRE ATT&CK for ICS v20.0 (Oct 2026)
* [x] Document-level SHA-256 source identity and zero-padded fact IDs
* [x] Linked `SourceReference` immutable memory structures
* [x] Architectural separation of predicates from evidence bases (`DIRECT_TELEMETRY` vs `ANALYST_REPORTED`)
* [x] **v0.4 Deterministic Structured Ingestion:** `CaseLoader` and `StructuredLogParser` directly compiling Sysmon, Windows Security, Firewall, SCADA audit, and CIP controller logs into the graph with zero extraction latency
* [x] **v0.4.1 Forensic Ingestion Hardening:** Source Trust Firewall (preventing untrusted telemetry assertion), Path Traversal Defense, Ingestion Completeness Reports, Raw Record Vault, and per-source stable fact IDs
* [x] **v0.4.2 Forensic Integration Closure:** Record-based Fact ID citation verifier (exact set membership & JSON support), Historian telemetry trust policy (`HISTORIAN_MIRROR` & `HISTORIAN_AUDIT`), typed process observation nodes (Purdue Level 0 `has_observation`, `measured_value`, `commanded_state`, `actual_state`, `interlock_status`), byte snapshot parsing, and separated ingestion completeness metrics
* [x] **v0.4.3 Temporal and Evidence Integrity:** Enforced degraded evidence policies (fail-closed when required sources missing), collision-free deterministic observation IDs per source record, claim-level predicate verification (validating statements against cited fact predicates), operation-specific historian predicates (`queried_tag_store`, `replicated_tags_to`), ISO-8601 UTC timestamp validation/normalization, and raw line retention in vault
* [x] **Project Aegis End-to-End Investigation Benchmark:** Complete vs degraded case benchmark asserting verified PLC logic download with citations under complete telemetry, and fail-closed rejection when controller audit is missing
* [ ] **Automated ATT&CK Mapping:** Direct lookup mapping of observed edges to MITRE ATT&CK for ICS v20.0 technique IDs.

---

## References
- Li et al., *"GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models"*, arXiv:2406.14550 (2024).
- NIST Special Publication 800-82 Revision 3, *"Guide to Operational Technology (OT) Security"* (2023/2026).
- MITRE ATT&CK® for Industrial Control Systems (ICS), Version 20.0 (Released October 13, 2026).
