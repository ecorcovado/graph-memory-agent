# GraphMemory Agent

### An offline-first, verifiable investigation assistant for air-gapped OT/ICS incident response (Purdue Model Levels 4 → 0).

A lightweight Python forensic engine that enables small, local LLMs (such as Qwen 2.5 Coder 7B or Llama 3 on Ollama) to navigate multi-hop industrial intrusion evidence without exceeding a 4K token window or leaking sensitive telemetry to third-party cloud APIs.

Built from core concepts in the *GraphReader* paper ([arXiv:2406.14550](https://arxiv.org/abs/2406.14550)), with a specific focus on **air-gapped industrial control systems (ICS), Purdue Enterprise Reference Architecture (PERA / ISA-95) IT-to-OT conduit verification, and fail-closed forensic triage**.

---

## Why this exists

### 1. Vector RAG fails at multi-hop links
Vector search (cosine similarity on text embeddings) works when the answer is in one chunk. It breaks down when answering multi-stage questions like:
> *"Did the enterprise compromise reach the PLC, and what safety interlock was bypassed?"*

The enterprise endpoint log on `CORP-WS-019` mentions an IP, the DMZ firewall log links that IP to an internal mirror, the SCADA audit log connects the mirror to an engineering workstation, and the controller audit log proves the CIP program download. An embedding search for "PLC bypass" retrieves the controller log and leaps over intermediate network conduits, hallucinating a direct connection.

### 2. Massive context windows degrade and leak confidential data
Dumping 50,000 lines of SIEM logs into a 128k context window triggers "lost-in-the-middle" attention degradation. More importantly, in security operations (SOC/DFIR), you cannot leak client network logs, internal IPs, or industrial control logic to third-party cloud APIs.

This project runs **100% locally** via Ollama. It compiles structured log records directly into a directed `MultiDiGraph` with typed relationships (`Subject -> Relation -> Object`), uses an autonomous agent loop that hops across entities, and logs intermediate clues into a bounded scratchpad memory with strict token budgets.

---

## Modular Deterministic Ingestion Architecture

GraphMemory Agent implements a 9-stage, fail-closed forensic ingestion pipeline that compiles structured log records directly into verified graph facts without ungrounded fact generation:

```
                  ┌────────────────────────┐
                  │     Case Manifest      │
                  │  (case_manifest.json)  │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      Case Loader       │
                  │ (Path Traversal Guard) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Source SHA-256 Digest  │
                  │   & Record Identity    │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │    Format Detection    │
                  │ (Schema & File Header) │
                  └───────────┬────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
   ┌───────────────────┐             ┌───────────────────┐
   │  Sysmon Adapter   │             │   Zeek Adapter    │
   │(EID 1, 3, 11, 22) │             │ (conn, dns, alert)│
   └─────────┬─────────┘             └─────────┬─────────┘
             │                                 │
             └────────────────┬────────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Normalised Event Model │
                  │   (NormalizedEvent)    │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │ Atomic Fact Generator  │
                  │(Per-Source Record IDs) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │  Evidence Validation   │
                  │ (Source Trust Firewall)│
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │      MultiDiGraph      │
                  │ (NetworkX Purdue PERA) │
                  └───────────┬────────────┘
                              │
                              ▼
                  ┌────────────────────────┐
                  │   GraphMemory Agent    │
                  │ (Claim Audit & Traversal│
                  └────────────────────────┘
```

1. **Case Manifest:** Resolves case metadata, Purdue zones, and ground truth specifications.
2. **Case Loader:** Canonical path resolution (`os.path.realpath`) defending against path traversal attacks.
3. **Source SHA-256 and Record Identity:** Single byte snapshot hashing (`full_sha`) generating deterministic record identifiers (`{source_type}_{line_idx:04d}`).
4. **Format Detection (`FormatDetector`):** Detects schema types (Sysmon, Zeek, Windows Security, CIP Controller, Historian, Physical Telemetry).
5. **Specialized Adapters (`core/adapters/`):**
   - `SysmonAdapter`: Process creation (EID 1), Network connection (EID 3), File creation (EID 11), Registry (EID 12-14), DNS query (EID 22).
   - `ZeekAdapter`: Connection telemetry (`conn.log`), DNS queries (`dns.log`), and security notices (`notice.log`).
   - `WindowsSecurityAdapter`: Remote network logon (EID 4624), local logon, and privilege assignment (EID 4672).
   - `FirewallAdapter`, `HistorianAdapter`, `ControllerAuditAdapter`, `ProcessTelemetryAdapter`: Industrial DMZ, supervisory, controller logic, and physical interlock observations.
6. **Normalised Event Model (`NormalizedEvent`):** Canonical dataclass standardizing event times, subjects, predicates, objects, raw records, and Purdue levels.
7. **Atomic Fact Generator:** Generates zero-padded, collision-free fact IDs (`{source_id}_R{line}_{record_id}`) and assemblies rich forensic summaries.
8. **Evidence Validation:** Enforces the `TRUSTED_TELEMETRY_SOURCES` allowlist; untrusted sources fail closed to `UNCLASSIFIED`.
9. **MultiDiGraph:** Builds typed directed graph in NetworkX with asset Purdue levels and edge classifications.
10. **GraphMemory Agent:** Traverses graph with bounded notebook, verifying all citations against registered cryptographic digests.

---

## Key Technical Guardrails

* **String-Aware & Escape-Aware JSON Scanner:** Balanced-brace scanner (`extract_json_objects`) tracking string boundaries and backslash escapes, reliably extracting structured answers even with embedded braces or escaped quotes in conversational prose.
* **Plaintext & Structured Claim Validation:** Both JSON and plaintext answers are validated against gathered evidence. Claims asserting PLC logic modification strictly require predicate `modified_logic` from `CONTROLLER_AUDIT` on the target controller specified in the case manifest (e.g. `plc-ballast-01`); assertions of safety interlock suppression require physical interlock telemetry on the target actuator (e.g. `xv-4012`). Generated text cannot hijack or change the target asset defined in the manifest.
* **XSS-Hardened Forensic HTML Reports:** All dynamic text (queries, model responses, raw record excerpts) is sanitized with `html.escape()`. Ingestion completeness (`COMPLETE` / `DEGRADED`) is audited separately from source byte hashes and attack path continuity.
* **Relational Conduit Verification Across Purdue Zones:** Evaluates stage presence vs verified conduit transitions across Purdue levels (L4 $\rightarrow$ L3.5 $\rightarrow$ L2 $\rightarrow$ L1 $\rightarrow$ L0) to distinguish active stages from unbroken end-to-end compromise paths.
* **Deterministic Fail-Closed Degraded Policy:** If required telemetry sources (such as controller CIP audits) are missing from the case manifest, the agent refuses to speculate and outputs `INSUFFICIENT EVIDENCE`.

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
python run_agent.py --case examples/purdue_ot_incident --question "Did the enterprise compromise reach the PLC?" --report reports/aegis_investigation_report.html

# 4. Run the comparative benchmark suite (Keyword Search vs Vector RAG vs Graph vs Ours)
python benchmark.py

# 5. Run offline unit tests (59 passing tests in ~0.2s)
python -m unittest tests/test_components.py
```

---

## Comparative Investigation Benchmark (Project Aegis)

> **Evaluation Scope & Methodology:** Evaluated on the Project Aegis synthetic OT intrusion case study (a single-scenario proof-of-concept demonstrating multi-stage network conduit reconstruction across air-gapped Purdue zones). Metrics are calculated dynamically from actual retrieved chunks, edge traversals, and answer citations against the ground truth transitions (`examples/purdue_ot_incident/ground_truth/expected_attack_path.json`):

Target Query:
> *"Did the enterprise compromise reach the PLC, and what safety interlock was bypassed?"*

| Methodology | Attack Path Continuity | Citation Provenance | Hallucination Rate | Degraded Evidence Policy |
| :--- | :--- | :--- | :--- | :--- |
| **Keyword Search (Grep/BM25)** | 0% (Disconnected) | 0.0% | N/A (No Synthesis) | NO (Cannot detect gaps) |
| **TF-IDF Top-k Retrieval** | 25.0% (Broken at DMZ - 1/4 edges) | 0.0% (No Cryptographic Verification) | 75.0% (Missing Intermediate Conduits) | NO (Speculates) |
| **Unconstrained Graph (NetworkX)** | 100.0% (4/4 edges) | 0.0% (Topology Only, No Hash Audit) | 0.0% (Graph Topology Validated) | NO (No policy enforcement) |
| **GraphMemory Agent (Ours)** | **100% (Unbroken Purdue Chain)** | **100.0% (Cryptographic Hashes)** | **0.0% (Strict Allowlist Enforcement)** | **YES (Degraded Policy Enforced)** |

### Ablation: Why is an LLM Agent needed if the Graph already has 100% Path Continuity?

* **Graph Topology Alone (NetworkX):** Graph shortest-path and BFS algorithms verify whether a path exists between two nodes. However, raw topology is completely agnostic to payload semantics: it cannot inspect process command-lines, detect privilege abuse, extract modified logic signatures, or synthesize a human-readable DFIR incident explanation.
* **TF-IDF / Chunk Retrieval Alone:** Top-k similarity retrieves the initial enterprise compromise and the final PLC download (due to word overlap with the query), completely jumping over intermediate DMZ proxies, jump hosts, and SCADA gateways—hallucinating a false direct connection.
* **GraphMemory Agent:** Combines graph constraints with a small local LLM. The directed graph forces the agent to follow verified physical and logical conduits step-by-step; the LLM interprets process arguments, user credentials, and industrial payloads at each hop; and the notebook enforces bounded token budgets and cryptographic provenance citations.

*Run the benchmark suite locally:*
```bash
python benchmark.py
```

---

## OT/ICS Investigation Workspace & Forensic Reporting

GraphMemory Agent provides an analyst-first investigation experience:
* **Terminal Executive Dashboard:** Automatically prints the Purdue Zone attack progression (Level 4 through Level 0) with status tags (`[SUPPORTED]` vs `[UNVERIFIED]`) and conduit verification.
* **Standalone Interactive HTML Report:** Generates a self-contained, dark-mode cybersecurity investigation report (`reports/investigation_report.html`) containing:
  * Executive KPI summary cards (Ingestion Completeness, Source Byte Digests, Attack Path Continuity, Evidence Verification).
  * Purdue Enterprise Reference Architecture (PERA) attack progression cards with conduit verification status.
  * Direct expandable raw record excerpts with file coordinates (`sysmon.jsonl:L1`, `firewall.jsonl:L2`).
  * Cryptographic source acquisition table matching SHA-256 byte digests.

```bash
# Generate the interactive HTML report
python run_agent.py --case examples/purdue_ot_incident --question "Did the enterprise compromise reach the PLC?" --report reports/aegis_investigation_report.html
```

---

## Project Structure

```
graph-memory-agent/
├── core/
│   ├── adapters/          # Modular log adapters
│   │   ├── base.py        # BaseLogAdapter & NormalizedEvent dataclass
│   │   ├── format_detector.py # Deterministic schema & file format detection
│   │   ├── sysmon.py      # Sysmon EID 1, 3, 11, 12-14, 22 adapter
│   │   ├── zeek.py        # Zeek conn.log, dns.log, notice.log adapter
│   │   ├── windows_security.py # Windows Security EID 4624, 4672 adapter
│   │   └── industrial.py  # Firewall, Historian, CIP Controller & Process adapters
│   ├── case_loader.py     # Deterministic structured log ingestion & pipeline coordinator
│   ├── budget_manager.py  # Prompt budget manager (4K token context bounds)
│   ├── graph_builder.py   # Document SHA-256 hashes, typed triples, and NetworkX MultiDiGraph
│   ├── notebook.py        # Linked SourceReference memory with provenance-preserving merging
│   ├── reporter.py        # Executive terminal dashboard & interactive HTML forensic report generator
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
│   └── test_components.py # 59 offline unit tests (adapters, fail-closed auditing, benchmark, reports)
├── benchmark.py           # 4-way comparative evaluation suite (Keyword vs TF-IDF Top-k vs Graph vs Ours)
├── run_agent.py           # CLI entry point (supports --file, --case, and --report)
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
* [x] **v0.4 Deterministic Structured Ingestion:** `CaseLoader` and `StructuredLogParser` compiling Sysmon, Windows Security, Firewall, SCADA audit, and CIP controller logs into the graph
* [x] **v0.4.1 Forensic Ingestion Hardening:** Source Trust Firewall, Path Traversal Defense, Ingestion Completeness Reports, Raw Record Vault, and per-source stable fact IDs
* [x] **v0.4.2 Forensic Integration Closure:** Record-based Fact ID citation verifier, Historian telemetry trust policy, typed process observation nodes, byte snapshot parsing, and separated completeness metrics
* [x] **v0.4.3 Temporal and Evidence Integrity:** Enforced degraded evidence policies (fail-closed when required sources missing), collision-free deterministic observation IDs, claim-level predicate verification, operation-specific historian predicates, ISO-8601 UTC timestamp normalization
* [x] **v0.5.0 Investigation Workspace & Benchmark:** Executive terminal dashboard, standalone interactive HTML incident reports, and 4-way comparative evaluation benchmark
* [x] **v0.5.1 Forensic Evaluation & Modular Architecture:** 9-stage modular adapter architecture (`core/adapters/` with `SysmonAdapter`, `ZeekAdapter`), string-aware JSON scanner, plaintext claim validation, HTML XSS protection, and conduit continuity verification
* [x] **v0.5.2 Autonomous Agent Benchmark & Zeek TSV Ingestion:** Autonomous `agent.run()` benchmark execution, native Zeek TSV log parsing with `#fields` / `#types` headers and Unix epoch float timestamp normalization, exact target claim validation (`plc-ballast-01` & valve `xv-4012`), additive investigation requirements, consecutive prefix conduit continuity, and 55 passing unit tests
* [x] **v0.5.3 Immutable Target Binding & Empirical Benchmark Metrics:** Strict immutable target binding (preventing answer claims from overriding expected target controller/actuator), all-predicate enforcement (`all()` set difference logic), dedicated Zeek TSV fixtures (`conn.log`, `dns.log`, `notice.log`) with `#separator`, `#types`, and escaped value handling, dynamic benchmark metric calculations, and 59 passing offline unit tests
* [ ] **Automated ATT&CK Mapping:** Direct lookup mapping of observed edges to MITRE ATT&CK for ICS v20.0 technique IDs.

---

## Related Work & References

- Li et al., *"GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models"*, arXiv:2406.14550 (2024).
- NIST Special Publication 800-82 Revision 3, *"Guide to Operational Technology (OT) Security"* (2023/2026).
- MITRE ATT&CK® for Industrial Control Systems (ICS), Version 20.0 (Released October 13, 2026).
- Wang et al., *"CyberSleuth: Benchmarking LLM Agents for Cyber Incident Investigation"*, arXiv:2407.03719 (2024).
- Gao et al., *"OntoLogX: Ontology-Grounded Log Analysis and ATT&CK Mapping"*, arXiv:2410.02100 (2024).
- *"Timesketch Digital Forensics Agent Integration"*, Google Open Source Security / Black Hat USA (2025).
