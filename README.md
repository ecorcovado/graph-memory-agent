# GraphMemory Agent

### An offline-first, verifiable investigation assistant for air-gapped OT/ICS incident response (Purdue Model Levels 4 → 0).

A lightweight Python forensic engine that enables small, local LLMs (such as Qwen 2.5 Coder 7B or Llama 3 on Ollama) to navigate multi-hop industrial intrusion evidence without exceeding a 4K token window or leaking sensitive telemetry to third-party cloud APIs.

Built from core concepts in the *GraphReader* paper ([arXiv:2406.14550](https://arxiv.org/abs/2406.14550)), with a specific focus on **air-gapped industrial control systems (ICS), Purdue Enterprise Reference Architecture (PERA / ISA-95) IT-to-OT conduit verification, and fail-closed forensic triage**.

---

## Why this exists

### 1. Flat chunk retrieval fails at multi-hop links
Flat retrieval (lexical TF-IDF or text embeddings) works when the answer is in one chunk. It breaks down when answering multi-stage questions like:
> *"Did the enterprise compromise reach the PLC, and what safety interlock was bypassed?"*

The enterprise endpoint log on `CORP-WS-019` mentions an IP, the DMZ firewall log links that IP to an internal mirror, the SCADA audit log connects the mirror to an engineering workstation, and the controller audit log proves the CIP program download. A flat search for "PLC bypass" retrieves the controller log and leaps over intermediate network conduits, hallucinating a direct connection.

### 2. Massive context windows degrade and leak confidential data
Dumping 50,000 lines of SIEM logs into a 128k context window triggers "lost-in-the-middle" attention degradation. More importantly, in security operations (SOC/DFIR), you cannot leak client network logs, internal IPs, or industrial control logic to third-party cloud APIs.

This project runs **100% locally** via Ollama. It compiles structured log records directly into a directed `MultiDiGraph` with typed relationships (`Subject -> Relation -> Object`), uses an autonomous agent loop that hops across entities, and logs intermediate clues into a bounded scratchpad memory with strict token budgets.

---

## Modular Deterministic Ingestion Architecture

GraphMemory Agent implements a 10-stage, fail-closed forensic ingestion and investigation pipeline that compiles structured log records directly into verified graph facts without ungrounded fact generation:

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
   │ (EID 1-3, 11-14)  │             │ (conn, dns, alert)│
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
3. **Source Ingestion Integrity & Tamper Detection:** Single byte snapshot hashing (`full_sha`) generating deterministic record identifiers (`{source_type}_{line_idx:04d}`). Guarantees post-acquisition file integrity, verifying that evidence files have not been modified since acquisition on disk.
4. **Format Detection (`FormatDetector`):** Detects schema types (Sysmon, Zeek, Windows Security, CIP Controller, Historian, Physical Telemetry).
5. **Specialized Adapters (`core/adapters/`):**
   - `SysmonAdapter`: Process creation (EID 1), Network connection (EID 3), File creation (EID 11), Registry (EID 12-14), DNS query (EID 22).
   - `ZeekAdapter`: Connection telemetry (`conn.log`), DNS queries (`dns.log`), and security notices (`notice.log`), supporting native TSV formats (`#separator`, `#set_separator`, `#fields`, `#types`, hex escape unescaping, and float epoch timestamp normalization).
   - `WindowsSecurityAdapter`: Remote network logon (EID 4624), local logon, and privilege assignment (EID 4672).
   - `FirewallAdapter`, `HistorianAdapter`, `ControllerAuditAdapter`, `ProcessTelemetryAdapter`: Industrial DMZ, supervisory, controller logic, and physical interlock observations.
6. **Normalised Event Model (`NormalizedEvent`):** Canonical dataclass standardizing event times, subjects, predicates, objects, raw records, and Purdue levels.
7. **Atomic Fact Generator:** Generates zero-padded, collision-free fact IDs (`{source_id}_R{line}_{record_id}`) and assemblies rich forensic summaries.
8. **Evidence Validation:** Enforces the `TRUSTED_TELEMETRY_SOURCES` allowlist; untrusted sources fail closed to `UNCLASSIFIED`.
9. **MultiDiGraph:** Builds typed directed graph in NetworkX with asset Purdue levels and edge classifications.
10. **GraphMemory Agent:** Traverses graph with bounded notebook, verifying all citations against registered cryptographic digests.

---

## Key Technical Guardrails

* **Three-Verdict Forensic Framework (SUPPORTED / REFUTED / INSUFFICIENT):** Distinctly separates positive defense verification from missing telemetry:
  * `SUPPORTED`: An unbroken conduit chain connects the enterprise breach to target OT assets, validated by record-level citations.
  * `REFUTED`: Positive security telemetry (e.g. firewall rule `action: BLOCK` / `blocked_connection_to`) verifies that lateral movement was actively blocked by boundary controls before reaching industrial assets.
  * `INSUFFICIENT EVIDENCE`: Critical telemetry sources are missing from case ingestion or conduits contain unmonitored gaps without positive proof of blockage.
* **Blocked-Edge Traversal Semantics:** Distinguishes evidence exploration from attack-path compromise progression. Edges representing boundary defenses (`blocked_connection_to`, `connection_denied`, firewall `action: BLOCK`) are preserved as immutable evidence records, but cannot be traversed as lateral movement transitions. The agent recognizes boundary enforcement and halts attack-path reconstruction at the perimeter.
* **Per-Claim Target Binding & Independent Validation:** Every entity asserted in unstructured text or structured JSON is validated independently against verified atomic facts. A verified fact for `plc-ballast-01` cannot substantiate an additional unsupported assertion regarding `plc-other-01`. Furthermore, unsupported positive claims cannot bypass verification by being combined with `INSUFFICIENT EVIDENCE`.
* **String-Aware & Escape-Aware JSON Scanner:** Balanced-brace scanner (`extract_json_objects`) tracking string boundaries and backslash escapes, reliably extracting structured answers even with embedded braces or escaped quotes in conversational prose.
* **Plaintext & Structured Claim Validation:** Both JSON and plaintext answers are validated against gathered evidence. Claims asserting PLC logic modification strictly require predicate `modified_logic` from `CONTROLLER_AUDIT` on the target controller specified in the query or case manifest (e.g. `plc-ballast-01`); assertions of safety interlock suppression require physical interlock telemetry on the target actuator (e.g. `xv-4012`). Expected controller and actuator are dynamically bound with **zero hardcoded fallback**; if unspecified in both the query and case manifest, assertions of controller modification immediately fail closed (`INSUFFICIENT EVIDENCE`).
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

# 5. Run offline unit tests (66 passing tests in ~0.2s)
python -m unittest tests/test_components.py
```

---

## Comparative Investigation Benchmark (Project Aegis)

> **Evaluation Scope & Methodology:** Evaluated on the Project Aegis synthetic OT intrusion testbed (demonstrating multi-stage network conduit reconstruction across air-gapped Purdue zones). Metrics are calculated dynamically on each execution directly from retrieved chunks, edge traversals, and cited atomic facts against ground-truth transitions (`examples/purdue_ot_incident/ground_truth/expected_attack_path.json`):

### Part 1: Positive Intrusion Scenario (Multi-Stage Purdue Compromise)

Target Query:
> *"Did the enterprise compromise reach the PLC, and what safety interlock was bypassed?"*

| Methodology | Purdue Conduit Continuity | Source Digest Coverage | Conduit Recall / Gap | Policy Enforcement |
| :--- | :--- | :--- | :--- | :--- |
| **Keyword Search (Grep/BM25)** | 0% (Disconnected) | N/A (No Digest Tracking) | 0.0% (4/4 edges missing) | NO (Cannot detect gaps) |
| **TF-IDF Top-k Retrieval** | 25.0% (Broken at DMZ - 1/4 edges) | N/A (No Digest Tracking) | 25.0% (3/4 edges missing) | NO (Speculates without verification) |
| **Unconstrained Graph (NetworkX)** | 100.0% (4/4 edges) | N/A (Topology Only, No Hashes) | 100.0% (Topology Overmatch) | NO (No policy enforcement) |
| **GraphMemory Agent (Ours)** | **100% (Unbroken Chain - 4/4 edges)** | **100.0% (SHA-256 Verified)** | **100.0% (4/4 edges verified)** | **PASS (3-Verdict Enforced)** |

> **Benchmark Metric Calculation Details:**
> * **TF-IDF Conduit Recall (25.0%):** Evaluated against the 4 ground-truth transitions. Top-k similarity matches 1 edge (`eng-ws-02` $\rightarrow$ `plc-ballast-01` due to keyword overlap with "PLC"), but misses 3 intermediate conduits (`corp-ws-019` $\rightarrow$ `dmz`, `dmz` $\rightarrow$ `scada`, and `plc` $\rightarrow$ `xv-4012`). The conduit gap rate is $75.0\%$ (3 of 4 edges missing), which causes flat retrieval to leap over the DMZ.
> * **Unconstrained Graph (100.0% Continuity / 0.0% Provenance):** Because all 4 topological transitions exist in the ingested graph, NetworkX pathfinding finds $4/4$ edges ($100\%$ topological recall). However, topology alone carries $0.0\%$ cryptographic provenance and completely lacks fail-closed missing evidence detection.

### Part 2: Negative Control Ablation (Intrusion Halted at Level 3.5 DMZ)

A critical failure mode in OT incident triage is *analyst and LLM confirmation bias*—falsely asserting that an intrusion reached Level 1 PLCs simply because industrial terms appear in the query or background process telemetry.

To measure this, `benchmark.py` evaluates a dedicated negative scenario (`examples/purdue_ot_incident_halted_dmz`) where the adversary breaches Level 4 and reaches the DMZ Historian Mirror, but egress into Level 2 SCADA is blocked by firewall rule `PERA-RULE-L35-TO-L2-BLOCK` (`action: BLOCK`). Normal background operations continue at Level 1/0 (routine CIP reads, valve interlocks remain `ARMED_AND_ACTIVE`).

Target Query:
> *"Did the enterprise compromise reach the PLC, and what safety interlock was bypassed?"*

| Evaluation Metric | Keyword Search | TF-IDF Top-k Retrieval | Unconstrained Graph | GraphMemory Agent (Ours) |
| :--- | :--- | :--- | :--- | :--- |
| **Purdue Continuity** | 0.0% (Disconnected) | 0.0% (No Bridge) | 100.0% (Topology Overmatch) | **25.0% (Halted at Level 3.5)** |
| **Conduit Integrity** | Broken | Broken | Undetected | **VERIFIED HALT AT DMZ** |
| **Forensic Verdict** | Disconnected | False Affirmation | False Affirmation | **REFUTED (Positive Defense Block)** |
| **Refutation Rate** | N/A (No synthesis) | 0/1 (Speculates) | 0/1 (Traverses) | **1/1 (100.0% Fail-Closed)** |
| **False Positive Breach Claims** | N/A | High (Synthesizes compromise) | High (Affirms operational link) | **0/1 (0.0% Zero Hallucination)** |

### Part 3: LLM vs Deterministic Rule-Stub Ablation (Isolating Model Contribution)

To rigorously separate the LLM's contribution from the deterministic graph rules, `benchmark.py` includes an ablation that replaces the local LLM with a template rule stub (`TemplateStubLLM`):

| Evaluation Condition | Autonomous LLM Agent | Template Rule Stub | Architectural Finding |
| :--- | :--- | :--- | :--- |
| **Positive Aegis Intrusion** | `SUPPORTED` (Verified) | `SUPPORTED` (Verified) | Security verdicts are identical between LLM and Stub. |
| **Negative Halted Intrusion** | `REFUTED` (Blocked) | `REFUTED` (Blocked) | Security boundaries and fail-closed rules are 100% deterministic. |
| **Degraded Telemetry Case** | `INSUFFICIENT EVIDENCE` | `INSUFFICIENT EVIDENCE` | Verifier prevents ungrounded speculation regardless of model weights. |
| **Forensic Prose Generation** | Dynamic contextual synthesis | Static template strings | The LLM's true contribution is semantic query parsing and human-readable prose. |

*Run the complete 3-part benchmark suite locally:*
```bash
python benchmark.py
```

---

## Known Limitations & Forensic Boundary Conditions

To maintain engineering integrity, the following limitations are explicitly acknowledged:

1. **Synthetic Proof-of-Concept Scenarios (Shared Authorship):** Both the positive intrusion scenario (`purdue_ot_incident`) and the negative control ablation (`purdue_ot_incident_halted_dmz`) were authored by the project maintainer. While they faithfully model real-world Purdue architectures (PERA / ISA-95) and realistic log formats (Sysmon, Zeek, Rockwell CIP), independent validation on third-party authored testbeds is required to prove generalized traversal.
2. **Absence of Real-World Log Noise & Entity Resolution:** The Project Aegis testbed uses clean, canonical identifiers (`corp-ws-019`, `plc-ballast-01`). Real-world industrial environments exhibit high background noise, DHCP IP reassignment, multi-homed dual-NIC assets, and hostname variations that require probabilistic entity resolution before graph construction.
3. **No Live Industrial Telemetry:** In accordance with Australian critical infrastructure protections (Security of Critical Infrastructure Act 2018) and strict industrial NDAs, live operational logs from operating mines, ports, or LNG refineries cannot be publicly distributed. High-fidelity synthetic logs are used in line with industry research standards.
4. **Scope of the LLM vs Deterministic Verifier:** The LLM does not establish forensic ground truth. As proven in the Part 3 ablation, ground truth verdicts (`SUPPORTED`, `REFUTED`, `INSUFFICIENT EVIDENCE`) are governed by deterministic graph traversal and set-membership verification. The LLM serves solely as a flexible natural-language interface and narrative synthesizer.
5. **Disk Snapshot Integrity vs Upstream Log Authenticity:** The SHA-256 byte digests guarantee post-acquisition file integrity (tamper detection once files are acquired on disk). They do not prove that an upstream Windows kernel or logging daemon was not tampered with prior to log generation.

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
│   ├── purdue_ot_incident/        # Project Aegis synthetic OT/ICS intrusion bundle (Levels 4 -> 0)
│   │   ├── case_manifest.json     # Case metadata (NIST SP 800-82 Rev. 3 & MITRE ATT&CK for ICS v20.0)
│   │   ├── incident_overview.txt  # Narrative incident report across Purdue Levels 4 -> 0
│   │   ├── enterprise/            # Level 4 Sysmon & Windows Security JSONL logs
│   │   ├── industrial_dmz/        # Level 3.5 DMZ Firewall, RDP, and Historian Mirror logs
│   │   ├── level3_operations/     # Level 3 Plant Historian audit logs
│   │   ├── level2_scada/          # Level 2 SCADA Gateway & Engineering Workstation logs
│   │   ├── level1_plc/            # Level 1 Controller CIP program download audit logs
│   │   ├── level0_process/        # Level 0 Sensor telemetry & safety valve trip events
│   │   └── ground_truth/          # Benchmark expected paths, atomic facts, and fail-closed answers
│   └── purdue_ot_incident_halted_dmz/ # Negative Control Scenario (Adversary halted at Level 3.5 DMZ)
│       ├── case_manifest.json     # Manifest with investigation_targets and zero-compromise ground truth
│       ├── enterprise/            # Level 4 Initial access and lateral attempt
│       ├── industrial_dmz/        # Level 3.5 Firewall rule blocking SCADA pivot (BLOCK)
│       ├── level1_plc/            # Level 1 Routine CIP tag polling (0x4C_READ_TAG, unmodified logic)
│       └── level0_process/        # Level 0 Nominal sensors and ARMED_AND_ACTIVE safety interlocks
├── tests/
│   └── test_components.py # 62 offline unit tests (adapters, zero-fallback targets, negative ablation, reports)
├── benchmark.py           # 2-part comparative evaluation suite (Positive multi-stage + Negative DMZ ablation)
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
* [x] **v0.5.4 Air-Gapped ICS Positioning & Manifest Targets:** Calibrated DFIR/OT problem framing around air-gapped triage without SaaS dependencies, schema-driven manifest target declarations in `case_manifest.json`, and dynamic benchmark reporting
* [x] **v0.5.5 Verified Academic Bibliography, Zero-Fallback Target Binding & Negative Control Ablation:** Fully verified and audited academic references (EMNLP 2024 GraphReader, arXiv:2508.20643 CyberSleuth, arXiv:2510.01409 OntoLogX, Black Hat USA 2025 Timesketch AI Agent, MITRE ATT&CK for ICS v20.0); eliminated all hardcoded target fallbacks in favor of strict query and manifest binding; added negative-control ablation scenario (`examples/purdue_ot_incident_halted_dmz`) proving 100% deterministic refusal on halted DMZ intrusions; and 62 passing offline unit tests
* [ ] **Automated ATT&CK Mapping:** Direct lookup mapping of observed edges to MITRE ATT&CK for ICS v20.0 technique IDs.

---

## Related Work & References

- Li et al., *"GraphReader: Building Graph-based Agent to Enhance Long-Context Abilities of Large Language Models"*, EMNLP 2024, arXiv:2406.14550 (2024).
- Fumero, Huang, Boffa, Giordano, Mellia, and Rossi, *"CyberSleuth: Autonomous Blue-Team LLM Agent for Web Attack Forensics"*, arXiv:2508.20643 (2025).
- Cotti, Drago, Rula, Bianchini, and Cerutti, *"OntoLogX: Ontology-Guided Knowledge Graph Extraction from Cybersecurity Logs with Large Language Models"*, arXiv:2510.01409 (2025) / *Advanced Intelligent Systems*.
- NIST Special Publication 800-82 Revision 3, *"Guide to Operational Technology (OT) Security"* (2023/2026).
- MITRE ATT&CK® for Industrial Control Systems (ICS), Version 20.0 (Released October 13, 2026).
- Kantchelian, van Dantzig, Kramer, Baeriswyl, and Köpper, *"Autonomous Timeline Analysis and Threat Hunting: An AI Agent for Timesketch"*, Google Open Source Security / Black Hat USA (2025).
