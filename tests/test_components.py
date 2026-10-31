"""
Comprehensive unit tests for GraphMemory-Agent core components.
Ensures MultiDiGraph directed construction, strict allowlist edge tiers,
fail-closed evidence auditing, multi-source provenance merging, and budget assembly function offline.
"""

import json
import os
import unittest
from core.notebook import Notebook, EvidenceNote, SourceReference
from core.graph_builder import (
    AtomicFact, KnowledgeGraph, normalize_entity, classify_relation,
    classify_relation_forensics, ForensicRelation, GraphBuilder,
    EVIDENCE_BASIS_TELEMETRY, EVIDENCE_BASIS_REPORTED, EVIDENCE_BASIS_INTEL,
    EVIDENCE_BASIS_UNCLASSIFIED, TRUSTED_TELEMETRY_SOURCES,
)
from core.case_loader import CaseLoader, NormalizedEvent, StructuredLogParser
from core.budget_manager import PromptBudgetManager
from core.agent import GraphReaderAgent


class MockOllamaClient:
    """Mock LLM client for offline deterministic unit testing."""
    def __init__(self, responses=None, model="mock-model"):
        self.responses = responses or {}
        self.call_history = []
        self.model = model

    def generate(self, prompt: str, temperature: float = 0.1, system: str = "") -> str:
        self.call_history.append((system, prompt))
        for key, val in self.responses.items():
            if key in prompt:
                return val
        return "mocked_response"


class TestGraphMemoryAgent(unittest.TestCase):

    def test_prompt_budget_manager_assembly_and_clamping(self):
        sys_p, user_p, tokens = PromptBudgetManager.assemble_prompt(
            system_text="System instructions.",
            query_and_plan="Question: What happened?",
            notebook_text="Note: Evidence found.",
            context_text="Context: Log lines.",
            instructions="Output answer.",
        )
        self.assertLessEqual(tokens, PromptBudgetManager.MAX_PROMPT_BUDGET)
        self.assertIn("Evidence found", user_p)

        # Test long instructions clamping
        long_inst = "Instruction " * 200
        _, user_p2, _ = PromptBudgetManager.assemble_prompt(
            query_and_plan="Query",
            instructions=long_inst,
        )
        self.assertIn("TRUNCATED", user_p2)

    def test_fact_extraction_uses_prompt_budget_manager(self):
        client = MockOllamaClient(responses={"Text to Deconstruct": "Fact 1 | host | executed | proc"})
        builder = GraphBuilder(client)
        builder.extract_atomic_facts("Sample chunk", chunk_id=0)

        # Verify that assemble_prompt was routed and called
        self.assertTrue(len(client.call_history) > 0)
        sys_called, prompt_called = client.call_history[0]
        self.assertIn("expert DFIR and Knowledge Graph analyst", sys_called)
        self.assertIn("Text to Deconstruct", prompt_called)

    def test_entity_normalization(self):
        self.assertEqual(normalize_entity("  WORKSTATION-402  "), "workstation-402")
        self.assertEqual(normalize_entity('"APT-SilverFox"'), "apt-silverfox")
        self.assertEqual(normalize_entity("System Health   Check"), "system health check")

    def test_strict_exact_predicate_allowlist(self):
        # Known observed events
        self.assertEqual(classify_relation("communicated_with"), "OBSERVED")
        self.assertEqual(classify_relation("executed"), "OBSERVED")
        self.assertEqual(classify_relation("modified_registry"), "OBSERVED")

        # Threat intel / context
        self.assertEqual(classify_relation("attributed_to"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("known_technique"), "INTELLIGENCE_REPORTED")

        # Co-occurrence
        self.assertEqual(classify_relation("co_occurs_with"), "CO_OCCURRENCE")

        # Substrings of observed predicates must NEVER be classified as OBSERVED
        self.assertEqual(classify_relation("not_executed"), "UNCLASSIFIED")
        self.assertEqual(classify_relation("unconnected"), "UNCLASSIFIED")
        self.assertEqual(classify_relation("registry_unmodified"), "UNCLASSIFIED")
        self.assertEqual(classify_relation("custom_random_predicate"), "UNCLASSIFIED")

    def test_biographical_predicates_not_observed(self):
        # Biographical facts must be INTELLIGENCE_REPORTED (asserted facts), never live OBSERVED telemetry
        self.assertEqual(classify_relation("founded_inside"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("architected"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("previously_worked_at"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("worked_at"), "INTELLIGENCE_REPORTED")

    def test_missing_edge_metadata_defaults_unclassified(self):
        kg = KnowledgeGraph()
        # Add an edge with NO classification attribute
        kg.graph.add_edge("nodeA", "nodeB", relation="some_link")

        rel_data = kg.get_neighbors_with_relations("nodeA")
        self.assertEqual(len(rel_data), 1)
        direction, rel, cls, target = rel_data[0]
        self.assertEqual(cls, "UNCLASSIFIED")

    def test_multi_source_provenance_preserving_deduplication(self):
        nb = Notebook(max_tokens=500)
        # Note from Source A with C0_F1 and excerpt A
        nb.add_note(step=1, source_node="nodeA", content="Identical summary of event.", fact_ids=["C0_F1"], source_excerpt="Excerpt A")
        # Same summary from Source B with C1_F3 and excerpt B
        nb.add_note(step=2, source_node="nodeB", content="Identical summary of event.", fact_ids=["C1_F3"], source_excerpt="Excerpt B")

        # Notes list should have 1 merged note preserving BOTH source nodes, fact IDs, and excerpts
        self.assertEqual(len(nb.notes), 1)
        note = nb.notes[0]
        self.assertIn("nodeA", note.source_nodes)
        self.assertIn("nodeB", note.source_nodes)
        self.assertIn("C0_F1", note.fact_ids)
        self.assertIn("C1_F3", note.fact_ids)
        self.assertIn("Excerpt A", note.source_excerpts)
        self.assertIn("Excerpt B", note.source_excerpts)

    def test_notebook_token_budget_and_eviction(self):
        nb = Notebook(max_tokens=50, max_single_note_tokens=40)
        self.assertTrue(nb.is_empty())

        nb.add_note(step=1, source_node="node1", content="Note 1: " + "a" * 120, fact_ids=["C0_F1"])
        nb.add_note(step=2, source_node="node2", content="Note 2: " + "b" * 120, fact_ids=["C0_F2"])

        self.assertLessEqual(nb.estimate_tokens(), 50)
        formatted = nb.format_for_prompt()
        self.assertNotIn("Note 1", formatted)
        self.assertIn("Note 2", formatted)
        self.assertTrue(nb.verify_citation("C0_F2"))

    def test_multidigraph_typed_directed_edges(self):
        kg = KnowledgeGraph()
        fact = AtomicFact(
            fact_id="C0_F1",
            chunk_id=0,
            text="WORKSTATION-402 communicated with 198.51.100.44",
            subject="workstation-402",
            relation="communicated_with",
            object_entity="198.51.100.44",
            key_elements=["workstation-402", "198.51.100.44"],
        )
        kg.atomic_facts[fact.fact_id] = fact
        kg.graph.add_edge(
            "workstation-402",
            "198.51.100.44",
            relation="communicated_with",
            classification="OBSERVED",
            fact_id=fact.fact_id,
        )

        self.assertTrue(kg.graph.has_edge("workstation-402", "198.51.100.44"))
        self.assertFalse(kg.graph.has_edge("198.51.100.44", "workstation-402"))

        rel_data = kg.get_neighbors_with_relations("workstation-402")
        self.assertEqual(len(rel_data), 1)
        direction, rel, cls, target = rel_data[0]
        self.assertEqual(direction, "outgoing")
        self.assertEqual(rel, "communicated_with")
        self.assertEqual(cls, "OBSERVED")
        self.assertEqual(target, "198.51.100.44")

    def test_ranked_candidate_neighbors(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("workstation-402")
        kg.graph.add_node("198.51.100.44")
        kg.graph.add_node("secondary_concept")

        kg.graph.add_edge("workstation-402", "198.51.100.44", relation="communicated_with", classification="OBSERVED")
        kg.graph.add_edge("workstation-402", "secondary_concept", relation="co_occurs_with", classification="CO_OCCURRENCE")

        client = MockOllamaClient()
        agent = GraphReaderAgent(client, kg)

        candidates = ["secondary_concept", "198.51.100.44"]
        ranked = agent.rank_candidate_neighbors("workstation-402", candidates, "What IP communicated?")
        self.assertEqual(ranked[0], "198.51.100.44")

    def test_empty_initial_node_selection_does_not_match_first_node(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("alpha_node")
        kg.graph.add_node("beta_node")
        kg.graph.add_node("gamma_node")

        # Mock returns empty string
        client = MockOllamaClient(responses={"Available Graph Nodes": ""})
        agent = GraphReaderAgent(client, kg)

        # Empty selection must not match alpha_node via empty substring; must fall back to keyword overlap
        selected = agent.select_initial_node(
            question="Where is gamma located?",
            plan="Investigate gamma"
        )
        self.assertEqual(selected, "gamma_node")

    def test_do_not_terminate_does_not_stop_exploration(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("nodeA")
        kg.graph.add_node("nodeB")
        kg.graph.add_edge("nodeA", "nodeB", relation="connected_to", classification="OBSERVED")

        # LLM response contains "DO NOT TERMINATE"
        client = MockOllamaClient(responses={
            "Decision": "DO NOT TERMINATE! ACTION: HOP | nodeB"
        })
        agent = GraphReaderAgent(client, kg)

        next_node, should_terminate = agent.explore_node(
            current_node="nodeA",
            question="Find node?",
            plan="Traverse",
            step=1,
            visited_nodes=set()
        )
        self.assertFalse(should_terminate)
        self.assertEqual(next_node, "nodeB")

    def test_exact_traversal_and_visited_safeguard(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("workstation-402")
        kg.graph.add_node("workstation-402-backup")
        kg.graph.add_edge("workstation-402", "workstation-402-backup", relation="connected_to", classification="OBSERVED")

        client = MockOllamaClient(responses={
            "Decision": "ACTION: HOP | workstation"
        })
        agent = GraphReaderAgent(client, kg)

        next_node, should_term = agent.explore_node(
            current_node="workstation-402",
            question="Where to go?",
            plan="Explore",
            step=1,
            visited_nodes={"workstation-402"}
        )
        self.assertEqual(next_node, "workstation-402-backup")

    def test_evidence_audit_fail_closed_on_fabricated_id(self):
        kg = KnowledgeGraph()
        client = MockOllamaClient(responses={
            "Verified Immutable Fact IDs": "The attacker used task SystemHealthCheck supported by fact C99_F99."
        })
        agent = GraphReaderAgent(client, kg)
        agent.notebook.add_note(step=1, source_node="host", content="Valid note", fact_ids=["C0_F1"])

        answer = agent.synthesize_answer("What task was used?")
        self.assertIn("INSUFFICIENT EVIDENCE", answer)
        self.assertIn("C99_F99", answer)

    def test_evidence_audit_fail_closed_on_uncited_claims(self):
        kg = KnowledgeGraph()
        client = MockOllamaClient(responses={
            "Verified Immutable Fact IDs": "The attacker definitely used SystemHealthCheck with complete certainty."
        })
        agent = GraphReaderAgent(client, kg)
        agent.notebook.add_note(step=1, source_node="host", content="Valid note", fact_ids=["C0_F1"])

        answer = agent.synthesize_answer("What task was used?")
        self.assertIn("INSUFFICIENT EVIDENCE", answer)
        self.assertIn("no verifiable citations", answer)


    def test_document_level_source_hash_and_fact_ids(self):
        client = MockOllamaClient(responses={"Text to Deconstruct": "Process executed | host01 | executed | payload.exe"})
        builder = GraphBuilder(client)
        sample_doc = "Initial incident log line.\nProcess executed on host01."
        kg = builder.build_graph(sample_doc)

        # 1. Document SHA-256 must be calculated and stored
        import hashlib
        expected_sha = hashlib.sha256(sample_doc.encode("utf-8")).hexdigest()
        source_id = list(kg.source_hashes.keys())[0]
        self.assertEqual(kg.source_hashes[source_id], expected_sha)

        # 2. Fact ID must incorporate document source ID and zero-padded ordinals
        self.assertTrue(len(kg.atomic_facts) > 0)
        first_fact_id = list(kg.atomic_facts.keys())[0]
        self.assertTrue(first_fact_id.startswith(source_id))
        self.assertIn("_C0000_F0001", first_fact_id)

    def test_source_reference_linking_in_evidence_notes(self):
        nb = Notebook(max_tokens=500)
        ref1 = SourceReference(
            source_id="sha_doc_1",
            source_record_id="sysmon_evt_101",
            fact_ids=("fact_01", "fact_02"),
            excerpt="powershell.exe executed with EncodedCommand",
            source_location="CORP-WS-019",
        )
        nb.add_note(
            step=1,
            content="Host executed obfuscated PowerShell.",
            source_refs=[ref1],
        )

        self.assertEqual(len(nb.notes), 1)
        note = nb.notes[0]
        # Direct linked SourceReference verification
        self.assertEqual(len(note.source_refs), 1)
        self.assertEqual(note.source_refs[0].source_id, "sha_doc_1")
        self.assertEqual(note.source_refs[0].source_record_id, "sysmon_evt_101")
        self.assertIn("fact_01", note.source_refs[0].fact_ids)
        # Helper property compatibility
        self.assertIn("CORP-WS-019", note.source_nodes)
        self.assertIn("fact_01", note.fact_ids)

    def test_separated_predicate_and_evidence_basis(self):
        # 1. Telemetry vs Report differentiation
        tier_tel, basis_tel = classify_relation_forensics("executed", source_type="SYSMON_EVENT")
        self.assertEqual(tier_tel, "OBSERVED")
        self.assertEqual(basis_tel, EVIDENCE_BASIS_TELEMETRY)

        tier_rep, basis_rep = classify_relation_forensics("executed", source_type="INCIDENT_REPORT")
        self.assertEqual(tier_rep, "OBSERVED")
        self.assertEqual(basis_rep, EVIDENCE_BASIS_REPORTED)

        # 2. Intelligence attribution differentiation
        tier_intel, basis_intel = classify_relation_forensics("attributed_to")
        self.assertEqual(tier_intel, "INTELLIGENCE_REPORTED")
        self.assertEqual(basis_intel, EVIDENCE_BASIS_INTEL)

        # 3. ForensicRelation tuple backward compatibility (unpackable as 4 items)
        rel = ForensicRelation("outgoing", "executed", "OBSERVED", "proc.exe", evidence_basis=basis_tel, source_type="SYSMON_EVENT")
        direction, predicate, classification, target = rel
        self.assertEqual(direction, "outgoing")
        self.assertEqual(predicate, "executed")
        self.assertEqual(classification, "OBSERVED")
        self.assertEqual(target, "proc.exe")
        self.assertEqual(rel.evidence_basis, EVIDENCE_BASIS_TELEMETRY)
        self.assertEqual(rel.source_type, "SYSMON_EVENT")

    def test_strict_full_line_action_parsing_negative_case(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("plc-controller-01")
        kg.graph.add_node("dmz-historian")
        kg.graph.add_edge("dmz-historian", "plc-controller-01", relation="connected_to", classification="OBSERVED")

        # Response contains "Do not use ACTION: TERMINATE." followed by a valid HOP action
        complex_decision = (
            "Do not use ACTION: TERMINATE.\n"
            "Continue investigating the PLC.\n"
            "ACTION: HOP | plc-controller-01"
        )
        client = MockOllamaClient(responses={"Decision": complex_decision})
        agent = GraphReaderAgent(client, kg)

        next_node, should_terminate = agent.explore_node(
            current_node="dmz-historian",
            question="Where did the lateral movement go?",
            plan="Trace connection to PLC",
            step=1,
            visited_nodes={"dmz-historian"},
        )
        # Must NOT terminate, and must successfully hop to plc-controller-01
        self.assertFalse(should_terminate)
        self.assertEqual(next_node, "plc-controller-01")

    def test_purdue_case_manifest_and_ground_truth_integrity(self):
        import json
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        manifest_path = os.path.join(base_dir, "examples", "purdue_ot_incident", "case_manifest.json")
        self.assertTrue(os.path.exists(manifest_path), f"Manifest missing at {manifest_path}")

        with open(manifest_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(manifest["case_id"], "AEGIS-OT-2026-001")
        self.assertEqual(manifest["compliance_frameworks"][0]["name"], "NIST SP 800-82 Rev. 3")
        self.assertEqual(manifest["compliance_frameworks"][1]["name"], "MITRE ATT&CK for ICS v20.0")

        # Verify all evidence files exist
        purdue_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        for category, files in manifest["evidence_sources"].items():
            for rel_file in files:
                file_path = os.path.join(purdue_dir, rel_file)
                self.assertTrue(os.path.exists(file_path), f"Evidence source file missing: {file_path}")

    def test_primary_attack_path_continuity(self):
        import json
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        attack_path_file = os.path.join(base_dir, "examples", "purdue_ot_incident", "ground_truth", "expected_attack_path.json")
        with open(attack_path_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        primary_path = data["primary_attack_path"]
        self.assertGreaterEqual(len(primary_path), 4)

        # Strict mathematical continuity: target of step N must be source of step N+1
        for current, following in zip(primary_path, primary_path[1:]):
            self.assertEqual(
                current["target_node"],
                following["source_node"],
                f"Path discontinuity: {current['target_node']} does not connect to {following['source_node']}",
            )

    def test_classify_relation_forensics_strict_allowlist_and_fallbacks(self):
        # 1. Unknown source type must fail closed to UNCLASSIFIED
        tier_unk, basis_unk = classify_relation_forensics("executed", source_type="UNKNOWN_IMPORT")
        self.assertEqual(tier_unk, "UNCLASSIFIED")
        self.assertEqual(basis_unk, EVIDENCE_BASIS_UNCLASSIFIED)

        # 2. Invalid explicit evidence basis must fail closed to UNCLASSIFIED
        tier_inv, basis_inv = classify_relation_forensics("communicated_with", evidence_basis="INVALID_BASIS")
        self.assertEqual(tier_inv, "UNCLASSIFIED")
        self.assertEqual(basis_inv, EVIDENCE_BASIS_UNCLASSIFIED)

        # 3. Known trusted telemetry source yields DIRECT_TELEMETRY
        tier_tel, basis_tel = classify_relation_forensics("executed", source_type="SYSMON_EVENT")
        self.assertEqual(tier_tel, "OBSERVED")
        self.assertEqual(basis_tel, EVIDENCE_BASIS_TELEMETRY)

    def test_case_loader_deterministic_ingestion(self):
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        # Check nodes across Purdue levels
        for expected_node in ["corp-ws-019", "dmz-hist-mirror", "scada-ignition-gw01", "eng-ws-02", "plc-ballast-01"]:
            self.assertTrue(kg.graph.has_node(expected_node), f"Missing expected node: {expected_node}")

        # Check critical PLC logic modification edge
        edge_data = kg.graph.get_edge_data("eng-ws-02", "plc-ballast-01")
        self.assertIsNotNone(edge_data)
        first_edge = list(edge_data.values())[0]
        self.assertEqual(first_edge["relation"], "modified_logic")
        self.assertEqual(first_edge["classification"], "OBSERVED")
        self.assertEqual(first_edge["evidence_basis"], EVIDENCE_BASIS_TELEMETRY)

        # Check atomic facts have genuine record IDs and source locations
        for fact in kg.atomic_facts.values():
            self.assertTrue(len(fact.source_record_id) > 0)
            self.assertTrue(len(fact.source_location) > 0)
            self.assertIn(":", fact.source_location)

    def test_missing_controller_audit_fail_closed(self):
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)

        # Ingest case with controller_audit.jsonl excluded
        kg_missing = loader.load_case(exclude_sources=["controller_audit.jsonl"])

        # eng-ws-02 must NOT have an edge to plc-ballast-01
        has_plc_edge = kg_missing.graph.has_edge("eng-ws-02", "plc-ballast-01")
        self.assertFalse(has_plc_edge, "eng-ws-02 unexpectedly connected to plc-ballast-01 when audit is excluded")

        # Mock agent synthesis: must fail closed when questioned on PLC compromise
        client = MockOllamaClient(responses={
            "Target Question": "The available evidence establishes access to the historian mirror and subsequent SCADA activity. However, there is no controller audit record proving that the PLC logic was modified by the same session. The attack path is therefore incomplete."
        })
        agent = GraphReaderAgent(client, kg_missing)
        agent.notebook.add_note(step=1, source_node="dmz-hist-mirror", content="Accessed SCADA gateway", fact_ids=["dummy_F001"])

        answer = agent.synthesize_answer("Did the enterprise compromise reach the PLC?")
        # Without fact citations proving PLC modification, verification fails closed or returns incomplete
        self.assertIn("INSUFFICIENT EVIDENCE", answer)

    def test_untrusted_source_cannot_assert_direct_telemetry(self):
        # Even if explicit basis is passed as DIRECT_TELEMETRY, untrusted source must fail closed to UNCLASSIFIED
        tier, basis = classify_relation_forensics(
            "executed",
            source_type="UNKNOWN_IMPORT",
            evidence_basis=EVIDENCE_BASIS_TELEMETRY,
        )
        self.assertEqual(tier, "UNCLASSIFIED")
        self.assertEqual(basis, EVIDENCE_BASIS_UNCLASSIFIED)

        # Incident report also cannot assert DIRECT_TELEMETRY
        tier_rep, basis_rep = classify_relation_forensics(
            "executed",
            source_type="INCIDENT_REPORT",
            evidence_basis=EVIDENCE_BASIS_TELEMETRY,
        )
        self.assertEqual(tier_rep, "UNCLASSIFIED")
        self.assertEqual(basis_rep, EVIDENCE_BASIS_UNCLASSIFIED)

    def test_missing_timestamp_defaults_to_unknown(self):
        # If timestamp is missing, it must be explicit UNKNOWN, never defaulting to ingestion time
        events = StructuredLogParser.parse_line(
            data={"event_id": 1, "host": "CORP-WS-019", "process": "powershell.exe"},
            source_id="test_src",
            source_type="SYSMON_EVENT",
            file_rel="sysmon.jsonl",
            line_idx=1,
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].event_time, "UNKNOWN")

    def test_rich_forensic_attributes_and_raw_vault_preserved(self):
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        # 1. Sysmon user attribute must be preserved in AtomicFact
        sysmon_facts = [f for f in kg.atomic_facts.values() if f.source_type == "SYSMON_EVENT" and f.relation == "executed"]
        self.assertTrue(len(sysmon_facts) > 0)
        first_sysmon = sysmon_facts[0]
        self.assertEqual(first_sysmon.attributes.get("user"), "corp\\j.miller")
        self.assertIn("user: corp\\j.miller", first_sysmon.text)

        # 2. Controller audit CIP hash signatures must be preserved
        cip_facts = [f for f in kg.atomic_facts.values() if f.source_type == "CONTROLLER_AUDIT"]
        self.assertTrue(len(cip_facts) > 0)
        first_cip = cip_facts[0]
        self.assertEqual(first_cip.attributes.get("prior_signature"), "0x4A1F")
        self.assertEqual(first_cip.attributes.get("new_signature"), "0x9E7B")
        self.assertIn("prior_signature: 0x4A1F", first_cip.text)
        self.assertIn("new_signature: 0x9E7B", first_cip.text)

        # 3. Raw record vault must store the original dictionary
        vault_key = (first_cip.source_id, first_cip.source_record_id)
        self.assertIn(vault_key, kg.raw_record_vault)
        raw_cip = kg.raw_record_vault[vault_key]
        self.assertEqual(raw_cip.get("model"), "Rockwell ControlLogix 1756-L83E")

    def test_ingestion_completeness_report_metrics(self):
        import os

        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)

        # Complete ingestion
        kg_full = loader.load_case()
        self.assertIsNotNone(kg_full.ingestion_report)
        self.assertTrue(kg_full.ingestion_report.is_complete)
        self.assertEqual(len(kg_full.ingestion_report.missing_sources), 0)
        self.assertGreater(kg_full.ingestion_report.facts_produced, 0)

        # Incomplete ingestion (missing controller audit)
        kg_partial = loader.load_case(exclude_sources=["controller_audit.jsonl"])
        self.assertFalse(kg_partial.ingestion_report.is_complete)
        self.assertIn("level1_plc/controller_audit.jsonl", kg_partial.ingestion_report.missing_sources)

    def test_citation_verifier_with_caseloader_record_ids(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        # Retrieve an actual real CaseLoader fact ID
        real_fact_id = list(kg.atomic_facts.keys())[0]
        self.assertIn("_R", real_fact_id)

        # 1. Plain text citing valid CaseLoader fact ID -> Success
        valid_response = f"The incident analysis confirms the activity. Citation: {real_fact_id}."
        client = MockOllamaClient(responses={"Question 1": valid_response})
        agent = GraphReaderAgent(client, kg)
        agent.notebook.add_note(step=1, source_node="test_node", content="Observed activity", fact_ids=[real_fact_id])
        ans = agent.synthesize_answer("Question 1")
        self.assertNotIn("INSUFFICIENT EVIDENCE", ans)
        self.assertIn(real_fact_id, ans)

        # 2. Structured JSON citing valid CaseLoader fact ID -> Success
        json_response = f'{{"answer": "Verified telemetry activity", "cited_fact_ids": ["{real_fact_id}"], "status": "SUPPORTED"}}'
        client_json = MockOllamaClient(responses={"Question 2": json_response})
        agent_json = GraphReaderAgent(client_json, kg)
        agent_json.notebook.add_note(step=1, source_node="test_node", content="Observed activity", fact_ids=[real_fact_id])
        ans_json = agent_json.synthesize_answer("Question 2")
        self.assertNotIn("INSUFFICIENT EVIDENCE", ans_json)

        # 3. Fabricated ID alongside valid ID -> Fail closed
        fabricated_id = "99999999999999999999999999999999_R1_fabricated_record"
        fraud_response = f"Activity confirmed: {real_fact_id} and {fabricated_id}."
        client_fraud = MockOllamaClient(responses={"Question 3": fraud_response})
        agent_fraud = GraphReaderAgent(client_fraud, kg)
        agent_fraud.notebook.add_note(step=1, source_node="test_node", content="Observed activity", fact_ids=[real_fact_id])
        ans_fraud = agent_fraud.synthesize_answer("Question 3")
        self.assertIn("INSUFFICIENT EVIDENCE", ans_fraud)
        self.assertIn("unverified fact IDs", ans_fraud)

    def test_historian_telemetry_source_trust_policy(self):
        # Both HISTORIAN_MIRROR and HISTORIAN_AUDIT must be trusted and classified as OBSERVED
        tier_mirror, basis_mirror = classify_relation_forensics(
            "connected_to",
            source_type="HISTORIAN_MIRROR",
            evidence_basis=EVIDENCE_BASIS_TELEMETRY,
        )
        self.assertEqual(tier_mirror, "OBSERVED")
        self.assertEqual(basis_mirror, EVIDENCE_BASIS_TELEMETRY)

        tier_audit, basis_audit = classify_relation_forensics(
            "connected_to",
            source_type="HISTORIAN_AUDIT",
            evidence_basis=EVIDENCE_BASIS_TELEMETRY,
        )
        self.assertEqual(tier_audit, "OBSERVED")
        self.assertEqual(basis_audit, EVIDENCE_BASIS_TELEMETRY)

    def test_source_type_mapping_trust_policy_coverage(self):
        # Every source mapped by CaseLoader must have a recognized trust policy
        for filename, source_type in CaseLoader.SOURCE_TYPE_MAPPING.items():
            self.assertIn(
                source_type,
                TRUSTED_TELEMETRY_SOURCES,
                f"Source type '{source_type}' from file '{filename}' lacks defined policy in TRUSTED_TELEMETRY_SOURCES"
            )

    def test_process_telemetry_typed_observation_nodes(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        # 1. Pressure transmitter must link to an observation node, not directly to raw measurement string
        self.assertTrue(kg.graph.has_node("pt-4012"))
        obs_nodes = [
            tgt for tgt, edge_data in kg.graph["pt-4012"].items()
            if any(attr.get("relation") == "has_observation" for attr in edge_data.values())
        ]
        self.assertTrue(len(obs_nodes) > 0, "pt-4012 must have an observation node via has_observation")
        pressure_obs = obs_nodes[0]

        # Observation node must link to measured value
        val_nodes = [
            tgt for tgt, edge_data in kg.graph[pressure_obs].items()
            if any(attr.get("relation") == "measured_value" for attr in edge_data.values())
        ]
        self.assertIn("12.4_bar", val_nodes)

        # 2. Valve XV-4012 must link to observation node with commanded, actual, and interlock states
        self.assertTrue(kg.graph.has_node("xv-4012"))
        valve_obs_nodes = [
            tgt for tgt, edge_data in kg.graph["xv-4012"].items()
            if any(attr.get("relation") == "has_observation" for attr in edge_data.values())
        ]
        self.assertTrue(len(valve_obs_nodes) > 0, "xv-4012 must have an observation node via has_observation")
        valve_obs = valve_obs_nodes[0]

        # Check commanded_state, actual_state, interlock_status predicates on observation node
        relations_from_obs = {
            attr.get("relation"): tgt
            for tgt, edge_dict in kg.graph[valve_obs].items()
            for attr in edge_dict.values()
        }
        self.assertIn("commanded_state", relations_from_obs)
        self.assertEqual(relations_from_obs["commanded_state"], "open_inhibited")
        self.assertIn("actual_state", relations_from_obs)
        self.assertEqual(relations_from_obs["actual_state"], "closed")
        self.assertIn("interlock_status", relations_from_obs)
        self.assertEqual(relations_from_obs["interlock_status"], "suppressed_by_plc_logic")

        # Must NOT have bogus 'connected_to 12.4_bar' edge
        if kg.graph.has_edge("pt-4012", "12.4_bar"):
            for edge_attr in kg.graph.get_edge_data("pt-4012", "12.4_bar").values():
                self.assertNotEqual(edge_attr.get("relation"), "connected_to")

    def test_empty_source_file_fails_completeness(self):
        import tempfile, shutil
        temp_dir = tempfile.mkdtemp()
        try:
            # Create manifest with an empty file
            empty_file = os.path.join(temp_dir, "empty_log.jsonl")
            with open(empty_file, "w", encoding="utf-8") as f:
                pass  # 0 bytes
            manifest_content = {
                "case_id": "TEST-EMPTY-001",
                "title": "Empty File Test",
                "evidence_sources": {
                    "test_zone": ["empty_log.jsonl"]
                }
            }
            manifest_path = os.path.join(temp_dir, "case_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_content, f)

            loader = CaseLoader(temp_dir)
            kg = loader.load_case()
            self.assertFalse(kg.ingestion_report.is_complete)
            self.assertIn("empty_log.jsonl", kg.ingestion_report.empty_sources)
            self.assertIsNotNone(kg.ingestion_report.degraded_reason)
            self.assertIn("Empty sources", kg.ingestion_report.degraded_reason)
        finally:
            shutil.rmtree(temp_dir)

    def test_separated_ingestion_report_metrics(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        rep = kg.ingestion_report
        self.assertIsNotNone(rep)
        self.assertGreater(rep.source_records_read, 0)
        self.assertGreater(rep.source_records_parsed, 0)
        self.assertEqual(rep.source_records_rejected, 0)
        self.assertGreater(rep.normalized_events_generated, 0)
        self.assertGreater(rep.atomic_facts_generated, 0)
        self.assertGreater(rep.graph_edges_generated, 0)
        # Property aliases for backward compatibility
        self.assertEqual(rep.facts_produced, rep.atomic_facts_generated)
        self.assertEqual(rep.edges_produced, rep.graph_edges_generated)
        self.assertEqual(rep.records_read, rep.source_records_read)
        self.assertEqual(rep.records_parsed, rep.source_records_parsed)

    def test_complete_vs_degraded_investigation_benchmark(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)

        # 1. Complete Case: All evidence present -> Agent verifies PLC modification with citations
        kg_complete = loader.load_case()
        self.assertTrue(kg_complete.ingestion_report.is_complete)
        cip_facts = [f for f in kg_complete.atomic_facts.values() if f.relation == "modified_logic" and f.object_entity == "plc-ballast-01"]
        self.assertTrue(len(cip_facts) > 0)
        fact_cip = cip_facts[0]

        valid_response = f"The compromise reached the PLC. Engineering workstation ENG-WS-02 downloaded modified logic to PLC-BALLAST-01 [{fact_cip.fact_id}]."
        client_complete = MockOllamaClient(responses={"Did the enterprise compromise reach the PLC?": valid_response})
        agent_complete = GraphReaderAgent(client_complete, kg_complete)
        agent_complete.notebook.add_note(step=1, source_node="eng-ws-02", content="Modified PLC logic", fact_ids=[fact_cip.fact_id])
        ans_complete = agent_complete.synthesize_answer("Did the enterprise compromise reach the PLC?")
        self.assertNotIn("INSUFFICIENT EVIDENCE", ans_complete)
        self.assertIn(fact_cip.fact_id, ans_complete)

        # 2. Degraded Case: controller_audit.jsonl removed -> Agent enforces degraded policy and fails closed
        kg_degraded = loader.load_case(exclude_sources=["controller_audit.jsonl"])
        self.assertFalse(kg_degraded.ingestion_report.is_complete)
        sysmon_facts = [f for f in kg_degraded.atomic_facts.values() if f.source_type == "SYSMON_EVENT"]
        dummy_id = sysmon_facts[0].fact_id

        fraud_attempt = f"The compromise reached the PLC via PowerShell execution on the workstation [{dummy_id}]."
        client_degraded = MockOllamaClient(responses={"Did the enterprise compromise reach the PLC?": fraud_attempt})
        agent_degraded = GraphReaderAgent(client_degraded, kg_degraded)
        agent_degraded.notebook.add_note(step=1, source_node="corp-ws-019", content="Executed powershell", fact_ids=[dummy_id])
        ans_degraded = agent_degraded.synthesize_answer("Did the enterprise compromise reach the PLC?")
        self.assertIn("INSUFFICIENT EVIDENCE", ans_degraded)
        self.assertIn("CONTROLLER_AUDIT", ans_degraded)

    def test_observation_nodes_do_not_collide_across_repeated_measurements(self):
        # 4 readings from the same sensor PT-4012 across time
        readings = [
            {"timestamp": "2026-10-28T14:00:00Z", "tag_id": "PT-4012", "value_bar": 6.2, "purdue_level": 0},
            {"timestamp": "2026-10-28T14:05:00Z", "tag_id": "PT-4012", "value_bar": 8.1, "purdue_level": 0},
            {"timestamp": "2026-10-28T14:10:00Z", "tag_id": "PT-4012", "value_bar": 12.4, "purdue_level": 0},
            {"timestamp": "2026-10-28T14:15:00Z", "tag_id": "PT-4012", "value_bar": 13.0, "purdue_level": 0},
        ]
        all_obs_nodes = set()
        for idx, r in enumerate(readings, 1):
            events = StructuredLogParser.parse_line(
                r,
                source_id="src_telemetry_001",
                source_type="PROCESS_TELEMETRY",
                file_rel="process_events.jsonl",
                line_idx=idx,
            )
            # Find the observation node
            obs_events = [e for e in events if e.predicate == "has_observation"]
            self.assertEqual(len(obs_events), 1)
            obs_node = obs_events[0].object_entity
            self.assertNotIn(obs_node, all_obs_nodes, f"Collision detected for observation node {obs_node} at line {idx}")
            all_obs_nodes.add(obs_node)

        # Must have exactly 4 unique observation nodes for 4 distinct readings
        self.assertEqual(len(all_obs_nodes), 4)

    def test_claim_level_predicate_validation_rejects_unsupported_claims(self):
        import os
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        case_dir = os.path.join(base_dir, "examples", "purdue_ot_incident")
        loader = CaseLoader(case_dir)
        kg = loader.load_case()

        # Get a real Sysmon fact (predicate: 'executed')
        sysmon_facts = [f for f in kg.atomic_facts.values() if f.relation == "executed"]
        sysmon_fid = sysmon_facts[0].fact_id

        # Model claims PLC modification but cites only Sysmon execution fact
        structured_payload = json.dumps({
            "claims": [
                {
                    "statement": "The workstation downloaded modified PLC logic to the controller.",
                    "fact_ids": [sysmon_fid],
                    "status": "SUPPORTED"
                }
            ]
        })
        client = MockOllamaClient(responses={"Target Question": structured_payload})
        agent = GraphReaderAgent(client, kg)
        agent.notebook.add_note(step=1, source_node="corp-ws-019", content="Executed powershell", fact_ids=[sysmon_fid])

        # Claim-level predicate check must fail closed because fact predicate is 'executed', not 'modified_logic'
        ans = agent.synthesize_answer("Target Question")
        self.assertIn("INSUFFICIENT EVIDENCE", ans)
        self.assertIn("modified_logic", ans)

    def test_historian_operation_specific_predicates(self):
        # 1. Historian Mirror uses queried_tag_store
        mirror_events = StructuredLogParser.parse_line(
            {"host": "dmz-hist-mirror", "query": "SELECT * FROM tags"},
            source_id="src_mirror",
            source_type="HISTORIAN_MIRROR",
            file_rel="historian_mirror.jsonl",
            line_idx=1,
        )
        self.assertEqual(len(mirror_events), 1)
        self.assertEqual(mirror_events[0].predicate, "queried_tag_store")
        tier_m, _ = classify_relation_forensics(mirror_events[0].predicate, "HISTORIAN_MIRROR", EVIDENCE_BASIS_TELEMETRY)
        self.assertEqual(tier_m, "OBSERVED")

        # 2. Historian Audit uses replicated_tags_to
        audit_events = StructuredLogParser.parse_line(
            {"peer_node": "dmz-hist-mirror", "host": "l3-hist-srv01", "transferred_tags": ["T1"]},
            source_id="src_audit",
            source_type="HISTORIAN_AUDIT",
            file_rel="historian_audit.jsonl",
            line_idx=1,
        )
        self.assertEqual(len(audit_events), 1)
        self.assertEqual(audit_events[0].predicate, "replicated_tags_to")
        tier_a, _ = classify_relation_forensics(audit_events[0].predicate, "HISTORIAN_AUDIT", EVIDENCE_BASIS_TELEMETRY)
        self.assertEqual(tier_a, "OBSERVED")

    def test_raw_record_vault_line_preservation_and_non_dict_rejection(self):
        import tempfile, shutil
        temp_dir = tempfile.mkdtemp()
        try:
            # Create a JSONL with non-dict values (array, int, string) alongside a valid dict
            test_file = os.path.join(temp_dir, "test_malformed.jsonl")
            with open(test_file, "w", encoding="utf-8") as f:
                f.write('{"timestamp": "2026-10-28T15:00:00Z", "host": "CORP-WS-019", "event_id": 1, "process": "cmd.exe"}\n')
                f.write('[1, 2, 3]\n')
                f.write('"simple string"\n')
                f.write('12345\n')

            manifest_content = {
                "case_id": "TEST-MALFORMED-001",
                "title": "Malformed Record Test",
                "evidence_sources": {
                    "enterprise": ["test_malformed.jsonl"]
                }
            }
            manifest_path = os.path.join(temp_dir, "case_manifest.json")
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(manifest_content, f)

            # Map the test file as SYSMON_EVENT
            CaseLoader.SOURCE_TYPE_MAPPING["test_malformed.jsonl"] = "SYSMON_EVENT"

            loader = CaseLoader(temp_dir)
            kg = loader.load_case()
            self.assertEqual(kg.ingestion_report.source_records_read, 4)
            self.assertEqual(kg.ingestion_report.source_records_parsed, 1)
            self.assertEqual(kg.ingestion_report.source_records_rejected, 3)

            # Check that raw line is preserved in vault
            vault_keys = list(kg.raw_record_vault.keys())
            self.assertEqual(len(vault_keys), 1)
            stored_record = kg.raw_record_vault[vault_keys[0]]
            self.assertIn("_raw_line", stored_record)
            self.assertIn("cmd.exe", stored_record["_raw_line"])
        finally:
            shutil.rmtree(temp_dir)

    def test_timestamp_validation_and_utc_normalization(self):
        # 1. ISO-8601 with offset
        events_offset = StructuredLogParser.parse_line(
            {"timestamp": "2026-10-28T17:00:00+02:00", "event_id": 1, "host": "H1", "process": "p.exe"},
            source_id="src1", source_type="SYSMON_EVENT", file_rel="sysmon.jsonl", line_idx=1
        )
        self.assertEqual(events_offset[0].event_time, "2026-10-28T15:00:00+00:00")

        # 2. Missing timestamp -> UNKNOWN
        events_none = StructuredLogParser.parse_line(
            {"event_id": 1, "host": "H1", "process": "p.exe"},
            source_id="src1", source_type="SYSMON_EVENT", file_rel="sysmon.jsonl", line_idx=2
        )
        self.assertEqual(events_none[0].event_time, "UNKNOWN")

        # 3. Invalid timestamp -> INVALID_TIMESTAMP
        events_inv = StructuredLogParser.parse_line(
            {"timestamp": "not-a-date", "event_id": 1, "host": "H1", "process": "p.exe"},
            source_id="src1", source_type="SYSMON_EVENT", file_rel="sysmon.jsonl", line_idx=3
        )
        self.assertEqual(events_inv[0].event_time, "INVALID_TIMESTAMP")


if __name__ == "__main__":
    unittest.main()
