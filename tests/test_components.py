"""
Comprehensive unit tests for GraphMemory-Agent core components.
Ensures MultiDiGraph directed construction, strict allowlist edge tiers,
fail-closed evidence auditing, multi-source provenance merging, and budget assembly function offline.
"""

import unittest
from core.notebook import Notebook, EvidenceNote, SourceReference
from core.graph_builder import (
    AtomicFact, KnowledgeGraph, normalize_entity, classify_relation,
    classify_relation_forensics, ForensicRelation, GraphBuilder,
    EVIDENCE_BASIS_TELEMETRY, EVIDENCE_BASIS_REPORTED, EVIDENCE_BASIS_INTEL,
    EVIDENCE_BASIS_UNCLASSIFIED,
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


if __name__ == "__main__":
    unittest.main()
