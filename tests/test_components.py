"""
Comprehensive unit tests for GraphMemory-Agent core components.
Ensures MultiDiGraph directed construction, strict allowlist edge tiers,
fail-closed evidence auditing, multi-source provenance merging, and budget assembly function offline.
"""

import unittest
from core.notebook import Notebook, EvidenceNote
from core.graph_builder import AtomicFact, KnowledgeGraph, normalize_entity, classify_relation, GraphBuilder
from core.budget_manager import PromptBudgetManager
from core.agent import GraphReaderAgent


class MockOllamaClient:
    """Mock LLM client for offline deterministic unit testing."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_history = []

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


if __name__ == "__main__":
    unittest.main()
