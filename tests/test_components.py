"""
Comprehensive unit tests for GraphMemory-Agent core components.
Ensures MultiDiGraph directed construction, forensic edge classifications,
centralized prompt budgeting, source-linked EvidenceNotes, and deterministic heuristics function offline.
"""

import unittest
from core.notebook import Notebook, EvidenceNote
from core.graph_builder import AtomicFact, KnowledgeGraph, normalize_entity, classify_relation
from core.budget_manager import PromptBudgetManager
from core.agent import GraphReaderAgent


class MockOllamaClient:
    """Mock LLM client for offline deterministic unit testing."""
    def __init__(self, responses=None):
        self.responses = responses or {}
        self.call_history = []

    def generate(self, prompt: str, temperature: float = 0.1, system: str = "") -> str:
        self.call_history.append(prompt)
        for key, val in self.responses.items():
            if key in prompt:
                return val
        return "mocked_response"


class TestGraphMemoryAgent(unittest.TestCase):

    def test_prompt_budget_manager(self):
        # Test token estimation
        self.assertEqual(PromptBudgetManager.estimate_tokens(""), 0)
        self.assertEqual(PromptBudgetManager.estimate_tokens("abcd"), 1)
        self.assertEqual(PromptBudgetManager.estimate_tokens("a" * 400), 100)

        # Test text clamping
        clamped = PromptBudgetManager.clamp_text("a" * 1000, max_tokens=50)
        self.assertLessEqual(PromptBudgetManager.estimate_tokens(clamped), 60)
        self.assertIn("TRUNCATED", clamped)

        # Test node list clamping
        huge_nodes = [f"node_{i}" for i in range(500)]
        clamped_nodes = PromptBudgetManager.clamp_node_list(huge_nodes, max_tokens=50)
        self.assertLess(len(clamped_nodes), 500)
        self.assertLessEqual(PromptBudgetManager.estimate_tokens(" ".join(clamped_nodes)), 60)

    def test_entity_normalization(self):
        self.assertEqual(normalize_entity("  WORKSTATION-402  "), "workstation-402")
        self.assertEqual(normalize_entity('"APT-SilverFox"'), "apt-silverfox")
        self.assertEqual(normalize_entity("System Health   Check"), "system health check")

    def test_edge_classification(self):
        self.assertEqual(classify_relation("communicated_with"), "OBSERVED")
        self.assertEqual(classify_relation("executed"), "OBSERVED")
        self.assertEqual(classify_relation("attributed_to"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("known_technique"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("co_occurs_with"), "CO_OCCURRENCE")

    def test_notebook_token_budget_and_eviction(self):
        # Budget = 50 tokens
        nb = Notebook(max_tokens=50, max_single_note_tokens=40)
        self.assertTrue(nb.is_empty())

        # Add 2 notes that exceed 50 tokens total (each ~35 tokens)
        nb.add_note(step=1, source_node="node1", content="Note 1: " + "a" * 120, fact_ids=["C0_F1"])
        nb.add_note(step=2, source_node="node2", content="Note 2: " + "b" * 120, fact_ids=["C0_F2"])

        # Budget should evict Note 1 to stay bounded
        self.assertLessEqual(nb.estimate_tokens(), 50)
        formatted = nb.format_for_prompt()
        self.assertNotIn("Note 1", formatted)
        self.assertIn("Note 2", formatted)
        self.assertIn("C0_F2", nb.get_all_fact_ids())

    def test_notebook_deduplication(self):
        nb = Notebook(max_tokens=500)
        nb.add_note(step=1, source_node="nodeA", content="Identical evidence found.")
        nb.add_note(step=2, source_node="nodeB", content="Identical evidence found.")
        self.assertEqual(len(nb.notes), 1)

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

        # Verify directional relations with classification
        rel_data = kg.get_neighbors_with_relations("workstation-402")
        self.assertEqual(len(rel_data), 1)
        direction, rel, cls, target = rel_data[0]
        self.assertEqual(direction, "outgoing")
        self.assertEqual(rel, "communicated_with")
        self.assertEqual(cls, "OBSERVED")
        self.assertEqual(target, "198.51.100.44")

    def test_initial_node_selection_fallbacks(self):
        kg = KnowledgeGraph()
        kg.graph.add_node("workstation-402")
        kg.graph.add_node("powershell.exe")
        kg.graph.add_node("apt-silverfox")
        kg.graph.add_edge("apt-silverfox", "workstation-402")
        kg.graph.add_edge("apt-silverfox", "powershell.exe")

        client = MockOllamaClient(responses={})
        agent = GraphReaderAgent(client, kg)

        # Keyword overlap fallback should select "workstation-402"
        selected = agent.select_initial_node(
            question="What happened on workstation-402?",
            plan="Investigate workstation-402"
        )
        self.assertEqual(selected, "workstation-402")

        # Hub degree fallback should select "apt-silverfox"
        hub_selected = agent.select_initial_node(
            question="Unknown anomaly detected?",
            plan="Search general systems"
        )
        self.assertEqual(hub_selected, "apt-silverfox")

    def test_evidence_verification_on_empty_notebook(self):
        kg = KnowledgeGraph()
        client = MockOllamaClient()
        agent = GraphReaderAgent(client, kg)

        answer = agent.synthesize_answer("What is the persistence task?")
        self.assertIn("INSUFFICIENT EVIDENCE", answer)


if __name__ == "__main__":
    unittest.main()
