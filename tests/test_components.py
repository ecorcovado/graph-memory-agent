"""
Comprehensive unit tests for GraphMemory-Agent core components.
Ensures MultiDiGraph directed construction, strict forensic edge tiers,
centralized prompt assembly, immutable EvidenceNotes, and deterministic verification function offline.
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

    def test_prompt_budget_manager_assembly(self):
        # Test combined prompt assembly within budget
        sys_p, user_p, tokens = PromptBudgetManager.assemble_prompt(
            system_text="System instructions.",
            query_and_plan="Question: What happened?",
            notebook_text="Note: Evidence found.",
            context_text="Context: Log lines.",
            instructions="Output answer.",
        )
        self.assertLessEqual(tokens, PromptBudgetManager.MAX_PROMPT_BUDGET)
        self.assertIn("Evidence found", user_p)

        # Test prompt clamping on massive inputs
        huge_text = "a" * 20000
        sys_p2, user_p2, tokens2 = PromptBudgetManager.assemble_prompt(
            system_text="System.",
            query_and_plan=huge_text,
            context_text=huge_text,
        )
        self.assertLessEqual(tokens2, PromptBudgetManager.MAX_PROMPT_BUDGET)

    def test_entity_normalization(self):
        self.assertEqual(normalize_entity("  WORKSTATION-402  "), "workstation-402")
        self.assertEqual(normalize_entity('"APT-SilverFox"'), "apt-silverfox")
        self.assertEqual(normalize_entity("System Health   Check"), "system health check")

    def test_strict_edge_classification_allowlist(self):
        # Known observed events
        self.assertEqual(classify_relation("communicated_with"), "OBSERVED")
        self.assertEqual(classify_relation("executed"), "OBSERVED")
        self.assertEqual(classify_relation("modified_registry"), "OBSERVED")

        # Threat intel
        self.assertEqual(classify_relation("attributed_to"), "INTELLIGENCE_REPORTED")
        self.assertEqual(classify_relation("known_technique"), "INTELLIGENCE_REPORTED")

        # Co-occurrence
        self.assertEqual(classify_relation("co_occurs_with"), "CO_OCCURRENCE")

        # Unrecognized / unknown predicate must NEVER default to OBSERVED
        self.assertEqual(classify_relation("unknown_custom_predicate_xyz"), "UNCLASSIFIED")
        self.assertEqual(classify_relation("arbitrary_verb"), "UNCLASSIFIED")

    def test_notebook_token_budget_and_eviction(self):
        nb = Notebook(max_tokens=50, max_single_note_tokens=40)
        self.assertTrue(nb.is_empty())

        # Add 2 notes that exceed 50 tokens total
        nb.add_note(step=1, source_node="node1", content="Note 1: " + "a" * 120, fact_ids=["C0_F1"])
        nb.add_note(step=2, source_node="node2", content="Note 2: " + "b" * 120, fact_ids=["C0_F2"])

        # Eviction keeps memory bounded
        self.assertLessEqual(nb.estimate_tokens(), 50)
        formatted = nb.format_for_prompt()
        self.assertNotIn("Note 1", formatted)
        self.assertIn("Note 2", formatted)
        self.assertTrue(nb.verify_citation("C0_F2"))

    def test_notebook_immutable_notes_and_deduplication(self):
        nb = Notebook(max_tokens=500)
        nb.add_note(step=1, source_node="nodeA", content="Identical evidence found.", fact_ids=["C0_F1"])
        nb.add_note(step=2, source_node="nodeB", content="Identical evidence found.", fact_ids=["C0_F2"])

        self.assertEqual(len(nb.notes), 1)
        self.assertIsInstance(nb.notes[0].fact_ids, tuple)

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

        # Verify directional relation helper
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
        kg.graph.add_node("some_secondary_concept")

        # Add an OBSERVED edge to IP, and a CO_OCCURRENCE edge to secondary concept
        kg.graph.add_edge("workstation-402", "198.51.100.44", relation="communicated_with", classification="OBSERVED")
        kg.graph.add_edge("workstation-402", "some_secondary_concept", relation="co_occurs_with", classification="CO_OCCURRENCE")

        client = MockOllamaClient()
        agent = GraphReaderAgent(client, kg)

        candidates = ["some_secondary_concept", "198.51.100.44"]
        ranked = agent.rank_candidate_neighbors("workstation-402", candidates, "What IP communicated?")
        
        # OBSERVED edge to 198.51.100.44 must rank first over CO_OCCURRENCE
        self.assertEqual(ranked[0], "198.51.100.44")

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

    def test_evidence_verification_empty_notebook(self):
        kg = KnowledgeGraph()
        client = MockOllamaClient()
        agent = GraphReaderAgent(client, kg)

        answer = agent.synthesize_answer("What is the persistence task?")
        self.assertIn("INSUFFICIENT EVIDENCE", answer)

    def test_deterministic_evidence_verification_flagging(self):
        kg = KnowledgeGraph()
        # Mock client returns an answer citing a fake/unverified fact ID C99_F99
        client = MockOllamaClient(responses={
            "Verified Immutable Fact IDs": "The attacker used task SystemHealthCheck supported by fact C99_F99."
        })
        agent = GraphReaderAgent(client, kg)
        agent.notebook.add_note(step=1, source_node="host", content="Valid note", fact_ids=["C0_F1"])

        answer = agent.synthesize_answer("What task was used?")
        # Audit notice must catch the unverified citation
        self.assertIn("FORENSIC AUDIT: Warning - Answer cited unverified fact IDs", answer)
        self.assertIn("C99_F99", answer)


if __name__ == "__main__":
    unittest.main()
