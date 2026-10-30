"""
Unit tests for GraphMemory-Agent core components.
Ensures graph construction, notebook indexing, and deterministic structures function offline.
"""

import unittest
from core.notebook import Notebook, Note
from core.graph_builder import AtomicFact, KnowledgeGraph


class TestGraphMemoryAgent(unittest.TestCase):

    def test_notebook_addition_and_formatting(self):
        nb = Notebook()
        self.assertTrue(nb.is_empty())
        
        nb.add_note(step=1, source_node="project aegis", content="Marcus Thorne is the lead engineer.")
        self.assertFalse(nb.is_empty())
        self.assertEqual(len(nb.notes), 1)

        formatted = nb.format_for_prompt()
        self.assertIn("Marcus Thorne is the lead engineer", formatted)
        self.assertIn("project aegis", formatted)

    def test_knowledge_graph_structure(self):
        kg = KnowledgeGraph()
        fact = AtomicFact(
            fact_id="C0_F1",
            chunk_id=0,
            text="Marcus Thorne previously worked at Sentinel Core Labs.",
            key_elements=["marcus thorne", "sentinel core labs"],
        )
        kg.atomic_facts[fact.fact_id] = fact
        kg.graph.add_edge("marcus thorne", "sentinel core labs", weight=1, fact_ids=[fact.fact_id])

        self.assertTrue(kg.graph.has_node("marcus thorne"))
        self.assertTrue(kg.graph.has_node("sentinel core labs"))
        self.assertTrue(kg.graph.has_edge("marcus thorne", "sentinel core labs"))
        self.assertEqual(list(kg.graph.neighbors("marcus thorne")), ["sentinel core labs"])


if __name__ == "__main__":
    unittest.main()
