"""
Notebook Scratchpad (Inspired by GraphReader - arXiv:2406.14550)
Maintains bounded working memory, deduplication, and token budget enforcement
to strictly guarantee local execution within a 4K context window.
"""

from dataclasses import dataclass
from typing import List


@dataclass
class Note:
    step: int
    source_node: str
    content: str


class Notebook:
    """
    Bounded working memory scratchpad for the exploring agent.
    Enforces a strict token budget and prevents memory bloat.
    """

    def __init__(self, max_tokens: int = 1500):
        self.notes: List[Note] = []
        self.max_tokens = max_tokens

    def estimate_tokens(self) -> int:
        """Heuristic token estimator (avg 4 chars per token)."""
        total_text = " ".join(f"{n.source_node} {n.content}" for n in self.notes)
        return max(0, int(len(total_text) / 4))

    def add_note(self, step: int, source_node: str, content: str):
        """Adds a note with deduplication and token budget enforcement."""
        content = content.strip()
        if not content:
            return

        # Deduplication check
        for n in self.notes:
            if content.lower() == n.content.lower():
                return

        self.notes.append(Note(step=step, source_node=source_node, content=content))
        self._enforce_budget()

    def _enforce_budget(self):
        """Evicts oldest notes if token budget is exceeded to maintain 4K prompt limits."""
        while self.estimate_tokens() > self.max_tokens and len(self.notes) > 1:
            self.notes.pop(0)

    def format_for_prompt(self) -> str:
        """Formats the notebook memory for LLM context inclusion."""
        if not self.notes:
            return "No notes recorded yet."
        formatted = []
        for n in self.notes:
            formatted.append(f"[Step {n.step} | Node: {n.source_node}]\n{n.content}")
        return "\n\n".join(formatted)

    def is_empty(self) -> bool:
        return len(self.notes) == 0

    def clear(self):
        self.notes.clear()
