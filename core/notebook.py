"""
Notebook Scratchpad (Inspired by GraphReader - arXiv:2406.14550)
Maintains working memory and recorded evidence during graph exploration.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class Note:
    step: int
    source_node: str
    content: str


class Notebook:
    """Working memory scratchpad for the exploring agent."""

    def __init__(self):
        self.notes: List[Note] = []

    def add_note(self, step: int, source_node: str, content: str):
        self.notes.append(Note(step=step, source_node=source_node, content=content.strip()))

    def format_for_prompt(self) -> str:
        if not self.notes:
            return "No notes recorded yet."
        formatted = []
        for n in self.notes:
            formatted.append(f"[Step {n.step} | Node: {n.source_node}] {n.content}")
        return "\n".join(formatted)

    def is_empty(self) -> bool:
        return len(self.notes) == 0
