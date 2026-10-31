"""
Notebook Scratchpad (Inspired by GraphReader - arXiv:2406.14550)
Maintains bounded, source-linked evidence notes with immutable fact IDs,
per-note length clamping, and strict token budget enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Set
from core.budget_manager import PromptBudgetManager


@dataclass
class EvidenceNote:
    step: int
    source_node: str
    fact_ids: List[str] = field(default_factory=list)
    source_excerpt: str = ""
    content: str = ""
    confidence: float = 1.0


class Notebook:
    """
    Bounded, provenance-backed working memory scratchpad for DFIR investigations.
    Guarantees that memory remains strictly bounded within prompt allocations.
    """

    def __init__(self, max_tokens: int = 1000, max_single_note_tokens: int = 300):
        self.notes: List[EvidenceNote] = []
        self.max_tokens = max_tokens
        self.max_single_note_tokens = max_single_note_tokens

    def estimate_tokens(self) -> int:
        """Estimates total tokens currently consumed across all notes."""
        total_text = " ".join(
            f"{n.source_node} {' '.join(n.fact_ids)} {n.source_excerpt} {n.content}"
            for n in self.notes
        )
        return PromptBudgetManager.estimate_tokens(total_text)

    def add_note(
        self,
        step: int,
        source_node: str,
        content: str,
        fact_ids: List[str] = None,
        source_excerpt: str = "",
        confidence: float = 1.0,
    ):
        """Adds a source-linked evidence note with length clamping and budget enforcement."""
        content = content.strip()
        if not content:
            return

        # Clamp individual note length to prevent a single oversized LLM output from overflowing
        clamped_content = PromptBudgetManager.clamp_text(content, self.max_single_note_tokens)
        clamped_excerpt = PromptBudgetManager.clamp_text(source_excerpt, 150)

        # Deduplication check
        for n in self.notes:
            if clamped_content.lower() == n.content.lower():
                return

        note = EvidenceNote(
            step=step,
            source_node=source_node,
            fact_ids=fact_ids or [],
            source_excerpt=clamped_excerpt,
            content=clamped_content,
            confidence=confidence,
        )
        self.notes.append(note)
        self._enforce_budget()

    def _enforce_budget(self):
        """Evicts oldest notes if total token budget is exceeded."""
        while self.estimate_tokens() > self.max_tokens and len(self.notes) > 1:
            self.notes.pop(0)

    def get_all_fact_ids(self) -> Set[str]:
        """Returns set of all unique atomic fact IDs cited in notebook evidence."""
        facts = set()
        for n in self.notes:
            facts.update(n.fact_ids)
        return facts

    def format_for_prompt(self) -> str:
        """Formats the notebook memory for LLM context inclusion."""
        if not self.notes:
            return "No verified notes recorded yet."
        formatted = []
        for n in self.notes:
            facts_str = f" [Facts: {', '.join(n.fact_ids)}]" if n.fact_ids else ""
            excerpt_str = f"\n  Source: \"{n.source_excerpt}\"" if n.source_excerpt else ""
            formatted.append(
                f"[Step {n.step} | Node: {n.source_node}{facts_str}]{excerpt_str}\n  Evidence: {n.content}"
            )
        return "\n\n".join(formatted)

    def is_empty(self) -> bool:
        return len(self.notes) == 0

    def clear(self):
        self.notes.clear()
