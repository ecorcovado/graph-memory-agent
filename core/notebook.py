"""
Notebook Scratchpad (Inspired by GraphReader - arXiv:2406.14550)
Maintains bounded, immutable source-linked EvidenceNotes with multi-source provenance,
per-note length clamping, and strict token budget enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Set, Tuple
from core.budget_manager import PromptBudgetManager


@dataclass(frozen=True)
class EvidenceNote:
    """Immutable evidence record with multi-source provenance references."""
    step: int
    source_nodes: Tuple[str, ...] = field(default_factory=tuple)
    fact_ids: Tuple[str, ...] = field(default_factory=tuple)
    source_excerpts: Tuple[str, ...] = field(default_factory=tuple)
    content: str = ""
    confidence: float = 1.0

    @property
    def source_node(self) -> str:
        return self.source_nodes[0] if self.source_nodes else ""

    @property
    def source_excerpt(self) -> str:
        return self.source_excerpts[0] if self.source_excerpts else ""


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
            f"{' '.join(n.source_nodes)} {' '.join(n.fact_ids)} {' '.join(n.source_excerpts)} {n.content}"
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
        """Adds an immutable source-linked evidence note with multi-source merging."""
        content = content.strip()
        if not content:
            return

        clamped_content = PromptBudgetManager.clamp_text(content, self.max_single_note_tokens)
        clamped_excerpt = PromptBudgetManager.clamp_text(source_excerpt, 150) if source_excerpt else ""
        immutable_fact_ids = tuple(sorted(set(fact_ids))) if fact_ids else ()

        # Provenance-preserving deduplication:
        # Merge source nodes, fact IDs, and source excerpts across multiple corroborating sources
        for idx, n in enumerate(self.notes):
            if clamped_content.lower() == n.content.lower():
                merged_nodes = tuple(sorted(set(n.source_nodes + (source_node,))))
                merged_ids = tuple(sorted(set(n.fact_ids + immutable_fact_ids)))
                new_excerpts = list(n.source_excerpts)
                if clamped_excerpt and clamped_excerpt not in new_excerpts:
                    new_excerpts.append(clamped_excerpt)
                merged_excerpts = tuple(new_excerpts)

                self.notes[idx] = EvidenceNote(
                    step=n.step,
                    source_nodes=merged_nodes,
                    fact_ids=merged_ids,
                    source_excerpts=merged_excerpts,
                    content=n.content,
                    confidence=max(n.confidence, confidence),
                )
                self._enforce_budget()
                return

        note = EvidenceNote(
            step=step,
            source_nodes=(source_node,),
            fact_ids=immutable_fact_ids,
            source_excerpts=(clamped_excerpt,) if clamped_excerpt else (),
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

    def verify_citation(self, fact_id: str) -> bool:
        """Verifies if a specific fact ID was legitimately gathered during investigation."""
        return fact_id in self.get_all_fact_ids()

    def format_for_prompt(self) -> str:
        """Formats the notebook memory for LLM context inclusion."""
        if not self.notes:
            return "No verified notes recorded yet."
        formatted = []
        for n in self.notes:
            nodes_str = ", ".join(n.source_nodes)
            facts_str = f" [Facts: {', '.join(n.fact_ids)}]" if n.fact_ids else ""
            excerpts_str = f"\n  Sources: {' | '.join(n.source_excerpts)}" if n.source_excerpts else ""
            formatted.append(
                f"[Step {n.step} | Sources: {nodes_str}{facts_str}]{excerpts_str}\n  Evidence: {n.content}"
            )
        return "\n\n".join(formatted)

    def is_empty(self) -> bool:
        return len(self.notes) == 0

    def clear(self):
        self.notes.clear()
