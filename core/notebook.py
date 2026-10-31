"""
Notebook Scratchpad (Inspired by GraphReader - arXiv:2406.14550)
Maintains bounded, immutable source-linked EvidenceNotes with fully-linked multi-source
provenance (SourceReference), per-note length clamping, and strict token budget enforcement.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Set, Tuple
from core.budget_manager import PromptBudgetManager


@dataclass(frozen=True)
class SourceReference:
    """Explicit immutable reference linking source document, record, fact IDs, and excerpt."""
    source_id: str
    source_record_id: str
    fact_ids: Tuple[str, ...] = field(default_factory=tuple)
    excerpt: str = ""
    source_location: str = ""


@dataclass(frozen=True)
class EvidenceNote:
    """Immutable evidence record with fully-linked multi-source provenance."""
    step: int
    content: str
    source_refs: Tuple[SourceReference, ...] = field(default_factory=tuple)
    confidence: float = 1.0

    @property
    def source_nodes(self) -> Tuple[str, ...]:
        nodes = []
        for ref in self.source_refs:
            for n in (ref.source_location, ref.source_record_id):
                if n and n not in nodes:
                    nodes.append(n)
        return tuple(nodes)

    @property
    def fact_ids(self) -> Tuple[str, ...]:
        ids = []
        for ref in self.source_refs:
            for fid in ref.fact_ids:
                if fid not in ids:
                    ids.append(fid)
        return tuple(sorted(set(ids)))

    @property
    def source_excerpts(self) -> Tuple[str, ...]:
        excerpts = []
        for ref in self.source_refs:
            if ref.excerpt and ref.excerpt not in excerpts:
                excerpts.append(ref.excerpt)
        return tuple(excerpts)

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
        source_node: str = "",
        content: str = "",
        fact_ids: Optional[List[str]] = None,
        source_excerpt: str = "",
        source_id: str = "src_doc",
        source_record_id: str = "",
        source_location: str = "",
        source_refs: Optional[List[SourceReference]] = None,
        confidence: float = 1.0,
    ):
        """Adds an immutable source-linked evidence note with linked multi-source provenance."""
        content = content.strip()
        if not content:
            return

        clamped_content = PromptBudgetManager.clamp_text(content, self.max_single_note_tokens)
        clamped_excerpt = PromptBudgetManager.clamp_text(source_excerpt, 150) if source_excerpt else ""
        immutable_fact_ids = tuple(sorted(set(fact_ids))) if fact_ids else ()

        # Build initial source references if not explicitly passed
        if source_refs:
            new_refs = tuple(source_refs)
        else:
            new_refs = (
                SourceReference(
                    source_id=source_id,
                    source_record_id=source_record_id or source_node,
                    fact_ids=immutable_fact_ids,
                    excerpt=clamped_excerpt,
                    source_location=source_location or source_node,
                ),
            )

        # Provenance-preserving deduplication:
        # If identical content exists, merge source references while keeping links intact
        for idx, n in enumerate(self.notes):
            if clamped_content.lower() == n.content.lower():
                existing_refs_list = list(n.source_refs)
                for incoming_ref in new_refs:
                    # Match by source_id and source_record_id if possible
                    merged = False
                    for r_idx, ex_ref in enumerate(existing_refs_list):
                        if (
                            ex_ref.source_id == incoming_ref.source_id
                            and ex_ref.source_record_id == incoming_ref.source_record_id
                        ):
                            combined_fids = tuple(sorted(set(ex_ref.fact_ids + incoming_ref.fact_ids)))
                            combined_excerpt = ex_ref.excerpt or incoming_ref.excerpt
                            existing_refs_list[r_idx] = SourceReference(
                                source_id=ex_ref.source_id,
                                source_record_id=ex_ref.source_record_id,
                                fact_ids=combined_fids,
                                excerpt=combined_excerpt,
                                source_location=ex_ref.source_location or incoming_ref.source_location,
                            )
                            merged = True
                            break
                    if not merged:
                        existing_refs_list.append(incoming_ref)

                self.notes[idx] = EvidenceNote(
                    step=n.step,
                    content=n.content,
                    source_refs=tuple(existing_refs_list),
                    confidence=max(n.confidence, confidence),
                )
                self._enforce_budget()
                return

        note = EvidenceNote(
            step=step,
            content=clamped_content,
            source_refs=new_refs,
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
