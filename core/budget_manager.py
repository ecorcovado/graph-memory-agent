"""
Central Prompt Budget Manager for Graph Memory Agent.
Enforces strict 4,096-token context allocation limits across all prompt components.
"""

from typing import List


class PromptBudgetManager:
    """
    Allocates and clamps prompt components to guarantee total prompt
    and model generation strictly stay within a 4,096-token context window.
    """

    TOTAL_CONTEXT = 4096
    OUTPUT_RESERVE = 500
    SAFETY_MARGIN = 396
    MAX_PROMPT_BUDGET = TOTAL_CONTEXT - OUTPUT_RESERVE - SAFETY_MARGIN  # 3,200 tokens

    ALLOCATIONS = {
        "system": 400,
        "query_and_plan": 300,
        "notebook": 1000,
        "retrieved_context": 1500,
    }

    @staticmethod
    def estimate_tokens(text: str) -> int:
        """Heuristic estimator: 1 token ≈ 4 characters for English / technical text."""
        if not text:
            return 0
        return max(1, int(len(text) / 4))

    @staticmethod
    def clamp_text(text: str, max_tokens: int) -> str:
        """Truncates text safely if it exceeds component token budget."""
        max_chars = max_tokens * 4
        if len(text) <= max_chars:
            return text
        return text[:max_chars] + "\n... [TRUNCATED TO FIT CONTEXT BUDGET]"

    @staticmethod
    def clamp_node_list(nodes: List[str], max_tokens: int = 400) -> List[str]:
        """
        Guarantees that large graphs with hundreds of nodes do not overflow
        the initial node-selection prompt.
        """
        clamped = []
        current_tokens = 0
        for n in nodes:
            n_tokens = PromptBudgetManager.estimate_tokens(n) + 1
            if current_tokens + n_tokens > max_tokens:
                break
            clamped.append(n)
            current_tokens += n_tokens
        return clamped
