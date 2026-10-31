"""
Central Prompt Budget Manager for Graph Memory Agent.
Enforces strict 4,096-token context allocation limits across all prompt components
and provides centralized, validated prompt assembly before calling LLM endpoints.
"""

from typing import Dict, List, Optional, Tuple


class PromptBudgetManager:
    """
    Allocates, clamps, and validates prompt components to guarantee total prompt
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
        """Heuristic estimator: 1 token ≈ 4 characters for technical / DFIR text."""
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
    def clamp_node_list(nodes: List[str], max_tokens: int = 300) -> List[str]:
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

    @classmethod
    def assemble_prompt(
        cls,
        system_text: str = "",
        query_and_plan: str = "",
        notebook_text: str = "",
        context_text: str = "",
        instructions: str = "",
    ) -> Tuple[str, str, int]:
        """
        Central prompt assembler. Clamps individual components to their allocations,
        verifies that combined prompt tokens do not exceed the 3,200 token budget,
        and returns (assembled_system_prompt, assembled_user_prompt, total_estimated_tokens).
        """
        clamped_system = cls.clamp_text(system_text, cls.ALLOCATIONS["system"])
        clamped_query = cls.clamp_text(query_and_plan, cls.ALLOCATIONS["query_and_plan"])
        clamped_nb = cls.clamp_text(notebook_text, cls.ALLOCATIONS["notebook"])
        clamped_ctx = cls.clamp_text(context_text, cls.ALLOCATIONS["retrieved_context"])

        user_prompt_sections = []
        if clamped_query:
            user_prompt_sections.append(clamped_query)
        if clamped_nb:
            user_prompt_sections.append(f"Recorded Evidence Memory:\n{clamped_nb}")
        if clamped_ctx:
            user_prompt_sections.append(f"Retrieved Context:\n{clamped_ctx}")
        if instructions:
            user_prompt_sections.append(instructions)

        user_prompt = "\n\n".join(user_prompt_sections)
        total_tokens = cls.estimate_tokens(clamped_system) + cls.estimate_tokens(user_prompt)

        # Hard safety clamp if combined prompt still exceeds max prompt budget
        if total_tokens > cls.MAX_PROMPT_BUDGET:
            excess_tokens = total_tokens - cls.MAX_PROMPT_BUDGET
            allowed_user_tokens = max(500, cls.estimate_tokens(user_prompt) - excess_tokens)
            user_prompt = cls.clamp_text(user_prompt, allowed_user_tokens)
            total_tokens = cls.estimate_tokens(clamped_system) + cls.estimate_tokens(user_prompt)

        return clamped_system, user_prompt, total_tokens
