"""
Graph-based Agent Loop (Inspired by GraphReader - arXiv:2406.14550)
Executes rational planning, node selection, coarse-to-fine graph traversal,
and deterministic evidence-verified answer reasoning within a strict 4K context budget.
"""

import json
import re
from typing import List, Optional, Tuple
from core.graph_builder import KnowledgeGraph, LocalOllamaClient
from core.notebook import Notebook
from core.budget_manager import PromptBudgetManager


class GraphReaderAgent:
    """
    Autonomous agent navigating a directed MultiDiGraph Knowledge Graph
    to answer complex multi-hop queries with evidence verification.
    """

    def __init__(
        self,
        llm_client: LocalOllamaClient,
        kg: KnowledgeGraph,
        max_steps: int = 8,
        verbose: bool = True,
        notebook_max_tokens: int = 1000,
    ):
        self.llm = llm_client
        self.kg = kg
        self.max_steps = max_steps
        self.verbose = verbose
        self.notebook = Notebook(max_tokens=notebook_max_tokens)

    def _log(self, title: str, text: str):
        if self.verbose:
            print(f"\n[{title}]")
            print(f"{text.strip()}")

    def generate_rational_plan(self, question: str) -> str:
        """Step 1: Decomposes question into a step-by-step rational plan (Figure 7)."""
        prompt = (
            f"You are an investigative cybersecurity assistant. Outline a concise, step-by-step rational plan "
            f"identifying what key entities, relationships, and clues we need to discover in order "
            f"to answer this question.\n\n"
            f"Question: {question}\n\n"
            f"Rational Plan (1-3 sentences max):"
        )
        plan = self.llm.generate(prompt=prompt, temperature=0.1)
        self._log("1. RATIONAL PLAN", plan)
        return plan

    def select_initial_node(self, question: str, plan: str) -> str:
        """
        Step 2: Picks the most relevant starting node from available graph nodes (Figure 8).
        Enforces prompt budgeting on candidate nodes and uses keyword/degree fallbacks.
        """
        nodes = list(self.kg.graph.nodes())
        if not nodes:
            return ""

        # Enforce budget manager clamp on candidate nodes list
        bounded_nodes = PromptBudgetManager.clamp_node_list(nodes, max_tokens=300)

        prompt = (
            f"Question: {question}\n"
            f"Plan: {plan}\n"
            f"Available Graph Nodes: {', '.join(bounded_nodes)}\n\n"
            f"Select exactly ONE starting node from the list above that provides the best entry point for the investigation.\n"
            f"Output ONLY the exact node name, nothing else."
        )
        selected = self.llm.generate(prompt=prompt, temperature=0.1).strip().lower()

        # 1. Exact or substring match against LLM output
        for node in nodes:
            if node == selected or node in selected or selected in node:
                self._log("2. INITIAL NODE SELECTED", f"Target: '{node}'")
                return node

        # 2. Fallback: match against question & plan keywords
        combined_text = f"{question} {plan}".lower()
        keyword_matches = [
            n for n in nodes
            if n in combined_text or any(w in n for w in combined_text.split() if len(w) > 4)
        ]
        if keyword_matches:
            best_match = max(keyword_matches, key=lambda n: len(n))
            self._log("2. INITIAL NODE SELECTED (Query Overlap Fallback)", f"Target: '{best_match}'")
            return best_match

        # 3. Fallback: Pick highest degree hub node
        hub_node = max(nodes, key=lambda n: self.kg.graph.degree(n))
        self._log("2. INITIAL NODE SELECTED (Hub Degree Fallback)", f"Target: '{hub_node}'")
        return hub_node

    def explore_node(
        self,
        current_node: str,
        question: str,
        plan: str,
        step: int,
        visited_nodes: Optional[set] = None,
    ) -> Tuple[Optional[str], bool]:
        """
        Step 3: Explores current node's atomic facts, records source-linked evidence in notebook,
        and decides whether to hop to a neighbor or terminate.
        """
        visited = visited_nodes or set()

        # 1. Fetch atomic facts for this node
        fact_ids = self.kg.node_to_facts.get(current_node, [])
        facts = [self.kg.atomic_facts[fid] for fid in fact_ids if fid in self.kg.atomic_facts]
        facts_text = "\n".join([f"- [{f.fact_id}]: {f.text}" for f in facts])
        source_excerpt = facts[0].text if facts else ""

        # 2. Fetch directional relations with forensic classifications
        all_neighbors = self.kg.get_adjacent_nodes(current_node)
        unvisited_neighbors = [n for n in all_neighbors if n not in visited]

        relations_data = self.kg.get_neighbors_with_relations(current_node)
        relations_text = "\n".join(
            [f"- ({direction}) -> [{rel} | {cls}] -> {target}" for direction, rel, cls, target in relations_data]
        ) if relations_data else "None"

        # 3. Ask agent to inspect facts and take intermediate source-linked notes
        notes_prompt = (
            f"You are an investigative agent gathering evidence to answer a multi-hop question.\n\n"
            f"Target Question: {question}\n"
            f"Rational Plan: {plan}\n"
            f"Current Node: '{current_node}'\n"
            f"Current Notebook Memory:\n{self.notebook.format_for_prompt()}\n\n"
            f"Atomic Facts at '{current_node}':\n{facts_text}\n\n"
            f"Instructions:\n"
            f"Extract all relevant facts or intermediate clues that help execute the plan or connect entities.\n"
            f"Even if a fact does not give the final answer, record it if it identifies an entity's role, relation, action, or target.\n"
            f"If no facts are relevant, reply 'NONE'. Otherwise, write a concise summary of the discovered facts:"
        )
        new_clues = self.llm.generate(prompt=notes_prompt, temperature=0.1).strip()

        is_none = new_clues.upper().strip() in ["NONE", "NONE.", "'NONE'", '"NONE"']
        if new_clues and not is_none:
            self.notebook.add_note(
                step=step,
                source_node=current_node,
                content=new_clues,
                fact_ids=fact_ids,
                source_excerpt=source_excerpt,
            )
            self._log(f"STEP {step}: NOTEBOOK UPDATED", f"[{current_node}] ->\n{new_clues}")
        else:
            self._log(f"STEP {step}: NOTEBOOK", f"No new clues extracted at [{current_node}].")

        # 4. Decision: Stop or Hop to Neighbor?
        decision_prompt = (
            f"Target Question: {question}\n"
            f"Rational Plan: {plan}\n"
            f"Current Discovered Clues in Notebook:\n{self.notebook.format_for_prompt()}\n\n"
            f"Current Graph Node: '{current_node}'\n"
            f"Connected Edges & Relations:\n{relations_text}\n\n"
            f"Available Neighbor Nodes: {', '.join(all_neighbors) if all_neighbors else 'None'}\n"
            f"Unvisited Neighbor Nodes: {', '.join(unvisited_neighbors) if unvisited_neighbors else 'None'}\n\n"
            f"Task:\n"
            f"Do we already have sufficient verified evidence in the notebook to definitively answer the target question?\n"
            f"- If YES: reply strictly 'ACTION: TERMINATE'\n"
            f"- If NO: select ONE neighbor node to explore next.\n"
            f"Reply strictly in this format:\n"
            f"ACTION: HOP | <exact_node_name>\n"
        )
        decision = self.llm.generate(prompt=decision_prompt, temperature=0.1).strip()
        self._log(f"STEP {step}: AGENT DECISION", decision)

        if "TERMINATE" in decision.upper():
            return None, True

        # Parse target neighbor from decision
        match = re.search(r"(?:HOP\s*\|\s*|HOP:\s*|HOP\s+TO\s+)(.+)", decision, re.IGNORECASE)
        target = match.group(1).strip().lower() if match else decision.strip().lower()

        # Validate target against actual graph neighbors
        for n in all_neighbors:
            if n == target or n in target or target in n:
                return n, False

        # Action Validation Safeguard: If target is invalid, do not guess arbitrary nodes
        if unvisited_neighbors:
            self._log(
                f"STEP {step}: ACTION VALIDATION WARNING",
                f"Target '{target}' is not an adjacent node. Diverting to unvisited neighbor '{unvisited_neighbors[0]}'",
            )
            return unvisited_neighbors[0], False

        self._log(f"STEP {step}: TERMINATION", "No valid adjacent unvisited neighbors remaining.")
        return None, True

    def synthesize_answer(self, question: str) -> str:
        """
        Step 4: Compiles all recorded notes, performs evidence verification,
        and reasons the final answer strictly from supported fact IDs.
        """
        if self.notebook.is_empty():
            return "INSUFFICIENT EVIDENCE: No relevant clues were gathered during graph traversal."

        verified_fact_ids = self.notebook.get_all_fact_ids()
        notes = self.notebook.format_for_prompt()

        prompt = (
            f"You are a DFIR technical investigator answering a question based strictly on gathered evidence.\n\n"
            f"Target Question: {question}\n\n"
            f"Recorded Evidence Notebook (Verified Fact IDs: {', '.join(sorted(verified_fact_ids))}):\n{notes}\n\n"
            f"Instructions:\n"
            f"1. Check if the gathered evidence contains a definitive, unbroken logical chain connecting the entities in the question.\n"
            f"2. Cite the exact Fact IDs (e.g. C0_F1) supporting each link in the chain.\n"
            f"3. If evidence is missing, inconclusive, or unsupported, explicitly state: 'INSUFFICIENT EVIDENCE: [explain missing link]'.\n"
            f"4. If evidence is sufficient, provide the concise evidence chain with cited Fact IDs and the final direct answer.\n"
            f"Answer:"
        )
        answer = self.llm.generate(prompt=prompt, temperature=0.1)
        self._log("FINAL ANSWER", answer)
        return answer

    def run(self, question: str) -> str:
        """Executes full autonomous GraphReader workflow."""
        self._log("STARTING INVESTIGATION", f"Query: {question}")

        plan = self.generate_rational_plan(question)
        current_node = self.select_initial_node(question, plan)

        if not current_node:
            return "ERROR: Unable to identify a valid entry-point entity in the knowledge graph."

        visited_nodes = set()
        step = 1

        while current_node and step <= self.max_steps:
            visited_nodes.add(current_node)
            next_node, should_terminate = self.explore_node(
                current_node=current_node,
                question=question,
                plan=plan,
                step=step,
                visited_nodes=visited_nodes,
            )

            if should_terminate:
                self._log("EXPLORATION COMPLETE", "Sufficient information gathered.")
                break

            if not next_node:
                self._log("EXPLORATION COMPLETE", "No further reachable nodes.")
                break

            if next_node in visited_nodes:
                adjacent = self.kg.get_adjacent_nodes(next_node)
                unvisited = [n for n in adjacent if n not in visited_nodes]
                if unvisited:
                    current_node = unvisited[0]
                else:
                    self._log("EXPLORATION COMPLETE", "All local neighbors explored.")
                    break
            else:
                current_node = next_node

            step += 1

        return self.synthesize_answer(question)
