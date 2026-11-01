"""
Graph-based Agent Loop (Inspired by GraphReader - arXiv:2406.14550)
Executes rational planning, node selection, coarse-to-fine graph traversal,
and deterministic evidence-verified answer reasoning within a strict 4K context budget.
"""

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from core.graph_builder import KnowledgeGraph, LocalOllamaClient
from core.notebook import Notebook, SourceReference
from core.budget_manager import PromptBudgetManager


@dataclass
class InvestigationRequirements:
    """Explicit evidence requirements policy for proving claims in an investigation."""
    required_source_types: Set[str] = field(default_factory=set)
    required_predicates: Set[str] = field(default_factory=set)
    target_claim: str = ""
    expected_controller: Optional[str] = None
    expected_actuator: Optional[str] = None


def extract_json_objects(text: str) -> List[dict]:
    """Safely extracts top-level JSON objects respecting quoted strings, escape characters, and balanced braces."""
    results = []
    # 1. Check for markdown code blocks ```json ... ```
    code_blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    for block in code_blocks:
        try:
            results.append(json.loads(block))
        except Exception:
            pass
    if results:
        return results

    # 2. String-aware and escape-aware balanced brace scanner
    start = -1
    depth = 0
    in_string = False
    escape = False

    for i, ch in enumerate(text):
        if escape:
            escape = False
            continue
        if ch == '\\':
            if in_string:
                escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if not in_string:
            if ch == '{':
                if depth == 0:
                    start = i
                depth += 1
            elif ch == '}':
                if depth > 0:
                    depth -= 1
                    if depth == 0 and start != -1:
                        candidate = text[start:i+1]
                        try:
                            results.append(json.loads(candidate))
                        except Exception:
                            pass
                        start = -1
    return results


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
        self.execution_log: List[Tuple[str, str]] = []

    def _log(self, title: str, text: str):
        self.execution_log.append((title, text))
        if self.verbose:
            print(f"\n[{title}]")
            print(f"{text.strip()}")

    def rank_candidate_neighbors(self, current_node: str, candidates: List[str], question: str) -> List[str]:
        """
        Ranks unvisited candidate neighbors by forensic edge tier, question keyword overlap, and connectivity.
        Prioritizes OBSERVED events over INTELLIGENCE_REPORTED or CO_OCCURRENCE.
        """
        if not candidates:
            return []

        q_words = set(question.lower().split())
        relations = self.kg.get_neighbors_with_relations(current_node)

        scored = []
        for cand in candidates:
            score = 0
            # Tier prioritization
            for _, _, cls, target in relations:
                if target == cand:
                    if cls == "OBSERVED":
                        score += 50
                    elif cls == "INTELLIGENCE_REPORTED":
                        score += 30
                    elif cls == "UNCLASSIFIED":
                        score += 15
                    elif cls == "CO_OCCURRENCE":
                        score += 5

            # Question keyword overlap
            if any(w in cand for w in q_words if len(w) > 3):
                score += 25

            # Graph connectivity
            score += self.kg.graph.degree(cand)
            scored.append((score, cand))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored]

    def generate_rational_plan(self, question: str) -> str:
        """Step 1: Decomposes question into a step-by-step rational plan (Figure 7)."""
        sys_prompt, user_prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text="You are an investigative cybersecurity assistant.",
            query_and_plan=f"Target Question: {question}",
            instructions="Outline a concise, step-by-step rational plan identifying key entities, relationships, and clues to discover.\nRational Plan (1-3 sentences max):",
        )
        plan = self.llm.generate(prompt=user_prompt, system=sys_prompt, temperature=0.1)
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

        sys_prompt, user_prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text="You are an expert incident investigator.",
            query_and_plan=f"Question: {question}\nPlan: {plan}",
            context_text=f"Available Graph Nodes: {', '.join(bounded_nodes)}",
            instructions="Select exactly ONE starting node from the list above that provides the best entry point for the investigation.\nOutput ONLY the exact node name, nothing else.",
        )
        selected = self.llm.generate(prompt=user_prompt, system=sys_prompt, temperature=0.1).strip().lower()

        # 1. Exact match against LLM output (requires non-empty selection to prevent empty substring bug)
        if selected:
            for node in nodes:
                if node == selected:
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
        sys_prompt, notes_prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text="You are an investigative agent gathering evidence to answer a multi-hop question.",
            query_and_plan=f"Target Question: {question}\nRational Plan: {plan}",
            notebook_text=self.notebook.format_for_prompt(),
            context_text=f"Current Node: '{current_node}'\nAtomic Facts at '{current_node}':\n{facts_text}",
            instructions=(
                "Instructions:\n"
                "Extract all relevant facts or intermediate clues that help execute the plan or connect entities.\n"
                "Even if a fact does not give the final answer, record it if it identifies an entity's role, relation, action, or target.\n"
                "If no facts are relevant, reply 'NONE'. Otherwise, write a concise summary of the discovered facts:"
            ),
        )
        new_clues = self.llm.generate(prompt=notes_prompt, system=sys_prompt, temperature=0.1).strip()

        is_none = new_clues.upper().strip() in ["NONE", "NONE.", "'NONE'", '"NONE"']
        if new_clues and not is_none:
            source_refs = []
            for f in facts:
                source_refs.append(
                    SourceReference(
                        source_id=f.source_id or "src_doc",
                        source_record_id=f.source_record_id or current_node,
                        fact_ids=(f.fact_id,),
                        excerpt=f.text,
                        source_location=f.source_location or f"Node:{current_node}",
                    )
                )
            self.notebook.add_note(
                step=step,
                source_node=current_node,
                content=new_clues,
                fact_ids=fact_ids,
                source_excerpt=source_excerpt,
                source_refs=source_refs if source_refs else None,
            )
            self._log(f"STEP {step}: NOTEBOOK UPDATED", f"[{current_node}] ->\n{new_clues}")
        else:
            self._log(f"STEP {step}: NOTEBOOK", f"No new clues extracted at [{current_node}].")

        # 4. Decision: Stop or Hop to Neighbor?
        sys_prompt_dec, decision_prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text="You are an investigative agent deciding the next graph traversal step.",
            query_and_plan=f"Target Question: {question}\nRational Plan: {plan}",
            notebook_text=self.notebook.format_for_prompt(),
            context_text=(
                f"Current Graph Node: '{current_node}'\n"
                f"Connected Edges & Relations:\n{relations_text}\n\n"
                f"Available Neighbor Nodes: {', '.join(all_neighbors) if all_neighbors else 'None'}\n"
                f"Unvisited Neighbor Nodes: {', '.join(unvisited_neighbors) if unvisited_neighbors else 'None'}"
            ),
            instructions=(
                "Task:\n"
                "Do we already have sufficient verified evidence in the notebook to definitively answer the target question?\n"
                "- If YES: reply strictly 'ACTION: TERMINATE'\n"
                "- If NO: select ONE neighbor node to explore next.\n"
                "Reply strictly in this format:\n"
                "ACTION: HOP | <exact_node_name>"
            ),
        )
        decision = self.llm.generate(prompt=decision_prompt, system=sys_prompt_dec, temperature=0.1).strip()
        self._log(f"STEP {step}: AGENT DECISION", decision)

        lines = [line.strip() for line in decision.splitlines() if line.strip()]

        # Strict full-line termination check: requires exact full-line ACTION: TERMINATE
        # Disallows match on phrases like 'Do not use ACTION: TERMINATE' or loose 'TERMINATE'
        for line in lines:
            if re.fullmatch(r"ACTION:\s*TERMINATE", line, re.IGNORECASE):
                return None, True

        # Parse target neighbor from structured action line
        target = None
        for line in lines:
            hop_match = re.fullmatch(r"ACTION:\s*HOP\s*\|\s*(.+)", line, re.IGNORECASE)
            if hop_match:
                target = hop_match.group(1).strip().lower()
                break

        # Resilient fallback parser if model deviated slightly from format
        if not target:
            match = re.search(r"(?:HOP\s*\|\s*|HOP:\s*|HOP\s+TO\s+)(.+)", decision, re.IGNORECASE)
            target = match.group(1).strip().lower() if match else decision.strip().lower()

        # Blocked Edge Traversal Safeguard: Agent never treats blocked connections as valid attack path transitions
        if target in unvisited_neighbors:
            if self.kg.is_transition_blocked(current_node, target):
                self._log(
                    f"STEP {step}: ATTACK PATH BLOCKED",
                    f"Lateral movement from '{current_node}' to '{target}' was blocked by boundary defense. Attack path cannot progress across blocked conduit.",
                )
                return None, True
            return target, False

        if target in visited:
            self._log(
                f"STEP {step}: TRAVERSAL SAFEGUARD",
                f"Target '{target}' was already visited. Selecting unvisited neighbor via ranked routing.",
            )

        # Action Validation Safeguard: Rank unvisited neighbors by forensic relevance instead of guessing arbitrarily
        ranked_unvisited = self.rank_candidate_neighbors(current_node, unvisited_neighbors, question)
        if ranked_unvisited:
            chosen = ranked_unvisited[0]
            if self.kg.is_transition_blocked(current_node, chosen):
                self._log(
                    f"STEP {step}: ATTACK PATH BLOCKED",
                    f"Lateral movement from '{current_node}' to '{chosen}' was blocked by boundary defense. Attack path cannot progress across blocked conduit.",
                )
                return None, True
            self._log(
                f"STEP {step}: ACTION VALIDATION ROUTING",
                f"Target '{target}' invalid or unranked. Ranked routing to '{chosen}' (tier/relevance priority)",
            )
            return chosen, False

        self._log(f"STEP {step}: TERMINATION", "No valid adjacent unvisited neighbors remaining.")
        return None, True

    def synthesize_answer(self, question: str, requirements: Optional[InvestigationRequirements] = None) -> str:
        """
        Step 4: Compiles all recorded notes, performs deterministic evidence verification,
        enforces degraded evidence policies and investigation requirements,
        validates cited fact IDs against immutable notebook facts, and fails closed if evidence is missing.
        """
        # 1. Enforce Investigation Requirements & Degraded Evidence Policy (Additive)
        q_lower = question.lower()
        manifest_targets = getattr(self.kg, "investigation_targets", {})
        default_ctrl = manifest_targets.get("primary_controller")
        default_act = manifest_targets.get("primary_actuator")

        if requirements is None:
            req_sources = set()
            req_preds = set()
            claims = []
            exp_ctrl = None
            exp_act = None

            if "plc" in q_lower and any(w in q_lower for w in ["modif", "compromis", "reach", "logic", "attack", "controller"]):
                req_sources.add("CONTROLLER_AUDIT")
                ctrl_preds = [p for p in manifest_targets.get("required_predicates", []) if p in ["modified_logic", "polled_controller", "firmware_update"]]
                if ctrl_preds:
                    req_preds.update(ctrl_preds)
                else:
                    req_preds.add("modified_logic")
                claims.append("PLC logic modification")
                ctrl_match = re.search(r"plc-[\w-]+", q_lower)
                exp_ctrl = ctrl_match.group(0) if ctrl_match else default_ctrl

            if any(w in q_lower for w in ["pressure", "valve", "dump", "interlock", "safety", "bypass"]):
                req_sources.add("PROCESS_TELEMETRY")
                act_preds = [p for p in manifest_targets.get("required_predicates", []) if p in ["interlock_status", "commanded_state", "measured_value", "has_observation"]]
                if act_preds:
                    req_preds.update(act_preds)
                else:
                    req_preds.add("interlock_status")
                claims.append("Physical process interlock telemetry")
                act_match = re.search(r"[a-z]{2}-\d{4}", q_lower)
                exp_act = act_match.group(0) if act_match else default_act

            if req_sources:
                requirements = InvestigationRequirements(
                    required_source_types=req_sources,
                    required_predicates=req_preds,
                    target_claim=" and ".join(claims),
                    expected_controller=exp_ctrl or default_ctrl,
                    expected_actuator=exp_act or default_act,
                )

        if requirements and hasattr(self.kg, "ingestion_report") and self.kg.ingestion_report:
            missing_sources = self.kg.ingestion_report.missing_sources
            missing_types = set()
            from core.case_loader import CaseLoader
            for ms in missing_sources:
                fname = ms.split("/")[-1].split("\\")[-1]
                stype = CaseLoader.SOURCE_TYPE_MAPPING.get(fname, fname)
                missing_types.add(stype)

            unmet_sources = requirements.required_source_types.intersection(missing_types)
            if unmet_sources:
                self._log("DEGRADED EVIDENCE POLICY ENFORCED", f"Missing required sources: {unmet_sources}")
                return (
                    f"INSUFFICIENT EVIDENCE: Investigation degraded. Required telemetry source(s) "
                    f"{sorted(list(unmet_sources))} missing from case ingestion; cannot substantiate "
                    f"claims regarding {requirements.target_claim}."
                )

        if self.notebook.is_empty():
            return "INSUFFICIENT EVIDENCE: No relevant clues were gathered during graph traversal."

        verified_fact_ids = self.notebook.get_all_fact_ids()
        notebook_facts = [self.kg.atomic_facts.get(fid) for fid in verified_fact_ids if fid in self.kg.atomic_facts]

        # Enforce that ALL required predicates and sources are substantiated in gathered notebook
        if requirements:
            if requirements.required_predicates:
                present_predicates = {f.relation for f in notebook_facts if f}
                missing_predicates = requirements.required_predicates - present_predicates
                if missing_predicates:
                    # Forensic Distinction: Check for positive evidence that the intrusion was actively blocked
                    blocked_facts = [
                        f for f in notebook_facts
                        if f and (
                            f.relation in ["blocked_connection_to", "connection_denied", "access_denied", "process_terminated"]
                            or f.attributes.get("action") in ["BLOCK", "DENY"]
                            or "block" in str(f.attributes.get("rule_id", "")).lower()
                        )
                    ]
                    if blocked_facts:
                        bf = blocked_facts[0]
                        self._log("DEFENSE EVIDENCE ENFORCED", f"Adversary path actively blocked: {bf.text}")
                        return (
                            f"REFUTED: Positive telemetry verifies adversary lateral movement was blocked "
                            f"({bf.subject} {bf.relation} {bf.object_entity} [{bf.fact_id}]). "
                            f"Adversary did not compromise target controller or process assets."
                        )

                    self._log("EVIDENCE POLICY ENFORCED", f"Missing required predicate(s): {missing_predicates}")
                    return (
                        f"INSUFFICIENT EVIDENCE: Gathered evidence lacks required predicate(s) "
                        f"{sorted(list(missing_predicates))} to prove {requirements.target_claim}."
                    )

            if requirements.required_source_types:
                present_sources = {f.source_type for f in notebook_facts if f}
                missing_sources = requirements.required_source_types - present_sources
                if missing_sources:
                    self._log("DEGRADED EVIDENCE POLICY ENFORCED", f"Missing evidence from required sources: {missing_sources}")
                    return (
                        f"INSUFFICIENT EVIDENCE: Investigation incomplete. Gathered notebook evidence lacks "
                        f"telemetry from required source type(s) {sorted(list(missing_sources))}."
                    )

        notes = self.notebook.format_for_prompt()

        sys_prompt, prompt, _ = PromptBudgetManager.assemble_prompt(
            system_text="You are a DFIR technical investigator answering a question based strictly on gathered evidence.",
            query_and_plan=f"Target Question: {question}",
            notebook_text=notes,
            context_text=f"Verified Immutable Fact IDs available: {', '.join(sorted(verified_fact_ids))}",
            instructions=(
                "Instructions:\n"
                "1. Check if the gathered evidence contains a definitive, unbroken logical chain connecting the entities in the question.\n"
                "2. Cite the exact Fact IDs supporting each link in the chain (choose strictly from the Verified Immutable Fact IDs listed above).\n"
                "3. If evidence is missing, inconclusive, or unsupported, explicitly state: 'INSUFFICIENT EVIDENCE: [explain missing link]'.\n"
                "4. If telemetry proves the attack path was actively blocked or halted by security controls, state: 'REFUTED: [explain defense blockage]'.\n"
                "5. If evidence is sufficient and unbroken, provide the concise evidence chain with cited Fact IDs and the final direct answer.\n"
                "Answer:"
            ),
        )
        answer = self.llm.generate(prompt=prompt, system=sys_prompt, temperature=0.1)

        # 2. Extract potential structured JSON output using balanced brace scanner
        parsed_jsons = extract_json_objects(answer)
        json_cited = set()
        structured_claims = []
        for pjson in parsed_jsons:
            if isinstance(pjson, dict):
                for item in pjson.get("cited_fact_ids", []):
                    if isinstance(item, str):
                        json_cited.add(item.strip())
                for claim in pjson.get("claims", []):
                    if isinstance(claim, dict):
                        structured_claims.append(claim)
                        for item in claim.get("fact_ids", []):
                            if isinstance(item, str):
                                json_cited.add(item.strip())

        # Exact Set Membership against verified immutable fact registry
        cited_fact_ids = {fid for fid in verified_fact_ids if fid in answer}
        cited_fact_ids.update(json_cited.intersection(verified_fact_ids))

        # 3. Detect fabricated or unverified Fact IDs cited in the answer
        candidate_patterns = [
            r"\b[a-f0-9]{8,64}_R\d+_[A-Za-z0-9_]+\b",
            r"\b[a-f0-9]{8,64}_C\d+_F\d+\b",
            r"\bC\d+_F\d+\b",
            r"\bFACT_[A-Za-z0-9_]+\b",
        ]
        candidate_ids = set()
        for pat in candidate_patterns:
            for match in re.findall(pat, answer):
                candidate_ids.add(match)
        candidate_ids.update(json_cited)

        invalid_citations = {
            cid for cid in candidate_ids
            if cid not in verified_fact_ids
        }

        if invalid_citations:
            self._log("EVIDENCE VERIFICATION FAILED", f"Answer cited unverified or fabricated Fact IDs: {invalid_citations}")
            return (
                f"INSUFFICIENT EVIDENCE: The investigation failed deterministic evidence validation. "
                f"The generated answer cited unverified fact IDs: {', '.join(sorted(invalid_citations))}"
            )

        if not cited_fact_ids and "insufficient evidence" not in answer.lower() and "refuted" not in answer.lower():
            self._log("EVIDENCE VERIFICATION FAILED", "Answer contained no citations to verified atomic facts.")
            return (
                "INSUFFICIENT EVIDENCE: The generated answer failed forensic validation because it contained "
                "no verifiable citations to atomic fact IDs."
            )

        # 4. Claim-Level Predicate and Entity Validation:
        all_cited_facts = [self.kg.atomic_facts.get(fid) for fid in cited_fact_ids if fid in self.kg.atomic_facts]
        expected_target_ctrl = (
            (requirements.expected_controller if requirements else None)
            or default_ctrl
        )
        expected_target_act = (
            (requirements.expected_actuator if requirements else None)
            or default_act
        )

        # A. Validate Structured Claims if present
        if structured_claims:
            for claim in structured_claims:
                stmt = claim.get("statement", "").lower()
                c_fids = claim.get("fact_ids", [])
                claim_facts = [self.kg.atomic_facts.get(fid) for fid in c_fids if fid in self.kg.atomic_facts]
                stmt_ctrls = set(re.findall(r"plc-[\w-]+", stmt))
                if ("plc" in stmt or "logic" in stmt or "controller" in stmt) and any(w in stmt for w in ["modif", "download", "logic", "compromis"]):
                    targets_to_check = stmt_ctrls if stmt_ctrls else ({expected_target_ctrl} if expected_target_ctrl else set())
                    if not targets_to_check:
                        self._log("CLAIM VALIDATION FAILED", "Structured claim asserts controller modification, but no target controller was specified in query or case manifest.")
                        return (
                            "INSUFFICIENT EVIDENCE: Structured claim asserts controller logic modification, but no target "
                            "controller was defined in query or case manifest."
                        )
                    # Every claimed controller in THIS claim statement must have verified predicate 'modified_logic' from CONTROLLER_AUDIT
                    for target_c in targets_to_check:
                        has_exact_target = any(
                            f.relation == "modified_logic"
                            and f.source_type == "CONTROLLER_AUDIT"
                            and f.object_entity == target_c
                            for f in claim_facts if f
                        )
                        if not has_exact_target:
                            self._log("CLAIM VALIDATION FAILED", f"Structured claim '{stmt}' does not verify predicate 'modified_logic' from CONTROLLER_AUDIT on target controller '{target_c}'.")
                            return (
                                f"INSUFFICIENT EVIDENCE: Claim '{claim.get('statement')}' cites facts that do not "
                                f"verify required predicate 'modified_logic' from CONTROLLER_AUDIT on target controller '{target_c}'."
                            )

        # B. Plaintext Claim Validation (prevents verification bypass on unstructured text)
        ans_lower = answer.lower()

        # Check for affirmative assertions of PLC / controller logic modification
        # An affirmative assertion claims compromise/modification/download, excluding explicit negative statements
        has_plc_keywords = ("plc" in ans_lower or "controller" in ans_lower) and any(w in ans_lower for w in ["modif", "download", "logic", "compromis", "injected"])
        is_negative_context = any(neg in ans_lower for neg in [
            "did not compromise", "no logic modification", "did not reach plc", "not reach the plc",
            "prevented from modifying", "without modifying", "unmodified"
        ])

        if has_plc_keywords and not is_negative_context:
            claimed_ctrls = set(re.findall(r"plc-[\w-]+", ans_lower))
            if not claimed_ctrls and not expected_target_ctrl:
                self._log("CLAIM VALIDATION FAILED", "Answer asserts controller modification, but no target controller was specified in query or case manifest.")
                return (
                    "INSUFFICIENT EVIDENCE: Answer asserts controller logic modification, but no target controller "
                    "was defined in query or case manifest."
                )

            # Independent Per-Claim Target Binding: EVERY claimed controller must be independently substantiated!
            # A valid fact for plc-ballast-01 CANNOT substantiate an additional unsupported claim for plc-other-01!
            if claimed_ctrls:
                for ctrl in sorted(list(claimed_ctrls)):
                    valid_ctrl_facts = [
                        f for f in all_cited_facts
                        if f and f.relation == "modified_logic"
                        and f.source_type == "CONTROLLER_AUDIT"
                        and f.object_entity == ctrl
                    ]
                    if not valid_ctrl_facts:
                        self._log("CLAIM VALIDATION FAILED", f"Assertion of PLC logic modification on '{ctrl}' lacks verified fact with predicate 'modified_logic' from CONTROLLER_AUDIT on '{ctrl}'.")
                        return (
                            f"INSUFFICIENT EVIDENCE: Generated answer asserts logic modification on controller '{ctrl}', "
                            f"but cited facts do not verify required predicate 'modified_logic' from CONTROLLER_AUDIT on '{ctrl}'."
                        )

            # Furthermore, if an expected target controller was required, it must be substantiated
            if expected_target_ctrl and expected_target_ctrl not in claimed_ctrls:
                self._log("CLAIM VALIDATION FAILED", f"Answer did not substantiate expected target controller '{expected_target_ctrl}'.")
                return (
                    f"INSUFFICIENT EVIDENCE: Generated answer claims modification on non-target controller(s) "
                    f"{sorted(list(claimed_ctrls))}, which does not substantiate expected target '{expected_target_ctrl}'."
                )

        # Check for affirmative assertions of safety interlock suppression / bypass
        has_interlock_keywords = "interlock" in ans_lower and any(w in ans_lower for w in ["bypass", "suppress", "inhibit", "override", "disarm"])
        is_interlock_negative = any(neg in ans_lower for neg in ["not bypassed", "no interlock was bypassed", "armed and active", "armed_and_active"])

        if has_interlock_keywords and not is_interlock_negative:
            claimed_tags = set(re.findall(r"[a-z]{2}-\d{4}", ans_lower))
            if not claimed_tags and not expected_target_act:
                self._log("CLAIM VALIDATION FAILED", "Answer asserts interlock bypass, but no target actuator was specified in query or case manifest.")
                return (
                    "INSUFFICIENT EVIDENCE: Answer asserts safety interlock bypass, but no target actuator "
                    "was defined in query or case manifest."
                )

            # Independent Per-Claim Tag Binding: EVERY claimed tag must have verified suppressed interlock telemetry!
            if claimed_tags:
                for tag in sorted(list(claimed_tags)):
                    valid_tag_facts = []
                    for f in all_cited_facts:
                        if not f:
                            continue
                        if f.source_type == "PROCESS_TELEMETRY" and f.relation == "interlock_status":
                            tag_match = (f.subject == tag or f.object_entity == tag or tag in f.text.lower())
                            status_val = str(f.object_entity or f.attributes.get("interlock_status", "")).lower()
                            is_bypassed = any(term in status_val for term in ["suppress", "bypass", "inhibit", "override", "disarm", "false", "0"])
                            if tag_match and is_bypassed:
                                valid_tag_facts.append(f)
                    if not valid_tag_facts:
                        self._log("CLAIM VALIDATION FAILED", f"Assertion of safety interlock bypass on tag '{tag}' lacks cited fact verifying suppressed interlock on '{tag}'.")
                        return (
                            f"INSUFFICIENT EVIDENCE: Generated answer asserts safety interlock bypass on tag '{tag}', "
                            f"but cited facts do not verify suppressed/bypassed interlock telemetry on '{tag}'."
                        )

            if expected_target_act and expected_target_act not in claimed_tags:
                self._log("CLAIM VALIDATION FAILED", f"Answer claims bypass on '{claimed_tags}', which does not substantiate expected actuator '{expected_target_act}'.")
                return (
                    f"INSUFFICIENT EVIDENCE: Generated answer claims bypass on non-target tag(s) "
                    f"{sorted(list(claimed_tags))}, which does not substantiate expected target '{expected_target_act}'."
                )

        self._log("EVIDENCE VERIFIED", f"All cited Fact IDs verified against notebook: {cited_fact_ids}")
        self._log("FINAL ANSWER", answer)
        return answer

    def run(self, question: str, requirements: Optional[InvestigationRequirements] = None) -> str:
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
                ranked = self.rank_candidate_neighbors(next_node, unvisited, question)
                if ranked:
                    current_node = ranked[0]
                else:
                    self._log("EXPLORATION COMPLETE", "All local neighbors explored.")
                    break
            else:
                current_node = next_node

            step += 1

        return self.synthesize_answer(question, requirements=requirements)
