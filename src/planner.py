"""
Agent-Planner Module for SymAgent.

Implements symbolic rule induction from KG (Section 4.1):
1. BM25 retrieval of seed questions with similar structure
2. BFS to sample closed paths from query entity to answer entity
3. Generalize closed paths to symbolic rules (first-order logic formulae)
4. Few-shot prompting with rule demonstrations M = {(q_seed_i, P_i)}
5. Generate rule body: p ~ pi_theta(·| rho_Plan, q, M)

The planner leverages LLM's inductive reasoning capability to extract
symbolic rules from KGs, guiding efficient question decomposition.
"""

import logging
import re
from typing import Any, Optional

from rank_bm25 import BM25Okapi

from .kg_environment import KGEnvironment
from .llm_client import LLMClient

logger = logging.getLogger(__name__)

# System prompt for the planner's symbolic rule induction
PLANNER_SYSTEM_PROMPT = """You are a symbolic rule induction assistant for knowledge graph reasoning.
Given a question and a set of demonstration examples (seed questions with their discovered relational paths),
your task is to generate the most likely symbolic rules (relational reasoning paths) that can answer the question.

The symbolic rules should follow the format:
r(x,y) <- r1(x,z1) ∧ r2(z1,z2) ∧ ... ∧ r_n(z_{n-1},y)

where the rule body forms a closed chain of relations.
You should output the potential reasoning paths as a list of relation chains, one per line.
Each relation chain should be a comma-separated list of relations enclosed in brackets.
Example: [relation1, relation2, relation3]

Output ONLY the reasoning paths, one per line."""


class AgentPlanner:
    """Agent-Planner: Symbolic rule induction from KG for question decomposition.

    As described in Section 4.1 of the paper:
    1. Retrieve seed questions via BM25 from the training set
    2. For each seed question, use BFS to sample closed paths from query entity
       to answer entity in the KG
    3. Generalize closed paths to symbolic rules by replacing entities with variables
    4. Use few-shot demonstrations to prompt LLM for rule generation

    Attributes:
        kg: The KG environment.
        llm: The LLM client.
        num_seed_questions: Number of seed questions to retrieve via BM25.
        max_bfs_depth: Maximum BFS depth for path sampling.
        max_paths_per_seed: Maximum closed paths per seed question.
        max_rules: Maximum symbolic rules to return.
    """

    def __init__(
        self,
        kg: KGEnvironment,
        llm: LLMClient,
        num_seed_questions: int = 3,
        max_bfs_depth: int = 4,
        max_paths_per_seed: int = 5,
        max_rules: int = 10,
        planner_temperature: float = 0.3,
    ):
        self.kg = kg
        self.llm = llm
        self.num_seed_questions = num_seed_questions
        self.max_bfs_depth = max_bfs_depth
        self.max_paths_per_seed = max_paths_per_seed
        self.max_rules = max_rules
        self.planner_temperature = planner_temperature

        # BM25 index over training questions for seed retrieval
        self._train_questions: list[dict[str, Any]] = []
        self._bm25: Optional[BM25Okapi] = None

    def build_seed_index(self, train_data: list[dict[str, Any]]) -> None:
        """Build BM25 index over training questions for seed retrieval.

        Args:
            train_data: List of training examples, each with 'question' key
                       and optionally 'question_entity' and 'answer_entities'.
        """
        self._train_questions = train_data
        tokenized = [q["question"].lower().split() for q in train_data]
        self._bm25 = BM25Okapi(tokenized)
        logger.info(
            f"Built seed question BM25 index with {len(train_data)} questions"
        )

    def retrieve_seed_questions(
        self,
        question: str,
        top_k: Optional[int] = None,
    ) -> list[dict[str, Any]]:
        """Retrieve seed questions similar to the given question using BM25.

        As per Section 4.1: "we employ BM25 to retrieve a set of seed questions
        {q_seed_i} from the training set, where each q_seed_i shares similar
        question structure with q."

        Args:
            question: The input question.
            top_k: Number of seeds to retrieve (defaults to num_seed_questions).

        Returns:
            List of seed question dicts.
        """
        if self._bm25 is None:
            logger.warning("Seed index not built. Call build_seed_index first.")
            return []

        k = top_k or self.num_seed_questions
        tokenized_query = question.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:k]

        seeds = [self._train_questions[i] for i in top_indices if scores[i] > 0]
        return seeds

    def sample_closed_paths(
        self,
        query_entity: str,
        answer_entities: list[str],
        max_depth: Optional[int] = None,
        max_paths: Optional[int] = None,
    ) -> list[list[str]]:
        """Sample closed paths from query entity to answer entities via BFS.

        As per Section 4.1: "For each q_seed_i, we adopt BFS to sample a set
        of closed paths P_i from the query entity e_q to the answer entity e_a
        within the KG G."

        Each path p_ij = r1(e_q, e1) ∧ r2(e1, e2) ... ∧ r_L(e_{L-1}, e_a)
        is a sequence of relations.

        Args:
            query_entity: The question entity.
            answer_entities: List of answer entities.
            max_depth: Maximum path depth.
            max_paths: Maximum paths per answer entity.

        Returns:
            List of relation paths (each path is a list of relation strings).
        """
        depth = max_depth or self.max_bfs_depth
        max_p = max_paths or self.max_paths_per_seed

        all_paths: list[list[str]] = []
        for ans_ent in answer_entities:
            paths = self.kg.bfs_find_paths(
                query_entity, ans_ent, max_depth=depth, max_paths=max_p
            )
            for path in paths:
                # Extract just the relation sequence
                rel_path = [rel for rel, _ in path]
                all_paths.append(rel_path)

        return all_paths

    def generalize_to_rules(self, paths: list[list[str]]) -> list[str]:
        """Generalize closed paths to symbolic rules.

        Replace specific entities with variables to form first-order logic rules.
        A path [r1, r2, ..., rn] becomes:
        r_h(x, y) <- r1(x, z1) ∧ r2(z1, z2) ∧ ... ∧ r_n(z_{n-1}, y)

        Args:
            paths: List of relation paths.

        Returns:
            List of symbolic rule strings.
        """
        rules = []
        for path in paths:
            if not path:
                continue
            variables = ["x"]
            for i in range(len(path) - 1):
                variables.append(f"z{i+1}")
            variables.append("y")

            body_parts = []
            for i, rel in enumerate(path):
                body_parts.append(f"{rel}({variables[i]}, {variables[i+1]})")

            head = f"r({variables[0]}, {variables[-1]})"
            body = " ∧ ".join(body_parts)
            rule = f"{head} <- {body}"
            rules.append(rule)

        return rules

    def build_demonstrations(
        self,
        seed_questions: list[dict[str, Any]],
    ) -> list[tuple[str, list[str]]]:
        """Build few-shot demonstrations M = {(q_seed_i, P_i)}.

        For each seed question, sample closed paths and generalize to rules.

        Args:
            seed_questions: List of seed question dicts with 'question',
                          'question_entity', and 'answer_entities'.

        Returns:
            List of (question, rules) tuples for few-shot demonstration.
        """
        demonstrations = []
        for seed in seed_questions:
            q = seed.get("question", "")
            q_ent = seed.get("question_entity", "")
            a_ents = seed.get("answer_entities", [])

            if not q_ent or not a_ents:
                continue

            paths = self.sample_closed_paths(q_ent, a_ents)
            rules = self.generalize_to_rules(paths)

            if rules:
                demonstrations.append((q, rules))

        return demonstrations

    def plan(
        self,
        question: str,
        question_entity: Optional[str] = None,
        answer_entities: Optional[list[str]] = None,
    ) -> list[list[str]]:
        """Generate symbolic rules for a given question.

        Implements the full planning pipeline (Equation 3):
        p ~ pi_theta(·| rho_Plan, q, M)

        Step 1: Retrieve seed questions via BM25
        Step 2: For each seed, sample closed paths via BFS and generalize to rules
        Step 3: Construct few-shot demonstrations M
        Step 4: Prompt LLM to generate appropriate rule bodies

        If question_entity and answer_entities are provided for the current question,
        also include paths from the KG directly.

        Args:
            question: The input question.
            question_entity: The linked entity from the question.
            answer_entities: Optional ground truth answer entities for direct path sampling.

        Returns:
            List of relation paths (each is a list of relation strings).
        """
        # Step 1: Retrieve seed questions
        seeds = self.retrieve_seed_questions(question)


        # Step 2: Build demonstrations from seeds
        demonstrations = self.build_demonstrations(seeds)

        # Step 2.5: If question_entity is not provided, use BM25 entity linking
        if question_entity is None:
            bm25_results = self.kg.bm25_retrieve_entities(question, top_k=1)
            if bm25_results:
                question_entity = bm25_results[0][0]
                logger.info(
                    f"Planner BM25 entity linking: '{question_entity}' "
                    f"for question: {question[:50]}..."
                )

        # Step 3: Use LLM for rule induction based on demonstrations (Eq. 3)
        # p ~ pi_theta(·|rho_Plan, q, M)
        if demonstrations:
            llm_paths = self._llm_induce_rules(question, demonstrations)
            all_paths = llm_paths
        else:
            # Fallback: if no demonstrations available, use direct KG paths
            # as a heuristic (engineering necessity, not part of Eq. 3)
            logger.info("No demonstrations available, falling back to direct KG paths")
            all_paths = []
            if question_entity:
                all_paths = self.kg.get_reasoning_paths(
                    question_entity,
                    max_depth=self.max_bfs_depth,
                    max_paths=self.max_rules,
                )

        # Deduplicate and limit
        seen: set[tuple[str, ...]] = set()
        unique_paths: list[list[str]] = []
        for path in all_paths:
            key = tuple(path)
            if key not in seen:
                seen.add(key)
                unique_paths.append(path)

        return unique_paths[: self.max_rules]

    def _llm_induce_rules(
        self,
        question: str,
        demonstrations: list[tuple[str, list[str]]],
    ) -> list[list[str]]:
        """Use LLM to induce symbolic rules from demonstrations.

        Constructs the few-shot prompt with demonstrations M = {(q_seed_i, P_i)}
        and prompts the LLM to generate rule bodies for the target question.

        Args:
            question: The target question.
            demonstrations: List of (seed_question, rules) pairs.

        Returns:
            List of relation paths extracted from LLM output.
        """
        # Build few-shot prompt
        demo_text = ""
        for i, (seed_q, rules) in enumerate(demonstrations):
            demo_text += f"Example {i+1}: {seed_q}\n"
            demo_text += "Rules:\n"
            for rule in rules:
                demo_text += f"  {rule}\n"
            # Also show as relation path
            paths = self._rules_to_paths(rules)
            for path in paths:
                demo_text += f"  Path: [{', '.join(path)}]\n"
            demo_text += "\n"

        user_prompt = (
            f"Here are some examples of questions and their symbolic rules "
            f"discovered from the knowledge graph:\n\n"
            f"{demo_text}"
            f"Now, given the following question, generate potential symbolic "
            f"rules (relational reasoning paths) that could help answer it:\n\n"
            f"Question: {question}\n\n"
            f"Output the reasoning paths as comma-separated relations in brackets, "
            f"one per line. Example: [relation1, relation2, relation3]"
        )

        try:
            response = self.llm.plan_generate(
                system_prompt=PLANNER_SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=self.planner_temperature,
            )
            return self._parse_paths_from_response(response)
        except Exception as e:
            logger.warning(f"LLM rule induction failed: {e}")
            # Fall back to demonstration paths
            paths: list[list[str]] = []
            for _, rules in demonstrations:
                paths.extend(self._rules_to_paths(rules))
            return paths

    def _parse_paths_from_response(self, response: str) -> list[list[str]]:
        """Parse relation paths from LLM response text.

        Looks for bracket-enclosed comma-separated relation lists.

        Args:
            response: Raw LLM response text.

        Returns:
            List of relation paths.
        """
        paths = []
        # Match patterns like [rel1, rel2, rel3]
        pattern = r"\[([^\]]+)\]"
        for match in re.finditer(pattern, response):
            rel_str = match.group(1)
            relations = [r.strip() for r in rel_str.split(",") if r.strip()]
            if relations:
                paths.append(relations)
        return paths

    def _rules_to_paths(self, rules: list[str]) -> list[list[str]]:
        """Convert symbolic rule strings to relation paths.

        Args:
            rules: List of rule strings.

        Returns:
            List of relation paths.
        """
        paths = []
        for rule in rules:
            # Extract relations from rule body (after <-)
            parts = rule.split("<-")
            if len(parts) < 2:
                continue
            body = parts[1].strip()
            # Split on ∧
            conjuncts = body.split("∧")
            relations = []
            for conj in conjuncts:
                conj = conj.strip()
                # Extract relation name (before parenthesis)
                match = re.match(r"([^\(]+)\(", conj)
                if match:
                    relations.append(match.group(1).strip())
            if relations:
                paths.append(relations)
        return paths

    def format_rules_for_prompt(self, paths: list[list[str]]) -> str:
        """Format discovered paths as a prompt string for the executor.

        Args:
            paths: List of relation paths.

        Returns:
            Formatted string for inclusion in executor prompt.
        """
        if not paths:
            return "No reasoning paths found."

        lines = ["Surrounding relational reasoning paths are:"]
        for i, path in enumerate(paths):
            lines.append(f"  [{', '.join(path)}]")
        lines.append(
            "There are multiple paths, maybe you should select one most potential path."
        )
        return "\n".join(lines)
