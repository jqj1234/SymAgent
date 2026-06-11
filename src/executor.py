"""
Agent-Executor Module for SymAgent.

Implements the Thought-Action-Observation loop (Section 4.2) with:
- Action tool library (Section 4.2.1):
  - getReasoningPath(entity, relation) - retrieve reasoning paths from planner
  - searchNeighbor(entity, relation) - search KG neighbors
  - wikiSearch(entity, relation) - search Wikipedia for missing info
  - extractTriples(entity, relation, document) - extract triples (auto-triggered)
  - finish(entity1, entity2, ..., entityN) - return final answer
- Interactive process (Section 4.2.2):
  - React-style Thought-Action-Observation loop
  - Automatic action selection based on symbolic rules + reflection
  - Local memory for trajectory recording
  - Missing triple identification
"""

import logging
import re
import urllib.request
from typing import Any, Optional

import wikipediaapi

from .kg_environment import KGEnvironment
from .llm_client import LLMClient
from .planner import AgentPlanner

logger = logging.getLogger(__name__)

# Default few-shot example from Figure 7 of the paper (Viggo Mortensen case study)
DEFAULT_FEW_SHOT_EXAMPLE = """Question: who did viggo mortensen play in lord of the rings?
Thought 1: To answer this question, I need to find potential relational reasoning paths by analyzing similar questions in the knowledge graph. This will help me discover the relational patterns that connect the entity to the answer.
Action 1: getReasoningPath(who did viggo mortensen play in lord of the rings?)
Observation 1: Surrounding relational reasoning paths are:
[film.film.starring, film.performance.character]
[film.film.music, film.film_song.film]
[fictional_universe.work_of_fiction.part_of_these_fictional_universes, fictional_universe.fictional_universe.works_set_here]
[film.actor.film, film.performance.character]
There are multiple paths, maybe you should select one most potential path.
Thought 2: From the reasoning paths, the most potential one to answer the question is [film.film.starring, film.performance.character] since it directly relates to the actor playing a character in which he starred in. I will search the neighbors of the entity Viggo Mortensen with the relation path.
Action 2: searchNeighbor(Viggo Mortensen, film.film.starring).
Observation 2: No entity found under this relation in the knowledge graph. You can keep searching on graph or go to invoke wikiSearch to retrieve relevant documents. You can also just answer if you think you can based on the information available and your knowledge.
Thought 3: Since there are no results under the relation 'film.film.starring' in the KG, I will search for the entity Viggo Mortensen in Wikipedia to find out the role he played in the Lord of the Rings series.
Action 3: wikiSearch(Viggo Mortensen, film.film.starring).
Observation 3: By searching, Viggo Mortensen's relevant documents are Known for; The Lord of the Rings: The Return of the King. 9.0. Aragorn; Captain Fantastic. 7.8. Ben; The Lord of the Rings: The Two Towers. 8.8. Aragorn. He went on to play supporting roles in several notable films during the 1990s. You should extract relevant factual triples about Viggo Mortensen to answer the question.
Thought 4: Based on the information from Wikipedia, I can get relevant information in the triples of (Viggo Mortensen, film.film.starring, The Lord of the Rings: The Return of the King), (Viggo Mortensen, film.film.starring, The Lord of the Rings: The Two Towers), and (Viggo Mortensen, film.film.starring, Captain Fantastic). According to the collected information, the next step is to get the role he played in the Lord of the Rings series according to the reasoning path film.performance.character to get the role.
Action 4: searchNeighbor(The Lord of the Rings: The Return of the King, film.performance.character).
Observation 4: m.019tyn.
Thought 5: According to the reasoning path, I have found the character Viggo Mortensen played in the Lord of the Rings series. The searchNeighbor returned the MID m.019tyn, so I will pass this MID directly to finish.
Action 5: finish(m.019tyn)."""

# Prompt template from Figure 5 of the paper
EXECUTOR_PROMPT_TEMPLATE = """You are a knowledge graph (KG) question-answering agent that interacts with a KG storing factual knowledge. When a user asks a question, solve it using interleaving Thought, Action, and Observation steps. Follow this strict format: "Thought: your thoughts.\\nAction: your next action."
Available actions:
1. getReasoningPath(question): Retrieve relational reasoning paths by finding similar questions and inducing symbolic rules from the knowledge graph. Use this first as a high-level planning step to discover potential relational patterns for answering the question.
2. searchNeighbor(entity, relation): Search the neighbors of the entity with the specified relation in the KG.
3. wikiSearch(entity, relation): Search Wikipedia for the entity with respect to the relation if the KG returns no relevant results. Extract relevant triples (entity, relation, entity) from the Wikipedia page.
4. finish(entity1, entity2, ..., entityN): Conclude the conversation with the final answer(s). Pass entities exactly as they appear in the observation — if searchNeighbor returned MIDs (e.g. m.019tyn), use MIDs; if wikiSearch returned names, use names. Do NOT convert between formats.
Steps to follow:
Start with getReasoningPath to obtain potential relational reasoning paths by analyzing similar questions in the knowledge graph. Your first thought should focus on finding potential reasoning relations from similar problems.
Follow the most plausible path step-by-step using searchNeighbor for each relation in the path (e.g., for path r1 -> r2, first use searchNeighbor(e1, r1), then use searchNeighbor(e2, r2)).
If searchNeighbor returns no valid information, use wikiSearch and extract relevant triples. You can also use searchWikidata to query Wikidata for additional structured data.
Continue following the relational reasoning path until enough information is gathered to answer the question.
Use finish to provide the final answer(s). IMPORTANT: pass the exact entities returned by searchNeighbor or wikiSearch — preserve MIDs if searchNeighbor returned MIDs, preserve names if wikiSearch returned names. Never convert between formats.
Use the following response format:
Thought: <your thoughts>
Action: <your next action>
Here are some examples:
{examples}
(END OF EXAMPLES)

Question: {question}
{history}"""


class Trajectory:
    """Records a single agent trajectory for self-learning.

    A trajectory H_n = (q, G, p, tau_0, a_0, o_0, ..., tau_{n-1}, a_{n-1}, o_{n-1})
    as defined in Equation 4.
    """

    def __init__(self, question: str):
        self.question = question
        self.steps: list[dict[str, str]] = []
        self.answer_entities: list[str] = []
        self.ground_truth_entities: list[str] = []
        self.reward: float = 0.0
        self.planned_paths: list[list[str]] = []

    def add_step(self, thought: str, action: str, observation: str) -> None:
        """Add a thought-action-observation step."""
        self.steps.append({
            "thought": thought,
            "action": action,
            "observation": observation,
        })

    def set_answer(self, entities: list[str]) -> None:
        """Set the final answer entities."""
        self.answer_entities = entities

    def set_reward(self, reward: float) -> None:
        """Set the outcome reward."""
        self.reward = reward

    def to_text(self) -> str:
        """Serialize trajectory to text for fine-tuning."""
        lines = [f"Question: {self.question}"]
        for i, step in enumerate(self.steps):
            lines.append(f"Thought {i+1}: {step['thought']}")
            lines.append(f"Action {i+1}: {step['action']}")
            lines.append(f"Observation {i+1}: {step['observation']}")
        if self.answer_entities:
            lines.append(f"Final Answer: {', '.join(self.answer_entities)}")
        return "\n".join(lines)

    def to_prompt_text(self) -> str:
        """Format trajectory as prompt continuation for fine-tuning.

        Returns only the generated parts (thoughts and actions),
        not the observations (which come from the environment).
        """
        parts = []
        for i, step in enumerate(self.steps):
            parts.append(f"Thought {i+1}: {step['thought']}")
            parts.append(f"Action {i+1}: {step['action']}")
            parts.append(f"Observation {i+1}: {step['observation']}")
        return "\n".join(parts)

    def __len__(self) -> int:
        return len(self.steps)


class ActionParser:
    """Parse action strings from LLM output into structured calls."""

    @staticmethod
    def parse(action_str: str) -> tuple[str, list[str]]:
        """Parse an action string into (action_name, arguments).

        Supported formats:
        - getReasoningPath(entity, relation)
        - searchNeighbor(entity, relation)
        - wikiSearch(entity, relation)
        - finish(entity1, entity2, ...)

        Args:
            action_str: Raw action string from LLM.

        Returns:
            Tuple of (action_name, list_of_argument_strings).
        """
        action_str = action_str.strip()
        # Remove "Action: " prefix if present
        if action_str.startswith("Action:"):
            action_str = action_str[len("Action:"):].strip()

        # Match action_name(args)
        match = re.match(r"(\w+)\((.*)\)", action_str, re.DOTALL)
        if not match:
            return "", []

        action_name = match.group(1)
        args_str = match.group(2).strip()

        # Parse arguments - split by comma but respect nested structures
        args = ActionParser._split_args(args_str)

        return action_name, args

    @staticmethod
    def _split_args(args_str: str) -> list[str]:
        """Split argument string by commas, respecting quotes and parens."""
        args = []
        current = []
        depth = 0
        in_quotes = False
        quote_char = None

        for ch in args_str:
            if ch in ('"', "'") and not in_quotes:
                in_quotes = True
                quote_char = ch
                current.append(ch)
            elif ch == quote_char and in_quotes:
                in_quotes = False
                quote_char = None
                current.append(ch)
            elif ch == '(' and not in_quotes:
                depth += 1
                current.append(ch)
            elif ch == ')' and not in_quotes:
                depth -= 1
                current.append(ch)
            elif ch == ',' and depth == 0 and not in_quotes:
                args.append("".join(current).strip().strip('"').strip("'"))
                current = []
            else:
                current.append(ch)

        if current:
            args.append("".join(current).strip().strip('"').strip("'"))

        return [a for a in args if a]


class AgentExecutor:
    """Agent-Executor: Interactive reasoning with KG environment.

    As described in Section 4.2, the executor engages in a Thought-Action-Observation
    loop to navigate the autonomous reasoning process. It leverages symbolic rules from
    the planner and environment feedback to dynamically adjust the reasoning process.

    The interactive trajectory at step n (Equation 4):
    H_n = (q, G, p, tau_0, a_0, o_0, ..., tau_{n-1}, a_{n-1}, o_{n-1})

    Attributes:
        kg: The KG environment.
        llm: The LLM client.
        planner: The Agent-Planner for symbolic rule generation.
        max_steps: Maximum number of interaction steps.
        reasoning_max_depth: Maximum depth for getReasoningPath BFS.
        reasoning_max_paths: Maximum paths returned by getReasoningPath.
    """

    def __init__(
        self,
        kg: KGEnvironment,
        llm: LLMClient,
        planner: AgentPlanner,
        max_steps: int = 10,
        reasoning_max_depth: int = 4,
        reasoning_max_paths: int = 10,
        wiki_max_summary_length: int = 2000,
    ):
        self.kg = kg
        self.llm = llm
        self.planner = planner
        self.max_steps = max_steps
        self.reasoning_max_depth = reasoning_max_depth
        self.reasoning_max_paths = reasoning_max_paths
        self.wiki_max_summary_length = wiki_max_summary_length

        # Auto-detect system proxy (Windows registry / macOS / Linux env vars)，比如到能直连 Wikipedia 的网络环境），可以把 executor.py:258-262 注释掉
        system_proxies = urllib.request.getproxies()
        self._proxy_url = system_proxies.get("https") or system_proxies.get("http")
        if self._proxy_url:
            logger.info(f"Detected system proxy: {self._proxy_url}")

        self.wikipedia = wikipediaapi.Wikipedia(
            user_agent="SymAgent/1.0 (https://github.com/symagent)",
            language="en",
            timeout=20.0,
            proxy=self._proxy_url,
        )
        self._extracted_triples: list[tuple[str, str, str]] = []
        self._few_shot_examples: str = DEFAULT_FEW_SHOT_EXAMPLE

    def set_few_shot_examples(self, examples: str) -> None:
        """Set few-shot examples for the executor prompt.

        Args:
            examples: Formatted example trajectories.
        """
        self._few_shot_examples = examples

    def execute(
        self,
        question: str,
        question_entity: Optional[str] = None,
        planned_paths: Optional[list[list[str]]] = None,
    ) -> Trajectory:
        """Execute the reasoning loop for a given question.

        Implements the interactive process described in Section 4.2.2.

        Args:
            question: The input question.
            question_entity: The linked entity from the question.
            planned_paths: Pre-computed symbolic rules from the planner.

        Returns:
            Trajectory recording the full reasoning process.
        """
        trajectory = Trajectory(question)

        # If question_entity is not provided, use BM25 entity linking
        if question_entity is None:
            bm25_results = self.kg.bm25_retrieve_entities(question, top_k=1)
            if bm25_results:
                question_entity = bm25_results[0][0]
                logger.info(f"BM25 entity linking: '{question_entity}' for question")

        # If no planned paths provided, run planner
        if planned_paths is None and question_entity:
            planned_paths = self.planner.plan(question, question_entity)
        trajectory.planned_paths = planned_paths or []

        # Build initial prompt
        history = ""
        finished = False

        for step in range(self.max_steps):
            if finished:
                break

            # Build the current prompt
            prompt = self._build_prompt(question, history)
            # Generate thought and action
            response = self.llm.execute_generate(
                prompt,
                stop=["Observation:"],
            )

            # Parse thought and action
            thought, action_str = self._parse_thought_action(response)
            if not action_str:
                # If parsing fails, treat as thought-only and generate action
                thought = response.strip()
                action_prompt = prompt + f"\nThought: {thought}\nAction:"
                action_response = self.llm.execute_generate(
                    action_prompt,
                )
                action_str = action_response.strip()

            # Parse action
            action_name, args = ActionParser.parse(action_str)

            # Execute action and get observation
            observation, finished = self._execute_action(
                action_name, args, question, question_entity, planned_paths
            )

            # Record step
            trajectory.add_step(thought, action_str, observation)

            # Set answer entities when finish action is executed
            if finished and action_name == "finish" and args:
                trajectory.set_answer(args)

            # Update history for next step
            history += f"\nThought {step+1}: {thought}"
            history += f"\nAction {step+1}: {action_str}"
            history += f"\nObservation {step+1}: {observation}"

        if not finished and trajectory.answer_entities:
            pass  # Already has answer from finish action
        elif not finished:
            # Max steps reached without finish
            logger.warning(
                f"Max steps ({self.max_steps}) reached for: {question}"
            )

        return trajectory

    def _build_prompt(self, question: str, history: str) -> str:
        """Build the executor prompt from template.

        Args:
            question: The input question.
            history: Accumulated interaction history.

        Returns:
            Complete prompt string.
        """
        return EXECUTOR_PROMPT_TEMPLATE.format(
            examples=self._few_shot_examples,
            question=question,
            history=history,
        )

    def _parse_thought_action(
        self, response: str
    ) -> tuple[str, str]:
        """Parse LLM response into thought and action components.

        Args:
            response: Raw LLM response.

        Returns:
            Tuple of (thought_text, action_text).
        """
        thought = ""
        action = ""

        # Try to extract Thought and Action sections
        # Handle multiple formats: "Thought N: ...", "Thought: ...", plain text
        thought_match = re.search(
            r"Thought\s*(?:\d+)?:\s*(.+?)(?=\n\s*Action\s*(?:\d+)?\s*:|\Z)",
            response,
            re.DOTALL,
        )
        action_match = re.search(
            r"Action\s*(?:\d+)?\s*:\s*(.+?)$", response, re.DOTALL
        )

        if thought_match:
            thought = thought_match.group(1).strip()
        if action_match:
            action = action_match.group(1).strip()

        # Remove any leading "Thought:" prefix from action if present
        if action and action.startswith("Thought"):
            thought_match2 = re.match(r"Thought\s*\d*\s*:\s*(.+)", action)
            if thought_match2:
                thought += " " + thought_match2.group(1).strip()
                action = ""

        # Fallback: if no clear separation, split at first action keyword
        if not action:
            for action_keyword in [
                "getReasoningPath",
                "searchNeighbor",
                "wikiSearch",
                "finish",
            ]:
                idx = response.find(action_keyword)
                if idx >= 0:
                    if not thought:
                        thought = response[:idx].strip()
                    action = response[idx:].strip()
                    break

        return thought, action

    def _execute_action(
        self,
        action_name: str,
        args: list[str],
        question: str,
        question_entity: Optional[str],
        planned_paths: Optional[list[list[str]]],
    ) -> tuple[str, bool]:
        """Execute an action and return the observation.

        Implements the action space defined in Section 4.2.1.

        Args:
            action_name: Name of the action to execute.
            args: Action arguments.
            question: The original question.
            question_entity: The question entity.
            planned_paths: Pre-computed symbolic rules.

        Returns:
            Tuple of (observation_string, is_finished).
        """
        if action_name == "getReasoningPath":
            return self._action_get_reasoning_path(
                args, question, question_entity, planned_paths
            )
        elif action_name == "searchNeighbor":
            return self._action_search_neighbor(args)
        elif action_name == "wikiSearch":
            return self._action_wiki_search(args, question)
        elif action_name == "finish":
            return self._action_finish(args)
        else:
            return (
                f"Invalid action: {action_name}. "
                f"Available actions: getReasoningPath, searchNeighbor, "
                f"wikiSearch, finish.",
                False,
            )

    def _resolve_entity_for_kg(self, entity: str) -> str:
        """Convert human-readable entity name to KG entity ID (MID) if needed.

        The KG stores entities as Freebase MIDs (e.g. m.0cr3d), but the LLM
        often emits human-readable names (e.g. 'Ina Garten') in action args.
        This method attempts to map names back to MIDs so that KG lookups
        (getReasoningPath, searchNeighbor) work correctly.

        Args:
            entity: Entity string from LLM action argument.

        Returns:
            Resolved entity identifier suitable for KG lookup.
        """
        # Already a direct KG entity (MID or exact match in adjacency list)
        if entity in self.kg._adj_out:
            return entity

        # Try name-to-MID mapping (loaded via kg.load_name_mapping)
        name2mid = getattr(self.kg, "_name2mid", None)
        if name2mid:
            ne = entity.lower().strip().replace("_", " ")
            if ne in name2mid:
                return name2mid[ne]

        # Fallback: entity2id keys might be names
        if hasattr(self.kg, "entity2id") and entity in self.kg.entity2id:
            return entity

        return entity

    def _action_get_reasoning_path(
        self,
        args: list[str],
        question: str,
        question_entity: Optional[str],
        planned_paths: Optional[list[list[str]]],
    ) -> tuple[str, bool]:
        """Execute getReasoningPath action.

        As per Section 4.1 of the paper: uses the Planner to find similar
        questions via BM25, sample closed paths via BFS, generalize to
        symbolic rules, and use LLM for rule induction.

        Args:
            args: Action arguments (question string, or empty to use original question).
            question: The original question.
            question_entity: The linked entity from the question.
            planned_paths: Pre-computed symbolic rules from the planner.

        Returns:
            Tuple of (observation_string, is_finished).
        """
        # Use the question from args if provided, otherwise use original question
        target_question = args[0] if args else question

        # Priority 1: Use pre-computed planned paths (from execute() call)
        if planned_paths:
            formatted = self.planner.format_rules_for_prompt(planned_paths)
            return formatted, False

        # Priority 2: Call Planner for full rule induction pipeline
        # Planner internally does: BM25 seed retrieval → BFS path sampling →
        # rule generalization → LLM rule induction
        paths = self.planner.plan(target_question, question_entity=question_entity)

        # Priority 3: Fallback to direct KG BFS if Planner returns nothing
        if not paths and question_entity:
            logger.info(
                f"Planner returned no paths, falling back to direct KG BFS "
                f"for entity: {question_entity}"
            )
            paths = self.kg.get_reasoning_paths(
                question_entity,
                max_depth=self.reasoning_max_depth,
                max_paths=self.reasoning_max_paths,
            )

        if not paths:
            return "No reasoning paths found.", False

        formatted = self.planner.format_rules_for_prompt(paths)
        return formatted, False

    def _action_search_neighbor(
        self, args: list[str]
    ) -> tuple[str, bool]:
        """Execute searchNeighbor(entity, relation) action.

        Returns neighbors of entity under the given relation in the KG.
        """
        if len(args) < 2:
            return "Error: searchNeighbor requires entity and relation arguments.", False

        entity = self._resolve_entity_for_kg(args[0])
        relation = args[1]

        neighbors = self.kg.search_neighbor_with_relation(entity, relation)

        if neighbors:
            obs = ", ".join(neighbors)
            return obs, False
        else:
            return (
                f"No entity found under this relation in the knowledge graph. "
                f"You can keep searching on graph or go to invoke wikiSearch "
                f"to retrieve relevant documents.",
                False,
            )

    def _action_wiki_search(
        self, args: list[str], question: str
    ) -> tuple[str, bool]:
        """Execute wikiSearch(entity, relation) action.

        Retrieves relevant documents from Wikipedia when KG information
        is insufficient. Automatically triggers extractTriples.
        """
        if len(args) < 2:
            return "Error: wikiSearch requires entity and relation arguments.", False

        entity = args[0]
        relation = args[1]

        # Search Wikipedia
        try:
            page = self.wikipedia.page(entity)
            if page.exists():
                summary = page.summary[:self.wiki_max_summary_length]
                # Auto-trigger extractTriples
                extracted = self.llm.extract_triples(
                    entity=entity,
                    relation=relation,
                    document=summary,
                    question=question,
                )
                # Add extracted triples to KG
                self._integrate_extracted_triples(extracted)
                observation = (
                    f"By searching, {entity}'s relevant documents are "
                    f"{summary}\n\nExtracted triples: {extracted}"
                )
                return observation, False
            else:
                return (
                    f"No Wikipedia page found for entity: {entity}. "
                    f"Try searching with a different entity name.",
                    False,
                )
        except Exception as e:
            logger.warning(f"Wikipedia search failed for {entity}: {e}")
            return f"Error searching Wikipedia: {e}", False

    def _action_finish(self, args: list[str]) -> tuple[str, bool]:
        """Execute finish(entity1, entity2, ...) action.

        Returns the final answer and signals completion.
        """
        if not args:
            return "Error: finish requires at least one entity argument.", False

        return f"Final answer: {', '.join(args)}", True

    def _integrate_extracted_triples(self, extraction_text: str) -> None:
        """Parse extracted triples and add them to the KG.

        Parses the LLM output for triples in format:
        [[entity, relation, object], ...]

        Tries JSON parsing first (most reliable), then falls back to
        regex matching for non-strict LLM outputs.

        Args:
            extraction_text: Raw LLM output with extracted triples.
        """
        try:
            # Strategy 1: Find outermost list-of-lists via JSON
            import ast
            bracket_start = extraction_text.find("[[")
            if bracket_start >= 0:
                # Find matching closing brackets
                depth = 0
                bracket_end = bracket_start
                for i in range(bracket_start, len(extraction_text)):
                    if extraction_text[i] == "[":
                        depth += 1
                    elif extraction_text[i] == "]":
                        depth -= 1
                        if depth == 0:
                            bracket_end = i + 1
                            break
                json_str = extraction_text[bracket_start:bracket_end]
                try:
                    triples = ast.literal_eval(json_str)
                    if isinstance(triples, list):
                        for triple in triples:
                            if isinstance(triple, (list, tuple)) and len(triple) >= 3:
                                h, r, t = str(triple[0]).strip(), str(triple[1]).strip(), str(triple[2]).strip()
                                if h and r and t:
                                    self.kg.add_triple(h, r, t)
                                    self._extracted_triples.append((h, r, t))
                        return
                except (ValueError, SyntaxError):
                    pass

            # Strategy 2: Regex fallback for individual [e, r, o] patterns
            pattern = r"\[([^\[\]]+?),\s*([^\[\]]+?),\s*([^\[\]]+?)\]"
            for match in re.finditer(pattern, extraction_text):
                h = match.group(1).strip().strip("'\"")
                r = match.group(2).strip().strip("'\"")
                t = match.group(3).strip().strip("'\"")
                if h and r and t:
                    self.kg.add_triple(h, r, t)
                    self._extracted_triples.append((h, r, t))
        except Exception as e:
            logger.warning(f"Failed to integrate extracted triples: {e}")

    def get_extracted_triples(self) -> list[tuple[str, str, str]]:
        """Get all triples extracted during the current session.

        These are triples identified as missing from the KG during
        the reasoning process (addressing RQ4 in the paper).
        """
        return list(self._extracted_triples)

    def reset(self) -> None:
        """Reset executor state for a new question."""
        self._extracted_triples = []


def compute_outcome_reward(
    predicted_entities: list[str],
    ground_truth_entities: list[str],
    kg: Optional[Any] = None,
) -> float:
    """Compute outcome-based reward using recall (Equation 6).

    r(mu_i) = Recall(A_{mu_i}, A_{gt}) = |A_{mu_i} ∩ A_{gt}| / |A_{gt}|

    Supports matching in both directions:
    - Direct string match (case-insensitive)
    - Name-to-MID and MID-to-name mapping (if kg with name mappings provided)

    Args:
        predicted_entities: Entities from the trajectory's final action.
        ground_truth_entities: Ground truth answer entities.
        kg: Optional KGEnvironment with _name2mid/_mid2name attributes.

    Returns:
        Recall value in [0, 1].
    """
    if not ground_truth_entities:
        return 0.0

    name2mid = getattr(kg, '_name2mid', None) if kg is not None else None
    mid2name = getattr(kg, '_mid2name', None) if kg is not None else None

    def normalize_entity(e: str) -> str:
        return e.lower().strip().replace("_", " ")

    def resolve_entity(e: str) -> set[str]:
        """Return all possible normalized forms of an entity."""
        forms = {normalize_entity(e)}
        ne = normalize_entity(e)
        if name2mid and ne in name2mid:
            forms.add(normalize_entity(name2mid[ne]))
        if mid2name:
            for mid, name in mid2name.items():
                if normalize_entity(name) == ne:
                    forms.add(normalize_entity(mid))
                elif normalize_entity(mid) == ne:
                    forms.add(normalize_entity(name))
        return forms

    gt_resolved: list[set[str]] = [resolve_entity(e) for e in ground_truth_entities]
    pred_resolved: list[set[str]] = [resolve_entity(e) for e in predicted_entities]

    matched_gt = set()
    matched_pred = set()
    for gi, g_forms in enumerate(gt_resolved):
        for pi, p_forms in enumerate(pred_resolved):
            if gi not in matched_gt and pi not in matched_pred:
                if g_forms & p_forms:
                    matched_gt.add(gi)
                    matched_pred.add(pi)
                    break

    return len(matched_gt) / len(ground_truth_entities)
