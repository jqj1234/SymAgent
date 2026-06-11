"""Tests for Agent-Executor module.

Covers:
  - ActionParser: parsing all action types
  - Trajectory: step recording, serialization (Eq. 4)
  - Outcome reward computation (Eq. 6)
  - AgentExecutor._build_prompt
  - AgentExecutor._parse_thought_action
  - AgentExecutor._execute_action (all 5 action types)
  - AgentExecutor.reset / get_extracted_triples
  - AgentExecutor._integrate_extracted_triples
"""

import unittest
from unittest.mock import MagicMock, patch

from src.executor import (
    ActionParser,
    AgentExecutor,
    Trajectory,
    compute_outcome_reward,
)
from src.kg_environment import KGEnvironment
from src.planner import AgentPlanner


class TestActionParser(unittest.TestCase):
    """Test cases for ActionParser (Section 4.2.1 - Action Space)."""

    def test_parse_search_neighbor(self):
        """Test parsing searchNeighbor action."""
        name, args = ActionParser.parse('searchNeighbor("Alice", "workFor")')
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_get_reasoning_path(self):
        """Test parsing getReasoningPath action."""
        name, args = ActionParser.parse('getReasoningPath("Alice", "workFor")')
        self.assertEqual(name, "getReasoningPath")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_wiki_search(self):
        """Test parsing wikiSearch action."""
        name, args = ActionParser.parse('wikiSearch("Alice", "workFor")')
        self.assertEqual(name, "wikiSearch")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_finish_single(self):
        """Test parsing finish action with single entity."""
        name, args = ActionParser.parse('finish("SanFrancisco")')
        self.assertEqual(name, "finish")
        self.assertEqual(args, ["SanFrancisco"])

    def test_parse_finish_multiple(self):
        """Test parsing finish action with multiple entities."""
        name, args = ActionParser.parse('finish("SanFrancisco", "MountainView")')
        self.assertEqual(name, "finish")
        self.assertEqual(args, ["SanFrancisco", "MountainView"])

    def test_parse_finish_three_entities(self):
        """Test parsing finish action with three entities."""
        name, args = ActionParser.parse('finish("A", "B", "C")')
        self.assertEqual(name, "finish")
        self.assertEqual(args, ["A", "B", "C"])

    def test_parse_with_action_prefix(self):
        """Test parsing action with 'Action: ' prefix."""
        name, args = ActionParser.parse('Action: searchNeighbor("Alice", "workFor")')
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_empty_string(self):
        """Test parsing empty string."""
        name, args = ActionParser.parse("")
        self.assertEqual(name, "")
        self.assertEqual(args, [])

    def test_parse_invalid_format(self):
        """Test parsing invalid action format."""
        name, args = ActionParser.parse("not a valid action")
        self.assertEqual(name, "")
        self.assertEqual(args, [])

    def test_parse_no_quotes(self):
        """Test parsing action without quotes."""
        name, args = ActionParser.parse("finish(Alice, Bob)")
        self.assertEqual(name, "finish")
        self.assertEqual(args, ["Alice", "Bob"])

    def test_parse_single_quotes(self):
        """Test parsing action with single quotes."""
        name, args = ActionParser.parse("searchNeighbor('Alice', 'workFor')")
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_mixed_quotes(self):
        """Test parsing action with mixed quotes."""
        name, args = ActionParser.parse('searchNeighbor("Alice", \'workFor\')')
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_with_spaces(self):
        """Test parsing action with extra whitespace."""
        name, args = ActionParser.parse('  searchNeighbor(  "Alice" , "workFor" )  ')
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(args, ["Alice", "workFor"])

    def test_parse_nested_parens(self):
        """Test parsing action with no arguments."""
        name, args = ActionParser.parse("finish()")
        self.assertEqual(name, "finish")
        self.assertEqual(args, [])

    def test_parse_entity_with_special_chars(self):
        """Test parsing entity names with special characters."""
        name, args = ActionParser.parse('searchNeighbor("The Lord of the Rings", "film.starring")')
        self.assertEqual(name, "searchNeighbor")
        self.assertEqual(len(args), 2)

    def test_split_args_filters_empty(self):
        """Test that _split_args filters out empty strings."""
        args = ActionParser._split_args("Alice, , Bob")
        self.assertEqual(args, ["Alice", "Bob"])


class TestTrajectory(unittest.TestCase):
    """Test cases for Trajectory (Eq. 4 - H_n)."""

    def test_add_step(self):
        """Test adding steps to trajectory."""
        traj = Trajectory("Where does Alice live?")
        traj.add_step(
            thought="I need to search for Alice's location",
            action='searchNeighbor("Alice", "liveIn")',
            observation="SanFrancisco",
        )
        self.assertEqual(len(traj), 1)
        self.assertEqual(traj.steps[0]["thought"], "I need to search for Alice's location")
        self.assertEqual(traj.steps[0]["action"], 'searchNeighbor("Alice", "liveIn")')
        self.assertEqual(traj.steps[0]["observation"], "SanFrancisco")

    def test_multiple_steps(self):
        """Test adding multiple steps."""
        traj = Trajectory("Test question")
        for i in range(5):
            traj.add_step(f"thought_{i}", f"action_{i}", f"obs_{i}")
        self.assertEqual(len(traj), 5)

    def test_set_answer(self):
        """Test setting answer entities."""
        traj = Trajectory("Where does Alice live?")
        traj.set_answer(["SanFrancisco"])
        self.assertEqual(traj.answer_entities, ["SanFrancisco"])

    def test_set_answer_multiple(self):
        """Test setting multiple answer entities."""
        traj = Trajectory("Test")
        traj.set_answer(["A", "B", "C"])
        self.assertEqual(traj.answer_entities, ["A", "B", "C"])

    def test_set_reward(self):
        """Test setting reward."""
        traj = Trajectory("Where does Alice live?")
        traj.set_reward(1.0)
        self.assertEqual(traj.reward, 1.0)

    def test_set_reward_partial(self):
        """Test setting partial reward."""
        traj = Trajectory("Test")
        traj.set_reward(0.5)
        self.assertEqual(traj.reward, 0.5)

    def test_to_text(self):
        """Test trajectory serialization to text."""
        traj = Trajectory("Where does Alice live?")
        traj.add_step("thinking", "acting", "observing")
        traj.set_answer(["SanFrancisco"])
        text = traj.to_text()
        self.assertIn("Where does Alice live?", text)
        self.assertIn("thinking", text)
        self.assertIn("acting", text)
        self.assertIn("observing", text)
        self.assertIn("SanFrancisco", text)

    def test_to_text_no_answer(self):
        """Test serialization when no answer is set."""
        traj = Trajectory("Test")
        traj.add_step("thought", "action", "obs")
        text = traj.to_text()
        self.assertIn("Test", text)
        self.assertNotIn("Final Answer", text)

    def test_to_prompt_text(self):
        """Test trajectory to prompt text format."""
        traj = Trajectory("Where does Alice live?")
        traj.add_step("thought1", "action1", "obs1")
        traj.add_step("thought2", "action2", "obs2")
        text = traj.to_prompt_text()
        self.assertIn("Thought 1: thought1", text)
        self.assertIn("Action 1: action1", text)
        self.assertIn("Observation 1: obs1", text)
        self.assertIn("Thought 2: thought2", text)

    def test_initial_state(self):
        """Test initial trajectory state."""
        traj = Trajectory("Test question")
        self.assertEqual(len(traj), 0)
        self.assertEqual(traj.answer_entities, [])
        self.assertEqual(traj.reward, 0.0)
        self.assertEqual(traj.planned_paths, [])
        self.assertEqual(traj.ground_truth_entities, [])

    def test_planned_paths_stored(self):
        """Test that planned_paths are stored."""
        traj = Trajectory("Test")
        traj.planned_paths = [["r1", "r2"], ["r3"]]
        self.assertEqual(len(traj.planned_paths), 2)


class TestOutcomeReward(unittest.TestCase):
    """Test cases for outcome-based reward computation (Eq. 6).

    r(mu_i) = Recall(A_{mu_i}, A_{gt}) = |A_{mu_i} ∩ A_{gt}| / |A_{gt}|
    """

    def test_perfect_match(self):
        """Test perfect prediction."""
        reward = compute_outcome_reward(["SanFrancisco"], ["SanFrancisco"])
        self.assertEqual(reward, 1.0)

    def test_partial_match(self):
        """Test partial prediction (recall)."""
        reward = compute_outcome_reward(["SanFrancisco"], ["SanFrancisco", "MountainView"])
        self.assertEqual(reward, 0.5)

    def test_no_match(self):
        """Test no matching entities."""
        reward = compute_outcome_reward(["London"], ["SanFrancisco"])
        self.assertEqual(reward, 0.0)

    def test_empty_prediction(self):
        """Test empty prediction list."""
        reward = compute_outcome_reward([], ["SanFrancisco"])
        self.assertEqual(reward, 0.0)

    def test_empty_ground_truth(self):
        """Test empty ground truth list."""
        reward = compute_outcome_reward(["SanFrancisco"], [])
        self.assertEqual(reward, 0.0)

    def test_both_empty(self):
        """Test both lists empty."""
        reward = compute_outcome_reward([], [])
        self.assertEqual(reward, 0.0)

    def test_case_insensitive(self):
        """Test case-insensitive matching."""
        reward = compute_outcome_reward(["sanfrancisco"], ["SanFrancisco"])
        self.assertEqual(reward, 1.0)

    def test_superset_prediction(self):
        """Test prediction has all ground truth plus extras."""
        reward = compute_outcome_reward(
            ["SanFrancisco", "London", "Paris"],
            ["SanFrancisco"],
        )
        self.assertEqual(reward, 1.0)

    def test_multiple_match_partial(self):
        """Test matching some of multiple ground truth answers."""
        reward = compute_outcome_reward(["A", "B"], ["A", "B", "C"])
        self.assertAlmostEqual(reward, 2 / 3)

    def test_whitespace_handling(self):
        """Test that whitespace is handled in matching."""
        reward = compute_outcome_reward([" SanFrancisco "], ["SanFrancisco"])
        self.assertEqual(reward, 1.0)

    def test_full_recall(self):
        """Test recall = 1 when all ground truth found."""
        reward = compute_outcome_reward(
            ["A", "B", "C", "D"],
            ["A", "B", "C"],
        )
        self.assertEqual(reward, 1.0)

    def test_recall_zero_when_none_match(self):
        """Test recall = 0 when no ground truth entities found."""
        reward = compute_outcome_reward(["X", "Y"], ["A", "B"])
        self.assertEqual(reward, 0.0)


class TestAgentExecutor(unittest.TestCase):
    """Test cases for AgentExecutor (Section 4.2)."""

    def setUp(self):
        """Set up test KG, planner, and executor."""
        self.kg = KGEnvironment()
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")
        self.kg.add_triple("Alice", "liveIn", "SanFrancisco")
        self.kg.add_triple("Bob", "workFor", "Google")
        self.kg.add_triple("Google", "locatedIn", "MountainView")

        self.mock_llm = MagicMock()
        self.planner = AgentPlanner(kg=self.kg, llm=self.mock_llm, max_rules=5)
        self.executor = AgentExecutor(
            kg=self.kg, llm=self.mock_llm, planner=self.planner, max_steps=10
        )

    def _mock_llm_response(self, thought, action):
        """Helper to create a mock LLM response."""
        return f"Thought: {thought}\nAction: {action}"

    # --- _build_prompt ---

    def test_build_prompt_includes_question(self):
        """Test that prompt includes the question."""
        prompt = self.executor._build_prompt("What is X?", "")
        self.assertIn("What is X?", prompt)

    def test_build_prompt_includes_history(self):
        """Test that prompt includes interaction history."""
        history = "\nThought 1: think\nAction 1: act\nObservation 1: obs"
        prompt = self.executor._build_prompt("Q?", history)
        self.assertIn("think", prompt)
        self.assertIn("act", prompt)
        self.assertIn("obs", prompt)

    def test_build_prompt_includes_available_actions(self):
        """Test that prompt lists available actions (Section 4.2.1)."""
        prompt = self.executor._build_prompt("Q?", "")
        self.assertIn("getReasoningPath", prompt)
        self.assertIn("searchNeighbor", prompt)
        self.assertIn("wikiSearch", prompt)
        self.assertIn("finish", prompt)

    def test_build_prompt_includes_examples(self):
        """Test that prompt includes few-shot examples."""
        prompt = self.executor._build_prompt("Q?", "")
        self.assertIn("Viggo Mortensen", prompt)

    # --- _parse_thought_action ---

    def test_parse_thought_action_standard(self):
        """Test parsing standard Thought/Action format."""
        response = "Thought: I need to search\nAction: searchNeighbor(Alice, workFor)"
        thought, action = self.executor._parse_thought_action(response)
        self.assertEqual(thought, "I need to search")
        self.assertEqual(action, "searchNeighbor(Alice, workFor)")

    def test_parse_thought_action_numbered(self):
        """Test parsing numbered Thought 1: / Action 1: format."""
        response = "Thought 1: I need to search\nAction 1: searchNeighbor(Alice, workFor)"
        thought, action = self.executor._parse_thought_action(response)
        self.assertEqual(thought, "I need to search")
        self.assertEqual(action, "searchNeighbor(Alice, workFor)")

    def test_parse_thought_action_fallback(self):
        """Test fallback parsing when format is non-standard."""
        response = "searchNeighbor(Alice, workFor)"
        thought, action = self.executor._parse_thought_action(response)
        self.assertEqual(action, "searchNeighbor(Alice, workFor)")

    def test_parse_thought_action_empty(self):
        """Test parsing empty response."""
        thought, action = self.executor._parse_thought_action("")
        self.assertEqual(thought, "")
        self.assertEqual(action, "")

    # --- _execute_action ---

    def test_execute_get_reasoning_path_two_args(self):
        """Test getReasoningPath(entity, relation) action."""
        obs, finished = self.executor._execute_action(
            "getReasoningPath", ["Alice", "workFor"], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("reasoning", obs.lower())

    def test_execute_get_reasoning_path_one_arg(self):
        """Test getReasoningPath(sub_question) with entity linking."""
        obs, finished = self.executor._execute_action(
            "getReasoningPath", ["Who does Alice work for?"], "Q?", "Alice", None
        )
        self.assertFalse(finished)

    def test_execute_get_reasoning_path_no_args_with_planned_paths(self):
        """Test getReasoningPath() with pre-computed planned paths."""
        obs, finished = self.executor._execute_action(
            "getReasoningPath", [], "Q?", "Alice", [["workFor", "locatedIn"]]
        )
        self.assertFalse(finished)
        self.assertIn("workFor", obs)

    def test_execute_search_neighbor_found(self):
        """Test searchNeighbor when entity found in KG."""
        obs, finished = self.executor._execute_action(
            "searchNeighbor", ["Alice", "workFor"], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("OpenAI", obs)

    def test_execute_search_neighbor_not_found(self):
        """Test searchNeighbor when entity not found."""
        obs, finished = self.executor._execute_action(
            "searchNeighbor", ["Alice", "bornIn"], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("No entity found", obs)

    def test_execute_search_neighbor_missing_args(self):
        """Test searchNeighbor with missing arguments."""
        obs, finished = self.executor._execute_action(
            "searchNeighbor", ["Alice"], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("Error", obs)

    def test_execute_finish(self):
        """Test finish action."""
        obs, finished = self.executor._execute_action(
            "finish", ["OpenAI"], "Q?", "Alice", None
        )
        self.assertTrue(finished)
        self.assertIn("OpenAI", obs)

    def test_execute_finish_no_args(self):
        """Test finish action with no arguments."""
        obs, finished = self.executor._execute_action(
            "finish", [], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("Error", obs)

    def test_execute_finish_multiple(self):
        """Test finish action with multiple entities."""
        obs, finished = self.executor._execute_action(
            "finish", ["A", "B", "C"], "Q?", "Alice", None
        )
        self.assertTrue(finished)
        self.assertIn("A", obs)
        self.assertIn("B", obs)
        self.assertIn("C", obs)

    def test_execute_invalid_action(self):
        """Test invalid action name."""
        obs, finished = self.executor._execute_action(
            "invalidAction", [], "Q?", "Alice", None
        )
        self.assertFalse(finished)
        self.assertIn("Invalid action", obs)

    # --- reset / get_extracted_triples ---

    def test_reset_clears_extracted_triples(self):
        """Test that reset clears extracted triples."""
        self.executor._extracted_triples = [("A", "r", "B")]
        self.executor.reset()
        self.assertEqual(self.executor.get_extracted_triples(), [])

    def test_get_extracted_triples_returns_list(self):
        """Test that get_extracted_triples returns a list."""
        self.assertIsInstance(self.executor.get_extracted_triples(), list)

    # --- _integrate_extracted_triples ---

    def test_integrate_extracted_triples_json(self):
        """Test integrating triples from JSON-formatted LLM output."""
        extraction = "[[Viggo Mortensen, film.film.starring, The Lord of the Rings]]"
        self.executor._integrate_extracted_triples(extraction)
        triples = self.executor.get_extracted_triples()
        self.assertEqual(len(triples), 1)
        self.assertEqual(triples[0][0], "Viggo Mortensen")
        self.assertIn(("Viggo Mortensen", "film.film.starring", "The Lord of the Rings"), self.kg)

    def test_integrate_extracted_triples_multiple(self):
        """Test integrating multiple extracted triples."""
        extraction = "[[A, r1, B], [A, r2, C], [D, r3, E]]"
        self.executor._integrate_extracted_triples(extraction)
        triples = self.executor.get_extracted_triples()
        self.assertEqual(len(triples), 3)

    def test_integrate_extracted_triples_regex_fallback(self):
        """Test regex fallback for non-JSON LLM output."""
        extraction = "Based on the document: [A, r1, B] and [C, r2, D]"
        self.executor._integrate_extracted_triples(extraction)
        triples = self.executor.get_extracted_triples()
        self.assertEqual(len(triples), 2)

    def test_integrate_extracted_triples_empty(self):
        """Test integrating empty extraction text."""
        self.executor._integrate_extracted_triples("No triples found.")
        triples = self.executor.get_extracted_triples()
        self.assertEqual(len(triples), 0)

    # --- set_few_shot_examples ---

    def test_set_few_shot_examples(self):
        """Test setting custom few-shot examples."""
        self.executor.set_few_shot_examples("Custom example")
        prompt = self.executor._build_prompt("Q?", "")
        self.assertIn("Custom example", prompt)


class TestAgentExecutorIntegration(unittest.TestCase):
    """Integration tests for executor with mocked LLM."""

    def setUp(self):
        """Set up test KG and executor."""
        self.kg = KGEnvironment()
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")

        self.mock_llm = MagicMock()
        self.planner = AgentPlanner(kg=self.kg, llm=self.mock_llm, max_rules=5)
        self.executor = AgentExecutor(
            kg=self.kg, llm=self.mock_llm, planner=self.planner, max_steps=5
        )

    def test_execute_simple_path(self):
        """Test executor completing a simple 2-step path."""
        # Step 1: getReasoningPath
        self.mock_llm.execute_generate.return_value = (
            "Thought: I need to find the reasoning path\n"
            "Action: getReasoningPath(Alice, workFor)"
        )
        # Step 2: searchNeighbor
        self.mock_llm.execute_generate.side_effect = [
            "Thought: Now search for Alice's employer\n"
            "Action: searchNeighbor(Alice, workFor)",
            "Thought: Found OpenAI\n"
            "Action: finish(OpenAI)",
        ]

        trajectory = self.executor.execute(
            "Where does Alice work?", "Alice", [["workFor"]]
        )

        self.assertTrue(len(trajectory) > 0)
        self.assertEqual(trajectory.answer_entities, ["OpenAI"])

    def test_execute_with_planned_paths(self):
        """Test executor uses pre-computed planned paths."""
        self.mock_llm.execute_generate.return_value = (
            "Thought: Let me search\n"
            "Action: searchNeighbor(Alice, workFor)"
        )

        trajectory = self.executor.execute(
            "Where does Alice work?", "Alice", [["workFor"]]
        )

        self.assertIsInstance(trajectory.planned_paths, list)


if __name__ == "__main__":
    unittest.main()
