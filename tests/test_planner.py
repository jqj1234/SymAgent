"""Tests for Agent-Planner module.

Covers:
  - BM25 seed question retrieval (Section 4.1, Step 1)
  - BFS closed path sampling (Section 4.1, Step 2)
  - Symbolic rule generalization (Section 4.1, Step 3 - Eq. 1)
  - Few-shot demonstration building (Section 4.1, Step 4)
  - LLM rule induction with mock (Section 4.1, Eq. 3)
  - LLM response path parsing
  - Rule-to-path conversion
  - Prompt formatting
  - Full plan() pipeline
"""

import unittest
from unittest.mock import MagicMock, patch

from src.kg_environment import KGEnvironment
from src.planner import AgentPlanner


class TestAgentPlanner(unittest.TestCase):
    """Test cases for AgentPlanner."""

    def setUp(self):
        """Set up test KG and planner."""
        self.kg = KGEnvironment()
        # Create a small knowledge graph
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")
        self.kg.add_triple("Alice", "liveIn", "SanFrancisco")
        self.kg.add_triple("Bob", "workFor", "Google")
        self.kg.add_triple("Google", "locatedIn", "MountainView")
        self.kg.add_triple("Bob", "liveIn", "MountainView")
        self.kg.add_triple("Alice", "knows", "Bob")
        self.kg.add_triple("SanFrancisco", "inState", "California")
        self.kg.add_triple("MountainView", "inState", "California")

        self.planner = AgentPlanner(
            kg=self.kg,
            llm=None,
            num_seed_questions=2,
            max_bfs_depth=3,
            max_paths_per_seed=3,
        )

        self.train_data = [
            {
                "question": "Where does Alice live?",
                "question_entity": "Alice",
                "answer_entities": ["SanFrancisco"],
            },
            {
                "question": "Where does Bob live?",
                "question_entity": "Bob",
                "answer_entities": ["MountainView"],
            },
            {
                "question": "Who does Alice work for?",
                "question_entity": "Alice",
                "answer_entities": ["OpenAI"],
            },
        ]
        self.planner.build_seed_index(self.train_data)

    # --- BM25 Retrieval (Section 4.1, Step 1) ---

    def test_retrieve_seed_questions(self):
        """Test BM25 seed question retrieval."""
        seeds = self.planner.retrieve_seed_questions(
            "Where does Alice live?", top_k=2
        )
        self.assertTrue(len(seeds) <= 2)

    def test_retrieve_seed_before_build(self):
        """Test retrieval returns empty if index not built."""
        planner2 = AgentPlanner(kg=self.kg, llm=None)
        seeds = planner2.retrieve_seed_questions("Where does Alice live?")
        self.assertEqual(seeds, [])

    def test_retrieve_seed_returns_full_dicts(self):
        """Test that retrieved seeds contain original data."""
        seeds = self.planner.retrieve_seed_questions("Alice", top_k=3)
        for seed in seeds:
            self.assertIn("question", seed)
            self.assertIn("question_entity", seed)
            self.assertIn("answer_entities", seed)

    def test_retrieve_seed_top_k_respected(self):
        """Test that top_k parameter is respected."""
        seeds = self.planner.retrieve_seed_questions("Alice", top_k=1)
        self.assertTrue(len(seeds) <= 1)

    # --- BFS Path Sampling (Section 4.1, Step 2) ---

    def test_sample_closed_paths_direct(self):
        """Test BFS closed path sampling with direct relation."""
        paths = self.planner.sample_closed_paths("Alice", ["SanFrancisco"])
        self.assertTrue(len(paths) > 0)
        all_rels = [tuple(p) for p in paths]
        self.assertIn(("liveIn",), all_rels)

    def test_sample_closed_paths_multi_hop(self):
        """Test BFS finds multi-hop paths."""
        paths = self.planner.sample_closed_paths("Alice", ["SanFrancisco"])
        all_rels = [tuple(p) for p in paths]
        # Should find: [workFor, locatedIn]
        self.assertIn(("workFor", "locatedIn"), all_rels)

    def test_sample_closed_paths_no_connection(self):
        """Test BFS with no path between entities."""
        self.kg.add_triple("Charlie", "livesIn", "London")
        paths = self.planner.sample_closed_paths("Alice", ["London"])
        self.assertEqual(len(paths), 0)

    def test_sample_closed_paths_max_paths(self):
        """Test BFS respects max_paths parameter."""
        paths = self.planner.sample_closed_paths("Alice", ["SanFrancisco"], max_paths=1)
        self.assertTrue(len(paths) <= 1)

    def test_sample_closed_paths_multiple_answers(self):
        """Test sampling paths to multiple answer entities."""
        paths = self.planner.sample_closed_paths("Alice", ["OpenAI", "SanFrancisco"])
        self.assertTrue(len(paths) > 0)
        # Should include paths to both targets
        flat_rels = [r for path in paths for r in path]
        self.assertIn("workFor", flat_rels)
        self.assertIn("liveIn", flat_rels)

    def test_sample_closed_paths_empty_answers(self):
        """Test sampling with empty answer list."""
        paths = self.planner.sample_closed_paths("Alice", [])
        self.assertEqual(len(paths), 0)

    # --- Rule Generalization (Section 4.1, Step 3, Eq. 1) ---

    def test_generalize_to_rules_two_hop(self):
        """Test generalizing a 2-hop path to FOL rule (Eq. 1)."""
        paths = [["workFor", "locatedIn"]]
        rules = self.planner.generalize_to_rules(paths)
        self.assertEqual(len(rules), 1)
        self.assertIn("workFor(x, z1)", rules[0])
        self.assertIn("locatedIn(z1, y)", rules[0])
        self.assertIn("<-", rules[0])
        # Rule head should be r(x, y)
        self.assertTrue(rules[0].startswith("r(x, y)"))

    def test_generalize_to_rules_one_hop(self):
        """Test generalizing a 1-hop path."""
        paths = [["liveIn"]]
        rules = self.planner.generalize_to_rules(paths)
        self.assertEqual(len(rules), 1)
        self.assertIn("liveIn(x, y)", rules[0])

    def test_generalize_to_rules_three_hop(self):
        """Test generalizing a 3-hop path with z1, z2 variables."""
        paths = [["knows", "workFor", "locatedIn"]]
        rules = self.planner.generalize_to_rules(paths)
        self.assertEqual(len(rules), 1)
        self.assertIn("knows(x, z1)", rules[0])
        self.assertIn("workFor(z1, z2)", rules[0])
        self.assertIn("locatedIn(z2, y)", rules[0])

    def test_generalize_to_rules_empty_path(self):
        """Test generalizing an empty path."""
        rules = self.planner.generalize_to_rules([[]])
        self.assertEqual(len(rules), 0)

    def test_generalize_to_rules_multiple(self):
        """Test generalizing multiple paths at once."""
        paths = [["workFor", "locatedIn"], ["liveIn"]]
        rules = self.planner.generalize_to_rules(paths)
        self.assertEqual(len(rules), 2)

    def test_generalize_to_rules_four_hop(self):
        """Test generalizing a 4-hop path with z1, z2, z3 variables."""
        paths = [["r1", "r2", "r3", "r4"]]
        rules = self.planner.generalize_to_rules(paths)
        self.assertEqual(len(rules), 1)
        self.assertIn("r1(x, z1)", rules[0])
        self.assertIn("r2(z1, z2)", rules[0])
        self.assertIn("r3(z2, z3)", rules[0])
        self.assertIn("r4(z3, y)", rules[0])

    # --- Demonstrations (Section 4.1, Step 4) ---

    def test_build_demonstrations(self):
        """Test building few-shot demonstrations M = {(q_seed_i, P_i)}."""
        seeds = [
            {
                "question": "Where does Alice live?",
                "question_entity": "Alice",
                "answer_entities": ["SanFrancisco"],
            }
        ]
        demos = self.planner.build_demonstrations(seeds)
        self.assertTrue(len(demos) > 0)
        q, rules = demos[0]
        self.assertEqual(q, "Where does Alice live?")
        self.assertTrue(len(rules) > 0)

    def test_build_demonstrations_skips_incomplete(self):
        """Test that seeds without entity/answer are skipped."""
        seeds = [
            {"question": "What is X?", "question_entity": "", "answer_entities": []},
            {"question": "Where does Alice live?"},
        ]
        demos = self.planner.build_demonstrations(seeds)
        self.assertEqual(len(demos), 0)

    def test_build_demonstrations_no_paths(self):
        """Test that seeds with no KG paths are skipped."""
        seeds = [
            {"question": "Q", "question_entity": "UnknownEntity", "answer_entities": ["UnknownAnswer"]}
        ]
        demos = self.planner.build_demonstrations(seeds)
        self.assertEqual(len(demos), 0)

    # --- Prompt Formatting ---

    def test_format_rules_for_prompt(self):
        """Test formatting rules as prompt text."""
        paths = [["workFor", "locatedIn"], ["liveIn"]]
        formatted = self.planner.format_rules_for_prompt(paths)
        self.assertIn("workFor", formatted)
        self.assertIn("locatedIn", formatted)
        self.assertIn("liveIn", formatted)
        self.assertIn("most potential path", formatted)

    def test_format_rules_empty(self):
        """Test formatting empty rules."""
        formatted = self.planner.format_rules_for_prompt([])
        self.assertEqual(formatted, "No reasoning paths found.")

    # --- Parsing Utilities ---

    def test_rules_to_paths(self):
        """Test converting rule strings back to relation paths."""
        rules = [
            "r(x, y) <- workFor(x, z1) ∧ locatedIn(z1, y)",
            "r(x, y) <- liveIn(x, y)",
        ]
        paths = self.planner._rules_to_paths(rules)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], ["workFor", "locatedIn"])
        self.assertEqual(paths[1], ["liveIn"])

    def test_rules_to_paths_invalid(self):
        """Test rules_to_paths with malformed rules."""
        rules = ["not a valid rule", ""]
        paths = self.planner._rules_to_paths(rules)
        self.assertEqual(len(paths), 0)

    def test_parse_paths_from_response(self):
        """Test parsing relation paths from LLM response text."""
        response = (
            "Here are the potential reasoning paths:\n"
            "[workFor, locatedIn]\n"
            "[liveIn]\n"
            "[knows, workFor, locatedIn]\n"
        )
        paths = self.planner._parse_paths_from_response(response)
        self.assertEqual(len(paths), 3)
        self.assertEqual(paths[0], ["workFor", "locatedIn"])
        self.assertEqual(paths[1], ["liveIn"])
        self.assertEqual(paths[2], ["knows", "workFor", "locatedIn"])

    def test_parse_paths_from_response_empty(self):
        """Test parsing empty response."""
        paths = self.planner._parse_paths_from_response("")
        self.assertEqual(paths, [])

    def test_parse_paths_from_response_no_brackets(self):
        """Test parsing response without bracket format."""
        paths = self.planner._parse_paths_from_response("no paths here")
        self.assertEqual(paths, [])

    def test_parse_paths_from_response_extra_whitespace(self):
        """Test parsing response with extra whitespace in brackets."""
        response = "[  workFor  ,  locatedIn  ]\n[  liveIn ]"
        paths = self.planner._parse_paths_from_response(response)
        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], ["workFor", "locatedIn"])
        self.assertEqual(paths[1], ["liveIn"])

    # --- LLM Rule Induction (Eq. 3) ---

    def test_llm_induce_rules_with_mock(self):
        """Test LLM rule induction with mocked LLM (Eq. 3)."""
        mock_llm = MagicMock()
        mock_llm.plan_generate.return_value = "[workFor, locatedIn]\n[liveIn]"

        planner = AgentPlanner(kg=self.kg, llm=mock_llm)
        demonstrations = [
            ("Where does Alice work?", ["r(x, y) <- workFor(x, z1) ∧ locatedIn(z1, y)"]),
        ]
        paths = planner._llm_induce_rules("Where does Bob work?", demonstrations)

        self.assertEqual(len(paths), 2)
        self.assertEqual(paths[0], ["workFor", "locatedIn"])
        self.assertEqual(paths[1], ["liveIn"])
        mock_llm.plan_generate.assert_called_once()

    def test_llm_induce_rules_fallback_on_error(self):
        """Test LLM rule induction falls back to demo paths on error."""
        mock_llm = MagicMock()
        mock_llm.plan_generate.side_effect = Exception("API Error")

        planner = AgentPlanner(kg=self.kg, llm=mock_llm)
        demonstrations = [
            ("Q1", ["r(x, y) <- r1(x, y)"]),
            ("Q2", ["r(x, y) <- r2(x, z1) ∧ r3(z1, y)"]),
        ]
        paths = planner._llm_induce_rules("test question", demonstrations)

        # Should fall back to demonstration paths
        self.assertEqual(len(paths), 2)
        self.assertIn(["r1"], paths)
        self.assertIn(["r2", "r3"], paths)

    def test_llm_induce_rules_empty_response(self):
        """Test LLM rule induction with empty LLM response (no brackets)."""
        mock_llm = MagicMock()
        mock_llm.plan_generate.return_value = "I don't know any paths."

        planner = AgentPlanner(kg=self.kg, llm=mock_llm)
        demonstrations = [("Q1", ["r(x, y) <- r1(x, y)"])]
        paths = planner._llm_induce_rules("test", demonstrations)

        # Empty response parsed to [] since no brackets found.
        # Fallback only triggers on exception, not on empty parse result.
        self.assertEqual(len(paths), 0)

    # --- Integration: plan() ---

    def test_plan_with_entity_no_llm(self):
        """Test plan() returns direct KG paths when no LLM/demonstrations available."""
        planner2 = AgentPlanner(kg=self.kg, llm=None)
        paths = planner2.plan("Where does Alice work?", question_entity="Alice")
        self.assertIsInstance(paths, list)
        for p in paths:
            self.assertIsInstance(p, list)
            for rel in p:
                self.assertIsInstance(rel, str)

    def test_plan_deduplicates_paths(self):
        """Test that plan() deduplicates returned paths."""
        planner2 = AgentPlanner(kg=self.kg, llm=None, max_rules=100)
        paths = planner2.plan("test", question_entity="Alice")
        path_tuples = [tuple(p) for p in paths]
        self.assertEqual(len(path_tuples), len(set(path_tuples)))

    def test_plan_respects_max_rules(self):
        """Test that plan() respects max_rules limit."""
        planner2 = AgentPlanner(kg=self.kg, llm=None, max_rules=3)
        paths = planner2.plan("test", question_entity="Alice")
        self.assertTrue(len(paths) <= 3)

    def test_plan_with_llm_and_demonstrations(self):
        """Test plan() with mocked LLM and demonstrations (full Eq. 3 pipeline)."""
        mock_llm = MagicMock()
        mock_llm.plan_generate.return_value = "[workFor, locatedIn]"

        planner = AgentPlanner(
            kg=self.kg, llm=mock_llm, num_seed_questions=2, max_rules=5
        )
        planner.build_seed_index(self.train_data)

        paths = planner.plan("Where does Bob work?", question_entity="Bob")
        self.assertIsInstance(paths, list)
        # LLM should have been called since demonstrations are available
        mock_llm.plan_generate.assert_called()


if __name__ == "__main__":
    unittest.main()
