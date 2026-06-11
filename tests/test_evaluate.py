"""Tests for Evaluation module.

Covers:
  - Entity normalization
  - Hits@1, Hits@3, Hits@10
  - F1 score
  - Accuracy
  - Evaluator class with mocked executor
"""

import unittest
from unittest.mock import MagicMock

from src.evaluate import Evaluator, accuracy, f1_score, hits_at_k, normalize_entity
from src.executor import Trajectory
from src.kg_environment import KGEnvironment
from src.llm_client import LLMClient
from src.planner import AgentPlanner


class TestNormalizeEntity(unittest.TestCase):
    """Test entity name normalization."""

    def test_lowercase(self):
        self.assertEqual(normalize_entity("SanFrancisco"), "sanfrancisco")

    def test_strip(self):
        self.assertEqual(normalize_entity("  Alice  "), "alice")

    def test_underscore_to_space(self):
        self.assertEqual(normalize_entity("San_Francisco"), "san francisco")

    def test_hyphen_to_space(self):
        self.assertEqual(normalize_entity("New-York"), "new york")

    def test_combined_transformations(self):
        """Test multiple transformations applied together."""
        self.assertEqual(normalize_entity("  New-York_City  "), "new york city")

    def test_empty_string(self):
        self.assertEqual(normalize_entity(""), "")

    def test_special_characters(self):
        """Test normalization preserves non-transformed characters."""
        self.assertEqual(normalize_entity("m.0bxtg"), "m.0bxtg")


class TestHitsAtK(unittest.TestCase):
    """Test Hits@k metric."""

    def test_hit_at_1(self):
        predicted = ["SanFrancisco", "MountainView", "London"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=1), 1.0)

    def test_miss_at_1(self):
        predicted = ["London", "SanFrancisco"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=1), 0.0)

    def test_hit_at_3(self):
        predicted = ["London", "Paris", "SanFrancisco"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=3), 1.0)

    def test_miss_at_3(self):
        predicted = ["London", "Paris", "Berlin"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=3), 0.0)

    def test_empty_predicted(self):
        self.assertEqual(hits_at_k([], ["SanFrancisco"], k=1), 0.0)

    def test_empty_ground_truth(self):
        self.assertEqual(hits_at_k(["SanFrancisco"], [], k=1), 0.0)

    def test_both_empty(self):
        self.assertEqual(hits_at_k([], [], k=1), 0.0)

    def test_hit_at_10(self):
        """Test Hits@10 with 10 predicted entities."""
        predicted = [f"entity_{i}" for i in range(10)]
        ground_truth = ["entity_9"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=10), 1.0)

    def test_miss_at_10(self):
        """Test Hits@10 miss."""
        predicted = [f"entity_{i}" for i in range(10)]
        ground_truth = ["entity_99"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=10), 0.0)

    def test_case_insensitive_hit(self):
        """Test that normalization makes hits case-insensitive."""
        predicted = ["sanfrancisco"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(hits_at_k(predicted, ground_truth, k=1), 1.0)


class TestF1Score(unittest.TestCase):
    """Test F1 score metric."""

    def test_perfect_match(self):
        predicted = ["A", "B", "C"]
        ground_truth = ["A", "B", "C"]
        self.assertEqual(f1_score(predicted, ground_truth), 1.0)

    def test_partial_match(self):
        predicted = ["A", "B"]
        ground_truth = ["A", "B", "C"]
        f1 = f1_score(predicted, ground_truth)
        self.assertAlmostEqual(f1, 0.8, places=3)

    def test_no_match(self):
        predicted = ["X", "Y"]
        ground_truth = ["A", "B"]
        self.assertEqual(f1_score(predicted, ground_truth), 0.0)

    def test_superset_prediction(self):
        predicted = ["A", "B", "C", "D"]
        ground_truth = ["A", "B"]
        f1 = f1_score(predicted, ground_truth)
        self.assertAlmostEqual(f1, 2 / 3, places=3)

    def test_both_empty(self):
        self.assertEqual(f1_score([], []), 1.0)

    def test_one_empty(self):
        self.assertEqual(f1_score(["A"], []), 0.0)
        self.assertEqual(f1_score([], ["A"]), 0.0)

    def test_case_insensitive(self):
        predicted = ["sanfrancisco"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(f1_score(predicted, ground_truth), 1.0)

    def test_precision_zero(self):
        """Test F1 when precision is 0."""
        predicted = ["X", "Y"]
        ground_truth = ["A", "B", "C"]
        self.assertEqual(f1_score(predicted, ground_truth), 0.0)

    def test_recall_zero(self):
        """Test F1 when recall is 0."""
        predicted = ["X"]
        ground_truth = ["A", "B"]
        self.assertEqual(f1_score(predicted, ground_truth), 0.0)


class TestAccuracy(unittest.TestCase):
    """Test accuracy metric."""

    def test_match(self):
        predicted = ["SanFrancisco", "London"]
        ground_truth = ["SanFrancisco", "Paris"]
        self.assertEqual(accuracy(predicted, ground_truth), 1.0)

    def test_no_match(self):
        predicted = ["London", "Paris"]
        ground_truth = ["SanFrancisco", "Berlin"]
        self.assertEqual(accuracy(predicted, ground_truth), 0.0)

    def test_empty(self):
        self.assertEqual(accuracy([], ["A"]), 0.0)
        self.assertEqual(accuracy(["A"], []), 0.0)

    def test_case_insensitive(self):
        predicted = ["sanfrancisco"]
        ground_truth = ["SanFrancisco"]
        self.assertEqual(accuracy(predicted, ground_truth), 1.0)


class TestEvaluator(unittest.TestCase):
    """Test Evaluator class with mocked components."""

    def setUp(self):
        """Set up mocked evaluator."""
        self.mock_kg = MagicMock()
        self.mock_llm = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_executor = MagicMock()

        self.evaluator = Evaluator(
            kg=self.mock_kg,
            llm=self.mock_llm,
            planner=self.mock_planner,
            executor=self.mock_executor,
            metrics=["hits@1", "f1", "accuracy"],
        )

    def test_evaluate_single_sample(self):
        """Test evaluation on a single sample."""
        # Set up mocks
        self.mock_planner.plan.return_value = [["r1", "r2"]]
        traj = Trajectory("Q?")
        traj.set_answer(["SanFrancisco"])
        self.mock_executor.execute.return_value = traj

        test_data = [
            {"question": "Q?", "question_entity": "E1", "answer_entities": ["SanFrancisco"]}
        ]

        results = self.evaluator.evaluate(test_data)
        self.assertIn("hits@1", results)
        self.assertEqual(results["hits@1"], 1.0)
        self.assertEqual(results["f1"], 1.0)
        self.assertEqual(results["accuracy"], 1.0)

    def test_evaluate_wrong_answer(self):
        """Test evaluation with wrong answer."""
        self.mock_planner.plan.return_value = [["r1"]]
        traj = Trajectory("Q?")
        traj.set_answer(["Wrong"])
        self.mock_executor.execute.return_value = traj

        test_data = [
            {"question": "Q?", "question_entity": "E1", "answer_entities": ["Correct"]}
        ]

        results = self.evaluator.evaluate(test_data)
        self.assertEqual(results["hits@1"], 0.0)
        self.assertEqual(results["f1"], 0.0)

    def test_evaluate_multiple_samples(self):
        """Test evaluation averages across samples."""
        self.mock_planner.plan.return_value = [["r1"]]

        traj1 = Trajectory("Q1")
        traj1.set_answer(["A"])
        traj2 = Trajectory("Q2")
        traj2.set_answer(["Wrong"])

        self.mock_executor.execute.side_effect = [traj1, traj2]

        test_data = [
            {"question": "Q1", "question_entity": "E1", "answer_entities": ["A"]},
            {"question": "Q2", "question_entity": "E2", "answer_entities": ["B"]},
        ]

        results = self.evaluator.evaluate(test_data)
        self.assertEqual(results["hits@1"], 0.5)

    def test_evaluate_max_samples(self):
        """Test max_samples limits evaluation."""
        self.mock_planner.plan.return_value = [["r1"]]
        traj = Trajectory("Q")
        traj.set_answer(["A"])
        self.mock_executor.execute.return_value = traj

        test_data = [
            {"question": f"Q{i}", "question_entity": "E", "answer_entities": ["A"]}
            for i in range(10)
        ]

        results = self.evaluator.evaluate(test_data, max_samples=3)
        # Should only evaluate 3 samples
        self.assertEqual(self.mock_executor.execute.call_count, 3)

    def test_evaluate_empty_data(self):
        """Test evaluation with empty data."""
        results = self.evaluator.evaluate([])
        self.assertEqual(results["hits@1"], 0.0)

    def test_compute_metrics_all(self):
        """Test _compute_metrics with all metric types."""
        self.evaluator.metrics = ["hits@1", "hits@3", "hits@10", "f1", "accuracy"]
        scores = self.evaluator._compute_metrics(["A", "B"], ["A", "B", "C"])
        self.assertIn("hits@1", scores)
        self.assertIn("hits@3", scores)
        self.assertIn("hits@10", scores)
        self.assertIn("f1", scores)
        self.assertIn("accuracy", scores)

    def test_evaluate_by_hop(self):
        """Test evaluation grouped by hop count."""
        self.mock_planner.plan.return_value = [["r1"]]
        traj = Trajectory("Q")
        traj.set_answer(["A"])
        self.mock_executor.execute.return_value = traj

        test_data = [
            {"question": "Q1", "question_entity": "E", "answer_entities": ["A"], "hop": 1},
            {"question": "Q2", "question_entity": "E", "answer_entities": ["A"], "hop": 1},
            {"question": "Q3", "question_entity": "E", "answer_entities": ["A"], "hop": 2},
        ]

        results = self.evaluator.evaluate_by_hop(test_data, hop_key="hop")
        self.assertIn(1, results)
        self.assertIn(2, results)


if __name__ == "__main__":
    unittest.main()
