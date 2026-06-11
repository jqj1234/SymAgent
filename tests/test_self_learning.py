"""Tests for Self-Learning Framework.

Covers:
  - TrajectoryPool: add, extend, filter, get_rewarded
  - heuristic_merge (Eq. 7) - all 4 branches
  - prepare_training_data (Eq. 8) - indicator function
  - SelfLearner: online_explore, self_refine, _build_refine_prompt
  - SelfLearner: _parse_refined_trajectory, run_iteration
"""

import unittest
from unittest.mock import MagicMock

from src.executor import Trajectory, compute_outcome_reward
from src.kg_environment import KGEnvironment
from src.self_learning import SelfLearner, TrajectoryPool, heuristic_merge, prepare_training_data


class TestTrajectoryPool(unittest.TestCase):
    """Test cases for TrajectoryPool."""

    def test_add_trajectory(self):
        """Test adding trajectories to pool."""
        pool = TrajectoryPool()
        traj = Trajectory("Where does Alice live?")
        traj.set_reward(1.0)
        pool.add(traj)
        self.assertEqual(len(pool), 1)

    def test_extend_pool(self):
        """Test extending pool with multiple trajectories."""
        pool = TrajectoryPool()
        trajs = [Trajectory(f"Q{i}") for i in range(5)]
        pool.extend(trajs)
        self.assertEqual(len(pool), 5)

    def test_filter_by_reward(self):
        """Test filtering trajectories by reward threshold."""
        pool = TrajectoryPool()
        for i in range(5):
            traj = Trajectory(f"Q{i}")
            traj.set_reward(float(i) / 4.0)
            pool.add(traj)

        filtered = pool.filter_by_reward(min_reward=0.5)
        self.assertEqual(len(filtered), 2)  # Only rewards 0.75 and 1.0

    def test_filter_by_reward_zero_threshold(self):
        """Test filtering with reward_threshold=0.0 keeps all with reward > 0."""
        pool = TrajectoryPool()
        for i in range(3):
            traj = Trajectory(f"Q{i}")
            traj.set_reward(float(i))
            pool.add(traj)

        filtered = pool.filter_by_reward(min_reward=0.0)
        self.assertEqual(len(filtered), 2)  # rewards 1.0 and 2.0 (not 0.0)

    def test_get_rewarded(self):
        """Test getting trajectories with their rewards."""
        pool = TrajectoryPool()
        traj = Trajectory("Q1")
        traj.set_reward(0.8)
        pool.add(traj)

        rewarded = pool.get_rewarded()
        self.assertEqual(len(rewarded), 1)
        self.assertEqual(rewarded[0][1], 0.8)

    def test_getitem(self):
        """Test index access to pool."""
        pool = TrajectoryPool()
        traj1 = Trajectory("Q1")
        traj2 = Trajectory("Q2")
        pool.add(traj1)
        pool.add(traj2)
        self.assertEqual(pool[0].question, "Q1")
        self.assertEqual(pool[1].question, "Q2")


class TestHeuristicMerge(unittest.TestCase):
    """Test cases for heuristic_merge (Equation 7).

    Eq. 7:
    D_0*(i) = {
        (mu_i, r(mu_i)),          if r(mu_i) > r(mu_hat_i)
        (mu_hat_i, r(mu_hat_i)),  if r(mu_i) < r(mu_hat_i)
        (t, r(t)),                if r(mu_i) = r(mu_hat_i) > 0, t = argmin |s|
        filtered,                 if r(mu_i) = r(mu_hat_i) = 0
    }
    """

    def _make_traj(self, question: str, reward: float, num_steps: int = 2) -> Trajectory:
        """Helper to create a trajectory with given properties."""
        traj = Trajectory(question)
        traj.set_reward(reward)
        for i in range(num_steps):
            traj.add_step(f"Thought {i}", f"Action {i}", f"Observation {i}")
        return traj

    def test_original_better(self):
        """Test branch 1: original trajectory has higher reward."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.8))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.5))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reward, 0.8)

    def test_refined_better(self):
        """Test branch 2: refined trajectory has higher reward."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.3))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.9))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reward, 0.9)

    def test_equal_nonzero_reward_selects_shorter(self):
        """Test branch 3: equal non-zero reward selects shorter trajectory."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.7, num_steps=5))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.7, num_steps=3))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 1)
        self.assertEqual(len(merged[0]), 3)  # Shorter one selected

    def test_equal_nonzero_reward_same_length(self):
        """Test branch 3 with same length: selects original (first)."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.5, num_steps=3))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.5, num_steps=3))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reward, 0.5)

    def test_both_zero_reward_filtered(self):
        """Test branch 4: both rewards are zero (filtered out)."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.0))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.0))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 0)

    def test_mixed_case(self):
        """Test mixed case with multiple trajectories."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.8))  # Branch 1: orig better
        orig.add(self._make_traj("Q2", reward=0.0))  # Branch 4: both 0
        orig.add(self._make_traj("Q3", reward=0.5, num_steps=4))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.5))
        refined.add(self._make_traj("Q2", reward=0.0))
        refined.add(self._make_traj("Q3", reward=0.5, num_steps=2))  # Branch 3: shorter

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged[0].reward, 0.8)  # Q1: orig better
        self.assertEqual(len(merged[1]), 2)       # Q3: shorter selected

    def test_mismatched_pool_sizes_raises(self):
        """Test that mismatched pool sizes raise an assertion error."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=1.0))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.5))
        refined.add(self._make_traj("Q2", reward=0.5))

        with self.assertRaises(AssertionError):
            heuristic_merge(orig, refined)

    def test_empty_pools(self):
        """Test merging two empty pools."""
        orig = TrajectoryPool()
        refined = TrajectoryPool()
        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 0)

    def test_single_trajectory_better_refined(self):
        """Test single trajectory where refined improves from 0 to 0.5."""
        orig = TrajectoryPool()
        orig.add(self._make_traj("Q1", reward=0.0))

        refined = TrajectoryPool()
        refined.add(self._make_traj("Q1", reward=0.5))

        merged = heuristic_merge(orig, refined)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reward, 0.5)


class TestPrepareTrainingData(unittest.TestCase):
    """Test prepare_training_data (Eq. 8 - Indicator Function).

    Eq. 8:
    L_SFT = -E_{mu~D*} [sum_j 1(x_j in A) * log pi_theta(x_j | x_{<j}, q)]

    Only thought and action tokens are trainable (1(x_j in A) = 1).
    Question and observation tokens are masked (1(x_j in A) = 0, label = -100).

    The implementation tokenizes the full prompt first (with special tokens),
    then tokenizes each segment separately (without special tokens) to compute
    the cursor position. The first token (BOS/special) is always masked.
    """

    def _make_mock_tokenizer(self, tokens_per_segment=3, has_bos=True):
        """Create a mock tokenizer that returns a fixed number of tokens per segment.

        The full prompt tokenization includes a BOS token if has_bos=True.
        Individual segment tokenizations (add_special_tokens=False) do not.

        Empty string tokenization:
        - With special tokens: [BOS] (1 token) if has_bos, else [] (0 tokens)
        - Without special tokens: [] (0 tokens)
        """
        mock_tokenizer = MagicMock()

        def mock_tokenize(text, **kwargs):
            add_special = kwargs.get("add_special_tokens", True)
            n = tokens_per_segment

            if not text and not text.strip():
                # Empty string tokenization for offset computation
                if add_special and has_bos:
                    return {"input_ids": [0], "attention_mask": [1]}  # BOS
                else:
                    return {"input_ids": [], "attention_mask": []}

            if add_special and has_bos:
                # Full tokenization: count segments, return BOS + segments * n
                lines = text.strip().split("\n")
                num_segments = len(lines)
                total = 1 + num_segments * n  # 1 BOS + segments
                return {"input_ids": list(range(total)), "attention_mask": [1] * total}
            else:
                # Individual segment: just N tokens
                return {"input_ids": list(range(n)), "attention_mask": [1] * n}

        mock_tokenizer.side_effect = mock_tokenize
        return mock_tokenizer

    def test_labels_mask_question_tokens(self):
        """Test that question tokens are masked with -100 in labels."""
        pool = TrajectoryPool()
        traj = Trajectory("Where does Alice live?")
        traj.add_step("I need to search", "searchNeighbor(Alice, liveIn)", "SanFrancisco")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)

        self.assertEqual(len(data), 1)
        labels = data[0]["labels"]

        # Token 0 is BOS (masked).
        # Tokens 1-3 are question (masked).
        self.assertEqual(labels[0], -100)
        self.assertEqual(labels[1], -100)
        self.assertEqual(labels[2], -100)
        self.assertEqual(labels[3], -100)

    def test_labels_unmask_thought_tokens(self):
        """Test that thought tokens are unmasked in labels."""
        pool = TrajectoryPool()
        traj = Trajectory("Test Q")
        traj.add_step("Think about it", "finish(A)", "obs")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        labels = data[0]["labels"]

        # Full prompt: 4 segments (question, thought, action, obs) -> 1+4*3=13 tokens
        # BOS: token 0 (masked)
        # Question: tokens 1-3 (masked)
        # Thought: tokens 4-6 (unmasked)
        for i in range(4, 7):
            self.assertNotEqual(labels[i], -100, f"Thought token at position {i} should be unmasked")

    def test_labels_unmask_action_tokens(self):
        """Test that action tokens are unmasked in labels."""
        pool = TrajectoryPool()
        traj = Trajectory("Test Q")
        traj.add_step("think", "finish(A)", "obs")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        labels = data[0]["labels"]

        # Action: tokens 7-9 (unmasked)
        for i in range(7, 10):
            self.assertNotEqual(labels[i], -100, f"Action token at position {i} should be unmasked")

    def test_labels_mask_observation_tokens(self):
        """Test that observation tokens are masked with -100."""
        pool = TrajectoryPool()
        traj = Trajectory("Test Q")
        traj.add_step("think", "finish(A)", "Result: A")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        labels = data[0]["labels"]

        # Observation: tokens 10-12 (masked)
        for i in range(10, 13):
            self.assertEqual(labels[i], -100, f"Observation token at position {i} should be masked")

    def test_multiple_trajectories(self):
        """Test prepare_training_data with multiple trajectories."""
        pool = TrajectoryPool()
        for i in range(3):
            traj = Trajectory(f"Question {i}")
            traj.add_step("thought", "action", "observation")
            pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        self.assertEqual(len(data), 3)

    def test_empty_trajectory(self):
        """Test prepare_training_data with trajectory that has no steps."""
        pool = TrajectoryPool()
        traj = Trajectory("Empty question")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        self.assertEqual(len(data), 1)
        # All tokens should be masked (only question + BOS, no steps)
        labels = data[0]["labels"]
        self.assertTrue(all(l == -100 for l in labels))

    def test_multiple_steps_per_trajectory(self):
        """Test prepare_training_data with trajectory having multiple steps.

        3 steps -> full prompt has 10 segments (1 question + 3*(thought+action+obs))
        BOS + 10*3 = 31 tokens
        Masked: BOS(0), Question(1-3), Obs1(7-9), Obs2(13-15), Obs3(19-21)
        Unmasked: Thought1(4-6), Action1(10-12), Thought2(16-18), Action2(22-24),
                  Thought3(28-30), Action3(25-27)
        """
        pool = TrajectoryPool()
        traj = Trajectory("Multi-step Q")
        traj.add_step("thought1", "action1", "obs1")
        traj.add_step("thought2", "action2", "obs2")
        traj.add_step("thought3", "action3", "obs3")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        self.assertEqual(len(data), 1)
        labels = data[0]["labels"]

        # Total: 1 BOS + 10 segments * 3 tokens = 31 tokens
        self.assertEqual(len(labels), 31)

        # BOS (0) and Question (1-3): masked
        for pos in [0, 1, 2, 3]:
            self.assertEqual(labels[pos], -100, f"Position {pos} should be masked")

        # Thought1 (4-6): unmasked
        for pos in range(4, 7):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Action1 (7-9): unmasked
        for pos in range(7, 10):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Obs1 (10-12): masked
        for pos in range(10, 13):
            self.assertEqual(labels[pos], -100, f"Position {pos} should be masked")

        # Thought2 (13-15): unmasked
        for pos in range(13, 16):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Action2 (16-18): unmasked
        for pos in range(16, 19):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Obs2 (19-21): masked
        for pos in range(19, 22):
            self.assertEqual(labels[pos], -100, f"Position {pos} should be masked")

        # Thought3 (22-24): unmasked
        for pos in range(22, 25):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Action3 (25-27): unmasked
        for pos in range(25, 28):
            self.assertNotEqual(labels[pos], -100, f"Position {pos} should be unmasked")

        # Obs3 (28-30): masked
        for pos in range(28, 31):
            self.assertEqual(labels[pos], -100, f"Position {pos} should be masked")


class TestSelfLearner(unittest.TestCase):
    """Test cases for SelfLearner class."""

    def setUp(self):
        """Set up test components."""
        self.kg = KGEnvironment()
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")

        self.mock_llm = MagicMock()
        self.mock_planner = MagicMock()
        self.mock_executor = MagicMock()

        self.learner = SelfLearner(
            kg=self.kg,
            llm=self.mock_llm,
            planner=self.mock_planner,
            executor=self.mock_executor,
            num_iterations=1,
            reward_threshold=0.0,
            model_name=None,  # Skip LoRA
        )

    def test_online_explore(self):
        """Test online exploration produces trajectories with rewards."""
        traj = Trajectory("Where does Alice work?")
        traj.add_step("thought", "action", "obs")
        traj.set_answer(["OpenAI"])
        self.mock_executor.execute.return_value = traj

        qa_pairs = [
            {"question": "Where does Alice work?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]

        pool = self.learner.online_explore(qa_pairs)
        self.assertEqual(len(pool), 1)
        self.assertEqual(pool[0].reward, 1.0)

    def test_online_explore_zero_reward(self):
        """Test online exploration with wrong answer."""
        traj = Trajectory("Where does Alice work?")
        traj.add_step("thought", "action", "obs")
        traj.set_answer(["Google"])  # Wrong answer
        self.mock_executor.execute.return_value = traj

        qa_pairs = [
            {"question": "Where does Alice work?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]

        pool = self.learner.online_explore(qa_pairs)
        self.assertEqual(len(pool), 0)  # Filtered out by reward_threshold=0.0

    def test_online_explore_multiple(self):
        """Test online exploration with multiple QA pairs."""
        def make_traj(q, answer, gt):
            traj = Trajectory(q)
            traj.add_step("thought", "action", "obs")
            traj.set_answer(answer)
            traj.ground_truth_entities = gt
            return traj

        self.mock_executor.execute.side_effect = [
            make_traj("Q1", ["OpenAI"], ["OpenAI"]),   # reward=1.0
            make_traj("Q2", ["Google"], ["OpenAI"]),    # reward=0.0
            make_traj("Q3", ["OpenAI"], ["OpenAI", "SF"]),  # reward=0.5
        ]

        qa_pairs = [
            {"question": "Q1", "question_entity": "Alice", "answer_entities": ["OpenAI"]},
            {"question": "Q2", "question_entity": "Alice", "answer_entities": ["OpenAI"]},
            {"question": "Q3", "question_entity": "Alice", "answer_entities": ["OpenAI", "SF"]},
        ]

        pool = self.learner.online_explore(qa_pairs)
        self.assertEqual(len(pool), 2)  # Q2 filtered (reward=0.0)

    def test_self_refine_success_prompt(self):
        """Test refine prompt for successful trajectory."""
        traj = Trajectory("Q?")
        traj.add_step("thought", "action", "obs")
        traj.set_answer(["OpenAI"])
        traj.set_reward(1.0)

        prompt = self.learner._build_refine_prompt(traj)
        self.assertIn("successful", prompt.lower())
        self.assertIn("concise", prompt.lower())

    def test_self_refine_failure_prompt(self):
        """Test refine prompt for failed trajectory."""
        traj = Trajectory("Q?")
        traj.add_step("thought", "action", "obs")
        traj.set_answer(["Wrong"])
        traj.set_reward(0.0)

        prompt = self.learner._build_refine_prompt(traj)
        self.assertIn("unsuccessful", prompt.lower())
        self.assertIn("went wrong", prompt.lower())

    def test_self_refine_includes_trajectory(self):
        """Test refine prompt includes original trajectory steps."""
        traj = Trajectory("Q?")
        traj.add_step("my thought", "my action", "my obs")
        traj.set_answer(["A"])
        traj.set_reward(0.5)

        prompt = self.learner._build_refine_prompt(traj)
        self.assertIn("my thought", prompt)
        self.assertIn("my action", prompt)
        self.assertIn("my obs", prompt)

    def test_parse_refined_trajectory(self):
        """Test parsing refined trajectory from LLM response."""
        response = (
            "Thought 1: I should search for Alice's employer\n"
            "Action 1: searchNeighbor(Alice, workFor)\n"
            "Thought 2: Found OpenAI\n"
            "Action 2: finish(OpenAI)\n"
        )

        traj = self.learner._parse_refined_trajectory(
            question="Where does Alice work?",
            response=response,
            original_reward=0.0,
            ground_truth_entities=["OpenAI"],
        )

        self.assertIsNotNone(traj)
        self.assertEqual(len(traj), 2)
        self.assertEqual(traj.answer_entities, ["OpenAI"])
        self.assertEqual(traj.reward, 1.0)

    def test_parse_refined_trajectory_empty_response(self):
        """Test parsing empty refined response returns None."""
        traj = self.learner._parse_refined_trajectory(
            question="Q?", response="No steps here", original_reward=0.0,
            ground_truth_entities=["A"],
        )
        self.assertIsNone(traj)

    def test_run_iteration(self):
        """Test single iteration: explore -> refine -> merge."""
        explore_traj = Trajectory("Q?")
        explore_traj.add_step("thought", "action", "obs")
        explore_traj.set_answer(["OpenAI"])
        explore_traj.set_reward(1.0)
        self.mock_executor.execute.return_value = explore_traj

        self.mock_llm.execute_generate.return_value = (
            "Thought 1: Correct thought\n"
            "Action 1: finish(OpenAI)\n"
        )

        qa_pairs = [
            {"question": "Q?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]

        merged = self.learner.run_iteration(qa_pairs, iteration=0)
        self.assertGreater(len(merged), 0)

    def test_save_and_load_trajectories(self):
        """Test trajectory save/load roundtrip."""
        import tempfile, os
        pool = TrajectoryPool()
        traj = Trajectory("Test Q")
        traj.add_step("thought", "action", "obs")
        traj.set_answer(["A"])
        traj.set_reward(0.8)
        traj.planned_paths = [["r1"]]
        pool.add(traj)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name

        try:
            self.learner._save_trajectories(pool, filepath)
            loaded = SelfLearner.load_trajectories(filepath)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].question, "Test Q")
            self.assertEqual(loaded[0].reward, 0.8)
            self.assertEqual(loaded[0].answer_entities, ["A"])
            self.assertEqual(loaded[0].planned_paths, [["r1"]])
        finally:
            os.unlink(filepath)


if __name__ == "__main__":
    unittest.main()
