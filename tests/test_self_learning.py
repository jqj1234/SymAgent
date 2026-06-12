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

    The implementation tokenizes each segment with add_special_tokens=False and
    concatenates the ids (so input_ids and label boundaries are consistent by
    construction), prepends a masked BOS, and appends a trainable EOS when the
    trajectory has at least one step.
    """

    class _FakeTokenizer:
        """Context-independent fake tokenizer: fixed token count per segment.

        Configured with explicit bos/eos ids and add_bos_token, matching the
        attributes prepare_training_data reads.
        """

        def __init__(self, tokens_per_segment=3, bos_token_id=101,
                     eos_token_id=99, add_bos_token=True):
            self.tokens_per_segment = tokens_per_segment
            self.bos_token_id = bos_token_id
            self.eos_token_id = eos_token_id
            self.add_bos_token = add_bos_token

        def __call__(self, text, add_special_tokens=True, return_tensors=None, **kwargs):
            # New implementation always calls per-segment with
            # add_special_tokens=False. Return fixed non-(-100) ids.
            n = self.tokens_per_segment
            ids = list(range(1, n + 1))
            return {"input_ids": ids, "attention_mask": [1] * n}

    def _make_mock_tokenizer(self, tokens_per_segment=3, has_bos=True):
        return self._FakeTokenizer(
            tokens_per_segment=tokens_per_segment,
            bos_token_id=101 if has_bos else None,
            add_bos_token=has_bos,
        )

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

        # BOS: token 0 (masked); Question: tokens 1-3 (masked)
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

    def test_labels_trainable_eos(self):
        """Test that a trainable EOS is appended at the end."""
        pool = TrajectoryPool()
        traj = Trajectory("Test Q")
        traj.add_step("think", "finish(A)", "obs")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        input_ids = data[0]["input_ids"]
        labels = data[0]["labels"]

        # Layout: BOS(0) Q(1-3) T(4-6) A(7-9) O(10-12) EOS(13) = 14 tokens
        self.assertEqual(len(input_ids), 14)
        self.assertEqual(input_ids[-1], 99)  # eos_token_id
        self.assertEqual(labels[-1], 99)     # trainable

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

    def test_empty_trajectory_dropped(self):
        """A trajectory with no steps has only masked tokens and is dropped."""
        pool = TrajectoryPool()
        traj = Trajectory("Empty question")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        # No trainable tokens -> dropped to avoid NaN loss.
        self.assertEqual(len(data), 0)

    def test_truncation_drops_all_masked_example(self):
        """If truncation removes every trainable token, the example is dropped."""
        pool = TrajectoryPool()
        traj = Trajectory("Q")
        traj.add_step("thought", "action", "obs")
        pool.add(traj)

        mock_tokenizer = self._make_mock_tokenizer(tokens_per_segment=3)
        # BOS(1) + Question(3) = 4 tokens, all masked. Thought starts at 4.
        data = prepare_training_data(pool, mock_tokenizer, max_length=4)
        self.assertEqual(len(data), 0)

    def test_multiple_steps_per_trajectory(self):
        """Test prepare_training_data with trajectory having multiple steps.

        Layout for 3 steps with 3 tokens/segment and BOS+EOS:
        BOS(0) Q(1-3)
        T1(4-6) A1(7-9) O1(10-12)
        T2(13-15) A2(16-18) O2(19-21)
        T3(22-24) A3(25-27) O3(28-30)
        EOS(31)
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

        # Total: 1 BOS + 10 segments * 3 tokens + 1 EOS = 32 tokens
        self.assertEqual(len(labels), 32)

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

        # EOS (31): unmasked
        self.assertNotEqual(labels[31], -100, "EOS should be trainable")


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
            exploration_mode="online",  # no local model in these unit tests
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


class TestExplorationMode(unittest.TestCase):
    """Test exploration_mode wiring (local self-training vs online distillation)."""

    def _make_learner(self, exploration_mode, model_name):
        kg = KGEnvironment()
        return SelfLearner(
            kg=kg,
            llm=MagicMock(),
            planner=MagicMock(),
            executor=MagicMock(),
            num_iterations=1,
            model_name=model_name,
            exploration_mode=exploration_mode,
        )

    def test_local_mode_loads_base_model_before_round0(self):
        """In local mode, run_full_loop loads the base model (no adapter) first."""
        learner = self._make_learner("local", model_name="/fake/Qwen")
        # Stub out the heavy bits.
        learner._switch_to_local_model = MagicMock()
        learner.run_iteration = MagicMock(return_value=TrajectoryPool())
        learner._save_trajectories = MagicMock()

        learner.run_full_loop([{"question": "q", "answer_entities": []}])

        # Round-0 local policy load: called with no adapter.
        learner._switch_to_local_model.assert_called_once_with(lora_path=None)

    def test_local_mode_requires_base_model(self):
        """local mode without base_model raises a clear error."""
        learner = self._make_learner("local", model_name=None)
        with self.assertRaises(ValueError):
            learner.run_full_loop([{"question": "q", "answer_entities": []}])

    def test_online_mode_does_not_preload_local_model(self):
        """In online mode, the API LLM stays the explorer (no round-0 swap)."""
        learner = self._make_learner("online", model_name="/fake/Qwen")
        learner._switch_to_local_model = MagicMock()
        learner.run_iteration = MagicMock(return_value=TrajectoryPool())
        learner._save_trajectories = MagicMock()
        # Empty merged pool -> no fine-tuning, so no switch at all.
        learner.run_full_loop([{"question": "q", "answer_entities": []}])

        learner._switch_to_local_model.assert_not_called()


if __name__ == "__main__":
    unittest.main()
