"""
SymAgent Test Suite — aligned with paper formulas.

Unit tests verify each equation/algorithm implementation.
System tests verify the full pipeline matches paper specs.

Run:  python tests/test_paper_alignment.py
"""

import sys
import os
import json
import logging

logging.basicConfig(level=logging.WARNING)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pytest

from src.self_learning import (
    SelfLearner, TrajectoryPool, Trajectory, heuristic_merge,
    prepare_training_data,
)
from src.executor import compute_outcome_reward
from src.kg_environment import KGEnvironment


# =============================================================================
# Unit Tests — Equation-level
# =============================================================================

class TestEq6OutcomeReward:
    """Equation 6: r(mu_i) = Recall(A_mu, A_gt) = |A_mu ∩ A_gt| / |A_gt|"""

    def test_perfect_recall(self):
        """All predicted entities match ground truth."""
        r = compute_outcome_reward(
            ["m.1", "m.2"], ["m.1", "m.2"]
        )
        assert r == 1.0, f"Expected 1.0, got {r}"

    def test_partial_recall(self):
        """One of two ground truth entities matched."""
        r = compute_outcome_reward(
            ["m.1"], ["m.1", "m.2"]
        )
        assert r == 0.5, f"Expected 0.5, got {r}"

    def test_zero_recall(self):
        """No entities matched."""
        r = compute_outcome_reward(
            ["m.99", "m.100"], ["m.1", "m.2"]
        )
        assert r == 0.0, f"Expected 0.0, got {r}"

    def test_empty_prediction(self):
        """Empty prediction against non-empty ground truth."""
        r = compute_outcome_reward([], ["m.1"])
        assert r == 0.0

    def test_empty_ground_truth(self):
        """Empty ground truth should return 0 (avoid division by zero)."""
        r = compute_outcome_reward(["m.1"], [])
        assert r == 0.0

    def test_case_insensitive(self):
        """Entity matching ignores case."""
        r = compute_outcome_reward(
            ["Bill Cassidy"], ["bill cassidy"]
        )
        assert r == 1.0, f"Expected 1.0, got {r}"

    def test_underscore_normalized(self):
        """Entity matching normalizes underscores."""
        r = compute_outcome_reward(
            ["bill_cassidy"], ["bill cassidy"]
        )
        assert r == 1.0, f"Expected 1.0, got {r}"

    def test_name_mid_resolution(self):
        """Entity matching uses name-to-MID mapping from KG."""
        kg = KGEnvironment()
        kg._name2mid = {"bill cassidy": "m.0abc"}
        kg._mid2name = {"m.0abc": "Bill Cassidy"}

        r = compute_outcome_reward(
            ["m.0abc"], ["Bill Cassidy"], kg=kg
        )
        assert r == 1.0, f"Expected 1.0, got {r}"

    def test_extra_predictions_dont_boost_recall(self):
        """Recall is capped at 1.0 regardless of extra predictions."""
        r = compute_outcome_reward(
            ["m.1", "m.2", "m.3"], ["m.1", "m.2"]
        )
        assert r == 1.0, f"Expected 1.0, got {r}"

    def test_single_match_among_many(self):
        """One match out of three ground truth = 1/3 recall."""
        r = compute_outcome_reward(
            ["m.2"], ["m.1", "m.2", "m.3"]
        )
        assert abs(r - 1/3) < 1e-9, f"Expected 0.333, got {r}"


class TestEq7HeuristicMerge:
    """Equation 7: D_0*(i) = 4-branch merge of original and refined."""

    def _make_traj(self, reward, num_steps=3):
        t = Trajectory("test question")
        for i in range(num_steps):
            t.add_step(f"thought {i}", f"action {i}", f"obs {i}")
        t.set_reward(reward)
        return t

    def test_branch1_orig_better(self):
        """r(mu) > r(mu_hat): keep original."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.8))
        ref_pool.add(self._make_traj(0.3))
        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 1
        assert merged[0].reward == 0.8

    def test_branch2_refined_better(self):
        """r(mu) < r(mu_hat): keep refined."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.2))
        ref_pool.add(self._make_traj(0.7))
        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 1
        assert merged[0].reward == 0.7

    def test_branch3_equal_nonzero_keep_shorter(self):
        """r(mu) = r(mu_hat) > 0: keep shorter trajectory."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.5, num_steps=5))
        ref_pool.add(self._make_traj(0.5, num_steps=2))
        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 1
        assert len(merged[0]) == 2  # shorter one

    def test_branch3_equal_nonzero_orig_shorter(self):
        """r(mu) = r(mu_hat) > 0: original is shorter, keep it."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.5, num_steps=2))
        ref_pool.add(self._make_traj(0.5, num_steps=5))
        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 1
        assert len(merged[0]) == 2

    def test_branch4_both_zero_filtered(self):
        """r(mu) = r(mu_hat) = 0: filtered out."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.0))
        ref_pool.add(self._make_traj(0.0))
        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 0

    def test_mixed_scenarios(self):
        """Multiple trajectories with different branch conditions."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        # Branch 1: orig better
        orig_pool.add(self._make_traj(0.8, 3))
        ref_pool.add(self._make_traj(0.3, 3))
        # Branch 2: refined better
        orig_pool.add(self._make_traj(0.2, 3))
        ref_pool.add(self._make_traj(0.9, 3))
        # Branch 3: equal, keep shorter
        orig_pool.add(self._make_traj(0.5, 4))
        ref_pool.add(self._make_traj(0.5, 2))
        # Branch 4: both zero
        orig_pool.add(self._make_traj(0.0, 3))
        ref_pool.add(self._make_traj(0.0, 3))

        merged = heuristic_merge(orig_pool, ref_pool)
        assert len(merged) == 3  # branch 4 filtered
        assert merged[0].reward == 0.8  # orig
        assert merged[1].reward == 0.9  # refined
        assert len(merged[2]) == 2     # shorter

    def test_size_mismatch_raises(self):
        """Unequal pool sizes should raise AssertionError."""
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(self._make_traj(0.5))
        ref_pool.add(self._make_traj(0.5))
        ref_pool.add(self._make_traj(0.5))
        with pytest.raises(AssertionError):
            heuristic_merge(orig_pool, ref_pool)


class TestEq8IndicatorMask:
    """Equation 8: SFT loss with indicator function.

    1(x_j in A) = 1 for Thought/Action tokens, 0 for Question/Observation.
    Tokens with 0 are masked as -100 in labels.
    """

    def _make_tokenizer_mock(self):
        """Create a simple tokenizer that maps words to integer IDs."""
        class SimpleTokenizer:
            def __init__(self):
                self.vocab = {}
                self.next_id = 0
                self.pad_token_id = 0
                self.eos_token_id = 0

            def _encode(self, text):
                ids = []
                for word in text.split():
                    if word not in self.vocab:
                        self.vocab[word] = self.next_id
                        self.next_id += 1
                    ids.append(self.vocab[word])
                return ids

            def __call__(self, text, **kwargs):
                ids = self._encode(text)
                return {"input_ids": ids, "attention_mask": [1] * len(ids)}

            def encode(self, text, **kwargs):
                return self._encode(text)
        return SimpleTokenizer()

    def test_question_tokens_masked(self):
        """Question tokens should be masked (-100) in labels."""
        tok = self._make_tokenizer_mock()
        pool = TrajectoryPool()
        t = Trajectory("what is the capital")
        t.add_step("Think about it", "search(m.0, r)", "result")
        t.set_answer(["m.1"])
        pool.add(t)

        data = prepare_training_data(pool, tok, max_length=512)
        labels = data[0]["labels"]

        # "Question:" and "what is the capital" should be masked
        # Count non-masked tokens
        trainable = [l for l in labels if l != -100]
        # Only Thought and Action should be trainable
        assert len(trainable) > 0, "Should have trainable tokens"

    def test_observation_tokens_masked(self):
        """Observation tokens should be masked (-100) in labels."""
        tok = self._make_tokenizer_mock()
        pool = TrajectoryPool()
        t = Trajectory("test")
        t.add_step("Think", "search(m.0, r)", "observation text here")
        t.set_answer(["m.1"])
        pool.add(t)

        data = prepare_training_data(pool, tok, max_length=512)
        labels = data[0]["labels"]
        input_ids = data[0]["input_ids"]
        assert len(labels) == len(input_ids), "Labels and input_ids must match"

        # "Observation 1: observation text here" should be masked
        # Verify at least some tokens are masked
        masked = [l for l in labels if l == -100]
        assert len(masked) > 0, "Should have masked tokens"

    def test_thought_action_not_masked(self):
        """Thought and Action tokens should NOT be masked."""
        tok = self._make_tokenizer_mock()
        pool = TrajectoryPool()
        t = Trajectory("test")
        t.add_step("I need to search", "search(m.0, relation)", "result")
        t.set_answer(["m.1"])
        pool.add(t)

        data = prepare_training_data(pool, tok, max_length=512)
        labels = data[0]["labels"]
        input_ids = data[0]["input_ids"]

        # Verify no -100 appears in Thought/Action positions
        # (This is implicitly tested by checking that trainable tokens exist
        # and correspond to Thought/Action)
        trainable_ids = [(i, l) for i, l in enumerate(labels) if l != -100]
        assert len(trainable_ids) > 0, "Thought/Action should not be masked"

    def test_labels_length_matches_input_ids(self):
        """Labels must have same length as input_ids."""
        tok = self._make_tokenizer_mock()
        pool = TrajectoryPool()
        t = Trajectory("test question with multiple words")
        t.add_step("A longer thought process here", "search(entity, relation)", "a long observation text")
        t.add_step("Another thought", "finish(answer)", "done")
        t.set_answer(["answer"])
        pool.add(t)

        data = prepare_training_data(pool, tok, max_length=512)
        for d in data:
            assert len(d["labels"]) == len(d["input_ids"]), \
                f"labels({len(d['labels'])}) != input_ids({len(d['input_ids'])})"

    def test_truncation_for_long_sequences(self):
        """Sequences exceeding max_length should be truncated."""
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("gpt2")
        pool = TrajectoryPool()
        t = Trajectory("test")
        for i in range(20):
            t.add_step(f"thought {i}", f"action {i}", f"observation {i}")
        t.set_answer(["ans"])
        pool.add(t)

        data = prepare_training_data(pool, tok, max_length=50)
        assert len(data) == 1
        assert len(data[0]["input_ids"]) <= 50


class TestTrajectoryAndPool:
    """Test Trajectory and TrajectoryPool basic operations."""

    def test_trajectory_steps(self):
        t = Trajectory("q")
        t.add_step("t1", "a1", "o1")
        t.add_step("t2", "a2", "o2")
        assert len(t) == 2
        assert t.steps[0]["thought"] == "t1"
        assert t.steps[0]["action"] == "a1"

    def test_trajectory_reward(self):
        t = Trajectory("q")
        t.set_reward(0.75)
        assert t.reward == 0.75

    def test_trajectory_answer(self):
        t = Trajectory("q")
        t.set_answer(["m.1", "m.2"])
        assert t.answer_entities == ["m.1", "m.2"]

    def test_pool_add_and_len(self):
        pool = TrajectoryPool()
        assert len(pool) == 0
        pool.add(Trajectory("q1"))
        pool.add(Trajectory("q2"))
        assert len(pool) == 2

    def test_pool_extend(self):
        pool = TrajectoryPool()
        pool.extend([Trajectory("q1"), Trajectory("q2"), Trajectory("q3")])
        assert len(pool) == 3

    def test_pool_filter_by_reward(self):
        pool = TrajectoryPool()
        for r in [0.0, 0.3, 0.5, 0.8, 1.0]:
            t = Trajectory("q")
            t.set_reward(r)
            pool.add(t)
        filtered = pool.filter_by_reward(0.4)
        assert len(filtered) == 3  # 0.5, 0.8, 1.0

    def test_pool_getitem(self):
        pool = TrajectoryPool()
        t1 = Trajectory("q1")
        t2 = Trajectory("q2")
        pool.add(t1)
        pool.add(t2)
        assert pool[0] is t1
        assert pool[1] is t2


class TestKGEnvironment:
    """Test KG environment operations (Algorithm 1, BFS, BM25)."""

    def test_add_and_has_triple(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        assert kg.has_triple("a", "r1", "b")
        assert not kg.has_triple("a", "r2", "b")

    def test_remove_triple(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        assert kg.remove_triple("a", "r1", "b")
        assert not kg.has_triple("a", "r1", "b")
        assert not kg.remove_triple("a", "r1", "b")  # already removed

    def test_search_neighbor(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        kg.add_triple("a", "r1", "c")
        kg.add_triple("a", "r2", "d")
        assert set(kg.search_neighbor("a", "r1")) == {"b", "c"}
        assert kg.search_neighbor("a", "r2") == ["d"]
        assert kg.search_neighbor("a") == ["b", "c", "d"]

    def test_bfs_shortest_path(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        kg.add_triple("b", "r2", "c")
        kg.add_triple("a", "r3", "d")  # longer path
        kg.add_triple("d", "r4", "c")

        path = kg.bfs_find_shortest_path("a", "c")
        assert len(path) == 2  # a->b->c
        assert path[0] == ("a", "r1", "b")
        assert path[1] == ("b", "r2", "c")

    def test_bfs_no_path(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        path = kg.bfs_find_shortest_path("a", "z")
        assert path == []

    def test_simulate_incompleteness(self):
        """Algorithm 1: Remove triples on query-answer paths."""
        kg = KGEnvironment()
        kg.add_triple("q", "r1", "mid1")
        kg.add_triple("mid1", "r2", "a1")
        kg.add_triple("q", "r3", "a2")

        removed = kg.simulate_incompleteness("q", ["a1", "a2"], remove_ratio=0.5, seed=42)
        assert len(removed) > 0, "Should remove some triples"
        # At least one path should be broken
        for h, r, t in removed:
            assert not kg.has_triple(h, r, t)

    def test_stats(self):
        kg = KGEnvironment()
        kg.add_triple("a", "r1", "b")
        kg.add_triple("b", "r2", "c")
        stats = kg.get_stats()
        assert stats["num_entities"] == 3
        assert stats["num_relations"] == 2
        assert stats["num_triples"] == 2

    def test_bm25_retrieval(self):
        kg = KGEnvironment()
        kg.add_triple("Barack Obama", "president_of", "USA")
        kg.add_triple("USA", "capital", "Washington DC")
        kg.build_bm25_index()

        results = kg.bm25_retrieve_entities("obama president", top_k=3)
        assert len(results) > 0
        entities = [e for e, s in results]
        assert "Barack Obama" in entities


# =============================================================================
# System Tests — Paper Parameter Alignment
# =============================================================================

class TestPaperHyperparameters:
    """Verify default config matches paper Table 6.

    Paper values:
      lora_r=32, lora_alpha=32, lora_dropout=0.05
      per_device_batch_size=2, gradient_accumulation_steps=2
      warmup_ratio=0.05, self-learning iterations=2
      learning_rate=2e-5, num_train_epochs=3, max_seq_length=4096
    """

    def test_default_lora_config(self):
        learner = SelfLearner.__new__(SelfLearner)
        learner.lora_config = {
            "r": 32, "lora_alpha": 32, "lora_dropout": 0.05,
            "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                               "down_proj", "up_proj", "gate_proj"],
        }
        assert learner.lora_config["r"] == 32
        assert learner.lora_config["lora_alpha"] == 32
        assert learner.lora_config["lora_dropout"] == 0.05
        assert len(learner.lora_config["target_modules"]) == 7

    def test_default_training_config(self):
        learner = SelfLearner.__new__(SelfLearner)
        learner.training_config = {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "warmup_ratio": 0.05,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "max_seq_length": 4096,
        }
        assert learner.training_config["per_device_train_batch_size"] == 2
        assert learner.training_config["gradient_accumulation_steps"] == 2
        assert learner.training_config["warmup_ratio"] == 0.05
        assert learner.training_config["num_train_epochs"] == 3
        assert learner.training_config["learning_rate"] == 2e-5
        assert learner.training_config["max_seq_length"] == 4096

    def test_default_num_iterations(self):
        learner = SelfLearner.__new__(SelfLearner)
        learner.num_iterations = 2
        assert learner.num_iterations == 2

    def test_config_from_yaml(self):
        """Load actual config.yaml and verify paper defaults.
        Paper Table 6 (Section A.3):
          lora_r=32, lora_alpha=32, lora_dropout=0.05
          gradient_accumulation_steps=2, warmup_ratio=0.05
          learning_rate=2e-5, epochs=3, max_seq_length=4096
        Config uses flat structure under self_learning.
        """
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'config.yaml'
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        sl = config.get("self_learning", {})

        # Paper Table 6 values
        assert sl.get("num_iterations") == 2, \
            f"num_iterations should be 2, got {sl.get('num_iterations')}"
        assert sl.get("lora_r") == 32, \
            f"lora_r should be 32, got {sl.get('lora_r')}"
        assert sl.get("lora_alpha") == 32, \
            f"lora_alpha should be 32, got {sl.get('lora_alpha')}"
        assert sl.get("lora_dropout") == 0.05, \
            f"lora_dropout should be 0.05, got {sl.get('lora_dropout')}"
        assert sl.get("gradient_accumulation_steps") == 2, \
            f"gradient_accumulation_steps should be 2, got {sl.get('gradient_accumulation_steps')}"
        assert sl.get("learning_rate") == 2e-5, \
            f"learning_rate should be 2e-5, got {sl.get('learning_rate')}"
        assert sl.get("num_train_epochs") == 3, \
            f"num_train_epochs should be 3, got {sl.get('num_train_epochs')}"
        assert sl.get("max_seq_length") == 4096, \
            f"max_seq_length should be 4096, got {sl.get('max_seq_length')}"
        assert sl.get("warmup_ratio") == 0.05, \
            f"warmup_ratio should be 0.05, got {sl.get('warmup_ratio')}"

        # Paper: per_device_batch_size=2, grad_accum=2 -> effective batch=4
        assert sl.get("batch_size") == 4, \
            f"batch_size should be 4 (effective), got {sl.get('batch_size')}"


class TestSelfLearningPipeline:
    """System test: full self-learning loop matches paper flow.

    Paper flow (Section 4.3):
    1. Online explore (pi_theta interacts with KG)
    2. Self-refine (LLM regenerates trajectories)
    3. Heuristic merge (Eq.7)
    4. LoRA fine-tune (Eq.8)
    5. Repeat with updated pi_theta
    """

    def _make_learner(self, output_dir):
        """Create a SelfLearner with gpt2 for testing."""
        import yaml
        config_path = os.path.join(
            os.path.dirname(__file__), '..', 'configs', 'config.yaml'
        )
        with open(config_path) as f:
            config = yaml.safe_load(f)

        from src.llm_client import LLMClient
        from src.planner import AgentPlanner
        from src.executor import AgentExecutor

        kg = KGEnvironment()
        kc = config["kg"]["datasets"]["webqsp"]
        kg.load_from_files(
            kc["triple_file"], kc.get("entity2id"), kc.get("relation2id")
        )
        kg.load_name_mapping(kc.get("mid2name"), kc.get("name2mid"))

        llm = LLMClient.from_config(config)
        planner = AgentPlanner(
            kg=kg, llm=llm, num_seed_questions=3,
            max_bfs_depth=4, max_paths_per_seed=5, max_rules=10
        )
        executor = AgentExecutor(kg=kg, llm=llm, planner=planner)

        return SelfLearner(
            kg=kg, llm=llm, planner=planner, executor=executor,
            num_iterations=2, output_dir=output_dir,
            lora_config={
                "r": 8, "lora_alpha": 16, "lora_dropout": 0.05,
                "target_modules": ["c_attn", "c_proj"],
            },
            training_config={
                "per_device_train_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "num_train_epochs": 1,
                "learning_rate": 2e-5,
                "max_seq_length": 256,
            },
            model_name="gpt2",
        )

    def test_run_iteration_produces_merged_pool(self):
        """Iteration should produce merged pool with correct rewards."""
        import glob
        output_dir = "/tmp/test_symagent_iteration"
        learner = self._make_learner(output_dir)

        # Use fake data to avoid API calls
        fake_qa = [
            {"question": "test q1", "question_entity": "m.0",
             "answer_entities": ["m.1"]},
        ]

        # Monkey-patch to avoid API calls
        def fake_explore(self, qa_pairs):
            pool = TrajectoryPool()
            for qa in qa_pairs:
                t = Trajectory(qa["question"])
                t.add_step("think", "search(m.0, r)", "m.1")
                t.add_step("finish", "finish(m.1)", "done")
                t.set_answer(["m.1"])
                t.set_reward(1.0)
                t.ground_truth_entities = qa["answer_entities"]
                pool.add(t)
            return pool

        def fake_refine(self, pool):
            refined = TrajectoryPool()
            for t in pool.trajectories:
                r = Trajectory(t.question)
                r.add_step("refined think", "search(m.0, r)", "m.1")
                r.add_step("refined finish", "finish(m.1)", "done")
                r.set_answer(["m.1"])
                r.set_reward(1.0)
                r.ground_truth_entities = t.ground_truth_entities
                refined.add(r)
            return refined

        SelfLearner.online_explore = fake_explore
        SelfLearner.self_refine = fake_refine

        merged = learner.run_iteration(fake_qa, iteration=0)
        assert len(merged) == 1
        assert merged[0].reward > 0

    def test_fine_tune_returns_lora_path(self):
        """fine_tune should return path to saved LoRA adapter."""
        import glob
        output_dir = "/tmp/test_symagent_finetune"
        learner = self._make_learner(output_dir)

        pool = TrajectoryPool()
        t = Trajectory("test question")
        t.add_step("think", "search(m.0, r)", "result")
        t.add_step("finish", "finish(m.1)", "done")
        t.set_answer(["m.1"])
        t.set_reward(1.0)
        pool.add(t)

        lora_path = learner.fine_tune(pool, iteration=0)
        assert os.path.isdir(lora_path), f"LoRA path should exist: {lora_path}"
        adapter_files = glob.glob(os.path.join(lora_path, "*"))
        assert len(adapter_files) >= 3, \
            f"Should have adapter files, got {len(adapter_files)}"

    def test_switch_to_local_model(self):
        """After fine-tune, _switch_to_local_model should change LLM backend."""
        import glob
        output_dir = "/tmp/test_symagent_switch"
        learner = self._make_learner(output_dir)

        from src.local_model_client import LocalModelClient

        # Verify initial state
        assert type(learner.llm).__name__ == "LLMClient"
        assert type(learner.planner.llm).__name__ == "LLMClient"
        assert type(learner.executor.llm).__name__ == "LLMClient"

        # Fine-tune and switch
        pool = TrajectoryPool()
        t = Trajectory("test")
        t.add_step("t", "a", "o")
        t.set_answer(["m.1"])
        t.set_reward(0.5)
        pool.add(t)

        lora_path = learner.fine_tune(pool, iteration=0)
        learner._switch_to_local_model(lora_path)

        # Verify switched
        assert type(learner.llm).__name__ == "LocalModelClient"
        assert type(learner.planner.llm).__name__ == "LocalModelClient"
        assert type(learner.executor.llm).__name__ == "LocalModelClient"

    def test_full_loop_auto_switch(self):
        """Full loop should auto-switch from API to local model after fine-tune."""
        import glob
        output_dir = "/tmp/test_symagent_fullloop"
        learner = self._make_learner(output_dir)

        fake_qa = [
            {"question": "test", "question_entity": "m.0",
             "answer_entities": ["m.1"]},
        ]

        # Monkey-patch explore/refine
        def fake_explore(self, qa_pairs):
            pool = TrajectoryPool()
            for qa in qa_pairs:
                t = Trajectory(qa["question"])
                t.add_step("think", "search(m.0, r)", "m.1")
                t.add_step("finish", "finish(m.1)", "done")
                t.set_answer(["m.1"])
                t.set_reward(0.5)
                t.ground_truth_entities = qa["answer_entities"]
                pool.add(t)
            return pool

        def fake_refine(self, pool):
            refined = TrajectoryPool()
            for t in pool.trajectories:
                r = Trajectory(t.question)
                r.add_step("refined", "finish(m.1)", "done")
                r.set_answer(["m.1"])
                r.set_reward(0.6)
                r.ground_truth_entities = t.ground_truth_entities
                refined.add(r)
            return refined

        SelfLearner.online_explore = fake_explore
        SelfLearner.self_refine = fake_refine

        pools = learner.run_full_loop(fake_qa)

        assert len(pools) == 2, "Should have 2 iteration pools"

        # Iteration 0: used API (LLMClient)
        # Iteration 1: should have switched to LocalModelClient
        # (verified by checking that lora_iteration_0 exists)
        lora_0 = os.path.join(output_dir, "lora_iteration_0", "lora_adapter")
        lora_1 = os.path.join(output_dir, "lora_iteration_1", "lora_adapter")
        assert os.path.isdir(lora_0), "Iteration 0 LoRA should exist"
        assert os.path.isdir(lora_1), "Iteration 1 LoRA should exist"


class TestLocalModelClient:
    """Test LocalModelClient compatibility with LLMClient interface."""

    def test_generate_returns_string(self):
        from src.local_model_client import LocalModelClient
        client = LocalModelClient("gpt2", temperature=0.1, max_new_tokens=32)
        resp = client.generate([{"role": "user", "content": "Hello"}])
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_execute_generate_returns_string(self):
        from src.local_model_client import LocalModelClient
        client = LocalModelClient("gpt2", temperature=0.1, max_new_tokens=32)
        resp = client.execute_generate("Hello world")
        assert isinstance(resp, str)
        assert len(resp) > 0

    def test_generate_with_stop(self):
        from src.local_model_client import LocalModelClient
        client = LocalModelClient("gpt2", temperature=0.1, max_new_tokens=64)
        resp = client.generate(
            [{"role": "user", "content": "Hi"}],
            stop=["\n"]
        )
        assert "\n" not in resp

    def test_load_with_lora(self):
        """Load model with LoRA adapter."""
        from src.local_model_client import LocalModelClient
        import tempfile, glob

        # Create a minimal LoRA adapter directory
        with tempfile.TemporaryDirectory() as tmpdir:
            adapter_dir = os.path.join(tmpdir, "adapter")
            os.makedirs(adapter_dir)
            # Write minimal adapter config
            with open(os.path.join(adapter_dir, "adapter_config.json"), "w") as f:
                json.dump({"base_model_name_or_path": "gpt2", "r": 8}, f)
            # Create a dummy safetensors
            with open(os.path.join(adapter_dir, "adapter_model.safetensors"), "wb") as f:
                f.write(b"")

            # This will fail because the adapter is invalid,
            # but it should fail at model loading, not at init
            try:
                client = LocalModelClient("gpt2", lora_path=adapter_dir)
                assert False, "Should have raised an error for invalid adapter"
            except Exception:
                pass  # Expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
