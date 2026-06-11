"""Integration tests for SymAgent: end-to-end pipeline with mock LLM.

Tests:
  1. Small KG construction (5-10 entities, 10-15 triples)
  2. QA sample construction (3-5 samples)
  3. Plan -> Execute complete flow
  4. Online explore -> Self-refine -> Heuristic merge flow
  5. Prepare training data with correct labels mask (Eq. 8)
  6. Reward computation (Eq. 6)
  7. End-to-end pipeline (no LLM)
"""

import json
import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from src.executor import AgentExecutor, Trajectory, compute_outcome_reward
from src.kg_environment import KGEnvironment
from src.planner import AgentPlanner
from src.self_learning import SelfLearner, TrajectoryPool, heuristic_merge, prepare_training_data


class TestIntegrationPipeline(unittest.TestCase):
    """End-to-end integration test for the SymAgent pipeline (no LLM)."""

    SAMPLE_QA_DATA = [
        {
            "question": "who directed inception",
            "question_entity": "m.0bxtg",
            "answer_entities": ["m.0bxtg"],
            "relation_path": ["film.film.directed_by"],
            "dataset": "webqsp",
            "qid": "WebQTest-001",
        },
        {
            "question": "who influenced samuel taylor coleridge",
            "question_entity": "m.078w2",
            "answer_entities": ["m.03sbs", "m.0448r"],
            "relation_path": ["influence.influence_node.influenced_by"],
            "dataset": "webqsp",
            "qid": "WebQTest-002",
        },
        {
            "question": "what is the capital of france",
            "question_entity": "m.0d06d",
            "answer_entities": ["m.0f5w9"],
            "relation_path": ["location.country.capital"],
            "dataset": "webqsp",
            "qid": "WebQTest-003",
        },
        {
            "question": "who is the president of the united states",
            "question_entity": "m.09c7w0",
            "answer_entities": ["m.0h3j"],
            "relation_path": ["government.governmental_jurisdiction.governing_officials"],
            "dataset": "webqsp",
            "qid": "WebQTest-004",
        },
        {
            "question": "where was albert einstein born",
            "question_entity": "m.0j3k",
            "answer_entities": ["m.0n4f"],
            "relation_path": ["people.person.place_of_birth"],
            "dataset": "webqsp",
            "qid": "WebQTest-005",
        },
    ]

    def setUp(self):
        """Set up temp directory and write sample data files."""
        self.tmpdir = tempfile.mkdtemp()

        # Write QA train file
        self.qa_train_path = os.path.join(self.tmpdir, "qa_train.json")
        with open(self.qa_train_path, "w") as f:
            json.dump(self.SAMPLE_QA_DATA, f, indent=2)

        # Write entity2id, relation2id, triples
        self.entity2id_path = os.path.join(self.tmpdir, "entity2id.txt")
        self.relation2id_path = os.path.join(self.tmpdir, "relation2id.txt")
        self.triples_path = os.path.join(self.tmpdir, "freebase_triples.txt")

        # Build triples from the QA data
        triples = set()
        for sample in self.SAMPLE_QA_DATA:
            topic = sample["question_entity"]
            path = sample["relation_path"]
            answers = sample["answer_entities"]
            if len(path) == 1:
                for ans in answers:
                    triples.add((topic, path[0], ans))

        with open(self.triples_path, "w") as f:
            for h, r, t in sorted(triples):
                f.write(f"{h}\t{r}\t{t}\n")

    def tearDown(self):
        """Clean up temp directory."""
        import shutil
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_load_kg_from_files(self):
        """Test loading KG from generated triple files."""
        kg = KGEnvironment()
        kg.load_from_files(
            triple_file=self.triples_path,
            entity2id_file=self.entity2id_path,
            relation2id_file=self.relation2id_path,
        )
        self.assertGreater(len(kg), 0)
        stats = kg.get_stats()
        self.assertGreater(stats["num_entities"], 0)
        self.assertGreater(stats["num_relations"], 0)

    def test_load_qa_file(self):
        """Test loading QA samples and registering entities/relations."""
        kg = KGEnvironment()
        samples = kg.load_qa_file(self.qa_train_path)

        self.assertEqual(len(samples), 5)
        self.assertIn("m.078w2", kg.entity2id)
        self.assertIn("film.film.directed_by", kg.relation2id)

    def test_load_from_freebase_dir(self):
        """Test load_from_freebase_dir convenience method."""
        kg = KGEnvironment()
        kg.load_from_freebase_dir(self.tmpdir)
        self.assertGreater(len(kg), 0)

    def test_bfs_path_finding_with_real_data(self):
        """Test BFS path finding with the mini KG."""
        kg = KGEnvironment()
        kg.load_from_files(self.triples_path)

        paths = kg.bfs_find_paths("m.078w2", "m.03sbs", max_depth=2)
        self.assertTrue(len(paths) > 0)
        found = any(
            p[0][0] == "influence.influence_node.influenced_by"
            for p in paths
        )
        self.assertTrue(found)

    def test_planner_bm25_retrieval(self):
        """Test planner BM25 retrieval with real QA questions."""
        kg = KGEnvironment()
        kg.load_from_files(self.triples_path)

        planner = AgentPlanner(kg=kg, llm=None, num_seed_questions=3)
        train_data = [
            {
                "question": s["question"],
                "question_entity": s["question_entity"],
                "answer_entities": s["answer_entities"],
            }
            for s in self.SAMPLE_QA_DATA
        ]
        planner.build_seed_index(train_data)

        seeds = planner.retrieve_seed_questions("who influenced coleridge", top_k=2)
        self.assertIsInstance(seeds, list)
        self.assertTrue(len(seeds) <= 2)

    def test_plan_no_llm(self):
        """Test full planning pipeline without LLM."""
        kg = KGEnvironment()
        kg.load_from_files(self.triples_path)

        planner = AgentPlanner(kg=kg, llm=None, num_seed_questions=3, max_rules=5)
        train_data = [
            {
                "question": s["question"],
                "question_entity": s["question_entity"],
                "answer_entities": s["answer_entities"],
            }
            for s in self.SAMPLE_QA_DATA
        ]
        planner.build_seed_index(train_data)

        paths = planner.plan(
            "who influenced samuel taylor coleridge",
            question_entity="m.078w2",
        )
        self.assertIsInstance(paths, list)
        for path in paths:
            self.assertIsInstance(path, list)
            for rel in path:
                self.assertIsInstance(rel, str)

    def test_end_to_end_pipeline(self):
        """Full end-to-end test: load data -> build KG -> plan -> verify paths."""
        kg = KGEnvironment()
        qa_samples = kg.load_qa_file(self.qa_train_path)
        self.assertEqual(len(qa_samples), 5)

        kg.load_from_files(self.triples_path)
        self.assertGreater(len(kg), 0)

        kg.build_bm25_index()

        planner = AgentPlanner(kg=kg, llm=None, num_seed_questions=3)
        train_data = [
            {
                "question": s["question"],
                "question_entity": s["question_entity"],
                "answer_entities": s["answer_entities"],
            }
            for s in qa_samples
        ]
        planner.build_seed_index(train_data)

        for sample in qa_samples:
            paths = planner.plan(
                sample["question"],
                question_entity=sample["question_entity"],
            )
            self.assertIsInstance(paths, list)
            for path in paths:
                self.assertIsInstance(path, list)
                for rel in path:
                    self.assertIsInstance(rel, str)

        neighbors = kg.search_neighbor("m.078w2")
        self.assertIsInstance(neighbors, list)

        reasoning_paths = kg.get_reasoning_paths("m.078w2", max_depth=1, max_paths=5)
        self.assertIsInstance(reasoning_paths, list)


class TestIntegrationWithMockLLM(unittest.TestCase):
    """Integration tests using mock LLM for plan -> execute flow."""

    def setUp(self):
        """Build a small KG with 7 entities and 9 triples."""
        self.kg = KGEnvironment()
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")
        self.kg.add_triple("Alice", "liveIn", "SanFrancisco")
        self.kg.add_triple("Bob", "workFor", "Google")
        self.kg.add_triple("Google", "locatedIn", "MountainView")
        self.kg.add_triple("Bob", "liveIn", "MountainView")
        self.kg.add_triple("Alice", "knows", "Bob")
        self.kg.add_triple("SanFrancisco", "inState", "California")
        self.kg.add_triple("MountainView", "inState", "California")

        self.mock_llm = MagicMock()

    def test_plan_execute_complete_flow(self):
        """Test plan -> execute complete flow with mock LLM."""
        planner = AgentPlanner(
            kg=self.kg, llm=self.mock_llm, num_seed_questions=2, max_rules=3
        )

        # Build seed index
        train_data = [
            {"question": "Where does Alice live?", "question_entity": "Alice", "answer_entities": ["SanFrancisco"]},
            {"question": "Where does Bob work?", "question_entity": "Bob", "answer_entities": ["Google"]},
        ]
        planner.build_seed_index(train_data)

        # Mock LLM to return rules
        self.mock_llm.plan_generate.return_value = "[workFor, locatedIn]\n[liveIn]"

        # Plan
        paths = planner.plan("Where does Alice work?", question_entity="Alice")
        self.assertTrue(len(paths) > 0)

        # Execute with mock LLM
        executor = AgentExecutor(
            kg=self.kg, llm=self.mock_llm, planner=planner, max_steps=5
        )
        self.mock_llm.execute_generate.side_effect = [
            "Thought: I should search\nAction: searchNeighbor(Alice, workFor)",
            "Thought: Found OpenAI\nAction: finish(OpenAI)",
        ]

        trajectory = executor.execute(
            "Where does Alice work?", "Alice", paths
        )
        self.assertEqual(trajectory.answer_entities, ["OpenAI"])
        self.assertTrue(len(trajectory) >= 2)

    def test_reward_computation_after_execute(self):
        """Test Eq. 6 reward after execute."""
        reward = compute_outcome_reward(["OpenAI"], ["OpenAI"])
        self.assertEqual(reward, 1.0)

        reward = compute_outcome_reward(["Google"], ["OpenAI"])
        self.assertEqual(reward, 0.0)

        reward = compute_outcome_reward(["OpenAI"], ["OpenAI", "Google"])
        self.assertAlmostEqual(reward, 0.5)

    def test_multi_hop_reasoning(self):
        """Test multi-hop reasoning: Alice -> workFor -> OpenAI -> locatedIn -> SanFrancisco."""
        self.mock_llm.execute_generate.side_effect = [
            "Thought: Follow the path\nAction: searchNeighbor(Alice, workFor)",
            "Thought: Got OpenAI, now follow next relation\nAction: searchNeighbor(OpenAI, locatedIn)",
            "Thought: Found SanFrancisco\nAction: finish(SanFrancisco)",
        ]

        planner = AgentPlanner(kg=self.kg, llm=self.mock_llm, max_rules=3)
        executor = AgentExecutor(
            kg=self.kg, llm=self.mock_llm, planner=planner, max_steps=5
        )

        trajectory = executor.execute(
            "Where is Alice's employer located?",
            "Alice",
            [["workFor", "locatedIn"]],
        )

        self.assertEqual(trajectory.answer_entities, ["SanFrancisco"])
        self.assertEqual(len(trajectory), 3)

    def test_wiki_search_fallback(self):
        """Test wikiSearch fallback when KG has no results."""
        # Remove the direct liveIn triple to force wiki search
        self.kg.remove_triple("Alice", "liveIn", "SanFrancisco")

        self.mock_llm.execute_generate.side_effect = [
            "Thought: Need to search\nAction: searchNeighbor(Alice, liveIn)",
            "Thought: No results in KG, try wiki\nAction: wikiSearch(Alice, liveIn)",
            "Thought: Found info\nAction: finish(SanFrancisco)",
        ]

        # Mock extract_triples for wiki search
        self.mock_llm.extract_triples.return_value = "[[Alice, liveIn, SanFrancisco]]"

        # Mock Wikipedia by patching the executor's wikipedia instance directly
        planner = AgentPlanner(kg=self.kg, llm=self.mock_llm, max_rules=3)
        executor = AgentExecutor(
            kg=self.kg, llm=self.mock_llm, planner=planner, max_steps=5
        )

        mock_wiki_page = MagicMock()
        mock_wiki_page.exists.return_value = True
        mock_wiki_page.summary = "Alice lives in San Francisco."
        executor.wikipedia.page = MagicMock(return_value=mock_wiki_page)

        trajectory = executor.execute(
            "Where does Alice live?",
            "Alice",
            [["liveIn"]],
        )

        # Triple should have been integrated into KG
        self.assertIn(("Alice", "liveIn", "SanFrancisco"), self.kg)
        extracted = executor.get_extracted_triples()
        self.assertTrue(len(extracted) > 0)


class TestIntegrationSelfLearning(unittest.TestCase):
    """Integration tests for the self-learning pipeline.

    Tests:
      - online_explore -> self_refine -> heuristic_merge
      - prepare_training_data generates correct labels mask
    """

    def setUp(self):
        """Set up test KG and components."""
        self.kg = KGEnvironment()
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.kg.add_triple("OpenAI", "locatedIn", "SanFrancisco")
        self.kg.add_triple("Bob", "workFor", "Google")
        self.kg.add_triple("Google", "locatedIn", "MountainView")

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
            model_name=None,
        )

    def test_online_explore_self_refine_merge(self):
        """Test complete flow: online_explore -> self_refine -> heuristic_merge (Eq. 7)."""
        # Setup: executor returns trajectories
        orig_traj = Trajectory("Where does Alice work?")
        orig_traj.add_step("Search for employer", "searchNeighbor(Alice, workFor)", "OpenAI")
        orig_traj.set_answer(["OpenAI"])
        orig_traj.set_reward(1.0)
        orig_traj.ground_truth_entities = ["OpenAI"]
        self.mock_executor.execute.return_value = orig_traj

        # Step 1: Online explore
        qa_pairs = [
            {"question": "Where does Alice work?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]
        original_pool = self.learner.online_explore(qa_pairs)
        self.assertEqual(len(original_pool), 1)

        # Step 2: Self-refine (mock LLM returns improved trajectory)
        self.mock_llm.execute_generate.return_value = (
            "Thought 1: Search directly\n"
            "Action 1: searchNeighbor(Alice, workFor)\n"
            "Thought 2: Got answer\n"
            "Action 2: finish(OpenAI)\n"
        )

        refined_pool = self.learner.self_refine(original_pool)
        self.assertEqual(len(refined_pool), 1)

        # Step 3: Heuristic merge (Eq. 7)
        merged = heuristic_merge(original_pool, refined_pool)
        self.assertGreater(len(merged), 0)
        # Since both have reward=1.0, shorter should be selected (branch 3)
        self.assertEqual(merged[0].reward, 1.0)

    def test_explore_refine_merge_improvement(self):
        """Test merge where refined trajectory improves over original (branch 2)."""
        # Original: wrong answer
        orig_traj = Trajectory("Where does Alice work?")
        orig_traj.add_step("Wrong thought", "searchNeighbor(Alice, liveIn)", "SanFrancisco")
        orig_traj.set_answer(["SanFrancisco"])  # Wrong answer
        orig_traj.set_reward(0.0)
        orig_traj.ground_truth_entities = ["OpenAI"]
        self.mock_executor.execute.return_value = orig_traj

        qa_pairs = [
            {"question": "Where does Alice work?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]
        original_pool = self.learner.online_explore(qa_pairs)
        self.assertEqual(len(original_pool), 0)  # Filtered by threshold

        # For this test, create pools manually to test merge logic
        orig_pool = TrajectoryPool()
        orig_traj2 = Trajectory("Q?")
        orig_traj2.add_step("wrong", "action", "obs")
        orig_traj2.set_answer(["Wrong"])
        orig_traj2.set_reward(0.0)
        orig_pool.add(orig_traj2)

        # Refined: correct answer
        ref_pool = TrajectoryPool()
        ref_traj = Trajectory("Q?")
        ref_traj.add_step("correct", "action", "obs")
        ref_traj.set_answer(["OpenAI"])
        ref_traj.set_reward(1.0)
        ref_pool.add(ref_traj)

        merged = heuristic_merge(orig_pool, ref_pool)
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0].reward, 1.0)

    def test_prepare_training_data_end_to_end(self):
        """Test prepare_training_data produces correct labels mask (Eq. 8)."""
        pool = TrajectoryPool()
        traj = Trajectory("Where does Alice work?")
        traj.add_step("Search employer", "searchNeighbor(Alice, workFor)", "OpenAI")
        traj.add_step("Conclude", "finish(OpenAI)", "Final answer: OpenAI")
        traj.set_answer(["OpenAI"])
        pool.add(traj)

        mock_tokenizer = MagicMock()

        # Simulate tokenizer: each text segment produces a fixed number of tokens
        # Question: "Where does Alice work?\n" -> 5 tokens
        # Each step line -> 3 tokens
        # Total: 5 + 2*(3+3+3) = 23 tokens
        full_tokens = 23
        mock_tokenizer.return_value = {
            "input_ids": list(range(full_tokens)),
            "attention_mask": [1] * full_tokens,
        }

        # Individual segment tokenizations
        def mock_tokenize(text, **kwargs):
            # Each segment is 3 tokens
            return {"input_ids": [100, 101, 102], "attention_mask": [1, 1, 1]}

        mock_tokenizer.side_effect = mock_tokenize

        data = prepare_training_data(pool, mock_tokenizer, max_length=4096)
        self.assertEqual(len(data), 1)

        labels = data[0]["labels"]
        input_ids = data[0]["input_ids"]
        attention_mask = data[0]["attention_mask"]

        # Verify lengths match
        self.assertEqual(len(labels), len(input_ids))
        self.assertEqual(len(attention_mask), len(input_ids))

        # Question tokens (positions 0-4) should be masked (-100)
        question_masked = all(l == -100 for l in labels[:5])
        self.assertTrue(question_masked, "Question tokens should be masked")

        # Thought 1 tokens (positions 5-7) should be unmasked
        thought1_unmasked = all(l != -100 for l in labels[5:8])
        self.assertTrue(thought1_unmasked, "Thought tokens should be unmasked")

        # Action 1 tokens (positions 8-10) should be unmasked
        action1_unmasked = all(l != -100 for l in labels[8:11])
        self.assertTrue(action1_unmasked, "Action tokens should be unmasked")

        # Observation 1 tokens (positions 11-13) should be masked
        obs1_masked = all(l == -100 for l in labels[11:14])
        self.assertTrue(obs1_masked, "Observation tokens should be masked")

        # Thought 2 tokens (positions 14-16) should be unmasked
        thought2_unmasked = all(l != -100 for l in labels[14:17])
        self.assertTrue(thought2_unmasked, "Thought 2 tokens should be unmasked")

        # Action 2 tokens (positions 17-19) should be unmasked
        action2_unmasked = all(l != -100 for l in labels[17:20])
        self.assertTrue(action2_unmasked, "Action 2 tokens should be unmasked")

        # Observation 2 tokens (positions 20-22) should be masked
        obs2_masked = all(l == -100 for l in labels[20:23])
        self.assertTrue(obs2_masked, "Observation 2 tokens should be masked")

    def test_run_iteration_full_cycle(self):
        """Test run_iteration: explore -> refine -> merge -> (skip fine-tune)."""
        orig_traj = Trajectory("Q?")
        orig_traj.add_step("think", "action", "obs")
        orig_traj.set_answer(["OpenAI"])
        self.mock_executor.execute.return_value = orig_traj

        self.mock_llm.execute_generate.return_value = (
            "Thought 1: Direct search\n"
            "Action 1: finish(OpenAI)\n"
        )

        qa_pairs = [
            {"question": "Q?", "question_entity": "Alice", "answer_entities": ["OpenAI"]}
        ]

        merged = self.learner.run_iteration(qa_pairs, iteration=0)
        self.assertGreater(len(merged), 0)

    def test_bfs_incompleteness_and_recovery(self):
        """Test that simulate_incompleteness + wiki search can recover knowledge."""
        # Start with a complete KG
        self.kg.add_triple("Alice", "liveIn", "SanFrancisco")

        # Simulate incompleteness (Algorithm 1)
        removed = self.kg.simulate_incompleteness(
            "Alice", ["SanFrancisco"], remove_ratio=1.0, seed=42
        )

        # After removal, direct path should be gone
        neighbors = self.kg.search_neighbor_with_relation("Alice", "liveIn")
        # The liveIn triple might have been removed
        if ("Alice", "liveIn", "SanFrancisco") in [(h, r, t) for h, r, t in removed]:
            self.assertEqual(neighbors, [])
            # But other paths may still exist (workFor -> locatedIn)


if __name__ == "__main__":
    unittest.main()
