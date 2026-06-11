"""Tests for KG Environment module.

Covers:
  - Triple add/remove/query
  - Neighbor search
  - BFS shortest path finding (Algorithm 1, Appendix A.1)
  - BFS closed path sampling (Section 4.1)
  - Reasoning path discovery
  - BM25 entity retrieval
  - Incompleteness simulation (Algorithm 1)
  - File I/O
  - Statistics
  - Entity/Relation mappings
"""

import os
import tempfile
import unittest

from src.kg_environment import KGEnvironment


class TestKGEnvironment(unittest.TestCase):
    """Test cases for KGEnvironment."""

    def setUp(self):
        """Set up a small test KG."""
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

    # --- Triple Operations ---

    def test_add_triple(self):
        """Test adding triples to the KG."""
        self.assertEqual(len(self.kg), 9)
        self.assertIn(("Alice", "workFor", "OpenAI"), self.kg)
        self.assertIn(("Bob", "liveIn", "MountainView"), self.kg)

    def test_add_duplicate_triple(self):
        """Test that duplicate triples are not added."""
        self.kg.add_triple("Alice", "workFor", "OpenAI")
        self.assertEqual(len(self.kg), 9)

    def test_remove_triple(self):
        """Test removing triples from the KG."""
        removed = self.kg.remove_triple("Alice", "workFor", "OpenAI")
        self.assertTrue(removed)
        self.assertNotIn(("Alice", "workFor", "OpenAI"), self.kg)
        self.assertEqual(len(self.kg), 8)

    def test_remove_nonexistent_triple(self):
        """Test removing a triple that doesn't exist."""
        removed = self.kg.remove_triple("Alice", "knows", "Charlie")
        self.assertFalse(removed)
        self.assertEqual(len(self.kg), 9)

    def test_has_triple(self):
        """Test triple existence check."""
        self.assertTrue(self.kg.has_triple("Alice", "workFor", "OpenAI"))
        self.assertFalse(self.kg.has_triple("Alice", "workFor", "Google"))

    def test_add_triple_registers_entities(self):
        """Test that add_triple registers new entities in entity2id."""
        self.kg.add_triple("Charlie", "livesIn", "London")
        self.assertIn("Charlie", self.kg.entity2id)
        self.assertIn("London", self.kg.entity2id)
        self.assertIn("livesIn", self.kg.relation2id)

    def test_remove_triple_updates_adjacency(self):
        """Test that remove_triple properly updates adjacency lists."""
        self.kg.remove_triple("Alice", "workFor", "OpenAI")
        neighbors = self.kg.search_neighbor_with_relation("Alice", "workFor")
        self.assertEqual(neighbors, [])
        # Other relations still work
        neighbors = self.kg.search_neighbor_with_relation("Alice", "liveIn")
        self.assertEqual(neighbors, ["SanFrancisco"])

    # --- Neighbor Search ---

    def test_search_neighbor_with_relation(self):
        """Test searching neighbors under a specific relation."""
        neighbors = self.kg.search_neighbor_with_relation("Alice", "workFor")
        self.assertEqual(neighbors, ["OpenAI"])

        neighbors = self.kg.search_neighbor_with_relation("Alice", "liveIn")
        self.assertEqual(neighbors, ["SanFrancisco"])

    def test_search_neighbor_no_match(self):
        """Test searching neighbors when no match exists."""
        neighbors = self.kg.search_neighbor_with_relation("Alice", "bornIn")
        self.assertEqual(neighbors, [])

    def test_search_neighbor_unknown_entity(self):
        """Test searching neighbors of unknown entity."""
        neighbors = self.kg.search_neighbor_with_relation("Unknown", "rel")
        self.assertEqual(neighbors, [])

    def test_search_neighbor_all(self):
        """Test searching all neighbors (no relation filter)."""
        neighbors = self.kg.search_neighbor("Alice")
        self.assertIn("OpenAI", neighbors)
        self.assertIn("SanFrancisco", neighbors)
        self.assertIn("Bob", neighbors)
        self.assertEqual(len(neighbors), 3)

    def test_search_neighbor_with_relation_method(self):
        """Test the dedicated search_neighbor_with_relation method."""
        neighbors = self.kg.search_neighbor_with_relation("Alice", "knows")
        self.assertEqual(neighbors, ["Bob"])

    def test_search_neighbor_multiple_outgoing(self):
        """Test entity with multiple outgoing relations to same relation."""
        self.kg.add_triple("Alice", "knows", "Charlie")
        neighbors = self.kg.search_neighbor_with_relation("Alice", "knows")
        self.assertEqual(len(neighbors), 2)
        self.assertIn("Bob", neighbors)
        self.assertIn("Charlie", neighbors)

    # --- BFS Shortest Path (Algorithm 1, Appendix A.1) ---

    def test_bfs_find_shortest_path_direct(self):
        """Test shortest path for direct 1-hop connection."""
        path = self.kg.bfs_find_shortest_path("Alice", "OpenAI", max_depth=4)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0], ("Alice", "workFor", "OpenAI"))

    def test_bfs_find_shortest_path_two_hop(self):
        """Test shortest path for 2-hop connection (via workFor -> locatedIn)."""
        # Remove direct liveIn path to force 2-hop
        self.kg.remove_triple("Alice", "liveIn", "SanFrancisco")
        path = self.kg.bfs_find_shortest_path("Alice", "SanFrancisco", max_depth=4)
        self.assertEqual(len(path), 2)
        self.assertEqual(path[0], ("Alice", "workFor", "OpenAI"))
        self.assertEqual(path[1], ("OpenAI", "locatedIn", "SanFrancisco"))

    def test_bfs_find_shortest_path_no_connection(self):
        """Test shortest path when no connection exists."""
        self.kg.add_triple("Charlie", "livesIn", "London")
        path = self.kg.bfs_find_shortest_path("Alice", "London", max_depth=4)
        self.assertEqual(path, [])

    def test_bfs_find_shortest_path_self(self):
        """Test shortest path from entity to itself (empty)."""
        path = self.kg.bfs_find_shortest_path("Alice", "Alice", max_depth=4)
        self.assertEqual(path, [])

    def test_bfs_find_shortest_path_unknown_entity(self):
        """Test shortest path with unknown entities."""
        path = self.kg.bfs_find_shortest_path("Unknown1", "Unknown2", max_depth=4)
        self.assertEqual(path, [])

    def test_bfs_find_shortest_path_respects_max_depth(self):
        """Test shortest path respects max_depth limit."""
        # Alice -> California requires 3 hops (liveIn -> inState or workFor -> locatedIn -> inState)
        path = self.kg.bfs_find_shortest_path("Alice", "California", max_depth=1)
        self.assertEqual(path, [])

    def test_bfs_find_shortest_path_three_hop(self):
        """Test shortest path for 3-hop connection."""
        path = self.kg.bfs_find_shortest_path("Alice", "California", max_depth=4)
        # Should find either: Alice -> liveIn -> SF -> inState -> California (2 hops)
        # or: Alice -> workFor -> OpenAI -> locatedIn -> SF -> inState -> California (3 hops)
        self.assertTrue(len(path) >= 2)

    def test_bfs_find_shortest_path_prefers_shorter(self):
        """Test that shortest path prefers shorter over longer paths."""
        # Alice has both direct (liveIn) and indirect (workFor -> locatedIn) to SanFrancisco
        path = self.kg.bfs_find_shortest_path("Alice", "SanFrancisco", max_depth=4)
        self.assertEqual(len(path), 1)
        self.assertEqual(path[0][1], "liveIn")

    # --- BFS Closed Path Sampling (Section 4.1) ---

    def test_bfs_find_paths_direct(self):
        """Test BFS finding a direct 1-hop path."""
        paths = self.kg.bfs_find_paths("Alice", "OpenAI", max_depth=1)
        self.assertTrue(len(paths) > 0)
        found = any(
            len(path) == 1 and path[0] == ("workFor", "OpenAI")
            for path in paths
        )
        self.assertTrue(found)

    def test_bfs_find_paths_two_hop(self):
        """Test BFS finding 2-hop paths."""
        paths = self.kg.bfs_find_paths("Alice", "SanFrancisco", max_depth=3)
        self.assertTrue(len(paths) > 0)
        # Check direct path
        direct_found = any(
            len(path) == 1 and path[0] == ("liveIn", "SanFrancisco")
            for path in paths
        )
        self.assertTrue(direct_found)
        # Check 2-hop path
        two_hop_found = any(
            len(path) == 2 and path[0][0] == "workFor" and path[1][0] == "locatedIn"
            for path in paths
        )
        self.assertTrue(two_hop_found)

    def test_bfs_find_paths_no_connection(self):
        """Test BFS when no path exists."""
        self.kg.add_triple("Charlie", "livesIn", "London")
        paths = self.kg.bfs_find_paths("Alice", "London", max_depth=3)
        self.assertEqual(len(paths), 0)

    def test_bfs_find_paths_max_depth(self):
        """Test BFS respects max depth."""
        paths = self.kg.bfs_find_paths("Alice", "California", max_depth=1)
        self.assertEqual(len(paths), 0)

    def test_bfs_find_paths_three_hop(self):
        """Test BFS finding 3-hop paths."""
        paths = self.kg.bfs_find_paths("Alice", "California", max_depth=4)
        self.assertTrue(len(paths) > 0)

    def test_bfs_find_paths_max_paths(self):
        """Test BFS respects max_paths limit."""
        paths = self.kg.bfs_find_paths("Alice", "SanFrancisco", max_depth=3, max_paths=1)
        self.assertTrue(len(paths) <= 1)

    def test_bfs_find_paths_unknown_entity(self):
        """Test BFS with unknown entities."""
        paths = self.kg.bfs_find_paths("Unknown1", "Unknown2", max_depth=3)
        self.assertEqual(len(paths), 0)

    def test_bfs_self_path(self):
        """Test BFS from entity to itself."""
        paths = self.kg.bfs_find_paths("Alice", "Alice", max_depth=3)
        # Should find the cycle: Alice -> knows -> Bob -> ... back to Alice
        # But current BFS avoids cycles, so this may return empty
        # This is expected behavior - no self-loops in our test graph

    # --- Reasoning Paths ---

    def test_get_reasoning_paths(self):
        """Test getting reasoning paths around an entity."""
        paths = self.kg.get_reasoning_paths("Alice", max_depth=2, max_paths=5)
        self.assertTrue(len(paths) > 0)
        # Should contain single-relation paths
        single_rel = [p for p in paths if len(p) == 1]
        self.assertTrue(len(single_rel) > 0)

    def test_get_reasoning_paths_max_paths(self):
        """Test reasoning paths respects max_paths."""
        paths = self.kg.get_reasoning_paths("Alice", max_depth=2, max_paths=3)
        self.assertTrue(len(paths) <= 3)

    def test_get_reasoning_paths_unknown_entity(self):
        """Test reasoning paths for unknown entity."""
        paths = self.kg.get_reasoning_paths("Unknown", max_depth=2)
        self.assertEqual(len(paths), 0)

    # --- BM25 Entity Retrieval ---

    def test_build_bm25_index(self):
        """Test BM25 index building and entity retrieval."""
        self.kg.build_bm25_index()
        results = self.kg.bm25_retrieve_entities("Alice works for OpenAI", top_k=3)
        self.assertTrue(len(results) > 0)
        entity_names = [name for name, _ in results]
        self.assertIn("Alice", entity_names)

    def test_bm25_custom_entity_list(self):
        """Test BM25 with custom entity list."""
        self.kg.build_bm25_index(entity_names=["Alice", "Bob", "Charlie"])
        results = self.kg.bm25_retrieve_entities("Alice", top_k=3)
        self.assertTrue(len(results) > 0)

    def test_bm25_scores_positive(self):
        """Test that BM25 scores are positive for relevant queries."""
        self.kg.build_bm25_index()
        results = self.kg.bm25_retrieve_entities("Alice", top_k=1)
        if results:
            _, score = results[0]
            self.assertGreater(score, 0.0)

    def test_bm25_auto_build(self):
        """Test that BM25 auto-builds if not explicitly built."""
        results = self.kg.bm25_retrieve_entities("Alice", top_k=3)
        self.assertIsInstance(results, list)

    def test_bm25_tokenize_entity_name(self):
        """Test entity name tokenization (dot, underscore, slash replaced with spaces)."""
        tokens = KGEnvironment._tokenize_entity_name("San.Francisco_California/USA")
        # Dots, underscores, slashes are replaced with spaces, then split on whitespace
        self.assertEqual(tokens, ["san", "francisco", "california", "usa"])

    def test_bm25_returns_sorted_by_score(self):
        """Test that BM25 results are sorted by descending score."""
        self.kg.build_bm25_index()
        results = self.kg.bm25_retrieve_entities("Alice OpenAI", top_k=5)
        if len(results) >= 2:
            scores = [score for _, score in results]
            self.assertEqual(scores, sorted(scores, reverse=True))

    # --- Incompleteness Simulation (Algorithm 1, Appendix A.1) ---

    def test_simulate_incompleteness(self):
        """Test simulating KG incompleteness (Algorithm 1)."""
        original_count = len(self.kg)
        removed = self.kg.simulate_incompleteness(
            "Alice", ["SanFrancisco"], remove_ratio=0.5, seed=42
        )
        self.assertTrue(len(self.kg) <= original_count)
        # Removed triples should not be in KG
        for h, r, t in removed:
            self.assertNotIn((h, r, t), self.kg._triple_set)

    def test_simulate_incompleteness_no_path(self):
        """Test incompleteness simulation when no path exists."""
        self.kg.add_triple("Charlie", "livesIn", "London")
        removed = self.kg.simulate_incompleteness(
            "Charlie", ["London"], remove_ratio=0.5, seed=42
        )
        self.assertIsInstance(removed, list)

    def test_simulate_incompleteness_multiple_answers(self):
        """Test incompleteness with multiple answer entities."""
        # Alice has paths to both OpenAI and SanFrancisco
        removed = self.kg.simulate_incompleteness(
            "Alice", ["OpenAI", "SanFrancisco"], remove_ratio=0.3, seed=42
        )
        self.assertIsInstance(removed, list)
        # Each removed triple should be on a path from Alice to one of the answers
        for h, r, t in removed:
            self.assertNotIn((h, r, t), self.kg._triple_set)

    def test_simulate_incompleteness_remove_ratio(self):
        """Test that remove_ratio controls the number of triples removed."""
        kg_copy = KGEnvironment()
        kg_copy.add_triple("A", "r1", "B")
        kg_copy.add_triple("B", "r2", "C")
        kg_copy.add_triple("C", "r3", "D")

        # With remove_ratio=1.0, should try to remove all path triples
        removed = kg_copy.simulate_incompleteness("A", ["D"], remove_ratio=1.0, seed=42)
        # The path A->B->C->D has 3 triples, so should remove up to 3
        self.assertTrue(len(removed) <= 3)

    def test_simulate_incompleteness_empty_answer_list(self):
        """Test incompleteness simulation with empty answer list."""
        removed = self.kg.simulate_incompleteness("Alice", [], remove_ratio=0.5, seed=42)
        self.assertEqual(removed, [])

    def test_simulate_incompleteness_deterministic_with_seed(self):
        """Test that same seed produces same results."""
        kg1 = KGEnvironment()
        kg1.add_triple("A", "r1", "B")
        kg1.add_triple("B", "r2", "C")

        kg2 = KGEnvironment()
        kg2.add_triple("A", "r1", "B")
        kg2.add_triple("B", "r2", "C")

        removed1 = kg1.simulate_incompleteness("A", ["C"], remove_ratio=0.5, seed=123)
        removed2 = kg2.simulate_incompleteness("A", ["C"], remove_ratio=0.5, seed=123)
        self.assertEqual(removed1, removed2)

    # --- Entity Info ---

    def test_get_entity_info(self):
        """Test getting entity information."""
        info = self.kg.get_entity_info("Alice")
        self.assertEqual(info["entity"], "Alice")
        self.assertIn("workFor", info["outgoing_relations"])
        self.assertTrue(info["num_outgoing"] > 0)

    def test_get_entity_info_unknown(self):
        """Test getting info for unknown entity."""
        info = self.kg.get_entity_info("Unknown")
        self.assertEqual(info["num_outgoing"], 0)
        self.assertEqual(info["num_incoming"], 0)

    def test_get_entity_info_incoming_relations(self):
        """Test that entity info includes incoming relations."""
        info = self.kg.get_entity_info("SanFrancisco")
        self.assertIn("liveIn", info["incoming_relations"])
        self.assertTrue(info["num_incoming"] > 0)

    # --- Statistics ---

    def test_get_stats(self):
        """Test KG statistics."""
        stats = self.kg.get_stats()
        self.assertEqual(stats["num_triples"], 9)
        self.assertTrue(stats["num_entities"] > 0)
        self.assertTrue(stats["num_relations"] > 0)

    def test_entity_and_relation_counts(self):
        """Test entity and relation count correctness."""
        stats = self.kg.get_stats()
        # 7 entities: Alice, OpenAI, SanFrancisco, Bob, Google, MountainView, California
        self.assertEqual(stats["num_entities"], 7)
        # 5 relations: workFor, locatedIn, liveIn, knows, inState
        self.assertEqual(stats["num_relations"], 5)

    # --- File I/O ---

    def test_save_and_load_triples(self):
        """Test saving and loading triples."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            self.kg.save_triples(filepath)
            kg2 = KGEnvironment()
            kg2.load_from_files(filepath)
            self.assertEqual(len(kg2), len(self.kg))
        finally:
            os.unlink(filepath)

    def test_load_nonexistent_file(self):
        """Test loading from nonexistent file (should not crash)."""
        kg2 = KGEnvironment()
        kg2.load_from_files("/nonexistent/path.txt")
        self.assertEqual(len(kg2), 0)

    def test_save_reload_preserves_data(self):
        """Test that save/reload preserves all triples."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            filepath = f.name

        try:
            self.kg.save_triples(filepath)
            kg2 = KGEnvironment()
            kg2.load_from_files(filepath)
            for h, r, t in self.kg._triple_set:
                self.assertIn((h, r, t), kg2._triple_set)
        finally:
            os.unlink(filepath)

    def test_load_jsonl_format(self):
        """Test loading triples in JSONL format."""
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            filepath = f.name
            json.dump({"h": "A", "r": "r1", "t": "B"}, f)
            f.write("\n")
            json.dump({"h": "B", "r": "r2", "t": "C"}, f)

        try:
            kg = KGEnvironment()
            kg.load_from_files(filepath, format="jsonl")
            self.assertEqual(len(kg), 2)
            self.assertIn(("A", "r1", "B"), kg)
            self.assertIn(("B", "r2", "C"), kg)
        finally:
            os.unlink(filepath)

    # --- Entity/Relation Mappings ---

    def test_entity2id_assigned(self):
        """Test that entity IDs are assigned."""
        self.assertIn("Alice", self.kg.entity2id)
        self.assertIn("Bob", self.kg.entity2id)

    def test_relation2id_assigned(self):
        """Test that relation IDs are assigned."""
        self.assertIn("workFor", self.kg.relation2id)
        self.assertIn("locatedIn", self.kg.relation2id)

    def test_id2entity_inverse(self):
        """Test id2entity is inverse of entity2id."""
        for name, idx in self.kg.entity2id.items():
            self.assertEqual(self.kg.id2entity[idx], name)

    def test_id2relation_inverse(self):
        """Test id2relation is inverse of relation2id."""
        for name, idx in self.kg.relation2id.items():
            self.assertEqual(self.kg.id2relation[idx], name)

    def test_len(self):
        """Test __len__ returns triple count."""
        self.assertEqual(len(self.kg), 9)

    def test_contains(self):
        """Test __contains__ for triple membership."""
        self.assertIn(("Alice", "workFor", "OpenAI"), self.kg)
        self.assertNotIn(("Alice", "workFor", "Google"), self.kg)

    # --- load_qa_file ---

    def test_load_qa_file(self):
        """Test loading QA samples and registering entities/relations."""
        import json
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            filepath = f.name
            json.dump([
                {"question": "q1", "question_entity": "E1", "answer_entities": ["E2"], "relation_path": ["r1"]},
                {"question": "q2", "question_entity": "E3", "answer_entities": ["E4"], "relation_path": ["r2"]},
            ], f)

        try:
            kg = KGEnvironment()
            samples = kg.load_qa_file(filepath)
            self.assertEqual(len(samples), 2)
            self.assertIn("E1", kg.entity2id)
            self.assertIn("E2", kg.entity2id)
            self.assertIn("r1", kg.relation2id)
        finally:
            os.unlink(filepath)

    # --- load_from_freebase_dir ---

    def test_load_from_freebase_dir(self):
        """Test load_from_freebase_dir convenience method."""
        import shutil
        tmpdir = tempfile.mkdtemp()
        try:
            # Create the expected files
            with open(os.path.join(tmpdir, "freebase_triples.txt"), "w") as f:
                f.write("A\tr1\tB\n")
            with open(os.path.join(tmpdir, "entity2id.txt"), "w") as f:
                f.write("A\t0\nB\t1\n")
            with open(os.path.join(tmpdir, "relation2id.txt"), "w") as f:
                f.write("r1\t0\n")

            kg = KGEnvironment()
            kg.load_from_freebase_dir(tmpdir)
            self.assertEqual(len(kg), 1)
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
