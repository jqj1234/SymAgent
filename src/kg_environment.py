"""
Knowledge Graph Environment for SymAgent.

Implements the KG as a dynamic environment (POMDP formulation) supporting:
- Entity/relation lookup and indexing
- BFS path sampling between query entity and answer entity
- Neighbor search with relation paths
- BM25 entity retrieval from questions
- Triple management (add/remove for incompleteness simulation)
"""

import json
import logging
import os
from collections import defaultdict, deque
from typing import Any, Optional

import networkx as nx
from rank_bm25 import BM25Okapi

logger = logging.getLogger(__name__)


class KGEnvironment:
    """Knowledge Graph Environment for SymAgent.

    The KG serves as a dynamic environment (POMDP) providing execution
    feedback rather than merely acting as a static knowledge repository.
    Supports Freebase and Wikidata style triple stores.

    Attributes:
        graph: NetworkX directed multigraph storing KG triples.
        entity2id: Mapping from entity name to integer ID.
        id2entity: Mapping from integer ID to entity name.
        relation2id: Mapping from relation name to integer ID.
        id2relation: Mapping from integer ID to relation name.
    """

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity2id: dict[str, int] = {}
        self.id2entity: dict[int, str] = {}
        self.relation2id: dict[str, int] = {}
        self.id2relation: dict[int, str] = {}
        self._adj_out: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._adj_in: dict[str, dict[str, list[str]]] = defaultdict(
            lambda: defaultdict(list)
        )
        self._bm25: Optional[BM25Okapi] = None
        self._bm25_entities: list[str] = []
        self._triple_set: set[tuple[str, str, str]] = set()

    def load_from_files(
        self,
        triple_file: str,
        entity2id_file: Optional[str] = None,
        relation2id_file: Optional[str] = None,
        format: str = "hrt",
    ) -> None:
        """Load KG from files.

        Args:
            triple_file: Path to triples file. Each line: head\\trelation\\ttail.
            entity2id_file: Optional path to entity-to-ID mapping.
            relation2id_file: Optional path to relation-to-ID mapping.
            format: Triple format - "hrt" (head relation tail tab-separated),
                    "jsonl" (JSON lines with h/r/t keys).
        """
        # Load entity and relation mappings if provided
        if entity2id_file and os.path.exists(entity2id_file):
            self._load_mapping(entity2id_file, self.entity2id, self.id2entity)
            logger.info(f"Loaded {len(self.entity2id)} entities")

        if relation2id_file and os.path.exists(relation2id_file):
            self._load_mapping(relation2id_file, self.relation2id, self.id2relation)
            logger.info(f"Loaded {len(self.relation2id)} relations")

        # Load triples
        if not os.path.exists(triple_file):
            logger.warning(f"Triple file not found: {triple_file}")
            return

        count = 0
        with open(triple_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if format == "hrt":
                    parts = line.split("\t")
                    if len(parts) >= 3:
                        h, r, t = parts[0], parts[1], parts[2]
                    else:
                        continue
                elif format == "jsonl":
                    data = json.loads(line)
                    h, r, t = data["h"], data["r"], data["t"]
                else:
                    continue

                self.add_triple(h, r, t)
                count += 1

        logger.info(f"Loaded {count} triples into KG environment")

    def load_from_freebase_dir(self, freebase_dir: str) -> None:
        """Load KG from a Freebase data directory built by build_kg_from_datasets.py.

        Expects:
          - freebase_dir/entity2id.txt
          - freebase_dir/relation2id.txt
          - freebase_dir/freebase_triples.txt

        Args:
            freebase_dir: Path to the Freebase data directory.
        """
        self.load_from_files(
            triple_file=os.path.join(freebase_dir, "freebase_triples.txt"),
            entity2id_file=os.path.join(freebase_dir, "entity2id.txt"),
            relation2id_file=os.path.join(freebase_dir, "relation2id.txt"),
        )

    def load_qa_file(self, qa_file: str) -> list[dict[str, Any]]:
        """Load unified QA samples from a JSON file.

        Expects list of dicts with keys:
          question, question_entity, answer_entities, relation_path, dataset, qid

        Registers all entities and relations from the QA samples into the KG
        index (without adding triples, since relation paths may have unknown
        intermediate entities).

        Args:
            qa_file: Path to a qa_train.json or qa_test.json file.

        Returns:
            List of QA sample dicts.
        """
        with open(qa_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        for sample in data:
            ent = sample.get("question_entity", "")
            if ent and ent not in self.entity2id:
                idx = len(self.entity2id)
                self.entity2id[ent] = idx
                self.id2entity[idx] = ent

            for ans in sample.get("answer_entities", []):
                if ans and ans not in self.entity2id:
                    idx = len(self.entity2id)
                    self.entity2id[ans] = idx
                    self.id2entity[idx] = ans

            for rel in sample.get("relation_path", []):
                if rel and rel not in self.relation2id:
                    idx = len(self.relation2id)
                    self.relation2id[rel] = idx
                    self.id2relation[idx] = rel

        logger.info(
            f"Loaded {len(data)} QA samples from {qa_file}, "
            f"registered {len(self.entity2id)} entities, "
            f"{len(self.relation2id)} relations"
        )
        return data

    def _load_mapping(
        self,
        filepath: str,
        name2id: dict[str, int],
        id2name: dict[int, str],
    ) -> None:
        """Load a name-to-ID mapping file."""
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t")
                if len(parts) >= 2:
                    name = parts[0]
                    idx = int(parts[1])
                    name2id[name] = idx
                    id2name[idx] = name

    def add_triple(self, head: str, relation: str, tail: str) -> None:
        """Add a triple to the KG.

        Args:
            head: Head entity name.
            relation: Relation name.
            tail: Tail entity name.
        """
        triple = (head, relation, tail)
        if triple in self._triple_set:
            return
        self._triple_set.add(triple)

        self.graph.add_edge(head, tail, relation=relation)

        # Update adjacency index for fast neighbor lookup
        self._adj_out[head][relation].append(tail)
        self._adj_in[tail][relation].append(head)

        # Register entities/relations if not already known
        for ent in (head, tail):
            if ent not in self.entity2id:
                idx = len(self.entity2id)
                self.entity2id[ent] = idx
                self.id2entity[idx] = ent
        if relation not in self.relation2id:
            idx = len(self.relation2id)
            self.relation2id[relation] = idx
            self.id2relation[idx] = relation

    def remove_triple(self, head: str, relation: str, tail: str) -> bool:
        """Remove a triple from the KG.

        Used to simulate KG incompleteness (Appendix A.1).

        Returns:
            True if the triple was removed, False if it didn't exist.
        """
        triple = (head, relation, tail)
        if triple not in self._triple_set:
            return False

        self._triple_set.discard(triple)

        # Remove from adjacency index
        if head in self._adj_out and relation in self._adj_out[head]:
            if tail in self._adj_out[head][relation]:
                self._adj_out[head][relation].remove(tail)
        if tail in self._adj_in and relation in self._adj_in[tail]:
            if head in self._adj_in[tail][relation]:
                self._adj_in[tail][relation].remove(head)

        # Remove from NetworkX graph
        if self.graph.has_edge(head, tail):
            edges_to_remove = []
            for key, data in self.graph[head][tail].items():
                if data.get("relation") == relation:
                    edges_to_remove.append(key)
            for key in edges_to_remove:
                self.graph.remove_edge(head, tail, key)

        return True

    def has_triple(self, head: str, relation: str, tail: str) -> bool:
        """Check if a triple exists in the KG."""
        return (head, relation, tail) in self._triple_set

    def search_neighbor(
        self,
        entity: str,
        relation: Optional[str] = None,
    ) -> list[str]:
        """Search neighbors of an entity in the KG.

        Implements the searchNeighbor(entity, relation) action tool.

        Args:
            entity: The entity to search neighbors for.
            relation: Optional relation filter. If None, returns all neighbors.

        Returns:
            List of neighbor entity names.
        """
        if entity not in self._adj_out:
            return []
        if relation is not None:
            return list(self._adj_out[entity].get(relation, []))
        # Return all neighbors across all relations
        neighbors = []
        for rel, tails in self._adj_out[entity].items():
            neighbors.extend(tails)
        return neighbors

    def search_neighbor_with_relation(
        self,
        entity: str,
        relation: str,
    ) -> list[str]:
        """Search neighbors of entity under a specific relation.

        Returns a formatted observation string as the KG environment response.
        Used by the executor's searchNeighbor action.

        Args:
            entity: The query entity.
            relation: The relation to follow.

        Returns:
            List of tail entities connected via the given relation.
        """
        neighbors = self.search_neighbor(entity, relation)
        return neighbors

    def bfs_find_shortest_path(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
    ) -> list[tuple[str, str, str]]:
        """BFS to find the shortest path from source to target.

        Returns the triples along the shortest path (not all paths).
        Used by simulate_incompleteness (Algorithm 1).

        Args:
            source: Source entity.
            target: Target entity.
            max_depth: Maximum search depth.

        Returns:
            List of (head, relation, tail) triples forming the shortest path.
            Empty list if no path exists.
        """
        if source not in self.entity2id or target not in self.entity2id:
            return []
        if source == target:
            return []

        # BFS queue: (current_entity, path_of_triples)
        queue: deque[tuple[str, list[tuple[str, str, str]]]] = deque()
        queue.append((source, []))
        visited: set[str] = {source}

        while queue:
            current, path_triples = queue.popleft()

            if len(path_triples) >= max_depth:
                continue

            if current not in self._adj_out:
                continue

            for relation, tails in self._adj_out[current].items():
                for tail in tails:
                    triple = (current, relation, tail)
                    new_path = path_triples + [triple]

                    if tail == target:
                        return new_path

                    if tail not in visited:
                        visited.add(tail)
                        queue.append((tail, new_path))

        return []

    def bfs_find_paths(
        self,
        source: str,
        target: str,
        max_depth: int = 4,
        max_paths: int = 5,
    ) -> list[list[tuple[str, str]]]:
        """BFS to find closed paths from source to target in the KG.

        Used by the Agent-Planner for symbolic rule induction.
        For each seed question, BFS samples closed paths from query entity
        to answer entity within the KG.

        Args:
            source: Source entity (query entity).
            target: Target entity (answer entity).
            max_depth: Maximum path length.
            max_paths: Maximum number of paths to return.

        Returns:
            List of paths, where each path is a list of (relation, tail_entity)
            tuples forming a chain from source to target.
        """
        if source not in self.entity2id or target not in self.entity2id:
            return []

        paths: list[list[tuple[str, str]]] = []
        # BFS queue: (current_entity, path_so_far, visited_nodes_on_path)
        queue: deque[tuple[str, list[tuple[str, str]], set[str]]] = deque()
        queue.append((source, [], {source}))
        visited_edges: set[tuple[str, str, str]] = set()

        while queue and len(paths) < max_paths:
            level_size = len(queue)
            for _ in range(level_size):
                if len(paths) >= max_paths:
                    break
                current, path, visited_nodes = queue.popleft()

                if len(path) >= max_depth:
                    continue

                if current not in self._adj_out:
                    continue

                for relation, tails in self._adj_out[current].items():
                    for tail in tails:
                        edge_key = (current, relation, tail)
                        if edge_key in visited_edges:
                            continue
                        visited_edges.add(edge_key)

                        new_path = path + [(relation, tail)]

                        if tail == target:
                            paths.append(new_path)
                            if len(paths) >= max_paths:
                                break
                        elif tail not in visited_nodes:
                            new_visited = visited_nodes | {tail}
                            queue.append((tail, new_path, new_visited))

        return paths

    def get_reasoning_paths(
        self,
        entity: str,
        max_depth: int = 2,
        max_paths: int = 10,
    ) -> list[list[str]]:
        """Get potential relational reasoning paths around an entity.

        Used by getReasoningPath action to discover surrounding relational
        patterns for question decomposition.

        Args:
            entity: The entity to explore around.
            max_depth: Maximum depth of reasoning paths.
            max_paths: Maximum number of paths to return.

        Returns:
            List of relation paths, where each path is a list of relation names.
        """
        paths: list[list[str]] = []
        # BFS to collect relation paths
        queue: deque[tuple[str, list[str]]] = deque()
        queue.append((entity, []))

        while queue and len(paths) < max_paths:
            current, rel_path = queue.popleft()

            if len(rel_path) >= max_depth:
                continue

            if current not in self._adj_out:
                continue

            for relation, tails in self._adj_out[current].items():
                for tail in tails[:3]:  # Limit branching
                    new_path = rel_path + [relation]
                    paths.append(new_path)
                    if len(paths) >= max_paths:
                        return paths
                    if len(new_path) < max_depth:
                        queue.append((tail, new_path))

        return paths

    def build_bm25_index(self, entity_names: Optional[list[str]] = None) -> None:
        """Build BM25 index for entity retrieval from questions.

        Args:
            entity_names: Optional list of entity names to index.
                         If None, indexes all entities in the KG.
        """
        if entity_names is not None:
            self._bm25_entities = entity_names
        else:
            self._bm25_entities = list(self.entity2id.keys())

        # Tokenize entity names for BM25
        tokenized_corpus = [
            self._tokenize_entity_name(name) for name in self._bm25_entities
        ]
        self._bm25 = BM25Okapi(tokenized_corpus)
        logger.info(
            f"Built BM25 index over {len(self._bm25_entities)} entities"
        )

    def bm25_retrieve_entities(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[tuple[str, float]]:
        """Retrieve entities from the KG using BM25 matching against a query.

        Args:
            query: Natural language query string.
            top_k: Number of top entities to return.

        Returns:
            List of (entity_name, score) tuples sorted by relevance.
        """
        if self._bm25 is None:
            self.build_bm25_index()

        tokenized_query = query.lower().split()
        scores = self._bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]

        return [
            (self._bm25_entities[i], float(scores[i]))
            for i in top_indices
            if scores[i] > 0
        ]

    @staticmethod
    def _tokenize_entity_name(name: str) -> list[str]:
        """Tokenize an entity name for BM25 indexing."""
        # Split on common delimiters and lowercase
        tokens = name.replace(".", " ").replace("_", " ").replace("/", " ")
        return tokens.lower().split()

    def load_name_mapping(self, mid2name_file: str, name2mid_file: Optional[str] = None) -> None:
        """Load MID-to-name and name-to-MID mappings for entity resolution.

        These mappings enable compute_outcome_reward to match entities
        across formats (e.g., 'Bill Cassidy' <-> 'm.0xyz').

        Args:
            mid2name_file: Path to mid2name.json.
            name2mid_file: Path to name2mid.json (optional, derived from mid2name if not provided).
        """
        import json
        with open(mid2name_file, 'r', encoding='utf-8') as f:
            self._mid2name = json.load(f)

        if name2mid_file and os.path.exists(name2mid_file):
            with open(name2mid_file, 'r', encoding='utf-8') as f:
                self._name2mid = json.load(f)
        else:
            # Derive name2mid from mid2name
            self._name2mid = {v.lower(): k for k, v in self._mid2name.items() if v}

        logger.info(
            f"Loaded name mappings: {len(self._mid2name)} mid2name, "
            f"{len(self._name2mid)} name2mid"
        )

    def get_entity_info(self, entity: str) -> dict[str, Any]:
        """Get information about an entity's connections in the KG.

        Args:
            entity: Entity name.

        Returns:
            Dict with outgoing_relations, incoming_relations, and counts.
        """
        out_rels = {}
        if entity in self._adj_out:
            for rel, tails in self._adj_out[entity].items():
                out_rels[rel] = list(tails)

        in_rels = {}
        if entity in self._adj_in:
            for rel, heads in self._adj_in[entity].items():
                in_rels[rel] = list(heads)

        return {
            "entity": entity,
            "outgoing_relations": out_rels,
            "incoming_relations": in_rels,
            "num_outgoing": sum(len(v) for v in out_rels.values()),
            "num_incoming": sum(len(v) for v in in_rels.values()),
        }

    def simulate_incompleteness(
        self,
        query_entity: str,
        answer_entities: list[str],
        remove_ratio: float = 0.3,
        seed: int = 42,
    ) -> list[tuple[str, str, str]]:
        """Simulate KG incompleteness by removing path triples.

        Follows Algorithm 1 from Appendix A.1 exactly:
        1. Initialize L <- []
        2. For each a_ent in a_ent_list:
           path <- BFS_find_shortest_path(G, q_ent, a_ent)
           L.extend(path)
        3. selected_triples <- random_select(L)
        4. For each t in selected_triples: G.remove(t)

        Args:
            query_entity: The question entity (q_ent).
            answer_entities: List of answer entities (a_ent_list).
            remove_ratio: Ratio of path triples to remove.
            seed: Random seed.

        Returns:
            List of removed triples.
        """
        import random

        rng = random.Random(seed)

        # Step 1: Initialize L <- []
        L: list[tuple[str, str, str]] = []

        # Step 2: For each a_ent, find shortest path and extend L
        for ans_ent in answer_entities:
            path_triples = self.bfs_find_shortest_path(
                query_entity, ans_ent, max_depth=4
            )
            L.extend(path_triples)

        if not L:
            return []

        # Step 3: Randomly select triples from L
        num_to_remove = max(1, int(len(L) * remove_ratio))
        selected = rng.sample(L, min(num_to_remove, len(L)))

        # Step 4: Remove selected triples from G
        removed = []
        for t in selected:
            if self.remove_triple(*t):
                removed.append(t)

        logger.info(
            f"Removed {len(removed)} triples to simulate incompleteness"
        )
        return removed

    def get_stats(self) -> dict[str, int]:
        """Return KG statistics."""
        return {
            "num_entities": len(self.entity2id),
            "num_relations": len(self.relation2id),
            "num_triples": len(self._triple_set),
        }

    def save_triples(self, filepath: str) -> None:
        """Save all triples to a file."""
        with open(filepath, "w", encoding="utf-8") as f:
            for h, r, t in sorted(self._triple_set):
                f.write(f"{h}\t{r}\t{t}\n")

    def __len__(self) -> int:
        return len(self._triple_set)

    def __contains__(self, triple: tuple[str, str, str]) -> bool:
        return triple in self._triple_set
