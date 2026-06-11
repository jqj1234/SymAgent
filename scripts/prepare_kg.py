#!/usr/bin/env python3
"""
Prepare Knowledge Graph for SymAgent.

Processes raw KG triples from downloaded datasets and builds:
  - entity2id.txt: Entity name to integer ID mapping
  - relation2id.txt: Relation name to integer ID mapping
  - freebase_triples.txt: Cleaned triple file (head \\t relation \\t tail)

Supports Freebase (WebQSP/CWQ), MetaQA, and GrailQA KG formats.

Usage:
  python scripts/prepare_kg.py --dataset webqsp --input_dir data/webqsp --output_dir data/webqsp
  python scripts/prepare_kg.py --dataset cwq --input data/cwq/raw_triples.txt --output_dir data/cwq
  python scripts/prepare_kg.py --dataset metaqa --input_dir data/metaqa --output_dir data/metaqa
"""

import argparse
import json
import logging
import os
from collections import defaultdict

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Triple loading from various formats
# ---------------------------------------------------------------------------

def load_triples_tsv(filepath: str, max_triples: int = 0) -> list[tuple[str, str, str]]:
    """Load triples from TSV file (head\\trelation\\ttail).

    Args:
        filepath: Path to triples file.
        max_triples: Maximum triples to load (0 = all).

    Returns:
        List of (head, relation, tail) tuples.
    """
    triples = []
    with open(filepath, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if max_triples and i >= max_triples:
                break
            line = line.strip()
            if not line:
                continue

            # Try JSON lines: {"h": ..., "r": ..., "t": ...}
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    h = data.get("h") or data.get("head") or data.get("subject", "")
                    r = data.get("r") or data.get("relation") or data.get("predicate", "")
                    t = data.get("t") or data.get("tail") or data.get("object", "")
                    if h and r and t:
                        triples.append((str(h), str(r), str(t)))
                    continue
                except json.JSONDecodeError:
                    pass

            # N-Triples: <s> <p> <o> .
            if line.startswith("<"):
                parts = line.rstrip(".").split()
                if len(parts) >= 3:
                    h = parts[0].strip("<>")
                    r = parts[1].strip("<>")
                    t = parts[2].strip("<>")
                    triples.append((h, r, t))
                    continue

            # TSV: h\\tr\\tt
            parts = line.split("\t")
            if len(parts) >= 3:
                triples.append((parts[0], parts[1], parts[2]))
                continue

            # Pipe-separated: h|r|t (MetaQA)
            parts = line.split("|")
            if len(parts) >= 3:
                triples.append((parts[0].strip(), parts[1].strip(), parts[2].strip()))

    return triples


def extract_triples_from_qa_json(
    filepath: str,
) -> tuple[list[tuple[str, str, str]], list[dict]]:
    """Extract KG triples and QA pairs from a dataset JSON file.

    Handles both WebQSP and CWQ formats by trying multiple key names.

    Returns:
        Tuple of (triples, qa_pairs).
    """
    with open(filepath, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # Normalize to a list
    items = raw
    if isinstance(raw, dict):
        for key in ("Questions", "data", "questions", "items"):
            if key in raw and isinstance(raw[key], list):
                items = raw[key]
                break

    triples: list[tuple[str, str, str]] = []
    qa_pairs: list[dict] = []

    for item in items:
        # --- Extract QA pair ---
        question = (
            item.get("Question") or item.get("question") or
            item.get("RawQuestion") or ""
        )
        q_ent = (
            item.get("QuestionEntity") or item.get("question_entity") or
            item.get("topic_entity", {}).get("mid", "") or ""
        )

        # Answers
        answers_raw = item.get("Answers") or item.get("answers") or []
        answer_entities = []
        for a in answers_raw:
            if isinstance(a, dict):
                ans = a.get("AnswerName") or a.get("answer") or a.get("entity", "")
                if ans:
                    answer_entities.append(str(ans))
            elif isinstance(a, str):
                answer_entities.append(a)

        # Hop
        comp = item.get("compositionality", {})
        hop = comp.get("depth", comp.get("comp", item.get("hop", 1)))

        qa_pairs.append({
            "question": str(question).strip(),
            "question_entity": str(q_ent).strip(),
            "answer_entities": answer_entities,
            "hop": int(hop) if isinstance(hop, (int, float)) else 1,
        })

        # --- Extract subgraph triples ---
        for key in ("SubGraph", "subgraph", "triples", "knowledge_graph"):
            subgraph = item.get(key, [])
            for triple_data in subgraph:
                if isinstance(triple_data, (list, tuple)) and len(triple_data) >= 3:
                    triples.append((str(triple_data[0]), str(triple_data[1]), str(triple_data[2])))
                elif isinstance(triple_data, dict):
                    h = triple_data.get("head") or triple_data.get("h", "")
                    r = triple_data.get("relation") or triple_data.get("r", "")
                    t = triple_data.get("tail") or triple_data.get("t", "")
                    if h and r and t:
                        triples.append((h, r, t))

    return triples, qa_pairs


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_indices(
    triples: list[tuple[str, str, str]],
    output_dir: str,
) -> tuple[dict[str, int], dict[str, int]]:
    """Build entity2id and relation2id mappings from triples.

    Args:
        triples: List of (head, relation, tail) tuples.
        output_dir: Directory to write index files.

    Returns:
        Tuple of (entity2id, relation2id) mappings.
    """
    entities: set[str] = set()
    relations: set[str] = set()

    for h, r, t in triples:
        entities.add(h)
        entities.add(t)
        relations.add(r)

    entity2id = {ent: idx for idx, ent in enumerate(sorted(entities))}
    relation2id = {rel: idx for idx, rel in enumerate(sorted(relations))}

    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "entity2id.txt"), "w", encoding="utf-8") as f:
        for ent, idx in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{ent}\t{idx}\n")

    with open(os.path.join(output_dir, "relation2id.txt"), "w", encoding="utf-8") as f:
        for rel, idx in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{rel}\t{idx}\n")

    logger.info(
        "Built indices: %d entities, %d relations -> %s",
        len(entity2id), len(relation2id), output_dir,
    )
    return entity2id, relation2id


def write_clean_triples(
    triples: list[tuple[str, str, str]],
    output_path: str,
) -> int:
    """Write deduplicated triples to TSV file.

    Returns:
        Number of unique triples written.
    """
    seen: set[tuple[str, str, str]] = set()
    unique: list[tuple[str, str, str]] = []
    for t in triples:
        if t not in seen:
            seen.add(t)
            unique.append(t)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for h, r, t in sorted(unique):
            f.write(f"{h}\t{r}\t{t}\n")

    logger.info("Wrote %d unique triples -> %s", len(unique), output_path)
    return len(unique)


# ---------------------------------------------------------------------------
# Subgraph extraction
# ---------------------------------------------------------------------------

def extract_subgraph(
    triples: list[tuple[str, str, str]],
    seed_entities: set[str],
    max_hops: int = 2,
) -> list[tuple[str, str, str]]:
    """Extract a subgraph around seed entities within max_hops.

    Args:
        triples: Full triple list.
        seed_entities: Entities to start from.
        max_hops: Maximum hops from seed entities.

    Returns:
        Filtered triples within the subgraph.
    """
    adj: dict[str, set[str]] = defaultdict(set)
    for h, r, t in triples:
        adj[h].add(t)
        adj[t].add(h)

    relevant = set(seed_entities)
    frontier = set(seed_entities)
    for _ in range(max_hops):
        next_frontier = set()
        for ent in frontier:
            next_frontier.update(adj.get(ent, set()))
        relevant.update(next_frontier)
        frontier = next_frontier

    subgraph = [
        (h, r, t) for h, r, t in triples
        if h in relevant and t in relevant
    ]
    logger.info(
        "Subgraph: %d triples, %d entities (from %d seeds, %d hops)",
        len(subgraph), len(relevant), len(seed_entities), max_hops,
    )
    return subgraph


# ---------------------------------------------------------------------------
# Dataset-specific preparation
# ---------------------------------------------------------------------------

def _find_triple_file(input_dir: str) -> str | None:
    """Find the KG triple file in a directory."""
    candidates = [
        "freebase_triples.txt", "fb_triples.txt", "triples.txt",
        "kb.txt", "knowledge_graph.txt",
    ]
    for fname in candidates:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            return fpath
    return None


def _find_qa_files(input_dir: str, split: str) -> list[str]:
    """Find QA JSON files for a given split."""
    candidates = [
        f"{split}.json",
        f"WebQSP.{split}.json",
        f"ComplexWebQuestions_{split}.json",
        f"{split}_set.json",
        f"{split}_processed.json",
    ]
    found = []
    for fname in candidates:
        fpath = os.path.join(input_dir, fname)
        if os.path.exists(fpath):
            found.append(fpath)
    return found


def prepare_webqsp(input_dir: str, output_dir: str) -> None:
    """Prepare KG for WebQSP dataset."""
    os.makedirs(output_dir, exist_ok=True)
    all_triples: list[tuple[str, str, str]] = []

    # Load standalone triple file
    triple_file = _find_triple_file(input_dir)
    if triple_file:
        logger.info("Loading triples from %s", triple_file)
        all_triples.extend(load_triples_tsv(triple_file))

    # Extract triples from QA files
    for split in ("train", "dev", "test"):
        for fpath in _find_qa_files(input_dir, split):
            logger.info("Extracting from %s", fpath)
            triples, qa = extract_triples_from_qa_json(fpath)
            all_triples.extend(triples)
            # Save processed QA
            out_path = os.path.join(output_dir, f"{split}.json")
            if not os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(qa, f, ensure_ascii=False, indent=2)
                logger.info("Saved %d QA pairs -> %s", len(qa), out_path)
            break  # Use first match only

    if all_triples:
        build_indices(all_triples, output_dir)
        write_clean_triples(all_triples, os.path.join(output_dir, "freebase_triples.txt"))
    else:
        logger.warning("No triples found for WebQSP in %s", input_dir)


def prepare_cwq(input_dir: str, output_dir: str) -> None:
    """Prepare KG for CWQ dataset."""
    os.makedirs(output_dir, exist_ok=True)
    all_triples: list[tuple[str, str, str]] = []

    triple_file = _find_triple_file(input_dir)
    if triple_file:
        logger.info("Loading triples from %s", triple_file)
        all_triples.extend(load_triples_tsv(triple_file))

    for split in ("train", "dev", "test"):
        for fpath in _find_qa_files(input_dir, split):
            logger.info("Extracting from %s", fpath)
            triples, qa = extract_triples_from_qa_json(fpath)
            all_triples.extend(triples)
            out_path = os.path.join(output_dir, f"{split}.json")
            if not os.path.exists(out_path):
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(qa, f, ensure_ascii=False, indent=2)
                logger.info("Saved %d QA pairs -> %s", len(qa), out_path)
            break

    if all_triples:
        build_indices(all_triples, output_dir)
        write_clean_triples(all_triples, os.path.join(output_dir, "freebase_triples.txt"))
    else:
        logger.warning("No triples found for CWQ in %s", input_dir)


def prepare_metaqa(input_dir: str, output_dir: str) -> None:
    """Prepare KG for MetaQA dataset."""
    os.makedirs(output_dir, exist_ok=True)

    # Load KB
    kb_path = os.path.join(input_dir, "kb.txt")
    if os.path.exists(kb_path):
        triples = load_triples_tsv(kb_path)
        build_indices(triples, output_dir)
        write_clean_triples(triples, os.path.join(output_dir, "kb.txt"))
    else:
        logger.warning("MetaQA kb.txt not found in %s", input_dir)


def prepare_grailqa(input_dir: str, output_dir: str) -> None:
    """Prepare KG for GrailQA dataset.

    GrailQA uses Wikidata/Freebase and the KG is typically shared with
    WebQSP. This mainly processes the QA files.
    """
    os.makedirs(output_dir, exist_ok=True)
    all_triples: list[tuple[str, str, str]] = []

    # Try to find any triple files
    triple_file = _find_triple_file(input_dir)
    if triple_file:
        all_triples.extend(load_triples_tsv(triple_file))

    if all_triples:
        build_indices(all_triples, output_dir)
        write_clean_triples(all_triples, os.path.join(output_dir, "freebase_triples.txt"))
    else:
        logger.info("GrailQA: No standalone triple file found (expected if sharing FB with WebQSP)")


# ---------------------------------------------------------------------------
# Neo4j export (optional)
# ---------------------------------------------------------------------------

def prepare_neo4j_format(
    triples: list[tuple[str, str, str]],
    output_dir: str,
) -> None:
    """Convert triples to Neo4j import format (CSV)."""
    os.makedirs(output_dir, exist_ok=True)

    entities: set[str] = set()
    with open(os.path.join(output_dir, "edges.csv"), "w", encoding="utf-8") as f:
        f.write(":START_ID,relation,:END_ID\n")
        for h, r, t in triples:
            f.write(f'"{h}","{r}","{t}"\n')
            entities.add(h)
            entities.add(t)

    with open(os.path.join(output_dir, "nodes.csv"), "w", encoding="utf-8") as f:
        f.write("entityId:ID,:LABEL\n")
        for ent in entities:
            f.write(f'"{ent}","Entity"\n')

    logger.info("Wrote Neo4j import files to %s", output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

DISPATCH = {
    "webqsp": prepare_webqsp,
    "cwq": prepare_cwq,
    "metaqa": prepare_metaqa,
    "grailqa": prepare_grailqa,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare KG data for SymAgent")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["webqsp", "cwq", "metaqa", "grailqa"],
        help="Dataset to prepare",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default=None,
        help="Input directory with raw data (default: data/<dataset>)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: same as input_dir)",
    )
    parser.add_argument(
        "--neo4j",
        action="store_true",
        help="Also generate Neo4j import format",
    )

    args = parser.parse_args()
    input_dir = args.input_dir or os.path.join("data", args.dataset)
    output_dir = args.output_dir or input_dir

    logger.info("Preparing KG for %s", args.dataset)
    logger.info("  Input:  %s", os.path.abspath(input_dir))
    logger.info("  Output: %s", os.path.abspath(output_dir))

    DISPATCH[args.dataset](input_dir, output_dir)

    if args.neo4j:
        triple_file = os.path.join(output_dir, "freebase_triples.txt")
        if os.path.exists(triple_file):
            triples = load_triples_tsv(triple_file)
            prepare_neo4j_format(triples, os.path.join(output_dir, "neo4j"))

    logger.info("Done.")


if __name__ == "__main__":
    main()
