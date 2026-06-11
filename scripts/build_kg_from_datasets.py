"""Build Freebase KG indices and unified QA files from WebQSP, CWQ, GrailQA datasets.

Reads the raw dataset JSON files, extracts:
  - Entities (topic entities + answer entities)
  - Relations (from SPARQL / InferentialChain / graph_query)
  - Relation paths for each QA pair

Outputs to data/freebase/:
  - entity2id.txt     (entity_mid \\t id)
  - relation2id.txt   (relation \\t id)
  - freebase_triples.txt (head \\t relation \\t tail)
  - qa_train.json     (unified QA train set)
  - qa_test.json      (unified QA test set)

Unified QA format:
  {
    "question": str,
    "question_entity": str,       # Freebase MID (e.g. "m.078w2")
    "answer_entities": [str],     # list of MIDs
    "relation_path": [str],       # extracted relation chain
    "dataset": str,               # source dataset
    "qid": str,                   # original question ID
  }
"""

import argparse
import json
import logging
import os
import re
from collections import defaultdict
from typing import Any, Optional

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# WebQSP extraction
# ---------------------------------------------------------------------------

def extract_webqsp(data: list[dict]) -> list[dict]:
    """Extract QA samples from WebQSP dataset.

    WebQSP format:
      - Parses[0].InferentialChain: list of relation strings (e.g. ["film.actor.film"])
      - Parses[0].Answers: list of {"AnswerArgument": MID, ...}
      - topic_entity: dict {MID: name}
      - QuestionId, RawQuestion
    """
    samples = []
    for item in data:
        parses = item.get("Parses", [])
        if not parses:
            continue

        parse = parses[0]
        inferential_chain = parse.get("InferentialChain", [])
        answers = [
            a["AnswerArgument"]
            for a in parse.get("Answers", [])
            if a.get("AnswerArgument", "").startswith("m.")
        ]

        # Extract topic entity MID from top-level dict
        topic_dict = item.get("topic_entity", {})
        if isinstance(topic_dict, dict):
            topic_mid = list(topic_dict.keys())[0] if topic_dict else ""
        else:
            topic_mid = str(topic_dict)

        if not topic_mid or not answers:
            continue

        samples.append({
            "question": item.get("RawQuestion", item.get("ProcessedQuestion", "")),
            "question_entity": topic_mid,
            "answer_entities": answers,
            "relation_path": inferential_chain,
            "dataset": "webqsp",
            "qid": item.get("QuestionId", ""),
        })

    logger.info(f"WebQSP: extracted {len(samples)} valid QA samples")
    return samples


# ---------------------------------------------------------------------------
# CWQ extraction
# ---------------------------------------------------------------------------

_CWQ_NS_PATTERN = re.compile(r"ns:([a-zA-Z0-9_.]+)")


def _extract_cwq_relations_from_sparql(sparql: str) -> list[str]:
    """Extract relation paths from CWQ SPARQL query.

    Looks for ns:xxx.yyy.zzz patterns in triple patterns (?s ns:rel ?o).
    Filters out type predicates and utility predicates.
    """
    relations = []
    for match in _CWQ_NS_PATTERN.finditer(sparql):
        rel = match.group(1)
        # Skip entity MIDs (m.xxxxx, g.xxxxx patterns)
        if re.match(r"^[mg]\.\w+$", rel):
            continue
        # Skip type/object.type predicates
        if rel in ("type.object.type", "type.object.name", "common.topic.alias"):
            continue
        if rel.startswith("type."):
            continue
        relations.append(rel)
    return relations


def extract_cwq(data: list[dict]) -> list[dict]:
    """Extract QA samples from CWQ dataset.

    CWQ format:
      - sparql: SPARQL query with ns:xxx.yyy relation patterns
      - topic_entity: dict {MID: name}
      - answer: string (often a literal value, not an entity MID)
    """
    samples = []
    for item in data:
        sparql = item.get("sparql", "")
        relations = _extract_cwq_relations_from_sparql(sparql)

        # Extract topic entity
        topic_dict = item.get("topic_entity", {})
        if isinstance(topic_dict, dict):
            topic_mid = list(topic_dict.keys())[0] if topic_dict else ""
        else:
            topic_mid = str(topic_dict)

        if not topic_mid:
            continue

        # CWQ answers are often literal strings, not MIDs
        # Try to extract MIDs from the answer string
        answer_raw = item.get("answer", "")
        answer_entities = []
        if isinstance(answer_raw, str):
            # Look for m.xxx patterns
            mid_matches = re.findall(r"\bm\.\w+", answer_raw)
            answer_entities = mid_matches
        elif isinstance(answer_raw, list):
            for a in answer_raw:
                if isinstance(a, dict):
                    arg = a.get("answer_argument", "")
                    if arg.startswith("m."):
                        answer_entities.append(arg)
                elif isinstance(a, str) and a.startswith("m."):
                    answer_entities.append(a)

        samples.append({
            "question": item.get("question", item.get("machine_question", "")),
            "question_entity": topic_mid,
            "answer_entities": answer_entities,
            "relation_path": relations,
            "dataset": "cwq",
            "qid": item.get("ID", ""),
        })

    logger.info(f"CWQ: extracted {len(samples)} valid QA samples")
    return samples


# ---------------------------------------------------------------------------
# GrailQA extraction
# ---------------------------------------------------------------------------

def extract_grailqa(data: list[dict]) -> list[dict]:
    """Extract QA samples from GrailQA dataset.

    GrailQA format:
      - graph_query.edges: list of {start, end, relation}
      - answer: list of {"answer_type": str, "answer_argument": str}
      - topic_entity: dict {MID: name}
      - s_expression: S-expression with JOIN patterns
    """
    samples = []
    for item in data:
        # Extract relations from graph_query edges
        gq = item.get("graph_query", {})
        edges = gq.get("edges", [])
        relations = [e["relation"] for e in edges if "relation" in e]

        # Also try extracting from s_expression as fallback
        if not relations:
            s_expr = item.get("s_expression", "")
            join_matches = re.findall(r"JOIN\s+([a-zA-Z0-9_.]+)", s_expr)
            relations = join_matches

        # Extract answers
        answer_entities = []
        for ans in item.get("answer", []):
            arg = ans.get("answer_argument", "")
            if arg.startswith("m."):
                answer_entities.append(arg)

        # Extract topic entity
        topic_dict = item.get("topic_entity", {})
        if isinstance(topic_dict, dict):
            topic_mid = list(topic_dict.keys())[0] if topic_dict else ""
        else:
            topic_mid = str(topic_dict)

        if not topic_mid:
            continue

        samples.append({
            "question": item.get("question", ""),
            "question_entity": topic_mid,
            "answer_entities": answer_entities,
            "relation_path": relations,
            "dataset": "grailqa",
            "qid": str(item.get("qid", "")),
        })

    logger.info(f"GrailQA: extracted {len(samples)} valid QA samples")
    return samples


# ---------------------------------------------------------------------------
# KG construction
# ---------------------------------------------------------------------------

def build_kg_indices(
    all_samples: list[dict],
    output_dir: str,
) -> dict[str, Any]:
    """Build entity2id, relation2id, and triples from all QA samples.

    Since we don't have the actual Freebase subgraph triples, we construct
    pseudo-triples from the QA relation paths:
      - (topic_entity, relation_path[0], __INTERMEDIATE_0__)
      - (__INTERMEDIATE_0__, relation_path[1], __INTERMEDIATE_1__)
      - ...
      - (__INTERMEDIATE_N__, relation_path[-1], answer_entity)

    This gives the KG environment enough structure for BFS path finding
    and relation reasoning.

    Additionally, we create direct (topic, relation, answer) triples for
    single-hop questions where the answer is known.
    """
    os.makedirs(output_dir, exist_ok=True)

    entity_set: set[str] = set()
    relation_set: set[str] = set()
    triples: set[tuple[str, str, str]] = set()

    # Intermediate node counter
    _intermediate_counter = 0

    def _intermediate_node() -> str:
        nonlocal _intermediate_counter
        node = f"__INTERMEDIATE_{_intermediate_counter}__"
        _intermediate_counter += 1
        return node

    for sample in all_samples:
        topic = sample["question_entity"]
        path = sample["relation_path"]
        answers = sample["answer_entities"]

        entity_set.add(topic)
        for rel in (path or []):
            relation_set.add(rel)

        if not path:
            continue

        if len(path) == 1 and answers:
            # Single-hop: create direct triples (topic, rel, answer)
            rel = path[0]
            for ans in answers:
                entity_set.add(ans)
                triples.add((topic, rel, ans))
        else:
            # Multi-hop: create chain with intermediate nodes
            current = topic
            for i, rel in enumerate(path):
                if i < len(path) - 1:
                    # Intermediate step
                    next_node = _intermediate_node()
                    triples.add((current, rel, next_node))
                    current = next_node
                else:
                    # Final step: connect to answer entities
                    for ans in answers:
                        entity_set.add(ans)
                        triples.add((current, rel, ans))
                    # If no answers, still create a placeholder
                    if not answers:
                        final = _intermediate_node()
                        triples.add((current, rel, final))

    # Also add all answer entities even if they weren't in triples
    for sample in all_samples:
        for ans in sample["answer_entities"]:
            entity_set.add(ans)

    # Build entity2id and relation2id mappings
    entity2id = {name: idx for idx, name in enumerate(sorted(entity_set))}
    relation2id = {name: idx for idx, name in enumerate(sorted(relation_set))}

    # Save entity2id.txt
    entity2id_path = os.path.join(output_dir, "entity2id.txt")
    with open(entity2id_path, "w") as f:
        for name, idx in sorted(entity2id.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{idx}\n")
    logger.info(f"Saved {len(entity2id)} entities to {entity2id_path}")

    # Save relation2id.txt
    relation2id_path = os.path.join(output_dir, "relation2id.txt")
    with open(relation2id_path, "w") as f:
        for name, idx in sorted(relation2id.items(), key=lambda x: x[1]):
            f.write(f"{name}\t{idx}\n")
    logger.info(f"Saved {len(relation2id)} relations to {relation2id_path}")

    # Save freebase_triples.txt
    triples_path = os.path.join(output_dir, "freebase_triples.txt")
    with open(triples_path, "w") as f:
        for h, r, t in sorted(triples):
            f.write(f"{h}\t{r}\t{t}\n")
    logger.info(f"Saved {len(triples)} triples to {triples_path}")

    return {
        "num_entities": len(entity2id),
        "num_relations": len(relation2id),
        "num_triples": len(triples),
    }


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------

def create_train_test_split(
    all_samples: list[dict],
    output_dir: str,
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[int, int]:
    """Split samples into train/test and save as JSON."""
    import random
    rng = random.Random(seed)

    rng.shuffle(all_samples)
    n_test = max(1, int(len(all_samples) * test_ratio))
    test_data = all_samples[:n_test]
    train_data = all_samples[n_test:]

    train_path = os.path.join(output_dir, "qa_train.json")
    test_path = os.path.join(output_dir, "qa_test.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Train: {len(train_data)}, Test: {len(test_data)}")
    return len(train_data), len(test_data)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Build Freebase KG indices from datasets"
    )
    parser.add_argument(
        "--data_dir", type=str, default="data",
        help="Root data directory",
    )
    parser.add_argument(
        "--output_dir", type=str, default="data/freebase",
        help="Output directory for KG indices",
    )
    parser.add_argument(
        "--test_ratio", type=float, default=0.2,
        help="Test set ratio",
    )
    args = parser.parse_args()

    all_samples: list[dict] = []

    # --- WebQSP ---
    webqsp_path = os.path.join(args.data_dir, "webqsp", "WebQSP.json")
    if os.path.exists(webqsp_path):
        with open(webqsp_path, "r") as f:
            webqsp_data = json.load(f)
        all_samples.extend(extract_webqsp(webqsp_data))
    else:
        logger.warning(f"WebQSP file not found: {webqsp_path}")

    # --- CWQ ---
    cwq_path = os.path.join(args.data_dir, "cwq", "cwq.json")
    if os.path.exists(cwq_path):
        with open(cwq_path, "r") as f:
            cwq_data = json.load(f)
        all_samples.extend(extract_cwq(cwq_data))
    else:
        logger.warning(f"CWQ file not found: {cwq_path}")

    # --- GrailQA ---
    grailqa_path = os.path.join(args.data_dir, "grailqa", "graliqa.json")
    if os.path.exists(grailqa_path):
        with open(grailqa_path, "r") as f:
            grailqa_data = json.load(f)
        all_samples.extend(extract_grailqa(grailqa_data))
    else:
        logger.warning(f"GrailQA file not found: {grailqa_path}")

    if not all_samples:
        logger.error("No samples extracted. Check dataset files.")
        return

    logger.info(f"Total extracted QA samples: {len(all_samples)}")

    # Build KG indices
    stats = build_kg_indices(all_samples, args.output_dir)
    logger.info(f"KG stats: {stats}")

    # Create train/test split
    train_n, test_n = create_train_test_split(
        all_samples, args.output_dir, args.test_ratio
    )
    logger.info(f"Done! Train={train_n}, Test={test_n}")


if __name__ == "__main__":
    main()
