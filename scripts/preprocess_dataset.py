"""Preprocess WebQSP, CWQ, GrailQA datasets into unified format for SymAgent.

Converts different dataset formats into a unified format:
{
    "qid": str,
    "question": str,
    "question_entity": str,      # Freebase MID
    "topic_entity_name": str,    # Entity name
    "answer_entities": list[str],  # List of answer entity MIDs
    "sparql": str,               # Original SPARQL query (if available)
    "compositionality_type": str,  # For CWQ
    "inferential_chain": list[str],  # For WebQSP
}
"""

import argparse
import json
import logging
import os
import re
from typing import Any

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def _extract_topic_entity(topic_entity_field: Any) -> tuple[str, str]:
    """Extract (mid, name) from topic_entity field.

    Handles both dict format {MID: name} and string format.
    """
    if isinstance(topic_entity_field, dict):
        for mid, name in topic_entity_field.items():
            return mid, name
        return "", ""
    return str(topic_entity_field), ""


def preprocess_webqsp(input_path: str, output_dir: str) -> dict[str, Any]:
    """Preprocess WebQSP dataset.

    Input format:
        {
            "QuestionId": str,
            "RawQuestion": str,
            "ProcessedQuestion": str,
            "Parses": [{
                "Answers": [{"AnswerArgument": str, "EntityName": str}],
                "InferentialChain": [str],
                "Sparql": str,
                "TopicEntityMid": str,
                "TopicEntityName": str,
            }],
            "topic_entity": {MID: name, ...},  # dict, not string
            "qid_topic_entity": {QID: name, ...},
        }
    """
    logger.info(f"Processing WebQSP from {input_path}")
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data:
        # Extract answers from first parse
        answers = []
        sparql = ""
        inferential_chain = []
        topic_mid = ""
        topic_name = ""

        parses = item.get("Parses", [])
        if parses:
            parse = parses[0]
            for ans in parse.get("Answers", []):
                ans_arg = ans.get("AnswerArgument", "")
                if ans_arg:
                    answers.append(ans_arg)
            sparql = parse.get("Sparql", "")
            inferential_chain = parse.get("InferentialChain", [])
            topic_mid = parse.get("TopicEntityMid", "")
            topic_name = parse.get("TopicEntityName", "")

        # Fallback to top-level dict {MID: name}
        if not topic_mid:
            topic_mid, topic_name = _extract_topic_entity(item.get("topic_entity", {}))

        processed.append({
            "qid": item.get("QuestionId", ""),
            "question": item.get("RawQuestion", item.get("ProcessedQuestion", "")),
            "question_entity": topic_mid,
            "topic_entity_name": topic_name,
            "answer_entities": answers,
            "sparql": sparql,
            "compositionality_type": "simple",
            "inferential_chain": inferential_chain,
        })

    # Save
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "webqsp_processed.json")
    with open(output_path, "w") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    logger.info(f"WebQSP: {len(processed)} samples saved to {output_path}")
    return {"total": len(processed), "with_answers": sum(1 for x in processed if x["answers"])}


def preprocess_cwq(input_path: str, output_dir: str) -> dict[str, Any]:
    """Preprocess CWQ dataset.

    Input format:
        {
            "ID": str,
            "question": str,
            "topic_entity": {MID: name, ...},  # dict, not string
            "answer": str,  # literal answer string (e.g. "2014 World Series")
            "sparql": str,
            "compositionality_type": str,
            "webqsp_ID": str,
        }
    """
    logger.info(f"Processing CWQ from {input_path}")
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data:
        # CWQ answers are literal strings, not entity MIDs
        answer_raw = item.get("answer", "")
        if isinstance(answer_raw, str):
            # Try to extract any MIDs from the answer
            mid_matches = re.findall(r"\bm\.\w+", answer_raw)
            answers = mid_matches if mid_matches else [answer_raw]
        elif isinstance(answer_raw, list):
            answers = answer_raw
        else:
            answers = [str(answer_raw)] if answer_raw else []

        # Extract topic entity from dict {MID: name}
        topic_mid, topic_name = _extract_topic_entity(item.get("topic_entity", {}))

        processed.append({
            "qid": item.get("ID", ""),
            "question": item.get("question", ""),
            "question_entity": topic_mid,
            "topic_entity_name": topic_name,
            "answer_entities": answers,
            "sparql": item.get("sparql", ""),
            "compositionality_type": item.get("compositionality_type", "composition"),
            "inferential_chain": [],
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "cwq_processed.json")
    with open(output_path, "w") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    logger.info(f"CWQ: {len(processed)} samples saved to {output_path}")
    return {"total": len(processed), "with_answers": sum(1 for x in processed if x["answers"])}


def preprocess_grailqa(input_path: str, output_dir: str) -> dict[str, Any]:
    """Preprocess GrailQA dataset.

    Input format:
        {
            "qid": int,
            "question": str,
            "answer": [{"answer_type": str, "answer_argument": str}],
            "topic_entity": {MID: name, ...},  # dict, not string
            "graph_query": {"nodes": [...], "edges": [...]},
            "sparql_query": str,
            "s_expression": str,
            "level": str,
        }
    """
    logger.info(f"Processing GrailQA from {input_path}")
    with open(input_path, "r") as f:
        raw_data = json.load(f)

    processed = []
    for item in raw_data:
        answers = []
        for ans in item.get("answer", []):
            ans_arg = ans.get("answer_argument", "")
            if ans_arg:
                answers.append(ans_arg)

        # Extract topic entity from dict {MID: name}
        topic_mid, topic_name = _extract_topic_entity(item.get("topic_entity", {}))

        processed.append({
            "qid": str(item.get("qid", "")),
            "question": item.get("question", ""),
            "question_entity": topic_mid,
            "topic_entity_name": topic_name,
            "answer_entities": answers,
            "sparql": item.get("sparql_query", ""),
            "compositionality_type": item.get("level", "unknown"),
            "inferential_chain": [],
        })

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "grailqa_processed.json")
    with open(output_path, "w") as f:
        json.dump(processed, f, ensure_ascii=False, indent=2)

    logger.info(f"GrailQA: {len(processed)} samples saved to {output_path}")
    return {"total": len(processed), "with_answers": sum(1 for x in processed if x["answers"])}


def create_train_test_split(processed_path: str, output_dir: str, test_ratio: float = 0.2):
    """Split processed data into train/test sets."""
    with open(processed_path, "r") as f:
        data = json.load(f)

    # Shuffle deterministically
    data.sort(key=lambda x: x["qid"])
    n_test = max(1, int(len(data) * test_ratio))
    test_data = data[:n_test]
    train_data = data[n_test:]

    base_name = os.path.basename(processed_path).replace("_processed.json", "")
    train_path = os.path.join(output_dir, f"{base_name}_train.json")
    test_path = os.path.join(output_dir, f"{base_name}_test.json")

    with open(train_path, "w") as f:
        json.dump(train_data, f, ensure_ascii=False, indent=2)
    with open(test_path, "w") as f:
        json.dump(test_data, f, ensure_ascii=False, indent=2)

    logger.info(f"Split: train={len(train_data)}, test={len(test_data)}")
    return train_path, test_path


def main():
    parser = argparse.ArgumentParser(description="Preprocess datasets for SymAgent")
    parser.add_argument("--data_dir", type=str, default="data")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--dataset", type=str, default="all",
                        choices=["all", "webqsp", "cwq", "grailqa"])
    parser.add_argument("--test_ratio", type=float, default=0.2)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    if args.dataset in ("all", "webqsp"):
        stats = preprocess_webqsp(
            os.path.join(args.data_dir, "webqsp", "WebQSP.json"),
            args.output_dir,
        )
        logger.info(f"WebQSP stats: {stats}")
        create_train_test_split(
            os.path.join(args.output_dir, "webqsp_processed.json"),
            args.output_dir,
            args.test_ratio,
        )

    if args.dataset in ("all", "cwq"):
        stats = preprocess_cwq(
            os.path.join(args.data_dir, "cwq", "cwq.json"),
            args.output_dir,
        )
        logger.info(f"CWQ stats: {stats}")
        create_train_test_split(
            os.path.join(args.output_dir, "cwq_processed.json"),
            args.output_dir,
            args.test_ratio,
        )

    if args.dataset in ("all", "grailqa"):
        stats = preprocess_grailqa(
            os.path.join(args.data_dir, "grailqa", "graliqa.json"),
            args.output_dir,
        )
        logger.info(f"GrailQA stats: {stats}")
        create_train_test_split(
            os.path.join(args.output_dir, "grailqa_processed.json"),
            args.output_dir,
            args.test_ratio,
        )

    logger.info("Preprocessing complete!")


if __name__ == "__main__":
    main()
