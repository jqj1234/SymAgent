#!/usr/bin/env python3
"""Filter multi-hop questions from WebQSP and CWQ datasets.

Paper Table 1 specifies filtered test set sizes:
  - WebQSP: test=247 (2-hop questions, max hop=2)
  - CWQ: test=316 (multi-hop questions, 2-4 hop)

This script:
  1. Loads processed data (train/valid/test splits)
  2. For WebQSP: keeps questions with exactly 2-hop paths (max hop=2)
  3. For CWQ: keeps questions with multi-hop paths (2-4 hop)
  4. Outputs filtered train/valid/test splits matching paper Table 1 sizes

Usage:
  python scripts/filter_multihop.py --dataset webqsp --input_dir data/processed --output_dir data/filtered
  python scripts/filter_multihop.py --dataset cwq --input_dir data/processed --output_dir data/filtered
"""

import argparse
import json
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def get_hop_count(sample: dict) -> int:
    """Determine the hop count of a QA sample.

    Uses the relation_path field: number of relations = hop count.
    Falls back to counting answer entities or other heuristics if
    relation_path is not available.

    Args:
        sample: QA sample dict.

    Returns:
        Hop count (integer).
    """
    relation_path = sample.get("relation_path", [])
    if relation_path and isinstance(relation_path, list):
        return len(relation_path)

    # Fallback: check for nested relation paths
    if "composition" in sample:
        comp = sample["composition"]
        if isinstance(comp, list):
            return len(comp)

    # Default: single hop
    return 1


def filter_webqsp(data: list[dict]) -> list[dict]:
    """Filter WebQSP data for 2-hop questions.

    Paper Table 1: WebQSP keeps questions with 2-hop paths (max hop=2).

    Args:
        data: List of QA sample dicts.

    Returns:
        Filtered list with only 2-hop questions.
    """
    filtered = [s for s in data if get_hop_count(s) == 2]
    return filtered


def filter_cwq(data: list[dict]) -> list[dict]:
    """Filter CWQ data for multi-hop questions (2-4 hop).

    Paper Table 1: CWQ keeps questions with multi-hop paths (2-4 hop).

    Args:
        data: List of QA sample dicts.

    Returns:
        Filtered list with only multi-hop (2-4 hop) questions.
    """
    filtered = [s for s in data if 2 <= get_hop_count(s) <= 4]
    return filtered


def load_json(filepath: str) -> list[dict]:
    """Load a JSON file as a list of dicts."""
    if not os.path.exists(filepath):
        logger.warning(f"File not found: {filepath}")
        return []
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, list):
        return data
    # Some formats nest data under a key
    if isinstance(data, dict):
        for key in ("data", "questions", "samples"):
            if key in data and isinstance(data[key], list):
                return data[key]
    return []


def save_json(data: list[dict], filepath: str) -> None:
    """Save data to a JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description="Filter multi-hop questions from WebQSP/CWQ datasets."
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["webqsp", "cwq"],
        help="Dataset to filter.",
    )
    parser.add_argument(
        "--input_dir",
        type=str,
        default="data/processed",
        help="Directory containing processed train/valid/test JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/filtered",
        help="Directory to save filtered files.",
    )
    args = parser.parse_args()

    filter_fn = filter_webqsp if args.dataset == "webqsp" else filter_cwq
    expected_test_size = 247 if args.dataset == "webqsp" else 316

    for split in ["train", "valid", "test"]:
        input_path = os.path.join(args.input_dir, f"{args.dataset}_{split}.json")
        output_path = os.path.join(args.output_dir, f"{args.dataset}_{split}.json")

        data = load_json(input_path)
        if not data:
            logger.info(f"No data found for {args.dataset} {split}, skipping.")
            continue

        filtered = filter_fn(data)

        logger.info(
            f"{args.dataset} {split}: {len(data)} -> {len(filtered)} "
            f"(filtered for multi-hop)"
        )

        if split == "test":
            logger.info(
                f"  Paper Table 1 expected test size: {expected_test_size}, "
                f"got: {len(filtered)}"
            )

        save_json(filtered, output_path)
        logger.info(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
