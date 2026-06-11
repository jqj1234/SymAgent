"""Reconstruct evaluation index/results from per-sample trajectory JSON files.

Usage:
  conda run -n llmDrone python scripts/rebuild_eval_from_trajectories.py \
    --dataset webqsp \
    --pattern "2026-05-16_14-27-44_sample_*.json" \
    --output-dir logs/rebuilt_webqsp_2026-05-16_14-27-44
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from typing import Any


def normalize_entity(entity: str) -> str:
    return entity.lower().strip().replace("_", " ").replace("-", " ")


def resolve_entity_forms(entity: str, name2mid: dict[str, str], mid2name: dict[str, str]) -> set[str]:
    forms = {normalize_entity(entity)}
    ne = normalize_entity(entity)
    if ne in name2mid:
        forms.add(normalize_entity(name2mid[ne]))
    for mid, name in mid2name.items():
        n_name = normalize_entity(name)
        n_mid = normalize_entity(mid)
        if n_name == ne:
            forms.add(n_mid)
        elif n_mid == ne:
            forms.add(n_name)
    return forms


def hits_at_k(pred_forms: list[set[str]], gt_forms: list[set[str]], k: int) -> float:
    if not gt_forms or not pred_forms:
        return 0.0
    for p_set in pred_forms[:k]:
        for g_set in gt_forms:
            if p_set & g_set:
                return 1.0
    return 0.0


def f1_score(pred_forms: list[set[str]], gt_forms: list[set[str]]) -> float:
    if not pred_forms and not gt_forms:
        return 1.0
    if not pred_forms or not gt_forms:
        return 0.0
    matched_gt = set()
    matched_pred = set()
    for gi, g_set in enumerate(gt_forms):
        for pi, p_set in enumerate(pred_forms):
            if gi not in matched_gt and pi not in matched_pred and g_set & p_set:
                matched_gt.add(gi)
                matched_pred.add(pi)
                break
    if not matched_gt:
        return 0.0
    precision = len(matched_pred) / len(pred_forms)
    recall = len(matched_gt) / len(gt_forms)
    return 2 * precision * recall / (precision + recall)


def classify_error(predicted: list[str], ground_truth: list[str], name2mid: dict[str, str]) -> tuple[str, str]:
    if not ground_truth:
        return "dataset_missing", "Ground truth is empty — dataset annotation missing"
    if not predicted:
        return "no_answer", "Agent produced no answer"
    missing = [p for p in predicted if normalize_entity(p) not in name2mid]
    if missing:
        return (
            "mapping_missing",
            f"{len(missing)}/{len(predicted)} predicted entities not in name2mid mapping table (e.g. {missing[0]!r})",
        )
    return "unknown", "Failed to match — reason unclear"


def infer_dataset_index(idx: int) -> int:
    # Old per-sample artifacts were emitted before dataset_index existed.
    # They were written sequentially in evaluation order, so use idx-1.
    return idx - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--pattern", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    mid2name_path = os.path.join("data", "freebase", "mid2name.json")
    with open(mid2name_path, "r", encoding="utf-8") as f:
        mid2name = json.load(f)
    name2mid = {v.lower(): k for k, v in mid2name.items() if v}

    files = sorted(glob.glob(os.path.join("logs", "trajectories", args.dataset, args.pattern)))
    if not files:
        raise FileNotFoundError(f"No files matched pattern: {args.pattern}")

    samples: list[dict[str, Any]] = []
    all_scores: dict[str, list[float]] = {"hits@1": [], "hits@3": [], "hits@10": [], "f1": []}
    run_timestamp = None

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            sample = json.load(f)
        if run_timestamp is None:
            run_timestamp = sample.get("timestamp")

        idx = int(sample["idx"])
        dataset_index = infer_dataset_index(idx)
        predicted = sample.get("predicted", [])
        ground_truth = sample.get("ground_truth", [])

        pred_forms = [resolve_entity_forms(e, name2mid, mid2name) for e in predicted]
        gt_forms = [resolve_entity_forms(e, name2mid, mid2name) for e in ground_truth]
        metrics = {
            "hits@1": hits_at_k(pred_forms, gt_forms, 1),
            "hits@3": hits_at_k(pred_forms, gt_forms, 3),
            "hits@10": hits_at_k(pred_forms, gt_forms, 10),
            "f1": f1_score(pred_forms, gt_forms),
        }
        error_type, error_detail = ("", "")
        if metrics["hits@1"] < 1.0:
            error_type, error_detail = classify_error(predicted, ground_truth, name2mid)

        sample["dataset_index"] = dataset_index
        sample["metrics"] = metrics
        sample["error_type"] = error_type
        sample["error_detail"] = error_detail
        samples.append(sample)

        for k, v in metrics.items():
            all_scores[k].append(v)

    results = {
        metric: (sum(values) / len(values) if values else 0.0)
        for metric, values in all_scores.items()
    }

    index_path = os.path.join(args.output_dir, f"rebuilt_index_{run_timestamp}.json")
    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_timestamp,
                "dataset": args.dataset,
                "total_samples": len(samples),
                "samples": samples,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    results_path = os.path.join(args.output_dir, f"rebuilt_results_{run_timestamp}.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "run_timestamp": run_timestamp,
                "dataset": args.dataset,
                "num_samples": len(samples),
                "metrics": results,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    error_lines = [
        f"# Error Summary ({run_timestamp})",
        "",
        f"**Total samples:** {len(samples)}",
        f"**Failed:** {sum(1 for s in samples if s.get('error_type'))}",
        f"**Pass rate:** {1 - sum(1 for s in samples if s.get('error_type'))/len(samples):.1%}",
        "",
        "## Failed Samples",
        "",
    ]
    for sample in samples:
        if sample.get("error_type"):
            error_lines.append(
                f"- dataset_index={sample['dataset_index']}, idx={sample['idx']}, "
                f"question={sample['question']}, error_type={sample['error_type']}, "
                f"detail={sample['error_detail']}"
            )
    errors_path = os.path.join(args.output_dir, f"rebuilt_errors_summary_{run_timestamp}.md")
    with open(errors_path, "w", encoding="utf-8") as f:
        f.write("\n".join(error_lines))

    print(f"Rebuilt index:   {index_path}")
    print(f"Rebuilt results: {results_path}")
    print(f"Rebuilt errors:  {errors_path}")
    print("Metrics:")
    for k, v in sorted(results.items()):
        print(f"  {k}: {v:.4f}")


if __name__ == "__main__":
    main()
