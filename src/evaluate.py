"""
Evaluation module for SymAgent.

Implements evaluation metrics as described in the paper:
- Hits@1, Hits@3, Hits@10
- F1 score
- Accuracy (as used in Table 2)

Supports WebQSP, CWQ, and GrailQA/MetaQA datasets.
"""

import json
import logging
import os
import re
from datetime import datetime
from typing import Any, Optional

from .executor import AgentExecutor, compute_outcome_reward, Trajectory
from .kg_environment import KGEnvironment
from .llm_client import LLMClient
from .planner import AgentPlanner

logger = logging.getLogger(__name__)


def normalize_entity(entity: str) -> str:
    """Normalize entity name for comparison.

    Handles different naming conventions (Freebase mid, Wikidata ID, etc.).
    """
    return entity.lower().strip().replace("_", " ").replace("-", " ")


def hits_at_k(
    predicted: list[str],
    ground_truth: list[str],
    k: int = 1,
) -> float:
    """Compute Hits@k metric.

    A hit occurs if any of the top-k predicted entities appears
    in the ground truth set.

    Args:
        predicted: Ranked list of predicted entity names.
        ground_truth: List of ground truth entity names.
        k: Cutoff rank.

    Returns:
        1.0 if hit, 0.0 otherwise.
    """
    if not ground_truth or not predicted:
        return 0.0

    gt_normalized = {normalize_entity(e) for e in ground_truth}
    top_k = predicted[:k]

    for pred in top_k:
        if normalize_entity(pred) in gt_normalized:
            return 1.0
    return 0.0


def f1_score(
    predicted: list[str],
    ground_truth: list[str],
) -> float:
    """Compute F1 score between predicted and ground truth entities.

    Args:
        predicted: List of predicted entity names.
        ground_truth: List of ground truth entity names.

    Returns:
        F1 score in [0, 1].
    """
    if not predicted and not ground_truth:
        return 1.0
    if not predicted or not ground_truth:
        return 0.0

    pred_set = {normalize_entity(e) for e in predicted}
    gt_set = {normalize_entity(e) for e in ground_truth}

    intersection = pred_set & gt_set
    if not intersection:
        return 0.0

    precision = len(intersection) / len(pred_set)
    recall = len(intersection) / len(gt_set)

    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def accuracy(
    predicted: list[str],
    ground_truth: list[str],
) -> float:
    """Compute accuracy (exact match on at least one answer).

    Args:
        predicted: List of predicted entity names.
        ground_truth: List of ground truth entity names.

    Returns:
        1.0 if any predicted entity matches any ground truth, 0.0 otherwise.
    """
    if not predicted or not ground_truth:
        return 0.0

    pred_set = {normalize_entity(e) for e in predicted}
    gt_set = {normalize_entity(e) for e in ground_truth}

    return 1.0 if pred_set & gt_set else 0.0


class Evaluator:
    """Evaluator for SymAgent on KGQA benchmarks.

    Evaluates the agent on datasets using standard metrics:
    Hits@1, Hits@3, Hits@10, F1, Accuracy.
    """

    def __init__(
        self,
        kg: KGEnvironment,
        llm: LLMClient,
        planner: AgentPlanner,
        executor: AgentExecutor,
        metrics: Optional[list[str]] = None,
        trajectory_dir: Optional[str] = None,
        dataset_name: str = "",
    ):
        self.kg = kg
        self.llm = llm
        self.planner = planner
        self.executor = executor
        self.metrics = metrics or ["hits@1", "hits@3", "hits@10", "f1", "accuracy"]
        self.trajectory_dir = trajectory_dir
        self.dataset_name = dataset_name
        self._run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self._index_entries: list[dict[str, Any]] = []

    def evaluate(
        self,
        test_data: list[dict[str, Any]],
        max_samples: Optional[int] = None,
        start_index: int = 0,
        end_index: Optional[int] = None,
        resume_index_file: Optional[str] = None,
    ) -> dict[str, float]:
        """Evaluate SymAgent on a test dataset.

        Args:
            test_data: List of test examples with 'question', 'question_entity',
                      and 'answer_entities'.
            max_samples: Optional limit on number of samples to evaluate.
            start_index: Global dataset start index (inclusive).
            end_index: Global dataset end index (exclusive).
            resume_index_file: Optional prior index_*.json to skip completed samples.

        Returns:
            Dict of metric_name -> score.
        """
        start_index = max(0, start_index)
        effective_end = len(test_data) if end_index is None else min(end_index, len(test_data))
        samples = test_data[start_index:effective_end]
        if max_samples is not None:
            samples = samples[:max_samples]

        completed_indices = self._load_completed_indices(resume_index_file)

        all_scores: dict[str, list[float]] = {
            m: [] for m in self.metrics
        }

        processed = 0
        for local_i, sample in enumerate(samples):
            dataset_index = start_index + local_i
            if dataset_index in completed_indices:
                logger.info(
                    "[%d/%d] Skipping completed sample at dataset_index=%d",
                    local_i + 1,
                    len(samples),
                    dataset_index,
                )
                continue

            question = sample["question"]
            q_ent = sample.get("question_entity", "")
            a_ents = sample.get("answer_entities", [])

            logger.info(
                f"[{local_i+1}/{len(samples)}] Evaluating: {question[:60]}..."
            )

            planned_paths = self.planner.plan(question, q_ent)
            self.executor.reset()
            trajectory = self.executor.execute(question, q_ent, planned_paths)
            predicted = trajectory.answer_entities

            sample_scores = self._compute_metrics(predicted, a_ents)
            for metric, score in sample_scores.items():
                all_scores[metric].append(score)

            error_type, error_detail = "", ""
            if sample_scores.get("hits@1", 0) < 1.0:
                error_type, error_detail = self._classify_error(
                    predicted, a_ents, sample_scores
                )

            if self.trajectory_dir:
                self._save_trajectory(
                    trajectory=trajectory,
                    idx=processed + 1,
                    dataset_index=dataset_index,
                    question=question,
                    predicted=predicted,
                    ground_truth=a_ents,
                    metrics=sample_scores,
                    error_type=error_type,
                    error_detail=error_detail,
                )
                self._save_index()
                self._save_error_summary()

            processed += 1
            if processed % 10 == 0:
                self._log_progress(all_scores, processed)

        results = self._aggregate_scores(all_scores)

        if self.trajectory_dir and self._index_entries:
            self._save_aggregate_artifacts(results)

        return results

    def _load_completed_indices(
        self, resume_index_file: Optional[str]
    ) -> set[int]:
        """Load completed dataset indices from a prior index file."""
        if not resume_index_file or not os.path.exists(resume_index_file):
            return set()
        with open(resume_index_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        return {
            int(sample["dataset_index"])
            for sample in data.get("samples", [])
            if "dataset_index" in sample
        }

    @staticmethod
    def _aggregate_scores(all_scores: dict[str, list[float]]) -> dict[str, float]:
        """Aggregate per-sample metric lists into final averages."""
        results = {}
        for metric, scores in all_scores.items():
            if scores:
                results[metric] = sum(scores) / len(scores)
            else:
                results[metric] = 0.0
        return results

    def _save_aggregate_artifacts(self, results: dict[str, float]) -> None:
        """Persist aggregate eval results for the current run."""
        assert self.trajectory_dir is not None
        results_path = os.path.join(
            self.trajectory_dir, f"results_{self._run_timestamp}.json"
        )
        payload = {
            "run_timestamp": self._run_timestamp,
            "dataset": self.dataset_name,
            "num_samples": len(self._index_entries),
            "metrics": results,
        }
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Per-run evaluation summary saved to %s", results_path)

    def merge_index_files(
        self,
        index_files: list[str],
        output_dir: Optional[str] = None,
    ) -> dict[str, float]:
        """Merge multiple index_*.json files and recompute final metrics."""
        merged: dict[int, dict[str, Any]] = {}
        for path in index_files:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            for sample in data.get("samples", []):
                dataset_index = int(sample["dataset_index"])
                if dataset_index in merged and merged[dataset_index] != sample:
                    raise ValueError(
                        f"Conflicting duplicate sample for dataset_index={dataset_index}: {path}"
                    )
                merged[dataset_index] = sample

        merged_samples = [merged[k] for k in sorted(merged)]
        all_scores: dict[str, list[float]] = {m: [] for m in self.metrics}
        for sample in merged_samples:
            for metric in self.metrics:
                if metric in sample.get("metrics", {}):
                    all_scores[metric].append(sample["metrics"][metric])
        results = self._aggregate_scores(all_scores)

        target_dir = output_dir or self.trajectory_dir
        if target_dir:
            os.makedirs(target_dir, exist_ok=True)
            merged_index_path = os.path.join(
                target_dir, f"merged_index_{self._run_timestamp}.json"
            )
            merged_results_path = os.path.join(
                target_dir, f"merged_results_{self._run_timestamp}.json"
            )
            with open(merged_index_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_timestamp": self._run_timestamp,
                        "dataset": self.dataset_name,
                        "source_files": index_files,
                        "total_samples": len(merged_samples),
                        "samples": merged_samples,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            with open(merged_results_path, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "run_timestamp": self._run_timestamp,
                        "dataset": self.dataset_name,
                        "source_files": index_files,
                        "num_samples": len(merged_samples),
                        "metrics": results,
                    },
                    f,
                    ensure_ascii=False,
                    indent=2,
                )
            logger.info("Merged evaluation artifacts saved to %s", target_dir)
        return results

    def _classify_error(
        self,
        predicted: list[str],
        ground_truth: list[str],
        metrics: dict[str, float],
    ) -> tuple[str, str]:
        """Automatically classify why a sample failed.

        Returns:
            Tuple of (error_category, human_readable_explanation).
        """
        if not ground_truth:
            return "dataset_missing", "Ground truth is empty — dataset annotation missing"

        if not predicted:
            return "no_answer", "Agent produced no answer (did not call finish or hit max steps)"

        # Check if predicted entities are missing from name2mid mapping
        name2mid = getattr(self.kg, "_name2mid", None)
        if name2mid:
            missing = [
                p for p in predicted
                if normalize_entity(p) not in name2mid
                and p not in self.kg._adj_out
            ]
            if missing:
                return (
                    "mapping_missing",
                    f"{len(missing)}/{len(predicted)} predicted entities not in "
                    f"name2mid mapping table (e.g. {missing[0]!r})",
                )

        # Check for partial match (right direction but data mismatch)
        pred_forms = [self._resolve_entity_forms(e) for e in predicted]
        gt_forms = [self._resolve_entity_forms(e) for e in ground_truth]
        any_match = any(
            p & g for p in pred_forms for g in gt_forms
        )
        if any_match:
            return (
                "partial_match",
                "Some entities resolved correctly but not fully matched "
                "(incomplete answer or format difference)",
            )

        return "unknown", "Failed to match — reason unclear"

    def _sanitize_filename(self, text: str, max_len: int = 40) -> str:
        """Create a filesystem-safe slug from question text."""
        slug = re.sub(r"[^\w\s-]", "", text.lower())
        slug = re.sub(r"[\s_]+", "_", slug).strip("_")
        return slug[:max_len] if slug else "unnamed"

    def _save_trajectory(
        self,
        trajectory: Trajectory,
        idx: int,
        dataset_index: int,
        question: str,
        predicted: list[str],
        ground_truth: list[str],
        metrics: dict[str, float],
        error_type: str = "",
        error_detail: str = "",
    ) -> None:
        """Persist a single trajectory as .txt (human-readable) and .json (structured)."""
        assert self.trajectory_dir is not None
        os.makedirs(self.trajectory_dir, exist_ok=True)

        slug = self._sanitize_filename(question)
        prefix = f"{self._run_timestamp}_sample_{idx:04d}_{slug}"

        # --- Text format: full reasoning trace ---
        txt_path = os.path.join(self.trajectory_dir, f"{prefix}.txt")
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(f"Question: {question}\n")
            f.write(f"Predicted:  {predicted}\n")
            f.write(f"GroundTruth: {ground_truth}\n")
            f.write(f"Metrics: {metrics}\n")
            f.write(f"Steps: {len(trajectory)}\n")
            if error_type:
                f.write(f"ErrorType: {error_type}\n")
                f.write(f"ErrorDetail: {error_detail}\n")
            f.write("=" * 60 + "\n")
            f.write(trajectory.to_text())
            f.write("\n")

        # --- JSON format: structured, easy to index ---
        json_path = os.path.join(self.trajectory_dir, f"{prefix}.json")
        traj_dict = {
            "timestamp": self._run_timestamp,
            "idx": idx,
            "dataset_index": dataset_index,
            "question": question,
            "predicted": predicted,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "num_steps": len(trajectory),
            "error_type": error_type,
            "error_detail": error_detail,
            "steps": [
                {
                    "thought": step["thought"],
                    "action": step["action"],
                    "observation": step["observation"],
                }
                for step in trajectory.steps
            ],
            "planned_paths": trajectory.planned_paths,
        }
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(traj_dict, f, ensure_ascii=False, indent=2)

        # Record for index
        self._index_entries.append(
            {
                "idx": idx,
                "dataset_index": dataset_index,
                "question": question,
                "predicted": predicted,
                "ground_truth": ground_truth,
                "metrics": metrics,
                "num_steps": len(trajectory),
                "error_type": error_type,
                "error_detail": error_detail,
                "txt_path": txt_path,
                "json_path": json_path,
            }
        )

    def _save_index(self) -> None:
        """Write a single index.json that catalogs all trajectories in this run."""
        assert self.trajectory_dir is not None
        index_path = os.path.join(
            self.trajectory_dir, f"index_{self._run_timestamp}.json"
        )
        index = {
            "run_timestamp": self._run_timestamp,
            "dataset": self.dataset_name,
            "total_samples": len(self._index_entries),
            "samples": self._index_entries,
        }
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(index, f, ensure_ascii=False, indent=2)
        logger.info("Trajectory index saved to %s", index_path)

    def _save_error_summary(self) -> None:
        """Generate a Markdown report grouping failed samples by error cause."""
        assert self.trajectory_dir is not None

        failed = [e for e in self._index_entries if e.get("error_type")]
        if not failed:
            return

        # Group by error_type
        groups: dict[str, list[dict[str, Any]]] = {}
        for entry in failed:
            et = entry.get("error_type", "unknown")
            groups.setdefault(et, []).append(entry)

        md_path = os.path.join(
            self.trajectory_dir, f"errors_summary_{self._run_timestamp}.md"
        )
        lines: list[str] = []
        lines.append(f"# Error Summary ({self._run_timestamp})")
        lines.append("")
        lines.append(f"**Total samples:** {len(self._index_entries)}")
        lines.append(f"**Failed:** {len(failed)}")
        lines.append(f"**Pass rate:** {1 - len(failed)/len(self._index_entries):.1%}")
        lines.append("")

        # Category breakdown table
        lines.append("## Breakdown by Cause")
        lines.append("")
        lines.append("| Category | Count | Description |")
        lines.append("|----------|-------|-------------|")
        for et, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
            detail = entries[0].get("error_detail", "")
            lines.append(f"| {et} | {len(entries)} | {detail[:60]}... |")
        lines.append("")

        # Detailed list per category
        for et, entries in sorted(groups.items(), key=lambda x: -len(x[1])):
            lines.append(f"## {et} ({len(entries)} samples)")
            lines.append("")
            for e in entries:
                idx = e["idx"]
                dataset_index = e.get("dataset_index")
                q = e["question"]
                pred = e["predicted"]
                gt = e["ground_truth"]
                detail = e.get("error_detail", "")
                lines.append(f"### #{idx} (dataset_index={dataset_index}): {q}")
                lines.append(f"- **Predicted:** `{pred}`")
                lines.append(f"- **GroundTruth:** `{gt}`")
                lines.append(f"- **Detail:** {detail}")
                lines.append("")

        with open(md_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logger.info("Error summary saved to %s", md_path)

    def _resolve_entity_forms(self, entity: str) -> set[str]:
        """Return all known forms of an entity using KG name mappings.

        Supports matching between Freebase MID (e.g. m.01_nrx) and
        human-readable names (e.g. 'Bob Hensgens').
        """
        forms = {normalize_entity(entity)}
        ne = normalize_entity(entity)
        name2mid = getattr(self.kg, '_name2mid', None)
        mid2name = getattr(self.kg, '_mid2name', None)
        if name2mid and ne in name2mid:
            forms.add(normalize_entity(name2mid[ne]))
        if mid2name:
            for mid, name in mid2name.items():
                n_name = normalize_entity(name)
                n_mid = normalize_entity(mid)
                if n_name == ne:
                    forms.add(n_mid)
                elif n_mid == ne:
                    forms.add(n_name)
        return forms

    def _compute_metrics(
        self,
        predicted: list[str],
        ground_truth: list[str],
    ) -> dict[str, float]:
        """Compute all configured metrics for a single prediction.

        Resolves entity names via KG mappings so that predicted human-readable
        names can match ground-truth Freebase MIDs (and vice versa).

        Args:
            predicted: Predicted entity names.
            ground_truth: Ground truth entity names.

        Returns:
            Dict of metric_name -> score.
        """
        pred_forms = [self._resolve_entity_forms(e) for e in predicted]
        gt_forms = [self._resolve_entity_forms(e) for e in ground_truth]

        scores = {}
        for metric in self.metrics:
            if metric == "hits@1":
                scores[metric] = self._hits_at_k(pred_forms, gt_forms, k=1)
            elif metric == "hits@3":
                scores[metric] = self._hits_at_k(pred_forms, gt_forms, k=3)
            elif metric == "hits@10":
                scores[metric] = self._hits_at_k(pred_forms, gt_forms, k=10)
            elif metric == "f1":
                scores[metric] = self._f1_score(pred_forms, gt_forms)
            elif metric == "accuracy":
                scores[metric] = self._accuracy(pred_forms, gt_forms)
        return scores

    @staticmethod
    def _hits_at_k(
        pred_forms: list[set[str]], gt_forms: list[set[str]], k: int
    ) -> float:
        if not gt_forms or not pred_forms:
            return 0.0
        for p_set in pred_forms[:k]:
            for g_set in gt_forms:
                if p_set & g_set:
                    return 1.0
        return 0.0

    @staticmethod
    def _f1_score(
        pred_forms: list[set[str]], gt_forms: list[set[str]]
    ) -> float:
        if not pred_forms and not gt_forms:
            return 1.0
        if not pred_forms or not gt_forms:
            return 0.0
        matched_gt = set()
        matched_pred = set()
        for gi, g_set in enumerate(gt_forms):
            for pi, p_set in enumerate(pred_forms):
                if gi not in matched_gt and pi not in matched_pred:
                    if g_set & p_set:
                        matched_gt.add(gi)
                        matched_pred.add(pi)
                        break
        if not matched_gt:
            return 0.0
        precision = len(matched_pred) / len(pred_forms)
        recall = len(matched_gt) / len(gt_forms)
        return 2 * precision * recall / (precision + recall)

    @staticmethod
    def _accuracy(
        pred_forms: list[set[str]], gt_forms: list[set[str]]
    ) -> float:
        if not pred_forms or not gt_forms:
            return 0.0
        for p_set in pred_forms:
            for g_set in gt_forms:
                if p_set & g_set:
                    return 1.0
        return 0.0

    def _log_progress(
        self, all_scores: dict[str, list[float]], n: int
    ) -> None:
        """Log intermediate evaluation results."""
        parts = [f"After {n} samples:"]
        for metric, scores in all_scores.items():
            if scores:
                avg = sum(scores[-n:]) / len(scores[-n:])
                parts.append(f"  {metric}: {avg:.4f}")
        logger.info("\n".join(parts))

    def evaluate_by_hop(
        self,
        test_data: list[dict[str, Any]],
        hop_key: str = "hop",
    ) -> dict[int, dict[str, float]]:
        """Evaluate performance grouped by question hop complexity.

        Args:
            test_data: Test data with hop information.
            hop_key: Key for hop count in data items.

        Returns:
            Dict mapping hop count to metric scores.
        """
        hop_groups: dict[int, list[dict]] = {}
        for item in test_data:
            hop = item.get(hop_key, 1)
            hop_groups.setdefault(hop, []).append(item)

        results = {}
        for hop, items in sorted(hop_groups.items()):
            logger.info(f"Evaluating hop={hop} ({len(items)} samples)")
            results[hop] = self.evaluate(items)

        return results
