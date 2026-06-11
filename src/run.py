"""
SymAgent Main Entry Point.

Provides CLI for running SymAgent in different modes:
  plan           - Run planning only (symbolic rule induction)
  execute        - Run execution only (thought-action-observation loop)
  explore        - Run online exploration only (generate trajectories)
  train          - Run self-learning training loop
  evaluate       - Evaluate on a test dataset
  full_pipeline  - Run the full pipeline (train + evaluate)

Usage:
  python -m src.run plan --config configs/config.yaml --question "..."
  python -m src.run execute --config configs/config.yaml --question "..."
  python -m src.run explore --config configs/config.yaml --dataset webqsp
  python -m src.run train --config configs/config.yaml --dataset webqsp
  python -m src.run evaluate --config configs/config.yaml --dataset webqsp
  python -m src.run full_pipeline --config configs/config.yaml --dataset webqsp
"""

import argparse
import json
import logging
import os
import sys
from typing import Any, Optional

import yaml

from .evaluate import Evaluator
from .executor import AgentExecutor, compute_outcome_reward
from .kg_environment import KGEnvironment
from .llm_client import LLMClient
from .planner import AgentPlanner
from .self_learning import SelfLearner

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration loading
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> dict[str, Any]:
    """Load YAML configuration file."""
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def setup_logging(config: dict[str, Any]) -> None:
    """Configure logging from config."""
    log_cfg = config.get("logging", {})
    log_dir = log_cfg.get("log_dir", "logs")
    os.makedirs(log_dir, exist_ok=True)

    level = getattr(logging, log_cfg.get("log_level", "INFO").upper(), logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(
                os.path.join(log_dir, "symagent.log"), encoding="utf-8"
            ),
        ],
    )


# ---------------------------------------------------------------------------
# Component initialization helpers
# ---------------------------------------------------------------------------

def init_kg(config: dict[str, Any], dataset: str) -> KGEnvironment:
    """Initialize KG environment from config."""
    kg = KGEnvironment()
    ds_cfg = config["kg"]["datasets"][dataset]

    triple_file = ds_cfg.get("triple_file")
    entity2id_file = ds_cfg.get("entity2id")
    relation2id_file = ds_cfg.get("relation2id")

    if triple_file and os.path.exists(triple_file):
        kg.load_from_files(
            triple_file=triple_file,
            entity2id_file=entity2id_file,
            relation2id_file=relation2id_file,
        )
        # Load name mappings for evaluation (mid <-> name resolution)
        mid2name_file = ds_cfg.get("mid2name")
        name2mid_file = ds_cfg.get("name2mid")
        if mid2name_file and os.path.exists(mid2name_file):
            kg.load_name_mapping(mid2name_file, name2mid_file)
        logger.info("KG loaded: %s", kg.get_stats())
    else:
        logger.warning(
            "KG triple file not found: %s. KG will be empty.", triple_file
        )

    return kg


def init_llm(config: dict[str, Any]) -> LLMClient:
    """Initialize LLM client from config."""
    return LLMClient.from_config(config)


def init_planner(
    config: dict[str, Any],
    kg: KGEnvironment,
    llm: LLMClient,
) -> AgentPlanner:
    """Initialize planner from config."""
    planner_cfg = config.get("planner", {})
    return AgentPlanner(
        kg=kg,
        llm=llm,
        num_seed_questions=planner_cfg.get("num_seed_questions", 3),
        max_bfs_depth=planner_cfg.get("max_bfs_depth", 4),
        max_paths_per_seed=planner_cfg.get("max_paths_per_seed", 5),
        max_rules=planner_cfg.get("max_rules", 10),
        planner_temperature=planner_cfg.get("planner_temperature", 0.3),
    )


def init_executor(
    config: dict[str, Any],
    kg: KGEnvironment,
    llm: LLMClient,
    planner: AgentPlanner,
) -> AgentExecutor:
    """Initialize executor from config."""
    executor_cfg = config.get("executor", {})
    return AgentExecutor(
        kg=kg,
        llm=llm,
        planner=planner,
        max_steps=executor_cfg.get("max_infer_steps", 10),
        reasoning_max_depth=executor_cfg.get("reasoning_max_depth", 4),
        reasoning_max_paths=executor_cfg.get("reasoning_max_paths", 10),
        wiki_max_summary_length=executor_cfg.get("wiki_max_summary_length", 2000),
    )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_dataset(
    config: dict[str, Any],
    dataset: str,
    split: str = "test",
) -> list[dict[str, Any]]:
    """Load a dataset split from file.

    Expected JSON format: list of dicts with keys:
      - question: str
      - question_entity: str
      - answer_entities: list[str]
      - hop: int (optional)

    Args:
        config: Full config dict.
        dataset: Dataset name (webqsp, cwq, metaqa).
        split: Data split (train, dev, test).

    Returns:
        List of sample dicts.
    """
    ds_cfg = config["kg"]["datasets"][dataset]
    file_key = f"{split}_file" if split != "valid" else "valid_file"
    filepath = ds_cfg.get(file_key)

    if not filepath or not os.path.exists(filepath):
        logger.error("Dataset file not found: %s", filepath)
        return []

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    logger.info(
        "Loaded %d samples from %s (%s/%s)",
        len(data), filepath, dataset, split,
    )
    return data


# ---------------------------------------------------------------------------
# CLI mode implementations
# ---------------------------------------------------------------------------

def run_plan(args: argparse.Namespace) -> None:
    """Run planning mode: generate symbolic rules.

    Supports two modes:
    - Single question: --question "..." --entity m.xxx
    - Batch (test set): --dataset webqsp [--max_samples N]
    """
    config = load_config(args.config)
    setup_logging(config)

    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)

    # Build seed index from training data
    train_data = load_dataset(config, args.dataset, "train")
    if train_data:
        planner.build_seed_index(train_data)

    if args.question:
        # Single question mode
        logger.info("Planning for: %s", args.question)
        paths = planner.plan(args.question, args.entity)

        print("\n=== Planned Symbolic Rules ===")
        for i, path in enumerate(paths):
            print(f"  Path {i+1}: [{', '.join(path)}]")
        print(f"\nFormatted for executor:")
        print(planner.format_rules_for_prompt(paths))
    else:
        # Batch mode: plan for test set
        test_data = load_dataset(config, args.dataset, "test")
        if not test_data:
            logger.error("No test data found. Exiting.")
            return
        kg.build_bm25_index()

        samples = test_data[:args.max_samples] if args.max_samples else test_data
        results = []
        for i, item in enumerate(samples):
            question = item["question"]
            q_ent = item.get("question_entity", "")
            logger.info("[%d/%d] Planning: %s", i + 1, len(samples), question[:60])
            paths = planner.plan(question, q_ent)
            results.append({"question": question, "planned_paths": paths})

        print(f"\n=== Planning Results ({len(results)} questions) ===")
        for r in results:
            print(f"  Q: {r['question'][:60]}...")
            for j, p in enumerate(r['planned_paths']):
                print(f"    Path {j+1}: [{', '.join(p)}]")

        # Save results
        output_path = args.output or os.path.join(
            config.get("logging", {}).get("log_dir", "logs"), "planned_paths.json"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info("Planned paths saved to %s", output_path)


def run_execute(args: argparse.Namespace) -> None:
    """Run execution mode: answer one or more questions.

    Supports two modes:
    - Single question: --question "..." --entity m.xxx
    - Batch (test set): --dataset webqsp [--max_samples N]
    """
    config = load_config(args.config)
    setup_logging(config)

    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)
    executor = init_executor(config, kg, llm, planner)

    # Build seed index
    train_data = load_dataset(config, args.dataset, "train")
    if train_data:
        planner.build_seed_index(train_data)
    kg.build_bm25_index()

    if args.question:
        # Single question mode
        logger.info("Executing: %s", args.question)
        paths = planner.plan(args.question, args.entity)
        trajectory = executor.execute(args.question, args.entity, paths)

        print("\n=== Execution Trajectory ===")
        print(trajectory.to_text())
        print(f"\n=== Answer ===")
        print(
            ", ".join(trajectory.answer_entities)
            if trajectory.answer_entities
            else "No answer found"
        )
        if args.verbose:
            print(f"\nExtracted triples: {executor.get_extracted_triples()}")
    else:
        # Batch mode: execute on test set
        test_data = load_dataset(config, args.dataset, "test")
        if not test_data:
            logger.error("No test data found. Exiting.")
            return

        samples = test_data[:args.max_samples] if args.max_samples else test_data
        trajectories = []
        for i, item in enumerate(samples):
            question = item["question"]
            q_ent = item.get("question_entity", "")
            a_ents = item.get("answer_entities", [])
            logger.info("[%d/%d] Executing: %s", i + 1, len(samples), question[:60])

            paths = planner.plan(question, q_ent)
            executor.reset()
            trajectory = executor.execute(question, q_ent, paths)
            reward = compute_outcome_reward(trajectory.answer_entities, a_ents, kg=kg)
            trajectory.set_reward(reward)

            trajectories.append({
                "question": question,
                "steps": trajectory.steps,
                "answer_entities": trajectory.answer_entities,
                "ground_truth": a_ents,
                "reward": reward,
            })

        # Summary
        total = len(trajectories)
        rewarded = sum(1 for t in trajectories if t["reward"] > 0)
        avg_reward = sum(t["reward"] for t in trajectories) / total if total else 0
        print(f"\n=== Execution Results ({total} questions) ===")
        print(f"  Successful (reward > 0): {rewarded}/{total}")
        print(f"  Average reward: {avg_reward:.4f}")

        # Save results
        output_path = args.output or os.path.join(
            config.get("logging", {}).get("log_dir", "logs"), "execution_results.json"
        )
        os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(trajectories, f, ensure_ascii=False, indent=2)
        logger.info("Execution results saved to %s", output_path)


def run_explore(args: argparse.Namespace) -> None:
    """Run online exploration: generate trajectories without training.

    This mode runs the agent on a dataset and saves the generated
    trajectories with their rewards. Useful for collecting training data.
    """
    config = load_config(args.config)
    setup_logging(config)

    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)
    executor = init_executor(config, kg, llm, planner)

    # Load data
    data = load_dataset(config, args.dataset, args.split)
    if not data:
        logger.error("No data found for split: %s", args.split)
        return

    # Build seed index
    train_data = load_dataset(config, args.dataset, "train")
    if train_data:
        planner.build_seed_index(train_data)
    kg.build_bm25_index()

    # Limit samples if requested
    samples = data[: args.max_samples] if args.max_samples else data

    trajectories = []
    for i, item in enumerate(samples):
        question = item["question"]
        q_ent = item.get("question_entity", "")
        a_ents = item.get("answer_entities", [])

        logger.info("[%d/%d] Exploring: %s", i + 1, len(samples), question[:60])

        # Plan and execute
        paths = planner.plan(question, q_ent)
        executor.reset()
        trajectory = executor.execute(question, q_ent, paths)

        # Compute reward
        reward = compute_outcome_reward(trajectory.answer_entities, a_ents, kg=kg)
        trajectory.set_reward(reward)

        trajectories.append({
            "question": question,
            "steps": trajectory.steps,
            "answer_entities": trajectory.answer_entities,
            "ground_truth": a_ents,
            "reward": reward,
            "planned_paths": trajectory.planned_paths,
        })

    # Summary
    total = len(trajectories)
    rewarded = sum(1 for t in trajectories if t["reward"] > 0)
    avg_reward = sum(t["reward"] for t in trajectories) / total if total else 0

    print(f"\n=== Exploration Results ===")
    print(f"  Total samples: {total}")
    print(f"  Successful (reward > 0): {rewarded}")
    print(f"  Average reward: {avg_reward:.4f}")

    # Save trajectories
    output_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(
        output_dir, f"{args.dataset}_{args.split}_trajectories.json"
    )
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(trajectories, f, ensure_ascii=False, indent=2)
    logger.info("Trajectories saved to %s", output_path)


def run_train(args: argparse.Namespace) -> None:
    """Run self-learning training loop."""
    config = load_config(args.config)
    setup_logging(config)

    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)
    executor = init_executor(config, kg, llm, planner)

    # Load data
    train_data = load_dataset(config, args.dataset, "train")
    if not train_data:
        logger.error("No training data found. Exiting.")
        return

    valid_data = load_dataset(config, args.dataset, "valid")
    if not valid_data:
        valid_data = None

    # Build seed index
    planner.build_seed_index(train_data)
    kg.build_bm25_index()

    # Initialize self-learner
    sl_cfg = config.get("self_learning", {})
    checkpoint_dir = config.get("logging", {}).get("checkpoint_dir", "checkpoints")

    lora_config = {
        "r": sl_cfg.get("lora_r", 32),
        "lora_alpha": sl_cfg.get("lora_alpha", 32),
        "lora_dropout": sl_cfg.get("lora_dropout", 0.05),
        "target_modules": sl_cfg.get("lora_target_modules", [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "down_proj", "up_proj", "gate_proj",
        ]),
    }

    training_config = {
        "per_device_train_batch_size": sl_cfg.get("batch_size", 4),
        "num_train_epochs": sl_cfg.get("num_train_epochs", 3),
        "learning_rate": sl_cfg.get("learning_rate", 2e-5),
        "max_seq_length": sl_cfg.get("max_seq_length", 4096),
        "warmup_ratio": sl_cfg.get("warmup_ratio", 0.05),
        "gradient_accumulation_steps": sl_cfg.get("gradient_accumulation_steps", 2),
    }

    planner_cfg = config.get("planner", {})
    self_learner = SelfLearner(
        kg=kg,
        llm=llm,
        planner=planner,
        executor=executor,
        num_iterations=sl_cfg.get("num_iterations", 2),
        reward_threshold=sl_cfg.get("reward_threshold", 0.0),
        output_dir=os.path.join(checkpoint_dir, args.dataset),
        lora_config=lora_config,
        training_config=training_config,
        refine_temperature=planner_cfg.get("refine_temperature", 0.3),
    )

    # Run training
    logger.info("Starting self-learning with %d training samples", len(train_data))
    all_pools = self_learner.run_full_loop(train_data, valid_data)

    total_trajectories = sum(len(p) for p in all_pools)
    logger.info("Training complete. Total trajectories: %d", total_trajectories)

    # Save final trajectories
    final_path = os.path.join(
        checkpoint_dir, args.dataset, "final_trajectories.json"
    )
    if all_pools:
        self_learner._save_trajectories(all_pools[-1], final_path)
        logger.info("Final trajectories saved to %s", final_path)


def run_evaluate(args: argparse.Namespace) -> None:
    """Run evaluation on a test dataset."""
    config = load_config(args.config)
    setup_logging(config)

    if args.merge_inputs:
        _run_merge_evaluate(args, config)
        return

    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)
    executor = init_executor(config, kg, llm, planner)

    # Load test data
    test_data = load_dataset(config, args.dataset, "test")
    if not test_data:
        logger.error("No test data found. Exiting.")
        return

    # Optionally build seed index from training data
    train_data = load_dataset(config, args.dataset, "train")
    if train_data:
        planner.build_seed_index(train_data)
    kg.build_bm25_index()

    # Initialize evaluator
    eval_cfg = config.get("evaluation", {})
    results_dir = config.get("logging", {}).get("log_dir", "logs")
    trajectory_dir = os.path.join(results_dir, "trajectories", args.dataset)
    evaluator = Evaluator(
        kg=kg,
        llm=llm,
        planner=planner,
        executor=executor,
        metrics=eval_cfg.get("metrics", ["hits@1", "hits@3", "hits@10", "f1", "accuracy"]),
        trajectory_dir=trajectory_dir,
        dataset_name=args.dataset,
    )

    logger.info("Evaluating on %d test samples", len(test_data))
    results = evaluator.evaluate(
        test_data,
        max_samples=args.max_samples,
        start_index=args.start_index,
        end_index=args.end_index,
        resume_index_file=args.resume_index_file,
    )

    print("\n=== Evaluation Results ===")
    for metric, score in sorted(results.items()):
        print(f"  {metric}: {score:.4f}")

    # Optionally evaluate by hop
    if args.by_hop:
        print("\n=== Results by Hop ===")
        hop_results = evaluator.evaluate_by_hop(test_data)
        for hop, hop_scores in sorted(hop_results.items()):
            print(f"\n  Hop {hop}:")
            for metric, score in sorted(hop_scores.items()):
                print(f"    {metric}: {score:.4f}")

    # Save results
    results_dir = config.get("logging", {}).get("log_dir", "logs")
    os.makedirs(results_dir, exist_ok=True)
    results_path = args.output or os.path.join(results_dir, f"{args.dataset}_eval_results.json")
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    logger.info("Results saved to %s", results_path)


def _run_merge_evaluate(args: argparse.Namespace, config: dict[str, Any]) -> None:
    """Merge multiple evaluation index files and recompute metrics."""
    kg = init_kg(config, args.dataset)
    llm = init_llm(config)
    planner = init_planner(config, kg, llm)
    executor = init_executor(config, kg, llm, planner)

    results_dir = config.get("logging", {}).get("log_dir", "logs")
    trajectory_dir = os.path.join(results_dir, "trajectories", args.dataset)
    evaluator = Evaluator(
        kg=kg,
        llm=llm,
        planner=planner,
        executor=executor,
        metrics=config.get("evaluation", {}).get("metrics", ["hits@1", "hits@3", "hits@10", "f1", "accuracy"]),
        trajectory_dir=trajectory_dir,
        dataset_name=args.dataset,
    )

    index_files = [p.strip() for p in args.merge_inputs.split(",") if p.strip()]
    merged_results = evaluator.merge_index_files(index_files, output_dir=args.merge_output_dir)

    print("\n=== Merged Evaluation Results ===")
    for metric, score in sorted(merged_results.items()):
        print(f"  {metric}: {score:.4f}")



def run_full_pipeline(args: argparse.Namespace) -> None:
    """Run the full pipeline: train then evaluate."""
    config = load_config(args.config)
    setup_logging(config)

    logger.info("=" * 60)
    logger.info("SymAgent Full Pipeline: dataset=%s", args.dataset)
    logger.info("=" * 60)

    # Phase 1: Train
    logger.info("Phase 1: Self-Learning Training")
    train_args = argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
    )
    run_train(train_args)

    # Phase 2: Evaluate
    logger.info("Phase 2: Evaluation")
    eval_args = argparse.Namespace(
        config=args.config,
        dataset=args.dataset,
        max_samples=args.max_samples,
        by_hop=args.by_hop,
        output=None,
        start_index=0,
        end_index=None,
        resume_index_file=None,
        merge_inputs=None,
        merge_output_dir=None,
    )
    run_evaluate(eval_args)

    logger.info("Full pipeline complete.")


# ---------------------------------------------------------------------------
# CLI argument parser
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    """Build the argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        prog="symagent",
        description="SymAgent: Neural-Symbolic Self-Learning Agent for KGQA",
    )
    parser.add_argument(
        "--config", type=str, default="configs/config.yaml",
        help="Path to config file (default: configs/config.yaml)",
    )

    subparsers = parser.add_subparsers(dest="mode", help="Run mode")

    # --- plan ---
    plan_parser = subparsers.add_parser("plan", help="Run planning (symbolic rule induction)")
    plan_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")
    plan_parser.add_argument("--question", type=str, default=None, help="Question to plan for (omit for batch mode)")
    plan_parser.add_argument("--entity", type=str, default=None, help="Question entity")
    plan_parser.add_argument("--max_samples", type=int, default=None, help="Max samples (batch mode)")
    plan_parser.add_argument("--output", type=str, default=None, help="Output file path")

    # --- execute ---
    exec_parser = subparsers.add_parser("execute", help="Run execution (answer a question)")
    exec_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")
    exec_parser.add_argument("--question", type=str, default=None, help="Question to answer (omit for batch mode)")
    exec_parser.add_argument("--entity", type=str, default=None, help="Question entity")
    exec_parser.add_argument("--max_samples", type=int, default=None, help="Max samples (batch mode)")
    exec_parser.add_argument("--output", type=str, default=None, help="Output file path")
    exec_parser.add_argument("--verbose", action="store_true", help="Show extracted triples")

    # --- explore ---
    explore_parser = subparsers.add_parser("explore", help="Run online exploration (generate trajectories)")
    explore_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")
    explore_parser.add_argument("--split", type=str, default="train", help="Data split to explore")
    explore_parser.add_argument("--max_samples", type=int, default=None, help="Max samples to explore")

    # --- train ---
    train_parser = subparsers.add_parser("train", help="Run self-learning training")
    train_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")

    # --- evaluate ---
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate on test set")
    eval_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")
    eval_parser.add_argument("--max_samples", type=int, default=None, help="Max test samples")
    eval_parser.add_argument("--by_hop", action="store_true", help="Group results by hop")
    eval_parser.add_argument("--output", type=str, default=None, help="Output file path")
    eval_parser.add_argument("--start_index", type=int, default=0, help="Start dataset index (inclusive)")
    eval_parser.add_argument("--end_index", type=int, default=None, help="End dataset index (exclusive)")
    eval_parser.add_argument("--resume_index_file", type=str, default=None, help="Prior index_*.json file to skip completed samples")
    eval_parser.add_argument("--merge_inputs", type=str, default=None, help="Comma-separated index_*.json files to merge")
    eval_parser.add_argument("--merge_output_dir", type=str, default=None, help="Directory to write merged artifacts")

    # --- full_pipeline ---
    full_parser = subparsers.add_parser("full_pipeline", help="Full pipeline: train + evaluate")
    full_parser.add_argument("--dataset", type=str, default="webqsp", help="Dataset name")
    full_parser.add_argument("--max_samples", type=int, default=None, help="Max test samples")
    full_parser.add_argument("--by_hop", action="store_true", help="Group results by hop")

    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

MODE_DISPATCH = {
    "plan": run_plan,
    "execute": run_execute,
    "explore": run_explore,
    "train": run_train,
    "evaluate": run_evaluate,
    "full_pipeline": run_full_pipeline,
}


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode is None:
        parser.print_help()
        sys.exit(1)

    dispatch = MODE_DISPATCH.get(args.mode)
    if dispatch is None:
        parser.error(f"Unknown mode: {args.mode}")

    dispatch(args)


if __name__ == "__main__":
    main()
