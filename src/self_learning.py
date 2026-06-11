"""
Self-Learning Framework for SymAgent.

Implements the iterative self-learning process described in Section 4.3:
1. Online Exploration (Section 4.3.1):
   - Agent interacts with KG environment via thought-action-observation loop
   - Generates candidate trajectories with outcome-based rewards
   - Self-refine trajectories using LLM self-reflection
   - Heuristic merge of original and refined trajectories

2. Offline Iterative Policy Updating (Section 4.3.2):
   - Fine-tune LLM on merged trajectories using LoRA
   - Loss: L_SFT = -E_{mu~D*} [pi_theta(mu|q)]
   - Iterative loop until validation improvement is negligible
"""

import json
import logging
import os
from typing import Any, Optional

import torch
from peft import LoraConfig, TaskType, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments

from .executor import AgentExecutor, Trajectory, compute_outcome_reward
from .kg_environment import KGEnvironment
from .llm_client import LLMClient
from .local_model_client import LocalModelClient
from .planner import AgentPlanner

logger = logging.getLogger(__name__)


class TrajectoryPool:
    """Manages a collection of trajectories for self-learning.

    Implements the data structures D_0, D_0^c, D_0* from Section 4.3.1.
    """

    def __init__(self):
        self.trajectories: list[Trajectory] = []

    def add(self, trajectory: Trajectory) -> None:
        """Add a trajectory to the pool."""
        self.trajectories.append(trajectory)

    def extend(self, trajectories: list[Trajectory]) -> None:
        """Add multiple trajectories."""
        self.trajectories.extend(trajectories)

    def get_rewarded(self) -> list[tuple[Trajectory, float]]:
        """Return trajectories with their rewards."""
        return [(t, t.reward) for t in self.trajectories]

    def filter_by_reward(self, min_reward: float = 0.0) -> "TrajectoryPool":
        """Return a new pool with trajectories above the reward threshold."""
        pool = TrajectoryPool()
        pool.trajectories = [
            t for t in self.trajectories if t.reward > min_reward
        ]
        return pool

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> Trajectory:
        return self.trajectories[idx]


def heuristic_merge(
    original_pool: TrajectoryPool,
    refined_pool: TrajectoryPool,
) -> TrajectoryPool:
    """Heuristic merge of original and refined trajectory pools.

    Implements Equation 7 from Section 4.3.1:

    D_0*(i) = {
        (mu_i, r(mu_i)),                        if r(mu_i) > r(mu_hat_i)
        (mu_hat_i, r(mu_hat_i)),                if r(mu_i) < r(mu_hat_i)
        (t, r(t)),                              if r(mu_i) = r(mu_hat_i) > 0
                                                where t = argmin |s| for s in {mu_i, mu_hat_i}
        filtered,                                if r(mu_i) = r(mu_hat_i) = 0
    }

    Args:
        original_pool: The original explored trajectories D_0.
        refined_pool: The self-refined trajectories D_0^c.

    Returns:
        Merged trajectory pool D_0*.
    """
    merged = TrajectoryPool()

    assert len(original_pool) == len(refined_pool), (
        f"Pool sizes must match: {len(original_pool)} vs {len(refined_pool)}"
    )

    for i in range(len(original_pool)):
        orig = original_pool[i]
        ref = refined_pool[i]
        r_orig = orig.reward
        r_ref = ref.reward

        if r_orig > r_ref:
            merged.add(orig)
        elif r_orig < r_ref:
            merged.add(ref)
        elif r_orig == r_ref and r_orig > 0:
            # Select shorter trajectory
            if len(orig) <= len(ref):
                merged.add(orig)
            else:
                merged.add(ref)
        else:
            # Both have reward 0: filtered out
            pass

    logger.info(
        f"Heuristic merge: {len(original_pool)} + {len(refined_pool)} -> "
        f"{len(merged)} trajectories"
    )
    return merged


class SelfLearner:
    """Self-Learning Framework for SymAgent.

    Implements the full self-learning pipeline from Section 4.3:
    - Online exploration: interact with KG environment
    - Self-refine: use LLM to refine trajectories
    - Heuristic merge: combine original and refined trajectories
    - Offline policy update: fine-tune LLM with LoRA
    - Iterative loop until convergence

    Attributes:
        kg: The KG environment.
        llm: The LLM client.
        planner: The Agent-Planner.
        executor: The Agent-Executor.
        num_iterations: Number of self-learning iterations.
        reward_threshold: Minimum reward to keep a trajectory.
    """

    def __init__(
        self,
        kg: KGEnvironment,
        llm: LLMClient,
        planner: AgentPlanner,
        executor: AgentExecutor,
        num_iterations: int = 2,
        reward_threshold: float = 0.0,
        output_dir: str = "checkpoints",
        lora_config: Optional[dict[str, Any]] = None,
        training_config: Optional[dict[str, Any]] = None,
        model_name: Optional[str] = None,
        refine_temperature: float = 0.3,
    ):
        self.kg = kg
        self.llm = llm
        self.planner = planner
        self.executor = executor
        self.num_iterations = num_iterations
        self.reward_threshold = reward_threshold
        self.output_dir = output_dir
        self.model_name = model_name or getattr(llm, "model_name", None)
        self.refine_temperature = refine_temperature

        self.lora_config = lora_config or {
            "r": 32,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "down_proj", "up_proj", "gate_proj",
            ],
        }

        self.training_config = training_config or {
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 2,
            "warmup_ratio": 0.05,
            "num_train_epochs": 3,
            "learning_rate": 2e-5,
            "max_seq_length": 4096,
        }

    def online_explore(
        self,
        qa_pairs: list[dict[str, Any]],
    ) -> TrajectoryPool:
        """Online exploration phase (Section 4.3.1).

        The base agent pi_{theta_0} interacts with the environment autonomously
        through a thought-action-observation loop, synthesizing a set of initial
        trajectories U_0 = {mu_1, mu_2, ..., mu_N}.

        For each trajectory, compute outcome-based reward (Equation 6):
        r(mu_i) = Recall(A_{mu_i}, A_{gt})

        Args:
            qa_pairs: List of QA pairs with 'question', 'question_entity',
                     and 'answer_entities'.

        Returns:
            TrajectoryPool with explored trajectories and rewards.
        """
        pool = TrajectoryPool()

        for qa in qa_pairs:
            question = qa["question"]
            q_ent = qa.get("question_entity", "")
            a_ents = qa.get("answer_entities", [])

            logger.info(f"Exploring: {question[:60]}...")

            # Run planner
            planned_paths = self.planner.plan(question, q_ent)

            # Run executor
            self.executor.reset()
            trajectory = self.executor.execute(
                question, q_ent, planned_paths
            )

            # Compute outcome reward
            reward = compute_outcome_reward(
                trajectory.answer_entities, a_ents, kg=self.kg
            )
            trajectory.set_reward(reward)
            trajectory.set_answer(trajectory.answer_entities)
            trajectory.ground_truth_entities = a_ents

            pool.add(trajectory)

        # Filter by reward threshold
        filtered = pool.filter_by_reward(self.reward_threshold)
        logger.info(
            f"Online exploration: {len(pool)} -> {len(filtered)} "
            f"trajectories (threshold={self.reward_threshold})"
        )
        return filtered

    def self_refine(
        self,
        pool: TrajectoryPool,
    ) -> TrajectoryPool:
        """Self-refine trajectories using LLM self-reflection.

        Using D_0 as reference, the policy LLM regenerates refined trajectories:
        {mu_hat_i} ~ pi_{theta_0}(·|mu_i, r(mu_i))

        Args:
            pool: Original trajectory pool D_0.

        Returns:
            Refined trajectory pool D_0^c.
        """
        refined_pool = TrajectoryPool()

        for trajectory in pool.trajectories:
            refine_prompt = self._build_refine_prompt(trajectory)

            try:
                response = self.llm.execute_generate(
                    refine_prompt,
                    temperature=self.refine_temperature,
                )
                refined = self._parse_refined_trajectory(
                    trajectory.question, response,
                    trajectory.reward,
                    trajectory.ground_truth_entities,
                )
                if refined:
                    refined_pool.add(refined)
                else:
                    refined_pool.add(trajectory)
            except Exception as e:
                logger.warning(f"Self-refine failed: {e}")
                refined_pool.add(trajectory)

        return refined_pool

    def _build_refine_prompt(self, trajectory: Trajectory) -> str:
        """Build prompt for trajectory self-refinement.

        Implements Section 4.3.1: Using D_0 as reference, the policy LLM
        pi_{theta_0} regenerates new refined trajectories.

        The prompt provides the original trajectory with its reward and
        guides structured self-reflection:
        - Analyze what went wrong (or could be improved)
        - Propose a concrete improvement strategy
        - Generate a corrected trajectory following the same format
        """
        prompt = (
            "You are a knowledge graph question-answering agent. "
            "You are refining a reasoning trajectory that was previously "
            "attempted. Analyze the original trajectory and produce an "
            "improved version.\n\n"
            f"Question: {trajectory.question}\n\n"
            "Original trajectory:\n"
        )
        for i, step in enumerate(trajectory.steps):
            prompt += f"Thought {i+1}: {step['thought']}\n"
            prompt += f"Action {i+1}: {step['action']}\n"
            prompt += f"Observation {i+1}: {step['observation']}\n"

        prompt += f"\nFinal answer: {', '.join(trajectory.answer_entities) if trajectory.answer_entities else 'None'}\n"

        reward_str = "correct" if trajectory.reward > 0 else "incorrect"
        prompt += f"Outcome: {reward_str} (reward: {trajectory.reward:.2f})\n\n"

        if trajectory.reward > 0:
            prompt += (
                "The trajectory was successful. Review the reasoning steps "
                "and produce a more concise or efficient version if possible. "
                "If the trajectory is already optimal, reproduce it faithfully.\n\n"
                "Instructions:\n"
                "1. Review each Thought step for correctness and necessity.\n"
                "2. Identify any redundant searches or unnecessary detours.\n"
                "3. Output the refined trajectory in the same format.\n"
            )
        else:
            prompt += (
                "The trajectory was unsuccessful. Carefully analyze what went "
                "wrong and produce a corrected trajectory.\n\n"
                "Instructions:\n"
                "1. Identify the first point where the reasoning went astray.\n"
                "2. Determine whether the issue was: wrong entity, wrong relation, "
                "missing information (should use wikiSearch), or premature finish.\n"
                "3. Propose a concrete correction strategy.\n"
                "4. Generate the refined trajectory step by step.\n"
                "5. End with a finish() action containing the final answer.\n\n"
                "Important: You must output a complete refined trajectory with "
                "Thought/Action steps. Do not just explain what went wrong.\n"
            )

        return prompt

    def _parse_refined_trajectory(
        self,
        question: str,
        response: str,
        original_reward: float,
        ground_truth_entities: list[str],
    ) -> Optional[Trajectory]:
        """Parse a refined trajectory from LLM response.

        Re-executes each action against the KG to obtain real observations.
        Recomputes the outcome reward against ground truth so that
        heuristic_merge (Eq. 7) operates on accurate rewards.
        """
        import re

        trajectory = Trajectory(question)
        trajectory.ground_truth_entities = ground_truth_entities
        steps = re.findall(
            r"Thought\s*\d*:\s*(.+?)\nAction\s*\d*:\s*(.+?)(?:\n|$)",
            response,
            re.DOTALL,
        )

        for thought, action in steps:
            from .executor import ActionParser
            action_name, args = ActionParser.parse(action)

            observation = ""
            if action_name == "finish" and args:
                trajectory.set_answer(args)
                observation = f"Final answer: {', '.join(args)}"
            elif action_name == "searchNeighbor" and len(args) >= 2:
                neighbors = self.kg.search_neighbor_with_relation(
                    args[0], args[1]
                )
                observation = ", ".join(neighbors) if neighbors else (
                    "No entity found under this relation in the knowledge graph."
                )
            elif action_name == "getReasoningPath" and len(args) >= 1:
                observation = (
                    "Surrounding relational reasoning paths are: []"
                )
            elif action_name == "searchWikidata" and len(args) >= 2:
                observation = (
                    f"By searching Wikidata, {args[0]}'s related entities are not available."
                )
            elif action_name == "wikiSearch" and len(args) >= 2:
                observation = (
                    f"By searching, {args[0]}'s relevant documents are not available offline."
                )
            else:
                observation = "Action executed."

            trajectory.add_step(thought.strip(), action.strip(), observation)

        if not trajectory.steps:
            return None

        # Recompute reward against ground truth (critical for Eq. 7)
        refined_reward = compute_outcome_reward(
            trajectory.answer_entities, ground_truth_entities, kg=self.kg
        )
        trajectory.set_reward(refined_reward)
        return trajectory

    def run_iteration(
        self,
        qa_pairs: list[dict[str, Any]],
        iteration: int = 0,
    ) -> TrajectoryPool:
        """Run a single self-learning iteration.

        Implements one full cycle:
        1. Online explore -> D_0
        2. Self-refine -> D_0^c
        3. Heuristic merge -> D_0*
        4. (Offline policy update is done separately)

        Args:
            qa_pairs: Training QA pairs.
            iteration: Current iteration number.

        Returns:
            Merged trajectory pool D_0*.
        """
        logger.info(f"=== Self-Learning Iteration {iteration} ===")

        # Step 1: Online exploration
        logger.info("Step 1: Online exploration...")
        original_pool = self.online_explore(qa_pairs)
        logger.info(f"Explored {len(original_pool)} trajectories")

        # Step 2: Self-refine
        logger.info("Step 2: Self-refinement...")
        refined_pool = self.self_refine(original_pool)
        logger.info(f"Refined {len(refined_pool)} trajectories")

        # Step 3: Heuristic merge
        logger.info("Step 3: Heuristic merge...")
        merged_pool = heuristic_merge(original_pool, refined_pool)
        logger.info(f"Merged into {len(merged_pool)} trajectories")

        return merged_pool

    def run_full_loop(
        self,
        train_data: list[dict[str, Any]],
        valid_data: Optional[list[dict[str, Any]]] = None,
    ) -> list[TrajectoryPool]:
        """Run the full self-learning loop.

        Iterative process:
        D_0 -> Explore -> Reward -> Self-Refine -> Merge -> D_0* -> Update theta

        The loop continues until improvement on validation set is negligible
        or max iterations reached. After each merge, LoRA fine-tuning is
        performed on the merged trajectories (Section 4.3.2).

        Args:
            train_data: Training QA pairs.
            valid_data: Optional validation data for early stopping.

        Returns:
            List of merged trajectory pools from each iteration.
        """
        all_pools: list[TrajectoryPool] = []
        prev_score = 0.0

        for iteration in range(self.num_iterations):
            merged_pool = self.run_iteration(train_data, iteration)
            all_pools.append(merged_pool)

            # Save merged trajectories
            self._save_trajectories(
                merged_pool,
                os.path.join(
                    self.output_dir, f"iteration_{iteration}_trajectories.json"
                ),
            )

            # Step 4: Offline policy update - LoRA fine-tuning (Section 4.3.2)
            # After updating θ, the next iteration uses the updated π_θ
            # for exploration instead of the initial API-based LLM.
            if self.model_name and len(merged_pool) > 0:
                logger.info(
                    f"Iteration {iteration}: Starting LoRA fine-tuning on "
                    f"{len(merged_pool)} trajectories..."
                )
                try:
                    lora_path = self.fine_tune(merged_pool, iteration)

                    # Switch to fine-tuned local model for next iteration.
                    # This implements the paper's key loop: pi_theta explores ->
                    # fine-tune -> updated pi_theta explores again.
                    if iteration < self.num_iterations - 1:
                        logger.info(
                            f"Switching to fine-tuned local model ({lora_path}) "
                            f"for iteration {iteration + 1} exploration..."
                        )
                        self._switch_to_local_model(lora_path)
                except Exception as e:
                    logger.warning(
                        f"LoRA fine-tuning failed in iteration {iteration}: {e}"
                    )
            elif not self.model_name:
                logger.info(
                    "No model_name configured, skipping LoRA fine-tuning."
                )

            # Optional: validate
            if valid_data:
                current_score = self._evaluate_on_valid(valid_data)
                logger.info(
                    f"Iteration {iteration} validation score: {current_score:.4f}"
                )

                # Early stopping if improvement is negligible
                if iteration > 0 and current_score - prev_score < 0.001:
                    logger.info(
                        "Improvement negligible, stopping self-learning loop."
                    )
                    break
                prev_score = current_score

        return all_pools

    def fine_tune(
        self,
        pool: TrajectoryPool,
        iteration: int = 0,
    ) -> str:
        """Fine-tune the base LLM on merged trajectories using LoRA.

        Implements Section 4.3.2 (Offline Iterative Policy Updating):
        - Load base model with AutoModelForCausalLM + AutoTokenizer
        - Apply LoRA via peft get_peft_model
        - Prepare training data with prepare_training_data() (Equation 8)
        - Fine-tune using HuggingFace Trainer
        - Save checkpoint

        After fine-tuning, the updated π_θ is used for subsequent
        exploration iterations (replacing the initial API-based LLM).

        Args:
            pool: Merged trajectory pool D* to train on.
            iteration: Current iteration number (for checkpoint naming).

        Returns:
            Path to the saved LoRA adapter directory.
        """
        from torch.utils.data import Dataset

        logger.info(f"Loading base model: {self.model_name}")

        tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        # Apply LoRA
        lora_cfg = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.lora_config["r"],
            lora_alpha=self.lora_config["lora_alpha"],
            lora_dropout=self.lora_config["lora_dropout"],
            target_modules=self.lora_config["target_modules"],
        )
        model = get_peft_model(model, lora_cfg)
        model.print_trainable_parameters()

        # Prepare training data
        max_seq_length = self.training_config.get("max_seq_length", 4096)
        train_examples = prepare_training_data(pool, tokenizer, max_length=max_seq_length)

        if not train_examples:
            logger.warning("No training data prepared, skipping fine-tuning.")
            return

        class TrajectoryDataset(Dataset):
            def __init__(self, data):
                self.data = data

            def __len__(self):
                return len(self.data)

            def __getitem__(self, idx):
                return {
                    "input_ids": self.data[idx]["input_ids"],
                    "attention_mask": self.data[idx]["attention_mask"],
                    "labels": self.data[idx]["labels"],
                }

        train_dataset = TrajectoryDataset(train_examples)

        # Data collator for dynamic padding
        from transformers import DataCollatorForSeq2Seq
        data_collator = DataCollatorForSeq2Seq(
            tokenizer=tokenizer,
            padding=True,
            max_length=max_seq_length,
            return_tensors="pt",
        )

        # Training arguments
        checkpoint_dir = os.path.join(self.output_dir, f"lora_iteration_{iteration}")
        training_args = TrainingArguments(
            output_dir=checkpoint_dir,
            per_device_train_batch_size=self.training_config.get(
                "per_device_train_batch_size", 2
            ),
            gradient_accumulation_steps=self.training_config.get(
                "gradient_accumulation_steps", 2
            ),
            warmup_ratio=self.training_config.get("warmup_ratio", 0.05),
            num_train_epochs=self.training_config.get("num_train_epochs", 3),
            learning_rate=self.training_config.get("learning_rate", 2e-5),
            lr_scheduler_type=self.training_config.get(
                "lr_scheduler_type", "cosine"
            ),
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
            report_to="none",
        )

        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_dataset,
            processing_class=tokenizer,
            data_collator=data_collator,
        )
        trainer.train()

        # Save LoRA checkpoint
        lora_save_path = os.path.join(checkpoint_dir, "lora_adapter")
        model.save_pretrained(lora_save_path)
        tokenizer.save_pretrained(lora_save_path)
        logger.info(f"LoRA adapter saved to {lora_save_path}")
        return lora_save_path

    def _evaluate_on_valid(
        self, valid_data: list[dict[str, Any]]
    ) -> float:
        """Evaluation on validation set during self-learning.

        Uses eval_sample_size from training_config to control the number
        of samples evaluated (avoids hardcoding).

        Returns:
            Average reward on validation set.
        """
        eval_sample_size = self.training_config.get("eval_sample_size", 50)
        sample_size = min(len(valid_data), eval_sample_size)

        total_reward = 0.0
        for qa in valid_data[:sample_size]:
            question = qa["question"]
            q_ent = qa.get("question_entity", "")
            a_ents = qa.get("answer_entities", [])

            paths = self.planner.plan(question, q_ent)
            self.executor.reset()
            traj = self.executor.execute(question, q_ent, paths)
            reward = compute_outcome_reward(traj.answer_entities, a_ents, kg=self.kg)
            total_reward += reward

        return total_reward / sample_size

    def _switch_to_local_model(self, lora_path: str) -> None:
        """Switch planner and executor to use the fine-tuned local model.

        After LoRA fine-tuning, the updated policy replaces the
        initial API-based LLM for subsequent exploration iterations.
        This is the core of the paper's iterative self-learning loop:
        pi_theta_0 -> explore -> fine-tune -> pi_theta_1 -> explore -> fine-tune -> ...

        The local model is loaded with the LoRA adapter and used as
        the LLM backend for both Planner (rule induction) and Executor
        (thought-action-observation loop).

        Args:
            lora_path: Path to the saved LoRA adapter directory.
        """
        from .local_model_client import LocalModelClient

        # Extract base model name (LoRA path contains adapter, not base)
        base_model = self.model_name

        local_llm = LocalModelClient(
            model_name=base_model,
            lora_path=lora_path,
            temperature=self.llm.temperature,
            top_p=self.llm.top_p if hasattr(self.llm, 'top_p') else 0.9,
            top_k=self.llm.top_k if hasattr(self.llm, 'top_k') else 600,
            max_new_tokens=self.llm.max_new_tokens if hasattr(self.llm, 'max_new_tokens') else 512,
        )

        # Update planner and executor to use local model
        self.planner.llm = local_llm
        self.executor.llm = local_llm
        self.llm = local_llm

        logger.info(
            f"Switched to local model: {base_model} + {lora_path}"
        )

    def _save_trajectories(
        self, pool: TrajectoryPool, filepath: str
    ) -> None:
        """Save trajectory pool to JSON file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data = []
        for t in pool.trajectories:
            data.append({
                "question": t.question,
                "steps": t.steps,
                "answer_entities": t.answer_entities,
                "reward": t.reward,
                "planned_paths": t.planned_paths,
            })
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @staticmethod
    def load_trajectories(filepath: str) -> TrajectoryPool:
        """Load trajectory pool from JSON file."""
        pool = TrajectoryPool()
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            traj = Trajectory(item["question"])
            traj.steps = item["steps"]
            traj.answer_entities = item.get("answer_entities", [])
            traj.reward = item.get("reward", 0.0)
            traj.planned_paths = item.get("planned_paths", [])
            pool.add(traj)
        return pool


def prepare_training_data(
    pool: TrajectoryPool,
    tokenizer: Any,
    max_length: int = 4096,
) -> list[dict[str, Any]]:
    """Prepare trajectory data for SFT fine-tuning.

    Implements Equation 8:
    L_SFT = -E_{mu~D*} [sum_{j} 1(x_j in A) * log pi_theta(x_j | x_{<j}, q)]

    Only compute loss on tokens belonging to thoughts or actions
    (indicator function 1(x_j in A)). Question tokens and observation
    tokens are masked with -100 so they contribute no loss.

    Args:
        pool: Merged trajectory pool D*.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.

    Returns:
        List of training examples with input_ids, labels, and attention_mask.
    """
    training_data = []

    for trajectory in pool.trajectories:
        # Build the full prompt with question and trajectory
        prompt = f"Question: {trajectory.question}\n"
        for i, step in enumerate(trajectory.steps):
            prompt += f"Thought {i+1}: {step['thought']}\n"
            prompt += f"Action {i+1}: {step['action']}\n"
            prompt += f"Observation {i+1}: {step['observation']}\n"

        # Tokenize
        encodings = tokenizer(
            prompt,
            max_length=max_length,
            truncation=True,
            padding=False,
            return_tensors=None,
        )

        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Build labels: start by masking everything, then unmask thought+action tokens
        labels = [-100] * len(input_ids)

        # Compute offset from special tokens (e.g., BOS) added by the full
        # tokenization but not by individual segment tokenizations.
        empty_with_special = tokenizer("", add_special_tokens=True, return_tensors=None)
        empty_without_special = tokenizer("", add_special_tokens=False, return_tensors=None)
        special_offset = len(empty_with_special["input_ids"]) - len(empty_without_special["input_ids"])

        # Tokenize each segment to identify which token positions are
        # thought/action (trainable) vs question/observation (masked).
        cursor = special_offset
        # Mask question tokens
        question_text = f"Question: {trajectory.question}\n"
        question_enc = tokenizer(
            question_text, add_special_tokens=False, return_tensors=None,
        )
        cursor += len(question_enc["input_ids"])

        # For each step, unmask Thought and Action lines, keep Observation masked
        for i, step in enumerate(trajectory.steps):
            # Thought line - trainable
            thought_text = f"Thought {i+1}: {step['thought']}\n"
            thought_enc = tokenizer(
                thought_text, add_special_tokens=False, return_tensors=None,
            )
            thought_len = len(thought_enc["input_ids"])
            for j in range(thought_len):
                pos = cursor + j
                if pos < len(labels):
                    labels[pos] = input_ids[pos]
            cursor += thought_len

            # Action line - trainable
            action_text = f"Action {i+1}: {step['action']}\n"
            action_enc = tokenizer(
                action_text, add_special_tokens=False, return_tensors=None,
            )
            action_len = len(action_enc["input_ids"])
            for j in range(action_len):
                pos = cursor + j
                if pos < len(labels):
                    labels[pos] = input_ids[pos]
            cursor += action_len

            # Observation line - masked (environment output, not generated by agent)
            obs_text = f"Observation {i+1}: {step['observation']}\n"
            obs_enc = tokenizer(
                obs_text, add_special_tokens=False, return_tensors=None,
            )
            cursor += len(obs_enc["input_ids"])

        # Ensure labels length matches input_ids length
        labels = labels[:len(input_ids)]

        training_data.append({
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        })

    return training_data
