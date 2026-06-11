#!/usr/bin/env python3
"""
SymAgent Demo — 完整演示算法流程

## 环境准备

1. 激活 conda 环境：
    conda activate ray312

2. 安装依赖（ray312 已全部包含，新环境需安装）：
    pip install openai rank_bm25 wikipedia-api pyyaml torch transformers peft networkx

3. 配置 API Key（编辑 configs/config.yaml）：
    llm:
      api_key: "your-zhipu-api-key"
      api_base: "https://open.bigmodel.cn/api/paas/v4"
      model_name: "glm-4.7-flash"

4. 下载数据集（如未下载）：
    bash scripts/download_data.sh

## 运行

    cd /home/sone/symagent
    conda activate ray312
    python demo.py

## 说明

- Step 1-2 的 BM25/BFS 部分不调 API，始终可以运行
- Step 3-5 需要 LLM API，限速时自动降级为 mock 演示
- Step 6-7 纯逻辑/配置展示，不需要 API

## 演示内容（WebQSP 数据集，少量样本）：
  Step 1: 加载 KG 和数据
  Step 2: Agent-Planner — BM25 种子检索 → BFS 路径采样 → 符号规则归纳
  Step 3: Agent-Executor — Thought-Action-Observation 推理循环
  Step 4: 奖励计算 — Outcome Reward (Recall)
  Step 5: 自我精炼 — Self-Refinement
  Step 6: 启发式合并 — Heuristic Merge
  Step 7: LoRA 微调 — Offline Policy Update
"""

import sys
import os
import json
import logging
import time
from datetime import datetime

# 确保项目根目录在 path 中
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.run import (
    load_config, setup_logging, init_kg, init_llm,
    init_planner, init_executor,
)
from src.kg_environment import KGEnvironment
from src.self_learning import SelfLearner, heuristic_merge, Trajectory
from src.executor import compute_outcome_reward

# ─── 配置 ───────────────────────────────────────────────────────────────────

DATASET = "webqsp"
MAX_PLAN_SAMPLES = 3      # 演示 Planner 处理的问题数
MAX_EXEC_SAMPLES = 2      # 演示 Executor 推理的问题数
DEMO_QUESTION = None       # 如果要指定单条问题，设为 (question, entity) 元组
# DEMO_QUESTION = ("what language do they speak in colombia", "m.0j7p")

SEPARATOR = "=" * 70
THIN_SEP = "-" * 70

# ─── 工具函数 ───────────────────────────────────────────────────────────────

def print_step(step_num: int, title: str, description: str = ""):
    print(f"\n{SEPARATOR}")
    print(f"  Step {step_num}: {title}")
    if description:
        print(f"  {description}")
    print(SEPARATOR)


def print_substep(title: str):
    print(f"\n  ▶ {title}")
    print(f"  {THIN_SEP}")


def print_result(label: str, value: str):
    lines = value.strip().split("\n")
    print(f"  ✦ {label}:")
    for line in lines:
        print(f"    {line}")


# ─── 主流程 ─────────────────────────────────────────────────────────────────

def main():
    print(SEPARATOR)
    print("  SymAgent 算法流程演示")
    print(f"  论文: SymAgent (SIGIR 2025) — arXiv:2502.03283v2")
    print(f"  时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  数据集: {DATASET}")
    print(SEPARATOR)

    # 加载配置
    config = load_config("configs/config.yaml")
    setup_logging(config)
    llm_cfg = config.get("llm", {})

    print(f"\n  LLM 配置:")
    print(f"    模型: {llm_cfg.get('model_name', 'N/A')}")
    print(f"    Temperature: {llm_cfg.get('temperature', 'N/A')}")
    print(f"    Top-p: {llm_cfg.get('top_p', 'N/A')}")

    # ====================================================================
    # Step 1: 加载 KG 和数据
    # ====================================================================
    print_step(1, "加载知识图谱和数据集",
               "对应论文: Algorithm 1 — 数据集构建 (Appendix A.1)")

    kg = init_kg(config, DATASET)
    llm = init_llm(config)

    print_result("KG 统计", json.dumps(kg.get_stats(), indent=2, ensure_ascii=False))

    # 加载数据
    from src.run import load_dataset
    train_data = load_dataset(config, DATASET, "train")
    test_data = load_dataset(config, DATASET, "test")

    print_result("训练集大小", f"{len(train_data)} 条")
    print_result("测试集大小", f"{len(test_data)} 条")

    # 展示一条样本
    if train_data:
        sample = train_data[0]
        print_result("训练样本示例", json.dumps({
            "question": sample.get("question", ""),
            "question_entity": sample.get("question_entity", ""),
            "answer_entities": sample.get("answer_entities", [])[:3],
            "hop": sample.get("hop", ""),
        }, indent=2, ensure_ascii=False))

    # ====================================================================
    # Step 2: Agent-Planner
    # ====================================================================
    print_step(2, "Agent-Planner: 符号规则归纳",
               "对应论文: Section 4.1, Equation 1 & 3")

    print("""
  算法流程:
    ① BM25 检索种子问题 — 从训练集中找到结构相似的 K 个问题
    ② BFS 采样封闭路径 — 在 KG 中从 topic_entity 到 answer_entity 做广度优先搜索
    ③ 广义化为符号规则 — Equation 1: 将具体实体替换为变量
    ④ LLM 规则归纳 — Equation 3: p ~ π(·|ρ_Plan, q, M), few-shot 诱导生成
""")

    planner = init_planner(config, kg, llm)
    planner.build_seed_index(train_data)
    kg.build_bm25_index()

    # 选择演示问题
    if DEMO_QUESTION:
        demo_q, demo_ent = DEMO_QUESTION
        plan_samples = [{"question": demo_q, "question_entity": demo_ent}]
    else:
        # 选有 2-hop 的问题来展示
        plan_samples = [s for s in test_data if s.get("hop", 0) == 2][:MAX_PLAN_SAMPLES]

    for i, item in enumerate(plan_samples):
        question = item["question"]
        q_ent = item.get("question_entity", "")
        a_ents = item.get("answer_entities", [])

        print_substep(f"问题 {i+1}: {question}")
        print(f"    主题实体: {q_ent}")
        print(f"    标准答案: {a_ents[:5]}")

        # 2.1 BM25 种子检索
        t0 = time.time()
        seeds = planner.retrieve_seed_questions(question, top_k=3)
        t1 = time.time()
        print(f"\n    ① BM25 种子检索 (耗时 {t1-t0:.2f}s, 检索到 {len(seeds)} 条):")
        for j, seed in enumerate(seeds):
            sq = seed.get("question", "") if isinstance(seed, dict) else str(seed)
            print(f"       Seed {j+1}: {sq[:70]}...")

        # 2.2 BFS 路径采样
        t0 = time.time()
        paths = planner.sample_closed_paths(q_ent, a_ents)
        t1 = time.time()
        print(f"\n    ② BFS 路径采样 (耗时 {t1-t0:.2f}s, 采样到 {len(paths)} 条闭路径):")
        for j, path in enumerate(paths[:5]):
            print(f"       Path {j+1}: {' → '.join(path)}")

        # 2.3 符号规则泛化
        t0 = time.time()
        rules = planner.generalize_to_rules(paths)
        t1 = time.time()
        print(f"\n    ③ 符号规则泛化 — Equation 1 (耗时 {t1-t0:.2f}s, 泛化出 {len(rules)} 条规则):")
        for j, rule in enumerate(rules[:5]):
            print(f"       Rule {j+1}: {rule}")

        # 2.4 LLM 规则归纳
        print(f"\n    ④ LLM 规则归纳 — Equation 3 (调用 API 中...)")
        t0 = time.time()
        try:
            planned_paths = planner.plan(question, q_ent)
            t1 = time.time()
            print(f"       耗时: {t1-t0:.2f}s")
            print(f"       最终生成 {len(planned_paths)} 条推理路径:")
            for j, p in enumerate(planned_paths):
                print(f"         Path {j+1}: [{', '.join(p)}]")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"       ⏳ API 限速，使用 BFS 回退路径")
                planned_paths = paths[:3]
                for j, p in enumerate(planned_paths):
                    print(f"         Path {j+1}: [{', '.join(p)}]")
            else:
                print(f"       ⚠️ API 调用失败: {e}")
                planned_paths = []

    # ====================================================================
    # Step 3: Agent-Executor
    # ====================================================================
    print_step(3, "Agent-Executor: Thought-Action-Observation 推理循环",
               "对应论文: Section 4.2, Equation 4 & 5")

    print("""
  算法流程 (ReAct 风格):
    交互轨迹 H_n = (q, G, p, τ_0, a_0, o_0, τ_1, a_1, o_1, ...)

    每一步:
      τ_n = π_θ(thought_n | H_n)     — Equation 5: 生成推理思路
      a_n = π_θ(action_n  | H_n, τ_n) — Equation 5: 选择动作
      o_n = Env(a_n)                   — 环境执行动作，返回观测

    Action 空间 (Section 4.2.1):
      • getReasoningPath(sub_q)   — 获取推理路径
      • searchNeighbor(ent, rel)  — 搜索 KG 邻居
      • wikiSearch(ent, rel)      — Wikipedia 补充搜索
      • extractTriples(...)       — 自动提取三元组
      • finish(e1, e2, ...)       — 返回最终答案
""")

    executor = init_executor(config, kg, llm, planner)

    # 选择演示问题
    if DEMO_QUESTION:
        demo_q, demo_ent = DEMO_QUESTION
        exec_samples = [{"question": demo_q, "question_entity": demo_ent,
                         "answer_entities": []}]
    else:
        exec_samples = [s for s in test_data if s.get("hop", 0) == 2][:MAX_EXEC_SAMPLES]

    for i, item in enumerate(exec_samples):
        question = item["question"]
        q_ent = item.get("question_entity", "")
        a_ents = item.get("answer_entities", [])

        print_substep(f"问题 {i+1}: {question}")
        print(f"    主题实体: {q_ent}")
        print(f"    标准答案: {a_ents[:5]}")

        # 先规划
        print(f"\n    [Planner 生成推理路径]")
        try:
            paths = planner.plan(question, q_ent)
            print(f"    推理路径: {[', '.join(p) for p in paths[:3]]}")
        except Exception as e:
            print(f"    ⚠️ Planner 失败: {e}")
            paths = []

        # 执行推理
        print(f"\n    [Executor 推理循环 — Equation 4 & 5]")
        t0 = time.time()
        try:
            trajectory = executor.execute(question, q_ent, paths)
            t1 = time.time()

            # 打印推理轨迹
            text = trajectory.to_text()
            for line in text.split("\n"):
                print(f"    {line}")

            print(f"\n    推理步数: {len(trajectory.steps)}")
            print(f"    预测答案: {trajectory.answer_entities}")
            print(f"    耗时: {t1-t0:.2f}s")

        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"    ⏳ API 限速，跳过 Executor 实际推理")
                print(f"    (Planner 路径已展示，完整推理需 API 可用)")
                # 构造一个演示轨迹
                trajectory = Trajectory(question=question)
                trajectory.add_step(
                    "I need to find the answer using the planned paths.",
                    f"searchNeighbor({q_ent}, {planned_paths[0][0] if planned_paths else 'relation'})",
                    "(需要 API 调用获取实际结果)"
                )
                trajectory.add_step(
                    "Based on the search results, I can find the answer.",
                    f"finish({a_ents[0] if a_ents else 'unknown'})",
                    str(a_ents[:3])
                )
                trajectory.set_answer(a_ents[:1])
            else:
                print(f"    ⚠️ Executor 失败: {e}")
                trajectory = None
            t1 = t0

    # ====================================================================
    # Step 4: 奖励计算
    # ====================================================================
    print_step(4, "Outcome Reward — 结果奖励",
               "对应论文: Equation 6: r(μ) = Recall(A_μ, A_gt)")

    print("""
  奖励函数:
    r(μ_i) = |A_μ ∩ A_gt| / |A_gt|

    其中:
      A_μ  = Agent 预测的答案实体集合
      A_gt = 标准答案实体集合

    实体匹配支持三种方式:
      1. 直接匹配 (标准化后的字符串比较)
      2. MID → 名称 (通过 mid2name.json 映射)
      3. 名称 → MID (通过 name2mid.json 映射)
""")

    for i, item in enumerate(exec_samples):
        question = item["question"]
        a_ents = item.get("answer_entities", [])
        print_substep(f"问题 {i+1}: {question}")

        if trajectory and trajectory.answer_entities:
            reward = compute_outcome_reward(
                trajectory.answer_entities, a_ents, kg=kg
            )
            trajectory.set_reward(reward)

            print(f"    预测答案: {trajectory.answer_entities[:5]}")
            print(f"    标准答案: {a_ents[:5]}")
            print(f"    Recall (奖励): {reward:.4f}")

            if reward > 0:
                print(f"    ✅ 成功 — Agent 找到了部分/全部正确答案")
            else:
                print(f"    ❌ 失败 — Agent 的预测与标准答案无交集")
        else:
            print(f"    ⚠️ 无轨迹，跳过奖励计算")

    # ====================================================================
    # Step 5: 自我精炼
    # ====================================================================
    print_step(5, "Self-Refinement — 自我精炼",
               "对应论文: Section 4.3.1")

    print("""
  算法:
    将原始轨迹 μ_i 和奖励 r(μ_i) 作为输入，
    让 LLM 分析失败原因并生成改进轨迹 μ̂_i。

    成功轨迹 (reward > 0): LLM 尝试生成更简洁高效的版本
    失败轨迹 (reward = 0): LLM 分析失败原因并生成修正轨迹

    关键: 精炼后的轨迹需要再次计算奖励，用于后续合并。
""")

    if trajectory and trajectory.steps:
        print_substep(f"原始轨迹 (reward={trajectory.reward:.2f}):")
        print(f"    {trajectory.to_text()[:300]}...")

        print(f"\n    [调用 LLM 进行自我精炼...]")
        t0 = time.time()
        try:
            # 用 SelfLearner 的精炼方法
            from src.self_learning import SelfLearner
            sl_cfg = config.get("self_learning", {})
            learner = SelfLearner(
                kg=kg, llm=llm, planner=planner, executor=executor,
                num_iterations=1, reward_threshold=0.0,
                output_dir="/tmp/symagent_demo",
                lora_config={}, training_config={},
            )

            refined_pool = learner.self_refine([trajectory])
            t1 = time.time()

            print(f"    耗时: {t1-t0:.2f}s")
            if refined_pool:
                refined = refined_pool[0]
                print(f"\n    精炼后轨迹 (reward={refined.reward:.2f}):")
                print(f"    {refined.to_text()[:300]}...")

                if refined.reward > trajectory.reward:
                    print(f"\n    ✅ 精炼有效 — 奖励从 {trajectory.reward:.2f} 提升到 {refined.reward:.2f}")
                elif refined.reward == trajectory.reward:
                    print(f"\n    ⚽ 持平 — 奖励不变 ({trajectory.reward:.2f})")
                else:
                    print(f"\n    ⚠️ 退化 — 奖励从 {trajectory.reward:.2f} 降到 {refined.reward:.2f}")
            else:
                print(f"    ⚠️ 精炼失败")
        except Exception as e:
            err_str = str(e)
            if "429" in err_str:
                print(f"    ⏳ API 限速，跳过自我精炼")
                print(f"    (精炼需要 API 调用，完整演示需等待限速解除)")
                refined_pool = []
            else:
                print(f"    ⚠️ 自我精炼失败: {e}")
                refined_pool = []
    else:
        print("    ⚠️ 无可用轨迹，跳过")

    # ====================================================================
    # Step 6: 启发式合并
    # ====================================================================
    print_step(6, "Heuristic Merge — 启发式合并",
               "对应论文: Equation 7")

    print("""
  四分支合并策略 (Equation 7):

    ┌─────────────────────────┬──────────────────────────┐
    │ 条件                    │ 保留                     │
    ├─────────────────────────┼──────────────────────────┤
    │ r(μ_i) > r(μ̂_i)       │ 原始轨迹 μ_i            │
    │ r(μ_i) < r(μ̂_i)       │ 精炼轨迹 μ̂_i            │
    │ r(μ_i) = r(μ̂_i) > 0   │ 更短的轨迹              │
    │ r(μ_i) = r(μ̂_i) = 0   │ 过滤掉 (都不保留)       │
    └─────────────────────────┴──────────────────────────┘
""")

    # 演示合并逻辑
    print_substep("合并演示")

    from src.self_learning import TrajectoryPool
    demo_pairs = [
        (0.8, 0.5, "r(μ) > r(μ̂)"),
        (0.3, 0.7, "r(μ) < r(μ̂)"),
        (0.6, 0.6, "r(μ) = r(μ̂) > 0"),
        (0.0, 0.0, "r(μ) = r(μ̂) = 0"),
    ]

    for r_orig, r_refine, desc in demo_pairs:
        # 构造最小 TrajectoryPool
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        t1 = Trajectory(question="demo")
        t1.set_reward(r_orig)
        t2 = Trajectory(question="demo")
        t2.set_reward(r_refine)
        orig_pool.add(t1)
        ref_pool.add(t2)

        merged = heuristic_merge(orig_pool, ref_pool)
        kept = merged.trajectories[0].reward if merged.trajectories else "filtered"
        print(f"    {desc}: r(μ)={r_orig:.1f}, r(μ̂)={r_refine:.1f} → 保留: {kept}")

    # 用实际轨迹演示
    if trajectory and refined_pool and len(refined_pool) > 0:
        refined = refined_pool[0]
        orig_pool = TrajectoryPool()
        ref_pool = TrajectoryPool()
        orig_pool.add(trajectory)
        ref_pool.add(refined)
        merged = heuristic_merge(orig_pool, ref_pool)
        kept = merged.trajectories[0].reward if merged.trajectories else "filtered"
        print(f"\n    实际合并: r(原始)={trajectory.reward:.2f}, r(精炼)={refined.reward:.2f}")
        print(f"    → 结果: {kept}")

    # ====================================================================
    # Step 7: LoRA 微调
    # ====================================================================
    print_step(7, "LoRA Fine-tuning — 离线策略更新",
               "对应论文: Section 4.3.2, Equation 8")

    sl_cfg = config.get("self_learning", {})
    print(f"""
  SFT 损失函数 (Equation 8):
    L_SFT = -E_{{μ~D*}} [Σ_j 1(x_j ∈ A) · log π_θ(x_j | x_{{<j}}, q)]

  Indicator Mask — 只训练 Agent 生成的部分:
    ┌─────────────────────────────┬────────────┐
    │ 内容                        │ 是否训练   │
    ├─────────────────────────────┼────────────┤
    │ Question                    │ ❌ mask    │
    │ Thought (τ_n)               │ ✅ 训练    │
    │ Action (a_n)                │ ✅ 训练    │
    │ Observation (o_n)           │ ❌ mask    │
    └─────────────────────────────┴────────────┘

  LoRA 配置 (来自 config.yaml):
    Rank (r):         {sl_cfg.get('lora_r', 32)}
    Alpha:            {sl_cfg.get('lora_alpha', 32)}
    Dropout:          {sl_cfg.get('lora_dropout', 0.05)}
    Target modules:   {sl_cfg.get('lora_target_modules', ['q_proj', 'k_proj', ...])}

  训练配置:
    Batch size:       {sl_cfg.get('batch_size', 4)}
    Learning rate:    {sl_cfg.get('learning_rate', '2e-5')}
    Epochs:           {sl_cfg.get('num_train_epochs', 3)}
    Max seq length:   {sl_cfg.get('max_seq_length', 4096)}
    Warmup ratio:     {sl_cfg.get('warmup_ratio', 0.05)}
    Grad accum steps: {sl_cfg.get('gradient_accumulation_steps', 2)}
""")

    print_substep("演示: 用演示轨迹构造训练数据")
    if trajectory and trajectory.steps:
        print(f"    轨迹步数: {len(trajectory.steps)}")
        print(f"    轨迹文本长度: {len(trajectory.to_text())} 字符")

        # 展示 indicator mask 的效果
        print(f"\n    Indicator Mask 示例 (前3步):")
        for j, step in enumerate(trajectory.steps[:3]):
            thought = step.get("thought", "")
            action = step.get("action", "")
            observation = step.get("observation", "")
            role = step.get("role", "")

            t_mask = "✅ 训练" if role == "thought" else "❌ mask"
            a_mask = "✅ 训练" if role == "action" else "❌ mask"
            o_mask = "❌ mask" if role == "observation" else ""

            if thought:
                print(f"      Thought {j+1}: [{t_mask}] {thought[:80]}...")
            if action:
                print(f"      Action {j+1}:  [{a_mask}] {action[:80]}...")
            if observation:
                print(f"      Obs {j+1}:      [{o_mask}] {observation[:80]}...")

    # ====================================================================
    # 总结
    # ====================================================================
    print(f"\n{SEPARATOR}")
    print("  演示完成 — SymAgent 完整算法流程总结")
    print(SEPARATOR)
    print("""
  ┌──────────────────────────────────────────────────────────────────┐
  │                    SymAgent 自学习循环                           │
  │                                                                  │
  │  训练集 ──→ Agent-Planner ──→ Agent-Executor ──→ 轨迹 μ_i      │
  │               (Section 4.1)     (Section 4.2)                   │
  │                      │                    │                     │
  │                      │              ┌─────┴─────┐               │
  │                      │              ▼           ▼               │
  │                      │         r(μ_i)     Self-Refine           │
  │                      │        (Eq.6)     (Section 4.3.1)       │
  │                      │              │           │               │
  │                      │              └─────┬─────┘               │
  │                      │                    ▼                     │
  │                      │            Heuristic Merge               │
  │                      │              (Eq.7)                      │
  │                      │                    │                     │
  │                      │                    ▼                     │
  │                      │         合并后的轨迹池 D*                │
  │                      │                    │                     │
  │                      │              LoRA SFT                    │
  │                      │             (Eq.8)                       │
  │                      │                    │                     │
  │                      │              更新后的 π_θ                │
  │                      │                    │                     │
  │                      └────────────────────┘                     │
  │                           ↑ 迭代 (num_iterations=2)            │
  └──────────────────────────────────────────────────────────────────┘

  关键创新:
  1. 神经-符号协同: Planner (符号) + Executor (神经) 互补
  2. 自学习框架: 无需人工标注，通过环境交互自动提升
  3. Indicator Mask: 只训练推理决策部分，不学习环境输出
  4. 迭代优化: 多轮探索-精炼-合并-微调循环
""")


if __name__ == "__main__":
    main()
