# SymAgent

**Neural-Symbolic Self-Learning Agent for Complex Reasoning over Knowledge Graphs**

复现论文：[SymAgent (SIGIR 2025)](https://arxiv.org/abs/2502.03283v2)

---

## 目录

- [项目结构](#项目结构)
- [环境配置](#环境配置)
- [配置文件详解](#配置文件详解)
- [运行模式](#运行模式)
  - [1. 规划模式 (plan)](#1-规划模式-plan)
  - [2. 推理模式 (execute)](#2-推理模式-execute)
  - [3. 评估模式 (evaluate)](#3-评估模式-evaluate)
  - [4. 训练模式 (train)](#4-训练模式-train)
  - [5. 完整流水线 (full_pipeline)](#5-完整流水线-full_pipeline)
- [自学习训练流程详解](#自学习训练流程详解)
- [LoRA 微调详解](#lora-微调详解)
- [数据说明](#数据说明)
  - [数据集概览](#数据集概览)
  - [数据字段说明](#数据字段说明)
  - [数据在系统中的使用方式](#数据在系统中的使用方式)
  - [数据预处理流程](#数据预处理流程)
  - [测试方法](#测试方法)
  - [评估指标说明](#评估指标说明)
- [核心算法与代码映射](#核心算法与代码映射)
- [常见问题](#常见问题)
- [依赖](#依赖)

---

## 项目结构

```
SymAgent/
├── src/                    # 核心源码
│   ├── run.py              # 启动入口
│   ├── kg_environment.py   # KG 动态环境
│   ├── planner.py          # Agent-Planner (符号规则诱导, Eq.1/3)
│   ├── executor.py         # Agent-Executor (Thought-Action-Observation, Eq.4-6)
│   ├── self_learning.py    # 自学习框架 (Eq.7-8, LoRA 微调)
│   ├── llm_client.py       # LLM 客户端 (OpenAI 兼容 API)
│   └── evaluate.py         # 评估模块 (Hits@k, F1, Accuracy)
├── scripts/                # 数据预处理脚本
│   ├── download_datasets.py        # 下载原始数据集
│   ├── preprocess_dataset.py       # 数据预处理
│   ├── filter_multihop.py          # 过滤多跳问题
│   ├── prepare_kg.py               # 准备 KG 数据
│   └── build_kg_from_datasets.py   # 从数据集构建 KG
├── configs/config.yaml     # 主配置文件
├── data/                   # 数据集
│   ├── freebase/           # Freebase KG (三元组 + 实体/关系映射)
│   ├── processed/          # 预处理后的 QA 样本 (训练/测试)
│   ├── webqsp/             # WebQSP 原始数据
│   ├── cwq/                # CWQ 原始数据
│   └── grailqa/            # GrailQA 原始数据
├── checkpoints/            # 训练 checkpoint 输出目录
├── logs/                   # 运行日志输出目录
└── paper/                  # 论文 PDF + 全文
```

---

## 环境配置

### 系统要求

- Python 3.10+
- PyTorch 2.0+ (LoRA 微调需要 CUDA)
- 7B 模型 LoRA 微调显存：bf16 约需 24GB+；**单卡 24GB 开 `load_in_4bit: true`(QLoRA) 即可**，约需 6-8GB
- 4-bit 量化需安装 `bitsandbytes`（`pip install bitsandbytes`）
- 评估模式：用在线 API 时不需要 GPU；用 `--lora_path` 评估本地模型则需要 GPU

### 快速开始

```bash
# 1. 激活环境
conda activate ray312

# 2. 安装依赖
pip install -r requirements.txt

# 3. 配置 API Key（编辑 configs/config.yaml）
#    或设置环境变量：export ZHIPU_API_KEY="your-key"

# 4. 运行算法流程演示
python demo.py

# 5. 运行评估
cd /home/sone/symagent
python run.py evaluate --dataset webqsp
```

### 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- `torch>=2.0.0` — 深度学习框架
- `transformers>=4.36.0` — HuggingFace 模型加载
- `peft>=0.7.0` — LoRA 微调
- `bitsandbytes>=0.41.0` — 4-bit(QLoRA)量化加载（`load_in_4bit` 必需）
- `openai>=1.0.0` — OpenAI 兼容 API 客户端
- `rank_bm25>=0.2.2` — BM25 检索
- `networkx>=3.0` — 图数据结构 (KG)
- `scikit-learn>=1.3.0` — 评估指标计算

### API 配置

编辑 `configs/config.yaml`，填入 LLM API 信息：

```yaml
llm:
  model_name: glm-4.7-flash          # API 推理模型
  api_base: https://open.bigmodel.cn/api/paas/v4
  api_key: YOUR_API_KEY
```

API Key 也可通过环境变量设置（优先级低于配置文件）：
```bash
export ZHIPU_API_KEY="your_key_here"
# 或
export OPENAI_API_KEY="your_key_here"
```

### LoRA 微调本地模型

训练模式的 LoRA 微调需要**本地模型权重**。在 config 的 `self_learning` 段设置：

```yaml
self_learning:
  base_model: /path/to/Qwen2.5-7B-Instruct   # LoRA 微调用的本地模型（HF 仓库名或本地路径）
  exploration_mode: online                   # 在线探索用哪个策略：local 自探索 / online 蒸馏
  load_in_4bit: true                         # 4-bit(QLoRA)加载，单卡 24GB 必开
```

- `base_model` 是微调的 backbone，区别于 `llm` 段的在线 API 推理端点。未设置则跳过 LoRA 微调，只采集/合并轨迹。
- `exploration_mode`：
  - `local` —— 用本地 `base_model` 自探索自训练（论文设定 π_θ 自我提升），需 `base_model`。单卡需同时容纳 7B 推理与微调，显存吃紧。
  - `online` —— 用在线 API 模型探索，微调时才加载一次本地 `base_model`（教师蒸馏变体）。**单卡 24GB 推荐此模式**，绕开探索阶段的显存峰值。
- `load_in_4bit`：微调时是否用 4-bit 量化加载。单卡 24GB 必须开，否则 bf16 的 7B 权重（~15GB）加长上下文 logits 峰值会 OOM。开启后自动启用 gradient checkpointing 进一步省显存（更慢）。

---

## 配置文件详解

所有参数均在 `configs/config.yaml` 中配置：

### LLM 配置 (`llm`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `model_name` | str | `glm-4.7-flash` | API 推理模型名称。用于 Planner 规则生成和 Executor 推理。支持任何 OpenAI 兼容 API 模型 |
| `api_base` | str | `https://open.bigmodel.cn/api/paas/v4` | API 端点地址。可改为 vLLM、Ollama 等本地部署的兼容端点 |
| `api_key` | str | `''` | API 密钥。也可通过 `ZHIPU_API_KEY` 或 `OPENAI_API_KEY` 环境变量提供 |
| `temperature` | float | `0.1` | 默认生成温度。越低越确定性，越高越随机。Planner 规则生成时自动使用 0.3，Executor 推理使用此值 |
| `top_p` | float | `0.9` | 核采样概率阈值。控制输出多样性 |
| `top_k` | int | `600` | Top-K 采样参数 |
| `max_new_tokens` | int | `512` | 单次生成的最大 token 数。影响 LLM 每步推理的输出长度上限 |
| `min_request_interval` | float | `2.0` | 两次 API 请求之间的最小间隔（秒）。用于避免触发 429 限速 |

### KG 配置 (`kg`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `data_dir` | str | `data` | 数据集根目录 |
| `backend` | str | `networkx` | KG 存储后端。目前仅支持 `networkx` |
| `freebase_to_wikidata_mapping` | str | — | Freebase 到 Wikidata 的 ID 映射文件路径 |
| `wikidata.endpoint` | str | — | Wikidata SPARQL 查询端点 |
| `wikidata.api_url` | str | — | Wikidata API URL |
| `wikidata.max_retries` | int | `3` | Wikidata 查询最大重试次数 |
| `wikidata.timeout` | int | `30` | Wikidata 查询超时时间（秒） |

#### 数据集配置 (`kg.datasets.<dataset_name>`)

每个数据集（`webqsp`/`cwq`/`metaqa`）支持以下参数：

| 参数 | 类型 | 说明 |
|------|------|------|
| `train_file` | str | 训练集 JSON 文件路径。每条数据包含 `question`、`question_entity`、`answer_entities` |
| `valid_file` | str | 验证集路径。用于自学习训练的 early stopping。为空则跳过验证 |
| `test_file` | str | 测试集路径。用于评估模式 |
| `triple_file` | str | KG 三元组文件。每行格式：`head\trelation\ttail` |
| `entity2id` | str | 实体到 ID 的映射文件。每行格式：`entity_name\tid` |
| `relation2id` | str | 关系到 ID 的映射文件。每行格式：`relation_name\tid` |
| `mid2name` | str | Freebase MID 到实体名称的 JSON 映射。用于奖励计算时的实体匹配 |
| `name2mid` | str | 实体名称到 Freebase MID 的 JSON 映射。`mid2name` 的反向映射 |

### Planner 配置 (`planner`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_seed_questions` | int | `3` | BM25 检索的种子问题数量。Planner 用 BM25 从训练集中检索结构相似的种子问题，用于构建 few-shot 示例。值越大，示例越多，但 API 调用也越多 |
| `max_bfs_depth` | int | `4` | BFS 采样推理路径的最大深度。控制从查询实体到答案实体的最大关系跳数 |
| `max_paths_per_seed` | int | `5` | 每个种子问题最多采样的闭路径数量。影响规则泛化的候选数量 |
| `max_rules` | int | `10` | Planner 最终返回的最大符号规则数量。控制传给 Executor 的推理路径上限 |
| `max_rule_length` | int | `4` | 单条规则的最大关系长度（即规则体中关系数量上限） |
| `bm25_top_k` | int | `5` | BM25 实体链接时返回的 Top-K 实体数量 |

### Executor 配置 (`executor`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `max_infer_steps` | int | `10` | Thought-Action-Observation 循环的最大步数。超过此步数未调用 `finish` 则强制终止 |
| `beam_width` | int | `1` | 推理束宽度。当前实现为贪心搜索（width=1） |
| `max_triples_per_step` | int | `50` | 每步最大三元组数量。限制 KG 搜索返回的邻居数 |
| `wiki_search_enabled` | bool | `true` | 是否启用 Wikipedia 搜索。当 KG 信息不足时 Executor 可调用 `wikiSearch` 补充信息。**网络无法访问 Wikipedia（如墙内环境）时设为 `false`**，否则每次 `wikiSearch` 会卡在超时/503 上浪费数十秒；关闭后 agent 改用 KG 或自身知识作答 |
| `wikidata_search_enabled` | bool | `true` | （**当前无效**）`searchWikidata` 动作尚未实现，此开关代码中未接入，可忽略 |

### 自学习训练配置 (`self_learning`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `base_model` | str | 无 | LoRA 微调的 backbone（HF 仓库名或本地权重路径）。未设置则跳过微调，只采集/合并轨迹 |
| `exploration_mode` | str | `local` | 在线探索阶段用哪个策略生成轨迹。`local`：用本地 `base_model` 自探索自训练（论文设定，需 `base_model`）；`online`：用 API 模型探索并蒸馏进本地模型（单卡 24GB 推荐） |
| `load_in_4bit` | bool | `false` | 微调时是否用 4-bit(QLoRA)量化加载 `base_model`。单卡 24GB 须开，否则 bf16 的 7B 会 OOM |
| `gradient_checkpointing` | bool | 跟随 `load_in_4bit` | 是否启用梯度检查点省显存（更慢）。默认与 `load_in_4bit` 一致 |
| `num_iterations` | int | `2` | 自学习迭代轮数。每轮执行：在线探索 → 自我精炼 → 启发式合并 → LoRA 微调。论文默认为 2 轮。注意 `online` 模式下探索者固定为 API，多轮收益很低，可设 1 |
| `exploration_budget` | int/null | `null` | 在线探索的预算（样本数）。`null` 表示使用全部训练数据 |
| `reward_threshold` | float | `0.0` | 轨迹奖励阈值。低于此值的轨迹在合并后被过滤。`0.0` 表示保留所有轨迹 |
| `batch_size` | int | `4` | LoRA 微调的每个 GPU 的 batch size |
| `learning_rate` | float | `2e-5` | LoRA 微调学习率。常用范围 1e-5 ~ 5e-5 |
| `num_train_epochs` | int | `3` | LoRA 微调的训练轮数 |
| `max_seq_length` | int | `4096` | 训练数据的最大序列长度。超过此长度的轨迹会被截断 |
| `warmup_ratio` | float | `0.05` | 学习率预热的比例。总训练步数的 5% 用于 warmup |
| `gradient_accumulation_steps` | int | `2` | 梯度累积步数。有效 batch size = `batch_size × gradient_accumulation_steps` |
| `lr_scheduler_type` | str | `cosine` | 学习率调度器。支持 `cosine`（余弦退火）、`linear`（线性衰减）、`constant` 等 |
| `eval_sample_size` | int | `50` | 自学习训练中每次验证的样本数量。用于 early stopping 判断 |

### LoRA 配置 (`self_learning.lora_*`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `lora_r` | int | `32` | LoRA 秩（rank）。控制低秩矩阵的维度。越大表达能力越强，但参数量也越多。常用值：8, 16, 32, 64 |
| `lora_alpha` | int | `32` | LoRA 缩放系数。实际缩放因子 = `alpha / r`。通常设为与 `r` 相同或 2 倍 |
| `lora_dropout` | float | `0.05` | LoRA 层的 dropout 率。用于防止过拟合 |
| `lora_target_modules` | list | 见下方 | 要应用 LoRA 的模型层名称。不同模型架构需要不同的层名 |

**不同模型的 `lora_target_modules` 配置：**

| 模型架构 | 代表模型 | target_modules |
|----------|----------|----------------|
| LLaMA/Qwen | Qwen2.5-7B, LLaMA-3-8B | `["q_proj", "k_proj", "v_proj", "o_proj", "down_proj", "up_proj", "gate_proj"]` |
| GPT-2 | GPT-2 | `["c_attn", "c_proj"]` |
| ChatGLM | GLM-4-9B | `["query_key_value", "dense"]` |

### 评估配置 (`evaluation`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `metrics` | list | `[hits@1, hits@3, hits@10, f1]` | 评估指标列表。可选：`hits@1`、`hits@3`、`hits@10`、`f1`、`accuracy` |
| `batch_size` | int | `1` | 评估时的 batch size（当前实现为逐条推理，此参数预留） |
| `max_eval_samples` | int/null | `null` | 评估时最大样本数。`null` 表示评估全部测试集 |

### 日志配置 (`logging`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `log_dir` | str | `logs` | 日志输出目录 |
| `checkpoint_dir` | str | `checkpoints` | 训练 checkpoint 保存目录 |
| `log_level` | str | `INFO` | 日志级别。可选：`DEBUG`、`INFO`、`WARNING`、`ERROR` |
| `save_steps` | int | `500` | 训练时每 N 步保存一次 checkpoint |

---

## 运行模式

统一入口：`python run.py <模式> [选项]`（等价于 `python -m src.run <模式> [选项]`）

### 全局选项

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `configs/config.yaml` | 配置文件路径 |
| `--dataset` | `webqsp` | 数据集名称。可选：`webqsp`、`cwq`、`metaqa` |

---

### 1. 规划模式 (plan)

仅运行 Agent-Planner，生成符号推理路径（不执行推理）。

**用途：** 验证 Planner 的规则生成能力，查看给定问题的推理路径。

```bash
python run.py plan --dataset webqsp
```

**额外选项：**

| 选项 | 说明 |
|------|------|
| `--max_samples N` | 只处理前 N 条测试样本 |
| `--output PATH` | 输出文件路径，默认 `logs/planned_paths.json` |

**输出示例：**
```json
[
  {
    "question": "what language do they speak in colombia south america",
    "planned_paths": [["location.country.languages_spoken", "common.topic.notable_types"]]
  }
]
```

**执行流程：**
1. 加载 KG 三元组 + 实体/关系映射
2. 用训练集构建 BM25 种子问题索引
3. 对每条测试问题：
   - BM25 检索相似种子问题
   - BFS 采样闭路径 → 泛化为符号规则
   - LLM few-shot 诱导生成规则（Eq.3）
4. 输出所有推理路径到 JSON 文件

---

### 2. 推理模式 (execute)

运行 Agent-Executor 的 Thought-Action-Observation 推理循环。

**用途：** 对单条问题或整个测试集进行交互式推理，输出推理轨迹和答案。

#### 单条问题推理

```bash
python run.py execute --question "who directed inception" --entity m.03_d0
```

| 选项 | 说明 |
|------|------|
| `--question` | 要推理的问题 |
| `--entity` | 问题中的主题实体（Freebase MID） |

**输出示例：**
```
=== Trajectory ===
Question: who directed inception
Thought 1: I need to find the director of inception.
Action 1: getReasoningPath(inception, directed by)
Observation 1: Surrounding relational reasoning paths are:
  [film.film.directed_by]
Thought 2: Based on the reasoning path, I'll search for inception's director.
Action 2: searchNeighbor(inception, film.film.directed_by)
Observation 2: m.0bxtg
Thought 3: I found the director entity m.0bxtg.
Action 3: finish(m.0bxtg)

Answer: ['m.0bxtg']
```

#### 测试集批量推理

```bash
python run.py execute --dataset webqsp --max_samples 20
```

| 选项 | 说明 |
|------|------|
| `--max_samples N` | 只推理前 N 条 |
| `--output PATH` | 输出路径，默认 `logs/execution_results.json` |

**执行流程：**
1. Planner 生成符号规则（推理路径）
2. Executor 执行 Thought-Action-Observation 循环：
   - `getReasoningPath` — 获取推理路径
   - `searchNeighbor` — 在 KG 中搜索邻居
   - `wikiSearch` — Wikipedia 补充搜索（KG 信息不足时）
   - `finish` — 返回最终答案
3. 计算奖励（Recall, Eq.6）

---

### 3. 评估模式 (evaluate)

在测试集上评估完整性能（Planner + Executor）。

**用途：** 计算标准 KGQA 指标，与论文结果对比。

```bash
# WebQSP MultiHop 评估
python run.py evaluate --dataset webqsp

# CWQ MultiHop 评估
python run.py evaluate --dataset cwq

# 限制样本数快速测试
python run.py evaluate --dataset webqsp --max_samples 50

# 用训练后的本地模型评估（base_model + LoRA adapter）
python run.py evaluate --dataset webqsp --lora_path checkpoints/webqsp/lora_iteration_1/lora_adapter

# 自定义输出路径
python run.py evaluate --dataset webqsp --output my_results.json
```

**额外选项：**

| 选项 | 说明 |
|------|------|
| `--max_samples N` | 只评估前 N 条测试样本。用于快速验证 |
| `--lora_path PATH` | 指定一个训练好的 LoRA adapter 目录。设置后用本地 `base_model` + adapter 评估（反映训练成果），不设置则用在线 API LLM（可作基线对比）。需配置 `self_learning.base_model` |
| `--output PATH` | 结果文件路径，默认 `logs/eval_<dataset>_results.json` |

**输出示例：**
```
=== Evaluation Results ===
Dataset: webqsp
  accuracy: 0.7256
  f1: 0.7145
  hits@1: 0.7080
  hits@3: 0.7434
  hits@10: 0.7611
```

**执行流程：**
1. 加载 KG + 训练集（构建种子索引）+ 测试集
2. 对每条测试问题：
   - Planner 生成推理路径
   - Executor 执行完整推理循环
   - 收集预测答案
3. 计算所有配置的指标并取平均
4. 每 10 条打印一次中间结果

---

### 4. 训练模式 (train)

运行完整的自学习训练循环（在线探索 + LoRA 微调）。

**用途：** 让 Agent 通过与环境交互自动提升推理能力。

```bash
# WebQSP 自学习训练
python run.py train --dataset webqsp

# CWQ 自学习训练
python run.py train --dataset cwq

# 只用前 5 条样本快速验证全流程（探索→合并→微调）
python run.py train --dataset webqsp --max_samples 5
```

**额外选项：**

| 选项 | 说明 |
|------|------|
| `--max_samples N` | 只用前 N 条训练样本探索/训练。用于快速跑通全流程 |

**前置条件：**
- 必须配置 `api_key`（`online` 探索模式下用于 LLM 推理）
- 如需 LoRA 微调，需配置 `self_learning.base_model` 指向本地模型
- 单卡 24GB 建议 `exploration_mode: online` + `load_in_4bit: true`
- 需要 GPU（微调阶段必需；`online` 模式探索仅需 API，`local` 模式探索也要 GPU）

**执行流程：**

```
迭代 0:
  ├── 在线探索 (online_explore)
  │   ├── 对每条训练问题调用 Planner + Executor
  │   ├── 生成推理轨迹 μ_i
  │   └── 计算结果奖励 r(μ_i) = Recall (Eq.6)
  ├── 自我精炼 (self_refine)
  │   ├── 将原始轨迹 + 奖励输入 LLM
  │   └── 生成改进轨迹 μ̂_i
  ├── 启发式合并 (heuristic_merge, Eq.7)
  │   └── 比较原始/精炼轨迹，保留更优者
  └── LoRA 微调 (fine_tune, Eq.8)
      ├── 加载本地模型 + 应用 LoRA
      ├── 构造 SFT 训练数据（仅 Thought/Action 参与损失）
      └── 训练并保存 checkpoint

迭代 1: (重复上述流程)
  - local 模式：用上一轮微调后的模型重新探索
  - online 模式：探索者仍为 API（不切本地模型），重新探索生成新轨迹
  ...

Early stopping: 验证集提升 < 0.001 时自动停止（需配置 valid_file）
```

**输出文件：** （`<dataset>` 为数据集名，如 `webqsp`）
```
checkpoints/<dataset>/
├── iteration_0_trajectories.json   # 第 0 轮合并后的轨迹（实际喂给微调的数据 D*）
├── iteration_1_trajectories.json   # 第 1 轮合并后的轨迹
├── final_trajectories.json         # 最后一轮轨迹副本
├── lora_iteration_0/
│   └── lora_adapter/               # 第 0 轮 LoRA adapter
│       ├── adapter_config.json
│       ├── adapter_model.safetensors
│       └── tokenizer 相关文件
└── lora_iteration_1/
    └── lora_adapter/               # 第 1 轮 LoRA adapter
```
> 注：只保存 LoRA adapter，不含 base 模型。推理时需 `base_model` + adapter 一起加载。

---

### 5. 完整流水线 (full_pipeline)

先执行自学习训练，再在测试集上评估。

```bash
python run.py full_pipeline --dataset webqsp

# 快速验证：训练和评估都只用前 5 条
python run.py full_pipeline --dataset webqsp --max_samples 5
```

**自动衔接：** Phase 2 评估会**自动使用 Phase 1 训出的最新一轮 LoRA adapter**（无需手动指定路径）。若训练未产出 adapter（如未配 `base_model`），则回退到在线 API LLM 评估。终端会打印实际使用的 adapter 路径。

**额外选项：**

| 选项 | 说明 |
|------|------|
| `--max_samples N` | 同时限制训练和评估的样本数 |
| `--by_hop` | 评估时按跳数分组统计 |

大致等价于：
```bash
python run.py train --dataset webqsp
python run.py evaluate --dataset webqsp --lora_path <最新一轮 adapter>
```

---

## 自学习训练流程详解

### 整体架构

```
训练集 → Agent 交互 → 轨迹生成 → 奖励计算 → 自我精炼 → 启发式合并 → LoRA 微调 → 迭代
```

### 第 1 阶段：在线探索 (Online Exploration, Section 4.3.1)

Agent（Planner + Executor）对训练集中的每个问题执行完整推理：

1. **Planner** 从训练集 BM25 检索相似种子问题，通过 BFS 采样闭路径并泛化为符号规则，再用 LLM few-shot 诱导生成目标问题的推理路径
2. **Executor** 沿推理路径执行 Thought-Action-Observation 循环，每步选择 `searchNeighbor` 或 `wikiSearch` 等动作
3. **奖励计算**：`r(μ_i) = Recall(A_μ, A_gt)` — 预测答案与真实答案的召回率（Eq.6）

### 第 2 阶段：自我精炼 (Self-Refinement)

对探索阶段生成的每条轨迹，将原始轨迹和奖励作为输入，让 LLM 重新生成改进轨迹：
- 成功轨迹（reward > 0）：LLM 尝试生成更简洁高效的版本
- 失败轨迹（reward = 0）：LLM 分析失败原因并生成修正轨迹

### 第 3 阶段：启发式合并 (Heuristic Merge, Eq.7)

对每对（原始轨迹 μ_i, 精炼轨迹 μ̂_i）按以下规则合并：

| 条件 | 保留 |
|------|------|
| r(μ_i) > r(μ̂_i) | 原始轨迹 μ_i |
| r(μ_i) < r(μ̂_i) | 精炼轨迹 μ̂_i |
| r(μ_i) = r(μ̂_i) > 0 | 更短的轨迹 |
| r(μ_i) = r(μ̂_i) = 0 | 过滤掉（都不保留） |

### 第 4 阶段：LoRA 微调 (Offline Policy Update, Section 4.3.2)

用合并后的轨迹池 D* 对 LLM 进行 SFT 微调。详见下方 [LoRA 微调详解](#lora-微调详解)。

---

## LoRA 微调详解

### 工作原理

LoRA（Low-Rank Adaptation）通过在模型原有权重旁添加低秩矩阵来实现高效微调：

```
原始权重 W ∈ R^{d×d}    →    W + ΔW = W + BA    其中 B ∈ R^{d×r}, A ∈ R^{r×d}
```

只有 A、B 两个低秩矩阵参与训练，原始权重 W 保持冻结。

### 训练数据构造 (Eq.8)

SFT 损失函数：
```
L_SFT = -E_{μ~D*} [Σ_j 1(x_j ∈ A) · log π_θ(x_j | x_{<j}, q)]
```

关键：**indicator mask** — 只有 Agent 生成的部分（Thought 和 Action）参与损失计算，Question 和 Observation（环境输出）被 mask 为 -100：

```
Question: what is the capital of France?     ← mask (-100, 不计算损失)
Thought 1: I need to search...               ← 参与损失计算 ✓
Action 1: searchNeighbor(m.fr, location.capital)  ← 参与损失计算 ✓
Observation 1: m.paris                        ← mask (-100, 环境输出)
```

这样模型只学习「如何推理和决策」，不学习「如何复述问题和观测」。

### 训练参数调优建议

| 场景 | 建议参数 |
|------|----------|
| 小数据集（<500 条轨迹） | `lora_r=16`, `num_train_epochs=5`, `learning_rate=5e-5` |
| 中等数据集（500-2000 条） | `lora_r=32`, `num_train_epochs=3`, `learning_rate=2e-5`（默认） |
| 大数据集（>2000 条） | `lora_r=64`, `num_train_epochs=2`, `learning_rate=1e-5` |
| 显存有限（<16GB） | `batch_size=1`, `gradient_accumulation_steps=4`, `max_seq_length=2048` |
| 防止过拟合 | `lora_dropout=0.1`, `num_train_epochs=1`, `reward_threshold=0.3` |

### 使用训练好的 LoRA Adapter

最简单的方式是直接用评估命令（自动加载 base + adapter）：

```bash
python run.py evaluate --dataset webqsp --lora_path checkpoints/webqsp/lora_iteration_1/lora_adapter
```

也可手动加载用于自定义推理：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# 加载基座模型（与训练时 self_learning.base_model 一致）
base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# 加载 LoRA adapter（路径含数据集子目录）
model = PeftModel.from_pretrained(base_model, "checkpoints/webqsp/lora_iteration_1/lora_adapter")
model.eval()

# 推理
prompt = "Question: who directed inception?"
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

---

## 数据说明

### 数据集概览

本项目使用基于 Freebase 知识图谱的多跳 KGQA（知识图谱问答）数据集。所有数据集共享同一套 Freebase KG 三元组存储。

#### 数据集规模

| 数据集 | 训练集 | 测试集 | 跳数分布 | KG | 说明 |
|--------|--------|--------|----------|-----|------|
| WebQSP (multihop) | 1,311 | 113 | 1-hop: 840, 2-hop: 471 (训练集); 全部 2-hop (测试集) | Freebase | 单跳和多跳知识图谱问答，测试集仅保留 2-hop 问题 |
| CWQ (multihop) | 2,328 | 316 | 全部 2-hop | Freebase | 复合多跳问答，含组合约束条件（conjunction/composition） |
| GrailQA | 800 | 200 | 混合 (i.i.d. / composition / constraint) | Freebase | 大规模通用知识图谱问答 |

#### KG 索引文件

所有数据集共用 `data/freebase/` 下的 KG 索引：

| 文件 | 行数 | 格式 | 说明 |
|------|------|------|------|
| `freebase_triples.txt` | 35,774 | `head\trelation\ttail` | KG 三元组，由 QA 样本的关系路径构建 |
| `entity2id.txt` | 24,207 | `entity_mid\tid` | 实体到整数 ID 的映射 |
| `relation2id.txt` | 1,168 | `relation_name\tid` | 关系到整数 ID 的映射 |
| `mid2name.json` | 13,764 | `{"m.078w2": "Samuel Taylor Coleridge", ...}` | Freebase MID 到实体名称的映射 |
| `name2mid.json` | — | `{"samuel taylor coleridge": "m.078w2", ...}` | 实体名称到 Freebase MID 的反向映射 |

#### KG 三元组的构建方式

由于本项目不使用完整 Freebase 子图，KG 三元组从 QA 样本的 `inferential_chain`（关系路径）中构建：

- **单跳问题**（1 个关系）：直接生成 `(topic_entity, relation, answer_entity)` 三元组
- **多跳问题**（N 个关系）：生成链式三元组，中间节点使用 `__INTERMEDIATE_xxx__` 占位符

```
例：关系路径 [r1, r2, r3]
三元组：
  (topic, r1, __INTERMEDIATE_0__)
  (__INTERMEDIATE_0__, r2, __INTERMEDIATE_1__)
  (__INTERMEDIATE_1__, r3, answer)
```

这使得 BFS 路径查找能够沿关系链导航，Planner 可以采样闭路径并泛化为符号规则。

---

### 数据字段说明

各数据集的字段略有差异，以下是统一说明：

#### WebQSP 数据格式

```json
{
  "question": "who is louisiana state senator",
  "question_entity": "m.04ly1",
  "question_entity_name": "Louisiana",
  "answer_entities": ["m.01_nrx", "m.019tyn", ...],
  "answer_entity_names": ["John Slidell", "Mary Landrieu", ...],
  "hop": 2,
  "qid": "WebQTest-575",
  "sparql": "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ...",
  "dataset": "webqsp",
  "inferential_chain": [
    "government.political_district.representatives",
    "government.government_position_held.office_holder"
  ]
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 自然语言问题 |
| `question_entity` | str | 问题主题实体的 Freebase MID（如 `m.04ly1`） |
| `question_entity_name` | str | 主题实体的可读名称（如 `Louisiana`） |
| `answer_entities` | list[str] | 标准答案实体 MID 列表，可能有多个答案 |
| `answer_entity_names` | list[str] | 标准答案实体的可读名称列表 |
| `hop` | int | 问题跳数（1=单跳, 2=二跳, ...），等于 `inferential_chain` 的长度 |
| `qid` | str | 问题唯一标识符 |
| `sparql` | str | 原始 SPARQL 查询语句 |
| `dataset` | str | 数据集来源标识（`webqsp`） |
| `inferential_chain` | list[str] | 推理关系链，即从主题实体到答案的完整关系路径 |

#### CWQ 数据格式

```json
{
  "question": "What religion with diety \"Uchchhishta Ganapati\" is practice in Indonesia?",
  "question_entity": {"m.03ryn": "Indonesia", "m.010x89zn": "Uchchhishta Ganapati"},
  "answer_entities": ["Hinduism"],
  "answer_entity_names": ["Hinduism"],
  "hop": 2,
  "compositionality_type": "conjunction",
  "sparql": "PREFIX ns: <http://rdf.freebase.com/ns/> SELECT ...",
  "dataset": "cwq"
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 自然语言问题 |
| `question_entity` | dict | 问题涉及的主题实体映射 `{MID: name, ...}`，CWQ 可能包含多个主题实体 |
| `answer_entities` | list[str] | 标准答案列表。注意 CWQ 的答案通常是字面值（如 `"Hinduism"`），而非 Freebase MID |
| `compositionality_type` | str | 问题组合类型：`conjunction`（合取约束）、`composition`（复合关系） |
| 其他字段 | — | 与 WebQSP 格式相同 |

#### GrailQA 数据格式

```json
{
  "qid": "2100023010000",
  "question": "find the fruit source for the wine 2005 lolonis ...",
  "topic_entity": "",
  "topic_entity_name": "",
  "answers": ["m.02ws9_1"],
  "sparql": "PREFIX rdf: <...> SELECT ...",
  "compositionality_type": "i.i.d.",
  "inferential_chain": []
}
```

| 字段 | 类型 | 说明 |
|------|------|------|
| `question` | str | 自然语言问题 |
| `topic_entity` | str | 主题实体 MID（部分样本可能为空） |
| `answers` | list[str] | 标准答案 MID 列表（注意字段名为 `answers`，非 `answer_entities`） |
| `compositionality_type` | str | `i.i.d.`（独立同分布）/ `composition` / `constraint` |

**注意：** GrailQA 的字段命名与 WebQSP/CWQ 不同（`topic_entity` vs `question_entity`，`answers` vs `answer_entities`），预处理脚本会统一转换。

---

### 数据在系统中的使用方式

#### 1. 训练集的用途

训练集在 SymAgent 中承担三个角色：

| 用途 | 代码位置 | 说明 |
|------|----------|------|
| BM25 种子检索 | `planner.py` → `build_seed_index()` | 用训练集的全部问题构建 BM25 索引。Planner 对每个测试问题，检索结构最相似的种子问题，用于构建 few-shot 示例 |
| KG 路径采样 | `planner.py` → `sample_closed_paths()` | 利用种子问题的 `question_entity` → `answer_entities` 在 KG 中做 BFS，采样闭路径作为符号规则来源 |
| 自学习探索 | `self_learning.py` → `online_explore()` | 自学习阶段，Agent 对训练集每个问题执行完整推理，生成轨迹并计算奖励 |

#### 2. 测试集的用途

测试集仅用于评估，不参与训练或规则学习：

| 用途 | 代码位置 | 说明 |
|------|----------|------|
| 评估性能 | `evaluate.py` → `Evaluator.evaluate()` | 对每条测试问题执行 Planner + Executor，计算 Hits@k / F1 / Accuracy |
| 早期停止 | `self_learning.py` → `_evaluate_on_valid()` | 自学习训练中，在验证集上定期评估，提升 < 0.001 时停止迭代 |

#### 3. KG 索引的用途

| 用途 | 代码位置 | 说明 |
|------|----------|------|
| BFS 路径查找 | `kg_environment.py` → `bfs_find_paths()` | Planner 用 BFS 从主题实体到答案实体采样闭路径 |
| 邻居搜索 | `kg_environment.py` → `search_neighbor()` | Executor 执行 `searchNeighbor(entity, relation)` 时查询 KG |
| BM25 实体链接 | `kg_environment.py` → `bm25_retrieve_entities()` | 当问题未提供 `question_entity` 时，用 BM25 从实体名中检索匹配实体 |
| 实体匹配 | `executor.py` → `compute_outcome_reward()` | 用 `mid2name`/`name2mid` 做实体名称 ↔ MID 双向匹配，计算奖励 |

#### 4. 配置文件中的数据路径映射

`configs/config.yaml` 中 `kg.datasets` 节点将数据集名称映射到具体文件：

```yaml
kg:
  datasets:
    webqsp:
      train_file: data/processed/webqsp_train_final.json    # 训练集
      test_file: data/processed/webqsp_test_multihop.json   # 测试集 (仅多跳)
      triple_file: data/freebase/freebase_triples.txt        # KG 三元组
      entity2id: data/freebase/entity2id.txt                 # 实体映射
      relation2id: data/freebase/relation2id.txt             # 关系映射
      mid2name: data/freebase/mid2name.json                  # MID→名称
      name2mid: data/freebase/name2mid.json                  # 名称→MID
```

---

### 数据预处理流程

本项目从三个公开数据集（WebQSP、CWQ、GrailQA）出发，经过预处理生成统一的 KGQA 数据。完整流程：

```
原始数据集
  ├── WebQSP:  data/webqsp/WebQSP.json
  ├── CWQ:     data/cwq/cwq.json
  └── GrailQA: data/grailqa/graliqa.json
        │
        ▼  python scripts/preprocess_dataset.py
预处理数据 (统一格式)
  ├── data/processed/webqsp_processed.json
  ├── data/processed/cwq_processed.json
  └── data/processed/grailqa_processed.json
        │
        ▼  python scripts/filter_multihop.py
多跳过滤数据
  ├── data/processed/webqsp_train_multihop.json  (仅 2-hop)
  ├── data/processed/webqsp_test_multihop.json   (仅 2-hop)
  ├── data/processed/cwq_train_multihop.json     (2-4 hop)
  └── data/processed/cwq_test_multihop.json      (2-4 hop)
        │
        ▼  python scripts/build_kg_from_datasets.py
KG 索引 + 统一 QA 文件
  ├── data/freebase/entity2id.txt
  ├── data/freebase/relation2id.txt
  ├── data/freebase/freebase_triples.txt
  ├── data/freebase/mid2name.json
  ├── data/freebase/name2mid.json
  └── data/freebase/qa_train.json / qa_test.json
```

各步骤说明：

1. **`download_datasets.py`**：从公开源下载 WebQSP、CWQ、GrailQA 的原始 JSON 文件到 `data/<dataset>/` 目录

2. **`preprocess_dataset.py`**：将各数据集的不同格式转换为统一格式，并按 `qid` 排序做 80/20 训练/测试划分
   - WebQSP：从 `Parses[0].Answers` 提取答案 MID，从 `InferentialChain` 提取关系路径
   - CWQ：从 SPARQL 的 `ns:xxx.yyy` 模式提取关系路径，答案为字面值
   - GrailQA：从 `graph_query.edges` 提取关系，从 `answer[].answer_argument` 提取答案

3. **`filter_multihop.py`**：按关系路径长度过滤多跳问题
   - WebQSP 测试集：仅保留 `hop == 2` 的问题（113 条）
   - CWQ 测试集：保留 `2 <= hop <= 4` 的问题（316 条）

4. **`build_kg_from_datasets.py`**：从所有 QA 样本的关系路径构建 KG 三元组索引
   - 单跳：直接生成 `(entity, relation, answer)` 三元组
   - 多跳：生成链式三元组，中间节点用 `__INTERMEDIATE_xxx__` 占位

---

### 测试方法

#### 快速验证（少量样本）

```bash
# 仅评估前 20 条测试样本，快速验证流水线是否正常
python run.py evaluate --dataset webqsp --max_samples 20
```

#### 完整评估

```bash
# WebQSP 多跳测试集完整评估（113 条）
python run.py evaluate --dataset webqsp

# CWQ 多跳测试集完整评估（316 条）
python run.py evaluate --dataset cwq
```

评估过程：对测试集每条问题依次执行 Planner（生成推理路径）→ Executor（Thought-Action-Observation 推理循环）→ 计算指标。

#### 单条问题推理调试

```bash
# 对单条问题执行完整推理，输出推理轨迹
python run.py execute --question "who directed inception" --entity m.03_d0
```

输出示例：
```
=== Trajectory ===
Question: who directed inception
Thought 1: I need to find the director of inception.
Action 1: getReasoningPath(inception, directed by)
Observation 1: Surrounding relational reasoning paths are:
  [film.film.directed_by]
Thought 2: Based on the reasoning path, I'll search for inception's director.
Action 2: searchNeighbor(inception, film.film.directed_by)
Observation 2: m.0bxtg
Thought 3: I found the director entity m.0bxtg.
Action 3: finish(m.0bxtg)

Answer: ['m.0bxtg']
```

#### 仅测试规划能力

```bash
# 仅运行 Planner，查看符号规则生成效果
python run.py plan --dataset webqsp --max_samples 10
```

#### 探索轨迹收集（不训练）

```bash
# 对训练集执行在线探索，收集轨迹和奖励（不触发 LoRA 微调）
python -m src.run explore --dataset webqsp --split train --max_samples 50
```

输出保存到 `logs/<dataset>_<split>_trajectories.json`，包含每条问题的完整推理轨迹、预测答案和奖励。

#### 自学习训练 + 评估

```bash
# 完整流水线：自学习训练 → 测试集评估
python run.py full_pipeline --dataset webqsp
```

#### 使用 `src/run.py` 入口（功能更完整）

```bash
# explore 模式（run.py 不支持，仅 src/run.py 支持）
python -m src.run explore --dataset webqsp --split train --max_samples 50

# 按跳数分组评估（src/run.py 支持 --by_hop）
python -m src.run evaluate --dataset webqsp --by_hop
```

---

### 评估指标说明

#### 指标定义

| 指标 | 公式 | 说明 |
|------|------|------|
| `hits@1` | `1` if `pred[0] ∈ GT` else `0` | 预测的第 1 个答案是否命中标准答案 |
| `hits@3` | `1` if `any(pred[:3]) ∈ GT` else `0` | 预测的前 3 个答案中是否有命中 |
| `hits@10` | `1` if `any(pred[:10]) ∈ GT` else `0` | 预测的前 10 个答案中是否有命中 |
| `f1` | `2 × P × R / (P + R)` | 预测集合与标准答案集合的精确率和召回率的调和平均 |
| `accuracy` | `1` if `pred ∩ GT ≠ ∅` else `0` | 预测答案与标准答案是否有至少一个交集 |

其中 P = Precision = `|pred ∩ GT| / |pred|`，R = Recall = `|pred ∩ GT| / |GT|`。

#### 指标计算方式（代码实现）

代码位置：`src/evaluate.py` → `hits_at_k()`, `f1_score()`, `accuracy()`

1. **实体标准化**：所有实体名称在比较前统一转为小写、去除下划线和连字符
   ```python
   normalize("Freebase_MID") → "freebase mid"
   ```

2. **实体匹配**：支持三种匹配方式（代码位置：`executor.py` → `compute_outcome_reward()`）：
   - **直接匹配**：预测实体与标准答案的标准化形式相同
   - **MID → 名称**：通过 `mid2name` 将预测的 MID 转为名称后匹配（如预测 `m.078w2`，标准答案为 `Samuel Taylor Coleridge`）
   - **名称 → MID**：通过 `name2mid` 将预测的名称转为 MID 后匹配

3. **结果聚合**：所有样本的指标取算术平均

#### 如何评价结果

| 指标 | 好的结果参考 | 含义 |
|------|-------------|------|
| `hits@1` | > 0.65 | 模型的首选答案有很大概率正确 |
| `hits@3` | > 0.70 | 拓宽候选后命中率显著提升 |
| `f1` | > 0.60 | 综合考量了多答案场景的精确率和召回率 |
| `accuracy` | > 0.70 | 至少找到一个正确答案的概率 |

**评价建议：**

- **`hits@1` 是最严格的指标**，直接反映模型的首选答案质量
- **`f1` 适合评估多答案场景**（如 WebQSP 的 "what languages..." 类问题可能有十几个正确答案），因为它同时考虑了精确率和召回率
- **`accuracy` 偏宽松**，只要找到一个正确答案就算对，适合粗略判断模型方向是否正确
- **CWQ 通常比 WebQSP 更难**，因为 CWQ 包含组合约束条件（conjunction），需要同时满足多个约束
- **对比时注意测试集版本**：本项目使用的是 multihop 过滤后的测试集（WebQSP 113 条, CWQ 316 条），与全量测试集的结果不可直接比较

**实体匹配对指标的影响：**

`mid2name`/`name2mid` 映射的正确加载对评估结果有显著影响。如果 Agent 输出的是实体名称（如 `"Aragorn"`）而标准答案是 Freebase MID（如 `"m.0gwlg"`），没有映射文件会导致所有匹配失败。因此 WebQSP 数据集配置中必须设置 `mid2name` 和 `name2mid` 路径（CWQ 配置中当前未设置，因为 CWQ 的答案本身就是字面值）。

---

## 核心算法与代码映射

| 论文公式 | 说明 | 代码位置 |
|----------|------|----------|
| Eq.1 | 符号规则泛化 `r(x,y) <- r1(x,z1) ∧ r2(z1,y)` | `planner.py` → `AgentPlanner.generalize_to_rules()` |
| Eq.3 | LLM 规则诱导 `p ~ π(·|ρ_Plan, q, M)` | `planner.py` → `AgentPlanner._llm_induce_rules()` |
| Eq.4 | 交互轨迹 `H_n = (q, G, p, τ_0, a_0, o_0, ...)` | `executor.py` → `Trajectory` 类 |
| Eq.6 | 结果奖励 `r(μ) = Recall(A_μ, A_gt)` | `executor.py` → `compute_outcome_reward()` |
| Eq.7 | 启发式合并（4 分支选择） | `self_learning.py` → `heuristic_merge()` |
| Eq.8 | SFT 损失（indicator mask） | `self_learning.py` → `prepare_training_data()` |
| Algo.1 | KG 不完整性模拟 | `kg_environment.py` → `KGEnvironment.simulate_incompleteness()` |
| Section 4.1 | BM25 种子检索 + BFS 路径采样 | `planner.py` → `retrieve_seed_questions()` + `sample_closed_paths()` |
| Section 4.2.1 | 动作工具库 (5 种动作) | `executor.py` → `AgentExecutor._execute_action()` |
| Section 4.2.2 | ReAct 推理循环 | `executor.py` → `AgentExecutor.execute()` |
| Section 4.3.1 | 在线探索 + 自我精炼 | `self_learning.py` → `SelfLearner.online_explore()` + `self_refine()` |
| Section 4.3.2 | 离线迭代策略更新 (LoRA SFT) | `self_learning.py` → `SelfLearner.fine_tune()` |

---

## 常见问题

### Q: 训练时提示 `Target modules not found in the base model`

A: `lora_target_modules` 配置与模型架构不匹配。不同模型的层名不同：
- LLaMA/Qwen 系列：`q_proj, k_proj, v_proj, o_proj, down_proj, up_proj, gate_proj`
- GPT-2：`c_attn, c_proj`
- ChatGLM：`query_key_value, dense`

### Q: 评估结果与论文不一致

A: 检查以下几点：
1. 测试集是否使用了 multihop 版本（`*_test_multihop.json`）
2. `mid2name`/`name2mid` 映射是否正确加载（影响实体匹配）
3. LLM 模型和温度参数是否与论文一致
4. `max_infer_steps` 是否足够（复杂多跳问题可能需要更多步）

### Q: API 调用遇到 429 限速

A: 增大 `min_request_interval`（如设为 3.0 秒），系统会自动指数退避重试（最多 5 次）。

### Q: LoRA 微调显存不足

A: 尝试以下方法：
1. **开启 `load_in_4bit: true`（QLoRA）** —— 单卡 24GB 首选，7B 从 ~15GB 压到 ~6GB
2. 减小 `batch_size` 到 1
3. 增大 `gradient_accumulation_steps`（如 4 或 8）
4. 减小 `max_seq_length`（如 2048）
5. 使用 `lora_r=16`（降低 LoRA 参数量）

### Q: 探索阶段就 OOM（不是微调阶段）

A: 多半是 `exploration_mode: local` 在单卡上加载了 7B 做推理，加长上下文 logits 峰值爆显存。单卡 24GB 改用 `exploration_mode: online`（探索走 API，微调时才加载本地模型）。

### Q: wikiSearch 卡住很久 / 报 503 / 超时

A: 服务器访问不了 Wikipedia（常见于墙内环境）。设 `executor.wiki_search_enabled: false` 关闭它，agent 改用 KG 或自身知识作答，避免每次卡数十秒。若有可直连的代理也可配 `https_proxy` 环境变量。

### Q: 训练只跑了 API 推理，没有触发 LoRA 微调

A: 检查 `self_learning.base_model` 是否配置了本地模型路径。为空时系统会跳过 LoRA 微调（仅采集/合并轨迹，不更新参数）。注意是 `self_learning.base_model`，不是 `llm` 段的参数。

---

## 依赖

Python 3.10+, PyTorch 2.0+, HuggingFace Transformers 4.36+, 支持 LoRA 微调的任意 LLM。

### 完整依赖列表

torch>=2.0.0
transformers>=4.36.0
accelerate>=0.25.0
peft>=0.7.0
openai>=1.0.0
rank_bm25>=0.2.2
wikipedia-api>=0.6.0
networkx>=3.0
numpy>=1.24.0
pandas>=2.0.0
pyyaml>=6.0
tqdm>=4.65.0
datasets>=2.14.0
scipy>=1.10.0
scikit-learn>=1.3.0
```