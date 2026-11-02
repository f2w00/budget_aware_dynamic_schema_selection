# 会话交接文档：预算感知 Schema 抽取实验

## 1. 项目目标

本项目面向大规模文化遗产原始数据，研究记录感知、预算感知的动态 Schema
抽取方法。核心目标不是单纯提高字段预测准确率，而是证明：

```text
在相近抽取质量下，Adaptive Schema 比仅通过 Prompt 指导 LLM 忽略无信息字段
消耗更少的输入 Token、输出 Token 和总 Token。
```

项目同时希望贴合 `pdcat` 的生产处理主题，体现快速 ML 模型和慢速 LLM 之间的
生产者-消费者关系，以及队列、背压、并发和任务预算调度。

## 2. 数据和参考结果

- `public.artifact_raw(artifact_id, raw_data)` 保存原始 JSON。
- `public.artifact_extract` 保存已有的 Full Schema 抽取结果。
- 两表通过 `artifact_id` 关联，实际交集为 `6763` 条。
- 原始输入必须来自 `artifact_raw.raw_data`，不能从 `artifact_extract` 反向生成。
- `artifact_extract` 当前只能作为 Full Schema 的弱监督参考，不能称为人工 Gold。
- 当前目标 Schema 有 18 个字段，定义位于 `simple2.json`。
- `field_weights.json` 保存字段价值权重和必选字段。

## 3. 已完成的数据和 ML 模型

已完成：

- 生成 `training_dataset.jsonl`，共 6763 条记录。
- 生成字段 `present` 和 `field_output_tokens` 标签。
- 多标签分层划分：train 5410、validation 677、test 676。
- 训练字段 present 预测模型。
- 训练字段输出 Token 预测模型。

当前模型结构：

```text
共享字符级 TF-IDF：char，ngram_range=(2, 5)
present：每字段 Logistic Regression
输出长度：每字段 Ridge，目标为 log1p(field_output_tokens)
```

模型文件目录：

```text
models/field_selector/
```

相关代码：

- `generate_training_dataset.py`
- `train_field_selector.py`
- `predict_field_selector.py`
- `split_dataset.py`

## 4. 四类预算定义

必须严格区分以下概念：

```text
C_full
    全字段 Full Schema 的估计成本。
    = Σ(字段名称 Token + 字段定义 Token + 预测输出 Token)
    当前不包含原始记录输入 Token。

B_required
    达到当前方法目标所需的选中字段预计成本。
    = Σ(选中字段名称 Token + 字段定义 Token + 预测输出 Token)

B_allocated
    Scheduler 实际分配给任务的预算。
    当前实现只记录该字段，尚未实现全局资源紧张时的动态降级。

C_actual
    LLM API 实际消耗。
    = 实际输入 Token + 实际输出 Token，包含实际重试产生的资源消耗。
```

重要区别：

```text
把所有字段预算相加得到的是 C_full，不是 Adaptive 任务预算。
Producer 的 B_required 应来自字段选择结果，只累加被选中的字段。
```

字段选择成本公式由 `llm_runner.fields_cost()` 统一实现，
`task_planner.py` 直接复用该函数。

## 5. 字段选择方法

当前支持的方法：

```text
full
    完整 18 字段 Schema。

prompt-null
    完整 Schema + Prompt-only 指令变体。
    无信息字段不要推断，直接输出 null；全部字段仍需输出。

prompt-omit
    完整 Schema + Prompt-only 指令变体。
    无信息字段允许直接省略，不出现在模型原始 JSON 中。
    runner 会将结果补齐为 null，并记录 missing_fields_expected=true。

fixed-k
    固定字段集合，按人工字段权重排序。

budget-k
    使用 0-1 背包，在预算 Token 内最大化字段加权效用。

lambda-x
    无硬预算选择，字段满足 utility - lambda * cost > 0 才选择。
```

选择器代码：

- `budget_selector.py`
- `unbudgeted_selector.py`
- `field_weights.json`

已知离线结果：

- 固定预算 B=100/200/400 的 Full Schema 非空字段覆盖率约为
  26.06%/53.46%/94.95%。
- 无硬预算验证集选择 `lambda=0.003`。
- 该设置测试集平均估计字段成本约 425.71 Token，平均选中 8.71 个字段，
  Full Schema 非空字段覆盖率约 99.00%。

这些是离线估计或弱监督回放结果，不是真实 LLM Gold 准确率。

## 6. Producer-Consumer 实现

### 6.1 任务规划器

`task_planner.py` 负责：

- 读取测试集原始记录。
- 每条记录只执行一次 ML 字段预测。
- 为各方法复用该预测结果。
- 计算 `C_full`。
- 计算当前方法的 `B_required`。
- 输出任务规划 JSONL。

示例：

```bash
uv run python task_planner.py \
  --split test \
  --limit 100 \
  --output experiments/task_plan.jsonl
```

任务规划行至少包含：

```text
task_id
artifact_id
method
method_group
setting
budget
lambda
selected_fields
full_estimated_cost
estimated_field_cost
required_budget
allocated_budget
budget_status
```

### 6.2 LLM runner

`llm_runner.py` 当前流程：

```text
ML 预测和任务构造 Producer
    ↓
有界 asyncio.Queue
    ↓
多个 LLM Consumer
    ↓
结果 JSONL Writer
```

当前实现特点：

- 默认队列容量是并发数的 2 倍。
- 可通过 `--queue-size` 调整队列容量。
- Consumer 数量由 `--max-concurrency` 控制。
- 队列按 FIFO 追加结束哨兵，Consumer 处理完真实任务后退出。
- `queue_wait_ms` 记录任务从进入队列到开始调用 LLM 的等待时间。
- `enqueue_ms` 记录 Producer 入队耗时，可反映背压。
- `pipeline_ms` 记录整个 Consumer 流水线耗时。
- 支持超时、指数退避重试、单条失败隔离和断点续跑。
- 成功结果会补齐缺失字段为 JSON `null`。
- `prompt-omit` 的缺失字段会通过 `missing_fields_expected` 标记为预期行为。

当前还没有真正实现：

- 多模型 Worker 的实际路由。
- 全局预算池和 Scheduler 动态降级。
- 任务状态数据库；目前状态主要通过 JSONL 结果行保存。

### 6.3 Prompt 模板

```text
1.prompt
    Full、Fixed 和 Adaptive 共用的基础模板。

1_prompt_null.prompt
    prompt-null 变体。

1_prompt_omit.prompt
    prompt-omit 变体。
```

每个方法单独记录 `prompt_variant` 和 `prompt_hash`。修改 Prompt 变体会改变
对应的 `experiment_id` 和断点判断。

## 7. 评估脚本

`evaluate_llm_experiments.py` 当前会：

- 读取成功的 JSONL 结果。
- 从 `artifact_extract` 读取弱监督参考值。
- 计算 presence precision/recall/F1。
- 计算 selected gold coverage 和 exact value recall。
- 汇总输入、输出、总 Token、延迟、队列等待、重试次数、实际成本和估计成本。
- 输出 `methods` 方法级汇总。
- 输出 `method_groups` 方法组汇总，例如 `full`、`prompt_only`。

运行示例：

```bash
uv run python evaluate_llm_experiments.py \
  --input experiments/serial_smoke_final.jsonl \
  --output reports/llm_smoke_final_metrics.json
```

当前评估仍使用 `artifact_extract` 弱监督参考。正式论文必须建立人工 Gold
或证据标注集，建议从测试集分层抽取 200～300 条。

## 8. 已完成验证

### 8.1 编译和 dry-run

以下检查通过：

```bash
uv run python -m py_compile \
  llm_runner.py task_planner.py evaluate_llm_experiments.py

uv run python llm_runner.py \
  --limit 5 \
  --dry-run \
  --output experiments/dry_run_check.jsonl
```

dry-run 生成 30 个任务，即 5 条记录 × 6 种默认方法。任务元数据检查通过，
Full、prompt-null、prompt-omit 使用相同的 18 个字段，但 Prompt 哈希不同。

### 8.2 任务规划器

```bash
uv run python task_planner.py \
  --limit 5 \
  --output experiments/task_plan_check.jsonl
```

检查结果：

- 生成 35 个任务，即 5 条记录 × 7 种方法。
- `C_full` 已包含预测输出 Token。
- Full、prompt-null、prompt-omit 的 `B_required == C_full`。
- 自适应方法的估计成本均不超过 `C_full`。
- task planner 和 runner 对自适应方法的成本估计一致。

### 8.3 真实 LLM 串行冒烟

最终验证文件：

```text
experiments/serial_smoke_final.jsonl
reports/llm_smoke_final_metrics.json
```

运行方式：

```bash
uv run python llm_runner.py \
  --limit 2 \
  --methods full prompt-null prompt-omit \
  --max-concurrency 1 \
  --output experiments/serial_smoke_final.jsonl
```

结果：

- 6/6 调用成功。
- 队列等待时间随任务排队增长，Producer 的背压可观测。
- `prompt-omit` 最终输出英文 Schema 键，`unexpected_fields=0`。
- `prompt-omit` 在两条记录上分别省略 6 个和 5 个字段。
- 2 条记录样本太小，不能据此做质量或成本结论。

曾出现的错误：早期 `prompt-omit` 模板偏离基础模板过大，模型输出了中文键名。
后来将模板改成与 `1.prompt` 最小差异，并增加“JSON 键名必须与目标字段列表
完全一致”的规则，最终冒烟通过。正式实验仍应持续检查 `unexpected_fields`。

## 9. 当前文件清单

核心实现：

- `llm_runner.py`：Prompt 构造、字段选择、LLM 调用、队列 Consumer、重试、断点。
- `task_planner.py`：Producer 任务计划与四类预算。
- `llm_prompt.py`：基础 Prompt 构造。
- `budget_selector.py`：硬预算 0-1 背包选择。
- `unbudgeted_selector.py`：lambda 无硬预算选择。
- `evaluate_llm_experiments.py`：真实 LLM 结果评估。

配置和模板：

- `1.prompt`
- `1_prompt_null.prompt`
- `1_prompt_omit.prompt`
- `simple2.json`
- `field_weights.json`
- `.env`：只从环境变量读取 API 配置，禁止提交密钥。

模型和数据：

- `training_dataset.jsonl`
- `models/field_selector/`
- `data/splits/`

文档：

- `doc/论文实验底稿.md`：项目论文实验底稿。
- `doc/research-plan.md`
- `doc/plan3.md`
- `doc/session_handoff_2026-09-01.md`：本文档。

## 10. 下一步实施顺序

### 必须先做

1. 建立 200～300 条人工 Gold 或原文证据标注集。
2. 明确字段值、字段存在性、证据片段和多值字段的标注规范。
3. 在同一批记录、同一模型、同一温度和同一并发设置下运行：

```text
Full
Prompt-only prompt-null
Prompt-only prompt-omit
Fixed-5 / Fixed-10
Adaptive budget-200 / budget-400
Adaptive lambda-0.003
```

4. 输出实际 Cost-Quality 曲线，而不是只输出离线估计曲线。
5. 计算配对记录级的 Token Reduction、质量差异和置信区间。

### 随后做

1. 增加真实的多模型配置，例如 fast worker 和 quality worker。
2. 将 `B_required`、`B_allocated` 和任务难度用于 Worker 路由。
3. 增加 Scheduler 的全局预算池和降级记录。
4. 增加任务状态持久化、恢复和幂等测试。
5. 比较 Producer-Consumer 与简单批量并发的吞吐、P50/P95 延迟和失败率。

## 11. 论文结论边界

当前可以声称：

```text
已经实现记录感知字段选择、预算估计、有界 Producer-Consumer 调度框架，
并完成 Prompt-only 基线和真实 LLM 串行冒烟验证。
```

当前不能声称：

```text
Adaptive Schema 已经在真实 Gold 质量上优于 Prompt-only。
Adaptive Schema 已经证明能够降低实际 API 费用。
多模型路由已经带来吞吐或质量收益。
```

这些结论必须等待人工 Gold、正式样本规模和完整对照实验完成。

## 12. 环境和运行注意事项

- Python 环境位于父目录：`/root/work/papers/.venv`。
- 在 `/root/work/papers/extraction` 目录执行 `uv run` 正常。
- 真实调用依赖 `.env` 中有效的 `DATABASE_URL`、`LLM_API_KEY`、可访问的
  `LLM_BASE_URL`。
- 默认模型是 `kg-assist`，默认并发读取 `LLM_MAX_CONCURRENCY`，没有价格配置时
  `actual_cost` 为 0 或 null，Token 才是可信的成本指标。
- 当前 Git 状态显示 extraction 目录整体为未跟踪状态，不要在未确认前提交或清理。
