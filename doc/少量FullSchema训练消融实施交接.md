# 少量 Full Schema 训练消融实施交接

## 1. 文档目的

本文档供下一位 Agent 实施训练样本量消融实验时使用。其目标是验证论文的核心部署叙事：

```text
少量代表性记录的 Full Schema 抽取
    -> 自动生成字段存在性和输出长度银标
    -> 训练轻量字段选择与长度预测模型
    -> 面向同一数据源的大规模记录动态裁剪 Schema
    -> 以一次性启动成本换取持续的质量-Token 收益
```

本轮任务只运行离线机器学习训练、预测、字段选择和评估，严禁调用 LLM API。真实 LLM
抽取仅在离线曲线完成后，针对少数代表训练规模单独决定是否执行。

## 2. 论文整体定位

### 2.1 研究问题

项目面向大规模文化遗产记录的结构化信息抽取。对每条记录直接输入完整 18 字段 Schema
会产生大量无信息字段、冗余输出和不必要的 Token 消耗。

本文方法不主张当前模型能够零样本迁移到任意博物馆。主线是同一数据源内的可部署流程：
先用少量 Full Schema 结果构造领域监督，再学习记录感知的字段选择器和长度预测器，为后续
大量同域记录选择高预期效用字段。

字段选择效用结合：

```text
字段存在概率 x 字段价值权重 / 预测字段成本
```

预算方法使用 0-1 背包在 `estimated_field_cost` 约束下选择字段；该预算只包含字段名、字段
定义和预测输出 Token，不是端到端 API Total Token 上限。

### 2.2 可主张与不可主张的结论

当前可以检验和报告：

- 少量 Full Schema 监督是否足以训练有效的字段选择器；
- 训练样本量增加时，字段预测、长度预测、字段覆盖率和估计成本如何变化；
- 启动监督成本在大规模同域处理时的理论摊销条件；
- 与 Full Schema、Prompt-only 和 SchemaRAG 的质量-Token 权衡。

当前不能声称：

- 任意博物馆之间的零样本泛化；
- 少量训练数据必然提升 Full Schema 的最终抽取质量；
- 仅依据 Presence 或 Exact Value Recall 即证明综合抽取质量更高；
- Silver/Model-assisted reference 等同于独立人工 Gold。

更严谨的结论措辞是“在显著降低 Token 的同时维持接近或更优的质量-成本权衡”；只有
统一 Quality 评估完成后，才可对“质量更优”作最终结论。

## 3. 已有数据与严格隔离

### 3.1 数据事实

| 数据集 | 数量 | 用途 |
| --- | ---: | --- |
| `training_dataset.jsonl` | 6763 | 现成 Full Schema 结果构造的训练样本与银标 |
| `data/splits/train_ids.json` | 5410 | 本轮允许抽取训练子集的唯一训练池 |
| `data/splits/validation_ids.json` | 677 | 固定阈值选择和模型配置验证集 |
| `data/splits/test_ids.json` | 676 | 固定离线测试集 |
| `data/gold/gold_main_ids.json` | 500 | 冻结真实 LLM 对比样本 |

`gold_main` 的 500 条记录全部在现有测试集内，且与训练集、验证集不存在重叠。训练量消融
不得从测试集或 `gold_main` 取任何记录，不得为了扩展训练量将 6763 条全部用作训练数据。

### 3.2 现有监督和模型

训练样本由 `generate_training_dataset.py` 从原始记录和既有 Full Schema 结果生成，包含：

```text
record_id
record_text
18 个 <field>_present 标签
18 个 field_output_tokens 标签
```

当前训练实现位于 `train_field_selector.py`：

- 输入：原始 JSON 的紧凑序列化文本；
- 特征：字符级 TF-IDF，2-5 gram；
- Presence：每字段一个 class-balanced Logistic Regression；
- Length：每字段一个 Ridge 对数长度回归；
- 阈值：仅在固定验证集选择；
- 测试：仅使用固定测试集。

本轮不替换模型架构，不修改 `field_weights.json`，不改变 `simple2.json` 的 18 字段 Schema。
目标是先得到可解释、可复现的训练量曲线，而不是同时改变多个因素。

## 4. 已有正式实验事实

### 4.1 内部方法

冻结条件为 `kg-assist-pro`、`temperature=0.0`、稳定 Prompt、并发 20、队列 40、最多 2 次
重试和 `gold_main` 500 条记录。内部 v3 Pro 结果文件为：

```text
experiments/gold_main_v3_stable_pro_results.jsonl
reports/gold_main_v3_stable_pro_all_methods_paired_metrics.json
```

主分析仅保留：`full`、`prompt-null`、`prompt-omit`、`budget-200`、`budget-400` 和
`lambda-0.003`。`fixed-5`、`fixed-10` 的字段集合偏向数据稠密字段，仅作补充诊断，
不进入主表、主排序或核心结论。

### 4.2 外部 SchemaRAG 基线

外部方法 `external-schemarag-k10` 是基于论文描述的适配重实现，不是官方代码复现。它在相同
LLM、Prompt、Schema 和样本条件下运行，结果见：

```text
experiments/external_schemarag_k10_gold_main_v3_pro.jsonl
reports/external_schemarag_k10_gold_main_v3_pro_metrics.json
```

七种拟比较方法因 `prompt-omit` 的 24 条最终 JSON 解析失败，共同成功样本为 476 条。已在
这 476 条上确认：`budget-200` 相比 SchemaRAG k=10 同时具有更高的 Presence/Exact Value
Recall/Gold 覆盖率，以及更低的平均 Total Token。该结论仍只限于现有 Presence、严格值匹配和
Token 指标，最终综合质量仍待统一评估。

### 4.3 不属于本轮的工作

以下工作已识别但不属于训练量消融：

- 非 `desc` 字段的 LLM Value Judge；
- `desc` 字段 embedding cosine score；
- 综合 Quality、bootstrap 置信区间和显著性分析；
- 代表训练规模的真实 LLM 重跑；
- 跨博物馆零样本或少样本迁移实验。

## 5. 本轮实施任务

### 5.1 训练规模与随机性

从固定 5410 条训练池构建以下训练规模：

```text
50 / 100 / 250 / 500 / 1000 / 2500 / 5410
```

每个规模使用 5 个预先固定的随机种子。所有规模应尽量使用嵌套训练集：同一随机种子下，
50 条是 100 条的子集，依此类推。抽样必须尽可能维持 18 个 Presence 标签的分布；极稀有字段
在最小规模中无法同时覆盖时，必须在元数据和报告中明确记录，不得伪造分层成功。

建议固定种子为：

```text
42 / 43 / 44 / 45 / 46
```

验证集、测试集、Gold、字段权重、Schema、TF-IDF 超参数和模型超参数必须在所有运行中保持
不变。每个训练规模只改变训练子集大小。

### 5.2 推荐执行顺序

1. 新增独立的训练量消融脚本或为现有训练脚本增加最小参数，优先复用现有训练和评估逻辑。
2. 先执行 `n=50, seed=42` 冒烟，验证训练、输出目录、指标和隔离检查。
3. 通过冒烟后执行 7 个训练规模乘以 5 个种子的完整矩阵。
4. 汇总每次运行的字段预测、长度预测和预算选择离线指标。
5. 生成学习曲线、汇总 CSV/JSON 和图形。
6. 用固定测试集或其 `gold_main` 子集进行离线预算选择评估；不得调用 LLM。
7. 完成后更新 `doc/正式实验记录.md`，记录配置、产物、关键曲线和结论边界。

### 5.3 输出隔离

禁止覆盖以下目录和文件：

```text
models/field_selector/
reports/field_selector_metrics.*
reports/field_output_token_metrics.*
experiments/gold_main_v3_stable_pro_results.jsonl
experiments/external_schemarag_k10_gold_main_v3_pro.jsonl
```

建议使用独立目录：

```text
models/training_size_ablation/<n>/seed_<seed>/
reports/training_size_ablation/runs/<n>/seed_<seed>/
reports/training_size_ablation/summary.json
reports/training_size_ablation/summary.csv
reports/training_size_ablation/figures/
```

每个运行必须保存训练 ID、训练池哈希、验证/测试 ID 哈希、随机种子、训练规模、代码参数、
模型元数据和指标。结果目录不可借用当前正式模型目录。

## 6. 离线评估要求

### 6.1 字段存在性预测

固定测试集上至少报告：

- 每字段 Precision、Recall、F1 和 Average Precision；
- 宏平均和微平均 Presence 指标；
- 稀有字段的正样本数与指标缺失情况；
- 相对完整训练池模型的性能恢复比例。

不得只汇报整体均值掩盖稀有字段退化。

### 6.2 输出长度预测

固定测试集上按字段报告：

- MAE；
- RMSE；
- Median Absolute Error；
- Spearman；
- 正样本数；
- 中位数回退模型与 Ridge 模型的使用情况。

### 6.3 预算选择离线指标

对 `budget-200` 和 `budget-400` 至少评估：

- Gold 非空字段覆盖率；
- 选中字段数量；
- `estimated_field_cost`；
- 与完整 5410 条模型相比的字段集合 Jaccard 相似度；
- 预算是否满足；
- 每字段被选中频率。

若使用 `gold_main` 子集计算覆盖率，必须单独标明它是固定测试集的子集，且不能把它当作训练
或验证数据。此处的成本仍是字段估计成本，不能称作实际 LLM Total Token。

### 6.4 汇总与图形

每个训练规模对 5 个种子报告均值、标准差和 95% 置信区间。至少生成：

1. 训练样本数对 Presence Macro/Micro-F1 的曲线；
2. 训练样本数对长度 MAE 的曲线；
3. 训练样本数对 `budget-200/400` Gold 覆盖率和估计成本的曲线；
4. 训练样本数对完整训练池模型性能恢复比例的曲线。

横轴应使用对数刻度或明确的离散训练规模，不得将不等间距规模画成误导性的线性间距。报告中
需要自动识别首次达到完整训练池模型 90%、95% 和 99% 性能的最小训练规模；若未达到，必须
明确写为未达到。

## 7. 启动成本与后续真实实验

训练量曲线完成后，才可选择代表点，例如 100、500、1000 和 5410 条，进行独立的真实 LLM
实验。真实实验前必须再次确认模型目录、样本集、Prompt、模型、温度和重试策略都固定。

用于论文的成本回收分析定义为：

```text
C_startup = n * C_full + C_train
S_per_record = C_full - C_adaptive
N_break_even = C_startup / S_per_record
```

其中 `C_full` 和 `C_adaptive` 必须使用相同样本、同一 API 计量口径下的实际 Total Token 或
明确单价换算后的成本。当前仅可将此作为后续分析框架，不能把离线估计成本直接代入并称为
真实成本回收点。

## 8. 验收标准

交付前必须逐项满足：

- [ ] 训练子集只来自 5410 条训练池，与验证、测试和 Gold 的交集为零；
- [ ] 7 个规模和 5 个种子均有可解析、独立保存的结果；
- [ ] 同种子下训练子集满足嵌套关系，或记录无法满足的明确原因；
- [ ] 每次运行保存训练 ID 和完整可复现元数据；
- [ ] 训练过程未调用 LLM API，未改写正式模型或正式结果；
- [ ] 汇总结果包含字段级和总体 Presence、长度、覆盖率与估计成本指标；
- [ ] 曲线报告均值、不确定性和相对完整训练池模型的恢复比例；
- [ ] 报告明确区分离线估计成本与真实 API Token/费用；
- [ ] 文档和正式实验记录均更新，且 `git diff --check` 通过。

## 9. 关键文件索引

| 文件 | 作用 |
| --- | --- |
| `training_dataset.jsonl` | 6763 条 Full Schema 银标训练样本 |
| `data/splits/*.json` | 固定 train/validation/test ID 划分 |
| `train_field_selector.py` | 现有训练、验证阈值选择和测试指标实现 |
| `split_dataset.py` | 多标签分层划分实现 |
| `budget_selector.py` | 预算约束字段选择 |
| `llm_runner.py` | 正式 LLM 实验运行器，仅供后续代表点真实运行 |
| `field_weights.json` | 固定字段价值权重配置 |
| `simple2.json` | 冻结 18 字段 Schema |
| `doc/论文实验底稿.md` | 论文主线与当前实验边界 |
| `doc/正式实验记录.md` | 已完成正式 LLM 实验记录 |
| `doc/SchemaRAG_k10_实施交接.md` | 外部 SchemaRAG 方法实现和结果交接 |
