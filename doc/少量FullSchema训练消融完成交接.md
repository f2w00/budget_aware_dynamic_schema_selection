# 少量 FullSchema 训练消融完成交接

## 1. 交接范围

本轮已完成 FullSchema 字段选择器的训练样本量消融实验。实验只进行离线机器学习训练、
预测、字段选择和评估，未调用 LLM API，也未覆盖正式模型、正式实验结果或现有正式评估报告。

本文件记录已完成的实现、数据隔离、结果、产物位置、复现命令和后续使用边界，供后续 Agent
继续分析或选择真实 LLM 实验代表规模时使用。

## 2. 实验配置

### 2.1 固定数据划分

| 数据或文件 | 数量 | 用途 |
| --- | ---: | --- |
| `training_dataset.jsonl` | 6763 | Full Schema 银标训练样本 |
| `data/splits/train_ids.json` | 5410 | 唯一训练池 |
| `data/splits/validation_ids.json` | 677 | 固定验证集和阈值选择 |
| `data/splits/test_ids.json` | 676 | 固定离线测试集 |
| `data/gold/gold_main_ids.json` | 500 | test 的固定子集，用于 Gold 覆盖评估 |

脚本启动时会检查：

- train、validation、test 三个划分覆盖数据集且互不重叠；
- `gold_main` 与 train、validation 无交集；
- `gold_main` 完全属于固定 test 子集；
- 所有训练子集只从 5410 条 train pool 中抽取。

### 2.2 训练规模和随机种子

训练规模为：

```text
50 / 100 / 250 / 500 / 1000 / 2500 / 5410
```

每个规模使用 5 个 seed：

```text
42 / 43 / 44 / 45 / 46
```

同一个 seed 下使用多标签缺口贪心顺序生成嵌套训练前缀，因此较小规模是较大规模的前缀。
抽样目标尽量维持 18 个 Presence 标签的分布；稀有标签无法在最小规模严格分层时，不伪造
分层结果，实际训练 ID 和抽样方式保存在各 run 的模型元数据中。

### 2.3 固定模型配置

- Schema：`simple2.json`，固定 18 个字段。
- 字段权重和 required fields：`field_weights.json`。
- 字段成本：`schema_costs.json`。
- 文本特征：字符级 TF-IDF，`ngram_range=(2, 5)`、`min_df=2`、`max_features=100000`、
  `sublinear_tf=True`。
- Presence 模型：每个字段一个 `LogisticRegression`，`class_weight="balanced"`、
  `solver="liblinear"`、`max_iter=1000`。
- Presence 阈值：只使用固定 validation 选择，test 不参与阈值选择。
- Length 模型：每个字段一个 `Ridge(alpha=10.0)`，对输出 token 数使用 `log1p` 回归并
  用 `expm1` 还原。
- 长度样本不足 `10` 个正样本时使用训练正样本中位数常数回退。
- Budget 选择：复用 `budget_selector.py` 的 0-1 knapsack，预算为 200 和 400。

## 3. 已完成实现

主脚本为 [`training_size_ablation.py`](../training_size_ablation.py)，完成以下工作：

1. 读取 JSONL 和固定 train/validation/test ID 划分。
2. 校验 `gold_main` 的测试集子集性质和数据隔离。
3. 为每个 seed 构建嵌套训练顺序。
4. 训练 7 个规模的 Presence 和 Length 模型。
5. 在固定 validation 上选择 Presence 阈值，在固定 test 上生成最终指标。
6. 对 budget-200 和 budget-400 逐条 test 记录进行字段选择。
7. 以同 seed 的完整 5410 模型作为逐记录字段集合 Jaccard 基线。
8. 保存模型、元数据、逐 run 指标、汇总统计、恢复阈值和学习曲线。

重要实现修正：预算 Jaccard 不是将所有记录的字段合并成一个 union 后计算，而是对每条
记录分别计算，再对 676 条 test 记录取均值；完整模型 baseline 也按 seed 缓存并复用。

## 4. 运行结果

完整矩阵为 35/35 个 run，7 个规模均有 5 个 seed。主要均值如下：

| 训练规模 | Presence Macro-F1 | Presence Micro-F1 | Length MAE | Budget-200 Gold 非空覆盖 | Budget-400 Gold 非空覆盖 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 50 | 0.5156 | 0.8909 | 4.9682 | 0.5217 | 0.9401 |
| 100 | 0.5187 | 0.9126 | 4.8046 | 0.5166 | 0.9413 |
| 250 | 0.5522 | 0.9296 | 3.7907 | 0.5173 | 0.9421 |
| 500 | 0.5940 | 0.9362 | 3.5017 | 0.5234 | 0.9459 |
| 1000 | 0.6245 | 0.9459 | 3.3123 | 0.5292 | 0.9455 |
| 2500 | 0.6916 | 0.9590 | 3.0084 | 0.5394 | 0.9479 |
| 5410 | 0.7139 | 0.9646 | 2.9678 | 0.5420 | 0.9483 |

每个 run 的详细报告还包含：

- 每字段 Presence Precision、Recall、F1、Average Precision 和测试集正样本数；
- Presence Macro-F1、Micro-F1；
- 每字段 Length MAE、RMSE、Median Absolute Error、Spearman、正样本数和模型类型；
- budget-200/400 的 Gold 非空记录数、Gold 非空覆盖率、整体 Gold 覆盖率、平均选中字段数；
- `estimated_cost`、预算满足率、逐记录 `jaccard_vs_full`；
- 18 个字段在 test 记录中的选择频率。

### 4.1 恢复阈值

阈值文件为 `reports/training_size_ablation/thresholds.json`：

| 指标 | 90% | 95% | 99% |
| --- | ---: | ---: | ---: |
| Presence Macro-F1 恢复率 | 未达到 | 未达到 | 未达到 |
| Presence Micro-F1 恢复率 | 100 | 2500 | 未达到 |
| Length MAE 恢复率 | 2500 | 2500 | 5410 |

恢复率相对于同 seed 的完整 5410 模型计算。Length MAE 使用完整模型 MAE 除以当前规模
MAE，因此数值越高越好。

## 5. 产物索引

### 5.1 代码和配置

| 产物 | 路径 | 说明 |
| --- | --- | --- |
| 消融脚本 | `training_size_ablation.py` | 训练、评估、汇总和绘图入口 |
| 原训练实现 | `train_field_selector.py` | 原有字段选择器逻辑参考 |
| 预算选择器 | `budget_selector.py` | 字段成本和 0-1 knapsack 选择 |
| 冻结 Schema | `simple2.json` | 18 字段定义 |
| 字段权重 | `field_weights.json` | required fields 和价值权重 |
| 字段成本 | `schema_costs.json` | estimated field cost 配置 |

### 5.2 模型

模型根目录为：

```text
models/training_size_ablation/
```

目录结构为：

```text
models/training_size_ablation/<size>/seed_<seed>/
```

本轮包含 7 个规模 × 5 个 seed，共 35 个模型目录、210 个模型和元数据文件。每个模型目录
包含：

```text
vectorizer.joblib
classifiers.joblib
length_regressors.joblib
constants.json
length_metadata.json
metadata.json
```

`metadata.json` 保存训练规模、seed、训练 ID、训练池 hash、validation/test ID hash、Gold
ID hash、字段列表、Python/sklearn 版本、训练时间、抽样方式和 `llm_api_called: false`。

### 5.3 报告和图形

逐 run 报告根目录：

```text
reports/training_size_ablation/runs/<size>/seed_<seed>/metrics.json
```

汇总文件：

```text
reports/training_size_ablation/summary.json
reports/training_size_ablation/summary.csv
reports/training_size_ablation/thresholds.json
```

学习曲线：

```text
reports/training_size_ablation/figures/presence_macro_f1.png
reports/training_size_ablation/figures/length_mae.png
reports/training_size_ablation/figures/budget_coverage.png
reports/training_size_ablation/figures/recovery.png
```

汇总按训练规模保存 5 个 seed 的均值、样本标准差和 `1.96 * std / sqrt(5)` 的 95% 正态
近似置信区间。图形横轴使用对数刻度，以避免不等间距训练规模造成视觉误导。

## 6. 复现命令

项目依赖应使用已有虚拟环境，不要修改 `pyproject.toml`：

```bash
.venv/bin/python -m py_compile training_size_ablation.py
```

单个冒烟运行示例：

```bash
.venv/bin/python training_size_ablation.py \
  --sizes 50 \
  --seeds 42 \
  --skip-plots \
  --model-root /tmp/training_size_ablation_smoke_models \
  --report-root /tmp/training_size_ablation_smoke_reports
```

完整矩阵命令：

```bash
.venv/bin/python training_size_ablation.py \
  --model-root models/training_size_ablation \
  --report-root reports/training_size_ablation
```

如果不希望覆盖现有消融产物，应将两个 root 都改为新的临时目录。脚本默认配置读取：

```text
training_dataset.jsonl
data/splits/
data/gold/gold_main_ids.json
simple2.json
schema_costs.json
field_weights.json
```

## 7. 验证状态

- `.venv/bin/python -m py_compile training_size_ablation.py`：通过。
- `n=50, seed=42` 冒烟：通过。
- 完整矩阵：35/35 成功。
- 汇总行数：7。
- 每个规模的 seed 数：5。
- 模型目录：35 个；模型和元数据文件：210 个。
- 逐 run `metrics.json`：35 个，均存在且可解析。
- 预算 Jaccard 范围：0.7079 至 1.0。
- 四张 PNG 图和三个汇总 JSON/CSV 文件：均存在。
- `git diff --check`：通过。

## 8. 结果解释边界和后续工作

本轮结果只说明离线字段存在性、输出长度预测和预算字段选择随训练样本量的变化，不能直接
等价为 LLM 端到端抽取质量变化。特别是：

- 监督标签来自已有 Full Schema 结果，是 silver/model-assisted reference，不是独立人工 Gold；
- `estimated_cost` 是字段选择阶段的离线估计成本，不是实际 API Total Token 或货币成本；
- 当前汇总置信区间是 5 个 seed 上的正态近似 CI，不是 bootstrap CI；
- 尚未完成非 `desc` 字段 Value Judge、`desc` embedding score 和综合 Quality；
- 尚未基于训练量曲线选择代表规模执行真实 LLM 重跑；
- 不应据此声称跨博物馆零样本泛化。

正式实验记录已同步更新：

```text
doc/正式实验记录.md
```

后续若要开展真实 LLM 代表点实验，建议先固定代表规模、确认模型和 Prompt 配置，再单独写入
新的 experiments/reports 文件，不要把真实 LLM 结果混入本轮离线消融目录。
