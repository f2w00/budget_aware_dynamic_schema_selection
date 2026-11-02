# 随稿复现材料归档（D1）

本目录冻结主实验 `main-comparison-pro` 的完整复现证据，回应审稿问题 32
（提供稳定文件与版本标识）。所有文件均离线生成，不触发 LLM/embedding。

## 内容

- `SHA256SUMS`：全部归档文件（raw 日志、meta、汇总、聚合产物、Gold/标注、各 seed 离线质量结果、分析脚本）的校验和。
- `manifest.json`：git 提交、逐次运行的 `workload_ids_sha256`、输入文件哈希、环境。
- `README.md`：本说明。

## 数据来源

- 主实验运行目录：`benchmarks/runs/main-comparison-pro/`（git 忽略），
  方法 budget-200、external-schemarag-k10、full，每次 2000 条记录、3 次重复、
  concurrency=16、queue_size=16。
- 聚合产物：`benchmarks/analysis/performance/` 及各离线分析目录
  （breakeven、c1_near_duplicate、c2_representativeness、c3_scan、initialization）。
- A6/A3 在线重跑：`benchmarks/runs/a6-a3-sweep/` 的 13 份逐条日志及
  `benchmarks/analysis/a6_a3/` 的预算曲线、基线表和图。
- A4 离线选择策略消融：`benchmarks/analysis/a4_selection_ablation/` 的固定测试集 Gold
  汇总、表格和图；Gold 真值为 `data/gold/agent_tasks/gold_main.jsonl`。
- A5 离线权重敏感性分析：`benchmarks/analysis/a5_weight_sensitivity/` 与
  `tex/v6-lncs/补充实验3.md`；比较人工、等权、训练频率、反频率压力和扰动权重。
- Gold 与标注：`data/gold/` + `gold_reference.prompt`。
- 各 seed 离线质量结果：`reports/`（quality_token_comparison 与训练/模型消融的逐 seed metrics）。
- 分析/绘图/归档脚本：`benchmarks/*.py`。

## 复现命令（来自各 run 的 meta.json command 字段）

```bash
# 从逐条日志重算性能指标（B1/B3，问题 9/11/12/13/26/31）
.venv/bin/python benchmarks/recompute_from_raw.py

# 离线汇总统计表（原有）
.venv/bin/python benchmarks/analyze_performance_results.py

# 盈亏平衡点分析（B5，问题 17）
.venv/bin/python benchmarks/analyze_breakeven.py

# 近重复检测与分组隔离可行性（C1，问题 19/20）
.venv/bin/python benchmarks/analyze_c1_near_duplicate.py

# 数据代表性与分层分析（C2，问题 20/21）
.venv/bin/python benchmarks/analyze_c2_representativeness.py

# 并发/队列解耦扫描（C3，问题 15）
.venv/bin/python benchmarks/analyze_c3_scan.py

# 预算响应与匹配基线（A6/A3，问题 5/7/8/18/25/30/34）
.venv/bin/python benchmarks/analyze_a6_a3.py

# 固定测试集 Gold 的选择策略消融（A4，问题 6/7/25/34）
.venv/bin/python benchmarks/analyze_a4_selection_ablation.py
A4_BUDGET=400 .venv/bin/python benchmarks/analyze_a4_selection_ablation.py

# 字段权重敏感性分析（A5，问题 6/25/34）
.venv/bin/python benchmarks/analyze_a5_weight_sensitivity.py

# 重新生成本归档清单
.venv/bin/python benchmarks/archive_artifacts.py
```

## 验证归档完整性

```bash
cd 仓库根目录
sha256sum -c benchmarks/archive/SHA256SUMS
```
