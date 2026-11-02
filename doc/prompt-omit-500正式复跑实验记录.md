# prompt-omit 500 条正式复跑实验记录

## 目的

本实验独立重做 `prompt-omit` 在 `gold_main` 500 条冻结记录上的真实 LLM 调用，
验证历史批次中 JSON 解析失败是否可复现，并在不混入历史结果的前提下给出完整的
离线评估结果。

## 固定输入与配置

- Gold ID：`data/gold/gold_main_ids.json`，500 条唯一记录；
- 冻结输入：`data/gold/prompt_omit_500_snapshot.jsonl`；
- 输入快照 SHA256：
  `a7f2cfe73b8903d6962ee9036091f748e0edf97b70004048607efcddab6100cc`；
- 参考：`data/gold/agent_tasks/gold_main.jsonl`，500 条 model-assisted silver reference；
- 方法：`prompt-omit`；
- 基础 Prompt：`1_stable.prompt`；
- omit 模板：`1_stable_omit.prompt`，hash 前 16 位为 `6a2682d1629c9b38`；
- Schema：`simple2.json`，hash 前 16 位为 `8c9969277e238283`；
- 服务：`kg-assist-pro` / `Qwen3-8B-AWQ`；
- temperature：0；thinking：关闭；并发：20；队列：40；最大重试：2；
- 不加载 M1，`selector_model_dir=null`，`prediction_ms=0`；
- 每条任务保留完整 18 个目标字段，允许模型省略字段后补为 `null`。

## 预检

- 输入快照由数据库按 Gold ID 原始顺序生成，500 条记录与 ID 集合完全一致；
- 500 条零网络 dry-run 通过：18 字段、`prompt_omit` 变体、冻结 raw 一致、无 M1；
- 2 条独立 live smoke 通过：2/2 成功、0 重试；
- smoke 输出位于 `experiments/prompt_omit_500_formal/smoke/`，未混入正式目录。

## 原始正式调用

正式调用输出：

```text
experiments/prompt_omit_500_formal/formal/prompt-omit.jsonl
```

500 条请求均已完成并落盘。477 条在原解析器下直接成功；23 条最终为
`JSONDecodeError`，每条已按最大重试次数请求三次，因此原始行的 `retry_count=2`。
23 条失败均是同一无歧义格式缺口：模型在 JSON 对象末尾将空字段写作
`"related_person":}`，即键后缺失值而紧邻对象结束符。它们不是服务失败、字段选择
失败或额外字段问题。原始响应、Token、延迟与三次失败尝试均保留在原始 JSONL 中。

## 解析修复与最终结果

为处理 `prompt-omit` 明确允许字段缺失的情况，`llm_runner.parse_json_response` 新增
仅对 `missing_fields_expected=true` 生效的规则：将键后的无歧义缺失值
`:\s*(?=[,}])` 规范化为 `: null`，并记录 `repaired_missing_json_values`。

随后使用 `benchmarks/repair_prompt_omit_500.py` 只对 23 条 `JSONDecodeError` 原始响应
离线重放。脚本拒绝修复其他错误类型、非该模式的 JSON 失败或含额外字段的结果；不发起
LLM 或 Embedding 调用。修复输出：

```text
experiments/prompt_omit_500_formal/formal-repaired/prompt-omit.jsonl
```

最终审计：

| 指标 | 结果 |
| --- | ---: |
| 记录数 / 唯一 ID | 500 / 500 |
| success | 500 |
| 原始直接成功 | 477 |
| 离线修复记录 | 23 |
| error | 0 |
| 选中字段数 | 全部 18 |
| M1 prediction 总时间 | 0 ms |
| 额外字段记录 | 0 |
| Prompt / Schema hash | 各 1 个，均与固定配置一致 |
| 冻结 raw 匹配 | 500/500 |

## 离线质量评估

评估命令：

```bash
uv run python evaluate_llm_experiments.py \
  --input experiments/prompt_omit_500_formal/formal-repaired/prompt-omit.jsonl \
  --gold data/gold/agent_tasks/gold_main.jsonl \
  --output reports/prompt_omit_500_formal_repaired_metrics.json
```

结果：

| 指标 | 值 |
| --- | ---: |
| Presence Macro-F1 | 0.7492 |
| Presence Micro-F1 | 0.7402 |
| Exact Value Recall | 0.6026 |
| selected gold coverage | 1.0000 |
| 平均输入 Token | 427.866 |
| 平均输出 Token | 102.766 |
| 平均总 Token | 530.632 |
| 平均请求延迟 | 1514.790 ms |
| 平均队列等待 | 3315.062 ms |
| 平均重试次数 | 0.092 |

`invalid_rows=0`，500 条均参与评估。指标只反映与 model-assisted silver reference 的一致性，
不构成独立人工事实正确性证明。

## 与历史批次的关系

历史 `experiments/gold_main_v3_stable_pro_results.jsonl` 的 `prompt-omit` 子集使用相同
stable omit Prompt 和 Schema，结果为 476/500 直接成功、24 条 `JSONDecodeError`，失败
同样集中在重试耗尽的 JSON 解析。历史结果不与本次结果合并。本次复跑以 23 条可审计
解析修复后实现 500/500 完整配对，验证该失败属于重复的输出格式问题，而非偶发服务故障。

## 产物与验证

```text
benchmarks/freeze_prompt_omit_500.py
benchmarks/run_prompt_omit_500.py
benchmarks/repair_prompt_omit_500.py
data/gold/prompt_omit_500_snapshot.jsonl
data/gold/prompt_omit_500_snapshot.manifest.json
experiments/prompt_omit_500_formal/dry-run/
experiments/prompt_omit_500_formal/smoke/
experiments/prompt_omit_500_formal/formal/
experiments/prompt_omit_500_formal/formal-repaired/
reports/prompt_omit_500_formal_repaired_metrics.json
```

已执行 `uv run python -m py_compile`（Runner、冻结、编排、修复和评估脚本）及
`git diff --check`，均通过。

## 边界

- 该实验只验证 `prompt-omit` 在这 500 条冻结 silver reference 上的输出和指标；
- 解析修复只处理无歧义的键后缺失值，不是通用 JSON 修复器；
- 原始请求阶段仍有 23 条模型输出不是合法 JSON，应报告为格式稳健性风险；
- 不能将本结果解释为独立人工 Gold 质量、未见泛化或货币成本结论。
