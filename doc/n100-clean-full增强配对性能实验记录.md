# n=100 clean-full 增强配对性能实验记录

## 1. 实验目的

旧 n=100 配对性能实验中，`full` 虽然保留全部 18 个字段，但 Runner 仍会加载 M1，
并对每条记录执行字段预测以生成统一遥测。这部分本地预测进入在线流水线，因此旧
`full` 不是干净基线。本实验重新测量不加载、不调用 M1 的 clean full，并通过每个
selector seed 各执行一次 AB 和 BA，降低方法顺序与服务时段的混杂。

本实验只验证同域部署负载上的系统时间、吞吐、Token 和遥测，不验证未见泛化或语义
质量保持。

## 2. 实现修正

- `llm_runner.py` 增加 `--records-file`，正式实验只读取冻结 JSONL，不实时访问数据库。
- `full`、`prompt-*` 和 `fixed-*` 不再加载或调用字段预测模型。
- clean full 固定保留 18 个字段，`prediction_ms=0`，估计字段成本和
  `selector_model_dir` 均为空。
- `budget-*`、`lambda` 和 SchemaRAG 仍按原逻辑使用预测。
- 新增 `record_end_to_end_ms`，从 Producer 构造记录任务开始，到结果写出前结束。
- 原 `latency_ms` 语义保持不变，只覆盖请求、解析和重试，不含队列等待与 Producer
  规划。

## 3. 冻结负载

冻结产物：

```text
data/performance/n100_clean_full_2000_snapshot.jsonl
data/performance/n100_clean_full_2000_snapshot.manifest.json
data/performance/n100_clean_full_orders/
```

- 记录数：2,000，ID 唯一；
- split 组成：train 1,760、validation 240、test 0；
- 源 ID SHA256：
  `831c1b9a7fcd84a09c17542e69114cc7f2a650b14cecc7e1c79a00e2058e274e`；
- snapshot SHA256：
  `942486f2a56f7ce8cc8a023706ad9d3fd73f1c664fd95eeba21ebcd1a0cfebbc`；
- raw-data SHA256：
  `a284e8654d5f501fa2efa8518db7c63a1272c5fd0603dfb847d60690e1d023d7`；
- Schema hash：`8c9969277e238283`；
- Prompt hash：`9062d195b48060a4`。

逐 ID 将新 snapshot 与旧正式 cohort 的 raw JSON 规范化比较，2,000/2,000 完全相等。
六个 pair 使用六个不同的冻结随机排列；同一 pair 的两种方法共享相同排列。

## 4. 增强配对协议

固定服务条件：`kg-assist-pro` / `Qwen3-8B-AWQ`、temperature=0、thinking 关闭、
stable Prompt、`c=16,q=16`、最多 2 次重试。

| pair | selector seed | repeat | 顺序 |
| ---: | ---: | ---: | --- |
| 1 | 42 | 1 | full -> budget-200 |
| 2 | 43 | 1 | budget-200 -> full |
| 3 | 44 | 1 | full -> budget-200 |
| 4 | 44 | 2 | budget-200 -> full |
| 5 | 43 | 2 | full -> budget-200 |
| 6 | 42 | 2 | budget-200 -> full |

每个 run 为 2,000 条。共 12 run、24,000 条正式逻辑 LLM 请求。pair 内连续执行；若
任一侧不完整，编排器归档整个 pair 后重跑，不允许只恢复半个 pair。

正式运行入口：

```bash
uv run python benchmarks/run_n100_clean_full_enhanced.py --execute --resume
```

## 5. 运行有效性

- 12 个正式 run 均通过 validation；
- 每个 run 均为 2,000 条成功、0 条错误、0 次重试；
- 6 个 clean-full run 的 `prediction_ms` 总计均为 0，模型目录均为空；
- 6 个 `budget-200` run 均绑定正确的 seed 42/43/44 n=100 M1；
- JSONL 按异步请求完成顺序写出，不保证与输入顺序相同；冻结顺序由 order 文件、运行
  元数据和完整 ID 集合共同审计；
- 正式运行未发生中断，不存在需要排除的半 pair；
- dry-run 12 个 run 和 live smoke 4 条记录不纳入正式统计。

正式产物：

```text
benchmarks/runs/n100-2000-clean-full-paired/
benchmarks/analysis/performance_n100_2000_clean_full_enhanced/
```

## 6. 方法均值

统计单位为 6 个完整、相邻的方法 pair；95% 区间使用 df=5 的 t 临界值 2.571。

| 方法 | pipeline 时间 | 95% CI | 有效吞吐 | 总 Token | E2E P95 | E2E P99 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clean full | 259.96 s | [255.31, 264.61] | 7.70 rec/s | 1,783,445 | 4848.86 ms | 5780.18 ms |
| n=100 budget-200 | 112.94 s | [103.80, 122.09] | 17.80 rec/s | 1,099,450 | 2147.03 ms | 2393.16 ms |

其他遥测均值：

| 方法 | 输入 Token | 输出 Token | 请求 P95 | 请求 P99 | 规划 P50 |
| --- | ---: | ---: | ---: | ---: | ---: |
| clean full | 1,399,176 | 384,269 | 2464.21 ms | 3532.57 ms | 0.14 ms |
| n=100 budget-200 | 965,037 | 134,413 | 1121.00 ms | 1324.63 ms | 9.45 ms |

`budget-200` 增加约 9 ms/条的本地字段规划，但减少的远程 LLM 负载远大于这项开销。

## 7. 配对效应

`budget-200` 相对同 pair 的 clean full：

| 指标 | 平均绝对差 | 平均相对差 | 相对差 95% CI |
| --- | ---: | ---: | ---: |
| pipeline 时间节省 | 147.02 s | 56.55% | [53.15%, 59.96%] |
| 有效吞吐增益 | 10.11 rec/s | 131.30% | [112.24%, 150.36%] |
| 总 Token 节省 | 683,995 | 38.35% | [38.20%, 38.50%] |
| 真实记录 E2E P95 节省 | 2701.82 ms | 55.76% | [52.54%, 58.97%] |
| 真实记录 E2E P99 节省 | 3387.02 ms | 58.63% | [52.44%, 64.83%] |
| 请求 P95 节省 | 1343.21 ms | 54.55% | [50.71%, 58.38%] |
| 请求 P99 节省 | 2207.94 ms | 62.52% | [57.09%, 67.95%] |

六个 pair 的 Token 节省率均约 38.2%--38.4%，方向和幅度稳定。pipeline 和吞吐更受
共享远程服务时段影响，但六个 pair 的方向均一致。

## 8. 顺序和 selector seed 描述性分析

| 分组 | pipeline 节省 | 吞吐增益 | Token 节省 |
| --- | ---: | ---: | ---: |
| full-first，n=3 | 55.42% | 124.57% | 38.35% |
| budget-first，n=3 | 57.69% | 138.03% | 38.35% |

| selector seed | pipeline 节省 | 吞吐增益 | Token 节省 |
| ---: | ---: | ---: | ---: |
| 42 | 57.51% | 135.38% | 38.44% |
| 43 | 53.36% | 114.39% | 38.45% |
| 44 | 58.80% | 144.14% | 38.17% |

AB 和 BA 两组均保持同方向；Token 节省几乎不受顺序影响。运行级时间差异不能完全归因
于 seed，因为 selector seed、运行时段和共享远程服务状态仍可能共同作用。

## 9. 初始化成本敏感性

基于本 cohort 方法均值：

- clean full 在线时间：259.96 s；
- n=100 `budget-200` 在线时间：112.94 s；
- 每条记录平均在线时间差：0.07351 s；
- 已部署冷加载 0.17826 s：break-even 约 2.43 条；
- 本地训练加加载 0.91539 s：break-even 约 12.45 条；
- 完整 FullSchema 银标生成成本未按本协议现场测量，不生成估计数字。

在线指标不含一次性训练或加载成本。clean full 不依赖 M1，因此若讨论生命周期成本，
加载和训练成本应归入 `budget-200`；完整生命周期还必须单列启动样本生成、银标整理、
环境和部署成本。

## 10. 与旧 cohort 的关系

旧目录 `benchmarks/runs/n100-2000-paired/` 保留审计，不覆盖也不删除。旧实验的 `full`
执行了 M1 预测，不能作为最终 clean-full 基线，也不能与本 cohort 混合平均。旧实验仍可
用于说明问题发现过程、Runner 遥测和 selector 稳定性；正式系统性能结论应使用本增强
cohort。

## 11. 结论边界

- 负载来自 train+validation，是同域部署性能负载，不是未见泛化测试。
- 6 个 pair 支持运行级描述和配对区间，但不足以作强显著性推断。
- 共享远程服务的精确模型制品、GPU 状态和外部负载不能完全冻结。
- 本实验没有独立人工 Gold，不能据此证明动态 Schema 保持语义质量。
- `budget-200` 实际包含 M1 预测、人工字段权重、强制字段和 0-1 背包，效果不能全部
  归因于 M1。
- 完整银标生成和部署成本未现场测量，不填造全生命周期数字。

## 12. 离线分析复现

```bash
uv run python benchmarks/analyze_n100_clean_full_enhanced.py
```

输出：

```text
benchmarks/analysis/performance_n100_2000_clean_full_enhanced/paired_runs.csv
benchmarks/analysis/performance_n100_2000_clean_full_enhanced/performance_statistics.json
```

离线分析不调用 LLM、Embedding 或数据库。
