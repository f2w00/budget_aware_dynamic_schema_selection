# n=100 clean SchemaRAG k=8 性能实验记录

## 1. 实验目的

在与 clean FullSchema 和 n=100 `budget-200` 相同的冻结负载、服务配置和在线流水线
条件下，测试字段数与 n=100 `budget-200` 接近的 SchemaRAG 外部重实现。n=100
`budget-200` 在此前 12,000 条正式输出中平均选择 7.852 个字段，中位数为 8，因而
本实验将 SchemaRAG 的 Top-k 设为 8。

本实验是独立的 SchemaRAG 性能 cohort，不与此前 clean full/budget 的 24,000 条
配对运行混合平均。由于运行时段不同，跨 cohort 比较只作描述性参考。

## 2. 固定条件

- 冻结负载：`data/performance/n100_clean_full_2000_snapshot.jsonl`；
- 每次运行：2,000 条；
- 六个预冻结顺序：`data/performance/n100_clean_full_orders/pair_1...pair_6...json`；
- 快照 SHA256：
  `942486f2a56f7ce8cc8a023706ad9d3fd73f1c664fd95eeba21ebcd1a0cfebbc`；
- raw_data SHA256：
  `a284e8654d5f501fa2efa8518db7c63a1272c5fd0603dfb847d60690e1d023d7`；
- Schema hash：`8c9969277e238283`；
- Prompt hash：`9062d195b48060a4`；
- LLM 服务：`kg-assist-pro` / `Qwen3-8B-AWQ`；
- Embedding 模型：由 `EMBEDDING_MODEL` 环境变量确定；
- temperature：0；
- thinking：关闭；
- 最大重试：2；
- Producer/Consumer：`c=16,q=16`；
- 方法：`external-schemarag-k8`；
- 正式请求：6 x 2,000 = 12,000 条。

## 3. SchemaRAG 实现口径

SchemaRAG 是基于方法描述的外部独立重实现，不是官方代码复现。当前实现：

1. 分别对字段名和字段定义进行 Embedding；
2. 按 0.5/0.5 加权并进行 L2 归一化；
3. 将固定查询指令和原始记录序列化内容拼接为 query；
4. 对每条记录执行 query Embedding；
5. 与 18 个字段向量计算点积，选择 Top-8 字段；
6. 使用 Top-8 字段生成动态 Schema 并调用 LLM。

字段向量可以缓存；每条记录的 query Embedding 纳入在线 Producer 和端到端时间，
Schema 向量构建/缓存时间单独记录。当前不包含字段分段、示例检索或二阶段重排序。

本实验已将 SchemaRAG 与 M1 完全解耦：不加载 M1、不执行字段预测，
`prediction_ms=0`、`selector_model_dir=null`，`estimated_field_cost=null`。因此结果
反映 Embedding Top-8 选择本身，不混入 M1 的预测计算或预测输出 Token 估计。

## 4. 运行矩阵与产物

六次正式运行对应既有六个冻结顺序：

| Pair | selector seed | repeat | 顺序文件 | 状态 |
| --- | ---: | ---: | --- | --- |
| 1 | 42 | 1 | pair 1 | 完成 |
| 2 | 43 | 1 | pair 2 | 完成 |
| 3 | 44 | 1 | pair 3 | 完成 |
| 4 | 44 | 2 | pair 4 | 完成 |
| 5 | 43 | 2 | pair 5 | 完成 |
| 6 | 42 | 2 | pair 6 | 完成 |

正式原始结果：

```text
benchmarks/runs/n100-2000-clean-full-schemarag-k8/
```

离线分析结果：

```text
benchmarks/analysis/performance_n100_2000_clean_full_schemarag_k8/
```

运行器：`benchmarks/run_n100_clean_full_schemarag_k8.py`；
分析器：`benchmarks/analyze_n100_clean_full_schemarag.py`。

首次严格额外键校验的六次运行未纳入正式统计，完整保存在：

```text
benchmarks/runs/n100-2000-clean-full-schemarag-k8-invalid-extra-fields/
```

该批次的主要问题是模型偶尔返回 `artifact_id`、`年代` 等未选字段，严格校验将其
误判为整条请求失败。修订后的正式协议使用显式 `--allow-extra-fields`：保留
`unexpected_fields` 审计信息，仅过滤未选字段，再按 Top-8 目标字段归一化结果。
该规则只在本 SchemaRAG k=8 cohort 中启用，其他实验默认仍保持严格校验。

## 5. 正式结果

以下统计以六个运行作为统计单位；置信区间使用 df=5 的 t 临界值 2.571，主要用于
描述当前服务时段和运行波动，不作强因果推断。

| 指标 | 均值 | SD | 95% CI |
| --- | ---: | ---: | ---: |
| 流水线时间 | 405.020 s | 86.331 s | 314.407--495.633 s |
| 有效吞吐 | 5.118 records/s | 1.028 | 4.040--6.197 |
| 总 Token | 1,085,024.8 | 37.0 | 1,084,986.0--1,085,063.7 |
| 输入 Token | 894,274.0 | 0.0 | 894,274.0--894,274.0 |
| 输出 Token | 190,750.8 | 37.0 | 190,712.0--190,789.7 |
| record E2E P50 | 4,312.5 ms | 201.7 ms | 4,100.8--4,524.2 ms |
| record E2E P95 | 11,557.1 ms | 5,096.0 ms | 6,208.3--16,905.9 ms |
| record E2E P99 | 45,955.1 ms | 16,292.5 ms | 28,854.4--63,055.8 ms |
| request P95 | 5,891.2 ms | 2,738.8 ms | 3,016.5--8,765.8 ms |
| request P99 | 30,038.1 ms | 20,685.8 ms | 8,326.2--51,750.1 ms |
| query Embedding 总时间 | 289.250 s | 92.032 s | 192.653--385.847 s |
| Schema Embedding 总时间 | 17.916 s | 0.740 s | 17.140--18.693 s |

六次运行均为：

- 2,000/2,000 条提交；
- 2,000/2,000 条成功；
- 0 error；
- 0 retry；
- 每条选择 8 个字段；
- M1 prediction 总时间为 0；
- query Embedding 遥测存在。

## 6. 额外键审计

正式 12,000 条成功结果中，有 558 条返回了目标字段之外的键，占 4.65%。额外键
出现次数为：

| 额外键 | 出现次数 |
| --- | ---: |
| `artifact_id` | 558 |
| `年代` | 534 |
| `类型` | 6 |
| `详细描述` | 6 |

这些额外键没有进入最终归一化结果，也没有改变选中字段数或目标字段的验证。该现象
说明当前 Prompt 下仍存在少量 Schema 边界遵循问题；虽然不影响本次请求成功率，
但在质量分析中应继续保留 `unexpected_fields` 作为协议遵循审计指标。

## 7. 与既有结果的关系

此前旧性能实验中的 SchemaRAG k=10 为：399.68 s、5.03 records/s、1,230,411
总 Token；该结果来自旧实现口径，且旧 full 仍含 M1 规划开销，不能直接与本次
clean full 形成严格三方法配对。

本次 k=8 的字段数与 n=100 `budget-200` 的平均选择规模相近，但 SchemaRAG 的
选择机制不同：

- `budget-200`：M1 字段预测 + 人工字段权重 + 必选字段 + 0-1 背包；
- SchemaRAG k=8：字段/记录 Embedding 相似度 Top-8。

因此 k=8 结果可用于回答“在近似字段数下，Embedding 检索是否仍有显著在线开销”，
但不能把两者差异全部归因于字段数量，也不能将本次六次运行写成与 clean
full/budget 同时段的严格配对比较。

## 8. 结果边界

- 负载来自 train+validation，适用于同域部署压力和流水线性能，不是未见泛化测试；
- 本 cohort 没有独立人工 Gold，不能用它证明抽取语义质量保持；
- 六次运行是运行级统计单位，不应将 2,000 条记录当成 2,000 个独立重复实验；
- 远程服务端 GPU、模型制品和外部负载没有完全冻结；
- query Embedding 是当前 SchemaRAG 重实现的在线额外成本，结论依赖当前服务和并发；
- 额外键被过滤并保留审计，不能当作“模型从未返回未选字段”；
- 不报告完整银标生成、环境部署和货币 API 成本，不能推出完整生命周期成本下降。

## 9. 验证状态

- SchemaRAG k=8 M1 解耦代码已通过 `uv run python -m py_compile`；
- 新增运行器和分析器已通过 `uv run python -m py_compile`；
- 六次小规模 dry-run 完成；
- 2 条 live smoke 完成，验证 8 字段、M1 prediction=0、Embedding 和 E2E 遥测；
- 修正版六次正式运行全部完成；
- 分析器重跑输出 6 runs；
- 12,000/12,000 正式请求成功；
- `git diff --check` 通过；
- 未修改论文正文文件。
