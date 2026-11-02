# SchemaRAG k=15 实施与结果交接

## 当前状态

`external-schemarag-k15` 已完成 Runner 接入、单元测试、10 条真实 smoke、冻结 500 条正式运行和
统一 Presence 评估。结果写入独立文件，没有覆盖 k=10 或既有 v3 结果。

本实现依据 *SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information
Extraction* 的论文描述适配完成。项目没有公开的官方 SchemaRAG 代码，因此不能表述为官方实现或
原论文实验的严格复现。

## 实现与固定条件

Runner 中仅替换 `raw_data -> selected_fields`。k=15 复用已有 18 个字段的 row embedding 缓存，
但使用独立方法名 `external-schemarag-k15` 和独立结果文件。

| 项目 | 配置 |
| --- | --- |
| Schema | `simple2.json`，18 个字段 |
| 选择数量 | `k=15`，无 required fields |
| Schema metadata | 字段名 + 完整中文字段定义 |
| row 向量 | `normalize(0.5 * name_embedding + 0.5 * definition_embedding)` |
| query | 完整 `raw_data` 的确定性 JSON + 固定中文指令 |
| 排序 | cosine 降序，平局按 `simple2.json` 顺序 |
| embedding | OpenAI 兼容 endpoint，模型名 `text-embedding`，服务端为 `Qwen3-Embedding-0.6B` |
| LLM | `kg-assist-pro`，temperature `0`，`enable_thinking=false` |
| Prompt | `1_stable.prompt` |
| 重试 | 最多 2 次重试，保留状态、错误和 Token 记录 |

缓存路径为 `experiments/schemarag_row_embeddings.json`，含 18 个 1024 维向量，配置哈希为
`ae8105a4c1507bfc`。哈希覆盖 Schema、模型名、权重和 query 指令，不包含 k，因为 k 不改变 row
向量。Runner 仍按方法名和 selector 哈希隔离断点续跑。

## 正式产物与完整性

| 产物 | 路径 |
| --- | --- |
| 原始运行结果 | `experiments/external_schemarag_k15_gold_main_v3_pro.jsonl` |
| Presence 指标 | `reports/external_schemarag_k15_gold_main_v3_pro_metrics.json` |
| 复现说明 | `doc/external_schemarag_k15_reproduction.md` |
| 正式实验记录 | `doc/正式实验记录.md` 第 14 节 |

500 条冻结 ID 全部完成：500 行、500 个唯一 `artifact_id`、500/500 成功、0 最终失败、0 次重试。
每条记录均选择 15 个合法字段并保存 18 个相似度。总 Input Token 为 184,298，Output Token 为
67,479，Total Token 为 251,777；平均分别为 368.596、134.958、503.554。平均估计字段成本为
448.796，实际货币成本为 0。

同一正式命令再次运行时提交数为 0，说明结果和 selector 哈希的断点续跑校验生效。

## 评估结果

评估使用 500 条 `data/gold/agent_tasks/gold_main.jsonl` reference，当前完成的是 Presence、字段覆盖、
Exact Value Recall 和运行效率指标；Gold 属于 model-assisted/silver reference，不是独立人工 Gold。

| 指标 | k=15 |
| --- | ---: |
| Presence Macro-F1 | 0.7324 |
| Presence Micro-F1 | 0.7224 |
| 平均 Gold 字段覆盖 | 0.9888 |
| 平均 Exact Value Recall | 0.5524 |
| 平均 Input Token | 368.60 |
| 平均 Output Token | 134.96 |
| 平均 Total Token | 503.55 |
| 平均 LLM 延迟 | 2625.10 ms |
| 平均排队等待 | 5098.60 ms |
| 平均重试次数 | 0 |

与已完成的 k=10 描述性对照如下：

| 方法 | Presence Macro-F1 | Presence Micro-F1 | Gold 覆盖 | Exact Value Recall | 平均 Total Token |
| --- | ---: | ---: | ---: | ---: | ---: |
| `external-schemarag-k10` | 0.7069 | 0.7022 | 0.8078 | 0.4796 | 432.92 |
| `external-schemarag-k15` | 0.7324 | 0.7224 | 0.9888 | 0.5524 | 503.55 |

k=15 相比 k=10 平均多 70.63 Total Token，Presence Micro-F1 提高 0.0203，Gold 字段覆盖提高
0.1811，Exact Value Recall 提高 0.0727。这些是当前 endpoint、模型、Prompt 和冻结样本上的
描述性差异，不能单独推导综合抽取质量或方法优越性。

## 结论边界

可以报告 k=15 的适配实现已稳定完成，500 条任务全部成功且无重试；相较 k=10，增加选择字段数
显著提高了 Gold 字段覆盖，并伴随 Token 和 Presence 指标上升。

不能报告其为官方代码复现，也不能使用原论文结果替代本项目结果。最终方法判断仍需补齐统一的
LLM Value Judge、`desc` Embedding Score 和冻结综合 Quality 评分，并在共同成功样本上比较。
