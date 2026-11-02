# SchemaRAG k=10 实施与结果交接

## 1. 当前状态

`external-schemarag-k10` 已完成适配实现、单元测试、10 条真实 LLM 冒烟、冻结 500 条
`gold_main` 正式运行和统一 Presence 评估。正式结果与既有 v3 内部方法结果相互独立，
未覆盖任何历史文件。

本项目没有公开的 SchemaRAG 官方代码。本实现是依据论文
*SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information
Extraction*（ACL 2026 Industry Track）的算法描述进行的适配重实现，不应表述为官方实现
或对原论文实验的严格复现。

## 2. 实现范围

外部方法只替换统一 Runner 中的以下步骤：

```text
raw_data -> selected_fields
```

其余流程保持不变：`simple2.json` 的 18 字段 Schema、稳定 Prompt、`kg-assist-pro`、
temperature `0`、关闭思考、响应解析、字段白名单校验、最多两次重试、Token 统计和 Gold
评估。

实现文件：

| 文件 | 作用 |
| --- | --- |
| `external_schemarag_selector.py` | SchemaRAG 字段检索、embedding 调用、缓存和相似度排序 |
| `llm_runner.py` | 注册 `external-schemarag-k10`，写入 selector 元数据并保护断点续跑 |
| `tests/test_external_schemarag_selector.py` | Top-k 平局排序、成本计算和缓存复用测试 |
| `doc/external_schemarag_k10_reproduction.md` | 方法来源、适配边界和运行说明 |

## 3. 冻结方法配置

| 项目 | 配置 |
| --- | --- |
| 方法名 | `external-schemarag-k10` |
| 字段数 | `k=10`，无 required fields |
| Schema metadata | `simple2.json` 的英文字段名与完整中文字段定义 |
| 字段名 embedding | `字段名：{field}` |
| 字段定义 embedding | `字段定义：{definition}` |
| row 向量 | `normalize(0.5 * name_embedding + 0.5 * definition_embedding)` |
| query | 确定性 JSON 序列化完整 `raw_data`，加固定中文检索指令 |
| 排序 | 归一化向量点积，即 cosine similarity；降序；平局按 `simple2.json` 字段顺序 |
| embedding endpoint | `.env` 中的 OpenAI 兼容 `EMBEDDING_*` 配置 |
| embedding 服务名 | `text-embedding`，服务端提供 `Qwen3-Embedding-0.6B` |
| 禁用功能 | segmentation、example retrieval、ANN、训练、`field_weights.json` |
| 选择成本 | 仅记录字段名、字段定义和预测输出 Token；不参与相似度排序 |

固定查询指令：

```text
给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。
```

## 4. 结果可复核性

row embedding 缓存：

```text
experiments/schemarag_row_embeddings.json
```

缓存含 18 个字段、1024 维向量，配置哈希为 `ae8105a4c1507bfc`。该哈希覆盖 Schema、
embedding 模型名、字段名/定义权重和查询指令；相同哈希也写入每条结果。Runner 对
`external-schemarag-*` 的断点续跑额外校验此哈希，因此模型、Schema、权重或指令改变后
不会错误跳过旧结果。

正式运行命令：

```bash
LLM_MODEL=kg-assist-pro LLM_TEMPERATURE=0.0 \
uv run python llm_runner.py \
  --methods external-schemarag-k10 \
  --ids-file data/gold/gold_main_ids.json \
  --limit 500 --max-concurrency 20 --queue-size 40 --max-retries 2 \
  --prompt 1_stable.prompt \
  --output experiments/external_schemarag_k10_gold_main_v3_pro.jsonl
```

评估命令：

```bash
uv run python evaluate_llm_experiments.py \
  --input experiments/external_schemarag_k10_gold_main_v3_pro.jsonl \
  --gold data/gold/agent_tasks/gold_main.jsonl \
  --require-methods external-schemarag-k10 \
  --output reports/external_schemarag_k10_gold_main_v3_pro_metrics.json
```

## 5. 正式 500 条结果

正式产物：

| 产物 | 路径 |
| --- | --- |
| 原始运行结果 | `experiments/external_schemarag_k10_gold_main_v3_pro.jsonl` |
| Presence 指标 | `reports/external_schemarag_k10_gold_main_v3_pro_metrics.json` |
| 完整实验记录 | `doc/正式实验记录.md` 第 13 节 |

运行完整性：

| 项目 | 结果 |
| --- | ---: |
| 冻结 ID 数 | 500 |
| 唯一 `artifact_id` | 500 |
| 最终成功 | 500 / 500 |
| 最终失败 | 0 |
| 重试任务数 | 0 |
| 总重试次数 | 0 |
| 每条选择字段数 | 10 |
| 每条记录的相似度数 | 18 |

Presence 与运行效率：

| 指标 | 结果 |
| --- | ---: |
| Presence Macro-F1 | 0.7069 |
| Presence Micro-F1 | 0.7022 |
| 平均 Gold 字段覆盖 | 0.8078 |
| 平均 Exact Value Recall | 0.4796 |
| 平均 Input Token | 327.51 |
| 平均 Output Token | 105.41 |
| 平均 Total Token | 432.92 |
| 平均估计字段成本 | 308.11 |
| 平均 LLM 延迟 | 2197.83 ms |
| 平均排队等待 | 4254.08 ms |
| 实际货币成本 | 0 |

`actual_cost=0` 的原因是自部署服务的输入、输出单价环境变量均为零，并不表示没有 Token
消耗；原始结果保存了 input、output 和 total Token，后续可配置内部单价后离线换算。

## 6. 与现有 v3 Pro 结果的初步对照

下列仅是 Presence 与平均总 Token 的描述性对照；所有方法均需使用相同 Quality 口径后，
才能形成主结论。

| 方法 | Presence Micro-F1 | 平均 Total Token |
| --- | ---: | ---: |
| `external-schemarag-k10` | 0.7022 | 432.92 |
| `full` | 0.7410 | 539.83 |
| `budget-200` | 0.7780 | 381.49 |
| `budget-400` | 0.7638 | 486.32 |
| `lambda-0.003` | 0.7816 | 441.28 |

SchemaRAG 相比 `full` 平均少用约 106.91 Total Token（约 19.8%），但当前 Presence Micro-F1
低于 `full`、`budget-200`、`budget-400` 和 `lambda-0.003`。平均 Gold 字段覆盖为 0.8078，
说明 Top-10 检索会漏掉部分 Gold 非空字段。这一结果支持“能降低输入规模”的描述，但不支持
“优于内部动态选择方法”或“总体抽取质量更高”的结论。

## 7. 指标解释

Presence 评估将 Gold 中非 `null` 的字段视为正例，衡量模型输出的字段是否非空；它不充分
衡量具体字段值是否正确。

- Macro-F1：先对每条记录计算 Precision、Recall 和 F1，再对 500 条记录取平均；每条记录
  权重相同。
- Micro-F1：先累计全部记录的 TP、FP、FN，再计算总体 Precision、Recall 和 F1；字段较多的
  记录对结果贡献更大。

本方法的 Macro-F1 与 Micro-F1 接近，分别为 0.7069 和 0.7022，未显示明显的记录规模失衡。

## 8. 结论边界与后续建议

可以报告：

- SchemaRAG k=10 的适配实现已稳定完成，500 条任务全部成功且无重试。
- 在当前 endpoint、模型、Prompt 和冻结样本上，方法实现了约 19.8% 的平均总 Token 降低
  （相对 v3 Pro `full`）。
- 每条结果均保留字段选择、完整相似度、配置哈希、原始输入、Token、重试和状态，可复核。

不能报告：

- 不能称为官方代码复现或严格复现实验。
- 不能引用原论文的质量数字作为本项目结果。
- 不能只依据 Presence 指标宣称总抽取质量更高或优于内部方法。
- 当前 Gold 是 model-assisted/silver reference，不是独立人工 Gold。

主线建议：

1. 将本方法作为外部基线纳入后续统一对照，但只与共同成功记录做配对分析。
2. 对所有拟比较方法补齐 LLM Value Judge 和 `desc` Embedding Score。
3. 使用已冻结的综合质量公式，进行 Quality-Token 配对比较后再确定主结果表与论文结论。
4. 如需货币成本，在不重跑结果的前提下，基于记录的 input/output Token 和明确的内部单价离线换算。
