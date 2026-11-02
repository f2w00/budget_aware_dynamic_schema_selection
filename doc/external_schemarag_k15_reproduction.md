# SchemaRAG k=15 复现说明

## 定位

`external-schemarag-k15` 是基于 SchemaRAG 论文算法描述的适配重实现。项目未使用官方代码，
因为当前没有公开的 SchemaRAG 官方实现；因此不能称为官方实现或原论文实验的严格复现。

本方法仅替换统一 Runner 的：

```text
raw_data -> selected_fields
```

其余流程冻结为 `simple2.json` 的 18 个字段、`1_stable.prompt`、`kg-assist-pro`、temperature 0、
关闭 thinking、统一响应解析、重试和 Token 记录。k=10 与 k=15 使用不同方法名和结果文件，互不覆盖。

## 冻结配置

| 项目 | 配置 |
| --- | --- |
| 方法名 | `external-schemarag-k15` |
| 选择字段数 | `k=15`，无 required fields |
| Schema metadata | `simple2.json` 字段名与完整中文字段定义 |
| row 向量 | `normalize(0.5 * name_embedding + 0.5 * definition_embedding)` |
| query | 完整 `raw_data` 的确定性 JSON 序列化，加固定中文检索指令 |
| 排序 | cosine similarity 降序，平局按 `simple2.json` 顺序 |
| embedding | OpenAI 兼容 endpoint，`.env` 中的 `EMBEDDING_*` |
| 禁用功能 | segmentation、example retrieval、ANN、训练、`field_weights.json` |
| 选择成本 | 仅记录字段名、定义和预测输出 Token，不参与检索排序 |

k=15 只改变选择数量，共享 `experiments/schemarag_row_embeddings.json` 中的 18 个字段 row 向量缓存。
缓存哈希不包含 k，因为 k 不改变 row 向量；结果行仍记录方法名和 selector 配置哈希。

## 正式运行

```bash
LLM_MODEL=kg-assist-pro LLM_TEMPERATURE=0.0 \
uv run python llm_runner.py \
  --methods external-schemarag-k15 \
  --ids-file data/gold/gold_main_ids.json \
  --limit 500 --max-concurrency 20 --queue-size 40 --max-retries 2 \
  --prompt 1_stable.prompt \
  --output experiments/external_schemarag_k15_gold_main_v3_pro.jsonl
```

```bash
uv run python evaluate_llm_experiments.py \
  --input experiments/external_schemarag_k15_gold_main_v3_pro.jsonl \
  --gold data/gold/agent_tasks/gold_main.jsonl \
  --require-methods external-schemarag-k15 \
  --output reports/external_schemarag_k15_gold_main_v3_pro_metrics.json
```

正式结果完成后，应补充 500 条完整性、Presence、Gold coverage、Exact Value Recall、Token、延迟、
重试和成功率，并与 k=10 做配对描述性比较。`actual_cost=0` 仅表示当前服务单价配置为零，不能解释为
没有 Token 消耗。
