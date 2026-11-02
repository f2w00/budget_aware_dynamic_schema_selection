# SchemaRAG k=10 适配重实现说明

## 方法来源

本文实现依据以下论文的算法描述：

- 题目：SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured
  Information Extraction
- 作者：Sin Yu Bonnie Ho, Arlie Coles, Erik Larsson, Eric Marshall, Nathan
  Bodenstab, Paul Vozila
- 版本：ACL 2026 Industry Track，Proceedings of ACL，pp. 1114-1127
- 官方页面：https://aclanthology.org/2026.acl-industry.78/
- PDF：https://aclanthology.org/2026.acl-industry.78.pdf
- 官方代码：未公开，因此本项目是基于论文描述的 re-implementation / adapted
  implementation，不是官方实现复现。

## 当前适配

方法名为 `external-schemarag-k10`。每个字段作为一个 Schema row，使用
`simple2.json` 的字段名和完整中文定义构造 metadata。两部分分别通过 embedding
服务编码，再以等权重组合并归一化：

```text
row_embedding = normalize(0.5 * name_embedding + 0.5 * definition_embedding)
```

每条完整 `raw_data` 按现有 Runner 的确定性 JSON 序列化方式编码，并添加固定查询指令：

```text
给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。
```

随后计算 cosine similarity，按相似度降序选择 Top-10；相似度相同时按
`simple2.json` 字段顺序。当前没有独立 category、enum 或层级 metadata，因此完整字段
定义承担可用的语义和值示例信息。

## 固定配置与边界

- Embedding endpoint：由 `EMBEDDING_BASE_URL`、`EMBEDDING_API_KEY` 和
  `EMBEDDING_MODEL` 配置，当前服务模型名为 `text-embedding`。
- Embedding 模型对应服务端提供的 `Qwen3-Embedding-0.6B`；服务端未提供 revision 时，
  以 endpoint 配置和运行记录为准。
- `k=10`，无 required fields。
- 不使用 `field_weights.json`；该文件仅用于本文内部 Adaptive selector。
- 不实现 segmentation、example retrieval、ANN 或训练 selector。
- embedding 只负责 `raw_data -> selected_fields`；LLM、Prompt、Schema、解析、重试和
  Token 统计均复用 `llm_runner.py`。
- `estimated_field_cost` 按现有成本口径记录，但不参与 SchemaRAG 字段排序。

## 运行命令

冒烟运行：

```bash
LLM_MODEL=kg-assist-pro LLM_TEMPERATURE=0.0 \
uv run python llm_runner.py \
  --methods external-schemarag-k10 \
  --ids-file data/gold/gold_main_ids.json \
  --limit 10 --max-concurrency 20 --queue-size 40 --max-retries 2 \
  --output experiments/external_schemarag_k10_gold_main_v3_pro_smoke.jsonl
```

正式运行使用同一命令，将 `--limit` 改为 `500` 并使用正式输出文件：

```text
experiments/external_schemarag_k10_gold_main_v3_pro.jsonl
```

评估和共同成功样本配对必须使用现有统一脚本，且不能把本方法与非共同成功记录直接
比较。当前结果完成前，不对质量优劣作结论。

## 产物与限制

row embedding 缓存位于 `experiments/schemarag_row_embeddings.json`，由 Schema、模型名和
metadata 权重哈希约束。相同配置哈希同时写入每条结果，并用于外部方法的断点续跑判断。
正式实验还应记录实际 embedding 服务版本、Prompt/schema hash、
提交数、成功/失败数、重试统计、输入/输出/总 Token 和共同成功样本规模。

本实现与原论文的语言、embedding endpoint、Schema 规模、LLM 和无 segmentation 设置均
不同，因此不能将原论文报告的数字代入当前实验，也不能称为严格复现原论文实验。
