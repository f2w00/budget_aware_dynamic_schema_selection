# SchemaRAG 方法说明与复现指南

> 目标：让另一个 Agent 能够准确理解并实现 SchemaRAG，作为当前文化遗产结构化抽取论文的外部 Baseline。
>
> 原论文：**SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information Extraction**
>
> 作者：Sin Yu Bonnie Ho, Arlie Coles, Erik Larsson, Eric Marshall, Nathan Bodenstab, Paul Vozila（Microsoft）
>
> 发表：**ACL 2026 Industry Track**, Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics, pp. 1114–1127.
>
> 官方论文：https://aclanthology.org/2026.acl-industry.78/
>
> 注意：本文件描述的是原论文方法。当前项目采用中文文化遗产数据，因此可以使用中文 embedding 模型进行适配，但论文中必须明确这是 re-implementation / adapted implementation，而不是严格复现实验结果。

---

## 1. SchemaRAG 解决什么问题

SchemaRAG 针对的是：

> LLM 根据一个大型、复杂的 JSON Schema，从非结构化文本中抽取结构化 key-value 数据。

传统做法是：

```text
Input Text
    +
Full Schema
    ↓
LLM
    ↓
Structured JSON
```

当 Schema 很大时，会产生三个问题：

1. Prompt 中包含大量与当前文本无关的 schema rows，增加 input token 和推理成本；
2. Schema 过大可能产生 lost-in-the-middle 等问题，使 LLM 难以关注真正相关的字段；
3. 大 Schema 可能接近或超过上下文窗口限制。

SchemaRAG 的核心思想：

> **不要把整个 Schema 都交给 LLM，而是先检索与当前文本最相关的 Schema 子集，再让 LLM 进行结构化抽取。**

即：

```text
Input Text
    ↓
Embedding
    ↓
Retrieve relevant schema rows
    ↓
Reduced Schema
    ↓
LLM
    ↓
Structured JSON
```

论文报告：在其 Nursing 和 Amazon 数据上，最高取得约 8.8% relative micro-F1 提升、47% latency reduction、48% token-cost reduction。

---

# 2. SchemaRAG 的核心思想

SchemaRAG 可以理解成：

> **RAG 的检索对象不是外部知识，而是 Output Schema 本身。**

普通 RAG：

```text
query
  ↓
retrieve documents
  ↓
LLM
```

SchemaRAG：

```text
input transcript
  ↓
retrieve relevant schema rows
  ↓
reduced schema
  ↓
LLM extraction
```

因此它不是在检索“答案”，而是在检索：

> **LLM 应该被允许/引导抽取哪些字段。**

---

# 3. 原论文的 Schema 定义

论文把 Schema 定义为多个 rows：

```text
S = {r1, r2, ..., rm}
```

每个 row 至少包含：

```text
row identifier
row name
value subschema
```

例如：

```json
{
  "type": "object",
  "properties": {
    "material": {
      "type": "string",
      "enum": ["bronze", "ceramic", "jade"]
    }
  }
}
```

复杂 Schema 可以具有 hierarchy。

论文实验主要使用两级 Schema：

```text
Category
    └── Row
```

一个 row 可以属于多个 category。

Value subschema 可以是：

- Single
- Multi
- Numeric
- 其他自定义类型

---

# 4. SchemaRAG 的第一步：构建 Schema Row Embedding

这是复现时最重要的部分。

对于每一个 schema row `ri`，构建 metadata：

```text
Mi = {
    row_name,
    category_names,
    possible_values,
    ...
}
```

最少包含：

1. row name
2. category names
3. 对 Single / Multi 类型，包含可能的 enum/value

例如：

```text
Row:
    material

Category:
    physical properties

Possible values:
    bronze, ceramic, jade, wood
```

---

# 5. Row Embedding

使用预训练 embedding model：

```text
H(text)
```

对每一种 metadata 分别做 embedding。

例如：

```text
e_name     = H("material")

e_category = H("physical properties")

e_values   = H("bronze ceramic jade wood")
```

然后进行加权组合：

```text
e_row =
    normalize(
        w_name * e_name
      + w_category * e_category
      + w_values * e_values
    )
```

更一般地：

```text
e_row =
    normalize(
        Σ w_j * H(m_ij)
    )
```

其中：

- `m_ij`：row 的第 j 种 metadata
- `w_j`：对应权重
- `H`：预训练 embedding model
- `normalize`：向量归一化

论文将这些权重作为 hyperparameter，可以通过随机搜索等方法调节。

### 重要

不要简单实现成：

```text
embedding(field_name)
```

原论文明确利用了 row metadata 的组合，包括 name、category、possible values 等。

---

# 6. 如果有标注的 Few-shot Examples

SchemaRAG 还支持一个可选组件：

> schema-row-labeled examples

一个 example：

```text
x_i = transcript text

y_i = annotated structured output
```

例如：

```text
Transcript:
"The patient has ..."

Annotated output:
{
    "blood_pressure": ...,
    "heart_rate": ...
}
```

对 example text：

```text
e_example = H(x_i)
```

注意：

> example embedding 直接对 transcript text 做 embedding。

因为 row embedding 和 example embedding 在同一个 embedding space 中，所以可以直接与输入文本计算 cosine similarity。

---

# 7. Schema Retrieval

对于输入 transcript：

```text
T
```

首先：

```text
e_T = H(T)
```

然后计算它与所有 candidate embeddings 的 cosine similarity。

Candidate embeddings 包括：

```text
E = E_row ∪ E_example
```

即：

```text
Schema row embeddings
+
Example embeddings
```

计算：

```text
rho(e) = cosine_similarity(e, e_T)
```

然后按照 similarity 从高到低排序。

---

# 8. 如何形成 Reduced Schema

这是 SchemaRAG 最核心的 retrieval algorithm。

设：

```text
k = number of retrieved embeddings
```

从最高 similarity 开始依次检查。

### 如果检索结果是 row embedding

直接把对应 row 加入：

```text
S_k
```

### 如果检索结果是 example embedding

把这个 example 对应的 annotated rows 加入：

```text
S_k
```

继续直到检查了 k 个 embeddings。

最终得到：

```text
S_k ⊆ S
```

然后把 `S_k` 转换成 JSON Schema，交给 LLM。

---

# 9. 一个非常重要的细节：k 不等于最终 row 数量

不要错误实现成：

```text
Top-K embedding = Top-K fields
```

原论文中：

```text
k = number of examined embeddings
```

而最终 schema rows 可能：

```text
< k
= k
> k
```

原因：

1. 多个 embeddings 可能对应同一个 row；
2. 一个 example annotation 可能包含多个 rows。

例如：

```text
k = 5

retrieved:
    row_A
    row_B
    example_X -> {row_C, row_D}
    row_E
    example_Y -> {row_B, row_F}
```

最终：

```text
S_k = {
    A, B, C, D, E, F
}
```

所以最终有 6 个 rows，而 k=5。

---

# 10. 最小复现版本

对于当前文化遗产项目，如果没有 labeled examples，可以先实现：

```text
Record
  ↓
Embedding
  ↓
Cosine similarity
  ↓
Top-k schema rows
  ↓
Reduced JSON Schema
  ↓
LLM extraction
```

这已经能作为 SchemaRAG 的主要 retrieval-based baseline。

不需要训练 selector。

---

# 11. 当前项目的中文适配

原论文实验使用：

```text
text-embedding-ada-002
GPT-4o-2024-08-06
```

当前文化遗产项目是中文数据，因此建议使用中文 embedding，例如：

```text
BAAI/bge-small-zh-v1.5
```

或者：

```text
BAAI/bge-base-zh-v1.5
```

LLM 则保持当前项目的 extraction model。

这属于：

> SchemaRAG re-implementation / adapted implementation

而不是：

> exact reproduction of the original experimental setup.

论文中必须说明 embedding backbone 与原论文不同。

---

# 12. 推荐的实现结构

建议实现成：

```text
schemarag/
├── schema.py
├── embedder.py
├── row_embedding.py
├── retriever.py
├── prompt.py
└── pipeline.py
```

### schema.py

负责解析 JSON Schema：

```text
Schema
 ├── Row
 │    ├── name
 │    ├── category
 │    ├── type
 │    └── enum
```

### embedder.py

统一 embedding 接口：

```python
embed(text) -> vector
```

### row_embedding.py

负责：

```text
row metadata
    ↓
multiple embeddings
    ↓
weighted interpolation
    ↓
normalized row embedding
```

### retriever.py

负责：

```python
retrieve(record, k)
```

输出：

```text
selected rows
similarity scores
```

### prompt.py

负责：

```text
selected rows
    ↓
reduced JSON Schema
```

### pipeline.py

完整流程：

```text
record
 ↓
embed
 ↓
retrieve
 ↓
build reduced schema
 ↓
LLM extraction
```

---

# 13. 最小伪代码

```python
# Offline: build schema index

for row in schema.rows:
    embeddings = []

    embeddings.append(
        w_name * embed(row.name)
    )

    embeddings.append(
        w_category * embed(row.category)
    )

    if row.enum:
        embeddings.append(
            w_values * embed(" ".join(row.enum))
        )

    row.embedding = normalize(sum(embeddings))
```

Online：

```python
def schemarag(record, k):

    record_embedding = embed(record)

    candidates = []

    for row in schema.rows:
        score = cosine(
            record_embedding,
            row.embedding
        )

        candidates.append(
            (score, row)
        )

    candidates.sort(
        key=lambda x: x[0],
        reverse=True
    )

    selected_rows = [
        row for score, row
        in candidates[:k]
    ]

    reduced_schema = build_json_schema(
        selected_rows
    )

    return reduced_schema
```

然后：

```python
schema = schemarag(record, k=K)

result = llm.extract(
    text=record,
    schema=schema
)
```

---

# 14. 可选的 Example Retrieval

如果以后有 labeled examples：

```python
candidates = []

for row in row_embeddings:
    candidates.append(
        (cosine(record_embedding, row.embedding),
         ("row", row))
    )

for example in example_embeddings:
    candidates.append(
        (cosine(record_embedding, example.embedding),
         ("example", example))
    )

candidates.sort(reverse=True)

for score, item in candidates[:k]:
    if item.type == "row":
        selected_rows.add(item.row)

    elif item.type == "example":
        selected_rows.update(
            item.annotated_rows
        )
```

同时可以把 retrieved examples 放进 LLM prompt，作为 few-shot examples。

---

# 15. Transcript Segmentation：第二个组件

原论文还有：

> Transcript Segmentation

用于较长文本。

流程：

```text
Long Transcript
      ↓
LLM segmentation
      ↓
t1, t2, ..., tn
      ↓
SchemaRAG(t1)
SchemaRAG(t2)
...
SchemaRAG(tn)
      ↓
n reduced schemas
      ↓
n extraction calls
```

每个 segment：

```text
segment_i
+
reduced_schema_i
+
retrieved examples
```

独立执行 extraction。

论文中的 latency 计算利用了这些 extraction 的并行性：

```text
latency =
segmentation latency
+
max(parallel extraction latency)
```

---

# 16. 当前项目建议：先不要实现 Segmentation

当前项目主要目的是比较：

```text
Full Schema
vs
SchemaRAG
vs
Existing Adaptive
vs
Budget-aware Adaptive
```

因此第一版 SchemaRAG baseline 建议：

```text
NO segmentation
```

即：

```text
whole artifact record
      ↓
SchemaRAG
      ↓
reduced schema
      ↓
LLM
```

原因：

如果同时加入 segmentation，就无法判断性能变化到底来自：

- schema reduction
- segmentation
- parallel extraction

如果以后需要严格复现原论文完整 pipeline，再单独增加 segmentation ablation。

---

# 17. SchemaRAG 原论文实验设计

原论文使用：

### Nursing

四个医院数据集：

- 每个约 48–50 transcripts
- 每个 hospital schema 10 row types
- 有人工 annotation
- 有 schema-row-labeled examples

### Amazon

- 48 products
- 804 annotated row-value pairs
- 554 unique rows
- schema 有 1906 rows
- transcript 比 Nursing 长约 10 倍

原论文 embedding：

```text
text-embedding-ada-002
```

LLM：

```text
GPT-4o-2024-08-06
```

temperature：

```text
0
```

默认：

```text
k = 60
```

主实验重复：

```text
3 runs
```

---

# 18. 原论文 baseline

SchemaRAG 主要比较：

### Baseline A

```text
Full Schema
```

### Baseline B

```text
Full Schema + Segmentation
```

SchemaRAG：

```text
Segmentation
+
Schema reduction
+
Examples
```

另外还有：

- row-name-only retrieval 等轻量 baseline
- Oracle schema reduction

Oracle：

> 直接使用 reference annotation 中真实出现的 rows 构造 reduced schema。

Oracle 用于表示 schema reduction 的理论上限，而不是一个实际可部署的方法。

---

# 19. 原论文质量评价

原论文没有使用人为加权的综合质量分数。

它使用结构化 IE 的：

```text
micro-F1
macro-F1
```

定义 TP / FP / FN。

Multi-valued rows 按 value 展开评价。

Invalid 或 schema 外输出会被丢弃。

可类型转换的值进行 normalization，例如：

```text
"4" → 4
```

再进行评分。

---

# 20. 原论文效率指标

同时报告：

### Latency

```text
average seconds / transcript
```

### Input tokens

```text
prompt/input tokens
```

### Output tokens

```text
completion/output tokens
```

### Total tokens

```text
input + output
```

另外可以根据模型价格计算：

```text
token cost
```

这与当前项目非常相关。

---

# 21. 当前项目应该怎么作为 baseline

建议实验统一成：

```text
B1: Full Schema + Prompt
B2: Fixed Top-K
B3: SchemaRAG (re-implemented)
B4: Existing Adaptive
B5: Ours: Budget-aware Adaptive
```

其中：

### B1

所有 18 fields 给 LLM：

```text
Full Schema
+
"only output fields supported by the text"
```

### B2

按某种固定策略选：

```text
Top-3
Top-5
Top-8
Top-10
```

### B3

SchemaRAG：

```text
embedding
+
schema metadata
+
cosine similarity
+
Top-K
```

### B4

当前已有 Adaptive 方法。

### B5

最终方法：

```text
field utility
+
field cost
+
budget constrained selection
```

---

# 22. SchemaRAG 与当前方法的核心区别

SchemaRAG：

```text
score_i = similarity(record, schema_i)

select Top-K
```

本质：

> relevance-based schema retrieval

当前方法：

```text
utility_i = learned_utility(record, field_i)

maximize:
    Σ utility_i

subject to:
    Σ cost_i <= Budget
```

本质：

> utility + cost + explicit budget constrained schema selection

因此论文里的方法关系可以非常清楚：

```text
Full Schema
      ↓
Fixed Top-K
      ↓
SchemaRAG
      ↓
Existing Adaptive
      ↓
Budget-aware Adaptive (Ours)
```

---

# 23. 不要把 SchemaRAG 实现错误成下面这些

### 错误 1：只 embedding field name

```text
embed("material")
```

这不是完整的 SchemaRAG row metadata embedding。

至少应该考虑：

```text
name
category
possible values
```

---

### 错误 2：把 similarity 当成最终 extraction

SchemaRAG 的 embedding retrieval 只是：

```text
schema selection
```

不是：

```text
information extraction
```

最终仍然需要 LLM。

---

### 错误 3：直接用 Top-K 字段但不保留 schema definition

必须把选中的 rows 转换成完整 JSON Schema，包括：

```text
field name
type
enum
constraints
hierarchy
```

否则比较不公平。

---

### 错误 4：用人工 Gold 直接训练 SchemaRAG

原方法是 training-free。

不需要训练 selector。

---

### 错误 5：用 Full Schema extraction 输出当测试 Gold

当前项目中：

```text
Full Schema output
```

可以作为 selector 的弱监督来源，但最终测试质量最好使用独立人工 Gold。

---

# 24. 当前论文中的推荐表述

如果使用 BGE：

> **SchemaRAG (Re-implemented).** We re-implement SchemaRAG following the original retrieval-based schema reduction procedure. For our Chinese cultural heritage corpus, we replace the original text-embedding-ada-002 encoder with BGE-small-zh-v1.5 while retaining the schema metadata embedding, cosine-similarity retrieval, and top-k schema reduction strategy.

不要写：

> We reproduce the original SchemaRAG results.

更准确：

> We re-implement / adapt SchemaRAG for our Chinese cultural heritage extraction setting.

---

# 25. 推荐 Agent 实现顺序

另一个 Agent 实际执行时应该严格按以下顺序：

```text
Step 1
Parse current JSON Schema

Step 2
Create metadata for every schema field

Step 3
Load BGE embedding model

Step 4
Offline-compute embedding for each field metadata

Step 5
For each artifact record:
    compute record embedding

Step 6
Calculate cosine similarity
between record and every schema field

Step 7
Select top-k

Step 8
Construct reduced JSON Schema

Step 9
Send original record + reduced schema to same LLM

Step 10
Evaluate:
    F1
    input tokens
    output tokens
    total tokens
    cost
    latency
```

先完成这个版本。

**不要先实现 segmentation、example retrieval、复杂 ANN、分布式系统。**

18 个字段的当前实验甚至可以直接 brute-force cosine similarity，不需要 FAISS/pgvector；只有 schema 扩展到几千/几万 fields 时才有必要考虑 ANN。

---

# 26. 当前项目的最终目标

SchemaRAG baseline 的意义不是证明：

> “我们重新发明了 dynamic schema。”

而是给当前方法一个强的外部参照：

```text
SchemaRAG:
    relevance → Top-K

Ours:
    utility + cost → budget optimization
```

最终应该回答：

> 在相同或更低 token budget 下，是否可以比 similarity-based SchemaRAG 保持更高的 extraction quality？

最核心的图建议：

```text
Quality
  ↑
  │             Full Schema
  │                ●
  │        Ours ●
  │          ●
  │   SchemaRAG ●
  │      ●
  │ Fixed Top-K
  │
  └────────────────────→ Token Cost
```

以及：

```text
Schema Size
10 → 18 → 30 → 50 → 80

比较：
Full Schema
SchemaRAG
Existing Adaptive
Ours
```

如果随着 schema size 增加，SchemaRAG 和你的方法相对于 Full Schema 的优势越来越明显，同时你的方法在相同 budget 下保持更高 F1，那么整个论文的实验故事就会非常完整。

---

## Reference

Ho, S. Y. B., Coles, A., Larsson, E., Marshall, E., Bodenstab, N., & Vozila, P. **SchemaRAG: Dynamic Large Schema Reduction for LLM-driven Structured Information Extraction.** Proceedings of ACL 2026, Industry Track, pp. 1114–1127.

官方论文：
https://aclanthology.org/2026.acl-industry.78/

论文 PDF：
https://aclanthology.org/2026.acl-industry.78.pdf
