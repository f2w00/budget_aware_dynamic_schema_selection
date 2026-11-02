# 强模型 + Agent 金标注方案

## 1. 文档目的

本项目不采用纯人工逐条标注，而是使用：

```text
原始 raw_data
    ↓
强模型生成初始抽取
    ↓
Agent 按字段定义、原文证据和一致性规则审核
    ↓
修正、确认并冻结
    ↓
Gold 标注集
```

该方案用于为 Full、Prompt-only、Fixed 和 Adaptive LLM 抽取提供统一的质量
参考集。下一个 Agent 必须把这份数据视为本项目的**模型辅助、证据约束、经
Agent 审核的 Gold 集**。

论文中建议称为：

```text
expert-model-assisted gold annotations
agent-adjudicated gold set
evidence-grounded reference annotations
```

不要声称这是完全独立于模型的人工金标准。因为标注初稿由强模型产生，Agent
也可能使用相同或相近的语言模型。必须在论文限制中说明这一点。

## 2. 标注目标

对测试集中的每条原始记录，逐项确认当前 Schema 的 18 个字段：

```text
name
category
form_type
usage
condition
dynasty
period
acq_type
region
site
culture
material
motif
craft
craft_school
related_event
related_person
```

强模型和 Agent 在生成阶段仍然要回答四个问题：

1. 原始记录中是否有足够证据支持该字段？
2. 如果有，标准字段值是什么？
3. 证据位于原始 JSON 的哪个路径、哪段文本？
4. 这个值是原文直接出现，还是由低风险规则推导得到？

核心原则：

```text
Gold 只能基于 artifact_raw.raw_data 及明确的字段归一化规则。
没有原文依据时必须在参考结果中写为 `null`，不允许依靠常识或外部知识补全。

正式实验文件可以采用轻量格式，不必把证据和审核过程全部嵌入每条结果：

```json
{
  "artifact_id": 5,
  "schema_version": "simple2-v1",
  "reference_model": "strong-model-v1",
  "reference_result": {
    "name": "北宋赵佶鸜鹆图轴",
    "category": "书画",
    "form_type": "轴",
    "usage": "观赏",
    "desc": "描绘三只鸜鹆相斗的宋代绘画作品",
    "condition": null,
    "dynasty": "宋",
    "period": "北宋",
    "acq_type": null,
    "region": null,
    "site": null,
    "culture": "中原文化",
    "material": "绢本",
    "motif": "鸜鹆相斗",
    "craft": "绘画",
    "craft_school": null,
    "related_event": null,
    "related_person": "赵佶"
  }
}
```

`reference_result` 必须包含 Schema 的全部 18 个字段。字段没有足够证据时写
`null`，不能省略字段。这样可以区分“强模型确认为空”和“标注文件漏字段”，
也可以直接计算字段存在性质量。

证据、`support_level`、`review_status` 和 `note` 属于审核过程数据。它们可以
保存在 draft 或审计文件中，但不要求复制进最终的轻量 reference 文件。
```

## 3. 数据抽样规模

推荐建立以下数据集：

```text
gold_main：500 条 test 集记录
gold_overlap：gold_main 中的 100 条，由第二个 Agent 或人工复核
gold_tail：可选，额外 50～100 条长尾字段诊断记录
```

资源有限时的最低方案：

```text
gold_main：300 条
gold_overlap：其中 50 条复核
```

资源充足时可以标注全部 `676` 条 test 记录。无论采用哪种规模，所有方法都
必须在完全相同的 `artifact_id` 集合上运行：

```text
gold_main = Full 输入记录 = Prompt-only 输入记录
           = Fixed 输入记录 = Adaptive 输入记录
```

## 4. 抽样要求

不能只按数据库顺序取前 N 条。应从 test 集进行分层抽样，至少覆盖：

```text
原始记录长度：短 / 中 / 长
Full Schema 弱监督非空字段数：低 / 中 / 高
文物类别
ML 预测字段数
字段预测置信度：高 / 中 / 低
长尾字段比例
```

抽样时必须保存抽样清单：

```text
data/gold/gold_main_ids.json
data/gold/gold_overlap_ids.json
data/gold/gold_tail_ids.json
```

抽样清单一旦用于正式实验就冻结，不能根据模型结果反复调整。

## 5. 推荐文件结构

主文件使用 JSONL，一条记录对应一行：

```text
data/gold/gold_main.jsonl
data/gold/gold_overlap.jsonl
data/gold/gold_tail.jsonl
```

正式 reference 文件的最小记录格式如下：

```json
{
  "artifact_id": 5,
  "schema_version": "simple2-v1",
  "reference_model": "strong-model-v1",
  "reference_result": {
    "name": "北宋赵佶鸜鹆图轴",
    "category": "书画",
    "form_type": "轴",
    "usage": "观赏",
    "desc": "描绘三只鸜鹆相斗的宋代绘画作品",
    "condition": null,
    "dynasty": "宋",
    "period": "北宋",
    "acq_type": null,
    "region": null,
    "site": null,
    "culture": "中原文化",
    "material": "绢本",
    "motif": "鸜鹆相斗",
    "craft": "绘画",
    "craft_school": null,
    "related_event": null,
    "related_person": "赵佶"
  }
}
```

必填顶层字段：

```text
artifact_id
schema_version
reference_model
reference_result
```

`reference_result` 的 18 个键必须全部存在，值只能是字段值、数组或 `null`。
缺失的 Schema 键视为不合法，而不是默认 `null`。

`raw_data` 不建议复制进 reference 文件。评估时根据 `artifact_id` 从
`public.artifact_raw` 读取，避免参考文件和数据库原始数据发生不一致。

最终实验只使用 `reference_result` 作为参考答案，不要求 reference 文件保存
`present`、`evidence`、`support_level` 或 `review_status`。其中：

```text
reference_result[field] 非 null → 参考答案认为字段存在
reference_result[field] == null  → 参考答案认为字段不存在或无法可靠抽取
```

因此，`null` 不是“没有标注”，而是强模型和 Agent 确认后的负标签。

### 5.1 审计过程文件

强模型初稿和 Agent 审核可选地保存在独立审计目录，不影响轻量 reference 文件：

```text
data/gold/drafts/<artifact_id>.json
data/gold/reviews/<artifact_id>.json
data/gold/adjudication_log.jsonl
```

审计文件可保留 `evidence`、`support_level`、`review_status`、`note` 和原始响应，
用于抽查强模型是否产生无依据值；正式评估不依赖这些字段。

## 6. 参考结果生成约束

强模型必须：

```text
1. 只根据 artifact_raw.raw_data 和 simple2.json 生成结果。
2. 使用完整的 18 个英文 Schema 键。
3. 无足够信息时输出 null，不允许省略键。
4. 不得使用外部知识补全相关人物、历史事件、窑口或来源地。
5. 保存强模型名称、版本和生成 Prompt 哈希。
```

Agent 的最低职责是结构审核：检查 JSON 合法性、18 个键完整性、键名正确性和
`null` 处理一致性。证据审核是推荐的质量控制步骤，但不再是最终 reference
文件的必填结构。

## 7. 18 个字段的判定边界

字段定义以 `simple2.json` 为准。以下规则用于减少 Agent 自由发挥。

```text
name
    文物名称、标题或原始记录明确给出的名称。

category
    陶器、瓷器、青铜器、书画等文物类别。

form_type
    鼎、瓶、俑、镜、卷、轴等器型或形制。

usage
    礼器、明器、兵器、食器、宗教文献等用途或功能。

desc
    原始记录明确支持的简短综合描述，不要求逐字复制全文。

condition
    完整、残缺、修复等明确保存状态。

dynasty
    朝代或政权。允许按照冻结的历史层级规则做低风险归一化。

period
    年号、早中晚期、具体历史阶段等比 dynasty 更细的时间信息。

acq_type
    考古发掘、传世、捐赠等明确取得方式。

region
    河南、陕西、江南等较大来源地区或区域。

site
    殷墟、马王堆、墓葬等具体出土、发现或来源地点。

culture
    原文明确给出的文化归属，不因朝代自动推断文化。

material
    陶、瓷、青铜、玉、绢本等主要材质。

motif
    龙纹、兽面纹、莲花纹等明确纹饰或装饰主题。

craft
    范铸、拉坯、刻花、镶嵌、绘画等制作工艺。

craft_school
    越窑、汝窑、具体流派或制作体系；不能仅由材质和时代推断。

related_event
    与记录直接相关且原文有依据的历史大事件；不能把朝代泛化为事件。

related_person
    原文明确提及或可由固定文本关系低风险确定的直接相关人物。
```

## 8. 归一化规则

归一化规则必须先写入版本化配置，再开始正式标注。建议文件为：

```text
data/gold/normalization_rules_v1.json
```

当前建议：

```text
北宋 → dynasty=宋，period=北宋
南宋 → dynasty=宋，period=南宋
盛唐 → dynasty=唐，period=盛唐
绢本 → 保留 material=绢本；不要未经规则改成绢
```

正式轻量 reference 只保存 `reference_result` 的最终值。若需要保留原始值和
归一化值，应放在审计文件中，不要在不同版本的 reference 文件中混用口径。

同义归一化必须保守。以下情况不能自动视为完全一致，除非规则文件明确规定：

```text
纸本 vs 纸
中原 vs 中原地区
宋 vs 北宋
书画 vs 绘画
```

## 9. 证据标注

每个 `present=true` 字段原则上至少有一条证据：

```json
"evidence": [
  {
    "json_path": "$.description",
    "quote": "青铜器表面镶嵌绿松石"
  }
]
```

证据要求：

- `json_path` 必须能定位到 `raw_data` 中的字段。
- `quote` 必须是原文中的短片段或其连续子串。
- 不得只写“根据上下文”而没有原文片段。
- 一个字段可有多条证据。
- `derived` 字段也必须保存推导所依赖的原文证据。

证据用于计算：

```text
Evidence-supported Precision
Unsupported Field Rate
```

## 10. Agent 标注工作流

### 阶段 A：强模型生成初稿

输入：

```text
artifact_id
完整 raw_data
18 个字段定义
字段归一化规则
```

输出：18 个字段的候选值、证据和审核辅助信息。最终 reference 中有值的字段
保留候选值，没有可靠信息的字段必须写为 `null`。初稿必须保留，不得覆盖。

```text
data/gold/drafts/<artifact_id>.json
```

### 阶段 B：Agent 逐字段审核

Agent 必须按以下顺序审核：

1. 检查候选值是否能在 raw_data 中找到证据。
2. 检查 JSONPath 和 quote 是否真实存在。
3. 检查字段是否与 `simple2.json` 定义匹配。
4. 检查是否把外部知识、常识或朝代知识误当成原文事实。
5. 执行已冻结的归一化规则。
6. 无充分证据则在最终 `reference_result` 中改为 `null`。
7. 检查字段之间的一致性，例如 dynasty 与 period。
8. 在审计文件中写入 `review_status` 和必要的 `note`（如果保存审计文件）。

### 阶段 C：冲突复核

以下情况必须进入复核队列：

```text
候选值没有可验证证据
Agent 无法判断 explicit 或 derived
两个值都可能成立
涉及 related_event 或 craft_school 的高风险推断
多值字段边界不明确
归一化规则没有覆盖
```

复核结果只能是：

```text
confirmed
corrected
rejected
needs_review
```

正式 Gold 文件中不允许保留 `needs_review`，所有记录必须在冻结前解决。

## 11. 多值字段

如果一个字段明确包含多个值，统一使用数组。以下示例属于审计过程中的字段
记录；最终轻量 reference 只保留该字段的数组值：

```json
{
  "present": true,
  "value": ["青铜", "绿松石"],
  "normalized_value": ["青铜", "绿松石"]
}
```

只有一个值时，是否仍使用数组必须在 Schema 评估器中统一。建议按照字段类型
配置；不要让同一个字段在不同记录中随意混用字符串和数组。

多值字段评估使用集合 Precision / Recall / F1，不使用字符串 Exact Match。

## 12. `desc` 字段

`desc` 不适合使用严格字符串完全匹配。Agent 应在审计过程中确认并输出证据
支持的简短事实摘要；最终轻量 reference 只保留摘要字符串：

```json
{
  "present": true,
  "value": "描绘三只鸜鹆相斗的宋代绘画作品",
  "support_level": "derived",
  "evidence": [
    {
      "json_path": "$.description",
      "quote": "本幅画三只鸜鹆，两只正在激烈相斗"
    }
  ]
}
```

`desc` 的主要评估方式：

```text
证据支持率
关键词或事实单元覆盖
语义相似度
人工抽查
```

## 13. 质量评分方案

正式实验不把整条 JSON 拼接后只计算一个 embedding 相似度，也不把字段 Exact
Match 作为主质量分数。采用以下三部分组成的质量总分：

```text
Quality
    = 0.5 × Presence-F1
    + 0.3 × LLM-Value-Score
    + 0.2 × Desc-Embedding-Score
```

三个分数都归一化到 `[0, 1]`，权重在正式实验前固定。论文中必须同时报告三个
分项，不能只报告综合分数。

### 13.1 Presence-F1

把 reference 和 prediction 的字段值转换为是否非空：

```text
reference_result[field] != null → gold_present=true
prediction[field] != null       → pred_present=true
```

在所有记录和 18 个字段上计算标准 Presence Precision、Recall 和 F1，并报告
Macro-F1 与 Micro-F1。额外报告 `Reference-present Recall` 和
`Reference-absent False Positive Rate`，避免“所有字段都输出 null”造成虚高结果。

### 13.2 LLM-Value-Score

LLM Judge 只评价非 `desc` 结构化字段的值，不重复评价 `desc`。Judge 输入至少
包含：

```text
原始 raw_data
字段名和字段定义
reference_result[field]
prediction[field]
```

Judge 只能根据 raw_data 判断，reference 只是模型辅助参考，不是绝对真理。字段
级评分固定为：

```text
1.00  值正确或语义等价
0.75  基本正确，但粒度或范围略有差异
0.50  部分正确，但存在明显缺失或不完整
0.25  与字段主题相关，但字段值明显不正确
0.00  错误、无关、无依据，或 reference 有值而 prediction 为 null
```

reference 和 prediction 都为 null 的字段不进入 Value-Score 分母，由
Presence-F1 评价空值判断。reference 有值但 prediction 为 null 时进入分母，
分数为 0。每个 Judge 结果应保存 `artifact_id`、字段名、分数、理由、Judge
模型版本和 Prompt 哈希。

### 13.3 Desc-Embedding-Score

只对 `desc` 计算 embedding 余弦相似度：

```text
reference_result["desc"]
prediction["desc"]
    ↓
同一个固定 embedding 模型
    ↓
cosine similarity
```

embedding 不能用于整条 JSON 的唯一分数，也不能替代结构化字段的 LLM Judge。
高相似度不保证朝代、人物或事件正确。当两者都为 null 时不进入分母；reference
有值而 prediction 为 null 时分数为 0。模型和阈值必须在 validation 或独立校准
集上固定，不能使用 test 结果调参。

### 13.4 空值和省略处理

```text
reference 非 null + prediction 非 null
    → Presence 判断 + LLM Value-Score

reference 非 null + prediction 为 null/省略
    → Presence Recall 损失；Value-Score=0

reference 为 null + prediction 非 null
    → Presence False Positive；Value-Score=0

reference 为 null + prediction 为 null/省略
    → Presence 判断正确；不进入 Value-Score
```

## 14. Gold 质量控制

强模型初稿不能直接进入 Gold。至少执行以下检查：

### 14.1 结构检查

- 每个 `artifact_id` 唯一。
- 18 个字段全部存在。
- `reference_result` 必须包含全部 18 个字段。
- 字段值只能是字符串、数字、布尔值、数组或 `null`。
- 不允许额外字段混入 `reference_result`。
- `reference_model` 和 `schema_version` 必须存在。
- 审计文件中的 evidence、support_level 和 review_status（如果保存）必须格式合法。

### 14.2 双重审核

`gold_overlap` 中的记录由第二个 Agent 或人工独立审核，不先查看初稿结论。
如果保留 overlap 审核，比较：

```text
字段非空/空值一致率
字段值语义一致率
证据支持判断一致率
```

冲突字段必须写入：

```text
data/gold/adjudication_log.jsonl
```

### 14.3 字段覆盖检查

正式冻结前统计每个字段：

```text
positive_count
negative_count
explicit_count
derived_count
evidence_count
```

如果长尾字段正例太少，增加 `gold_tail` 诊断集，但不能把诊断集和自然分布
主评估集无说明地混合。

## 15. 与实验结果的对应方式

所有 LLM 结果按 `artifact_id` 与 Gold 对齐：

```text
gold_main.jsonl
    ├── Full
    ├── Prompt-only prompt-null
    ├── Prompt-only prompt-omit
    ├── Fixed-5 / Fixed-10
    ├── Adaptive budget-200 / budget-400
    └── Adaptive lambda-0.003
```

`prompt-omit` 的省略不能自动计为正确：

```text
Gold present=true + 模型省略
    → Recall 损失

Gold present=false + 模型省略
    → 正确

Gold present=false + 模型输出值
    → False Positive / Unsupported Field
```

统一输出以下指标：

```text
Presence Macro-F1 / Micro-F1
Reference-present Recall
Reference-absent False Positive Rate
LLM-Value-Score（不包含 desc）
Desc-Embedding-Score
Quality = 0.5 × Presence-F1 + 0.3 × LLM-Value-Score
          + 0.2 × Desc-Embedding-Score
Input / Output / Total Tokens
C_full / B_required / C_actual
P50 / P95 延迟
失败率和重试率
```

## 16. 论文表述和限制

推荐方法描述：

```text
We constructed an evidence-grounded, model-assisted gold set by asking a
stronger extraction model to produce draft annotations and using an Agent to
verify, correct, and adjudicate every field against the original raw record.
```

必须报告的限制：

- Gold 初稿不是完全独立人工标注。
- 强模型和被测 LLM 可能共享语言知识或系统偏差。
- Agent 审核也可能继承强模型初稿错误。
- 证据约束降低幻觉风险，但不能保证所有字段解释均无偏差。
- 结果应称为“经 Agent 审核的模型辅助 Gold”，不应宣称绝对真实标签。

## 17. 下一个 Agent 的执行顺序

1. 检查 test ID 文件、`simple2.json` 和 raw 数据访问是否正常。
2. 实现分层抽样脚本，输出 `gold_main_ids.json` 和 `gold_overlap_ids.json`。
3. 实现强模型 reference 生成器，要求完整 18 字段并将无信息字段写为 null。
4. 实现 Agent 结构审核器，检查 JSON、键完整性、键名和 null 语义。
5. 可选实现独立审计文件，保存证据、修正记录和原始响应。
6. 先处理 30～50 条试标记录，检查边界和归一化规则。
7. 冻结 `schema_version`、`reference_model` 和评分 Prompt 版本。
8. 扩展到 300～500 条轻量 reference 主集。
9. 用 reference 主集运行所有 LLM 方法。
10. 用 Presence-F1、LLM-Value-Score 和 desc embedding 计算质量总分。
11. 用配对记录级指标比较 Adaptive 和 Prompt-only 的质量与成本。

## 18. 当前结论边界

在 Gold 集完成前，只能报告：

```text
强模型 + Agent 金标注流程已设计或实现。
```

Gold 集完成且质量控制通过后，才可以报告：

```text
在经证据审核的模型辅助 Gold 集上，比较不同 Schema 策略的抽取质量。
```

只有正式 LLM 实验和 Token 统计完成后，才可以讨论：

```text
Adaptive Schema 是否在相近质量下比 Prompt-only 消耗更少。
```
