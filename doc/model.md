# Lightweight Record-aware Schema Budget Controller

## 1. 模型定位

在大规模文化遗产数据结构化抽取任务中，直接使用完整领域 Schema 进行大模型抽取会带来较高成本。

由于不同文物记录的信息丰富程度存在明显差异：

* 有些记录只有名称和年代
* 有些记录包含详细描述、工艺、纹饰、历史背景等信息

因此，不同记录不应该使用完全相同的抽取 Schema。

本项目设计一个：

**Lightweight Record-aware Schema Budget Controller**

作为结构化抽取前置模块。

其目标：

> 在低成本情况下，根据输入记录特点预测合适的 Schema 范围和抽取预算，使后续大模型只处理必要的信息字段。

---

# 2. 模型在整体系统中的位置

完整流程：

```
文化遗产记录

        |
        v

Lightweight Schema Budget Controller

        |
        +----------------+
        |                |
        v                v

字段选择            抽取预算控制

        |                |

        +----------------+

                |

                v

动态 Schema

                |

                v

LLM结构化抽取

                |

                v

标准化文化遗产数据
```

该模型位于：

> 原始数据和大模型抽取之间。

---

# 3. 模型解决的问题

## 3.1 固定 Schema 问题

传统流程：

```
Record

+

完整领域 Schema

+

LLM

```

缺点：

* 简单记录包含大量无关字段
* Prompt 长度增加
* 推理成本增加
* 大规模处理效率降低

---

## 3.2 动态需求问题

不同记录需要不同信息范围。

例如：

### 简单记录

输入：

```
青花盘
```

可能只需要：

```
名称
年代
材质
器型
```

### 复杂记录

输入：

```
明代景德镇窑青花缠枝莲纹盘，
具有复杂纹饰和工艺描述
```

需要：

```
名称
年代
材质
器型
纹饰
工艺
历史背景
```

因此需要一个自动判断机制。

---

# 4. 模型核心任务

模型不负责直接抽取文物信息。

它负责：

## Task 1：Schema需求预测

预测当前记录需要哪些字段。

例如：

输入：

```
明代青花莲纹盘
```

输出：

```
name        高需求
dynasty     高需求
material    高需求
decoration  高需求
collector   低需求
```

---

## Task 2：Budget预测

预测当前记录应该投入多少抽取资源。

例如：

简单记录：

```
低预算
少字段
```

复杂记录：

```
高预算
多字段
```

---

## Task 3：字段数量控制

根据记录复杂度决定：

```
选择多少字段
```

避免固定 Top-K。

---

# 5. 输入信息

模型输入不是完整文本理解任务，而是低成本记录分析。

输入可以包括：

## 文本特征

* 文本长度
* Token 数量
* 描述完整程度

## 结构特征

* 已存在字段数量
* 缺失字段数量
* 数据类别

## 领域特征

* 文物类别
* 年代信息
* 材质信息
* 器型信息

这些特征用于判断：

```
Record Complexity

↓

Schema Requirement
```

---

# 6. 输出结果

模型输出字段需求评分：

示例：

```json
{
"name":0.98,
"dynasty":0.95,
"material":0.92,
"form":0.90,
"decoration":0.75,
"technique":0.60,
"collector":0.10
}
```

以及预算信息：

```json
{
"budget_level":"medium",
"max_fields":10
}
```

之后生成当前记录专属 Schema。

---

# 7. 训练方式

由于 55 万条数据没有人工字段标注，因此采用自动生成监督信号。

流程：

```
部分样本

        |

        v

完整 Schema 抽取

        |

        v

生成字段使用标签

        |

        v

训练轻量模型

        |

        v

应用到大规模数据
```

---

# 8. 训练数据生成

选择部分记录：

例如：

```
5000~10000 条
```

使用完整 Schema 进行一次高质量抽取。

根据抽取结果生成标签：

例如：

输入：

```
青花盘
```

完整抽取：

```
name:
青花盘

dynasty:
明

material:
瓷

decoration:
null
```

转换为：

```
name=1
dynasty=1
material=1
decoration=0
```

形成多标签训练数据。

---

# 9. 模型类型

该模块不需要大型神经网络。

目标是：

* 快速
* 稳定
* 可扩展

可采用：

* Logistic Regression
* LightGBM
* XGBoost
* Small MLP

重点不是模型复杂度，而是：

> 用极低成本完成大规模 Schema 决策。

---

# 10. 推理流程

对于每条新记录：

```
输入记录

    |

    v

提取轻量特征

    |

    v

Budget Controller预测

    |

    v

生成字段排序

    |

    v

确定 Schema

    |

    v

调用LLM抽取
```

---

# 11. 与已有 Schema Selection 方法区别

传统 Schema Selection：

```
输入

↓

寻找相关字段

↓

Schema
```

关注：

> 字段相关性。

本模型：

```
输入

↓

判断记录复杂度

↓

决定预算

↓

决定字段范围

↓

Schema
```

关注：

> 大规模抽取中的资源分配。

---

# 12. 论文中的定位

模型定位：

不是一个简单字段分类器。

而是：

> A lightweight record-aware budget controller for adaptive schema allocation in large-scale structured extraction.

核心贡献：

1. 让 Schema 选择过程具有记录感知能力
2. 让抽取预算根据数据复杂度动态变化
3. 避免大规模数据处理中固定 Schema 带来的资源浪费

---

# 13. 实验验证

主要比较：

## Full Schema

所有字段输入。

---

## Fixed Schema

固定字段数量。

---

## Adaptive Budget Controller

动态字段和预算。

评价：

质量：

* Field-level F1
* Completeness

效率：

* Token 消耗
* 推理时间
* 成本

目标：

证明：

```
较低成本

+

接近完整 Schema 的质量

+

更好的扩展能力
```

---

# 14. 一句话总结

Lightweight Record-aware Schema Budget Controller 是一个位于数据和大模型之间的小型决策模型，它通过理解单条文化遗产记录的复杂度，动态决定抽取 Schema 和资源预算，从而支持几十万级文化遗产数据的高效结构化处理。

---

# 15. 当前实验结论

## 15.1 数据与模型

当前使用两张表通过 `artifact_id` 对齐后的 6763 条记录：

```text
artifact_raw.raw_data       → record_text
artifact_extract.*          → 字段 present 弱监督标签
```

数据划分如下：

```text
训练集：5410 条
验证集：677 条
测试集：676 条
```

当前模型采用字符级 TF-IDF 与多标签 Logistic Regression：

```text
原始 JSON
    ↓
字符级 TF-IDF（2～5-gram）
    ↓
字段 present 分类器
    ↓
各字段需求分数
```

字段标签定义为：Full Schema 抽取结果中的字段非空时为 `1`，为空时为 `0`。

## 15.2 字段预测效果

测试集上的主要结果如下：

| 字段 | F1 |
| --- | ---: |
| `category` | 1.000 |
| `desc` | 1.000 |
| `name` | 1.000 |
| `material` | 0.998 |
| `period` | 0.998 |
| `dynasty` | 0.988 |
| `form_type` | 0.988 |
| `related_person` | 0.848 |
| `usage` | 0.846 |
| `region` | 0.816 |
| `condition` | 0.800 |
| `motif` | 0.731 |
| `culture` | 0.584 |
| `related_event` | 0.471 |
| `craft_school` | 0.316 |
| `site` | 0.286 |
| `acq_type` | 0.182 |
| `craft` | 0.000 |

## 15.3 当前判断

当前模型表现具有明显的字段差异：

1. 高频字段预测效果较好。`dynasty`、`material`、`form_type` 等字段可以用于第一版动态 Schema 选择。
2. 中频字段基本可用。`usage`、`region`、`condition`、`motif` 和 `related_person` 可以使用模型连续分数参与字段排序。
3. 长尾字段预测不稳定。`craft`、`site`、`acq_type`、`craft_school` 和 `related_event` 的正样本数量很少，当前结果只能作为初步参考。
4. `name`、`category` 和 `desc` 在当前数据中正样本率为 100%，模型将其作为常量字段处理。因此这些字段的 F1 不代表分类器学习能力，更适合直接作为必选字段。

## 15.4 结果解释边界

当前测试标签来自 Full Schema 抽取结果的非空判断，因此本实验衡量的是：

```text
模型复现 Full Schema 字段非空模式的能力
```

它还不能完全等同于：

```text
模型判断原始文本中是否存在真实且正确字段信息的能力
```

原因是 Full Schema 结果属于弱监督来源，其中可能包含抽取错误或模型幻觉。当前结果可用于完成第一版字段选择闭环，但不应直接作为人工核验后的字段抽取准确率。

## 15.5 下一步

下一步将模型输出的字段分数与字段动态成本结合：

```text
字段需求分数
    +
字段 Schema 定义成本
    +
字段预期输出成本
    ↓
预算约束下的动态 Schema 选择
```

最终需要比较 Full Schema、Fixed Schema 和 Adaptive Schema 在不同预算下的质量与成本，包括字段覆盖率、Schema Token、LLM 总成本和处理时间。

## 15.6 字段输出 Token 预测效果

在共享 TF-IDF 编码基础上，为每个字段增加 Ridge 回归头，使用非空字段的
`field_output_tokens` 标签训练。回归目标使用 `log1p(Token 数)`，预测后还原为
非负整数。

字段输出长度预测使用测试集非空字段样本评估，主要指标为平均绝对误差（MAE）
和中位绝对误差：

| 字段 | 测试非空样本数 | MAE（Token） | 中位绝对误差（Token） |
| --- | ---: | ---: | ---: |
| `category` | 676 | 0.11 | 0.05 |
| `material` | 674 | 0.24 | 0.10 |
| `dynasty` | 507 | 0.30 | 0.13 |
| `period` | 674 | 0.65 | 0.28 |
| `form_type` | 669 | 0.98 | 0.62 |
| `name` | 676 | 2.76 | 2.10 |
| `usage` | 428 | 2.17 | 1.48 |
| `motif` | 162 | 2.50 | 1.55 |
| `desc` | 676 | 6.90 | 2.42 |
| `region` | 24 | 1.55 | 1.14 |
| `related_person` | 94 | 1.35 | 0.71 |
| `site` | 6 | 4.69 | 4.33 |
| `related_event` | 5 | 4.57 | 4.63 |
| `condition` | 6 | 18.86 | 13.63 |

部分长尾字段采用训练集非空 Token 数中位数作为回归兜底，或由于测试样本很少，
其指标不适合进行强结论。

## 15.7 成本预测结论

当前字段输出 Token 预测的整体结论是：

1. 高频、短文本字段（如 `category`、`material`、`dynasty`）的长度预测误差较小，
   可以用于动态 Schema 的相对成本排序。
2. `form_type`、`usage`、`motif` 等字段的预测误差处于可接受范围，可以作为预算
   估计的一部分，但不应视为精确账单。
3. `desc` 的 MAE 约为 6.90 Token，中位绝对误差约为 2.42 Token。平均误差受少数
   超长描述影响较大，因此预算选择时应为该字段保留安全余量。
4. `condition`、`site`、`related_event` 等长尾字段样本不足，当前长度预测仅能
   作为粗略估计，不能据此作出稳定的字段成本结论。

当前可采用以下动态字段成本估计：

```text
dynamic_field_cost(record, field)
    = field_name_tokens
    + schema_value_tokens
    + predicted_output_tokens(record, field)
```

该估计适合用于：

```text
字段成本排序
预算约束下的 Schema 选择
不同策略的相对成本比较
```

该估计暂不能直接等同于实际 API 账单，因为当前输出 Token 标签来自 Full Schema
结果中的字段值，尚未覆盖不同动态 Schema 对输出长度的交互影响、固定 Prompt
开销和重试开销。预算选择时应增加安全余量，尤其是对 `desc` 等长文本字段。

---

# 16. 后续实验任务与项目最小说明

## 16.1 项目要解决的问题

当前项目面向大规模文化遗产记录的结构化抽取。固定使用完整 Schema 会让每条记录
都处理大量可能为空或价值较低的字段，增加 Schema Token、输出 Token 和 API 成本。

项目希望验证：

```text
根据每条记录的内容选择字段，是否能在降低抽取成本的同时保留主要信息。
```

## 16.2 数据与模型关系

两张 PostgreSQL 表通过唯一的 `artifact_id` 对齐：

```text
artifact_raw.raw_data
    → 原始记录
    → 模型输入 record_text

artifact_extract.*
    → 已完成的 Full Schema 抽取结果
    → present 和字段输出 Token 的弱监督标签
```

`artifact_extract` 是已有的 Full Schema 结果，不是 Adaptive Schema 的结果。

当前模型只负责两个预测任务：

```text
record_text
    → 每个字段的 present 分数
    → 每个字段的预测输出 Token 数
```

当前模型不负责预测：

```text
记录复杂度
预算等级
原始记录输入 Token 数
```

当前模型文件位于：

```text
models/field_selector/
```

当前模型采用共享字符级 TF-IDF 编码，并连接字段 present 分类器和字段输出长度回归器。

## 16.3 三种实验方法

后续实验必须在同一批原始记录上比较三种方法：

```text
Full Schema
    artifact_raw.raw_data + 完整 Schema
    已有结果：artifact_extract

Fixed Schema
    artifact_raw.raw_data + 固定字段集合
    需要重新调用 LLM

Adaptive Schema
    artifact_raw.raw_data + 模型选择的动态字段集合
    需要重新调用 LLM
```

其中，Full Schema 作为参考基准；Fixed Schema 和 Adaptive Schema 必须使用相同的
原始输入、相同的 LLM、相同的提示词格式、相同的输出格式和相同的重试策略。

## 16.4 下一步实验任务

### 任务一：确定实验样本

从已有 6763 条对齐记录中固定抽取一批实验样本，建议先使用测试集中的 `100～300`
条记录。每种方法必须处理完全相同的 `artifact_id` 集合。

### 任务二：确定 Fixed Schema 基线

使用字段价值权重和全局字段统计确定固定字段集合，建议至少测试：

```text
Fixed-2
Fixed-5
Fixed-10
```

也可以使用固定成本版本替代固定字段数量，但所有 Fixed 配置必须提前固定，不能
根据测试结果临时修改。

### 任务三：执行 Adaptive Schema 选择

对每条实验记录执行：

```text
原始 JSON
    → 字段 present 分数
    → 字段输出 Token 预测
    → 字段价值权重
    → Schema 成本计算
    → 预算约束选择
    → 动态 Schema
```

建议先测试预算：

```text
100 / 200 / 400 / 800
```

其中 `800` 作为接近完整 Schema 的高预算参考，`100` 作为低预算压力测试。

### 任务四：重新调用 LLM

对同一批原始记录分别执行 Fixed Schema 和 Adaptive Schema 抽取。每次调用至少保存：

```text
artifact_id
method
budget
selected_fields
schema_definition
extraction_result
input_tokens
output_tokens
total_cost
latency
retry_count
error
```

Full Schema 结果也应转换为相同的记录格式，便于统一评估。

### 任务五：进行质量和成本评估

如果暂时没有人工 Gold 标注，使用 `artifact_extract` 作为 Full Schema 参考，并将
质量指标命名为：

```text
Full Schema agreement
```

不能把该指标直接称为真实抽取准确率。

主要质量指标：

```text
字段级 Precision、Recall、F1
Full Schema 非空字段覆盖率
选中字段中与 Full Schema 一致的比例
```

主要成本指标：

```text
实际输入 Token
实际输出 Token
总 Token
API 成本
平均延迟
重试次数
```

主要比较方式：

```text
相同预算下：Adaptive 是否比 Fixed 覆盖更多 Full Schema 字段
相同覆盖率下：Adaptive 是否比 Fixed 消耗更少 Token 和费用
```

### 任务六：检查预测成本误差

将模型预测的字段输出 Token 与动态 Schema 实际调用的输出 Token 比较：

```text
预测输出 Token
    vs.
实际输出 Token
```

报告每个字段和每个预算档位的 MAE、相对误差和高估/低估比例，重点检查 `desc`、
`condition` 等长度变化较大的字段。

## 16.5 实验结果文件

建议将每次 LLM 调用结果保存为 JSONL：

```text
experiments/extraction_results.jsonl
```

统一评估结果保存为：

```text
reports/extraction_quality_metrics.json
reports/extraction_cost_metrics.json
reports/quality_cost_curve.csv
```

所有实验都应保留：

```text
模型版本
字段权重文件
预算值
实验样本 ID
LLM 和 Prompt 配置
```

## 16.6 当前阶段完成标准

本阶段完成的最低标准是：

1. 同一批原始记录完成 Full、Fixed 和 Adaptive 三类结果对比。
2. Adaptive 在每个预算档位都生成不超过预算的字段 Schema。
3. 保存每次调用的实际 Token、费用、延迟和结果。
4. 输出质量、成本和 Full Schema agreement 指标。
5. 绘制预算与字段覆盖率、实际成本之间的关系。

完成以上实验后，才能判断当前方法是否真正带来质量-成本收益；在此之前，离线
回放结果只能说明字段选择器的模拟效果，不能代表真实 LLM 抽取效果。
