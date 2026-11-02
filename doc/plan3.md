# Budget-aware Adaptive Schema Selection for Large-Scale Cultural Heritage Extraction

## 1. 项目概述

本项目研究面向大规模文化遗产数据的高效结构化抽取方法。

当前拥有约 55 万条文物记录，数据存在以下特点：

* 数据规模大，需要低成本自动处理
* 记录信息完整程度差异明显
* 部分记录包含丰富描述信息
* 部分记录只有基础属性
* 不同记录对结构化字段的需求不同

传统结构化抽取方法通常采用固定 Schema：

```
Record

+

Complete Schema

+

LLM Extraction

↓

Structured Data
```

该方法虽然能够获得较完整的信息，但存在：

* 大量无关字段参与推理
* 输入 Token 浪费
* 抽取成本增加
* 不适合大规模数据处理

因此，本项目提出一种：

**Budget-aware Adaptive Schema Selection Framework**

通过学习字段价值，并结合预算约束动态选择 Schema，实现低成本、高质量的大规模结构化抽取。

---

# 2. 核心研究问题

对于一条文化遗产记录：

已知：

* 候选字段集合
* 字段处理成本
* 当前抽取预算

如何选择最有价值的字段，使：

* 信息质量最大化
* 抽取成本最小化

形式化为：

```
Maximum Information Utility

under Limited Extraction Budget
```

---

# 3. 总体流程

系统流程：

```
55万文物记录

        |

        v

Full Schema Extraction
(生成监督数据)

        |

        v

Field Utility Learning

        |

        +----------------+
        |                |
        v                v

Field Importance     Field Cost Estimation

        |                |

        +----------------+

                |

                v

Budget-aware Schema Selection

                |

                v

Dynamic Schema

                |

                v

LLM Structured Extraction

                |

                v

Structured Cultural Heritage Dataset
```

---

# 4. 方法组成

## 4.1 Full Schema Extraction（数据构造阶段）

首先使用完整领域 Schema 对部分数据进行完整抽取。

目的：

生成字段级监督信号。

输入：

```
文物记录
+
完整 Schema
```

输出：

```
完整结构化结果
```

例如：

```json
{
"name":"青花盘",
"dynasty":"明",
"material":"瓷",
"form":"盘",
"decoration":"缠枝莲纹",
"provenance":null
}
```

该结果用于后续字段价值学习。

---

# 5. Field Utility Learning

## 5.1 目标

学习：

> 对于当前记录，每个字段的重要程度。

输入：

```
Record

+

Candidate Field
```

输出：

```
Field Utility Score
```

例如：

输入：

```
明代青花缠枝莲纹盘
```

输出：

```
name              0.99
dynasty           0.95
material          0.92
decoration        0.88
provenance        0.20
collector         0.05
```

---

## 5.2 训练数据

利用已有完整 Schema 抽取结果自动生成弱监督标签。

字段标签来源：

* 字段是否成功抽取
* 字段信息丰富程度
* 字段有效性评价

形成：

```
(record, field, utility)
```

训练样本。

---

# 6. Field Cost Estimation

## 6.1 目标

估计每个字段加入 Schema 后产生的成本。

成本包括：

* 字段描述 token
* 输出 token
* 推理开销

例如：

| 字段                 | 成本         |
| ------------------ | ---------- |
| name               | 10 tokens  |
| material           | 20 tokens  |
| form               | 30 tokens  |
| decoration         | 80 tokens  |
| historical_context | 200 tokens |

---

# 7. Budget-aware Schema Selection

## 7.1 输入

当前记录：

```
Field Utility Scores

+

Field Costs

+

Budget B
```

---

## 7.2 目标

选择：

价值最高的字段组合。

优化目标：

```
Maximize:

Selected Field Utility


Subject to:

Selected Field Cost <= Budget
```

即：

在有限 Token / 计算预算下获得最大信息收益。

---

## 7.3 输出

动态 Schema。

例如：

低预算：

```
name
dynasty
material
form
```

高预算：

```
name
dynasty
material
form
decoration
technique
historical_context
```

---

# 8. Budget 定义

Budget 表示：

> 当前记录允许使用的最大 Schema 相关资源。

主要形式：

## Token Budget

例如：

```
B = 200 tokens
```

表示：

选择字段后：

Schema 总成本不能超过 200 tokens。

---

# 9. 最终抽取流程

动态 Schema 生成后：

输入：

```
Original Record

+

Adaptive Schema
```

进入 LLM。

输出：

```
Structured JSON
```

用于：

* 文物数据库建设
* 知识库构建
* 后续检索分析

---

# 10. 实验设计

## 10.1 数据

数据：

```
550,000 Cultural Heritage Records
```

特点：

* 大规模
* 字段稀疏
* 信息密度不同

---

# 10.2 Baseline

## Baseline 1: Full Schema Extraction

所有字段。

特点：

质量高，成本高。

---

## Baseline 2: Fixed Top-K Schema

固定选择字段数量。

例如：

Top-5

Top-10

---

## Baseline 3: Random Schema

随机字段选择。

---

## Proposed Method

Budget-aware Adaptive Schema Selection。

---

# 11. 评价指标

## 质量指标

* Field-level Precision
* Field-level Recall
* F1 Score
* Completeness

## 成本指标

* Input Tokens
* Output Tokens
* API Cost
* Processing Time

## 综合指标

Quality-Cost Tradeoff。

---

# 12. 预期贡献

## Contribution 1

提出一种面向大规模文化遗产数据的预算感知 Schema 自适应抽取框架。

## Contribution 2

利用完整 Schema 抽取结果构造弱监督数据，学习字段级价值。

## Contribution 3

将 Schema 选择转化为预算约束优化问题，实现成本和质量之间的平衡。

## Contribution 4

在 55 万级真实文化遗产数据上验证方法有效性。

---

# 13. 当前开发优先级

## 第一阶段

完成：

* Schema 定义
* Full Schema 抽取
* 字段统计分析

---

## 第二阶段

完成：

* Field Utility 数据生成
* 字段价值模型训练
* 字段成本统计

---

## 第三阶段

完成：

* Budget-aware Selection
* 动态 Schema 生成
* LLM 抽取实验

---

## 第四阶段

完成：

* Baseline 对比
* 成本分析
* 论文实验图表

---

# 14. 一句话总结

本项目提出一种预算感知的动态 Schema 选择方法，通过学习每条文化遗产记录中字段的重要性，并结合字段成本约束，在有限抽取预算下自动生成最优 Schema，从而降低大规模 LLM 结构化抽取成本，同时保持数据质量。
