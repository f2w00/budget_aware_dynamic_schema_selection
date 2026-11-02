# Adaptive Schema Budgeting for Large-Scale Cultural Heritage Data Structured Extraction

## 1. 项目背景

本项目面向大规模文化遗产数据的自动结构化处理任务。

当前拥有约 55 万条文物记录，数据来源复杂，存在以下特点：

* 数据规模大，需要高效处理
* 记录信息完整程度差异明显
* 部分记录包含详细描述文本
* 部分记录只有简单名称或基础属性
* 不同文物记录需要关注的信息字段不同

例如：

简单记录：

```
青花盘
明代
瓷
```

复杂记录：

```
明代景德镇窑青花缠枝莲纹盘，
胎质细腻，盘心绘缠枝莲纹，
具有明确时代、工艺和纹饰特征……
```

如果所有记录使用同一个完整领域 Schema 进行抽取，会造成：

* 大量无关字段参与推理
* 输入 Token 增加
* 抽取成本增加
* 简单记录被过度处理

因此，需要一种能够根据记录特点动态调整抽取规模的方法。

---

# 2. 研究目标

提出一种：

**Adaptive Schema Budgeting Framework**

用于解决：

> 在大规模异构文化遗产数据结构化过程中，如何根据单条记录的信息复杂度动态分配 Schema 和抽取资源。

核心目标：

1. 根据记录特点判断其处理需求
2. 动态决定需要使用的 Schema 范围
3. 动态控制字段数量
4. 降低结构化抽取成本
5. 保持较高的数据质量

---

# 3. 核心思想

传统结构化抽取流程：

```
Record

+

Fixed Schema

+

LLM

↓

Structured Data
```

问题：

所有记录使用相同 Schema。

例如：

对于：

```
铜镜
```

没有必要询问：

```
收藏来源
修复历史
铭文内容
考古层位
```

但是对于：

```
汉代出土铜镜，带铭文，具有云雷纹装饰
```

这些字段可能非常重要。

---

本项目提出：

```
Record

↓

理解记录复杂度

↓

动态 Schema Budget

↓

选择适合字段

↓

结构化抽取

↓

标准数据
```

核心变化：

从：

```
One Schema for All Records
```

变成：

```
One Record → One Adaptive Schema
```

---

# 4. 系统总体架构

整体流程：

```
                    Cultural Heritage Records
                              |
                              v
                 Lightweight Record Analyzer
                              |
                              v
                 Schema Budget Controller
                              |
              +---------------+---------------+
              |                               |
              v                               v
     Dynamic Field Selection          Adaptive Top-K Control
              |                               |
              +---------------+---------------+
                              |
                              v
                 Structured Information Extraction
                              |
                              v
                 Validation and Standardization
                              |
                              v
                 Cultural Heritage Knowledge Data
```

---

# 5. Lightweight Record Analyzer

## 5.1 作用

负责以低成本方式理解输入记录。

目标：

判断当前记录：

* 信息丰富程度
* 数据复杂程度
* 可能需要的抽取范围

---

## 5.2 为什么需要轻量分析

如果每条记录首先调用大模型或复杂语义检索：

```
550000 records

↓

550000 次复杂分析
```

本身会产生大量成本。

因此，本项目首先使用轻量级分析模块：

```
Record

↓

Cheap Feature Analysis

↓

Budget Decision
```

使整个流程能够扩展到几十万甚至百万级数据。

---

# 6. Schema Budget Controller（核心模块）

## 6.1 功能

Schema Budget Controller 是整个系统的核心决策模块。

它负责：

* 判断当前记录应该投入多少抽取资源
* 决定 Schema 使用规模
* 控制后续字段选择范围

---

## 6.2 与传统方法区别

传统：

```
所有记录

↓

完整 Schema

↓

LLM Extraction
```

本方法：

```
简单记录

↓

小 Schema

↓

快速抽取


复杂记录

↓

大 Schema

↓

完整抽取
```

---

## 6.3 设计目标

实现：

```
Quality - Cost Balance
```

即：

在保持数据质量的情况下：

* 减少无效字段
* 降低 Token 消耗
* 提高处理速度

---

# 7. Dynamic Schema Selection

## 7.1 功能

从完整领域 Schema 中选择当前记录相关字段。

例如：

完整 Schema：

```
Artifact

├── Name
├── Dynasty
├── Material
├── Form
├── Decoration
├── Technique
├── Provenance
├── Excavation
├── Restoration
├── Collector
└── Historical Context
```

对于简单记录：

```
青花盘
```

可能选择：

```
Name
Dynasty
Material
Form
```

对于复杂记录：

```
明代青花缠枝莲纹盘……
```

可能选择：

```
Name
Dynasty
Material
Form
Decoration
Technique
Historical Context
```

---

# 8. Budget-aware Field Selection

## 8.1 核心思想

字段选择不仅考虑：

```
字段是否相关
```

还考虑：

```
字段处理成本
```

因此目标不是简单选择最多字段。

而是在有限预算下：

选择收益最高的字段组合。

---

## 8.2 字段选择目标

希望达到：

```
Maximum Information Gain

under Limited Budget
```

即：

```
有限计算资源

↓

获得最大结构化信息
```

---

# 9. Adaptive Top-K Schema Control

## 9.1 问题

固定 Top-K：

例如：

```
每条记录选择 Top-10 字段
```

存在问题：

简单记录：

```
10 个字段过多
```

复杂记录：

```
10 个字段不足
```

---

## 9.2 动态控制

本项目根据：

* 记录复杂度
* 字段需求
* 当前预算

动态决定：

```
K = number of selected fields
```

结果：

```
简单记录:

K 小


复杂记录:

K 大
```

---

# 10. Structured Extraction

动态 Schema 生成后：

输入：

```
Original Record

+

Adaptive Schema
```

进入结构化抽取模型。

输出：

统一结构：

```json
{
"name":"",
"dynasty":"",
"material":"",
"decoration":""
}
```

用于：

* 文物数据库建设
* 知识图谱构建
* 检索增强系统
* 文化遗产研究分析

---

# 11. Validation and Feedback

抽取结果需要进一步验证。

包括：

* 字段完整性
* 格式正确性
* 信息一致性

验证结果可以反馈：

* 优化 Schema 选择
* 调整预算策略
* 改进抽取流程

---

# 12. 实验设计

## 12.1 数据集

主要数据：

```
550,000 Cultural Heritage Records
```

特点：

* 大规模
* 信息稀疏
* 字段缺失严重

---

# 13. Baseline 对比

## Baseline 1

Full Schema Extraction

```
所有字段

↓

LLM

↓

结构化结果
```

---

## Baseline 2

Fixed Top-K

```
固定字段数量

↓

抽取
```

---

## Baseline 3

Schema Retrieval / Pruning 方法

用于比较已有动态 Schema 方法。

---

## Proposed Method

Adaptive Schema Budgeting

```
Record Analysis

↓

Budget Controller

↓

Dynamic Schema

↓

Extraction
```

---

# 14. 评价指标

## 数据质量

* Field-level F1
* Completeness
* Extraction Accuracy

## 成本指标

* Input Tokens
* Output Tokens
* Latency
* API Cost

## 扩展能力

测试：

```
10K
50K
100K
300K
550K
```

观察：

* 处理速度
* 成本增长
* 系统稳定性

---

# 15. 论文贡献总结

可能贡献：

## Contribution 1

提出一种面向大规模文化遗产数据的 Adaptive Schema Budgeting 框架。

## Contribution 2

提出轻量级 Record-aware Schema Budget Controller，在结构化抽取前进行低成本决策。

## Contribution 3

提出 Budget-aware Field Selection 和 Adaptive Top-K Schema Control，实现动态字段选择。

## Contribution 4

在 55 万级文化遗产数据上验证方法的质量和扩展能力。

---

# 16. 当前实现优先级

第一阶段：

完成：

* 数据整理
* Schema 定义
* Baseline 抽取流程

第二阶段：

实现：

* Lightweight Analyzer
* Budget Controller
* Dynamic Schema Selection

第三阶段：

实验：

* Quality comparison
* Cost reduction
* Scalability evaluation

---

# 项目一句话总结

本项目提出一种轻量级、自适应的 Schema Budgeting 方法，使大规模文化遗产记录能够根据自身信息复杂度动态选择抽取字段，在保持结构化质量的同时降低大规模 LLM 数据抽取成本。
