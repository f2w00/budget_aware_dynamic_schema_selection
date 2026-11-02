# Large-Scale Adaptive Schema Budgeting for Structured Record Enrichment

## 1. 项目背景

我们拥有一个大规模文化遗产数据集：

* 约 55 万条文物记录
* 数据来源为文化遗产/文物数据库
* 数据具有明显的信息不完整和异构特点

记录形式不统一：

### 丰富记录

包含：

* 名称
* 年代
* 材质
* 器型
* 纹饰
* 工艺
* 出土地
* 说明文本

例如：

```
明代青花缠枝莲纹盘，景德镇窑制，
胎质细腻，盘心绘缠枝莲纹……
```

### 稀疏记录

可能只有：

```
青花盘
明代
瓷
```

甚至：

```
铜镜
```

---

## 2. 研究问题

传统结构化抽取方法通常假设：

```
Document
+
Fixed Schema
+
LLM
=
Structured Data
```

但是在大规模文化遗产数据中：

1. 每条记录的信息量不同
2. 不同记录需要关注的字段不同
3. 固定 Schema 会产生大量无效计算

例如：

对于：

```
青花盘
```

没有必要让模型考虑：

```
修复历史
收藏来源
铭文
考古层位
```

但是对于：

```
明代景德镇窑青花缠枝莲纹盘，1985年出土……
```

这些字段可能非常重要。

因此提出问题：

> 如何根据单条记录的信息复杂度，动态选择 Schema，并动态分配抽取计算预算？

---

# 3. 核心方法

提出：

## Adaptive Schema Budgeting (ASB)

整体流程：

```
Input Record

      |
      v

Record Complexity Estimation

      |
      v

Schema Budget Allocation

      |
      +----------------+
      |                |
      v                v

Field Selection    Adaptive Top-K

      |
      v

LLM Structured Extraction

      |
      v

Validation / Expansion
```

---

# 4. 方法模块

## 4.1 Record Complexity Estimation

目标：

判断当前记录需要多少抽取资源。

输入特征：

### 文本特征

* 文本长度
* token 数量
* 描述完整度

### 结构化特征

* 已存在字段数量
* 字段缺失比例

### 语义特征

* record embedding
* entity 数量
* 信息密度

定义：

```
Complexity(record)
```

用于后续预算分配。

---

# 5. Dynamic Schema Selection

目标：

从全局 Schema 中选择当前记录需要的字段。

例如：

全局：

```
Artifact Schema:

name
dynasty
period
material
form
decoration
technique
inscription
provenance
excavation
restoration
collector
...
```

对于简单记录：

```
青花盘
```

选择：

```
name
dynasty
material
form
```

对于复杂记录：

选择：

```
name
dynasty
material
form
decoration
technique
provenance
```

---

字段相关性计算：

```
relevance(field, record)

=
embedding_similarity
+
historical
```
