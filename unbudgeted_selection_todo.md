# 无硬预算条件下的自适应字段选择待办

## 1. 目标

当前预算选择器要求预先指定 Token 预算，例如 `100`、`200` 或 `400`。本任务研究
另一种使用方式：不指定硬预算，而是让系统根据字段的可抽取性、任务价值和预期成本
自动决定是否选择字段。

目标不是简单设置统一概率 cutoff，而是尽量以较低成本达到接近 Full Schema 的字段
覆盖效果。

核心流程：

```text
原始 JSON
    ↓
字段 present 分数
    ↓
字段输出 Token 预测
    ↓
字段价值权重
    ↓
无硬预算的成本感知选择
    ↓
动态 Schema
```

---

## 2. 当前已有基础

当前已经具备：

```text
field_scores
predicted_output_tokens
schema_costs.json
field_weights.json
```

其中：

```text
field_score
    → 该字段在当前记录中可能有 Full Schema 输出的分数

field_value_weight
    → 该字段对当前任务的价值先验

field_cost
    → 字段名 Token + Schema 描述 Token + 预测输出 Token
```

当前固定预算选择器位于：

```text
budget_selector.py
```

现有离线回放结果：

| 预算 | 平均估计成本 | 平均选中字段数 | Full Schema 非空字段覆盖率 |
| ---: | ---: | ---: | ---: |
| 100 | 90.52 | 2.75 | 26.06% |
| 200 | 192.67 | 4.81 | 53.46% |
| 400 | 385.04 | 8.23 | 94.95% |

---

## 3. 推荐方法：成本感知净收益选择

### 3.1 字段效用

对每条记录和每个字段计算预期效用：

```text
utility(record, field)
    = field_score(record, field)
    × field_value_weight(field)
```

它表示该字段在当前记录中可能有内容，同时该字段对任务具有一定价值。

### 3.2 字段成本

字段成本定义为：

```text
cost(record, field)
    = field_name_tokens
    + schema_value_tokens
    + predicted_output_tokens(record, field)
```

该成本暂时不包含原始记录输入 Token，因为同一条原始记录无论选择多少字段，通常都
需要输入一次。原始输入成本应在真实 API 实验中作为总成本单独报告。

### 3.3 净收益

引入成本惩罚系数 `lambda`：

```text
net_gain(record, field)
    = utility(record, field)
    - lambda × cost(record, field)
```

无硬预算时选择：

```text
net_gain(record, field) > 0
```

其中：

```text
lambda 小 → 选择更多字段，效果接近 Full Schema
lambda 大 → 更重视成本，选择更少字段
```

该方法比统一概率 cutoff 更合理，因为它同时考虑：

```text
字段可抽取性
字段任务价值
字段 Schema 成本
字段预期输出长度
```

---

## 4. 推荐的全局选择形式

对于没有硬预算的情况，不应仅对每个字段独立判断。可以将字段集合选择写成：

```text
maximize Σ [utility(record, field) - lambda × cost(record, field)] × x_field

subject to:

x_field ∈ {0, 1}
```

在字段相互独立的近似下，直接选择净收益为正的字段即可；如果以后加入字段互补或
冗余关系，可继续使用组合优化。

为了控制一次选择过多字段，建议实现两种模式：

### 模式 A：净收益模式

```text
选择所有 net_gain > 0 的字段
```

适合研究：

```text
不同 lambda 对成本和覆盖率的影响
```

### 模式 B：目标覆盖率模式

在验证集上搜索最小成本的 `lambda`，使 Full Schema 非空字段覆盖率达到目标：

```text
minimize average_cost
subject to average_coverage >= target_coverage
```

建议目标：

```text
90%
95%
98%
```

推理时使用验证集确定的 `lambda`，不能使用测试集调参。

---

## 5. 核心字段策略

即使没有硬预算，也建议保留核心字段配置：

```text
name
category
```

核心字段可以有两种处理方式：

```text
必选：直接加入动态 Schema
软必选：只要 present_score 超过较低阈值就加入
```

当前建议使用“必选”模式，因为这两个字段是基础数据库字段，且当前数据中非空率
为 100%。

`desc` 不建议默认必选。虽然当前数据中非空率为 100%，但该字段输出长度较长，
应让成本感知选择器根据 `lambda` 决定是否加入。

---

## 6. cutoff 的替代方案

### 不推荐的统一 cutoff

```text
field_score > 0.5
```

问题：

- 不考虑字段价值；
- 不考虑输出长度；
- 不考虑 Schema 描述成本；
- 高成本字段和低成本字段使用相同标准；
- 不能直接产生成本-覆盖率曲线。

### 推荐的成本感知 cutoff

如果需要保留 cutoff 形式，应使用字段和成本相关的动态条件：

```text
field_score × field_value_weight
    > lambda × field_cost
```

它与净收益条件完全等价，但解释更直观：字段的预期价值必须超过其成本惩罚。

---

## 7. 待实现任务

### 任务一：实现无硬预算选择器

新增脚本或扩展现有选择器：

```text
unbudgeted_selector.py
```

[已完成] 已实现净收益选择器，直接选择满足 `utility - lambda × cost > 0` 的字段，
并保留配置中的核心字段。

输入：

```text
模型预测结果
schema_costs.json
field_weights.json
lambda
required_fields
```

输出：

```json
{
  "lambda": 0.01,
  "selected_fields": ["name", "category", "dynasty", "material"],
  "estimated_cost": 180,
  "total_utility": 3.2,
  "total_net_gain": 1.4,
  "dynamic_schema": {}
}
```

### 任务二：实现 lambda 网格搜索

在验证集上测试一组 `lambda`：

```text
0.0001 / 0.0003 / 0.001 / 0.003 / 0.01 / 0.03 / 0.1
```

实际取值应根据字段效用和 Token 成本的数量级调整。

每个 `lambda` 记录：

```text
平均估计成本
平均选中字段数
Full Schema 非空字段覆盖率
加权字段效用
净收益
```

[已完成] 已实现验证集网格搜索，默认测试七个 `lambda` 值并记录成本、字段数、
覆盖率、效用和净收益。

### 任务三：实现目标覆盖率选择

对目标覆盖率执行：

```text
90% / 95% / 98%
```

自动选择验证集上达到目标覆盖率所需的最低平均成本对应的 `lambda`。

[已完成] 当前验证集结果中，`lambda=0.003` 同时满足 90%、95% 和 98% 平均覆盖率
目标，验证集平均估计成本约为 426.68 Token。

保存配置：

```text
unbudgeted_policy.json
```

示例：

```json
{
  "policies": {
    "coverage_90": {"lambda": 0.01},
    "coverage_95": {"lambda": 0.005},
    "coverage_98": {"lambda": 0.001}
  }
}
```

### 任务四：测试集冻结评估

使用验证集确定 `lambda` 后，在测试集上只进行一次最终评估。

禁止根据测试集结果再次修改：

```text
lambda
字段权重
required_fields
```

[已完成] 使用验证集确定的 `lambda=0.003` 在测试集上完成冻结评估：平均估计成本
约 425.71 Token，平均选中 8.71 个字段，Full Schema 非空字段覆盖率约 98.998%。

### 任务五：与固定预算方法比较

比较：

```text
Fixed Schema
Fixed Budget Schema
Unbudgeted lambda Schema
```

主要指标：

```text
平均实际或估计成本
Full Schema 字段覆盖率
加权字段效用
平均选中字段数
质量-成本曲线
```

[部分完成] 已生成无硬预算策略的离线结果；与固定预算方法的统一质量-成本曲线仍待
整理。

### 任务六：真实 LLM 验证

离线选择结果确认后，对同一批原始记录重新调用 LLM，比较：

```text
Full Schema
Fixed Schema
Fixed Budget Adaptive Schema
Unbudgeted Adaptive Schema
```

必须保存实际：

```text
input_tokens
output_tokens
latency
retry_count
extraction_result
```

[待完成] 尚未执行真实 LLM 抽取。

---

## 8. 论文中的方法表述

建议将该方法称为：

```text
Cost-aware Utility Maximization for Unbudgeted Schema Selection
```

中文可表述为：

```text
无硬预算条件下的成本感知效用最大化 Schema 选择
```

论文中应明确：

```text
field_score 提供记录级可抽取性估计；
field_value_weight 提供任务相关价值先验；
predicted_output_tokens 提供条件输出成本估计；
lambda 控制质量与成本之间的权衡。
```

因此，该方法不是声称自动发现了字段的客观价值，而是使用任务相关价值先验完成
成本感知的字段选择。

推荐核心公式：

```text
U(r, f) = P(f is extractable | r) × V(f)

C(r, f) = SchemaCost(f) + E[OutputTokens | r, f]

S_r = {f | U(r, f) - lambda × C(r, f) > 0}
```

---

## 9. 当前状态

### 已完成

- [x] 字段 present 分数预测；
- [x] 字段输出 Token 数预测；
- [x] 字段价值权重配置；
- [x] 固定预算 0-1 背包选择；
- [x] 100、200、400 Token 离线回放。

### 待完成

- [x] 无硬预算净收益选择器；
- [x] `lambda` 验证集搜索；
- [x] 90%、95%、98% 目标覆盖率策略；
- [x] 无硬预算方法的测试集冻结评估；
- [ ] 与固定预算方法的质量-成本曲线比较；
- [ ] 无硬预算 Adaptive Schema 的真实 LLM 抽取；
- [ ] 预测成本与实际 API 成本误差分析。

## 10. 当前结论边界

在无硬预算选择器和真实 LLM 实验完成前，只能认为该方法是一个待验证的选择策略。

现有 100、200、400 Token 结果属于固定预算离线回放，不能直接作为无硬预算策略的
结果，也不能证明真实 LLM 抽取质量已经接近 Full Schema。
