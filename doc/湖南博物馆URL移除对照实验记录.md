# 湖南博物馆 URL 移除对照实验记录

## 目的

检验原始记录中的 `来源URL` 是否影响同一湖南博物馆 cohort 的 M1 字段预测、预算
Schema 抽取质量和实际 Token。该实验仅改变输入中的 URL 字段，保留相同的 500 个
`artifact_id`、Prompt、Schema、服务配置和 100/50/350 切分协议。

## 输入净化与协议

- 原始 cohort：`artifact_id=300000001..300000500`，连续 500 条。
- `without_url` 输入从原始冻结快照逐条派生，仅删除 `来源URL`；500/500 条均删除该字段，
  `source_url` 和 `image_url` 删除计数均为 0。
- 原始快照 SHA256：
  `655584615d81e2293883eef636eb8339196453d0f9e97e2b9aca323483c65f36`。
- 净化快照：`data/hunan_museum_300000001_300000500_without_url/snapshot.jsonl`，SHA256：
  `c00e14d72bd7daf5e116f900bf02d85ed02529423e682309c10433a54d90e761`。
- `without_url` full silver：500/500 success、0 retry、18 字段；stable Prompt、
  `simple2.json`、`kg-assist-pro`、temperature 0、thinking 关闭、c=20/q=40、retry=2。
- 新 silver 单独生成切分与 M1：100 train / 50 validation / 350 held-out，split seed
  `20260917`；Budget-200 和 Budget-400 各运行同一 350 held-out 记录，均 350/350
  success、0 retry。并发仅为执行效率，不作为性能结果。

## With/Without URL 对照

| 指标 | With URL | Without URL | 差异（without - with） |
| --- | ---: | ---: | ---: |
| 原始 M1 Presence Micro-F1 | 0.9295 | 0.9176 | -0.0119 |
| Held-out full 平均 total Token | 535.4371 | 492.6943 | -42.7429 |
| Budget-200 Presence Micro-F1 | 0.6709 | 0.6630 | -0.0079 |
| Budget-200 Presence Macro-F1 | 0.6709 | 0.6658 | -0.0051 |
| Budget-200 Exact Value Recall | 0.5082 | 0.5142 | +0.0060 |
| Budget-200 平均 total Token | 414.4029 | 370.0629 | -44.3400 |
| Budget-400 Presence Micro-F1 | 0.8938 | 0.9147 | +0.0209 |
| Budget-400 Presence Macro-F1 | 0.8786 | 0.9035 | +0.0249 |
| Budget-400 Exact Value Recall | 0.7484 | 0.7267 | -0.0217 |
| Budget-400 平均 total Token | 512.0000 | 469.8171 | -42.1829 |

`without_url` 的 Budget-200 选择 9 字段，平均估计成本 `197.2629`；Budget-400 选择
16 字段，平均估计成本 `393.6371`。对应的实际 Token 为 input/output/total：
Budget-200 `300.0257/70.0371/370.0629`，Budget-400
`347.2914/122.5257/469.8171`。

## 结论与边界

- 删除 `来源URL` 稳定降低所有方法的实际总 Token，每条约减少 42--44 Token；这是明确的
  输入压缩收益。
- 质量影响不是单向的：Budget-400 的 Presence Micro-F1 提高 0.0209，而 Budget-200
  降低 0.0079，原始 M1 Micro-F1 也降低 0.0119。因此不能声称 URL 移除普遍提升质量。
- 两个版本分别以各自 full 输出生成 silver labels；full 不同，质量差异同时包含输入移除对
  模型输出标签的影响，不能解释为只改变了 M1 特征。
- 参考为模型生成的 silver，不是人工 Gold；不据此声称人工事实正确性、跨机构泛化或端到端
  性能提升。

## 产物

- 净化脚本：`benchmarks/sanitize_hunan_without_url.py`。
- without_url full：
  `experiments/hunan_museum_300000001_300000500_without_url_full_silver/formal/full.jsonl`。
- without_url 模型与离线报告：
  `models/hunan_museum_300000001_300000500_without_url_n100_m1/`、
  `reports/hunan_museum_300000001_300000500_without_url_n100_m1/`。
- without_url 正式预算输出：
  `experiments/hunan_museum_300000001_300000500_without_url_budget_quality/formal/`。
