# n=100 动态 Schema + stable Prompt 异常实验交接

## 1. 交接结论

本轮实验用于核查 `1_stable.prompt` 在动态裁剪 Schema 场景下的行为。原始字符模型和 M1

当前最稳妥的结论是：

> `1_stable.prompt` 与动态裁剪后的 Schema 组合存在严重异常。异常同时出现在原始模型和
> M1 中，暂不能归因于 M1 的特征工程、训练样本选择或字段选择结果。

这些结果是**诊断实验**，不是动态 Schema 性能结论，也不应与此前正常的 FullSchema 结果

## 2. 实验协议

| 项目 | 设置 |
| --- | --- |
| 测试数据 | 固定 `data/gold/gold_main_ids.json`，500 条 |
| Gold 评估 | `data/gold/agent_tasks/gold_main.jsonl` |
| LLM | `kg-assist-pro`（由 `.env` 的 `LLM_MODEL` 配置） |
| Temperature | `0`（由 `.env` 的 `LLM_TEMPERATURE` 配置） |
| Prompt | `1_stable.prompt` |
| Schema 选择 | `budget-200` |
| 训练规模 | `n=100` |
| 训练 seed | `42/43/44` |
| 并发与重试 | 并发 20、队列 40、最多重试 2 次 |
| 每组请求数 | 500 |

模型和测试集：

- 原始字符模型：`models/training_size_ablation/100/seed_<seed>`
- M1 字符+词级模型：`models/model_strategy_ablation/m1_by_size/100/seed_<seed>`
- 统一入口：`llm_runner.py`

## 3. 结果文件

### 3.1 原始字符模型

结果：

```text
experiments/n100_char_stable_budget200_seed42_gold_main_v3_pro.jsonl
experiments/n100_char_stable_budget200_seed43_gold_main_v3_pro.jsonl
experiments/n100_char_stable_budget200_seed44_gold_main_v3_pro.jsonl
```

评估报告：

```text
reports/n100_char_stable_budget200_gold_main_v3_pro_metrics.json
```

三 seed 结果：

| Seed | 输入 Token | 输出 Token | 总 Token | Presence Micro-F1 | Exact Value Recall |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 42 | 318.2 | 44.2 | 362.4 | 0.0248 | 0.0094 |
| 43 | 318.1 | 44.0 | 362.2 | 0.0254 | 0.0091 |
| 44 | 318.1 | 44.2 | 362.3 | 0.0261 | 0.0094 |

seed=42 中仅 15/500 条记录出现任意非空值；其余记录基本为全字段 `null`。

### 3.2 M1 字符+词级模型

结果：

```text
experiments/n100_m1_stable_budget200_seed42_gold_main_v3_pro.jsonl
experiments/n100_m1_stable_budget200_seed43_gold_main_v3_pro.jsonl
experiments/n100_m1_stable_budget200_seed44_gold_main_v3_pro.jsonl
```

评估报告：

```text
reports/n100_m1_stable_budget200_gold_main_v3_pro_metrics.json
```

三 seed 结果均接近：

| 指标 | 三 seed 约值 |
| --- | ---: |
| 输入 Token | 316--318 |
| 输出 Token | 43.5--43.7 |
| 总 Token | 360--361 |
| Presence Micro-F1 | 约 0.025 |
| Exact Value Recall | 约 0.009 |

M1 seed=42 中仅 13/500 条记录出现任意非空值。三组均为 500/500 success，没有 API 最终
失败或评估 invalid row。

## 4. 典型 Prompt

本轮实际使用 `1_stable.prompt`。动态 Schema 的一条样例只包含 7 个字段：

```text
目标字段名称：
[
  "name",
  "category",
  "form_type",
  "dynasty",
  "period",
  "material",
  "motif"
]

重点字段说明：
{
  "category": "文物类别",
  "form_type": "器型或形制",
  "dynasty": "朝代或政权",
  "material": "主要材质"
}
```

Prompt 的规则包含：

```text
只能根据源数据中明确出现或可直接归纳的信息填写字段
不得使用外部知识或领域常识补全
没有明确可靠依据时输出 JSON null
```

原始数据则使用中文键名，例如 `文物名称`、`类型`、`年代`、`详细描述`，目标字段使用英文
键名，例如 `name`、`category`、`dynasty`。Prompt 没有显式给出这些中文源字段到英文目标
字段的映射。

## 5. 已排除因素

- 原始模型和 M1 都出现相同现象，降低了“M1 模型产物损坏”的可能性。
- 每组均为 500 条成功响应，`invalid_rows=0`，不是 API 请求失败造成的低分。
- 原始响应本身就是合法 JSON，且字段值多数为 `null`，不是评估解析错误。
- M1 与原始模型均使用 `budget-200`，选择字段数量约 7--8 个，说明不是某一个模型单独选出空 Schema。
- 三个训练 seed 的结果都一致，不能解释为单个 seed 的抽样波动。

## 6. 与历史 FullSchema 的关系

历史 FullSchema 实验使用同一个 `1_stable.prompt`，但传入完整 18 字段 Schema，结果正常。
这说明不能简单地说 `1_stable.prompt` 在所有情况下都会输出全 `null`。

当前观察到的差异是：

```text
完整 18 字段 Schema + 1_stable.prompt -> 历史结果正常
动态 7--8 字段 Schema + 1_stable.prompt -> 原始模型和 M1 均近乎全空
```

因此候选原因包括：

1. LLM 服务端或 Prompt 模板对动态目标结构有特殊行为；
2. `1_stable.prompt` 的强约束在字段裁剪后导致模型过度保守；
3. 英文目标字段与中文原始字段缺乏显式映射；
4. 动态 Schema 的输入 JSON 结构与 FullSchema 的稳定任务上下文不兼容；
5. 服务端可能存在缓存、模板路由、模型版本或请求参数差异。

### 6.1 服务器重启后的 FullSchema 复核

模型服务器重启后，使用相同的 `1_stable.prompt`、同一批 `gold_main` 前 10 条记录和完整
18 字段 Schema 重新执行了 FullSchema smoke test：

```text
experiments/retry_full_stable_smoke10_v3_pro.jsonl
```

结果为 10/10 API success，但 0/10 记录存在任意非空抽取值；每条原始响应均为完整 18 个字段
的全 `null` JSON。平均输入、输出和总 Token 分别为 375.5、99.0 和 474.5。

这一复核改变了此前“只有动态 Schema 与 stable Prompt 不兼容”的判断。当前证据表明：

```text
服务器重启后：
完整 18 字段 Schema + 1_stable.prompt -> 10/10 全空
动态 7--8 字段 Schema + 1_stable.prompt -> 10/10 全空
```

因此，优先排查 LLM 服务端实际模型版本、部署状态、隐藏 system prompt、模板路由和请求参数。
不能再将异常主要归因于字段选择器或动态 Schema。

## 7. 复现命令

原始字符模型示例：

```bash
.venv/bin/python llm_runner.py \
  --methods budget-200 \
  --limit 0 \
  --ids-file data/gold/gold_main_ids.json \
  --output experiments/n100_char_stable_budget200_seed42_gold_main_v3_pro.jsonl \
  --prompt 1_stable.prompt \
  --model-dir models/training_size_ablation/100/seed_42 \
  --experiment-id n100-char-stable-budget200-v3-pro \
  --run-label n100-char-stable-budget200-seed42 \
  --max-concurrency 20 \
  --queue-size 40 \
  --max-retries 2
```

M1 只需将 `--model-dir` 和输出文件、实验标签替换为：

```text
models/model_strategy_ablation/m1_by_size/100/seed_42
n100_m1_stable_budget200_seed42_gold_main_v3_pro.jsonl
n100-m1-stable-budget200-v3-pro
n100-m1-stable-budget200-seed42
```

评估命令示例：

```bash
.venv/bin/python evaluate_llm_experiments.py \
  --input experiments/n100_char_stable_budget200_seed42_gold_main_v3_pro.jsonl \
           experiments/n100_char_stable_budget200_seed43_gold_main_v3_pro.jsonl \
           experiments/n100_char_stable_budget200_seed44_gold_main_v3_pro.jsonl \
  --gold data/gold/agent_tasks/gold_main.jsonl \
  --output reports/n100_char_stable_budget200_gold_main_v3_pro_metrics.json
```

## 8. 后续核查建议

在重新消耗 500 条 API 请求前，先做以下小规模检查：

1. 固定同一条原始记录，分别发送完整 18 字段 Schema 和裁剪 7--8 字段 Schema；保存完整请求
   messages 和响应。
2. 直接在服务端或模型网关日志中确认当前实际加载的模型版本、权重和启动时间。
3. 使用完全相同的任务规则，分别测试：完整字段定义、字段名列表、显式中英文映射三种 Prompt。
4. 核对服务端实际模型版本、请求 body、temperature、system message、默认 max tokens 和
   是否存在缓存。
5. 检查响应中的 usage 是否由服务端真实返回，不要只依据客户端估算。
6. 先跑 5--10 条 smoke test；只有 FullSchema 和动态 Schema 均恢复非空抽取后，才重新跑
   完整 500 条。
7. 诊断期间不要覆盖本交接中两个结果文件组；新模板和新实验使用新的文件名及独立目录。

## 9. 验证状态

- 原始模型：3/3 组完成，每组 500/500 success。
- M1 模型：3/3 组完成，每组 500/500 success。
- 两份评估报告：`invalid_rows=0`，`gold_records=500`。
- 两种模型均确认大量全 `null`。
- 服务器重启后的 FullSchema smoke test：10/10 success、10/10 全空。
- 本轮没有修改旧实验结果，没有提交 git。
