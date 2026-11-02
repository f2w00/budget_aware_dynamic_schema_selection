"""汇总不同实验批次的质量-Token 指标，保留样本口径和来源说明。"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).parent


def load(path: str) -> dict[str, Any]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def mean(rows: list[dict[str, Any]], key: str) -> float | None:
    values = [row[key] for row in rows if row.get(key) is not None]
    return sum(values) / len(values) if values else None


def row(
    name: str,
    source: str,
    sample_scope: str,
    method_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    fields = {
        "method": name,
        "source": source,
        "sample_scope": sample_scope,
        "runs": len(method_rows),
        "records_per_run": mean(method_rows, "records"),
    }
    for key in (
        "average_presence_f1",
        "presence_micro_f1",
        "average_exact_value_recall",
        "average_selected_gold_coverage",
        "average_input_tokens",
        "average_output_tokens",
        "average_total_tokens",
        "average_estimated_field_cost",
    ):
        fields[key] = mean(method_rows, key)
    return fields


def main() -> None:
    n100_budget = load(
        "reports/n100_m1_budget_gold_main_v3_pro_metrics.json"
    )["methods"]
    m1_stable = load(
        "reports/kg_assist_pro_n100_m1_stable_budget200_gold_main_v3_pro_metrics.json"
    )["methods"]
    m1_aligned = load(
        "reports/kg_assist_pro_n100_m1_stable_budget200_aligned476_metrics.json"
    )["methods"]
    m1_stable_extra = load(
        "reports/kg_assist_pro_n100_m1_stable_lambda_budget400_gold_main_v3_pro_metrics.json"
    )["methods"]
    m1_aligned_extra = {
        "lambda": load(
            "reports/kg_assist_pro_n100_m1_stable_lambda_aligned476_metrics.json"
        )["methods"],
        "budget400": load(
            "reports/kg_assist_pro_n100_m1_stable_budget400_aligned476_metrics.json"
        )["methods"],
    }
    old_internal = load(
        "reports/gold_main_v3_stable_pro_all_methods_paired_metrics.json"
    )["methods"]
    schemarag_k10 = load(
        "reports/external_schemarag_k10_gold_main_v3_pro_metrics.json"
    )["methods"]["external-schemarag-k10"]
    schemarag_k15 = load(
        "reports/external_schemarag_k15_gold_main_v3_pro_metrics.json"
    )["methods"]["external-schemarag-k15"]

    rows = []
    for budget in (200, 400):
        runs = [
            n100_budget[f"n100-m1-budget{budget}-seed{seed}"]
            for seed in (42, 43, 44)
        ]
        rows.append(
            row(
                f"M1 n=100 + budget-{budget}",
                "reports/n100_m1_budget_gold_main_v3_pro_metrics.json",
                "gold_main 500条；3个seed均为完整500条",
                runs,
            )
        )

    stable_runs = [
        m1_stable[
            f"kg-assist-pro-n100-m1-stable-budget200-seed{seed}"
        ]
        for seed in (42, 43, 44)
    ]
    rows.append(
        row(
            "M1 n=100 + budget-200 (stable Prompt)",
            "reports/kg_assist_pro_n100_m1_stable_budget200_gold_main_v3_pro_metrics.json",
            "gold_main 500条；3个seed均为完整500条；kg-assist-pro",
            stable_runs,
        )
    )
    aligned_runs = [
        m1_aligned[
            f"kg-assist-pro-n100-m1-stable-budget200-seed{seed}"
        ]
        for seed in (42, 43, 44)
    ]
    rows.append(
        row(
            "M1 n=100 + budget-200 (stable Prompt, aligned476)",
            "reports/kg_assist_pro_n100_m1_stable_budget200_aligned476_metrics.json",
            "与历史八方法共同成功 artifact_id 对齐的476条；3个seed",
            aligned_runs,
        )
    )
    for kind, label, report_name in (
        ("lambda", "lambda=-0.003", "lambda"),
        ("budget400", "budget-400", "budget400"),
    ):
        runs = [
            m1_stable_extra[
                f"kg-assist-pro-n100-m1-stable-{kind.replace('budget400', 'budget400')}-seed{seed}"
            ]
            for seed in (42, 43, 44)
        ]
        rows.append(
            row(
                f"M1 n=100 + {label} (stable Prompt)",
                "reports/kg_assist_pro_n100_m1_stable_lambda_budget400_gold_main_v3_pro_metrics.json",
                "gold_main 500条；3个seed均为完整500条；kg-assist-pro",
                runs,
            )
        )
        aligned_methods = m1_aligned_extra[kind]
        aligned_runs = [
            aligned_methods[
                f"kg-assist-pro-n100-m1-stable-{kind}-seed{seed}"
            ]
            for seed in (42, 43, 44)
        ]
        rows.append(
            row(
                f"M1 n=100 + {label} (stable Prompt, aligned476)",
                f"reports/kg_assist_pro_n100_m1_stable_{report_name}_aligned476_metrics.json",
                "与历史八方法共同成功 artifact_id 对齐的476条；3个seed",
                aligned_runs,
            )
        )

    for name, label in (
        ("full", "FullSchema"),
        ("budget-200", "历史内部 budget-200"),
        ("budget-400", "历史内部 budget-400"),
        ("lambda-0.003", "历史内部 lambda=-0.003"),
    ):
        rows.append(
            row(
                label,
                "reports/gold_main_v3_stable_pro_all_methods_paired_metrics.json",
                "gold_main 500条中的七方法共同成功476条",
                [old_internal[name]],
            )
        )

    for name, metrics in (
        ("SchemaRAG k=10（外部重实现）", schemarag_k10),
        ("SchemaRAG k=15（外部重实现）", schemarag_k15),
    ):
        rows.append(
            row(
                name,
                "对应 external_schemarag_*_gold_main_v3_pro_metrics.json",
                "gold_main 500条；独立外部重实现",
                [metrics],
            )
        )

    output = ROOT / "reports/quality_token_comparison.json"
    output.write_text(
        json.dumps(
            {
                "description": "质量-Token 跨实验批次对照；不同 sample_scope 不作严格显著性比较",
                "metric_notes": {
                    "average_presence_f1": "逐记录 Presence F1 的平均值",
                    "presence_micro_f1": "聚合字段级 TP/FP/FN 的 Micro-F1",
                    "average_exact_value_recall": "逐记录 Exact Value Recall 的平均值",
                    "average_selected_gold_coverage": "选中字段对 Gold 非空字段的平均覆盖率",
                    "average_estimated_field_cost": "字段选择离线估计成本，非实际 API 成本",
                },
                "rows": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    csv_path = ROOT / "reports/quality_token_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    markdown = [
        "# 质量-Token 跨方法对照",
        "",
        "本报告由 `compile_quality_token_comparison.py` 从各方法原始评估报告生成。",
        "不同样本口径只作描述性对照，不作严格显著性比较。",
        "",
        "| 方法 | 样本口径 | runs | Presence F1 | Micro-F1 | Exact Value Recall | Gold覆盖 | 输入Token | 总Token |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for item in rows:
        markdown.append(
            "| {method} | {sample_scope} | {runs} | {average_presence_f1:.4f} | "
            "{presence_micro_f1:.4f} | {average_exact_value_recall:.4f} | "
            "{average_selected_gold_coverage:.4f} | {average_input_tokens:.1f} | "
            "{average_total_tokens:.1f} |".format(**item)
        )
    markdown.extend(
        [
            "",
            "## 解读边界",
            "- M1 的三 seed 结果是独立的完整 500 条实验；其均值可用于评估 n=100 训练方案的稳定性。",
            "- 历史内部方法使用七方法共同成功的 476 条，不能与新实验直接作严格配对显著性比较。",
            "- SchemaRAG k=10/k=15 是外部方法的重实现，不是官方代码复现。",
            "- `average_actual_cost=0` 或估计字段成本不代表真实 API 无成本；真实成本应按 Token 单价计算。",
            "- 当前指标尚未包含 Value Judge、描述 embedding 或综合 Quality。",
            "",
            "## 当前主结论",
            "- M1 n=100 + budget-200 在新 500 条批次中同时取得较高质量和较低 Token，三 seed 结果稳定。",
            "- budget-400 并未随覆盖率提高而提高端到端质量，说明选中字段更多不必然更好。",
            "- M1 n=100 + budget-200 与历史 FullSchema 的样本子集不同，现阶段应表述为强候选和描述性对照，",
            "  不能表述为严格同批优于 FullSchema。",
        ]
    )
    (ROOT / "reports/quality_token_comparison.md").write_text(
        "\n".join(markdown) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
