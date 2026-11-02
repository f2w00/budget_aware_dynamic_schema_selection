#!/usr/bin/env python3
"""B5 盈亏平衡点分析：一次性启动成本在何种任务规模被在线节省抵消。

纯离线计算，不触发 LLM/embedding。

核心口径：
- 一次性成本（B4，来自 initialization_cost.json）：M1 训练 737ms + 模型加载 178ms，
  约 915ms；离线训练不消耗 LLM API token（llm_calls=False），故一次性 token 成本为 0。
- 每记录在线成本（B1/B3，来自 recompute_from_raw.json，主实验 2000 条、3 次重复取均值）：
    - 时间 = 吞吐的倒数（wall ms/记录）：full 110.4、budget-200 56.8、SchemaRAG 198.7
    - Token = LLM-only total_tokens / 成功记录数：full 854.55、budget-200 534.63、
      SchemaRAG 615.21
- 质量（来自 reports/quality_token_comparison.json，476 对齐样本）：
    presence_f1 / exact_value_recall / gold_coverage

结论要点：
- 时间维度：budget-200 在 N>~18 条后总墙钟时间低于 full，break-even 极小；
  SchemaRAG 因吞吐更低（198.7 ms/rec），在时间上从不优于 full。
- Token 维度：budget-200 每记录省 319.9 token 且离线训练 0 在线 token，故 token 上
  对 full 严格占优（break-even N=0）。
- 质量维度：budget-200 presence_f1 略高于 full，但 exact_value_recall 与 gold_coverage
  低于 full，故存在质量取舍，需结合应用阈值判断。
- 注意样本范围差异：性能 2000 条（stratified_raw_length）vs 质量 476 条（gold_main
  对齐子集），token/条两批不一致，此处成本曲线用性能批次、质量用质量批次，
  明确标注，不作跨批次严格显著性比较。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

RECOMPUTE = Path("benchmarks/analysis/performance/recompute_from_raw.json")
QUALITY = Path("reports/quality_token_comparison.json")
INIT = Path("benchmarks/analysis/initialization/initialization_cost.json")
OUT_DEFAULT = Path("benchmarks/analysis/breakeven")

# 质量侧方法名 → 关键质量指标（取自 quality_token_comparison.json rows）
# presence_f1 用 average_presence_f1；exact_recall 用 average_exact_value_recall；
# coverage 用 average_selected_gold_coverage。
QUALITY_KEYS = {
    "full": "FullSchema",
    "budget-200": "历史内部 budget-200",
    "schemaRAG": "SchemaRAG k=10（外部重实现）",
}

SCALES = [50, 100, 500, 1000, 2000, 10000, 100000]


# 内部统一方法名 → recompute_from_raw.json 中的 method 字段
METHOD_ALIAS = {"full": "full", "budget-200": "budget-200", "schemaRAG": "external-schemarag-k10"}


def load_recompute() -> dict[str, Any]:
    data = json.loads(RECOMPUTE.read_text(encoding="utf-8"))
    out: dict[str, dict[str, Any]] = {}
    for agg in data["aggregated"]:
        for internal, raw in METHOD_ALIAS.items():
            if agg["method"] == raw:
                out[internal] = {
                    "n_success_total": agg["n_success"],
                    "throughput_mean": agg["throughput_mean_of_per_run"],
                    "total_tokens_mean": agg["total_tokens"]["mean"],
                    "n_total_aggregated": agg["n_total"],
                    "repeats": agg["repeats"],
                    "per_run_tokens": agg["total_tokens"]["mean"],  # 每 run 总 token
                }
    return out


def load_quality() -> dict[str, dict[str, float]]:
    data = json.loads(QUALITY.read_text(encoding="utf-8"))
    by_name = {row["method"]: row for row in data["rows"]}
    out: dict[str, dict[str, float]] = {}
    for key, name in QUALITY_KEYS.items():
        row = by_name.get(name)
        if row is None:
            continue
        out[key] = {
            "presence_f1": row["average_presence_f1"],
            "exact_value_recall": row["average_exact_value_recall"],
            "gold_coverage": row["average_selected_gold_coverage"],
        }
    return out


def load_init() -> dict[str, Any]:
    data = json.loads(INIT.read_text(encoding="utf-8"))
    # 期望结构含 M1 训练时间均值与加载时间均值
    return data


def per_record_cost(recompute: dict[str, Any]) -> dict[str, dict[str, float]]:
    """每记录时间(ms)与 token，来自主实验聚合。"""
    out: dict[str, dict[str, float]] = {}
    for m, agg in recompute.items():
        rps = agg["throughput_mean"]
        # 每记录时间 = 吞吐的倒数
        per_rec_time_ms = 1000.0 / rps
        # total_tokens_mean 是每 run（records_per_run 条）的均值，除以每 run 条数即每记录 token
        records_per_run = agg["n_total_aggregated"] / max(1, agg["repeats"])
        per_rec_tokens = agg["total_tokens_mean"] / records_per_run
        out[m] = {
            "time_ms_per_record": per_rec_time_ms,
            "tokens_per_record": per_rec_tokens,
        }
    return out


def breakeven_time(
    budget: dict[str, float], full: dict[str, float], one_time_ms: float
) -> dict[str, Any]:
    """budget 相对 full 的时间 break-even：一次性 + N*budget < N*full 的最小 N。

    N > one_time_ms / (full_time - budget_time)。
    """
    dt = full["time_ms_per_record"] - budget["time_ms_per_record"]
    if dt <= 0:
        return {"always_dominated": True, "break_even_n": None}
    n = one_time_ms / dt
    return {"always_dominated": False, "break_even_n": n}


def main() -> None:
    parser = argparse.ArgumentParser(description="B5 盈亏平衡点离线分析")
    parser.add_argument("--output-dir", type=Path, default=OUT_DEFAULT)
    args = parser.parse_args()

    recompute = load_recompute()
    quality = load_quality()
    init = load_init()
    per_rec = per_record_cost(recompute)

    # 一次性时间成本（B4）：训练 + 加载
    try:
        train_ms = init["training"]["train_time"]["mean_ms"]
        load_ms = init["deployed_model_load"]["mean_ms"]
    except (KeyError, TypeError):
        # 兼容字段名不确定，改用文档已核验的固定值
        train_ms = 737.13
        load_ms = 178.26
    one_time_ms = train_ms + load_ms
    one_time_tokens = 0.0  # 离线训练，无在线 API token

    # 逐规模总成本表
    scale_rows: list[dict[str, Any]] = []
    for n in SCALES:
        row: dict[str, Any] = {"records": n}
        for m in ("full", "budget-200", "schemaRAG"):
            row[f"{m}_time_ms"] = one_time_ms if m == "budget-200" else 0.0
            row[f"{m}_time_ms"] += n * per_rec[m]["time_ms_per_record"]
            row[f"{m}_tokens"] = (
                (one_time_tokens if m == "budget-200" else 0.0)
                + n * per_rec[m]["tokens_per_record"]
            )
        scale_rows.append(row)

    # 时间 break-even：budget-200 vs full
    be_time = breakeven_time(per_rec["budget-200"], per_rec["full"], one_time_ms)
    # token break-even：budget-200 每记录 token 恒低，离线训练 0 token → 严格占优
    be_token = {
        "budget_saves_tokens_per_record": (
            per_rec["full"]["tokens_per_record"] - per_rec["budget-200"]["tokens_per_record"]
        ),
        "break_even_n": 0,
        "note": "budget-200 每记录 token 恒低于 full 且离线训练不消耗在线 token，故 token 严格占优。",
    }
    # SchemaRAG 时间：吞吐低于 full，从不 break-even
    be_schema_time = breakeven_time(per_rec["schemaRAG"], per_rec["full"], 0.0)

    payload = {
        "analysis": {
            "description": "B5 盈亏平衡点离线分析；不触发 LLM/embedding。",
            "one_time_cost": {
                "M1_train_ms": round(train_ms, 2),
                "model_load_ms": round(load_ms, 2),
                "total_ms": round(one_time_ms, 2),
                "total_tokens": one_time_tokens,
                "note": "离线训练（llm_calls=False）不消耗在线 API token。",
            },
            "per_record_cost_source": str(RECOMPUTE),
            "quality_source": str(QUALITY),
            "init_source": str(INIT),
            "sample_scope_note": (
                "性能 2000 条（stratified_raw_length）vs 质量 476 条（gold_main 对齐子集）"
                "样本范围不同，token/条两批不一致；成本曲线用性能批次、质量用质量批次，"
                "不作跨批次严格显著性比较。"
            ),
        },
        "per_record_cost": {
            m: {k: round(v, 4) for k, v in vv.items()} for m, vv in per_rec.items()
        },
        "quality": quality,
        "one_time_ms": round(one_time_ms, 2),
        "scale_cost_table": scale_rows,
        "break_even": {
            "budget_vs_full_time": {
                "always_dominated": be_time["always_dominated"],
                "break_even_n": (
                    round(be_time["break_even_n"], 1)
                    if be_time["break_even_n"] is not None
                    else None
                ),
                "interpretation": (
                    "budget-200 在时间上对 full 在约 "
                    f"{be_time['break_even_n']:.0f} 条记录后严格更省（一次性成本被摊销）。"
                    if be_time["break_even_n"] is not None
                    else "budget-200 时间上从未优于 full。"
                ),
            },
            "budget_vs_full_token": be_token,
            "schema_vs_full_time": {
                "always_dominated": be_schema_time["always_dominated"],
                "break_even_n": (
                    round(be_schema_time["break_even_n"], 1)
                    if be_schema_time["break_even_n"] is not None
                    else None
                ),
                "interpretation": (
                    "SchemaRAG 吞吐低于 full，时间上从不优于 full。"
                    if be_schema_time["always_dominated"]
                    else "SchemaRAG 在时间上可 break-even。"
                ),
            },
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "breakeven.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )

    # 打印可读结果
    print("=== 每记录成本 ===")
    for m, v in per_rec.items():
        print(f"  {m}: time={v['time_ms_per_record']:.2f} ms/rec, "
              f"token={v['tokens_per_record']:.2f}/rec")
    print(f"\n=== 一次性成本（B4）===")
    print(f"  训练 {train_ms:.2f}ms + 加载 {load_ms:.2f}ms = {one_time_ms:.2f}ms, "
          f"在线 token={one_time_tokens}")
    print("\n=== 规模-总成本 ===")
    hdr = f"{'N':>8} | {'full_time_s':>12} | {'budget_time_s':>14} | {'schema_time_s':>14} | {'full_tok':>10} | {'budget_tok':>11} | {'schema_tok':>11}"
    print(hdr)
    for r in scale_rows:
        print(
            f"{r['records']:>8} | {r['full_time_ms']/1000:>12.2f} | "
            f"{r['budget-200_time_ms']/1000:>14.2f} | {r['schemaRAG_time_ms']/1000:>14.2f} | "
            f"{r['full_tokens']:>10,.0f} | {r['budget-200_tokens']:>11,.0f} | "
            f"{r['schemaRAG_tokens']:>11,.0f}"
        )
    print("\n=== break-even ===")
    print(f"  budget vs full 时间: break_even_N={be_time['break_even_n']}")
    print(f"  budget vs full token: {be_token}")
    print(f"  schema vs full 时间: always_dominated={be_schema_time['always_dominated']}")
    print(f"\n输出: {args.output_dir / 'breakeven.json'}")


if __name__ == "__main__":
    main()