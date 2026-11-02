#!/usr/bin/env python
"""A6/A3 离线分析：预算网格（A6）与匹配基线（A3）。

读取 benchmarks/runs/a6-a3-sweep/raw/*.jsonl（2000 条冻结负载，c16/q16，每次 1 repeat），
逐条重算 token/延迟/失败/预算利用率/选中字段覆盖，输出 per-method 汇总。

产出 benchmarks/analysis/a6_a3/：
  - a6_a3.json         汇总（method -> 指标）
  - a6_budget_curve.csv  预算网格成本-覆盖率响应（A6）
  - a3_baseline.csv      匹配基线对比（A3）
  - 2 张图：a6_budget_response、a3_baseline_compare

不触发 LLM/embedding，纯离线统计。
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SWEEP = ROOT / "benchmarks/runs/a6-a3-sweep/raw"
OUT_DIR = ROOT / "benchmarks/analysis/a6_a3"

REQUIRED = ["name", "category"]

# 复用的主实验运行（2000 条冻结负载，c16/q16，r1），补齐 A6 曲线与 A3 基线。
REUSE = {
    "full": ("main-comparison-pro", "main-comparison-pro-full-c16-q16-r1.jsonl"),
    "budget-200": ("main-comparison-pro", "main-comparison-pro-budget-200-c16-q16-r1.jsonl"),
    "external-schemarag-k10": (
        "main-comparison-pro",
        "main-comparison-pro-external-schemarag-k10-c16-q16-r1.jsonl",
    ),
}


def percentile(vals: list[float], p: float) -> float | None:
    if not vals:
        return None
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100.0
    lo = math.floor(idx)
    hi = math.ceil(idx)
    if lo == hi:
        return s[lo]
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def compute_per_record_tokens(rec: dict) -> tuple[float, float, float]:
    """顶层 input/output/total tokens（含重试累计），与 recompute_from_raw 口径一致。"""
    return (
        float(rec.get("input_tokens") or 0),
        float(rec.get("output_tokens") or 0),
        float(rec.get("total_tokens") or 0),
    )


def analyze_method(name: str, rows: list[dict]) -> dict:
    n_total = len(rows)
    n_success = sum(1 for r in rows if r.get("status") == "success")
    n_error = n_total - n_success

    in_tok = out_tok = tot_tok = 0.0
    llm_ms = queue_wait_ms = 0.0
    lat = []
    alloc = req = 0.0
    n_sel = []
    missing = 0
    unplanned = 0

    for r in rows:
        i, o, t = compute_per_record_tokens(r)
        in_tok += i
        out_tok += o
        tot_tok += t
        llm_ms += float(r.get("llm_ms") or 0)
        queue_wait_ms += float(r.get("queue_wait_ms") or 0)
        alloc += float(r.get("allocated_budget") or 0)
        req += float(r.get("required_budget") or 0)
        n_sel.append(len(r.get("selected_fields") or []))
        if r.get("budget_status") != "planned":
            unplanned += 1
        if r.get("missing_fields_expected"):
            missing += 1
        lv = r.get("latency_ms")
        if lv is not None:
            lat.append(float(lv))

    # 覆盖率代理：选中字段数占 18 字段 schema 的比例（性能负载无 Gold，仅成本/覆盖代理）。
    mean_sel = sum(n_sel) / len(n_sel) if n_sel else 0.0
    return {
        "method": name,
        "n_total": n_total,
        "n_success": n_success,
        "n_error": n_error,
        "success_rate": n_success / n_total if n_total else 0.0,
        "total_tokens": tot_tok,
        "input_tokens": in_tok,
        "output_tokens": out_tok,
        "tokens_per_record": tot_tok / n_total if n_total else 0.0,
        "llm_ms_total": llm_ms,
        "queue_wait_ms_total": queue_wait_ms,
        "latency_p50_ms": percentile(lat, 50),
        "latency_p95_ms": percentile(lat, 95),
        "latency_p99_ms": percentile(lat, 99),
        "allocated_budget_total": alloc,
        "required_budget_total": req,
        "budget_headroom_ratio": req / alloc if alloc else None,
        "mean_selected_fields": mean_sel,
        "selected_coverage": mean_sel / 18.0,
        "missing_fields_records": missing,
        "non_planned_records": unplanned,
        "n_records": n_total,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A6/A3 预算网格与基线离线分析")
    parser.add_argument("--raw-dir", type=Path, default=SWEEP)
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    raw_dir: Path = args.raw_dir
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # 按方法聚合（一个方法一个 jsonl）。
    results: dict[str, dict] = {}
    # 复用主实验运行。
    for name, (subdir, fn) in REUSE.items():
        fp = ROOT / "benchmarks/runs" / subdir / "raw" / fn
        if not fp.exists():
            print(f"警告：缺少复用点 {fp}")
            continue
        rows = [json.loads(line) for line in fp.open("r", encoding="utf-8")]
        results[name] = analyze_method(name, rows)
        print(f"{name}(reuse): n={results[name]['n_success']}/{results[name]['n_total']} "
              f"tok/rec={results[name]['tokens_per_record']:.1f} "
              f"sel={results[name]['mean_selected_fields']:.2f} "
              f"p99={results[name]['latency_p99_ms']:.0f}ms")
    # 本次扫描运行。
    files = sorted(raw_dir.glob("*.jsonl"))
    if not files:
        print(f"警告：{raw_dir} 下无 jsonl")
    for fp in files:
        name = fp.stem
        rows = [json.loads(line) for line in fp.open("r", encoding="utf-8")]
        results[name] = analyze_method(name, rows)
        print(f"{name}: n={results[name]['n_success']}/{results[name]['n_total']} "
              f"tok/rec={results[name]['tokens_per_record']:.1f} "
              f"sel={results[name]['mean_selected_fields']:.2f} "
              f"p99={results[name]['latency_p99_ms']:.0f}ms")

    (out_dir / "a6_a3.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # A6 预算响应：提取 budget 类方法（本次扫描名 a6-budget-<N>，复用名 budget-<N>）。
    budget_rows = sorted(
        [
            (int(n.rsplit("-", 1)[1]), v)
            for n, v in results.items()
            if n.startswith("budget-") or n.startswith("a6-budget-")
        ]
    )
    if budget_rows:
        csv = ["method,budget,tokens_per_record,selected_fields,coverage_pct,p99_ms,success_rate"]
        for b, v in budget_rows:
            csv.append(
                f"{v['method']},{b},{v['tokens_per_record']:.2f},"
                f"{v['mean_selected_fields']:.2f},{v['selected_coverage']*100:.1f},"
                f"{v['latency_p99_ms']:.0f},{v['success_rate']:.4f}"
            )
        (out_dir / "a6_budget_curve.csv").write_text("\n".join(csv) + "\n")

        budgets = [b for b, _ in budget_rows]
        toks = [v["tokens_per_record"] for _, v in budget_rows]
        covs = [v["selected_coverage"] * 100 for _, v in budget_rows]
        p99s = [v["latency_p99_ms"] for _, v in budget_rows]

        fig, axs = plt.subplots(1, 3, figsize=(15, 4))
        axs[0].plot(budgets, toks, "o-")
        axs[0].set_xlabel("Budget (token)"); axs[0].set_ylabel("tokens/record"); axs[0].set_title("A6: cost")
        axs[1].plot(budgets, covs, "s-")
        axs[1].set_xlabel("Budget (token)"); axs[1].set_ylabel("selected coverage %"); axs[1].set_title("A6: coverage")
        axs[2].plot(budgets, p99s, "^-")
        axs[2].set_xlabel("Budget (token)"); axs[2].set_ylabel("P99 latency ms"); axs[2].set_title("A6: tail latency")
        for ax in axs:
            ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / "a6_budget_response.png", dpi=150)
        plt.close(fig)

    # A3 基线对比：全部方法，按 tokens_per_record 排序。
    all_rows = sorted(results.values(), key=lambda v: v["tokens_per_record"])
    csv = ["method,group,tokens_per_record,selected_fields,coverage_pct,p99_ms,success_rate"]
    group_map = {
        "full": "full", "prompt-null": "full", "prompt-omit": "full",
    }

    def classify(n: str) -> str:
        # 本次扫描运行名带 a6-/a3- 前缀，归一化后按方法名分类。
        base = n.split("-", 1)[1] if n.startswith(("a6-", "a3-")) else n
        if base.startswith("budget-"):
            return "budget"
        if base.startswith("fixed-"):
            return "fixed"
        if base.startswith("lambda-"):
            return "lambda"
        return group_map.get(base, "other")

    for v in all_rows:
        name = v["method"]
        group = classify(name)
        csv.append(
            f"{name},{group},{v['tokens_per_record']:.2f},{v['mean_selected_fields']:.2f},"
            f"{v['selected_coverage']*100:.1f},{v['latency_p99_ms']:.0f},{v['success_rate']:.4f}"
        )
    (out_dir / "a3_baseline.csv").write_text("\n".join(csv) + "\n")

    # A3 图：成本(tok/rec) vs 覆盖率，标注分组颜色。
    names = [v["method"] for v in all_rows]
    toks = [v["tokens_per_record"] for v in all_rows]
    covs = [v["selected_coverage"] * 100 for v in all_rows]
    colors = []
    for v in all_rows:
        g = classify(v["method"])
        colors.append(
            "#e74c3c" if g == "budget" else
            "#3498db" if g == "fixed" else
            "#9b59b6" if g == "lambda" else
            "#2ecc71"
        )
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.scatter(toks, covs, c=colors, s=80)
    for n, x, y in zip(names, toks, covs):
        ax.annotate(n, (x, y), textcoords="offset points", xytext=(6, 6), fontsize=8)
    ax.set_xlabel("tokens/record (cost)"); ax.set_ylabel("selected coverage %")
    ax.set_title("A3 baselines: cost vs coverage")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "a3_baseline_compare.png", dpi=150)
    plt.close(fig)

    print(f"\n完成。输出到 {out_dir}")


if __name__ == "__main__":
    main()