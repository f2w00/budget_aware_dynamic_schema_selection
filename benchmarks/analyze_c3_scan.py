#!/usr/bin/env python3
"""C3 解耦扫描汇总：固定 q 扫 c 与固定 c 扫 q。

从 c3-scan-decoupled 各网格点的 raw/*.jsonl 与 *.meta.json 离线重算，
输出并发扫描 / 队列扫描两张表 + 吞吐/尾延迟曲线，识别饱和点与背压。

数据来源（已核验）：
- Set A（固定 c=16，扫 q∈{4,8,16,32,64}）：
    q4/q8/q16/q32 来自顶层 raw/（c3-scan-fixc16-*），
    q64 来自 setA-q64-rerun（干净重跑，submitted=500）。
- Set B（固定 q=16，扫 c∈{1,2,4,8,16,32}）：
    c1/c2/c4/c8/c32 来自 setB-c{c}，c16 复用 Set A 的 q16。
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import median, mean, stdev

# C3 实验根目录
C3 = Path("benchmarks/runs/c3-scan-decoupled")

# 每个网格点 -> (meta 文件, raw 文件)。手工列出确保使用干净数据。
POINTS: dict[tuple[int, int], dict[str, Path]] = {
    # Set B：固定 q=16，扫 c
    (1, 16): {"dir": C3 / "setB-c1", "prefix": "c3-scan-fixq16-budget-200-c1-q16-r1"},
    (2, 16): {"dir": C3 / "setB-c2", "prefix": "c3-scan-fixq16-budget-200-c2-q16-r1"},
    (4, 16): {"dir": C3 / "setB-c4", "prefix": "c3-scan-fixq16-budget-200-c4-q16-r1"},
    (8, 16): {"dir": C3 / "setB-c8", "prefix": "c3-scan-fixq16-budget-200-c8-q16-r1"},
    (32, 16): {"dir": C3 / "setB-c32", "prefix": "c3-scan-fixq16-budget-200-c32-q16-r1"},
    # Set A：固定 c=16，扫 q
    (16, 4): {"dir": C3, "prefix": "c3-scan-fixc16-budget-200-c16-q4-r1"},
    (16, 8): {"dir": C3, "prefix": "c3-scan-fixc16-budget-200-c16-q8-r1"},
    (16, 16): {"dir": C3, "prefix": "c3-scan-fixc16-budget-200-c16-q16-r1"},
    (16, 32): {"dir": C3, "prefix": "c3-scan-fixc16-budget-200-c16-q32-r1"},
    # q64 用干净重跑（顶层 raw 的 q64 submitted=182 被超时打断，弃用）
    (16, 64): {"dir": C3 / "setA-q64-rerun", "prefix": "concurrency-scan-budget-200-c16-q64-r1"},
}


def percentile(sorted_values: list[float], p: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * (p / 100.0)
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    frac = rank - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def load_point(c: int, q: int) -> dict:
    cfg = POINTS[(c, q)]
    meta = json.loads((cfg["dir"] / f"{cfg['prefix']}.meta.json").read_text(encoding="utf-8"))
    raw_path = cfg["dir"] / "raw" / f"{cfg['prefix']}.jsonl"
    rows = []
    with raw_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    latency = sorted(float(r["latency_ms"]) for r in rows if "latency_ms" in r)
    queue_wait = sorted(float(r["queue_wait_ms"]) for r in rows if "queue_wait_ms" in r)
    success = sum(1 for r in rows if r.get("status") == "success")
    total = len(rows)
    pipe_ms = meta["pipeline_ms"]
    return {
        "concurrency": c,
        "queue_size": q,
        "n_total": total,
        "n_success": success,
        "n_error": total - success,
        "pipeline_ms": pipe_ms,
        "throughput_rps": success / (pipe_ms / 1000.0),
        "latency_p50_ms": percentile(latency, 50),
        "latency_p95_ms": percentile(latency, 95),
        "latency_p99_ms": percentile(latency, 99),
        "latency_max_ms": latency[-1] if latency else None,
        "queue_wait_p50_ms": percentile(queue_wait, 50),
        "queue_wait_p99_ms": percentile(queue_wait, 99),
        "queue_average": meta.get("queue_average"),
        "queue_peak": meta.get("queue_peak"),
        "producer_blocked_ms": meta.get("producer_blocked_ms"),
        "enqueue_ms": meta.get("enqueue_ms"),
        "submitted": meta.get("submitted"),
        "workload_ids_sha256": meta.get("workload_ids_sha256"),
        "temperature": meta.get("temperature"),
        "command": meta.get("command"),
    }


def plot_scan(set_rows: list[dict], x_key: str, x_label: str, output_dir: Path, stem: str) -> None:
    """画吞吐 + 尾延迟随 x_key 变化的双面板图。"""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "figure.dpi": 120,
    })
    x = [r[x_key] for r in set_rows]
    thr = [r["throughput_rps"] for r in set_rows]
    p50 = [r["latency_p50_ms"] / 1000 for r in set_rows]
    p99 = [r["latency_p99_ms"] / 1000 for r in set_rows]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.2))
    axes[0].plot(x, thr, marker="o", color="#1f77b4")
    axes[0].set_xlabel(x_label)
    axes[0].set_ylabel("Throughput (records/s)")
    axes[0].set_title("Throughput")
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)
    axes[1].plot(x, p50, marker="o", label="P50", color="#2ca02c")
    axes[1].plot(x, p99, marker="s", label="P99", color="#d62728")
    axes[1].set_xlabel(x_label)
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_title("Tail latency")
    axes[1].legend()
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)

    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    import matplotlib.pyplot as _plt
    _plt.close(fig)


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    fields = list(rows[0])
    import csv as _csv
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = _csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="C3 解耦扫描离线汇总")
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/analysis/c3_scan"))
    args = parser.parse_args()

    results = []
    for (c, q) in sorted(POINTS):
        try:
            results.append(load_point(c, q))
        except FileNotFoundError as exc:
            print(f"跳过缺失点 c={c} q={q}: {exc}")

    # 分两组：Set A 固定 c=16，Set B 固定 q=16
    set_a = sorted([r for r in results if r["concurrency"] == 16], key=lambda r: r["queue_size"])
    set_b = sorted([r for r in results if r["queue_size"] == 16], key=lambda r: r["concurrency"])

    payload = {
        "analysis": {
            "description": (
                "C3 解耦扫描：固定并发扫队列 + 固定队列扫并发，从 raw 离线重算。"
                "覆盖 budget-200，limit=500（499 条有效），temperature=0.0。"
                "q64 采用干净重跑（setA-q64-rerun，submitted=500），弃用被超时打断的 submitted=182 记录。"
            ),
            "source_dir": str(C3),
        },
        "set_A_fixed_concurrency_scan_q": set_a,
        "set_B_fixed_queue_scan_c": set_b,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "c3_scan.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    write_csv(args.output_dir / "set_A_fixed_c_scan_q.csv", set_a)
    write_csv(args.output_dir / "set_B_fixed_q_scan_c.csv", set_b)

    # 生成两张图：吞吐/尾延迟 随 队列大小、随 并发 变化
    plot_scan(set_a, "queue_size", "Queue size (c=16 fixed)", args.output_dir, "c3_fixed_c_scan_q")
    plot_scan(set_b, "concurrency", "Concurrency (q=16 fixed)", args.output_dir, "c3_fixed_q_scan_c")

    # 简洁控制台输出
    for label, rows in (("Set A (c=16, 扫 q)", set_a), ("Set B (q=16, 扫 c)", set_b)):
        print(f"\n=== {label} ===")
        for r in rows:
            print(
                f"  c={r['concurrency']:>2} q={r['queue_size']:>3} | "
                f"thr={r['throughput_rps']:>6.2f} rps | "
                f"p50={r['latency_p50_ms']:>6.0f} p95={r['latency_p95_ms']:>6.0f} "
                f"p99={r['latency_p99_ms']:>7.0f}ms | "
                f"success={r['n_success']}/{r['n_total']} | "
                f"qwait_p99={r['queue_wait_p99_ms']:>6.0f}ms | "
                f"producer_blocked={r['producer_blocked_ms']/1000:>7.1f}s | "
                f"queue_peak={r['queue_peak']}"
            )

    print(json.dumps({"output": str(args.output_dir / "c3_scan.json")}, ensure_ascii=False))


if __name__ == "__main__":
    main()