#!/usr/bin/env python3
"""从第二部分离线统计 CSV 生成论文性能图表。"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


DEFAULT_INPUT_DIR = Path("benchmarks/analysis/performance")
DEFAULT_OUTPUT_DIR = Path("benchmarks/analysis/performance/figures")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成第二部分论文性能图表")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def number(row: dict[str, Any], field: str) -> float:
    return float(row[field])


def style_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "figure.dpi": 120,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_main_comparison(rows: list[dict[str, Any]], output_dir: Path) -> None:
    order = ["full", "budget-200", "external-schemarag-k10"]
    labels = {
        "full": "FullSchema",
        "budget-200": "Adaptive B=200",
        "external-schemarag-k10": "SchemaRAG k=10",
    }
    colors = {"full": "#4C566A", "budget-200": "#2F6B9A", "external-schemarag-k10": "#C45A3C"}
    data = {row["method"]: row for row in rows}
    selected = [data[method] for method in order]
    names = [labels[row["method"]] for row in selected]
    x = range(len(selected))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axes[0].bar(
        x,
        [number(row, "effective_records_per_second_mean") for row in selected],
        color=[colors[row["method"]] for row in selected],
    )
    axes[0].set_ylabel("Effective throughput (records/s)")
    axes[0].set_xticks(list(x), names, rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)
    axes[0].set_title("Throughput")

    axes[1].bar(
        x,
        [number(row, "total_tokens_total_mean") / 1e6 for row in selected],
        color=[colors[row["method"]] for row in selected],
    )
    axes[1].set_ylabel("Total tokens (million)")
    axes[1].set_xticks(list(x), names, rotation=18, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)
    axes[1].set_title("Token workload")
    save_figure(fig, output_dir, "main_throughput_and_tokens")


def plot_main_latency(rows: list[dict[str, Any]], output_dir: Path) -> None:
    order = ["full", "budget-200", "external-schemarag-k10"]
    labels = {
        "full": "FullSchema",
        "budget-200": "Adaptive B=200",
        "external-schemarag-k10": "SchemaRAG k=10",
    }
    colors = {"full": "#4C566A", "budget-200": "#2F6B9A", "external-schemarag-k10": "#C45A3C"}
    data = {row["method"]: row for row in rows}
    selected = [data[method] for method in order]
    names = [labels[row["method"]] for row in selected]
    x = range(len(selected))

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axes[0].bar(
        x,
        [number(row, "pipeline_ms_mean") / 1000 for row in selected],
        color=[colors[row["method"]] for row in selected],
    )
    axes[0].set_ylabel("Pipeline time (s)")
    axes[0].set_xticks(list(x), names, rotation=18, ha="right")
    axes[0].grid(axis="y", linestyle=":", alpha=0.7)
    axes[0].set_title("End-to-end completion time")

    axes[1].bar(
        x,
        [number(row, "latency_ms_p95_mean") / 1000 for row in selected],
        color=[colors[row["method"]] for row in selected],
        label="P95",
    )
    axes[1].bar(
        [value + 0.22 for value in x],
        [number(row, "latency_ms_p99_mean") / 1000 for row in selected],
        width=0.22,
        color="#A3BE8C",
        label="P99",
    )
    axes[1].set_ylabel("Latency (s)")
    axes[1].set_xticks([value + 0.11 for value in x], names, rotation=18, ha="right")
    axes[1].grid(axis="y", linestyle=":", alpha=0.7)
    axes[1].set_title("Tail latency")
    axes[1].legend(frameon=True)
    save_figure(fig, output_dir, "main_time_and_tail_latency")


def plot_endpoint_scaling(rows: list[dict[str, Any]], output_dir: Path) -> None:
    methods = {"full": ("FullSchema", "#4C566A"), "budget-200": ("Adaptive B=200", "#2F6B9A")}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    for method, (label, color) in methods.items():
        selected = sorted(
            (row for row in rows if row["method"] == method),
            key=lambda row: int(row["concurrency"]),
        )
        concurrency = [int(row["concurrency"]) for row in selected]
        axes[0].plot(
            concurrency,
            [number(row, "effective_records_per_second_mean") for row in selected],
            marker="o",
            linewidth=1.7,
            color=color,
            label=label,
        )
        axes[1].plot(
            concurrency,
            [number(row, "latency_ms_p99_mean") / 1000 for row in selected],
            marker="o",
            linewidth=1.7,
            color=color,
            label=label,
        )
    axes[0].set_xlabel("Consumer concurrency")
    axes[0].set_ylabel("Effective throughput (records/s)")
    axes[0].set_title("Endpoint scaling (c=1/32)")
    axes[1].set_xlabel("Consumer concurrency")
    axes[1].set_ylabel("P99 latency (s)")
    axes[1].set_title("P99 tail latency (c=1/32)")
    for axis in axes:
        axis.set_xticks([1, 32])
        axis.grid(True, linestyle=":", alpha=0.7)
        axis.legend(frameon=True)
    save_figure(fig, output_dir, "endpoint_scaling_and_tail_latency")


def plot_token_throughput(rows: list[dict[str, Any]], output_dir: Path) -> None:
    labels = {
        "full": "FullSchema",
        "budget-200": "Adaptive B=200",
        "external-schemarag-k10": "SchemaRAG k=10",
    }
    colors = {"full": "#4C566A", "budget-200": "#2F6B9A", "external-schemarag-k10": "#C45A3C"}
    fig, axis = plt.subplots(figsize=(5.2, 4.0))
    for row in rows:
        method = row["method"]
        axis.scatter(
            number(row, "total_tokens_total_mean") / 1e6,
            number(row, "effective_records_per_second_mean"),
            s=70,
            color=colors[method],
            label=labels[method],
            zorder=3,
        )
    axis.set_xlabel("Total tokens (million)")
    axis.set_ylabel("Effective throughput (records/s)")
    axis.set_title("Token workload versus throughput")
    axis.grid(True, linestyle=":", alpha=0.7)
    axis.legend(frameon=True)
    save_figure(fig, output_dir, "token_throughput_tradeoff")


def plot_initialization_sensitivity(output_dir: Path) -> None:
    """绘制一次性初始化成本对累计性能差异的离线敏感性曲线。"""
    records = list(range(1, 2001))
    full_rate = 9.0573
    budget_rate = 17.61
    full_time = [count / full_rate for count in records]
    budget_online = [count / budget_rate for count in records]
    budget_with_load = [0.178 + value for value in budget_online]
    budget_with_training = [0.915 + value for value in budget_online]

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.5))
    axes[0].plot(records, full_time, color="#4C566A", linewidth=1.8, label="FullSchema")
    axes[0].plot(
        records,
        budget_with_load,
        color="#2F6B9A",
        linewidth=1.8,
        label="Adaptive B=200 + load",
    )
    axes[0].plot(
        records,
        budget_with_training,
        color="#2F6B9A",
        linestyle="--",
        linewidth=1.5,
        label="Adaptive B=200 + train/load",
    )
    axes[0].axvline(4, color="#A3BE8C", linestyle=":", linewidth=1.2)
    axes[0].axvline(18, color="#D08770", linestyle=":", linewidth=1.2)
    axes[0].set_xscale("log")
    axes[0].set_xlabel("Workload size (records)")
    axes[0].set_ylabel("Cumulative time (s)")
    axes[0].set_title("Accumulated time")
    axes[0].grid(True, linestyle=":", alpha=0.7)
    axes[0].legend(frameon=True, fontsize=7)

    load_share = [0.178 / (0.178 + value) * 100 for value in budget_online]
    training_share = [0.915 / (0.915 + value) * 100 for value in budget_online]
    axes[1].plot(
        records,
        load_share,
        color="#A3BE8C",
        linewidth=1.8,
        label="Load only (0.178 s)",
    )
    axes[1].plot(
        records,
        training_share,
        color="#D08770",
        linestyle="--",
        linewidth=1.8,
        label="Train + load (0.915 s)",
    )
    axes[1].set_xscale("log")
    axes[1].set_xlabel("Workload size (records)")
    axes[1].set_ylabel("Startup share of total time (%)")
    axes[1].set_title("Startup-cost share")
    axes[1].grid(True, linestyle=":", alpha=0.7)
    axes[1].legend(frameon=True, fontsize=7)

    save_figure(fig, output_dir, "initialization_cost_sensitivity")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    style_plot()
    main_rows = read_csv(args.input_dir / "main_comparison.csv")
    endpoint_rows = read_csv(args.input_dir / "endpoint_scaling.csv")
    plot_main_comparison(main_rows, args.output_dir)
    plot_main_latency(main_rows, args.output_dir)
    plot_endpoint_scaling(endpoint_rows, args.output_dir)
    plot_token_throughput(main_rows, args.output_dir)
    plot_initialization_sensitivity(args.output_dir)
    print(f"generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
