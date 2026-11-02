#!/usr/bin/env python3
"""绘制 Character、M1 与 M2 的离线模型策略和训练规模对照图。"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

SKILL_SCRIPTS = Path("/root/.agents/skills/nature-figure/scripts")
sys.path.insert(0, str(SKILL_SCRIPTS))

from audit_panel_alignment import require_matplotlib_panel_alignment


ROOT = Path(__file__).resolve().parent
CHARACTER_REPORT = ROOT / "reports/training_size_ablation/summary.json"
M1_REPORT = ROOT / "reports/model_strategy_ablation/m1_by_size/comparison.json"
M2_REPORT = ROOT / "reports/model_strategy_ablation/m2/comparison.json"
OUTPUT = ROOT / "tex/model_strategy_learning_curves"

METRICS = (
    ("presence_macro_f1", "Presence Macro-F1", (0.45, 0.76)),
    ("presence_micro_f1", "Presence Micro-F1", (0.86, 0.98)),
    ("length_mae", "Length MAE (tokens)", (2.0, 5.8)),
)

COLORS = {
    "Character TF--IDF": "#5B6770",
    "M1 early fusion": "#167C80",
    "M2 late fusion": "#C35B34",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def summarize_runs(runs: list[dict]) -> dict[int, dict[str, tuple[float, float]]]:
    grouped: dict[int, list[dict]] = defaultdict(list)
    for run in runs:
        grouped[int(run["size"])].append(run)

    summary: dict[int, dict[str, tuple[float, float]]] = {}
    for size, values in grouped.items():
        summary[size] = {}
        for metric, _, _ in METRICS:
            scores = np.asarray([item[metric] for item in values], dtype=float)
            ci95 = 1.96 * scores.std(ddof=1) / np.sqrt(len(scores))
            summary[size][metric] = (float(scores.mean()), float(ci95))
    return summary


def character_summary(report: dict) -> dict[int, dict[str, tuple[float, float]]]:
    result: dict[int, dict[str, tuple[float, float]]] = {}
    for row in report["summary"]:
        result[int(row["size"])] = {
            metric: (float(row[f"{metric}_mean"]), float(row[f"{metric}_ci95"]))
            for metric, _, _ in METRICS
        }
    return result


def plot_series(
    axis: plt.Axes,
    summary: dict[int, dict[str, tuple[float, float]]],
    metric: str,
    label: str,
    marker: str,
) -> None:
    sizes = sorted(summary)
    means = [summary[size][metric][0] for size in sizes]
    errors = [summary[size][metric][1] for size in sizes]
    axis.errorbar(
        sizes,
        means,
        yerr=errors,
        color=COLORS[label],
        marker=marker,
        markersize=4.5,
        linewidth=1.6,
        capsize=2.2,
        label=label,
        zorder=3,
    )


def main() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "axes.titlesize": 7,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.5,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.8,
        }
    )

    character = character_summary(load_json(CHARACTER_REPORT))
    m1 = summarize_runs(load_json(M1_REPORT)["runs"])
    m2 = summarize_runs(load_json(M2_REPORT)["runs"])
    figure, axes = plt.subplots(1, 3, figsize=(7.25, 2.45), sharex=True)

    for index, (axis, (metric, ylabel, ylim)) in enumerate(zip(axes, METRICS, strict=True)):
        plot_series(axis, character, metric, "Character TF--IDF", "o")
        plot_series(axis, m1, metric, "M1 early fusion", "s")

        plot_series(axis, m2, metric, "M2 late fusion", "D")
        axis.set_xscale("log")
        axis.set_xticks(sorted(character))
        axis.set_xticklabels([str(size) for size in sorted(character)])
        axis.set_ylim(*ylim)
        axis.set_xlabel("Training samples")
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", color="#D9DEE2", linewidth=0.55, zorder=0)
        axis.text(
            -0.18,
            1.04,
            chr(ord("a") + index),
            transform=axis.transAxes,
            fontweight="bold",
            fontsize=8,
            va="bottom",
        )

    axes[0].legend(loc="lower right", handlelength=1.6)
    figure.subplots_adjust(left=0.075, right=0.995, bottom=0.24, top=0.93, wspace=0.42)
    figure.canvas.draw()
    require_matplotlib_panel_alignment(
        figure,
        json_out=f"{OUTPUT}.alignment.json",
        overlay_svg=f"{OUTPUT}.alignment.svg",
        tolerance_pt=1.5,
        gutter_tolerance_pt=1.5,
        require_panel_labels=True,
        strict=True,
    )
    figure.savefig(f"{OUTPUT}.svg", bbox_inches="tight")
    figure.savefig(f"{OUTPUT}.pdf", bbox_inches="tight")
    figure.savefig(f"{OUTPUT}.png", dpi=600, bbox_inches="tight")


if __name__ == "__main__":
    main()
