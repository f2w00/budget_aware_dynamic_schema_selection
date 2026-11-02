#!/usr/bin/env python3
"""绘制离线估计成本与 Full Schema 字段覆盖率曲线。"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="绘制离线成本-覆盖率曲线")
    parser.add_argument("--budget-report", type=Path, default=Path("reports/budget_replay.json"))
    parser.add_argument("--unbudgeted-report", type=Path, default=Path("reports/unbudgeted_replay.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("figures"))
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_points(budget_report: dict[str, Any], unbudgeted_report: dict[str, Any]) -> list[dict[str, Any]]:
    points = []
    for row in budget_report["budgets"]:
        points.append({
            "method": "Fixed-budget adaptive",
            "setting": f"B={row['budget']}",
            "lambda": "",
            "estimated_cost": row["average_estimated_cost"],
            "coverage": row["average_gold_coverage"],
            "selected_fields": row["average_selected_fields"],
        })
    for row in unbudgeted_report["test"]:
        points.append({
            "method": "Unbudgeted adaptive",
            "setting": f"lambda={row['lambda']}",
            "lambda": row["lambda"],
            "estimated_cost": row["average_estimated_cost"],
            "coverage": row["average_gold_coverage"],
            "selected_fields": row["average_selected_fields"],
        })
    return points


def write_csv(path: Path, points: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=points[0].keys())
        writer.writeheader()
        writer.writerows(points)


def plot(points: list[dict[str, Any]], output_dir: Path) -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.labelsize": 10,
        "axes.titlesize": 11,
        "legend.fontsize": 8,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    colors = {
        "Fixed-budget adaptive": "#2F6B9A",
        "Unbudgeted adaptive": "#C45A3C",
    }
    markers = {
        "Fixed-budget adaptive": "o",
        "Unbudgeted adaptive": "s",
    }
    for method in colors:
        subset = sorted(
            (point for point in points if point["method"] == method),
            key=lambda point: point["estimated_cost"],
        )
        ax.plot(
            [point["estimated_cost"] for point in subset],
            [point["coverage"] * 100 for point in subset],
            marker=markers[method],
            linewidth=1.7,
            markersize=4.5,
            label=method,
            color=colors[method],
        )
    ax.axhline(100, color="#666666", linestyle="--", linewidth=0.9, label="Full Schema reference")
    ax.set_xlabel("Average estimated field cost (tokens)")
    ax.set_ylabel("Full Schema non-empty field coverage (%)")
    ax.set_title("Offline estimated cost--coverage curve")
    ax.set_xlim(left=0)
    ax.set_ylim(0, 105)
    ax.grid(True, linestyle=":", linewidth=0.6, alpha=0.7)
    ax.legend(loc="lower right", frameon=True)
    fig.tight_layout()
    fig.savefig(output_dir / "quality_cost_curve.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "quality_cost_curve.png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    points = build_points(load_json(args.budget_report), load_json(args.unbudgeted_report))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "quality_cost_curve.csv", points)
    plot(points, args.output_dir)
    print(json.dumps({
        "points": len(points),
        "pdf": str(args.output_dir / "quality_cost_curve.pdf"),
        "png": str(args.output_dir / "quality_cost_curve.png"),
        "csv": str(args.output_dir / "quality_cost_curve.csv"),
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
