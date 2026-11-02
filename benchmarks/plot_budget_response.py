#!/usr/bin/env python3
"""根据英文主稿 tab:budget-response 的九个在线预算点生成预算响应图。"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt

OUTPUT = Path("tex/v6-lncs/figures/budget_response.png")

BUDGET = [60, 100, 150, 200, 300, 400, 600, 800, 1200]
TOKENS = [326.7, 404.0, 482.4, 534.6, 627.4, 730.7, 856.1, 857.6, 858.0]
FIELDS = [2.80, 4.15, 5.90, 7.52, 10.13, 13.73, 18.00, 18.00, 18.00]
FIELD_PCT = [15.5, 23.1, 32.8, 41.8, 56.3, 76.3, 100.0, 100.0, 100.0]
P99 = [1240, 993, 1186, 1445, 1706, 2818, 3883, 3870, 3915]


def main() -> None:
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

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.2, 2.8))

    color_tokens = "#1f77b4"
    color_fields = "#d62728"
    color_pct = "#2ca02c"
    color_p99 = "#9467bd"

    ax1.plot(BUDGET, TOKENS, marker="o", linestyle="-",
             color=color_tokens, label="LLM tokens/record")
    ax1.set_xlabel("Budget capacity B")
    ax1.set_ylabel("LLM tokens/record")
    ax1.set_ylim(250, 950)
    ax1.tick_params(axis="y", labelcolor=color_tokens)
    ax1.legend(loc="lower right")

    ax1b = ax1.twinx()
    ax1b.plot(BUDGET, FIELDS, marker="s", linestyle="--",
              color=color_fields, label="Fields/record")
    ax1b.set_ylabel("Fields/record")
    ax1b.set_ylim(0, 20)
    ax1b.tick_params(axis="y", labelcolor=color_fields)
    ax1b.legend(loc="upper left")

    ax1.axvline(200, color="gray", linestyle=":", linewidth=1)
    ax1.text(200, 260, "B=200", rotation=90, fontsize=7, color="gray",
             ha="right", va="bottom")

    ax2.plot(BUDGET, FIELD_PCT, marker="o", linestyle="-",
             color=color_pct, label="Fields/18 (%)")
    ax2.set_xlabel("Budget capacity B")
    ax2.set_ylabel("Fields/18 (%)")
    ax2.set_ylim(0, 110)
    ax2.tick_params(axis="y", labelcolor=color_pct)
    ax2.legend(loc="lower right")

    ax2b = ax2.twinx()
    ax2b.plot(BUDGET, P99, marker="^", linestyle="--",
              color=color_p99, label="P99 (ms)")
    ax2b.set_ylabel("P99 (ms)")
    ax2b.set_ylim(0, 4500)
    ax2b.tick_params(axis="y", labelcolor=color_p99)
    ax2b.legend(loc="upper left")

    ax2.axvline(200, color="gray", linestyle=":", linewidth=1)
    ax2.text(200, 8, "B=200", rotation=90, fontsize=7, color="gray",
             ha="right", va="bottom")

    fig.tight_layout()
    fig.savefig(OUTPUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {OUTPUT}")


if __name__ == "__main__":
    main()