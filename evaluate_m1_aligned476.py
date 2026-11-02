"""将 M1 stable 结果对齐到历史八方法共同成功的 476 条记录。"""

from __future__ import annotations

import argparse
import json
import sys
import subprocess
from pathlib import Path
from typing import Any


METHODS = (
    "prompt-omit",
    "full",
    "prompt-null",
    "fixed-5",
    "budget-200",
    "fixed-10",
    "lambda-0.003",
    "budget-400",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="评估 M1 与历史 476 条共同记录对齐结果")
    parser.add_argument(
        "--historical",
        type=Path,
        default=Path("experiments/gold_main_v3_stable_pro_results.jsonl"),
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        type=Path,
        default=[
            Path(
                "experiments/kg_assist_pro_n100_m1_stable_budget200_seed42_gold_main_v3_pro.jsonl"
            ),
            Path(
                "experiments/kg_assist_pro_n100_m1_stable_budget200_seed43_gold_main_v3_pro.jsonl"
            ),
            Path(
                "experiments/kg_assist_pro_n100_m1_stable_budget200_seed44_gold_main_v3_pro.jsonl"
            ),
        ],
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/aligned476_m1_stable_budget200"),
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=Path(
            "reports/kg_assist_pro_n100_m1_stable_budget200_aligned476_metrics.json"
        ),
    )
    parser.add_argument(
        "--gold",
        type=Path,
        default=Path("data/gold/agent_tasks/gold_main.jsonl"),
    )
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def historical_common_ids(path: Path) -> list[int]:
    rows = read_jsonl(path)
    successful: dict[str, set[int]] = {method: set() for method in METHODS}
    for item in rows:
        method = item.get("method") or item.get("run_label")
        if method in successful and item.get("status") == "success":
            successful[method].add(item["artifact_id"])
    common = set.intersection(*(successful[method] for method in METHODS))
    return sorted(common)


def filter_result(path: Path, ids: set[int], output: Path) -> int:
    rows = read_jsonl(path)
    kept = [row for row in rows if row.get("artifact_id") in ids]
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept),
        encoding="utf-8",
    )
    return len(kept)


def main() -> None:
    args = parse_args()
    ids = historical_common_ids(args.historical)
    if len(ids) != 476:
        raise SystemExit(f"历史共同记录数应为476，实际为{len(ids)}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    id_path = args.output_dir / "historical_common_ids.json"
    id_path.write_text(json.dumps(ids, indent=2) + "\n", encoding="utf-8")

    filtered = []
    counts = {}
    for input_path in args.inputs:
        output_path = args.output_dir / input_path.name
        counts[input_path.name] = filter_result(input_path, set(ids), output_path)
        filtered.append(output_path)

    command = [
        sys.executable,
        "evaluate_llm_experiments.py",
        "--input",
        *[str(path) for path in filtered],
        "--gold",
        str(args.gold),
        "--output",
        str(args.report),
    ]
    subprocess.run(command, check=True)
    report = json.loads(args.report.read_text(encoding="utf-8"))
    report["alignment"] = {
        "historical_source": str(args.historical),
        "historical_methods": list(METHODS),
        "common_artifact_ids": str(id_path),
        "common_records": len(ids),
        "filtered_input_records": counts,
        "note": "M1 结果仅按历史共同 artifact_id 对齐；历史方法指标仍来自原报告。",
    }
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"common_records": len(ids), "filtered": counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
