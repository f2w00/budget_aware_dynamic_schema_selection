#!/usr/bin/env python3
"""离线分析 n=100 配对运行的字段选择稳定性、结构分层和完整性。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from statistics import mean, stdev
from typing import Any


SEEDS = (42, 43, 44)
METHODS = ("full", "budget-200")
EXPECTED_HASH = (
    "831c1b9a7fcd84a09c17542e69114cc7f2a650b14cecc7e1c79a00e2058e274e"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="分析 n=100 配对运行的离线产物")
    parser.add_argument(
        "--runs-dir", type=Path, default=Path("benchmarks/runs/n100-2000-paired")
    )
    parser.add_argument(
        "--ids-file", type=Path, default=Path("data/performance/performance_ids_2000.json")
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=Path("data/performance/performance_ids_2000.manifest.json"),
    )
    parser.add_argument(
        "--train-ids", type=Path, default=Path("data/splits/train_ids.json")
    )
    parser.add_argument(
        "--validation-ids", type=Path,
        default=Path("data/splits/validation_ids.json"),
    )
    parser.add_argument(
        "--test-ids", type=Path, default=Path("data/splits/test_ids.json")
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("benchmarks/analysis/n100-2000-offline"),
    )
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return float("nan")
    index = min(len(ordered) - 1, max(0, math.ceil(fraction * len(ordered)) - 1))
    return ordered[index]


def summary(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"n": 0, "mean": float("nan"), "sd": float("nan"), "p50": float("nan")}
    return {
        "n": len(values),
        "mean": mean(values),
        "sd": stdev(values) if len(values) > 1 else 0.0,
        "p50": percentile(values, 0.5),
    }


def compact_json_bytes(value: Any) -> int:
    return len(json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=str).encode())


def structure_metrics(value: Any) -> dict[str, int]:
    max_depth = 0
    arrays = 0
    nonempty_leaves = 0

    def visit(node: Any, depth: int) -> None:
        nonlocal max_depth, arrays, nonempty_leaves
        max_depth = max(max_depth, depth)
        if isinstance(node, dict):
            for child in node.values():
                visit(child, depth + 1)
        elif isinstance(node, list):
            arrays += 1
            for child in node:
                visit(child, depth + 1)
        elif node is not None and str(node) != "":
            nonempty_leaves += 1

    visit(value, 0)
    return {
        "raw_bytes": compact_json_bytes(value),
        "top_level_keys": len(value) if isinstance(value, dict) else 0,
        "max_depth": max_depth,
        "array_count": arrays,
        "nonempty_leaf_count": nonempty_leaves,
    }


def load_run_rows(runs_dir: Path) -> dict[tuple[int, str], list[dict[str, Any]]]:
    loaded = {}
    for seed in SEEDS:
        for method in METHODS:
            run = f"n100-s{seed}-{method}-c16-q16"
            path = runs_dir / "raw" / f"{run}.jsonl"
            rows = read_jsonl(path)
            if len(rows) != 2000:
                raise ValueError(f"{run} 记录数不是 2000: {len(rows)}")
            loaded[(seed, method)] = rows
    return loaded


def check_integrity(
    args: argparse.Namespace,
    rows: dict[tuple[int, str], list[dict[str, Any]]],
    manifest: dict[str, Any],
    expected_ids: list[int],
) -> dict[str, Any]:
    train = set(read_json(args.train_ids))
    validation = set(read_json(args.validation_ids))
    test = set(read_json(args.test_ids))
    expected_set = set(expected_ids)
    partition = {
        "train": len(expected_set & train),
        "validation": len(expected_set & validation),
        "test": len(expected_set & test),
        "other": len(expected_set - train - validation - test),
    }
    run_checks = []
    for (seed, method), run_rows in rows.items():
        run = f"n100-s{seed}-{method}-c16-q16"
        run_checks.append({
            "run": run,
            "records": len(run_rows),
            "unique_ids": len({row["artifact_id"] for row in run_rows}),
            "success": sum(row.get("status") == "success" for row in run_rows),
            "errors": sum(row.get("status") != "success" for row in run_rows),
            "model_dir": sorted({row.get("selector_model_dir") for row in run_rows}),
            "run_labels": sorted({row.get("run_label") for row in run_rows}),
            "experiment_ids": sorted({row.get("experiment_id") for row in run_rows}),
            "ids_match": {row["artifact_id"] for row in run_rows} == expected_set,
        })
    return {
        "manifest_hash": manifest.get("ids_sha256"),
        "expected_hash": EXPECTED_HASH,
        "manifest_hash_matches": manifest.get("ids_sha256") == EXPECTED_HASH,
        "manifest_selected_records": manifest.get("selected_records"),
        "expected_records": len(expected_ids),
        "expected_ids_unique": len(expected_set) == len(expected_ids),
        "partition": partition,
        "all_in_train_or_validation": partition["test"] == 0 and partition["other"] == 0,
        "runs": run_checks,
        "all_runs_complete": all(
            item["records"] == 2000 and item["unique_ids"] == 2000
            and item["success"] == 2000 and item["errors"] == 0
            and item["ids_match"] for item in run_checks
        ),
    }


def assign_strata(rows: list[dict[str, Any]]) -> dict[int, int]:
    sizes = sorted(
        (structure_metrics(row["raw_data"])["raw_bytes"], row["artifact_id"])
        for row in rows
    )
    return {
        artifact_id: index // 400 + 1
        for index, (_, artifact_id) in enumerate(sizes)
    }


def structure_report(rows: dict[tuple[int, str], list[dict[str, Any]]]) -> dict[str, Any]:
    base = rows[(42, "full")]
    strata = assign_strata(base)
    metrics_by_id = {
        row["artifact_id"]: structure_metrics(row["raw_data"]) for row in base
    }
    report: dict[str, Any] = {"strata_definition": "按 raw JSON bytes 排序后每 400 条一层"}
    for method in METHODS:
        groups = defaultdict(lambda: defaultdict(list))
        for seed in SEEDS:
            for row in rows[(seed, method)]:
                group = groups[strata[row["artifact_id"]]]
                metrics = metrics_by_id[row["artifact_id"]]
                group["raw_bytes"].append(metrics["raw_bytes"])
                group["top_level_keys"].append(metrics["top_level_keys"])
                group["max_depth"].append(metrics["max_depth"])
                group["array_count"].append(metrics["array_count"])
                group["nonempty_leaf_count"].append(metrics["nonempty_leaf_count"])
                group["total_tokens"].append(float(row["total_tokens"]))
                group["latency_ms"].append(float(row["latency_ms"]))
                group["planning_ms"].append(float(row["planning_ms"]))
                group["queue_wait_ms"].append(float(row["queue_wait_ms"]))
        report[method] = {
            str(stratum): {metric: summary(values) for metric, values in group.items()}
            for stratum, group in sorted(groups.items())
        }
    report["stratum_sizes"] = {str(i): 400 for i in range(1, 6)}
    report["overall_structure"] = {
        metric: summary([values[metric] for values in metrics_by_id.values()])
        for metric in next(iter(metrics_by_id.values()))
    }
    return report


def stability_report(rows: dict[tuple[int, str], list[dict[str, Any]]]) -> dict[str, Any]:
    by_seed = {}
    selections = {}
    for seed in SEEDS:
        selections[seed] = {
            row["artifact_id"]: set(row.get("selected_fields", []))
            for row in rows[(seed, "budget-200")]
        }
        counts = Counter(field for fields in selections[seed].values() for field in fields)
        by_seed[str(seed)] = {
            "selected_field_count": summary([len(fields) for fields in selections[seed].values()]),
            "estimated_field_cost": summary([
                float(row["estimated_field_cost"])
                for row in rows[(seed, "budget-200")]
            ]),
            "field_selection_frequency": {
                field: {"count": count, "rate": count / 2000}
                for field, count in sorted(counts.items())
            },
        }
    pairwise = []
    for left, right in combinations(SEEDS, 2):
        jaccard = []
        exact = []
        for artifact_id in selections[left]:
            intersection = len(selections[left][artifact_id] & selections[right][artifact_id])
            union = len(selections[left][artifact_id] | selections[right][artifact_id])
            jaccard.append(intersection / union if union else 1.0)
            exact.append(selections[left][artifact_id] == selections[right][artifact_id])
        pairwise.append({
            "seed_pair": [left, right],
            "jaccard": summary(jaccard),
            "exact_same_rate": mean(exact),
        })
    cross_counts = Counter(
        field for artifact_id in selections[42]
        for field in set.union(*(selections[seed][artifact_id] for seed in SEEDS))
    )
    return {
        "scope": "budget-200，三个 n=100 seed，按相同 artifact_id 对齐",
        "per_seed": by_seed,
        "pairwise": pairwise,
        "cross_seed_presence": {
            field: {"records": count, "rate": count / 2000}
            for field, count in sorted(cross_counts.items())
        },
    }


def main() -> None:
    args = parse_args()
    expected_ids = read_json(args.ids_file)
    manifest = read_json(args.manifest)
    rows = load_run_rows(args.runs_dir)
    integrity = check_integrity(args, rows, manifest, expected_ids)
    structure = structure_report(rows)
    stability = stability_report(rows)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "integrity_audit.json", integrity)
    write_json(args.output_dir / "structure_stratified_statistics.json", structure)
    write_json(args.output_dir / "selector_stability.json", stability)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "integrity": integrity["all_runs_complete"] and integrity["all_in_train_or_validation"],
        "pairwise_jaccard": [item["jaccard"]["mean"] for item in stability["pairwise"]],
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
