#!/usr/bin/env python3
"""在湖南博物馆 held-out 集上运行预算 Schema 的质量对照。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
COHORT_DIR = ROOT / "data/hunan_museum_300000001_300000500"
IDS = COHORT_DIR / "test_ids.json"
SNAPSHOT = COHORT_DIR / "snapshot.jsonl"
MODEL_DIR = ROOT / "models/hunan_museum_300000001_300000500_n100_m1/seed_42"
PROMPT = ROOT / "1_stable.prompt"
SCHEMA = ROOT / "simple2.json"
OUTPUT_DIR = ROOT / "experiments/hunan_museum_300000001_300000500_budget_quality"
EXPERIMENT_ID = "hunan-museum-300000001-300000500-n100-quality-kg-assist-pro"
RUN_LABEL_PREFIX = "hunan-museum-n100"
METHODS = ("budget-200", "budget-400")
CONCURRENCY = 20
QUEUE_SIZE = 40
MAX_RETRIES = 2


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate(
    path: Path,
    expected_ids: list[int],
    snapshot_rows: dict[int, dict[str, Any]],
    method: str,
    dry_run: bool,
) -> dict[str, Any]:
    rows = read_rows(path)
    artifact_ids = [int(row["artifact_id"]) for row in rows]
    budget = int(method.removeprefix("budget-"))
    run_label = f"{RUN_LABEL_PREFIX}-{method}-stable"
    statuses = {row.get("status") for row in rows}
    valid = (
        len(rows) == len(expected_ids)
        and len(set(artifact_ids)) == len(expected_ids)
        and set(artifact_ids) == set(expected_ids)
        and all(row.get("method") == method for row in rows)
        and all(row.get("run_label") == run_label for row in rows)
        and all(row.get("experiment_id") == EXPERIMENT_ID for row in rows)
        and all(row.get("prompt_variant") == "base" for row in rows)
        and all(row.get("selector_model_dir") == str(MODEL_DIR) for row in rows)
        and all((row.get("prediction_ms") or 0) > 0 for row in rows)
        and all(row.get("budget") == budget for row in rows)
        and all(row.get("estimated_field_cost") is not None for row in rows)
        and all(row.get("estimated_field_cost") <= budget for row in rows)
        and all(row.get("raw_data") == snapshot_rows[row["artifact_id"]]["raw_data"] for row in rows)
    )
    if dry_run:
        valid = valid and statuses == {None}
    else:
        valid = valid and statuses == {"success"} and all(
            set(row.get("result", {})) == set(row.get("selected_fields", []))
            for row in rows
        )
    return {
        "valid": valid,
        "method": method,
        "budget": budget,
        "rows": len(rows),
        "unique_ids": len(set(artifact_ids)),
        "status_counts": {
            str(status): sum(row.get("status") == status for row in rows)
            for status in statuses
        },
        "selected_field_count_range": [
            min(len(row.get("selected_fields", [])) for row in rows),
            max(len(row.get("selected_fields", [])) for row in rows),
        ] if rows else None,
        "estimated_cost_range": [
            min(row.get("estimated_field_cost") or 0 for row in rows),
            max(row.get("estimated_field_cost") or 0 for row in rows),
        ] if rows else None,
        "prompt_hashes": sorted({row.get("prompt_hash") for row in rows}),
        "schema_hashes": sorted({row.get("schema_hash") for row in rows}),
        "raw_data_snapshot_match": all(
            row.get("raw_data") == snapshot_rows[row["artifact_id"]]["raw_data"]
            for row in rows
        ),
    }


def run_one(args: argparse.Namespace, method: str, dry_run: bool, limit: int, smoke: bool) -> dict[str, Any]:
    stage = "dry-run" if dry_run else "smoke" if smoke else "formal"
    run_dir = args.output_dir / stage / method
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / f"{method}.jsonl"
    validation_path = run_dir / "validation.json"
    if raw.exists() and not args.resume:
        raise SystemExit(f"拒绝覆盖已有输出: {raw}")
    if raw.exists() and args.resume:
        previous = json.loads(validation_path.read_text(encoding="utf-8")) if validation_path.exists() else {}
        if previous.get("valid"):
            return previous
        raw.rename(raw.with_name(raw.name + f".partial.{int(time.time())}"))

    command = [
        sys.executable,
        str(ROOT / "llm_runner.py"),
        "--methods",
        method,
        "--ids-file",
        str(IDS),
        "--records-file",
        str(SNAPSHOT),
        "--limit",
        str(limit),
        "--output",
        str(raw),
        "--prompt",
        str(PROMPT),
        "--schema",
        str(SCHEMA),
        "--model-dir",
        str(MODEL_DIR),
        "--max-concurrency",
        str(CONCURRENCY),
        "--queue-size",
        str(QUEUE_SIZE),
        "--max-retries",
        str(MAX_RETRIES),
        "--experiment-id",
        EXPERIMENT_ID,
        "--run-label",
        f"{RUN_LABEL_PREFIX}-{method}-stable",
    ]
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["LLM_TEMPERATURE"] = "0"
    env["LLM_MAX_CONCURRENCY"] = str(CONCURRENCY)
    with (run_dir / "runner.stdout.log").open("w", encoding="utf-8") as stdout, (
        run_dir / "runner.stderr.log"
    ).open("w", encoding="utf-8") as stderr:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    if completed.returncode != 0:
        raise SystemExit(f"{method} Runner 失败，returncode={completed.returncode}，见 {run_dir}")

    expected_ids = [int(value) for value in json.loads(IDS.read_text(encoding="utf-8"))][:limit]
    snapshot_rows = {row["artifact_id"]: row for row in read_rows(SNAPSHOT)}
    actual = raw.with_suffix(".dry_run.jsonl") if dry_run else raw
    report = validate(actual, expected_ids, snapshot_rows, method, dry_run)
    report.update(
        {
            "stage": stage,
            "command": command,
            "snapshot_sha256": sha256(SNAPSHOT),
            "ids_sha256": sha256(IDS),
            "model_sha256": sha256(MODEL_DIR / "model.joblib"),
            "prompt_sha256": sha256(PROMPT),
            "schema_sha256": sha256(SCHEMA),
            "temperature": 0,
            "max_concurrency": CONCURRENCY,
            "queue_size": QUEUE_SIZE,
            "max_retries": MAX_RETRIES,
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    validation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["valid"]:
        raise SystemExit(f"{method} 输出审计失败: {json.dumps(report, ensure_ascii=False)}")
    return report


def main() -> None:
    global COHORT_DIR, IDS, SNAPSHOT, MODEL_DIR, OUTPUT_DIR, EXPERIMENT_ID, RUN_LABEL_PREFIX
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--run-label-prefix", default=RUN_LABEL_PREFIX)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    COHORT_DIR = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    IDS = COHORT_DIR / "test_ids.json"
    SNAPSHOT = COHORT_DIR / "snapshot.jsonl"
    MODEL_DIR = args.model_dir if args.model_dir.is_absolute() else ROOT / args.model_dir
    OUTPUT_DIR = args.output_dir
    EXPERIMENT_ID = args.experiment_id
    RUN_LABEL_PREFIX = args.run_label_prefix
    if args.smoke and not args.execute:
        raise SystemExit("--smoke 需要 --execute")
    required = (IDS, SNAPSHOT, MODEL_DIR / "model.joblib")
    if any(not path.exists() for path in required):
        raise SystemExit("缺少冻结 held-out 输入或 n=100 模型")
    if args.smoke:
        reports = [run_one(args, method, False, 2, True) for method in METHODS]
    elif args.execute:
        reports = [run_one(args, method, False, 350, False) for method in METHODS]
    else:
        reports = [run_one(args, method, True, 350, False) for method in METHODS]
    print(json.dumps({"runs": reports}, ensure_ascii=False))


if __name__ == "__main__":
    main()
