#!/usr/bin/env python3
"""运行湖南博物馆固定 500 条 full silver，并校验完整性。"""

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
IDS = COHORT_DIR / "ids.json"
SNAPSHOT = COHORT_DIR / "snapshot.jsonl"
PROMPT = ROOT / "1_stable.prompt"
SCHEMA = ROOT / "simple2.json"
OUTPUT_DIR = ROOT / "experiments/hunan_museum_300000001_300000500_full_silver"
RUN_LABEL = "hunan-museum-300000001-300000500-full-stable"
EXPERIMENT_ID = "hunan-museum-300000001-300000500-full-silver-kg-assist-pro"


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
    dry_run: bool,
) -> dict[str, Any]:
    rows = read_rows(path)
    artifact_ids = [int(row["artifact_id"]) for row in rows]
    statuses = {row.get("status") for row in rows}
    valid = (
        len(rows) == len(expected_ids)
        and len(set(artifact_ids)) == len(expected_ids)
        and set(artifact_ids) == set(expected_ids)
        and all(row.get("method") == "full" for row in rows)
        and all(row.get("run_label") == RUN_LABEL for row in rows)
        and all(row.get("experiment_id") == EXPERIMENT_ID for row in rows)
        and all(row.get("prompt_variant") == "base" for row in rows)
        and all(len(row.get("selected_fields", [])) == 18 for row in rows)
        and all(row.get("selector_model_dir") is None for row in rows)
        and all((row.get("prediction_ms") or 0) == 0 for row in rows)
        and all(
            row.get("raw_data") == snapshot_rows[row["artifact_id"]]["raw_data"]
            for row in rows
        )
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
        "rows": len(rows),
        "unique_ids": len(set(artifact_ids)),
        "status_counts": {
            str(status): sum(row.get("status") == status for row in rows)
            for status in statuses
        },
        "selected_field_counts": sorted({len(row.get("selected_fields", [])) for row in rows}),
        "prompt_hashes": sorted({row.get("prompt_hash") for row in rows}),
        "schema_hashes": sorted({row.get("schema_hash") for row in rows}),
        "raw_data_snapshot_match": all(
            row.get("raw_data") == snapshot_rows[row["artifact_id"]]["raw_data"]
            for row in rows
        ),
    }


def run(args: argparse.Namespace, dry_run: bool, limit: int, smoke: bool) -> dict[str, Any]:
    stage = "dry-run" if dry_run else "smoke" if smoke else "formal"
    run_dir = args.output_dir / stage
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / "full.jsonl"
    validation_path = run_dir / "validation.json"
    if raw.exists() and not args.resume:
        raise SystemExit(f"拒绝覆盖已有输出: {raw}")
    if raw.exists() and args.resume:
        expected_ids = [int(value) for value in json.loads(IDS.read_text(encoding="utf-8"))]
        snapshot_rows = {row["artifact_id"]: row for row in read_rows(SNAPSHOT)}
        actual = raw.with_suffix(".dry_run.jsonl") if dry_run else raw
        previous = validate(actual, expected_ids[:limit], snapshot_rows, dry_run)
        if previous["valid"]:
            return previous
        raw.rename(raw.with_name(raw.name + f".partial.{int(time.time())}"))

    command = [
        sys.executable,
        str(ROOT / "llm_runner.py"),
        "--methods",
        "full",
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
        "--max-concurrency",
        str(args.max_concurrency),
        "--queue-size",
        str(args.queue_size),
        "--max-retries",
        "2",
        "--experiment-id",
        EXPERIMENT_ID,
        "--run-label",
        RUN_LABEL,
    ]
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["LLM_TEMPERATURE"] = "0"
    env["LLM_MAX_CONCURRENCY"] = "1"
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
        raise SystemExit(f"Runner 失败，returncode={completed.returncode}，见 {run_dir}")

    expected_ids = [int(value) for value in json.loads(IDS.read_text(encoding="utf-8"))]
    snapshot_rows = {row["artifact_id"]: row for row in read_rows(SNAPSHOT)}
    actual = raw.with_suffix(".dry_run.jsonl") if dry_run else raw
    report = validate(actual, expected_ids[:limit], snapshot_rows, dry_run)
    report.update(
        {
            "stage": stage,
            "command": command,
            "snapshot_sha256": sha256(SNAPSHOT),
            "ids_sha256": sha256(IDS),
            "prompt_sha256": sha256(PROMPT),
            "schema_sha256": sha256(SCHEMA),
            "temperature": 0,
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "max_retries": 2,
            "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
    )
    validation_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if not report["valid"]:
        raise SystemExit(f"输出审计失败: {json.dumps(report, ensure_ascii=False)}")
    return report


def main() -> None:
    global COHORT_DIR, IDS, SNAPSHOT, OUTPUT_DIR, RUN_LABEL, EXPERIMENT_ID
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--cohort-dir", type=Path, default=COHORT_DIR)
    parser.add_argument("--run-label", default=RUN_LABEL)
    parser.add_argument("--experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--max-concurrency", type=int, default=1)
    parser.add_argument("--queue-size", type=int, default=1)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    COHORT_DIR = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    IDS = COHORT_DIR / "ids.json"
    SNAPSHOT = COHORT_DIR / "snapshot.jsonl"
    OUTPUT_DIR = args.output_dir
    RUN_LABEL = args.run_label
    EXPERIMENT_ID = args.experiment_id
    if args.smoke and not args.execute:
        raise SystemExit("--smoke 需要 --execute")
    if args.max_concurrency < 1 or args.queue_size < 1:
        raise SystemExit("并发和队列大小必须为正整数")
    if not IDS.exists() or not SNAPSHOT.exists():
        raise SystemExit("缺少冻结 cohort")
    if args.smoke:
        report = run(args, dry_run=False, limit=2, smoke=True)
    elif args.execute:
        report = run(args, dry_run=False, limit=500, smoke=False)
    else:
        report = run(args, dry_run=True, limit=500, smoke=False)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
