#!/usr/bin/env python3
"""运行并审计独立的 prompt-omit 500 条实验。"""

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
IDS = ROOT / "data/gold/gold_main_ids.json"
SNAPSHOT = ROOT / "data/gold/prompt_omit_500_snapshot.jsonl"
PROMPT = ROOT / "1_stable_omit.prompt"
SCHEMA = ROOT / "simple2.json"
OUTPUT_DIR = ROOT / "experiments/prompt_omit_500_formal"
RUN_LABEL = "prompt-omit-500-stable-omit"
EXPERIMENT_ID = "prompt-omit-500-stable-omit-kg-assist-pro"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def validate_raw(
    path: Path,
    expected_ids: list[int],
    snapshot_rows: dict[int, dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    rows = read_rows(path)
    ids = [row.get("artifact_id") for row in rows]
    statuses = {row.get("status") for row in rows}
    expected_set = set(expected_ids)
    actual_set = set(ids)
    valid = (
        len(rows) == len(expected_ids)
        and len(actual_set) == len(expected_ids)
        and actual_set == expected_set
        and all(row.get("method") == "prompt-omit" for row in rows)
        and all(row.get("run_label") == RUN_LABEL for row in rows)
        and all(row.get("experiment_id") == EXPERIMENT_ID for row in rows)
        and all(row.get("prompt_variant") == "prompt_omit" for row in rows)
        and all(len(row.get("selected_fields", [])) == 18 for row in rows)
        and all(row.get("selector_model_dir") is None for row in rows)
        and all((row.get("prediction_ms") or 0) == 0 for row in rows)
        and all(row.get("missing_fields_expected") is True for row in rows)
        and all(
            row.get("raw_data") == snapshot_rows.get(row.get("artifact_id"), {}).get("raw_data")
            for row in rows
        )
    )
    if dry_run:
        valid = valid and statuses == {None}
    else:
        valid = valid and statuses == {"success"}
    return {
        "valid": valid,
        "rows": len(rows),
        "unique_ids": len(actual_set),
        "status_counts": {str(status): sum(row.get("status") == status for row in rows)
                           for status in statuses},
        "selected_field_counts": sorted({len(row.get("selected_fields", [])) for row in rows}),
        "prompt_hashes": sorted({row.get("prompt_hash") for row in rows}),
        "schema_hashes": sorted({row.get("schema_hash") for row in rows}),
        "missing_fields_expected": sorted({row.get("missing_fields_expected") for row in rows}),
        "prediction_ms_total": sum(row.get("prediction_ms") or 0 for row in rows),
        "selector_model_dirs": sorted({row.get("selector_model_dir") for row in rows}),
        "prompt_variants": sorted({row.get("prompt_variant") for row in rows}),
        "raw_data_snapshot_match": all(
            row.get("raw_data") == snapshot_rows.get(row.get("artifact_id"), {}).get("raw_data")
            for row in rows
        ),
    }


def run(
    args: argparse.Namespace,
    ids: list[int],
    snapshot_rows: dict[int, dict[str, Any]],
    dry_run: bool,
    limit: int,
    smoke: bool = False,
) -> dict[str, Any]:
    suffix = "dry-run" if dry_run else "smoke" if smoke else "formal"
    run_dir = args.output_dir / suffix
    run_dir.mkdir(parents=True, exist_ok=True)
    raw = run_dir / ("prompt-omit-smoke.jsonl" if smoke else "prompt-omit.jsonl")
    stdout = run_dir / "runner.stdout.log"
    stderr = run_dir / "runner.stderr.log"
    validation = run_dir / "validation.json"
    if raw.exists() and not args.resume:
        raise SystemExit(f"拒绝覆盖已有输出，请使用 --resume: {raw}")
    if raw.exists() and args.resume:
        existing_report = None
        if validation.exists():
            existing_report = json.loads(validation.read_text(encoding="utf-8"))
        if existing_report and existing_report.get("valid"):
            return existing_report
        old = raw.with_name(raw.name + f".partial.{int(time.time())}")
        raw.rename(old)

    command = [
        sys.executable, str(ROOT / "llm_runner.py"),
        "--methods", "prompt-omit", "--ids-file", str(IDS),
        "--records-file", str(SNAPSHOT), "--limit", str(limit),
        "--output", str(raw), "--prompt", str(ROOT / "1_stable.prompt"),
        "--prompt-omit-template", str(PROMPT), "--schema", str(SCHEMA),
        "--max-concurrency", "20", "--queue-size", "40", "--max-retries", "2",
        "--experiment-id", EXPERIMENT_ID, "--run-label", RUN_LABEL,
    ]
    if dry_run:
        command.append("--dry-run")
    env = os.environ.copy()
    env["LLM_TEMPERATURE"] = "0"
    env["LLM_MAX_CONCURRENCY"] = "20"
    completed = subprocess.run(command, cwd=ROOT, env=env, text=True,
                               stdout=stdout.open("w"), stderr=stderr.open("w"),
                               check=False)
    if completed.returncode != 0:
        raise SystemExit(f"Runner 失败，returncode={completed.returncode}，见 {stderr}")
    result_path = raw.with_suffix(".dry_run.jsonl") if dry_run else raw
    report = validate_raw(result_path, ids[:limit], snapshot_rows, dry_run)
    report.update({"dry_run": dry_run, "command": command, "returncode": completed.returncode,
                   "snapshot_sha256": sha256(SNAPSHOT), "prompt_sha256": sha256(PROMPT),
                   "schema_sha256": sha256(SCHEMA), "validated_at": time.strftime("%Y-%m-%dT%H:%M:%S")})
    validation.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if not report["valid"]:
        raise SystemExit(f"输出审计失败: {json.dumps(report, ensure_ascii=False)}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_DIR)
    parser.add_argument("--execute", action="store_true", help="执行真实 LLM 调用")
    parser.add_argument("--smoke", action="store_true", help="仅运行 2 条，不纳入正式结果")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    ids = json.loads(IDS.read_text(encoding="utf-8"))
    if len(ids) != 500 or len(ids) != len(set(ids)):
        raise SystemExit("Gold ID 文件必须包含 500 个唯一 ID")
    if not SNAPSHOT.exists():
        raise SystemExit(f"缺少冻结快照，请先运行 freeze_prompt_omit_500.py: {SNAPSHOT}")
    if args.smoke and not args.execute:
        raise SystemExit("--smoke 需要 --execute")
    snapshot_rows = {
        row["artifact_id"]: row
        for row in read_rows(SNAPSHOT)
    }
    if set(snapshot_rows) != set(ids):
        raise SystemExit("冻结快照中的 artifact_id 集合与 Gold ID 不一致")
    if args.smoke:
        report = run(args, ids, snapshot_rows, dry_run=False, limit=2, smoke=True)
    elif args.execute:
        report = run(args, ids, snapshot_rows, dry_run=False, limit=500)
    else:
        report = run(args, ids, snapshot_rows, dry_run=True, limit=500)
    print(json.dumps(report, ensure_ascii=False))


if __name__ == "__main__":
    main()
