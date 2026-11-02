#!/usr/bin/env python3
"""在同一冻结顺序上串行运行 full、budget-200 和 SchemaRAG k=8。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from benchmark_pipeline import parse_runner_summary, read_jsonl, summarize, validate, write_summaries


COHORT = "n100-2000-three-way-paired"
METHODS = ("full", "budget-200", "external-schemarag-k8")
SEQUENCE = (
    ("full", "budget-200", "external-schemarag-k8"),
    ("budget-200", "external-schemarag-k8", "full"),
    ("external-schemarag-k8", "full", "budget-200"),
    ("full", "external-schemarag-k8", "budget-200"),
    ("external-schemarag-k8", "budget-200", "full"),
    ("budget-200", "full", "external-schemarag-k8"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行三方法冻结负载配对实验")
    parser.add_argument("--snapshot", type=Path, default=Path("data/performance/n100_clean_full_2000_snapshot.jsonl"))
    parser.add_argument("--manifest", type=Path, default=Path("data/performance/n100_clean_full_2000_snapshot.manifest.json"))
    parser.add_argument("--model-root", type=Path, default=Path("models/model_strategy_ablation/m1"))
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/runs/n100-2000-three-way-paired"))
    parser.add_argument("--runner", type=Path, default=Path("llm_runner.py"))
    parser.add_argument("--python", dest="python_executable", default=sys.executable)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--max-concurrency", type=int, default=16)
    parser.add_argument("--queue-size", type=int, default=16)
    parser.add_argument("--max-retries", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_ids(path: Path) -> list[int]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
        raise ValueError(f"顺序文件格式错误: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"顺序文件包含重复 ID: {path}")
    return ids


def build_cohort_manifest(source: dict[str, Any], snapshot: Path) -> dict[str, Any]:
    source_orders = source.get("pair_orders", [])
    if len(source_orders) != 6:
        raise ValueError("源快照必须包含六个冻结顺序")
    blocks = []
    for index, (source_order, sequence) in enumerate(zip(source_orders, SEQUENCE, strict=True), 1):
        ids_file = Path(source_order["ids_file"])
        ids = load_ids(ids_file)
        blocks.append({
            "block_index": index,
            "selector_seed": source_order["selector_seed"],
            "repeat": source_order["repeat"],
            "sequence": list(sequence),
            "ids_file": str(ids_file),
            "ids_sha256": hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest(),
            "source_pair_index": source_order["pair_index"],
        })
    return {
        "cohort": COHORT,
        "workload_type": "frozen_raw_data_snapshot",
        "methods": list(METHODS),
        "formal_logical_requests": 6 * 3 * 2000,
        "snapshot_file": str(snapshot),
        "snapshot_sha256": sha256_file(snapshot),
        "source_snapshot_manifest": str(source.get("snapshot_file", "")),
        "schema_hash": source.get("schema_hash"),
        "prompt_hash": source.get("prompt_hash"),
        "max_concurrency": 16,
        "queue_size": 16,
        "max_retries": 2,
        "temperature": 0.0,
        "blocks": blocks,
    }


def run_name(block: dict[str, Any], method: str) -> str:
    return f"block{block['block_index']}-s{block['selector_seed']}-r{block['repeat']}-{method}-c16-q16"


def paths(output_dir: Path, name: str, dry_run: bool) -> dict[str, Path]:
    log_suffix = ".dry" if dry_run else ""
    raw_suffix = ".dry_run.jsonl" if dry_run else ".jsonl"
    return {
        "raw": output_dir / "raw" / f"{name}{raw_suffix}",
        "stdout": output_dir / "logs" / f"{name}{log_suffix}.stdout.log",
        "stderr": output_dir / "logs" / f"{name}{log_suffix}.stderr.log",
        "validation": output_dir / "validation" / f"{name}{log_suffix}.json",
        "meta": output_dir / f"{name}{log_suffix}.meta.json",
    }


def valid_run(path_set: dict[str, Path], ids: list[int], method: str, name: str, experiment_id: str) -> bool:
    if not path_set["raw"].exists() or not path_set["validation"].exists() or not path_set["meta"].exists():
        return False
    validation = json.loads(path_set["validation"].read_text(encoding="utf-8"))
    if not validation.get("valid"):
        return False
    rows = read_jsonl(path_set["raw"])
    if len(rows) != len(ids) or {row.get("artifact_id") for row in rows} != set(ids):
        return False
    return all(
        row.get("method") == method
        and row.get("run_label") == name
        and row.get("experiment_id") == experiment_id
        and row.get("status") == "success"
        for row in rows
    )


def archive_block(output_dir: Path, block: dict[str, Any]) -> None:
    stamp = time.strftime("%Y%m%d-%H%M%S")
    archive = output_dir / "partial_blocks" / f"block{block['block_index']}-{stamp}"
    archive.mkdir(parents=True, exist_ok=False)
    for method in block["sequence"]:
        name = run_name(block, method)
        for path_set in (paths(output_dir, name, False), paths(output_dir, name, True)):
            for path in path_set.values():
                if path.exists():
                    shutil.move(str(path), archive / path.name)


def run_one(args: argparse.Namespace, manifest: dict[str, Any], block: dict[str, Any], method: str, dry_run: bool) -> dict[str, Any]:
    name = run_name(block, method)
    experiment_id = f"{COHORT}-{name}"
    path_set = paths(args.output_dir, name, dry_run)
    if any(path.exists() for path in path_set.values()):
        raise FileExistsError(f"拒绝覆盖既有运行产物: {name}")
    ids_file = Path(block["ids_file"])
    ids = load_ids(ids_file)
    expected_ids = ids[: args.limit] if args.limit > 0 else ids
    model_dir = args.model_root / f"seed_{block['selector_seed']}"
    if method == "budget-200" and not (model_dir / "model.joblib").exists():
        raise FileNotFoundError(f"缺少 n=100 M1 模型: {model_dir}")
    command = [
        args.python_executable, str(args.runner), "--methods", method,
        "--ids-file", str(ids_file), "--records-file", str(args.snapshot),
        "--limit", str(args.limit), "--max-concurrency", str(args.max_concurrency),
        "--queue-size", str(args.queue_size), "--max-retries", str(args.max_retries),
        "--run-label", name, "--experiment-id", experiment_id,
        "--output", str(args.output_dir / "raw" / f"{name}.jsonl"),
        "--allow-extra-fields",
    ]
    if method == "budget-200":
        command.extend(["--model-dir", str(model_dir)])
    if dry_run:
        command.append("--dry-run")
    environment = os.environ.copy()
    environment["LLM_TEMPERATURE"] = str(args.temperature)
    started_at = time.time()
    started = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True, env=environment, check=False)
    wall_clock_ms = round((time.perf_counter() - started) * 1000, 2)
    path_set["stdout"].write_text(completed.stdout, encoding="utf-8")
    path_set["stderr"].write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"运行失败 {name}，详见 {path_set['stderr']}")
    runner_meta = parse_runner_summary(completed.stdout)
    rows = read_jsonl(path_set["raw"])
    validation = validate(rows, expected_ids, [method], name, runner_meta.get("submitted"), experiment_id)
    validation["returncode"] = completed.returncode
    validation["all_success"] = (
        True
        if dry_run
        else all(row.get("status") == "success" for row in rows)
    )
    validation["expected_snapshot_sha256"] = manifest["snapshot_sha256"]
    validation["valid"] = (
        validation["valid"]
        and validation["all_success"]
        and len(rows) == len(expected_ids)
    )
    write_json(path_set["validation"], validation)
    if not validation["valid"]:
        raise RuntimeError(f"运行校验失败 {name}")
    meta = {
        "cohort": COHORT, "run_id": name, "experiment_id": experiment_id,
        "method": method, "block_index": block["block_index"],
        "selector_seed": block["selector_seed"], "repeat": block["repeat"],
        "order_position": block["sequence"].index(method) + 1,
        "sequence": block["sequence"], "ids_file": str(ids_file),
        "ids_sha256": block["ids_sha256"], "snapshot_file": str(args.snapshot),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "raw_data_sha256": manifest.get("raw_data_sha256"),
        "schema_hash": manifest.get("schema_hash"), "prompt_hash": manifest.get("prompt_hash"),
        "selector_training_size": 100 if method == "budget-200" else None,
        "selector_model_dir": str(model_dir) if method == "budget-200" else None,
        "clean_full": method == "full", "top_k": 8 if method == "external-schemarag-k8" else None,
        "max_concurrency": args.max_concurrency, "queue_size": args.queue_size,
        "max_retries": args.max_retries, "temperature": args.temperature,
        "records_requested": len(expected_ids), "dry_run": dry_run,
        "started_at_unix": started_at, "wall_clock_ms": wall_clock_ms,
        "command": command, "returncode": completed.returncode, **runner_meta,
    }
    write_json(path_set["meta"], meta)
    return summarize(rows, meta) | {"validation_valid": True}


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.max_concurrency != 16 or args.queue_size != 16 or args.max_retries != 2:
        raise SystemExit("协议要求 limit>=0、max-concurrency=16、queue-size=16、max-retries=2")
    source = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256_file(args.snapshot) != source.get("snapshot_sha256"):
        raise ValueError("冻结快照 SHA256 与源 manifest 不匹配")
    manifest = build_cohort_manifest(source, args.snapshot)
    dry_run = not args.execute
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.resume and any(args.output_dir.iterdir()):
        raise SystemExit("正式/预检输出目录非空；如需恢复必须显式使用 --resume")
    for directory in ("raw", "logs", "validation", "partial_blocks"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / f"cohort{'.dry' if dry_run else ''}.json", manifest | {"dry_run": dry_run})
    summary_dir = args.output_dir / "dry_run" if dry_run else args.output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for block in manifest["blocks"]:
        block_paths = [paths(args.output_dir, run_name(block, method), dry_run) for method in METHODS]
        block_ids = load_ids(Path(block["ids_file"]))
        block_complete = all(
            valid_run(path_set, block_ids[: args.limit] if args.limit > 0 else block_ids,
                      method, run_name(block, method), f"{COHORT}-{run_name(block, method)}")
            for path_set, method in zip(block_paths, METHODS, strict=True)
        )
        if args.resume and not dry_run and block_complete:
            for path_set in block_paths:
                meta = json.loads(path_set["meta"].read_text(encoding="utf-8"))
                summaries.append(summarize(read_jsonl(path_set["raw"]), meta) | {"validation_valid": True})
            write_summaries(summary_dir, summaries)
            continue
        if args.resume and not dry_run and any(path.exists() for path_set in block_paths for path in path_set.values()):
            archive_block(args.output_dir, block)
        try:
            for method in block["sequence"]:
                summaries.append(run_one(args, manifest, block, method, dry_run))
                write_summaries(summary_dir, summaries)
        except Exception:
            if not dry_run:
                archive_block(args.output_dir, block)
            raise
    write_json(args.output_dir / f"validation{'.dry' if dry_run else ''}.json", {
        "valid": len(summaries) == 18 and all(row.get("validation_valid") for row in summaries),
        "runs": summaries,
    })
    print(json.dumps({"runs": len(summaries), "blocks": 6, "dry_run": dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
