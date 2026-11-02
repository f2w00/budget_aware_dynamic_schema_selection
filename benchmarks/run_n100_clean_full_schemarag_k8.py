#!/usr/bin/env python3
"""在冻结负载上串行运行六次 clean SchemaRAG k=8 端到端实验。"""

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

from benchmark_pipeline import (
    parse_runner_summary,
    read_jsonl,
    summarize,
    validate,
    write_summaries,
)


METHOD = "external-schemarag-k8"
COHORT = "n100-2000-clean-full-schemarag-k8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 clean SchemaRAG k=8 的 12k 端到端实验")
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=Path("data/performance/n100_clean_full_2000_snapshot.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/performance/n100_clean_full_2000_snapshot.manifest.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("benchmarks/runs/n100-2000-clean-full-schemarag-k8"),
    )
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
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_ids(path: Path) -> list[int]:
    ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(ids, list) or not all(isinstance(item, int) for item in ids):
        raise ValueError(f"顺序文件格式错误: {path}")
    if len(ids) != len(set(ids)):
        raise ValueError(f"顺序文件包含重复 ID: {path}")
    return ids


def run_name(pair: dict[str, Any]) -> str:
    return (
        f"pair{pair['pair_index']}-s{pair['selector_seed']}-r{pair['repeat']}-"
        "external-schemarag-k8-c16-q16"
    )


def run_paths(output_dir: Path, name: str, dry_run: bool) -> dict[str, Path]:
    suffix = ".dry" if dry_run else ""
    raw_suffix = ".dry_run.jsonl" if dry_run else ".jsonl"
    return {
        "raw": output_dir / "raw" / f"{name}{raw_suffix}",
        "stdout": output_dir / "logs" / f"{name}{suffix}.stdout.log",
        "stderr": output_dir / "logs" / f"{name}{suffix}.stderr.log",
        "validation": output_dir / "validation" / f"{name}{suffix}.json",
        "meta": output_dir / f"{name}{suffix}.meta.json",
    }


def complete_run(paths: dict[str, Path]) -> bool:
    if not paths["raw"].exists() or not paths["validation"].exists():
        return False
    return bool(json.loads(paths["validation"].read_text(encoding="utf-8")).get("valid"))


def archive_run(output_dir: Path, paths: dict[str, Path], name: str) -> None:
    archive = output_dir / "partial_runs" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
    archive.mkdir(parents=True, exist_ok=False)
    for path in paths.values():
        if path.exists():
            shutil.move(str(path), archive / path.name)


def run_one(
    args: argparse.Namespace,
    manifest: dict[str, Any],
    pair: dict[str, Any],
    dry_run: bool,
) -> dict[str, Any]:
    name = run_name(pair)
    experiment_id = f"{COHORT}-{name}"
    paths = run_paths(args.output_dir, name, dry_run)
    if any(path.exists() for path in paths.values()):
        raise FileExistsError(f"拒绝覆盖既有运行产物: {name}")
    ids_file = Path(pair["ids_file"])
    ids = load_ids(ids_file)
    expected_ids = ids[: args.limit] if args.limit > 0 else ids
    command = [
        args.python_executable,
        str(args.runner),
        "--methods",
        METHOD,
        "--ids-file",
        str(ids_file),
        "--records-file",
        str(args.snapshot),
        "--limit",
        str(args.limit),
        "--max-concurrency",
        str(args.max_concurrency),
        "--queue-size",
        str(args.queue_size),
        "--max-retries",
        str(args.max_retries),
        "--run-label",
        name,
        "--experiment-id",
        experiment_id,
        "--output",
        str(args.output_dir / "raw" / f"{name}.jsonl"),
        "--allow-extra-fields",
    ]
    if dry_run:
        command.append("--dry-run")

    environment = os.environ.copy()
    environment["LLM_TEMPERATURE"] = str(args.temperature)
    started_at = time.time()
    started = time.perf_counter()
    completed = subprocess.run(
        command, capture_output=True, text=True, env=environment, check=False
    )
    wall_clock_ms = round((time.perf_counter() - started) * 1000, 2)
    paths["stdout"].write_text(completed.stdout, encoding="utf-8")
    paths["stderr"].write_text(completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        raise RuntimeError(f"运行失败 {name}，详见 {paths['stderr']}")

    runner_meta = parse_runner_summary(completed.stdout)
    rows = read_jsonl(paths["raw"])
    validation = validate(
        rows,
        expected_ids,
        [METHOD],
        name,
        runner_meta.get("submitted"),
        experiment_id,
    )
    validation["returncode"] = completed.returncode
    write_json(paths["validation"], validation)
    if not validation["valid"]:
        raise RuntimeError(f"运行校验失败 {name}")

    meta = {
        "cohort": COHORT,
        "run_id": name,
        "experiment_id": experiment_id,
        "method": METHOD,
        "pair_index": pair["pair_index"],
        "selector_seed": pair["selector_seed"],
        "repeat": pair["repeat"],
        "source_pair_sequence": pair["sequence"],
        "selector_training_size": None,
        "selector_model_dir": None,
        "clean_schemarag": True,
        "top_k": 8,
        "max_concurrency": args.max_concurrency,
        "queue_size": args.queue_size,
        "max_retries": args.max_retries,
        "temperature": args.temperature,
        "ids_file": str(ids_file),
        "ids_sha256": pair["ids_sha256"],
        "snapshot_file": str(args.snapshot),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "raw_data_sha256": manifest["raw_data_sha256"],
        "records_requested": len(expected_ids),
        "dry_run": dry_run,
        "started_at_unix": started_at,
        "wall_clock_ms": wall_clock_ms,
        "command": command,
        "returncode": completed.returncode,
        **runner_meta,
    }
    write_json(paths["meta"], meta)
    return summarize(rows, meta) | {"validation_valid": True}


def main() -> None:
    args = parse_args()
    if args.limit < 0 or args.max_concurrency != 16 or args.queue_size != 16:
        raise SystemExit("协议要求 limit>=0 且 max-concurrency=queue-size=16")
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if sha256_file(args.snapshot) != manifest.get("snapshot_sha256"):
        raise ValueError("冻结快照 SHA256 与 manifest 不匹配")
    pairs = manifest.get("pair_orders", [])
    if len(pairs) != 6:
        raise ValueError("SchemaRAG 增强协议必须包含六个顺序")

    for directory in ("raw", "logs", "validation", "partial_runs"):
        (args.output_dir / directory).mkdir(parents=True, exist_ok=True)
    dry_run = not args.execute
    summary_dir = args.output_dir / "dry_run" if dry_run else args.output_dir
    summary_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        args.output_dir / f"cohort{'.dry' if dry_run else ''}.json",
        {
            "cohort": COHORT,
            "formal_logical_requests": 12000,
            "method": METHOD,
            "top_k": 8,
            "pairs": pairs,
            "snapshot_file": str(args.snapshot),
            "snapshot_sha256": manifest["snapshot_sha256"],
            "temperature": args.temperature,
            "max_concurrency": args.max_concurrency,
            "queue_size": args.queue_size,
            "max_retries": args.max_retries,
            "dry_run": dry_run,
        },
    )

    summaries = []
    for pair in pairs:
        name = run_name(pair)
        paths = run_paths(args.output_dir, name, dry_run)
        if args.resume and not dry_run and complete_run(paths):
            meta = json.loads(paths["meta"].read_text(encoding="utf-8"))
            summaries.append(summarize(read_jsonl(paths["raw"]), meta) | {"validation_valid": True})
        else:
            if args.resume and not dry_run and any(path.exists() for path in paths.values()):
                archive_run(args.output_dir, paths, name)
            summaries.append(run_one(args, manifest, pair, dry_run))
        write_summaries(summary_dir, summaries)

    validations = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted((args.output_dir / "validation").glob("*.json"))
        if (".dry." in path.name) == dry_run
    ]
    write_json(
        args.output_dir / f"validation{'.dry' if dry_run else ''}.json",
        {"valid": len(validations) == 6 and all(row["valid"] for row in validations), "runs": validations},
    )
    print(json.dumps({"runs": len(summaries), "dry_run": dry_run}, ensure_ascii=False))


if __name__ == "__main__":
    main()
