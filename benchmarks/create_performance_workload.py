#!/usr/bin/env python3
"""从非测试划分构建可复现的分层性能工作负载。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg

# 允许从项目根目录以外直接执行本脚本。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from llm_runner import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="构建冻结的流水线性能工作负载")
    parser.add_argument(
        "--source-ids-files",
        nargs="+",
        type=Path,
        default=[Path("data/splits/train_ids.json"), Path("data/splits/validation_ids.json")],
        help="候选 ID 文件，默认使用训练集和验证集",
    )
    parser.add_argument(
        "--exclude-ids-file",
        type=Path,
        default=Path("data/splits/test_ids.json"),
        help="需排除的 ID 文件，默认排除质量测试集",
    )
    parser.add_argument("--size", type=int, default=2000)
    parser.add_argument("--strata", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20260904)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/performance/performance_ids_2000.json"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/performance/performance_ids_2000.manifest.json"),
    )
    return parser.parse_args()


def load_ids(path: Path) -> list[int]:
    values = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(values, list) or not all(isinstance(value, int) for value in values):
        raise ValueError(f"ID 文件必须为整数 JSON 数组: {path}")
    if len(values) != len(set(values)):
        raise ValueError(f"ID 文件包含重复 artifact_id: {path}")
    return values


def load_records(database_url: str, ids: list[int]) -> list[tuple[int, Any]]:
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT artifact_id, raw_data FROM public.artifact_raw "
            "WHERE artifact_id = ANY(%s) ORDER BY artifact_id",
            (ids,),
        )
        return list(cursor.fetchall())


def raw_length(raw_data: Any) -> int:
    return len(
        json.dumps(raw_data, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8")
    )


def stratified_sample(
    records: list[tuple[int, Any]], size: int, strata: int, seed: int
) -> tuple[list[int], list[dict[str, int]]]:
    if size < 1 or strata < 1:
        raise ValueError("size 和 strata 必须为正数")
    if size > len(records):
        raise ValueError(f"请求 {size} 条，但候选记录仅有 {len(records)} 条")
    if strata > len(records):
        raise ValueError("strata 不能大于候选记录数")

    ranked = sorted((raw_length(raw_data), artifact_id) for artifact_id, raw_data in records)
    bucket_sizes = [
        len(ranked) * (index + 1) // strata - len(ranked) * index // strata
        for index in range(strata)
    ]
    quotas = [size * bucket_size / len(ranked) for bucket_size in bucket_sizes]
    counts = [int(quota) for quota in quotas]
    remainder = size - sum(counts)
    for index in sorted(
        range(strata), key=lambda value: quotas[value] - counts[value], reverse=True
    )[:remainder]:
        counts[index] += 1
    sampled: list[int] = []
    summary = []
    rng = random.Random(seed)

    # 按记录长度的等频区间抽取，使性能负载覆盖短、中、长输入。
    for index in range(strata):
        start = len(ranked) * index // strata
        end = len(ranked) * (index + 1) // strata
        bucket = ranked[start:end]
        count = counts[index]
        selected = rng.sample(bucket, count)
        sampled.extend(artifact_id for _, artifact_id in selected)
        summary.append(
            {
                "stratum": index + 1,
                "candidate_records": len(bucket),
                "selected_records": count,
                "min_raw_bytes": bucket[0][0],
                "max_raw_bytes": bucket[-1][0],
            }
        )

    rng.shuffle(sampled)
    return sampled, summary


def ids_hash(ids: list[int]) -> str:
    payload = json.dumps(ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.environ.get("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")

    source_ids = []
    for path in args.source_ids_files:
        source_ids.extend(load_ids(path))
    if len(source_ids) != len(set(source_ids)):
        raise SystemExit("候选 ID 文件之间存在重复 artifact_id")
    excluded_ids = set(load_ids(args.exclude_ids_file)) if args.exclude_ids_file else set()
    candidate_ids = [artifact_id for artifact_id in source_ids if artifact_id not in excluded_ids]
    excluded_from_source = len(source_ids) - len(candidate_ids)
    records = load_records(database_url, candidate_ids)
    found_ids = {artifact_id for artifact_id, _ in records}
    missing_ids = sorted(set(candidate_ids) - found_ids)
    if missing_ids:
        raise SystemExit(f"artifact_raw 缺少 {len(missing_ids)} 个候选 ID")

    ids, strata = stratified_sample(records, args.size, args.strata, args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "workload_type": "stratified_raw_length",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_ids_files": [str(path) for path in args.source_ids_files],
        "exclude_ids_file": str(args.exclude_ids_file) if args.exclude_ids_file else None,
        "source_records": len(source_ids),
        "excluded_records": excluded_from_source,
        "candidate_records": len(records),
        "selected_records": len(ids),
        "strata": args.strata,
        "seed": args.seed,
        "ids_sha256": ids_hash(ids),
        "stratum_summary": strata,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
