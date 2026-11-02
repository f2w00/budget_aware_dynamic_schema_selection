#!/usr/bin/env python3
"""从冻结的 test split 中生成可复现的 Gold 分层抽样清单。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import psycopg

from llm_runner import load_dotenv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="生成 Gold 分层抽样 ID")
    parser.add_argument("--split-file", type=Path, default=Path("data/splits/test_ids.json"))
    parser.add_argument("--main-size", type=int, default=500)
    parser.add_argument("--overlap-size", type=int, default=75)
    parser.add_argument("--tail-size", type=int, default=75)
    parser.add_argument("--output-dir", type=Path, default=Path("data/gold"))
    parser.add_argument("--seed", default="gold-v1")
    return parser.parse_args()


def raw_text(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def stable_key(artifact_id: int, seed: str) -> int:
    digest = hashlib.sha256(f"{seed}:{artifact_id}".encode()).hexdigest()
    return int(digest, 16)


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    test_ids = json.loads(args.split_file.read_text(encoding="utf-8"))
    with psycopg.connect(database_url) as connection, connection.cursor() as cursor:
        cursor.execute(
            "SELECT artifact_id, raw_data FROM public.artifact_raw "
            "WHERE artifact_id = ANY(%s)",
            (test_ids,),
        )
        records = {int(row[0]): row[1] for row in cursor.fetchall()}

    missing = sorted(set(map(int, test_ids)) - set(records))
    if missing:
        raise SystemExit(f"artifact_raw 缺少 {len(missing)} 条 test 记录")

    rows = []
    for artifact_id, value in records.items():
        length = len(raw_text(value))
        rows.append({
            "artifact_id": artifact_id,
            "raw_chars": length,
            "stable_key": stable_key(artifact_id, args.seed),
        })
    rows.sort(key=lambda row: row["stable_key"])

    # 主集按原始文本长度分层，避免随机样本只覆盖短记录。
    bins = {0: [], 1: [], 2: [], 3: []}
    for index, row in enumerate(sorted(rows, key=lambda item: item["raw_chars"])):
        bins[min(index * 4 // len(rows), 3)].append(row)
    main = []
    for bucket in bins.values():
        quota = round(args.main_size * len(bucket) / len(rows))
        main.extend(sorted(bucket, key=lambda row: row["stable_key"])[:quota])
    main = sorted(main, key=lambda row: row["stable_key"])[:args.main_size]

    main_ids = {row["artifact_id"] for row in main}
    remaining = [row for row in rows if row["artifact_id"] not in main_ids]
    remaining.sort(key=lambda row: (-row["raw_chars"], row["stable_key"]))
    tail = remaining[:args.tail_size]
    tail_ids = {row["artifact_id"] for row in tail}
    # overlap 用于二次审核，必须是主集的稳定子集，不能从剩余数据抽取。
    overlap = sorted(main, key=lambda row: row["stable_key"])[:args.overlap_size]
    overlap_ids = {row["artifact_id"] for row in overlap}
    if len(main) != args.main_size or len(overlap) != args.overlap_size:
        raise SystemExit("抽样数量不足：请检查请求规模是否超过 test 集容量")
    if len(tail) != args.tail_size:
        raise SystemExit("tail 抽样数量不足：请减小 main 或 tail 的请求规模")
    if not overlap_ids <= main_ids or main_ids & tail_ids:
        raise SystemExit("抽样集合关系错误：overlap 应属于 main，tail 应与 main 互斥")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "gold_main_ids.json": [row["artifact_id"] for row in main],
        "gold_overlap_ids.json": [row["artifact_id"] for row in overlap],
        "gold_tail_ids.json": [row["artifact_id"] for row in tail],
    }
    for name, ids in outputs.items():
        (args.output_dir / name).write_text(
            json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    manifest = {
        "source_split": str(args.split_file),
        "seed": args.seed,
        "test_count": len(rows),
        "main_count": len(main),
        "overlap_count": len(overlap),
        "tail_count": len(tail),
        "main_raw_char_min_max": [min(row["raw_chars"] for row in main), max(row["raw_chars"] for row in main)],
        "tail_raw_char_min_max": [min(row["raw_chars"] for row in tail), max(row["raw_chars"] for row in tail)],
    }
    (args.output_dir / "gold_sampling_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
