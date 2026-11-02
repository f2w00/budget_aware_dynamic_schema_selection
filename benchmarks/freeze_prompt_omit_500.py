#!/usr/bin/env python3
"""冻结 prompt-omit 正式实验所需的 500 条 raw_data。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import psycopg

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from llm_runner import load_dotenv


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ids", type=Path, default=Path("data/gold/gold_main_ids.json"))
    parser.add_argument("--output", type=Path,
                        default=Path("data/gold/prompt_omit_500_snapshot.jsonl"))
    parser.add_argument("--manifest", type=Path,
                        default=Path("data/gold/prompt_omit_500_snapshot.manifest.json"))
    args = parser.parse_args()

    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    ids = json.loads(args.ids.read_text(encoding="utf-8"))
    if len(ids) != 500 or len(ids) != len(set(ids)):
        raise SystemExit("Gold ID 必须是 500 条且唯一")
    query = """
        SELECT raw.artifact_id, raw.raw_data
        FROM unnest(%s::bigint[]) WITH ORDINALITY AS requested(artifact_id, ordinal)
        JOIN public.artifact_raw AS raw ON raw.artifact_id = requested.artifact_id
        ORDER BY requested.ordinal
    """
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query, (ids,)).fetchall()
    if [row[0] for row in rows] != ids:
        raise SystemExit("数据库记录缺失或返回顺序不匹配")

    payload = "".join(
        json.dumps({"artifact_id": aid, "raw_data": raw},
                   ensure_ascii=False, separators=(",", ":")) + "\n"
        for aid, raw in rows
    ).encode("utf-8")
    if args.output.exists():
        raise SystemExit(f"拒绝覆盖已有快照: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(payload)
    manifest = {
        "purpose": "prompt-omit 500条正式实验冻结输入",
        "ids_file": str(args.ids),
        "ids_sha256": sha256_bytes(json.dumps(ids, separators=(",", ":")).encode()),
        "record_count": len(rows),
        "snapshot_sha256": sha256_bytes(payload),
        "artifact_ids": ids,
    }
    args.manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
                             encoding="utf-8")
    print(json.dumps({"records": len(rows), "snapshot": str(args.output),
                      "snapshot_sha256": manifest["snapshot_sha256"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
