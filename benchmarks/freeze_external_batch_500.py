#!/usr/bin/env python3
"""冻结外部 artifact 批次的分层 500 条字段选择实验输入。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import psycopg


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from llm_runner import load_dotenv


THRESHOLD = 400000001
COHORT_DIR = ROOT / "data/external_batch_400000001"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def load_ids(path: Path) -> set[int]:
    if path.suffix == ".jsonl":
        return {
            int(json.loads(line)["record_id"])
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
    return {int(value) for value in json.loads(path.read_text(encoding="utf-8"))}


def main() -> None:
    parser = argparse.ArgumentParser(description="冻结外部批次 500 条分层输入")
    parser.add_argument("--output-dir", type=Path, default=COHORT_DIR)
    parser.add_argument("--seed", type=int, default=20260916)
    parser.add_argument("--size", type=int, default=500)
    parser.add_argument("--strata", type=int, default=5)
    args = parser.parse_args()
    if args.size <= 0 or args.strata <= 0 or args.size % args.strata:
        raise SystemExit("size 必须为正数，且能被 strata 整除")

    output_dir = args.output_dir if args.output_dir.is_absolute() else ROOT / args.output_dir
    snapshot = output_dir / "snapshot.jsonl"
    ids_output = output_dir / "ids.json"
    manifest_output = output_dir / "manifest.json"
    for path in (snapshot, ids_output, manifest_output):
        if path.exists():
            raise SystemExit(f"拒绝覆盖已有冻结产物: {path}")

    old_sets = {
        "training_dataset": load_ids(ROOT / "training_dataset.jsonl"),
        "gold_main": load_ids(ROOT / "data/gold/gold_main_ids.json"),
        "performance_2000": load_ids(ROOT / "data/performance/performance_ids_2000.json"),
    }
    old_ids = set().union(*old_sets.values())

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    per_stratum = args.size // args.strata
    query = """
        WITH candidates AS (
            SELECT artifact_id, raw_data, octet_length(raw_data::text) AS raw_bytes
            FROM public.artifact_raw
            WHERE artifact_id > %(threshold)s
        ), ranked AS (
            SELECT artifact_id, raw_data, raw_bytes,
                   ntile(%(strata)s) OVER (ORDER BY raw_bytes, artifact_id) AS stratum
            FROM candidates
        ), sampled AS (
            SELECT artifact_id, raw_data, raw_bytes, stratum,
                   row_number() OVER (
                       PARTITION BY stratum
                       ORDER BY md5(%(seed)s || artifact_id::text)
                   ) AS sample_rank
            FROM ranked
        )
        SELECT artifact_id, raw_data, raw_bytes, stratum
        FROM sampled
        WHERE sample_rank <= %(per_stratum)s
        ORDER BY stratum, sample_rank
    """
    count_query = """
        WITH ranked AS (
            SELECT ntile(%(strata)s) OVER (
                       ORDER BY octet_length(raw_data::text), artifact_id
                   ) AS stratum
            FROM public.artifact_raw
            WHERE artifact_id > %(threshold)s
        )
        SELECT stratum, count(*) FROM ranked GROUP BY stratum ORDER BY stratum
    """
    params = {
        "threshold": THRESHOLD,
        "strata": args.strata,
        "seed": str(args.seed),
        "per_stratum": per_stratum,
    }
    with psycopg.connect(database_url) as connection:
        candidates = connection.execute(
            "SELECT count(*) FROM public.artifact_raw WHERE artifact_id > %s",
            (THRESHOLD,),
        ).fetchone()[0]
        stratum_counts = connection.execute(count_query, params).fetchall()
        rows = connection.execute(query, params).fetchall()

    artifact_ids = [int(row[0]) for row in rows]
    if len(rows) != args.size or len(artifact_ids) != len(set(artifact_ids)):
        raise SystemExit(f"分层抽样记录数异常: {len(rows)}")
    overlap = sorted(old_ids & set(artifact_ids))
    if overlap:
        raise SystemExit(f"新 cohort 与既有实验重叠: {overlap[:5]}")
    selected_by_stratum = {
        stratum: sum(row[3] == stratum for row in rows)
        for stratum in range(1, args.strata + 1)
    }
    if set(selected_by_stratum.values()) != {per_stratum}:
        raise SystemExit(f"各层样本数异常: {selected_by_stratum}")

    output_dir.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(
            {"artifact_id": row[0], "raw_data": row[1]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
        + "\n"
        for row in rows
    ).encode("utf-8")
    snapshot.write_bytes(payload)
    ids_output.write_text(json.dumps(artifact_ids, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    manifest: dict[str, Any] = {
        "purpose": "external batch n=100 field selection generalization",
        "artifact_id_threshold_exclusive": THRESHOLD,
        "candidate_records": candidates,
        "sampling": {
            "method": "raw JSON byte-length NTILE strata, deterministic MD5 selection",
            "seed": args.seed,
            "size": args.size,
            "strata": args.strata,
            "selected_per_stratum": selected_by_stratum,
            "candidate_per_stratum": {stratum: count for stratum, count in stratum_counts},
        },
        "artifact_ids_sha256": sha256_bytes(
            json.dumps(artifact_ids, separators=(",", ":")).encode("utf-8")
        ),
        "snapshot_sha256": sha256_bytes(payload),
        "record_count": len(rows),
        "raw_bytes": {
            "min": min(int(row[2]) for row in rows),
            "max": max(int(row[2]) for row in rows),
        },
        "overlap_with_existing": {name: len(set(artifact_ids) & ids) for name, ids in old_sets.items()},
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "records": len(rows),
        "snapshot": str(snapshot),
        "snapshot_sha256": manifest["snapshot_sha256"],
        "selected_per_stratum": selected_by_stratum,
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
