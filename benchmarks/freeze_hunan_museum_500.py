#!/usr/bin/env python3
"""冻结湖南省博物馆连续 ID 批次的前 500 条原始记录。"""

from __future__ import annotations

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


FIRST_ARTIFACT_ID = 300_000_001
SIZE = 500
COHORT_DIR = ROOT / "data/hunan_museum_300000001_300000500"


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
    snapshot = COHORT_DIR / "snapshot.jsonl"
    ids_output = COHORT_DIR / "ids.json"
    manifest_output = COHORT_DIR / "manifest.json"
    for path in (snapshot, ids_output, manifest_output):
        if path.exists():
            raise SystemExit(f"拒绝覆盖已有冻结产物: {path}")

    existing_sets = {
        "training_dataset": load_ids(ROOT / "training_dataset.jsonl"),
        "gold_main": load_ids(ROOT / "data/gold/gold_main_ids.json"),
        "performance_2000": load_ids(ROOT / "data/performance/performance_ids_2000.json"),
        "external_batch_400000001": load_ids(
            ROOT / "data/external_batch_400000001/ids.json"
        ),
    }
    existing_ids = set().union(*existing_sets.values())

    load_dotenv(ROOT / ".env")
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")

    last_artifact_id = FIRST_ARTIFACT_ID + SIZE - 1
    query = """
        SELECT artifact_id, raw_data, octet_length(raw_data::text) AS raw_bytes
        FROM public.artifact_raw
        WHERE artifact_id BETWEEN %s AND %s
        ORDER BY artifact_id ASC
    """
    with psycopg.connect(database_url) as connection:
        rows = connection.execute(query, (FIRST_ARTIFACT_ID, last_artifact_id)).fetchall()

    artifact_ids = [int(row[0]) for row in rows]
    expected_ids = list(range(FIRST_ARTIFACT_ID, last_artifact_id + 1))
    if artifact_ids != expected_ids:
        raise SystemExit(
            "数据库缺少或错序湖南博物馆连续 ID："
            f"期望 {FIRST_ARTIFACT_ID}..{last_artifact_id}，实际 {len(artifact_ids)} 条"
        )
    overlap = sorted(existing_ids & set(artifact_ids))
    if overlap:
        raise SystemExit(f"连续前500条与既有实验重叠: {overlap[:5]}")

    COHORT_DIR.mkdir(parents=True, exist_ok=True)
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
    ids_output.write_text(
        json.dumps(artifact_ids, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    manifest: dict[str, Any] = {
        "purpose": "Hunan Museum continuous artifact_id 300000001 through 300000500",
        "first_artifact_id_requested": FIRST_ARTIFACT_ID,
        "last_artifact_id_requested": last_artifact_id,
        "selection": "fixed continuous artifact_id interval; no sampling",
        "record_count": len(rows),
        "first_artifact_id": artifact_ids[0],
        "last_artifact_id": artifact_ids[-1],
        "artifact_ids_sha256": sha256_bytes(
            json.dumps(artifact_ids, separators=(",", ":")).encode("utf-8")
        ),
        "snapshot_sha256": sha256_bytes(payload),
        "raw_bytes": {
            "min": min(int(row[2]) for row in rows),
            "max": max(int(row[2]) for row in rows),
        },
        "overlap_with_existing": {
            name: len(set(artifact_ids) & ids) for name, ids in existing_sets.items()
        },
    }
    manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
