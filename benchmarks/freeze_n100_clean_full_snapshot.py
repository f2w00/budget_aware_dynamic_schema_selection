#!/usr/bin/env python3
"""冻结 n=100 clean-full 性能实验的 raw_data 快照与配对顺序。"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from llm_runner import iter_records, load_dotenv, load_ids, text_hash


PAIR_SPECS = (
    (42, 1, ("full", "budget-200")),
    (43, 1, ("budget-200", "full")),
    (44, 1, ("full", "budget-200")),
    (44, 2, ("budget-200", "full")),
    (43, 2, ("full", "budget-200")),
    (42, 2, ("budget-200", "full")),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="冻结 clean-full 2,000 条性能负载")
    parser.add_argument(
        "--ids-file",
        type=Path,
        default=Path("data/performance/performance_ids_2000.json"),
    )
    parser.add_argument(
        "--source-manifest",
        type=Path,
        default=Path("data/performance/performance_ids_2000.manifest.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/performance/n100_clean_full_2000_snapshot.jsonl"),
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/performance/n100_clean_full_2000_snapshot.manifest.json"),
    )
    parser.add_argument(
        "--order-dir",
        type=Path,
        default=Path("data/performance/n100_clean_full_orders"),
    )
    return parser.parse_args()


def canonical(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def main() -> None:
    args = parse_args()
    for path in (args.output, args.manifest):
        if path.exists():
            raise FileExistsError(f"拒绝覆盖既有冻结产物: {path}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.order_dir.mkdir(parents=True, exist_ok=True)

    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")
    ids = load_ids(args.ids_file, Path("data/splits"), "test")
    rows = [
        {"artifact_id": artifact_id, "raw_data": raw_data}
        for artifact_id, raw_data in iter_records(database_url, ids, 0)
    ]
    if len(rows) != len(ids):
        raise ValueError(f"数据库仅返回 {len(rows)}/{len(ids)} 条记录")

    snapshot_payload = b"".join(canonical(row) + b"\n" for row in rows)
    args.output.write_bytes(snapshot_payload)
    raw_hash = sha256_bytes(b"".join(canonical(row["raw_data"]) for row in rows))
    order_files = []
    for pair_index, (seed, repeat, sequence) in enumerate(PAIR_SPECS, 1):
        pair_ids = ids.copy()
        random.Random(20260911 + pair_index).shuffle(pair_ids)
        order_path = args.order_dir / f"pair_{pair_index}_s{seed}_r{repeat}.json"
        if order_path.exists():
            raise FileExistsError(f"拒绝覆盖既有顺序文件: {order_path}")
        order_path.write_text(
            json.dumps(pair_ids, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        order_files.append(
            {
                "pair_index": pair_index,
                "selector_seed": seed,
                "repeat": repeat,
                "sequence": list(sequence),
                "ids_file": str(order_path),
                "ids_sha256": sha256_bytes(canonical(pair_ids)),
            }
        )

    source_manifest = json.loads(args.source_manifest.read_text(encoding="utf-8"))
    schema = json.loads(Path("simple2.json").read_text(encoding="utf-8"))
    manifest = {
        "workload_type": "frozen_raw_data_snapshot",
        "selected_records": len(rows),
        "source_ids_file": str(args.ids_file),
        "source_ids_sha256": source_manifest.get("ids_sha256"),
        "snapshot_file": str(args.output),
        "snapshot_sha256": sha256_bytes(snapshot_payload),
        "raw_data_sha256": raw_hash,
        "schema_hash": text_hash(json.dumps(schema, ensure_ascii=False, sort_keys=True)),
        "prompt_hash": text_hash(Path("1.prompt").read_text(encoding="utf-8")),
        "source_manifest": str(args.source_manifest),
        "source_split_counts": {"train": 1760, "validation": 240, "test": 0},
        "pair_orders": order_files,
    }
    args.manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False))


if __name__ == "__main__":
    main()
