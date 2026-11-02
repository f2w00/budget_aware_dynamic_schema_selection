#!/usr/bin/env python3
"""从湖南博物馆 full silver 输出冻结 held-out 质量评估参考。"""

from __future__ import annotations

import json
import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEST_IDS = ROOT / "data/hunan_museum_300000001_300000500/test_ids.json"
SILVER = ROOT / "experiments/hunan_museum_300000001_300000500_full_silver/formal/full.jsonl"
OUTPUT = ROOT / "data/hunan_museum_300000001_300000500/heldout_full_silver_reference.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-dir", type=Path, default=ROOT / "data/hunan_museum_300000001_300000500")
    parser.add_argument("--silver", type=Path, default=SILVER)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    cohort_dir = args.cohort_dir if args.cohort_dir.is_absolute() else ROOT / args.cohort_dir
    silver_path = args.silver if args.silver.is_absolute() else ROOT / args.silver
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise SystemExit(f"拒绝覆盖已有参考: {output}")
    expected_ids = [
        int(value)
        for value in json.loads((cohort_dir / "test_ids.json").read_text(encoding="utf-8"))
    ]
    silver_rows = {
        int(row["artifact_id"]): row
        for line in silver_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for row in [json.loads(line)]
    }
    if any(row.get("status") != "success" or row.get("method") != "full"
           for row in silver_rows.values()):
        raise SystemExit("full silver 存在失败或非 full 记录")
    if set(expected_ids) - set(silver_rows):
        raise SystemExit("full silver 缺少 held-out ID")

    references = [
        {
            "artifact_id": artifact_id,
            "schema_version": "simple2-v1",
            "reference_result": {
                field: silver_rows[artifact_id]["result"].get(field)
                for field in silver_rows[artifact_id]["selected_fields"]
            },
        }
        for artifact_id in expected_ids
    ]
    if len(references) != 350 or len({row["artifact_id"] for row in references}) != 350:
        raise SystemExit("held-out reference 不是 350 条唯一记录")
    output.write_text(
        "".join(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                for row in references),
        encoding="utf-8",
    )
    print(json.dumps({"records": len(references), "output": str(output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
