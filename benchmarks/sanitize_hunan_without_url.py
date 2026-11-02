#!/usr/bin/env python3
"""为湖南博物馆 cohort 生成不含来源 URL 的可审计输入副本。"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data/hunan_museum_300000001_300000500"
OUTPUT = ROOT / "data/hunan_museum_300000001_300000500_without_url"
REMOVED_FIELDS = ("来源URL", "source_url", "image_url")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    source_snapshot = SOURCE / "snapshot.jsonl"
    source_ids = SOURCE / "ids.json"
    output_snapshot = OUTPUT / "snapshot.jsonl"
    output_ids = OUTPUT / "ids.json"
    manifest = OUTPUT / "manifest.json"
    if not source_snapshot.exists() or not source_ids.exists():
        raise SystemExit("缺少原始湖南 cohort")
    for path in (output_snapshot, output_ids, manifest):
        if path.exists():
            raise SystemExit(f"拒绝覆盖已有净化产物: {path}")

    source_rows = [
        json.loads(line)
        for line in source_snapshot.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = json.loads(source_ids.read_text(encoding="utf-8"))
    if len(source_rows) != 500 or [row["artifact_id"] for row in source_rows] != ids:
        raise SystemExit("原始快照不是预期的 500 条连续 cohort")

    removed_counts = {field: 0 for field in REMOVED_FIELDS}
    sanitized_rows = []
    for row in source_rows:
        raw_data = dict(row["raw_data"])
        for field in REMOVED_FIELDS:
            if field in raw_data:
                del raw_data[field]
                removed_counts[field] += 1
        sanitized_rows.append({"artifact_id": row["artifact_id"], "raw_data": raw_data})
    if removed_counts["来源URL"] != 500:
        raise SystemExit(f"来源URL 删除计数异常: {removed_counts['来源URL']}")
    if any(any(field in row["raw_data"] for field in REMOVED_FIELDS) for row in sanitized_rows):
        raise SystemExit("净化后仍存在 URL 或 image URL 字段")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
        for row in sanitized_rows
    ).encode("utf-8")
    output_snapshot.write_bytes(payload)
    output_ids.write_text(json.dumps(ids, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest.write_text(
        json.dumps(
            {
                "purpose": "Hunan Museum without_url experimental input",
                "source_snapshot": str(source_snapshot.relative_to(ROOT)),
                "source_snapshot_sha256": sha256(source_snapshot),
                "source_ids_sha256": sha256(source_ids),
                "record_count": len(sanitized_rows),
                "removed_fields": removed_counts,
                "sanitized_snapshot_sha256": hashlib.sha256(payload).hexdigest(),
                "artifact_ids_sha256": sha256(output_ids),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"records": len(sanitized_rows), "removed": removed_counts}, ensure_ascii=False))


if __name__ == "__main__":
    main()
