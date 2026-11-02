#!/usr/bin/env python3
"""审计 model-assisted reference 的空值语义和高风险字段原文支持。"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from llm_runner import load_dotenv, load_records


HIGH_RISK_FIELDS = {
    "condition",
    "acq_type",
    "region",
    "site",
    "culture",
    "craft",
    "craft_school",
    "related_event",
    "related_person",
}
NULL_STRINGS = {"null", "none", "n/a", "未知", "未详", "无"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="审计 Gold reference 的原文支持")
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--schema", type=Path, default=Path("simple2.json"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/gold/reviews/gold_reference_audit.jsonl"),
    )
    parser.add_argument(
        "--stats",
        type=Path,
        default=Path("reports/gold_reference_audit.json"),
    )
    return parser.parse_args()


def is_null_string(value: Any) -> bool:
    return isinstance(value, str) and value.strip().lower() in NULL_STRINGS


def values(value: Any) -> list[Any]:
    return value if isinstance(value, list) else [value]


def main() -> None:
    args = parse_args()
    load_dotenv(Path(".env"))
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise SystemExit("缺少 DATABASE_URL")

    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    references = [
        json.loads(line)
        for line in args.reference.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [int(row["artifact_id"]) for row in references]
    raw_by_id = dict(load_records(database_url, ids, len(ids)))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.stats.parent.mkdir(parents=True, exist_ok=True)
    reviewed = 0
    candidate_count = 0
    null_string_count = 0
    direct_match_count = 0
    unsupported_count = 0
    with args.output.open("w", encoding="utf-8") as output:
        for reference in references:
            artifact_id = int(reference["artifact_id"])
            raw_data = raw_by_id.get(artifact_id)
            if raw_data is None:
                raise SystemExit(f"artifact_raw 缺少 artifact_id={artifact_id}")
            raw_text = json.dumps(raw_data, ensure_ascii=False, sort_keys=True)
            findings = []
            for field in schema:
                value = reference["reference_result"].get(field)
                if is_null_string(value):
                    null_string_count += 1
                    findings.append({
                        "field": field,
                        "value": value,
                        "reason": "null_string",
                        "direct_raw_match": False,
                    })
                if field not in HIGH_RISK_FIELDS or value is None or is_null_string(value):
                    continue
                matches = []
                for item in values(value):
                    if not isinstance(item, (str, int, float, bool)):
                        continue
                    matches.append(str(item) in raw_text)
                direct_match = bool(matches) and all(matches)
                if direct_match:
                    direct_match_count += 1
                else:
                    unsupported_count += 1
                findings.append({
                    "field": field,
                    "value": value,
                    "reason": "high_risk_non_null",
                    "direct_raw_match": direct_match,
                })
            if findings:
                candidate_count += 1
                output.write(json.dumps({
                    "artifact_id": artifact_id,
                    "reference_model": reference.get("reference_model"),
                    "findings": findings,
                    "review_action": "check artifact_raw before accepting reference",
                }, ensure_ascii=False, separators=(",", ":")) + "\n")
            reviewed += 1

    stats = {
        "reference": str(args.reference),
        "reviewed_records": reviewed,
        "records_with_findings": candidate_count,
        "null_string_findings": null_string_count,
        "high_risk_direct_raw_matches": direct_match_count,
        "high_risk_without_direct_raw_match": unsupported_count,
        "high_risk_fields": sorted(HIGH_RISK_FIELDS),
    }
    args.stats.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()
