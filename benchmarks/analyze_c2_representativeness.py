#!/usr/bin/env python3
"""C2 数据代表性与分层分析（纯离线，不触发 LLM）。

回应审稿问题 20/21：对 2000 条在线性能负载做代表性分层，并与源数据库
（training_dataset.jsonl，6763 条）对比，确认样本构成与结论适用范围。

分层维度（raw_data 可得）：
- 记录类型（类型）
- 描述长度（详细描述，字符数与 UTF-8 字节数）
- 年代
- 字段非空/缺失模式（raw_data 共 5 字段，评估是否存在缺失）

同时对比源数据库的 18 字段 label 稀疏度（来自 split_stats.json / 源 labels），
说明性能负载与开发分布的关系。
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

PERF_RAW_DIR = Path("benchmarks/runs/main-comparison-pro/raw")
SOURCE = Path("training_dataset.jsonl")
SPLIT_STATS = Path("data/splits/split_stats.json")
OUT_DIR = Path("benchmarks/analysis/c2_representativeness")

RAW_FIELDS = ("artifact_id", "年代", "文物名称", "类型", "详细描述")


def load_perf_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(PERF_RAW_DIR.glob("*.jsonl")):
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    # 文本字段在 raw_data 子对象内，摊平到顶层便于分层统计
                    rec.update(rec.get("raw_data", {}) or {})
                    rows.append(rec)
    # 去重：同一 artifact_id 在多 run 重复出现，仅取唯一记录
    seen: set[str] = set()
    uniq: list[dict[str, Any]] = []
    for r in rows:
        aid = r.get("artifact_id")
        if aid is not None and aid not in seen:
            seen.add(aid)
            uniq.append(r)
    return uniq


def load_source_records() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with SOURCE.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rec = json.loads(line)
                text = rec.get("record_text", "")
                try:
                    raw = json.loads(text)
                except Exception:
                    raw = {}
                rows.append({"raw": raw, "labels": rec.get("labels", {})})
    return rows


def desc_metrics(record: dict[str, Any]) -> tuple[int, int]:
    d = str(record.get("详细描述", ""))
    return len(d), len(d.encode("utf-8"))


def type_dist(rows: list[dict[str, Any]]) -> Counter:
    return Counter(str(r.get("类型", "") or "").strip() or "(空)" for r in rows)


def dynasty_dist(rows: list[dict[str, Any]]) -> Counter:
    return Counter(str(r.get("年代", "") or "").strip() or "(空)" for r in rows)


def summarize_strata(
    rows: list[dict[str, Any]], label: str
) -> dict[str, Any]:
    types = type_dist(rows)
    dynasties = dynasty_dist(rows)
    char_lens = [desc_metrics(r)[0] for r in rows]
    byte_lens = [desc_metrics(r)[1] for r in rows]
    char_lens.sort()
    byte_lens.sort()

    # 字段缺失模式
    field_empty: dict[str, int] = {}
    for f in RAW_FIELDS:
        field_empty[f] = sum(1 for r in rows if str(r.get(f, "")).strip() in ("", "None", "null"))

    def pct(seq: list[int], p: float) -> int:
        if not seq:
            return 0
        idx = int((len(seq) - 1) * p / 100.0)
        return seq[idx]

    return {
        "label": label,
        "n": len(rows),
        "type_dist": dict(types.most_common()),
        "dynasty_dist": dict(dynasties.most_common(20)),
        "desc_char_len": {
            "min": char_lens[0] if char_lens else 0,
            "p50": pct(char_lens, 50),
            "p90": pct(char_lens, 90),
            "p99": pct(char_lens, 99),
            "max": char_lens[-1] if char_lens else 0,
        },
        "desc_byte_len": {
            "min": byte_lens[0] if byte_lens else 0,
            "p50": pct(byte_lens, 50),
            "p90": pct(byte_lens, 90),
            "p99": pct(byte_lens, 99),
            "max": byte_lens[-1] if byte_lens else 0,
        },
        "field_empty": field_empty,
    }


def source_label_sparsity(source: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(source)
    counters: dict[str, int] = {}
    for r in source:
        for k, v in (r.get("labels") or {}).items():
            counters[k] = counters.get(k, 0) + (1 if v else 0)
    return {
        "n": n,
        "field_positive_count": {k: v for k, v in sorted(counters.items())},
        "field_positive_rate": {
            k: round(v / n, 4) for k, v in sorted(counters.items())
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="C2 数据代表性与分层分析")
    parser.add_argument("--output-dir", type=Path, default=OUT_DIR)
    args = parser.parse_args()

    perf = load_perf_records()
    source = load_source_records()

    perf_strata = summarize_strata(perf, "performance_2000")
    source_strata = summarize_strata(
        [{"类型": r["raw"].get("类型", ""), "年代": r["raw"].get("年代", ""),
          "详细描述": r["raw"].get("详细描述", ""), "文物名称": r["raw"].get("文物名称", ""),
          "artifact_id": r["raw"].get("artifact_id", "")} for r in source],
        "source_full",
    )

    label_sparsity = source_label_sparsity(source)

    # 类型构成对比（性能负载 vs 源）
    perf_types = perf_strata["type_dist"]
    src_types = source_strata["type_dist"]
    type_compare = {
        k: {
            "perf_count": perf_types.get(k, 0),
            "perf_pct": round(perf_types.get(k, 0) / len(perf) * 100, 2),
            "source_count": src_types.get(k, 0),
            "source_pct": round(src_types.get(k, 0) / len(source) * 100, 2),
        }
        for k in sorted(set(perf_types) | set(src_types), key=lambda x: -perf_types.get(x, 0))
    }

    payload = {
        "analysis": {
            "description": "C2 数据代表性与分层：性能负载 2000 条 vs 源数据库；纯离线统计。",
            "perf_source": "benchmarks/runs/main-comparison-pro/raw",
            "source": str(SOURCE),
            "note": (
                "性能负载按字节长度 5 层分层（见 workload manifest）；raw_data 仅 5 字段且全存在，"
                "无字段缺失。源 18 字段稀疏度来自训练 label。"
            ),
        },
        "performance_load": perf_strata,
        "source_full": source_strata,
        "source_18field_sparsity": label_sparsity,
        "type_comparison": type_compare,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "c2_representativeness.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output": str(args.output_dir / "c2_representativeness.json"),
                      "perf_n": len(perf), "source_n": len(source)}, ensure_ascii=False))


if __name__ == "__main__":
    main()