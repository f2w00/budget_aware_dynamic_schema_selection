#!/usr/bin/env python3
"""为字段选择模型生成固定的多标签训练、验证和测试 ID 划分。"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from iterstrat.ml_stratifiers import MultilabelStratifiedShuffleSplit


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="划分字段选择训练数据集")
    parser.add_argument("--input", type=Path, default=Path("training_dataset.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_rows(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError as error:
            raise SystemExit(f"第 {line_number} 行 JSON 无效: {error}") from error
    if not rows:
        raise SystemExit("训练数据集为空")
    return rows


def split_indices(labels: np.ndarray, seed: int) -> tuple[np.ndarray, ...]:
    first = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.2,
        random_state=seed,
    )
    train_idx, remaining_idx = next(first.split(np.zeros(len(labels)), labels))
    remaining_labels = labels[remaining_idx]
    second = MultilabelStratifiedShuffleSplit(
        n_splits=1,
        test_size=0.5,
        random_state=seed,
    )
    validation_relative, test_relative = next(
        second.split(np.zeros(len(remaining_labels)), remaining_labels)
    )
    return train_idx, remaining_idx[validation_relative], remaining_idx[test_relative]


def field_stats(rows: list[dict[str, Any]], indices: np.ndarray, fields: list[str]) -> dict[str, Any]:
    counts = Counter()
    for index in indices:
        for field in fields:
            counts[field] += int(rows[index]["labels"][f"{field}_present"])
    size = len(indices)
    return {
        field: {
            "positive": counts[field],
            "negative": size - counts[field],
            "positive_rate": round(counts[field] / size, 6) if size else 0.0,
        }
        for field in fields
    }


def main() -> None:
    args = parse_args()
    rows = load_rows(args.input)
    ids = [row["record_id"] for row in rows]
    if len(set(ids)) != len(ids):
        raise SystemExit("record_id 存在重复，无法安全划分")

    label_keys = list(rows[0]["labels"])
    fields = [key.removesuffix("_present") for key in label_keys]
    labels = np.array(
        [[int(row["labels"][key]) for key in label_keys] for row in rows],
        dtype=np.int8,
    )
    train_idx, validation_idx, test_idx = split_indices(labels, args.seed)
    splits = {
        "train": train_idx,
        "validation": validation_idx,
        "test": test_idx,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, indices in splits.items():
        values = [ids[index] for index in indices]
        (args.output_dir / f"{name}_ids.json").write_text(
            json.dumps(values, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    stats = {
        "source": str(args.input),
        "seed": args.seed,
        "total": len(rows),
        "sizes": {name: len(indices) for name, indices in splits.items()},
        "fields": fields,
        "field_stats": {
            name: field_stats(rows, indices, fields)
            for name, indices in splits.items()
        },
    }
    (args.output_dir / "split_stats.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(stats["sizes"], ensure_ascii=False))


if __name__ == "__main__":
    main()
