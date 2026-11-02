#!/usr/bin/env python3
"""使用已训练的字段选择模型预测一条原始 JSON 记录。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np


def serialize_raw_data(raw_data: Any) -> str:
    return json.dumps(raw_data, ensure_ascii=False, separators=(",", ":"), default=str)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="预测原始记录的字段需求")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--input-json", type=Path)
    group.add_argument("--json")
    parser.add_argument("--model-dir", type=Path, default=Path("models/field_selector"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    raw_data = json.loads(args.input_json.read_text(encoding="utf-8")) if args.input_json else json.loads(args.json)
    record_text = serialize_raw_data(raw_data)
    vectorizer = joblib.load(args.model_dir / "vectorizer.joblib")
    classifiers = joblib.load(args.model_dir / "classifiers.joblib")
    length_regressors = joblib.load(args.model_dir / "length_regressors.joblib")
    constants = json.loads((args.model_dir / "constants.json").read_text(encoding="utf-8"))
    length_metadata = json.loads(
        (args.model_dir / "length_metadata.json").read_text(encoding="utf-8")
    )
    thresholds = json.loads((args.model_dir / "thresholds.json").read_text(encoding="utf-8"))
    metadata = json.loads((args.model_dir / "metadata.json").read_text(encoding="utf-8"))

    matrix = vectorizer.transform([record_text])
    scores = {}
    predicted_output_tokens = {}
    selected = []
    for field in metadata["fields"]:
        if field in constants:
            score = constants[field]
        else:
            score = float(classifiers[field].predict_proba(matrix)[0, 1])
        scores[field] = round(score, 6)
        length_info = length_metadata[field]
        if length_info["type"] == "median_constant":
            predicted_length = length_info["value"]
        else:
            predicted_length = length_regressors[field].predict(matrix)[0]
            predicted_length = max(0.0, float(np.expm1(predicted_length)))
        predicted_output_tokens[field] = max(0, round(predicted_length))
        if score >= thresholds[field]:
            selected.append(field)
    print(json.dumps({
        "field_scores": scores,
        "predicted_output_tokens": predicted_output_tokens,
        "selected_by_threshold": selected,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
