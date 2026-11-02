#!/usr/bin/env python3
"""基于论文描述重实现的 SchemaRAG 字段检索器。"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Protocol

import numpy as np
from openai import OpenAI


K = 10
NAME_WEIGHT = 0.5
DEFINITION_WEIGHT = 0.5
QUERY_INSTRUCTION = "给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。"


class Embedder(Protocol):
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...


class OpenAIEmbedder:
    """调用项目配置的 OpenAI 兼容 embedding 服务。"""

    def __init__(self) -> None:
        base_url = os.environ.get("EMBEDDING_BASE_URL")
        api_key = os.environ.get("EMBEDDING_API_KEY")
        model = os.environ.get("EMBEDDING_MODEL")
        if not base_url or not api_key or not model:
            raise RuntimeError(
                "缺少 EMBEDDING_BASE_URL、EMBEDDING_API_KEY 或 EMBEDDING_MODEL"
            )
        timeout = float(os.environ.get("EMBEDDING_TIMEOUT_SECONDS", "60"))
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=timeout)
        self.model = model
        self.batch_size = max(1, int(os.environ.get("EMBEDDING_BATCH_SIZE", "32")))

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors: list[list[float]] = []
        for start in range(0, len(texts), self.batch_size):
            response = self.client.embeddings.create(
                model=self.model,
                input=texts[start : start + self.batch_size],
            )
            data = sorted(response.data, key=lambda item: item.index)
            vectors.extend([list(item.embedding) for item in data])
        if len(vectors) != len(texts):
            raise RuntimeError("embedding 服务返回数量与输入数量不一致")
        return vectors


class OfflineStubEmbedder:
    """仅用于 dry-run 和本地构造检查，禁止代替正式 embedding 结果。"""

    model = "offline-stub"

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = []
        for text in texts:
            digest = hashlib.sha256(text.encode("utf-8")).digest()
            vector = [float(digest[index % len(digest)] + 1) for index in range(32)]
            vectors.append(vector)
        return vectors


def query_text(raw_data: Any) -> str:
    record = json.dumps(raw_data, ensure_ascii=False, separators=(",", ":"), default=str)
    return f"{QUERY_INSTRUCTION}\n原始记录：{record}"


def normalize(vector: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vector)
    if norm == 0:
        raise ValueError("embedding 向量范数为 0")
    return vector / norm


def config_hash(schema: dict[str, str], model: str) -> str:
    payload = json.dumps(
        {
            "schema": schema,
            "model": model,
            "name_weight": NAME_WEIGHT,
            "definition_weight": DEFINITION_WEIGHT,
            "query_instruction": QUERY_INSTRUCTION,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def selector_config_hash(schema: dict[str, str], model: str) -> str:
    """暴露配置哈希，供 Runner 的断点续跑判断使用。"""
    return config_hash(schema, model)


def load_or_create_row_embeddings(
    schema: dict[str, str],
    embedder: Embedder,
    cache_path: Path | None = None,
) -> dict[str, list[float]]:
    """缓存只绑定 Schema、模型和组合权重，避免复用陈旧 row 向量。"""
    model = getattr(embedder, "model", "test-embedder")
    cache_key = config_hash(schema, model)
    if cache_path and cache_path.exists():
        cached = json.loads(cache_path.read_text(encoding="utf-8"))
        if cached.get("config_hash") == cache_key:
            return cached["embeddings"]

    name_vectors = embedder.embed([f"字段名：{field}" for field in schema])
    definition_vectors = embedder.embed(
        [f"字段定义：{definition}" for definition in schema.values()]
    )
    embeddings = {}
    for field, name_vector, definition_vector in zip(
        schema, name_vectors, definition_vectors, strict=True
    ):
        combined = (
            NAME_WEIGHT * np.asarray(name_vector, dtype=float)
            + DEFINITION_WEIGHT * np.asarray(definition_vector, dtype=float)
        )
        embeddings[field] = normalize(combined).tolist()
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(
            json.dumps(
                {"config_hash": cache_key, "model": model, "embeddings": embeddings},
                ensure_ascii=False,
            )
            + "\n",
            encoding="utf-8",
        )
    return embeddings


def select_fields(
    raw_data: Any,
    schema: dict[str, str],
    costs: dict[str, Any],
    prediction: dict[str, Any],
    embedder: Embedder | None = None,
    cache_path: Path | None = None,
    k: int = K,
) -> dict[str, Any]:
    if not 1 <= k <= len(schema):
        raise ValueError(f"k 必须在 1 到 {len(schema)} 之间")
    if embedder is None:
        embedder = (
            OfflineStubEmbedder()
            if os.environ.get("EMBEDDING_OFFLINE_STUB") == "1"
            else OpenAIEmbedder()
        )
    schema_embedding_started = time.perf_counter()
    row_embeddings = load_or_create_row_embeddings(schema, embedder, cache_path)
    schema_embedding_ms = (time.perf_counter() - schema_embedding_started) * 1000
    # 字段向量可缓存；每条原始记录的查询向量必须计入在线检索开销。
    query_embedding_started = time.perf_counter()
    record_embedding = normalize(np.asarray(embedder.embed([query_text(raw_data)])[0]))
    query_embedding_ms = (time.perf_counter() - query_embedding_started) * 1000

    ranked = []
    schema_order = {field: index for index, field in enumerate(schema)}
    for field in schema:
        score = float(np.dot(record_embedding, row_embeddings[field]))
        ranked.append((score, schema_order[field], field))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected_fields = [field for _, _, field in ranked[:k]]
    selected_cost = sum(
        int(costs["fields"][field]["field_name_tokens"])
        + int(costs["fields"][field]["schema_value_tokens"])
        + int(prediction.get("predicted_output_tokens", {}).get(field, 0))
        for field in selected_fields
    )
    scores = {field: score for score, _, field in ranked}
    return {
        "selected_fields": selected_fields,
        "estimated_cost": selected_cost,
        "similarities": scores,
        "query_embedding_ms": round(query_embedding_ms, 2),
        "schema_embedding_ms": round(schema_embedding_ms, 2),
        "method_metadata": {
            "algorithm": "SchemaRAG re-implemented/adapted",
            "embedding_model": getattr(embedder, "model", "test-embedder"),
            "k": k,
            "name_weight": NAME_WEIGHT,
            "definition_weight": DEFINITION_WEIGHT,
            "query_instruction": QUERY_INSTRUCTION,
            "segmentation": False,
            "example_retrieval": False,
        },
    }
