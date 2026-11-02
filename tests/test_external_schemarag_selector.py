from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from external_schemarag_selector import select_fields


class FakeEmbedder:
    model = "fake-embedding"

    def __init__(self, vectors):
        self.vectors = vectors
        self.calls = []

    def embed(self, texts):
        self.calls.append(texts)
        return [self.vectors[text] for text in texts]


class SchemaRAGSelectorTest(unittest.TestCase):
    def test_selects_top_k_with_schema_order_tiebreak(self):
        schema = {
            "first": "first definition",
            "second": "second definition",
            "third": "third definition",
        }
        vectors = {
            "字段名：first": [1, 0],
            "字段名：second": [0, 1],
            "字段名：third": [1, 0],
            "字段定义：first definition": [1, 0],
            "字段定义：second definition": [0, 1],
            "字段定义：third definition": [1, 0],
            "给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。\n"
            "原始记录：{}": [1, 0],
        }
        costs = {
            "fields": {
                field: {"field_name_tokens": 1, "schema_value_tokens": 1}
                for field in schema
            }
        }
        prediction = {"predicted_output_tokens": {field: 2 for field in schema}}

        result = select_fields(
            {}, schema, costs, prediction, FakeEmbedder(vectors), k=2
        )

        self.assertEqual(result["selected_fields"], ["first", "third"])
        self.assertEqual(result["estimated_cost"], 8)
        self.assertEqual(result["method_metadata"]["k"], 2)

    def test_row_cache_is_reused(self):
        schema = {"field": "definition"}
        vectors = {
            "字段名：field": [1, 0],
            "字段定义：definition": [1, 0],
            "给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。\n"
            "原始记录：{}": [1, 0],
        }
        embedder = FakeEmbedder(vectors)
        costs = {
            "fields": {
                "field": {"field_name_tokens": 1, "schema_value_tokens": 1}
            }
        }
        prediction = {"predicted_output_tokens": {"field": 1}}
        with TemporaryDirectory() as directory:
            cache = Path(directory) / "rows.json"
            select_fields({}, schema, costs, prediction, embedder, cache, k=1)
            first_call_count = len(embedder.calls)
            select_fields({}, schema, costs, prediction, embedder, cache, k=1)

        self.assertEqual(len(embedder.calls), first_call_count + 1)

    def test_selects_fifteen_fields_from_eighteen_field_schema(self):
        schema = {f"field_{index:02d}": "definition" for index in range(18)}
        vectors = {
            f"字段名：field_{index:02d}": [1, 0]
            for index in range(18)
        }
        vectors.update({
            "字段定义：definition": [1, 0],
            "给定一条文物原始记录，检索该记录中可能存在可靠信息的目标字段。\n"
            "原始记录：{}": [1, 0],
        })
        costs = {
            "fields": {
                field: {"field_name_tokens": 1, "schema_value_tokens": 1}
                for field in schema
            }
        }
        prediction = {
            "predicted_output_tokens": {field: 1 for field in schema}
        }

        result = select_fields(
            {}, schema, costs, prediction, FakeEmbedder(vectors), k=15
        )

        self.assertEqual(len(result["selected_fields"]), 15)
        self.assertEqual(result["method_metadata"]["k"], 15)
        self.assertEqual(set(result["selected_fields"]), set(list(schema)[:15]))
