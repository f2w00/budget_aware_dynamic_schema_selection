import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.benchmark_pipeline import ids_hash, load_workload_manifest, summarize
from benchmarks.create_performance_workload import stratified_sample


class BenchmarkPipelineTest(unittest.TestCase):
    def test_summarize_includes_stage_and_queue_metrics(self):
        rows = [
            {
                "status": "success",
                "latency_ms": 100.0,
                "queue_wait_ms": 20.0,
                "enqueue_blocked_ms": 5.0,
                "planning_ms": 10.0,
                "prediction_ms": 2.0,
                "selection_ms": 3.0,
                "embedding_ms": 2.0,
                "schema_embedding_ms": 0.5,
                "prompt_build_ms": 5.0,
                "llm_ms": 80.0,
                "parse_validate_ms": 1.0,
                "retry_backoff_ms": 0.0,
                "estimated_field_cost": 40,
                "total_tokens": 50,
                "retry_count": 0,
                "input_tokens": 30,
                "output_tokens": 20,
                "attempts": [],
            }
        ]
        summary = summarize(rows, {"pipeline_ms": 200.0, "dry_run": False})

        self.assertEqual(summary["planning_ms_total"], 10.0)
        self.assertEqual(summary["embedding_ms_total"], 2.0)
        self.assertEqual(summary["schema_embedding_ms_total"], 0.5)
        self.assertEqual(summary["llm_ms_total"], 80.0)
        self.assertEqual(summary["parse_validate_ms_total"], 1.0)
        self.assertEqual(summary["enqueue_blocked_ms_total"], 5.0)
        self.assertEqual(summary["queue_wait_ratio"], 0.2)

    def test_stratified_sample_is_reproducible_and_balanced(self):
        records = [(index, {"text": "x" * index}) for index in range(1, 21)]

        first_ids, first_summary = stratified_sample(records, size=10, strata=5, seed=42)
        second_ids, second_summary = stratified_sample(records, size=10, strata=5, seed=42)

        self.assertEqual(first_ids, second_ids)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(len(first_ids), 10)
        self.assertEqual(len(set(first_ids)), 10)
        self.assertEqual([row["selected_records"] for row in first_summary], [2, 2, 2, 2, 2])

    def test_workload_manifest_must_match_ids(self):
        ids = [5, 10, 15]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "workload.manifest.json"
            path.write_text(
                json.dumps(
                    {
                        "workload_type": "stratified_raw_length",
                        "selected_records": len(ids),
                        "ids_sha256": ids_hash(ids),
                    }
                ),
                encoding="utf-8",
            )

            result = load_workload_manifest(path, ids)

        self.assertEqual(result["workload_records"], 3)
        self.assertEqual(result["workload_ids_sha256"], ids_hash(ids))


if __name__ == "__main__":
    unittest.main()
