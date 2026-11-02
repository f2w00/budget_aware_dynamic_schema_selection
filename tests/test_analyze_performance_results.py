import unittest

from benchmarks.analyze_performance_results import compare_to_full, confidence_interval, group_rows


class PerformanceAnalysisTest(unittest.TestCase):
    def test_confidence_interval_requires_supported_repeat_count(self):
        self.assertEqual(confidence_interval([1.0]), (None, None))
        lower, upper = confidence_interval([1.0, 3.0])
        self.assertLess(lower, 1.0)
        self.assertGreater(upper, 3.0)

    def test_group_and_relative_comparison(self):
        rows = []
        for method, pipeline, throughput, tokens in (
            ("full", 100.0, 10.0, 1000.0),
            ("budget-200", 50.0, 20.0, 600.0),
        ):
            for repeat in (1, 2):
                rows.append(
                    {
                        "method": method,
                        "concurrency": 16,
                        "queue_size": 16,
                        "records": 100,
                        "success_records": 100,
                        "error_records": 0,
                        "retry_count_total": 0,
                        "pipeline_ms": pipeline,
                        "effective_records_per_second": throughput,
                        "total_tokens_total": tokens,
                        "latency_ms_p95": 20.0,
                        "latency_ms_p99": 30.0,
                        "rss_peak_bytes": 1000.0,
                        "queue_average": 10.0,
                        "producer_blocked_ms": 40.0,
                    }
                )

        groups = group_rows(rows)
        comparison = compare_to_full(groups)

        self.assertEqual(len(groups), 2)
        self.assertEqual(groups[0]["repeats"], 2)
        self.assertEqual(comparison[0]["pipeline_time_change_pct"], -50.0)
        self.assertEqual(comparison[0]["throughput_change_pct"], 100.0)
        self.assertEqual(comparison[0]["token_change_pct"], -40.0)


if __name__ == "__main__":
    unittest.main()
