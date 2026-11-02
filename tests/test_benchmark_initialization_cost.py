from __future__ import annotations

import unittest
from pathlib import Path

from benchmarks.benchmark_initialization_cost import ensure_isolated_output, percentiles


class InitializationBenchmarkTest(unittest.TestCase):
    def test_percentiles_for_single_value(self) -> None:
        self.assertEqual(percentiles([12.5]), {
            "mean_ms": 12.5,
            "std_ms": 0.0,
            "p50_ms": 12.5,
            "p95_ms": 12.5,
            "max_ms": 12.5,
        })

    def test_models_directory_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            ensure_isolated_output(Path("models/new-benchmark-model"))
