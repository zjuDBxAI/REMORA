import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from summarize_dynamic_frontier import aggregate_repetitions  # noqa: E402
from audit_dynamic_inversions import (  # noqa: E402
    comparison_statistics,
    score_statistics,
)


def make_row(prefix: str, qps: float, latency: float, digest: str = "same"):
    return {
        "label": "ood-rpq",
        "prefix": prefix,
        "mode": "rpq",
        "beam_width": 100,
        "threads": 1,
        "warmup_queries": 100,
        "recall": 0.77,
        "qps": qps,
        "mean_latency_ms": latency,
        "p50_query_latency_ms": latency * 0.9,
        "p95_query_latency_ms": latency * 1.2,
        "p99_query_latency_ms": latency * 1.4,
        "mean_comparisons": 3500.0,
        "mean_expansions": 109.0,
        "mean_visited_nodes": 3501.0,
        "mean_final_candidate_pool": 100.0,
        "_id_sha256": digest,
    }


class DynamicFrontierAggregationTest(unittest.TestCase):
    def test_repetitions_use_median_and_report_variation(self):
        rows = aggregate_repetitions(
            [
                make_row("run-1", 700.0, 1.4),
                make_row("run-2", 720.0, 1.3),
                make_row("run-3", 710.0, 1.5),
            ]
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["timing_repetitions"], 3)
        self.assertEqual(rows[0]["qps"], 710.0)
        self.assertEqual(rows[0]["qps_min"], 700.0)
        self.assertEqual(rows[0]["qps_max"], 720.0)
        self.assertEqual(rows[0]["mean_latency_ms"], 1.4)
        self.assertGreater(rows[0]["qps_cv_percent"], 0.0)

    def test_repetitions_require_identical_result_ids(self):
        with self.assertRaisesRegex(ValueError, "non-deterministic result IDs"):
            aggregate_repetitions(
                [
                    make_row("run-1", 700.0, 1.4, "first"),
                    make_row("run-2", 720.0, 1.3, "second"),
                ]
            )


class DynamicInversionStatisticsTest(unittest.TestCase):
    def test_score_and_paired_recovery_statistics(self):
        exact = np.asarray([0.2, 0.1, 0.3, 0.4], dtype=np.float32)
        adc = np.asarray([-0.1, 0.0, 0.2, 0.3], dtype=np.float32)
        rpq = np.asarray([0.1, -0.05, 0.25, 0.2], dtype=np.float32)
        mask = np.ones(4, dtype=bool)
        adc_stats = score_statistics(exact, adc, mask, 1e-3, 0.0)
        self.assertEqual(adc_stats["nonpositive_inversions"], 2)
        self.assertAlmostEqual(adc_stats["nonpositive_inversion_rate"], 0.5)
        paired = comparison_statistics(exact, adc, rpq, mask, 0.0)
        self.assertEqual(paired["recovered_adc_inversions"], 1)
        self.assertEqual(paired["new_inversions_from_adc_correct"], 0)
        self.assertEqual(paired["persistent_inversions"], 1)
        self.assertEqual(paired["net_inversions_removed"], 1)


if __name__ == "__main__":
    unittest.main()
