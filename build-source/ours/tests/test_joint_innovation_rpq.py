#!/usr/bin/env python3

from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

from joint_innovation_rpq import (  # noqa: E402
    assemble_model,
    build_joint_transform,
    initialize_side_codes,
    optimal_bit_allocation,
    pack_codes,
    read_model,
    read_side_codes,
    residual_score_correction,
    unpack_group,
    write_model,
)


class JointInnovationTest(unittest.TestCase):
    def test_transform_realizes_metric(self) -> None:
        rng = np.random.default_rng(7)
        residual = rng.normal(size=(4000, 6)).astype(np.float32)
        residual[:, 1] += 0.7 * residual[:, 0]
        raw = rng.normal(size=(6, 6))
        metric = np.eye(6) + raw.T @ raw
        mean, forward, inverse, spectrum, diagnostics = build_joint_transform(
            residual, metric, 1e-7
        )
        self.assertTrue(np.all(spectrum > 0.0))
        self.assertLess(diagnostics["transform_inverse_max_error"], 1e-8)
        self.assertTrue(
            np.allclose(
                forward.astype(np.float64) @ forward.astype(np.float64).T,
                metric,
                rtol=2e-5,
                atol=2e-5,
            )
        )
        error = rng.normal(size=(32, 6))
        lhs = np.sum((error @ forward) ** 2, axis=1)
        rhs = np.einsum("bi,ij,bj->b", error, metric, error)
        self.assertTrue(np.allclose(lhs, rhs, rtol=2e-5, atol=2e-5))
        self.assertTrue(np.allclose(forward @ inverse, np.eye(6), atol=2e-5))
        self.assertEqual(mean.shape, (6,))

    def test_exact_rate_dp(self) -> None:
        def record(loss: float) -> dict:
            return {"diagnostics": {"validation_mse": loss}}

        curves = [
            {1: record(8.0), 2: record(4.0), 3: record(3.0)},
            {1: record(9.0), 2: record(8.0), 3: record(2.0)},
        ]
        allocation, loss = optimal_bit_allocation(curves, 4)
        self.assertEqual(allocation, [1, 3])
        self.assertEqual(loss, 10.0)

    def test_model_codes_and_scoring_roundtrip(self) -> None:
        groups = [np.asarray([0, 2]), np.asarray([1, 3])]
        bits = [2, 2]
        curves = []
        for group_index in range(2):
            centers = np.asarray(
                [
                    [0.0, 0.0],
                    [1.0 + group_index, 0.0],
                    [0.0, 2.0 + group_index],
                    [1.0, 1.0],
                ],
                dtype=np.float32,
            )
            curves.append(
                {
                    2: {
                        "centers": centers,
                        "counts": np.ones(4, dtype=np.int64),
                        "diagnostics": {"validation_mse": 1.0},
                    }
                }
            )
        model = assemble_model(
            artifact_id=11,
            node_count=3,
            metric_id=13,
            metric_lambda=4.0,
            residual_mean=np.asarray([0.1, 0.2, 0.3, 0.4], dtype=np.float32),
            forward_transform=np.eye(4, dtype=np.float32),
            inverse_transform=np.eye(4, dtype=np.float32),
            innovation_eigenvalues=np.ones(4, dtype=np.float32),
            groups=groups,
            bits=bits,
            curves=curves,
            variant="test",
        )
        raw_codes = [
            np.asarray([0, 1, 3], dtype=np.uint8),
            np.asarray([2, 3, 1], dtype=np.uint8),
        ]
        packed = pack_codes(raw_codes, bits)
        self.assertTrue(np.array_equal(unpack_group(packed, bits, 0), raw_codes[0]))
        self.assertTrue(np.array_equal(unpack_group(packed, bits, 1), raw_codes[1]))
        query = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
        score = residual_score_correction(packed, query, model)
        expected = []
        for row in range(3):
            reconstruction = model.residual_mean.copy()
            for group in range(2):
                reconstruction[groups[group]] += curves[group][2]["centers"][
                    raw_codes[group][row]
                ]
            expected.append(float(query @ reconstruction))
        self.assertTrue(np.allclose(score, expected, atol=1e-6))
        with tempfile.TemporaryDirectory() as directory:
            model_path = str(Path(directory) / "model.npz")
            side_path = str(Path(directory) / "codes.bin")
            write_model(model_path, model)
            loaded = read_model(model_path)
            side = initialize_side_codes(side_path, loaded)
            side[:] = packed
            side.flush()
            del side
            loaded_codes = read_side_codes(side_path, loaded)
            self.assertTrue(np.array_equal(loaded_codes, packed))


if __name__ == "__main__":
    unittest.main()
