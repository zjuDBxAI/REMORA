import sys
import unittest
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from residual_opq import (  # noqa: E402
    eigenvalue_balanced_rotation_from_covariance,
    random_orthogonal_rotation,
    transformed_residual_covariance,
)


class ResidualOPQTest(unittest.TestCase):
    def test_random_rotation_is_reproducible_and_orthogonal(self):
        first = random_orthogonal_rotation(12, 17)
        second = random_orthogonal_rotation(12, 17)
        np.testing.assert_array_equal(first, second)
        np.testing.assert_allclose(first.T @ first, np.eye(12), atol=2e-6)

    def test_covariance_estimator_centers_values(self):
        rng = np.random.default_rng(4)
        values = rng.normal(size=(41, 2, 3)).astype(np.float32)
        offset = np.asarray(
            [2.0, -1.0, 0.5, 3.0, -4.0, 1.5], dtype=np.float32
        ).reshape(1, 2, 3)
        values += offset
        actual = transformed_residual_covariance(
            values, torch.device("cpu"), batch_size=7
        )
        flat = values.reshape(values.shape[0], -1).astype(np.float64)
        expected = np.cov(flat, rowvar=False, bias=True)
        np.testing.assert_allclose(actual, expected, atol=2e-6)

    def test_eigenvalue_balancing_reduces_group_log_volume_range(self):
        eigenvalues = np.asarray(
            [256, 128, 64, 32, 16, 8, 4, 2], dtype=np.float64
        )
        covariance = np.diag(eigenvalues)
        rotation, diagnostics = eigenvalue_balanced_rotation_from_covariance(
            covariance, groups=2
        )
        np.testing.assert_allclose(rotation.T @ rotation, np.eye(8), atol=2e-6)
        rotated = (
            rotation.astype(np.float64).T
            @ covariance
            @ rotation.astype(np.float64)
        )
        np.testing.assert_allclose(
            rotated - np.diag(np.diag(rotated)), 0.0, atol=1e-8
        )
        balanced_logs = [
            float(np.sum(np.log(np.diag(rotated)[start : start + 4])))
            for start in (0, 4)
        ]
        naive_logs = [
            float(np.sum(np.log(eigenvalues[start : start + 4])))
            for start in (0, 4)
        ]
        self.assertLess(
            abs(balanced_logs[0] - balanced_logs[1]),
            abs(naive_logs[0] - naive_logs[1]),
        )
        self.assertLess(diagnostics["group_log_determinant_range"], 1e-10)


if __name__ == "__main__":
    unittest.main()
