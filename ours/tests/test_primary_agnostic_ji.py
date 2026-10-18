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
    build_joint_transform,
    fit_vq,
    pack_codes,
    unpack_group,
)
from primary_quantizer_manifest import (  # noqa: E402
    PrimaryQuantizerManifest,
    compute_artifact_id,
    read_manifest,
    write_fbin,
    write_manifest,
)
from residual_metric_io import (  # noqa: E402
    ResidualMetric,
    compute_metric_id,
    read_residual_metric,
    write_residual_metric,
)


class PrimaryAgnosticJiTest(unittest.TestCase):
    def test_zero_bit_vq_and_code_packing(self) -> None:
        import torch

        rng = np.random.default_rng(13)
        train = rng.normal(size=(100, 3)).astype(np.float32)
        validation = rng.normal(size=(20, 3)).astype(np.float32)
        centers, counts, diagnostics = fit_vq(
            train,
            validation,
            bits=0,
            iterations=3,
            batch_size=32,
            device=torch.device("cpu"),
            seed=9,
        )
        self.assertEqual(centers.shape, (1, 3))
        self.assertEqual(int(counts.sum()), train.shape[0])
        self.assertEqual(diagnostics["codewords"], 1)
        packed = pack_codes(
            [np.zeros(5, dtype=np.uint8), np.arange(5, dtype=np.uint8)],
            [0, 3],
        )
        np.testing.assert_array_equal(unpack_group(packed, [0, 3], 0), 0)
        np.testing.assert_array_equal(unpack_group(packed, [0, 3], 1), np.arange(5))

    def test_manifest_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base = np.arange(24, dtype=np.float32).reshape(6, 4)
            reconstruction = base - 0.25
            base_path = root / "base.fbin"
            reconstruction_path = root / "reconstruction.fbin"
            write_fbin(str(base_path), base)
            write_fbin(str(reconstruction_path), reconstruction)
            query_path = root / "query.fbin"
            write_fbin(str(query_path), base[:2])
            artifact_id = compute_artifact_id(
                "test-primary", 6, 4, str(reconstruction_path), 16
            )
            parity_path = root / "parity.json"
            parity_path.write_text('{"passed": true}', encoding="utf-8")
            manifest = PrimaryQuantizerManifest(
                name="test-primary",
                artifact_id=artifact_id,
                node_count=6,
                dim=4,
                base_fbin=str(base_path),
                reconstruction_fbin=str(reconstruction_path),
                query_fbin=str(query_path),
                nominal_code_bits_per_node=16,
                physical_bytes_per_node=3.0,
                scorer_parity_json=str(parity_path),
            )
            path = root / "manifest.json"
            write_manifest(str(path), manifest)
            loaded = read_manifest(str(path))
            self.assertEqual(loaded.artifact_id, artifact_id)
            self.assertEqual(loaded.name, "test-primary")

    def test_pure_psd_metric_roundtrip_and_transform(self) -> None:
        rng = np.random.default_rng(19)
        query = rng.normal(size=(200, 5))
        query[:, 4] = query[:, 0] + query[:, 1]
        moment = query.T @ query / query.shape[0]
        values, vectors = np.linalg.eigh(moment)
        factor = (vectors * np.sqrt(np.maximum(values, 0.0))) @ vectors.T
        metric_id = compute_metric_id(
            7,
            5,
            1,
            "ood-query",
            0.0,
            moment,
            geometry="full",
            metric_form="pure",
        )
        metric = ResidualMetric(
            artifact_id=7,
            metric_id=metric_id,
            dim=5,
            groups=1,
            mode="ood-query",
            metric_lambda=0.0,
            normalized_second_moment=moment.astype(np.float32),
            matrices=moment.astype(np.float32),
            factors=factor.astype(np.float32),
            observations=200,
            total_weight=200.0,
            geometry="full",
            metric_form="pure",
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "pure.npz"
            write_residual_metric(str(path), metric)
            loaded = read_residual_metric(str(path))
            self.assertEqual(loaded.metric_form, "pure")
            residual = rng.normal(size=(500, 5)).astype(np.float32)
            _, forward, inverse, _, diagnostics = build_joint_transform(
                residual, loaded.matrices, 1e-6
            )
            self.assertTrue(np.all(np.isfinite(forward)))
            self.assertTrue(np.all(np.isfinite(inverse)))
            self.assertLess(diagnostics["transform_inverse_max_error"], 1e-6)
            self.assertEqual(diagnostics["innovation_clipped_dimensions"], 1)


if __name__ == "__main__":
    unittest.main()
