"""Native CPU codeword assignment preserves the variable-rate JI objective."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

PYTHON_DIR = Path(__file__).resolve().parents[1] / "python"
if str(PYTHON_DIR) not in sys.path:
    sys.path.insert(0, str(PYTHON_DIR))

import faiss
import numpy as np
import torch

from joint_innovation_rpq import (assign_vq, assemble_model, fit_vq, read_side_codes,
    unpack_group, vq_backend_info, write_model)
from primary_quantizer_manifest import PrimaryQuantizerManifest, write_fbin, write_manifest


def oracle(values, centers):
    distance = np.sum((values[:, None, :].astype(np.float64) - centers[None, :, :]) ** 2, axis=2)
    ids = np.argmin(distance, axis=1)
    return ids, float(distance[np.arange(len(values)), ids].mean())


class FaissVQ(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.threads = faiss.omp_get_max_threads()
        cls.torch_threads = torch.get_num_threads()
        faiss.omp_set_num_threads(2)
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        faiss.omp_set_num_threads(cls.threads)
        torch.set_num_threads(cls.torch_threads)

    def test_fp32_assignment_full_codebook_noncontiguous_and_short_tail(self):
        rng = np.random.default_rng(312)
        values = rng.normal(size=(137, 24)).astype(np.float32)[:, ::2]
        centers = rng.normal(size=(64, 12)).astype(np.float32)
        labels, mse = assign_vq(values, centers, 32, torch.device("cpu"))
        expected, loss = oracle(values, centers)
        np.testing.assert_array_equal(labels, expected)
        self.assertEqual(labels.dtype, np.uint8)
        self.assertAlmostEqual(mse, loss, places=5)

    def test_tied_codewords_choose_first_and_zero_bit_group_works(self):
        for count in (1, 64, 256):
            values = np.ones((35, 12), dtype=np.float32)
            centers = np.zeros((count, 12), dtype=np.float32)
            labels, mse = assign_vq(values, centers, 32, torch.device("cpu"))
            np.testing.assert_array_equal(labels, 0)
            self.assertEqual(mse, 12)

    def test_more_than_256_codewords_keeps_uint16_ids(self):
        centers = np.stack((np.arange(257), np.zeros(257)), axis=1).astype(np.float32)
        labels, mse = assign_vq(centers[-5:], centers, 3, torch.device("cpu"))
        np.testing.assert_array_equal(labels, np.arange(252, 257))
        self.assertEqual(labels.dtype, np.uint16)
        self.assertEqual(mse, 0)

    def test_cpu_assignment_does_not_use_torch_distance_or_argmin(self):
        with patch("torch.min", side_effect=AssertionError("torch reduction")), patch(
                "torch.argmin", side_effect=AssertionError("torch selection")), patch(
                "torch.matmul", side_effect=AssertionError("torch distances")):
            labels, mse = assign_vq(np.eye(8, dtype=np.float32), np.eye(8, dtype=np.float32), 3, torch.device("cpu"))
        np.testing.assert_array_equal(labels, np.arange(8))
        self.assertEqual(mse, 0)

    def test_fit_preserves_initialization_centroid_update_and_empty_cluster(self):
        train = np.array([[0, 0], [0, 2], [10, 10], [10, 12]], dtype=np.float32)
        initial = np.array([[0, 0], [10, 10], [100, 100], [-100, -100]], dtype=np.float32)
        centers, counts, diagnostics = fit_vq(train, train, bits=2, iterations=8,
            batch_size=3, device=torch.device("cpu"), seed=17, initial=initial)
        np.testing.assert_array_equal(centers, [[0, 1], [10, 11], [100, 100], [-100, -100]])
        np.testing.assert_array_equal(counts, [2, 2, 0, 0])
        self.assertEqual(diagnostics["empty_clusters"], 2)
        self.assertEqual(diagnostics["validation_mse"], 1)
        self.assertEqual(diagnostics["assignment_backend"]["backend"], "faiss-cpu-flat-l2-v1")

    def test_invalid_vectors_codebooks_and_batch_sizes_fail(self):
        good = np.ones((5, 3), dtype=np.float32)
        for values, centers, batch in ((good, good, 0), (good, good[:, :2], 2),
                (good * np.nan, good, 2), (good, good * np.inf, 2),
                (good, np.empty((0, 3), dtype=np.float32), 2)):
            with self.assertRaises(ValueError): assign_vq(values, centers, batch, torch.device("cpu"))
        labels, mse = assign_vq(good[:0], good, 2, torch.device("cpu"))
        self.assertEqual(len(labels), 0)
        self.assertEqual(mse, 0)

    def test_non_avx512_faiss_is_rejected_and_cuda_backend_is_retained(self):
        with patch("faiss.get_compile_options", return_value="OPTIMIZE AVX2"):
            with self.assertRaisesRegex(RuntimeError, "AVX-512"):
                assign_vq(np.ones((2, 3)), np.ones((1, 3)), 2, torch.device("cpu"))
        self.assertEqual(vq_backend_info(torch.device("cuda:0"))["backend"], "torch")

    def test_primary_agnostic_encoder_packs_expected_codes_and_records_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rng = np.random.default_rng(887)
            residual = rng.normal(size=(23, 4)).astype(np.float32)
            reconstruction = np.zeros_like(residual)
            write_fbin(root / "base.fbin", residual)
            write_fbin(root / "reconstruction.fbin", reconstruction)
            write_fbin(root / "query.fbin", residual[:2])
            (root / "parity.json").write_text(json.dumps(dict(passed=True)))
            manifest = PrimaryQuantizerManifest("fixture", 11, 23, 4, str(root / "base.fbin"),
                str(root / "reconstruction.fbin"), str(root / "query.fbin"), 32, 4,
                scorer_parity_json=str(root / "parity.json"))
            write_manifest(root / "manifest.json", manifest)
            groups = [np.array([0, 2]), np.array([1, 3])]
            curves = [{2: dict(centers=rng.normal(size=(4, 2)).astype(np.float32),
                counts=np.ones(4, dtype=np.int64), diagnostics=dict(validation_mse=1))} for _ in groups]
            model = assemble_model(artifact_id=11, node_count=23, metric_id=13, metric_lambda=0,
                residual_mean=np.zeros(4), forward_transform=np.eye(4), inverse_transform=np.eye(4),
                innovation_eigenvalues=np.ones(4), groups=groups, bits=[2, 2], curves=curves, variant="fixture")
            write_model(root / "model.npz", model)
            # Choose the encoder beside the imported implementation so staged
            # validation and installed production validation use the same code.
            import joint_innovation_rpq
            script = Path(joint_innovation_rpq.__file__).with_name("encode_primary_agnostic_joint_innovation.py")
            result = subprocess.run([sys.executable, str(script), "--primary-manifest", str(root / "manifest.json"),
                "--joint-model", str(root / "model.npz"), "--out-side-codes", str(root / "side.codes"),
                "--out-summary", str(root / "summary.json"), "--batch-size", "7"],
                env=dict(os.environ, OMP_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2", MKL_NUM_THREADS="2"),
                capture_output=True, text=True, timeout=30)
            self.assertEqual(result.returncode, 0, result.stderr)
            packed = read_side_codes(root / "side.codes", model)
            for group in range(2):
                expected, _ = oracle(residual[:, groups[group]], curves[group][2]["centers"])
                np.testing.assert_array_equal(unpack_group(packed, model.bits, group), expected)
            summary = json.loads((root / "summary.json").read_text())
            self.assertEqual(summary["assignment_backend"]["backend"], "faiss-cpu-flat-l2-v1")
            self.assertEqual(summary["side_bits_per_node"], 4)


if __name__ == "__main__": unittest.main()
