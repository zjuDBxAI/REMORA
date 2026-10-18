import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fixed_candidate_io import (  # noqa: E402
    VISITED_MAGIC,
    VISITED_VERSION,
    write_id_matrix,
)
from audit_variable_primary_pq import main as audit_main  # noqa: E402
from drb_io import DrbModel, write_codes, write_model  # noqa: E402
from encode_variable_primary_pq import main as encode_main  # noqa: E402
from train_variable_primary_pq import main as train_main  # noqa: E402
from variable_primary_pq import nearest_metric_codes  # noqa: E402
from variable_primary_pq_io import (  # noqa: E402
    read_variable_primary_codes,
    read_variable_primary_model,
    unpack_variable_primary_codes,
)


def write_fbin(path: Path, values: np.ndarray) -> None:
    values = np.ascontiguousarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        np.asarray(values.shape, dtype="<u4").tofile(handle)
        values.tofile(handle)


def write_visited(path: Path, rows: list[np.ndarray]) -> None:
    offsets = np.zeros(len(rows) + 1, dtype="<u8")
    for index, row in enumerate(rows):
        offsets[index + 1] = offsets[index] + row.size
    ids = np.concatenate(rows).astype("<u4")
    with open(path, "wb") as handle:
        handle.write(
            struct.pack(
                "<8sIQIQ",
                VISITED_MAGIC,
                VISITED_VERSION,
                0,
                len(rows),
                ids.size,
            )
        )
        offsets.tofile(handle)
        ids.tofile(handle)


class VariablePrimaryPQTest(unittest.TestCase):
    def test_metric_assignment_matches_mahalanobis_bruteforce(self):
        rng = np.random.default_rng(19)
        values = rng.normal(size=(17, 3, 2)).astype(np.float32)
        centers = rng.normal(size=(3, 256, 2)).astype(np.float32)
        matrices = np.asarray(
            [
                [[2.0, 0.4], [0.4, 1.2]],
                [[1.1, -0.2], [-0.2, 2.4]],
                [[3.0, 0.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        factors = np.linalg.cholesky(matrices).astype(np.float32)
        actual = nearest_metric_codes(
            values,
            centers,
            factors,
            device=torch.device("cpu"),
            batch_size=7,
        )
        expected = np.empty_like(actual)
        for group in range(values.shape[1]):
            error = values[:, group, None, :] - centers[None, group, :, :]
            distance = np.einsum(
                "nkd,de,nke->nk",
                error.astype(np.float64),
                matrices[group].astype(np.float64),
                error.astype(np.float64),
            )
            expected[:, group] = np.argmin(distance, axis=1)
        np.testing.assert_array_equal(actual, expected)

    def test_small_end_to_end_exact_budget_layout(self):
        rng = np.random.default_rng(83)
        nodes, query_count = 600, 4
        subspaces, dsub, dim = 8, 2, 16
        centroids = rng.normal(size=(subspaces, 256, dsub)).astype(np.float32)
        codes = rng.integers(0, 256, size=(nodes, subspaces), dtype=np.uint8)
        reconstructed = centroids[
            np.arange(subspaces, dtype=np.int64)[None, :], codes
        ].reshape(nodes, dim)
        base = reconstructed + rng.normal(0.0, 0.2, size=(nodes, dim)).astype(
            np.float32
        )
        queries = rng.normal(size=(query_count, dim)).astype(np.float32)
        primary = DrbModel(
            artifact_id=731,
            node_count=nodes,
            dim=dim,
            subspaces=subspaces,
            codewords=256,
            dsub=dsub,
            refined_subspaces=0,
            has_transform=False,
            transform=np.eye(dim, dtype=np.float32),
            transform_bias=np.zeros(dim, dtype=np.float32),
            centroids=centroids,
            selected_subspaces=np.empty(0, dtype=np.uint32),
            directions=np.empty((0, dsub), dtype=np.float32),
            thresholds=np.empty((0, 256), dtype=np.float32),
            beta=np.empty((0, 256, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq_path = root / "pq.bin"
            codes_path = root / "codes.bin"
            base_path = root / "base.fbin"
            query_path = root / "queries.fbin"
            model_dir = root / "model"
            cache_path = root / "cache.npy"
            variable_codes_path = root / "variable.codes.bin"
            visited_path = root / "visited.csrbin"
            adc_path = root / "adc.ids.ibin"
            truth_path = root / "truth.ids.ibin"
            audit_dir = root / "audit"
            write_model(str(pq_path), primary)
            write_codes(str(codes_path), codes, primary.artifact_id)
            write_fbin(base_path, base)
            write_fbin(query_path, queries)

            train_argv = [
                "train_variable_primary_pq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--out-dir", str(model_dir),
                "--extra-bits", "8",
                "--sample-size", str(nodes),
                "--iterations", "2",
                "--subspace-block-size", "4",
                "--solver-device", "cpu",
                "--batch-size", "128",
                "--log-interval", "0",
                "--training-cache", str(cache_path),
            ]
            with mock.patch.object(sys, "argv", train_argv):
                train_main()
            model_path = model_dir / "variable_primary_pq_model.npz"
            model = read_variable_primary_model(str(model_path))
            self.assertEqual(model.refined_subspaces, 8)
            self.assertEqual(model.code_bits_per_node, 72)
            self.assertEqual(model.code_bytes_per_node, 9)
            self.assertEqual(model.centroids_9bit.shape, (8, 512, 2))

            encode_argv = [
                "encode_variable_primary_pq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--variable-model", str(model_path),
                "--out-codes", str(variable_codes_path),
                "--solver-device", "cpu",
                "--batch-size", "128",
                "--subspace-block-size", "4",
                "--log-interval", "0",
            ]
            with mock.patch.object(sys, "argv", encode_argv):
                encode_main()
            records, width = read_variable_primary_codes(
                str(variable_codes_path), model
            )
            self.assertEqual(width, 9)
            self.assertEqual(records.shape, (nodes, 9))
            low, codes_9bit = unpack_variable_primary_codes(records, model)
            self.assertEqual(low.shape, (nodes, 8))
            self.assertEqual(codes_9bit.shape, (nodes, 8))
            self.assertTrue(np.all(codes_9bit < 512))

            candidate_rows = [np.arange(nodes, dtype=np.uint32)] * query_count
            write_visited(visited_path, candidate_rows)
            adc_ids = np.empty((query_count, 10), dtype=np.uint32)
            truth = np.empty((query_count, 10), dtype=np.uint32)
            for query_id, query in enumerate(queries):
                adc_score = reconstructed @ query
                exact_score = base @ query
                node_ids = np.arange(nodes, dtype=np.uint32)
                adc_ids[query_id] = node_ids[
                    np.lexsort((node_ids, -adc_score.astype(np.float64)))[:10]
                ]
                truth[query_id] = node_ids[
                    np.lexsort((node_ids, -exact_score.astype(np.float64)))[:10]
                ]
            write_id_matrix(str(adc_path), adc_ids)
            write_id_matrix(str(truth_path), truth)
            audit_argv = [
                "audit_variable_primary_pq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--query-fbin", str(query_path),
                "--variable-model", str(model_path),
                "--variable-codes", str(variable_codes_path),
                "--visited-csr", str(visited_path),
                "--adc-result-ids", str(adc_path),
                "--ground-truth", str(truth_path),
                "--out-dir", str(audit_dir),
                "--topk", "10",
                "--log-interval", "0",
            ]
            with mock.patch.object(sys, "argv", audit_argv):
                audit_main()
            summary = json.loads((audit_dir / "summary.json").read_text())
            self.assertEqual(summary["adc_replay"]["exact_order_match_rate"], 1.0)
            self.assertEqual(summary["code_bits_per_node"], 72)
            self.assertEqual(summary["code_bytes_per_node"], 9)
            self.assertEqual(summary["selected_subspace_count"], 8)


if __name__ == "__main__":
    unittest.main()
