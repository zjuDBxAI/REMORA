import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from fixed_candidate_io import (  # noqa: E402
    VISITED_MAGIC,
    VISITED_VERSION,
    write_id_matrix,
)
from audit_group_residual_vq import main as audit_main  # noqa: E402
from drb_io import DrbModel, write_codes, write_model  # noqa: E402
from encode_group_residual_vq import main as encode_main  # noqa: E402
from group_residual_vq import (  # noqa: E402
    build_selected_query_metric,
    build_groups,
    extract_group_residual,
    lloyd_group_kmeans,
)
from group_residual_vq_io import (  # noqa: E402
    GroupResidualVQModel,
    compute_encoding_id,
    initialize_group_side_codes,
    read_group_model,
    read_group_side_codes,
    write_group_model,
)
from train_group_residual_vq import main as train_main  # noqa: E402


def write_fbin(path: Path, values: np.ndarray) -> None:
    payload = np.ascontiguousarray(values, dtype="<f4")
    with open(path, "wb") as handle:
        np.asarray(payload.shape, dtype="<u4").tofile(handle)
        payload.tofile(handle)


def write_visited(path: Path, candidate_rows: list[np.ndarray]) -> None:
    offsets = np.zeros(len(candidate_rows) + 1, dtype="<u8")
    for index, row in enumerate(candidate_rows):
        offsets[index + 1] = offsets[index] + row.size
    ids = np.concatenate(candidate_rows).astype("<u4")
    with open(path, "wb") as handle:
        handle.write(
            struct.pack(
                "<8sIQIQ",
                VISITED_MAGIC,
                VISITED_VERSION,
                0,
                len(candidate_rows),
                ids.size,
            )
        )
        offsets.tofile(handle)
        ids.tofile(handle)


class GroupResidualVQTest(unittest.TestCase):
    def test_group_helpers_and_artifact_roundtrip(self):
        selected = np.asarray([0, 1, 2, 3], dtype=np.uint32)
        np.testing.assert_array_equal(
            build_groups(selected, 2, "risk-round-robin"), [[0, 2], [1, 3]]
        )
        residual = np.asarray(
            [[[-2.0], [-2.0]], [[-1.0], [-1.0]], [[1.0], [1.0]], [[2.0], [2.0]]],
            dtype=np.float32,
        )
        prototypes, counts, diagnostics = lloyd_group_kmeans(
            residual,
            codewords=2,
            iterations=5,
            convergence_tolerance=0.0,
            device=__import__("torch").device("cpu"),
            batch_size=2,
            seed=1,
            log_interval=0,
        )
        self.assertEqual(prototypes.shape, (2, 2, 1))
        np.testing.assert_array_equal(np.sum(counts, axis=1), [4, 4])
        self.assertGreaterEqual(diagnostics["iterations"], 1)

        query_values = np.asarray(
            [[1.0, 0.0, 0.0, 2.0], [2.0, 0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        normalized, matrices, factors, metric_diagnostics = (
            build_selected_query_metric(
                query_values,
                np.asarray([[1, 0]], dtype=np.uint32),
                dsub=2,
                metric_lambda=1.0,
            )
        )
        self.assertEqual(normalized.shape, (1, 4, 4))
        self.assertGreater(float(np.max(matrices - np.eye(4))), 0.0)
        np.testing.assert_allclose(
            matrices,
            np.einsum("gde,gfe->gdf", factors, factors),
            rtol=2e-5,
            atol=2e-6,
        )
        self.assertEqual(len(metric_diagnostics), 1)

        full_prototypes = np.zeros((1, 256, 4), dtype=np.float32)
        groups = np.asarray([[0, 1]], dtype=np.uint32)
        model = GroupResidualVQModel(
            artifact_id=7,
            encoding_id=compute_encoding_id(7, groups, full_prototypes),
            node_count=3,
            dim=4,
            subspaces=2,
            dsub=2,
            groups=groups,
            codewords=256,
            prototypes=full_prototypes,
            prototype_counts=np.zeros((1, 256), dtype=np.int64),
            training_objective="test",
        )
        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "model.npz"
            side_path = Path(directory) / "side.bin"
            write_group_model(str(model_path), model)
            restored = read_group_model(str(model_path))
            side = initialize_group_side_codes(str(side_path), restored)
            side[:] = np.asarray([[1], [2], [3]], dtype=np.uint8)
            side.flush()
            del side
            mapped, width = read_group_side_codes(str(side_path), restored)
            self.assertEqual(width, 1)
            np.testing.assert_array_equal(mapped[:, 0], [1, 2, 3])
    def test_small_end_to_end_training_encoding_and_audit(self):
        rng = np.random.default_rng(9)
        nodes, queries_count = 300, 4
        codes = rng.integers(0, 2, size=(nodes, 2), dtype=np.uint8)
        centroids = np.asarray(
            [
                [[-1.0, 0.0], [1.0, 0.0]],
                [[0.0, -1.0], [0.0, 1.0]],
            ],
            dtype=np.float32,
        )
        reconstructed = np.empty((nodes, 4), dtype=np.float32)
        reconstructed[:, :2] = centroids[0, codes[:, 0]]
        reconstructed[:, 2:] = centroids[1, codes[:, 1]]
        base = reconstructed + rng.normal(0.0, 0.2, size=(nodes, 4)).astype(
            np.float32
        )
        queries = rng.normal(size=(queries_count, 4)).astype(np.float32)
        pq = DrbModel(
            artifact_id=77,
            node_count=nodes,
            dim=4,
            subspaces=2,
            codewords=2,
            dsub=2,
            refined_subspaces=0,
            has_transform=False,
            transform=np.eye(4, dtype=np.float32),
            transform_bias=np.zeros(4, dtype=np.float32),
            centroids=centroids,
            selected_subspaces=np.empty(0, dtype=np.uint32),
            directions=np.empty((0, 2), dtype=np.float32),
            thresholds=np.empty((0, 2), dtype=np.float32),
            beta=np.empty((0, 2, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq_path = root / "pq.bin"
            codes_path = root / "codes.bin"
            base_path = root / "base.fbin"
            query_path = root / "query.fbin"
            selection_path = root / "selection.npz"
            mse_dir = root / "mse"
            side_path = root / "side.bin"
            visited_path = root / "visited.csrbin"
            adc_path = root / "adc.ibin"
            truth_path = root / "truth.ibin"
            audit_dir = root / "audit"
            query_aware_dir = root / "query-aware"
            query_aware_side = root / "query-aware-side.bin"
            query_aware_audit = root / "query-aware-audit"
            write_model(str(pq_path), pq)
            write_codes(str(codes_path), codes, pq.artifact_id)
            write_fbin(base_path, base)
            write_fbin(query_path, queries)
            np.savez_compressed(
                selection_path,
                artifact_id=np.asarray(77, dtype=np.uint64),
                node_count=np.asarray(nodes, dtype=np.uint64),
                subspaces=np.asarray(2, dtype=np.uint32),
                dsub=np.asarray(2, dtype=np.uint32),
                refined_subspaces=np.asarray(2, dtype=np.uint32),
                selected_subspaces=np.asarray([0, 1], dtype=np.uint32),
                subspace_risks=np.asarray([2.0, 1.0], dtype=np.float64),
            )

            train_argv = [
                "train_group_residual_vq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--selection-artifact", str(selection_path),
                "--out-dir", str(mse_dir),
                "--selected-subspaces", "2",
                "--groups", "1",
                "--sample-size", str(nodes),
                "--iterations", "2",
                "--solver-device", "cpu",
                "--batch-size", "64",
                "--log-interval", "0",
                "--kmeans-log-interval", "0",
            ]
            with mock.patch.object(sys, "argv", train_argv):
                train_main()
            mse_model_path = mse_dir / "group_residual_vq_model.npz"
            encode_argv = [
                "encode_group_residual_vq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--group-model", str(mse_model_path),
                "--out-side-codes", str(side_path),
                "--solver-device", "cpu",
                "--batch-size", "64",
                "--log-interval", "0",
            ]
            with mock.patch.object(sys, "argv", encode_argv):
                encode_main()

            candidate_rows = [np.arange(nodes, dtype=np.uint32)] * queries_count
            write_visited(visited_path, candidate_rows)
            adc_ids = np.empty((queries_count, 10), dtype=np.uint32)
            truth_ids = np.empty((queries_count, 10), dtype=np.uint32)
            for qid, query in enumerate(queries):
                ids = np.arange(nodes, dtype=np.uint32)
                adc_order = np.lexsort((ids, -(reconstructed @ query)))[:10]
                exact_order = np.lexsort((ids, -(base @ query)))[:10]
                adc_ids[qid] = ids[adc_order]
                truth_ids[qid] = ids[exact_order]
            write_id_matrix(str(adc_path), adc_ids)
            write_id_matrix(str(truth_path), truth_ids)
            audit_argv = [
                "audit_group_residual_vq.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--query-fbin", str(query_path),
                "--group-model", str(mse_model_path),
                "--side-codes", str(side_path),
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
            self.assertEqual(summary["side_bits_per_node"], 8)
            self.assertIn("group_vq_mse", summary["modes"])

            query_aware_argv = train_argv.copy()
            query_aware_argv[
                query_aware_argv.index(str(mse_dir))
            ] = str(query_aware_dir)
            query_aware_argv.extend(
                [
                    "--assignment-metric", "ood-query",
                    "--query-fbin", str(query_path),
                    "--metric-lambda", "1",
                    "--query-limit", str(queries_count),
                ]
            )
            with mock.patch.object(sys, "argv", query_aware_argv):
                train_main()
            query_aware_model_path = (
                query_aware_dir / "group_residual_vq_model.npz"
            )
            query_aware_model = read_group_model(str(query_aware_model_path))
            self.assertEqual(query_aware_model.assignment_metric, "ood-query")
            self.assertEqual(query_aware_model.metric_lambda, 1.0)
            self.assertFalse(query_aware_model.uses_euclidean_assignment)
            query_aware_encode_argv = encode_argv.copy()
            query_aware_encode_argv[
                query_aware_encode_argv.index(str(mse_model_path))
            ] = str(query_aware_model_path)
            query_aware_encode_argv[
                query_aware_encode_argv.index(str(side_path))
            ] = str(query_aware_side)
            with mock.patch.object(sys, "argv", query_aware_encode_argv):
                encode_main()
            query_aware_audit_argv = audit_argv.copy()
            query_aware_audit_argv[
                query_aware_audit_argv.index(str(mse_model_path))
            ] = str(query_aware_model_path)
            query_aware_audit_argv[
                query_aware_audit_argv.index(str(side_path))
            ] = str(query_aware_side)
            query_aware_audit_argv[
                query_aware_audit_argv.index(str(audit_dir))
            ] = str(query_aware_audit)
            with mock.patch.object(sys, "argv", query_aware_audit_argv):
                audit_main()
            query_aware_summary = json.loads(
                (query_aware_audit / "summary.json").read_text()
            )
            self.assertIn("group_vq_query_aware", query_aware_summary["modes"])
            self.assertEqual(query_aware_summary["metric_lambda"], 1.0)


if __name__ == "__main__":
    unittest.main()
