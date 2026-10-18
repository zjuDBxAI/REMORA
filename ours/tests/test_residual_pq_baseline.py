import csv
import json
import subprocess
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
from audit_residual_pq_baseline import main as audit_main  # noqa: E402
from audit_residual_pq_certificates import (  # noqa: E402
    ResidualRun,
    compute_node_etas,
    compute_query_lh,
)
from build_residual_metric import main as metric_main  # noqa: E402
from compare_paired_recall import main as compare_paired_main  # noqa: E402
from drb_io import DrbModel, write_codes, write_model  # noqa: E402
from encode_residual_pq_baseline import main as encode_main  # noqa: E402
from residual_pq_baseline import (  # noqa: E402
    coded_residual_reconstruction,
    extract_full_residual,
    reconstruct_primary_pq,
    transform_grouped_residual,
)
from build_quantized_query_base_knn import exact_quantized_knn  # noqa: E402
from quantized_roargraph import reconstruct_quantized_batch  # noqa: E402
from residual_pq_baseline_io import (  # noqa: E402
    RESIDUAL_PQ_ONLINE_MAGIC,
    read_residual_pq_model,
    read_residual_pq_side_codes,
    write_residual_pq_online_model,
)
from residual_metric_io import read_residual_metric  # noqa: E402
from train_additive_residual_quantizer import main as additive_train_main  # noqa: E402
from train_residual_pq_baseline import main as train_main  # noqa: E402
from verify_residual_pq_degeneracy import main as verify_main  # noqa: E402


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


class ResidualPQBaselineTest(unittest.TestCase):
    def test_small_end_to_end(self):
        rng = np.random.default_rng(31)
        nodes, query_count = 300, 4
        dim, subspaces, dsub, primary_codewords = 8, 4, 2, 2
        centroids = rng.normal(
            size=(subspaces, primary_codewords, dsub)
        ).astype(np.float32)
        codes = rng.integers(
            0, primary_codewords, size=(nodes, subspaces), dtype=np.uint8
        )
        reconstructed = centroids[
            np.arange(subspaces, dtype=np.int64)[None, :], codes
        ].reshape(nodes, dim)
        base = reconstructed + rng.normal(0.0, 0.25, size=(nodes, dim)).astype(
            np.float32
        )
        queries = rng.normal(size=(query_count, dim)).astype(np.float32)
        pq = DrbModel(
            artifact_id=918,
            node_count=nodes,
            dim=dim,
            subspaces=subspaces,
            codewords=primary_codewords,
            dsub=dsub,
            refined_subspaces=0,
            has_transform=False,
            transform=np.eye(dim, dtype=np.float32),
            transform_bias=np.zeros(dim, dtype=np.float32),
            centroids=centroids,
            selected_subspaces=np.empty(0, dtype=np.uint32),
            directions=np.empty((0, dsub), dtype=np.float32),
            thresholds=np.empty((0, primary_codewords), dtype=np.float32),
            beta=np.empty((0, primary_codewords, 2), dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            pq_path = root / "pq.bin"
            codes_path = root / "codes.bin"
            base_path = root / "base.fbin"
            query_path = root / "query.fbin"
            model_dir = root / "model"
            training_cache_path = root / "residual-cache.npy"
            metric_zero_path = root / "metric-zero.npz"
            metric_zero_dir = root / "model-metric-zero"
            metric_zero_side_path = root / "side-metric-zero.bin"
            metric_one_path = root / "metric-one.npz"
            metric_full_path = root / "metric-full.npz"
            metric_full_dir = root / "model-metric-full"
            metric_full_opq_dir = root / "model-metric-full-opq"
            metric_full_multistart_dir = root / "model-metric-full-multistart"
            metric_full_side_path = root / "side-metric-full.bin"
            additive_dir = root / "model-additive"
            additive_side_path = root / "side-additive.bin"
            conditioned_dir = root / "model-additive-conditioned"
            conditioned_side_path = root / "side-additive-conditioned.bin"
            conditioned_online_path = root / "online-additive-conditioned.bin"
            metric_decision_path = root / "metric-decision.npz"
            metric_event_path = root / "metric-events.csv"
            metric_one_dir = root / "model-metric-one"
            metric_one_side_path = root / "side-metric-one.bin"
            parity_path = root / "lambda-zero-parity.json"
            side_path = root / "side.bin"
            online_model_path = root / "online.bin"
            visited_path = root / "visited.csrbin"
            adc_path = root / "adc.ibin"
            truth_path = root / "truth.ibin"
            audit_dir = root / "audit"
            paired_path = root / "paired.json"
            write_model(str(pq_path), pq)
            write_codes(str(codes_path), codes, pq.artifact_id)
            write_fbin(base_path, base)
            write_fbin(query_path, queries)
            with open(metric_event_path, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=["query_id", "kind", "weight"]
                )
                writer.writeheader()
                writer.writerow({"query_id": 0, "kind": "inversion", "weight": 1.0})
                writer.writerow({"query_id": 1, "kind": "protect", "weight": 0.5})

            expected_reconstruction = reconstruct_primary_pq(codes, pq)
            np.testing.assert_allclose(expected_reconstruction, reconstructed)
            residual = extract_full_residual(base, codes, pq, groups=2)
            self.assertEqual(residual.shape, (nodes, 2, 4))
            np.testing.assert_allclose(
                residual.reshape(nodes, dim), base - reconstructed, atol=1e-7
            )

            train_argv = [
                "train_residual_pq_baseline.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--out-dir", str(model_dir),
                "--groups", "2",
                "--codewords", "256",
                "--sample-size", str(nodes),
                "--iterations", "2",
                "--solver-device", "cpu",
                "--batch-size", "64",
                "--log-interval", "0",
                "--kmeans-log-interval", "0",
                "--training-cache", str(training_cache_path),
            ]
            with mock.patch.object(sys, "argv", train_argv):
                train_main()
            model_path = model_dir / "residual_pq_model.npz"
            model = read_residual_pq_model(str(model_path))
            self.assertEqual(model.group_dim, 4)
            self.assertEqual(model.side_bits_per_node, 16)

            for metric_lambda, metric_path in (
                (0.0, metric_zero_path),
                (1.0, metric_one_path),
            ):
                metric_argv = [
                    "build_residual_metric.py",
                    "--pq-model", str(pq_path),
                    "--query-fbin", str(query_path),
                    "--mode", "ood-query",
                    "--out-model", str(metric_path),
                    "--groups", "2",
                    "--metric-lambda", str(metric_lambda),
                    "--query-start", "0",
                    "--query-limit", str(query_count),
                    "--batch-size", "2",
                ]
                with mock.patch.object(sys, "argv", metric_argv):
                    metric_main()
            metric_zero = read_residual_metric(str(metric_zero_path))
            metric_one = read_residual_metric(str(metric_one_path))
            metric_full_argv = [
                "build_residual_metric.py",
                "--pq-model", str(pq_path),
                "--query-fbin", str(query_path),
                "--mode", "ood-query",
                "--geometry", "full",
                "--out-model", str(metric_full_path),
                "--groups", "2",
                "--metric-lambda", "1",
                "--query-start", "0",
                "--query-limit", str(query_count),
                "--batch-size", "2",
            ]
            with mock.patch.object(sys, "argv", metric_full_argv):
                metric_main()
            metric_full = read_residual_metric(str(metric_full_path))
            self.assertEqual(metric_full.geometry, "full")
            self.assertEqual(metric_full.matrices.shape, (dim, dim))
            decision_metric_argv = [
                "build_residual_metric.py",
                "--pq-model", str(pq_path),
                "--query-fbin", str(query_path),
                "--mode", "decision-event",
                "--events-csv", str(metric_event_path),
                "--out-model", str(metric_decision_path),
                "--groups", "2",
                "--metric-lambda", "1",
                "--query-start", "0",
                "--query-limit", str(query_count),
            ]
            with mock.patch.object(sys, "argv", decision_metric_argv):
                metric_main()
            metric_decision = read_residual_metric(str(metric_decision_path))
            self.assertEqual(metric_decision.mode, "decision-event")
            self.assertEqual(metric_decision.observations, 2)
            expected_identity = np.broadcast_to(
                np.eye(4, dtype=np.float32), (2, 4, 4)
            )
            np.testing.assert_array_equal(metric_zero.matrices, expected_identity)
            self.assertGreater(
                float(np.max(metric_one.matrices - expected_identity)), 0.0
            )

            for metric_path, output_dir in (
                (metric_zero_path, metric_zero_dir),
                (metric_one_path, metric_one_dir),
            ):
                metric_train_argv = train_argv.copy()
                metric_train_argv[
                    metric_train_argv.index(str(model_dir))
                ] = str(output_dir)
                metric_train_argv.extend(["--metric-model", str(metric_path)])
                metric_train_argv.append("--reuse-training-cache")
                with mock.patch.object(sys, "argv", metric_train_argv):
                    train_main()
            metric_zero_model = read_residual_pq_model(
                str(metric_zero_dir / "residual_pq_model.npz")
            )
            metric_one_model = read_residual_pq_model(
                str(metric_one_dir / "residual_pq_model.npz")
            )
            np.testing.assert_array_equal(metric_zero_model.prototypes, model.prototypes)
            np.testing.assert_array_equal(
                metric_zero_model.prototype_counts, model.prototype_counts
            )
            self.assertTrue(metric_zero_model.uses_euclidean_assignment)
            self.assertFalse(metric_one_model.uses_euclidean_assignment)

            for output_dir, rotation in (
                (metric_full_dir, "none"),
                (metric_full_opq_dir, "opq"),
            ):
                full_train_argv = train_argv.copy()
                full_train_argv[full_train_argv.index(str(model_dir))] = str(output_dir)
                full_train_argv.extend(
                    [
                        "--metric-model", str(metric_full_path),
                        "--residual-rotation", rotation,
                        "--rotation-iterations", "1",
                        "--rotation-lloyd-iterations", "1",
                        "--reuse-training-cache",
                    ]
                )
                with mock.patch.object(sys, "argv", full_train_argv):
                    train_main()
            multistart_argv = train_argv.copy()
            multistart_argv[multistart_argv.index(str(model_dir))] = str(
                metric_full_multistart_dir
            )
            multistart_argv.extend(
                [
                    "--metric-model", str(metric_full_path),
                    "--residual-rotation", "opq",
                    "--rotation-iterations", "1",
                    "--rotation-lloyd-iterations", "1",
                    "--rotation-initialization", "multistart",
                    "--rotation-random-restarts", "1",
                    "--rotation-restart-initialization", "sample",
                    "--rotation-checkpoint", "best",
                    "--reuse-training-cache",
                ]
            )
            with mock.patch.object(sys, "argv", multistart_argv):
                train_main()
            full_model = read_residual_pq_model(
                str(metric_full_dir / "residual_pq_model.npz")
            )
            full_opq_model = read_residual_pq_model(
                str(metric_full_opq_dir / "residual_pq_model.npz")
            )
            full_multistart_model = read_residual_pq_model(
                str(metric_full_multistart_dir / "residual_pq_model.npz")
            )
            with open(
                metric_full_multistart_dir / "train_summary.json", encoding="utf-8"
            ) as handle:
                multistart_summary = json.load(handle)
            self.assertEqual(
                len(multistart_summary["kmeans"]["rotation_search"]["starts"]), 3
            )
            self.assertIn(
                multistart_summary["kmeans"]["rotation_search"]["selected"]["start"],
                {"identity", "covariance-balanced", "random-1"},
            )
            for transformed_model in (
                full_model, full_opq_model, full_multistart_model
            ):
                self.assertTrue(transformed_model.has_residual_transform)
                np.testing.assert_allclose(
                    transformed_model.residual_transform
                    @ transformed_model.residual_inverse_transform,
                    np.eye(dim),
                    atol=2e-5,
                )
                np.testing.assert_allclose(
                    transformed_model.residual_transform
                    @ transformed_model.residual_transform.T,
                    metric_full.matrices,
                    atol=2e-5,
                )

            additive_common_argv = [
                "train_additive_residual_quantizer.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--metric-model", str(metric_full_path),
                "--groups", "2",
                "--codewords", "256",
                "--sample-size", str(nodes),
                "--initialization-iterations", "1",
                "--joint-sweeps", "1",
                "--coordinate-iterations", "1",
                "--solver-device", "cpu",
                "--batch-size", "64",
                "--training-cache", str(training_cache_path),
                "--reuse-training-cache",
                "--log-interval", "0",
            ]
            with mock.patch.object(
                sys,
                "argv",
                additive_common_argv
                + ["--out-dir", str(additive_dir), "--conditional-rounds", "0"],
            ):
                additive_train_main()
            with mock.patch.object(
                sys,
                "argv",
                additive_common_argv
                + [
                    "--out-dir", str(conditioned_dir),
                    "--conditional-rank", "2",
                    "--conditional-rounds", "1",
                    "--conditional-epochs", "1",
                    "--initial-residual-model",
                    str(additive_dir / "residual_pq_model.npz"),
                    "--freeze-additive-codebooks",
                ],
            ):
                additive_train_main()
            additive_model = read_residual_pq_model(
                str(additive_dir / "residual_pq_model.npz")
            )
            conditioned_model = read_residual_pq_model(
                str(conditioned_dir / "residual_pq_model.npz")
            )
            self.assertEqual(additive_model.quantizer_kind, "additive")
            self.assertEqual(additive_model.prototypes.shape, (2, 256, dim))
            self.assertFalse(additive_model.has_primary_conditioning)
            self.assertTrue(conditioned_model.has_primary_conditioning)
            self.assertEqual(conditioned_model.conditional_rank, 2)
            np.testing.assert_array_equal(
                conditioned_model.prototypes, additive_model.prototypes
            )
            self.assertEqual(
                conditioned_model.conditional_embeddings.shape,
                (subspaces, primary_codewords, 2),
            )

            encode_argv = [
                "encode_residual_pq_baseline.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--residual-model", str(model_path),
                "--out-side-codes", str(side_path),
                "--solver-device", "cpu",
                "--batch-size", "64",
                "--log-interval", "0",
            ]
            with mock.patch.object(sys, "argv", encode_argv):
                encode_main()
            side, width = read_residual_pq_side_codes(str(side_path), model)
            self.assertEqual(side.shape, (nodes, 2))
            self.assertEqual(width, 2)
            quantized = reconstruct_quantized_batch(
                codes, side, pq, model
            )
            brute_scores = queries @ quantized.T
            expected_ids = np.argsort(-brute_scores, axis=1)[:, :10]
            knn_ids, knn_scores = exact_quantized_knn(
                queries,
                codes,
                side,
                pq,
                model,
                topk=10,
                query_batch_size=2,
                base_batch_size=64,
                device=torch.device("cpu"),
                log_interval=0,
            )
            np.testing.assert_array_equal(knn_ids, expected_ids)
            np.testing.assert_allclose(
                knn_scores,
                np.take_along_axis(brute_scores, expected_ids, axis=1),
                atol=1e-5,
            )
            full_encode_argv = encode_argv.copy()
            full_encode_argv[full_encode_argv.index(str(model_path))] = str(
                metric_full_dir / "residual_pq_model.npz"
            )
            full_encode_argv[full_encode_argv.index(str(side_path))] = str(
                metric_full_side_path
            )
            with mock.patch.object(sys, "argv", full_encode_argv):
                encode_main()
            full_side, _ = read_residual_pq_side_codes(
                str(metric_full_side_path), full_model
            )
            full_quantized = reconstruct_quantized_batch(
                codes, full_side, pq, full_model
            )
            full_ids, full_scores = exact_quantized_knn(
                queries,
                codes,
                full_side,
                pq,
                full_model,
                topk=10,
                query_batch_size=2,
                base_batch_size=64,
                device=torch.device("cpu"),
                log_interval=0,
            )
            full_brute_scores = queries @ full_quantized.T
            full_expected_ids = np.argsort(-full_brute_scores, axis=1)[:, :10]
            np.testing.assert_array_equal(full_ids, full_expected_ids)
            np.testing.assert_allclose(
                full_scores,
                np.take_along_axis(
                    full_brute_scores, full_expected_ids, axis=1
                ),
                atol=1e-5,
            )
            for additive_candidate, additive_side_path_candidate in (
                (additive_model, additive_side_path),
                (conditioned_model, conditioned_side_path),
            ):
                additive_encode_argv = encode_argv.copy()
                additive_encode_argv[
                    additive_encode_argv.index(str(model_path))
                ] = str(
                    additive_dir / "residual_pq_model.npz"
                    if additive_candidate is additive_model
                    else conditioned_dir / "residual_pq_model.npz"
                )
                additive_encode_argv[
                    additive_encode_argv.index(str(side_path))
                ] = str(additive_side_path_candidate)
                additive_encode_argv.extend(
                    ["--additive-coordinate-sweeps", "1"]
                )
                if additive_candidate is conditioned_model:
                    additive_encode_argv.extend(
                        [
                            "--additive-initial-model",
                            str(additive_dir / "residual_pq_model.npz"),
                            "--additive-initial-side-codes",
                            str(additive_side_path),
                        ]
                    )
                with mock.patch.object(sys, "argv", additive_encode_argv):
                    encode_main()
                additive_side, _ = read_residual_pq_side_codes(
                    str(additive_side_path_candidate), additive_candidate
                )
                additive_quantized = reconstruct_quantized_batch(
                    codes, additive_side, pq, additive_candidate
                )
                additive_brute_scores = queries @ additive_quantized.T
                additive_expected_ids = np.argsort(
                    -additive_brute_scores, axis=1
                )[:, :10]
                additive_ids, additive_scores = exact_quantized_knn(
                    queries,
                    codes,
                    additive_side,
                    pq,
                    additive_candidate,
                    topk=10,
                    query_batch_size=2,
                    base_batch_size=64,
                    device=torch.device("cpu"),
                    log_interval=0,
                )
                np.testing.assert_array_equal(
                    additive_ids, additive_expected_ids
                )
                np.testing.assert_allclose(
                    additive_scores,
                    np.take_along_axis(
                        additive_brute_scores, additive_expected_ids, axis=1
                    ),
                    atol=1e-5,
                )
            write_residual_pq_online_model(
                str(conditioned_online_path), pq, conditioned_model
            )
            with open(conditioned_online_path, "rb") as handle:
                self.assertEqual(handle.read(8), RESIDUAL_PQ_ONLINE_MAGIC)
            conditioned_side, _ = read_residual_pq_side_codes(
                str(conditioned_side_path), conditioned_model
            )
            conditioned_run = ResidualRun(
                name="conditioned",
                model_path=str(conditioned_dir / "residual_pq_model.npz"),
                side_codes_path=str(conditioned_side_path),
                model=conditioned_model,
                side_codes=conditioned_side,
            )
            certificate_nodes = np.arange(8, dtype=np.uint32)
            certificate_etas = compute_node_etas(
                certificate_nodes,
                base,
                pq,
                codes,
                [conditioned_run],
                batch_size=8,
                log_interval=0,
            )["conditioned"][:8]
            coded_residual = transform_grouped_residual(
                (base[:8] - reconstructed[:8]).reshape(8, 2, 4),
                conditioned_model.residual_transform,
            )
            coded_approximation = coded_residual_reconstruction(
                conditioned_side[:8], codes[:8], conditioned_model
            ).reshape(8, 2, 4)
            manual_etas = np.linalg.norm(
                (coded_residual - coded_approximation).reshape(8, dim), axis=1
            )
            np.testing.assert_allclose(
                certificate_etas, manual_etas, rtol=1e-5, atol=1e-6
            )
            query_lh = compute_query_lh(
                np.arange(query_count, dtype=np.int64),
                queries,
                pq,
                [conditioned_run],
                log_interval=0,
            )["conditioned"]
            expected_lh = np.linalg.norm(
                queries @ conditioned_model.residual_inverse_transform.T,
                axis=1,
            )
            np.testing.assert_allclose(
                query_lh, expected_lh, rtol=1e-5, atol=1e-6
            )
            probe = ROOT / "build" / "roar_drb_tests"
            if probe.exists():
                completed = subprocess.run(
                    [
                        str(probe),
                        str(conditioned_online_path),
                        str(codes_path),
                        str(conditioned_side_path),
                    ],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                cpp_scores = np.fromstring(completed.stdout, sep=" ")
                conditioned_quantized = reconstruct_quantized_batch(
                    codes[:8], conditioned_side[:8], pq, conditioned_model
                )
                probe_query = 0.1 * np.arange(1, dim + 1, dtype=np.float32)
                np.testing.assert_allclose(
                    cpp_scores,
                    conditioned_quantized @ probe_query,
                    rtol=1e-5,
                    atol=1e-6,
                )
            write_residual_pq_online_model(
                str(online_model_path), pq, model
            )
            with open(online_model_path, "rb") as handle:
                self.assertEqual(handle.read(8), RESIDUAL_PQ_ONLINE_MAGIC)

            for metric_model_path, output_path in (
                (metric_zero_dir / "residual_pq_model.npz", metric_zero_side_path),
                (metric_one_dir / "residual_pq_model.npz", metric_one_side_path),
            ):
                metric_encode_argv = encode_argv.copy()
                metric_encode_argv[
                    metric_encode_argv.index(str(model_path))
                ] = str(metric_model_path)
                metric_encode_argv[
                    metric_encode_argv.index(str(side_path))
                ] = str(output_path)
                with mock.patch.object(sys, "argv", metric_encode_argv):
                    encode_main()
            metric_zero_side, _ = read_residual_pq_side_codes(
                str(metric_zero_side_path), metric_zero_model
            )
            np.testing.assert_array_equal(metric_zero_side, side)
            verify_argv = [
                "verify_residual_pq_degeneracy.py",
                "--euclidean-model", str(model_path),
                "--euclidean-side-codes", str(side_path),
                "--metric-zero-model", str(
                    metric_zero_dir / "residual_pq_model.npz"
                ),
                "--metric-zero-side-codes", str(metric_zero_side_path),
                "--out-json", str(parity_path),
                "--chunk-size", "64",
            ]
            with mock.patch.object(sys, "argv", verify_argv):
                verify_main()
            self.assertTrue(json.loads(parity_path.read_text())["passed"])

            candidate_rows = [np.arange(nodes, dtype=np.uint32)] * query_count
            write_visited(visited_path, candidate_rows)
            adc_ids = np.empty((query_count, 10), dtype=np.uint32)
            truth_ids = np.empty((query_count, 10), dtype=np.uint32)
            ids = np.arange(nodes, dtype=np.uint32)
            for qid, query in enumerate(queries):
                adc_order = np.lexsort((ids, -(reconstructed @ query)))[:10]
                exact_order = np.lexsort((ids, -(base @ query)))[:10]
                adc_ids[qid] = ids[adc_order]
                truth_ids[qid] = ids[exact_order]
            write_id_matrix(str(adc_path), adc_ids)
            write_id_matrix(str(truth_path), truth_ids)

            audit_argv = [
                "audit_residual_pq_baseline.py",
                "--pq-model", str(pq_path),
                "--codes", str(codes_path),
                "--base-fbin", str(base_path),
                "--query-fbin", str(query_path),
                "--residual-model", str(model_path),
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
            self.assertEqual(summary["covered_residual_dimensions"], dim)
            self.assertEqual(summary["side_bits_per_node"], 16)
            self.assertIn("residual_pq", summary["modes"])
            self.assertEqual(summary["modes"]["exact_residual"]["recall"], 1.0)
            paired_argv = [
                "compare_paired_recall.py",
                "--ground-truth", str(truth_path),
                "--reference-ids", str(audit_dir / "residual_pq.ids.ibin"),
                "--candidate-ids", str(audit_dir / "residual_pq.ids.ibin"),
                "--bootstrap-samples", "100",
                "--bootstrap-batch-size", "20",
                "--out-json", str(paired_path),
            ]
            with mock.patch.object(sys, "argv", paired_argv):
                compare_paired_main()
            paired = json.loads(paired_path.read_text())
            self.assertEqual(paired["recall_delta"], 0.0)
            self.assertFalse(paired["significantly_better_95"])


if __name__ == "__main__":
    unittest.main()
