#!/usr/bin/env python3
"""Train the database-MSE member of the fixed-budget group residual-VQ audit."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import (
    build_selected_query_metric,
    build_groups,
    extract_group_residual,
    lloyd_group_kmeans,
    resolve_device,
)
from group_residual_vq_io import (
    GroupResidualVQModel,
    compute_encoding_id,
    compute_metric_id,
    write_group_model,
)
from residual_pq_baseline import (
    is_identity_metric,
    lloyd_metric_group_kmeans,
)
from selected_subspaces_io import read_selected_subspaces


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument(
        "--selection-artifact",
        required=True,
        help="NPZ containing selected_subspaces and subspace_risks",
    )
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--assignment-metric",
        choices=("euclidean", "ood-query"),
        default="euclidean",
    )
    parser.add_argument("--query-fbin", default="")
    parser.add_argument("--metric-lambda", type=float, default=0.0)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=10000)
    parser.add_argument("--trace-epsilon", type=float, default=1e-12)
    parser.add_argument("--selected-subspaces", type=int, default=64)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument(
        "--group-policy",
        choices=("risk-round-robin", "contiguous"),
        default="risk-round-robin",
    )
    parser.add_argument("--sample-size", type=int, default=500000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--initialization",
        choices=("sample", "kmeans++"),
        default="sample",
        help=(
            "centroid initialization for metric Lloyd; kmeans++ runs D^2 "
            "sampling in the Mahalanobis-transformed residual space"
        ),
    )
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument("--kmeans-log-interval", type=int, default=1)
    parser.add_argument(
        "--training-cache",
        default="",
        help="optional .npy residual-cache path to keep; a temporary cache is used otherwise",
    )
    parser.add_argument(
        "--reuse-training-cache",
        action="store_true",
        help="reuse an existing --training-cache after validating metadata",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.selected_subspaces <= 0
        or args.groups <= 0
        or args.sample_size < 256
        or args.iterations <= 0
        or args.batch_size <= 0
        or args.metric_lambda < 0.0
        or args.query_start < 0
        or args.query_limit <= 0
        or args.trace_epsilon <= 0.0
        or (args.reuse_training_cache and not args.training_cache)
    ):
        raise ValueError("invalid group residual-VQ training configuration")
    if args.assignment_metric == "ood-query" and not args.query_fbin:
        raise ValueError("--query-fbin is required for ood-query assignment")
    if args.assignment_metric == "euclidean" and (
        args.query_fbin or args.metric_lambda != 0.0
    ):
        raise ValueError(
            "Euclidean assignment does not accept query data or a nonzero lambda"
        )
    device = resolve_device(args.solver_device)
    pq_model = read_model(args.pq_model)
    if pq_model.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen R=0 model")
    selection_model = read_selected_subspaces(args.selection_artifact)
    if (
        selection_model.artifact_id != pq_model.artifact_id
        or selection_model.node_count != pq_model.node_count
        or selection_model.subspaces != pq_model.subspaces
        or selection_model.dsub != pq_model.dsub
        or selection_model.refined_subspaces < args.selected_subspaces
    ):
        raise ValueError("selection artifact is incompatible or too small")
    selected = np.asarray(
        selection_model.selected_subspaces[: args.selected_subspaces],
        dtype=np.uint32,
    )
    groups = build_groups(selected, args.groups, args.group_policy)
    group_dim = groups.shape[1] * pq_model.dsub
    identity = np.broadcast_to(
        np.eye(group_dim, dtype=np.float32),
        (groups.shape[0], group_dim, group_dim),
    ).copy()
    normalized_second_moment = np.zeros_like(identity)
    metric_matrices = identity
    metric_factors = identity.copy()
    metric_diagnostics: list[dict[str, float | int]] = []
    metric_artifact_id = 0
    if args.assignment_metric == "ood-query":
        queries = fbin_memmap(args.query_fbin)
        query_end = args.query_start + args.query_limit
        if queries.shape[1] != pq_model.dim or query_end > queries.shape[0]:
            raise ValueError("requested metric query range is invalid")
        transformed_queries = transform_vectors(
            np.asarray(queries[args.query_start:query_end], dtype=np.float32),
            pq_model,
        )
        (
            normalized_second_moment,
            metric_matrices,
            metric_factors,
            metric_diagnostics,
        ) = build_selected_query_metric(
            transformed_queries,
            groups,
            pq_model.dsub,
            args.metric_lambda,
            args.trace_epsilon,
        )
        metric_artifact_id = compute_metric_id(
            pq_model.artifact_id,
            groups,
            args.metric_lambda,
            normalized_second_moment,
        )
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("PQ codes do not match the model")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the model")

    rng = np.random.default_rng(args.seed)
    sample_size = min(int(args.sample_size), pq_model.node_count)
    sample_ids = (
        np.arange(pq_model.node_count, dtype=np.int64)
        if sample_size == pq_model.node_count
        else np.sort(rng.choice(pq_model.node_count, sample_size, replace=False))
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    temporary_cache = not bool(args.training_cache)
    if temporary_cache:
        descriptor, cache_path = tempfile.mkstemp(
            prefix="group_residual_vq_", suffix=".npy"
        )
        os.close(descriptor)
    else:
        cache_path = args.training_cache
    cache_meta_path = f"{cache_path}.meta.json"
    expected_cache_shape = (sample_size, groups.shape[0], group_dim)
    expected_cache_meta = {
        "version": 1,
        "artifact_id": int(pq_model.artifact_id),
        "node_count": int(pq_model.node_count),
        "groups": groups.astype(int).tolist(),
        "group_dim": int(group_dim),
        "sample_size": int(sample_size),
        "sample_seed": int(args.seed),
    }
    if args.reuse_training_cache:
        if not os.path.exists(cache_path) or not os.path.exists(cache_meta_path):
            raise ValueError(
                "--reuse-training-cache requested but cache or metadata is missing"
            )
        with open(cache_meta_path, encoding="utf-8") as handle:
            cache_meta = json.load(handle)
        if cache_meta != expected_cache_meta:
            raise ValueError("reused training cache metadata does not match this run")
        cache = np.load(cache_path, mmap_mode="r+")
        if cache.shape != expected_cache_shape or cache.dtype != np.float32:
            raise ValueError("reused training cache has incompatible shape or dtype")
    else:
        cache = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_cache_shape,
        )
    try:
        if not args.reuse_training_cache:
            for start in range(0, sample_size, args.batch_size):
                end = min(sample_size, start + args.batch_size)
                ids = sample_ids[start:end]
                batch_codes = np.asarray(codes[ids], dtype=np.uint8)
                vectors_t = transform_vectors(
                    np.asarray(base[ids], dtype=np.float32), pq_model
                )
                cache[start:end] = extract_group_residual(
                    vectors_t, batch_codes, pq_model, groups
                )
                if args.log_interval > 0 and (
                    end % args.log_interval < args.batch_size or end == sample_size
                ):
                    print(f"[group-residual-cache] {end}/{sample_size}", flush=True)
            cache.flush()
            with open(cache_meta_path, "w", encoding="utf-8") as handle:
                json.dump(expected_cache_meta, handle, indent=2, sort_keys=True)
        else:
            print(
                f"[group-residual-cache] reusing {cache_path} "
                f"shape={expected_cache_shape}",
                flush=True,
            )
        if is_identity_metric(metric_matrices) and args.initialization == "sample":
            prototypes, counts, diagnostics = lloyd_group_kmeans(
                cache,
                codewords=256,
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                log_interval=args.kmeans_log_interval,
            )
            diagnostics["assignment_kernel"] = "euclidean_exact"
        else:
            prototypes, counts, diagnostics = lloyd_metric_group_kmeans(
                cache,
                metric_factors=metric_factors,
                codewords=256,
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                initialization=args.initialization,
                log_interval=args.kmeans_log_interval,
            )
            diagnostics["assignment_kernel"] = (
                "euclidean_metric_exact"
                if is_identity_metric(metric_matrices)
                else "mahalanobis"
            )
    finally:
        del cache
        if temporary_cache and os.path.exists(cache_path):
            os.unlink(cache_path)
        if temporary_cache and os.path.exists(cache_meta_path):
            os.unlink(cache_meta_path)

    encoding_id = compute_encoding_id(
        pq_model.artifact_id,
        groups,
        prototypes,
        metric_matrices=(
            None if is_identity_metric(metric_matrices) else metric_matrices
        ),
    )
    model = GroupResidualVQModel(
        artifact_id=pq_model.artifact_id,
        encoding_id=encoding_id,
        node_count=pq_model.node_count,
        dim=pq_model.dim,
        subspaces=pq_model.subspaces,
        dsub=pq_model.dsub,
        groups=groups,
        codewords=256,
        prototypes=prototypes,
        prototype_counts=counts,
        training_objective=(
            "database_residual_mse_lloyd"
            if is_identity_metric(metric_matrices)
            else "selected_residual_ood_query_metric_assignment_lloyd"
        ),
        assignment_metric=args.assignment_metric,
        metric_lambda=args.metric_lambda,
        metric_artifact_id=metric_artifact_id,
        metric_matrices=metric_matrices,
        metric_factors=metric_factors,
    )
    model_path = output / "group_residual_vq_model.npz"
    write_group_model(str(model_path), model)
    summary = {
        "version": 1,
        "method": "fixed_budget_group_residual_vector_quantization",
        "training_objective": model.training_objective,
        "assignment_metric": model.assignment_metric,
        "metric_lambda": model.metric_lambda,
        "metric_artifact_id": model.metric_artifact_id,
        "uses_euclidean_assignment": model.uses_euclidean_assignment,
        "metric_definition": "A_g = I + lambda * normalized_second_moment_g",
        "metric_normalization": "group_dim_over_trace",
        "metric_query_range": (
            [args.query_start, args.query_start + args.query_limit]
            if args.assignment_metric == "ood-query"
            else None
        ),
        "metric_queries_used": (
            args.query_limit if args.assignment_metric == "ood-query" else 0
        ),
        "metric_diagnostics": metric_diagnostics,
        "group_policy": args.group_policy,
        "groups": groups.astype(int).tolist(),
        "group_count": model.group_count,
        "subspaces_per_group": model.subspaces_per_group,
        "group_dim": model.group_dim,
        "codewords_per_group": model.codewords,
        "bits_per_group": model.bits_per_group,
        "side_bits_per_node": model.side_bits_per_node,
        "side_bytes_per_node": model.side_bytes_per_node,
        "sample_size": sample_size,
        "kmeans": diagnostics,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "selection_artifact": os.path.abspath(args.selection_artifact),
            "query_fbin": (
                os.path.abspath(args.query_fbin) if args.query_fbin else None
            ),
            "model": os.path.abspath(model_path),
        },
        "config": vars(args),
    }
    with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote group residual-VQ model: {model_path}")
    print(f"wrote summary: {output / 'train_summary.json'}")


if __name__ == "__main__":
    main()
