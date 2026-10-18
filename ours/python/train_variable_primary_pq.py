#!/usr/bin/env python3
"""Train an exact-budget 8/9-bit variable-rate primary-PQ baseline.

The OPQ transform is frozen.  A 512-centroid codebook is trained for every
subspace, then the R subspaces with the largest database-MSE reduction over the
existing 256-centroid codebook receive one extra bit.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import resolve_device
from variable_primary_pq import (
    assigned_8bit_mse,
    lloyd_9bit_kmeans,
    split_initialize_9bit_centroids,
)
from variable_primary_pq_io import (
    VariablePrimaryPQModel,
    compute_encoding_id,
    write_variable_primary_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--extra-bits", type=int, default=64)
    parser.add_argument("--sample-size", type=int, default=1000000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument("--split-scale", type=float, default=0.5)
    parser.add_argument("--subspace-block-size", type=int, default=8)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument(
        "--training-cache",
        default="",
        help="optional .npy transformed-vector sample cache to keep",
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
        args.extra_bits <= 0
        or args.sample_size < 512
        or args.iterations <= 0
        or args.convergence_tolerance < 0.0
        or args.split_scale < 0.0
        or args.subspace_block_size <= 0
        or args.batch_size <= 0
        or (args.reuse_training_cache and not args.training_cache)
    ):
        raise ValueError("invalid variable-primary-PQ training configuration")
    device = resolve_device(args.solver_device)
    primary = read_model(args.pq_model)
    if primary.refined_subspaces != 0 or primary.codewords != 256:
        raise ValueError("--pq-model must be a frozen 8-bit R=0 primary model")
    if args.extra_bits > primary.subspaces:
        raise ValueError("extra bits cannot exceed the number of subspaces")
    codes = read_codes(args.codes, expected_artifact_id=primary.artifact_id)
    base = fbin_memmap(args.base_fbin, primary.node_count)
    if codes.shape != (primary.node_count, primary.subspaces):
        raise ValueError("primary codes do not match the model")
    if base.shape != (primary.node_count, primary.dim):
        raise ValueError("base vectors do not match the model")

    rng = np.random.default_rng(args.seed)
    sample_size = min(int(args.sample_size), primary.node_count)
    sample_ids = (
        np.arange(primary.node_count, dtype=np.int64)
        if sample_size == primary.node_count
        else np.sort(rng.choice(primary.node_count, sample_size, replace=False))
    )
    sample_codes = np.asarray(codes[sample_ids], dtype=np.uint8)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    temporary_cache = not bool(args.training_cache)
    if temporary_cache:
        descriptor, cache_path = tempfile.mkstemp(
            prefix="variable_primary_pq_", suffix=".npy"
        )
        os.close(descriptor)
    else:
        cache_path = args.training_cache
    cache_meta_path = f"{cache_path}.meta.json"
    expected_meta = {
        "version": 1,
        "primary_artifact_id": int(primary.artifact_id),
        "node_count": int(primary.node_count),
        "dim": int(primary.dim),
        "subspaces": int(primary.subspaces),
        "dsub": int(primary.dsub),
        "sample_size": int(sample_size),
        "sample_seed": int(args.seed),
    }
    cache_shape = (sample_size, primary.subspaces, primary.dsub)
    if args.reuse_training_cache:
        if not os.path.exists(cache_path) or not os.path.exists(cache_meta_path):
            raise ValueError("requested transformed-vector cache is missing")
        with open(cache_meta_path, encoding="utf-8") as handle:
            actual_meta = json.load(handle)
        if actual_meta != expected_meta:
            raise ValueError("reused transformed-vector cache metadata mismatch")
        cache = np.load(cache_path, mmap_mode="r")
        if cache.shape != cache_shape or cache.dtype != np.float32:
            raise ValueError("reused transformed-vector cache shape mismatch")
        print(f"[variable-primary-cache] reusing {cache_path}", flush=True)
    else:
        cache = np.lib.format.open_memmap(
            cache_path, mode="w+", dtype=np.float32, shape=cache_shape
        )
        for start in range(0, sample_size, args.batch_size):
            end = min(sample_size, start + args.batch_size)
            transformed = transform_vectors(
                np.asarray(base[sample_ids[start:end]], dtype=np.float32), primary
            )
            cache[start:end] = transformed.reshape(
                end - start, primary.subspaces, primary.dsub
            )
            if args.log_interval > 0 and (
                end % args.log_interval < args.batch_size or end == sample_size
            ):
                print(f"[variable-primary-cache] {end}/{sample_size}", flush=True)
        cache.flush()
        with open(cache_meta_path, "w", encoding="utf-8") as handle:
            json.dump(expected_meta, handle, indent=2, sort_keys=True)

    try:
        mse_8bit = assigned_8bit_mse(cache, sample_codes, primary.centroids)
        initial = split_initialize_9bit_centroids(
            cache,
            sample_codes,
            primary.centroids,
            split_scale=args.split_scale,
        )
        centroids_9bit = np.empty(
            (primary.subspaces, 512, primary.dsub), dtype=np.float32
        )
        counts_9bit = np.empty((primary.subspaces, 512), dtype=np.int64)
        mse_9bit = np.empty(primary.subspaces, dtype=np.float64)
        diagnostics = []
        for begin in range(0, primary.subspaces, args.subspace_block_size):
            end = min(primary.subspaces, begin + args.subspace_block_size)
            centers, counts, mse, block_diag = lloyd_9bit_kmeans(
                cache[:, begin:end],
                initial[begin:end],
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                log_prefix=f"9bit-kmeans-{begin}-{end}",
            )
            centroids_9bit[begin:end] = centers
            counts_9bit[begin:end] = counts
            mse_9bit[begin:end] = mse
            block_diag.update({"subspace_begin": begin, "subspace_end": end})
            diagnostics.append(block_diag)
    finally:
        del cache
        if temporary_cache and os.path.exists(cache_path):
            os.unlink(cache_path)
        if temporary_cache and os.path.exists(cache_meta_path):
            os.unlink(cache_meta_path)

    utility = mse_8bit - mse_9bit
    indices = np.arange(primary.subspaces, dtype=np.int64)
    ranking = np.lexsort((indices, -utility))
    selected = ranking[: args.extra_bits].astype(np.uint32)
    encoding_id = compute_encoding_id(
        primary.artifact_id,
        selected,
        centroids_9bit,
        centroids_8bit=primary.centroids,
        metric_matrices=np.broadcast_to(
            np.eye(primary.dsub, dtype=np.float32),
            (primary.subspaces, primary.dsub, primary.dsub),
        ),
    )
    counts_8bit = np.stack(
        [
            np.bincount(sample_codes[:, subspace], minlength=256)
            for subspace in range(primary.subspaces)
        ],
        axis=0,
    ).astype(np.int64)
    model = VariablePrimaryPQModel(
        primary_artifact_id=primary.artifact_id,
        encoding_id=encoding_id,
        node_count=primary.node_count,
        dim=primary.dim,
        subspaces=primary.subspaces,
        dsub=primary.dsub,
        selected_subspaces=selected,
        centroids_9bit=centroids_9bit,
        centroid_counts_9bit=counts_9bit,
        mse_8bit=mse_8bit,
        mse_9bit=mse_9bit,
        utility=utility,
        selection_policy="top_database_mse_reduction_from_8_to_9_bits",
        training_objective="database_reconstruction_mse_lloyd",
        assignment_metric="euclidean",
        metric_lambda=0.0,
        metric_artifact_id=0,
        centroids_8bit=np.asarray(primary.centroids, dtype=np.float32).copy(),
        centroid_counts_8bit=counts_8bit,
    )
    model_path = output / "variable_primary_pq_model.npz"
    write_variable_primary_model(str(model_path), model)
    summary = {
        "version": 1,
        "method": "exact_budget_variable_bit_primary_pq",
        "primary_opq_frozen": True,
        "selection_policy": model.selection_policy,
        "training_objective": model.training_objective,
        "node_count": model.node_count,
        "dim": model.dim,
        "subspaces": model.subspaces,
        "dsub": model.dsub,
        "base_bits_per_subspace": 8,
        "refined_bits_per_selected_subspace": 9,
        "selected_subspaces": selected.astype(int).tolist(),
        "selected_subspace_count": model.refined_subspaces,
        "code_bits_per_node": model.code_bits_per_node,
        "code_bytes_per_node": model.code_bytes_per_node,
        "sample_size": sample_size,
        "sample_mse": {
            "existing_8bit_total": float(np.sum(mse_8bit)),
            "variable_8_9bit_total": float(
                np.sum(mse_8bit) - np.sum(utility[selected])
            ),
            "selected_utility_total": float(np.sum(utility[selected])),
            "negative_utility_subspaces": int(np.sum(utility < 0.0)),
            "mse_8bit_by_subspace": mse_8bit.tolist(),
            "mse_9bit_by_subspace": mse_9bit.tolist(),
            "utility_by_subspace": utility.tolist(),
        },
        "kmeans_blocks": diagnostics,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "model": os.path.abspath(model_path),
        },
        "config": vars(args),
    }
    with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"selected subspaces: {selected.astype(int).tolist()}")
    print(f"wrote variable-primary-PQ model: {model_path}")
    print(f"wrote summary: {output / 'train_summary.json'}")


if __name__ == "__main__":
    main()
