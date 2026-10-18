#!/usr/bin/env python3
"""Build residual samples and Reg/Pure OOD metrics for one primary quantizer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap
from primary_quantizer_manifest import read_manifest
from residual_metric_io import (
    ResidualMetric,
    compute_metric_id,
    write_residual_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--query-fbin", default="")
    parser.add_argument("--query-start", type=int, default=10000)
    parser.add_argument("--query-limit", type=int, default=10000)
    parser.add_argument("--residual-sample-size", type=int, default=1000000)
    parser.add_argument("--residual-batch-size", type=int, default=65536)
    parser.add_argument("--lambda-values", default="2 3 4 5 6 8")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--out-dir", required=True)
    return parser.parse_args()


def matrix_factor(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values, vectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    tolerance = 1e-9 * max(float(values[-1]), 1.0)
    if float(values[0]) < -tolerance:
        raise ValueError("query second moment is not positive semidefinite")
    clipped = np.maximum(values, 0.0)
    factor = (vectors * np.sqrt(clipped)) @ vectors.T
    return factor, values


def emit_metric(
    output: Path,
    artifact_id: int,
    moment: np.ndarray,
    observations: int,
    metric_form: str,
    metric_lambda: float,
) -> dict:
    dim = int(moment.shape[0])
    trace = float(np.trace(moment))
    if trace <= 0.0:
        raise ValueError("query second moment has zero trace")
    normalized = (dim / trace) * moment
    if metric_form == "pure":
        matrix = moment
        factor, eigenvalues = matrix_factor(matrix)
        name = "pure-ood"
    else:
        matrix = np.eye(dim, dtype=np.float64) + metric_lambda * normalized
        factor = np.linalg.cholesky(matrix)
        eigenvalues = np.linalg.eigvalsh(matrix)
        name = f"regularized-lambda-{metric_lambda:g}"
    metric_id = compute_metric_id(
        artifact_id,
        dim,
        1,
        "ood-query",
        metric_lambda,
        normalized,
        geometry="full",
        metric_form=metric_form,
    )
    metric = ResidualMetric(
        artifact_id=artifact_id,
        metric_id=metric_id,
        dim=dim,
        groups=1,
        mode="ood-query",
        metric_lambda=metric_lambda,
        normalized_second_moment=normalized.astype(np.float32),
        matrices=matrix.astype(np.float32),
        factors=factor.astype(np.float32),
        observations=observations,
        total_weight=float(observations),
        geometry="full",
        metric_form=metric_form,
    )
    path = output / "metrics" / f"{name}.npz"
    write_residual_metric(str(path), metric)
    return {
        "name": name,
        "path": str(path.resolve()),
        "metric_form": metric_form,
        "metric_lambda": metric_lambda,
        "raw_trace": trace,
        "min_eigenvalue": float(eigenvalues[0]),
        "max_eigenvalue": float(eigenvalues[-1]),
        "numerical_rank": int(
            np.sum(eigenvalues > 1e-8 * max(float(eigenvalues[-1]), 1.0))
        ),
    }


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0
        or args.query_limit <= 0
        or args.residual_sample_size <= 0
        or args.residual_batch_size <= 0
    ):
        raise ValueError("invalid primary-agnostic input configuration")
    manifest = read_manifest(args.primary_manifest)
    query_path = args.query_fbin or manifest.query_fbin
    queries = fbin_memmap(query_path)
    query_end = args.query_start + args.query_limit
    if queries.shape[1] != manifest.dim or query_end > queries.shape[0]:
        raise ValueError("query range/dimension is incompatible with primary manifest")
    q = np.asarray(queries[args.query_start:query_end], dtype=np.float64)
    moment = q.T @ q / q.shape[0]
    moment = 0.5 * (moment + moment.T)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    sample_size = min(args.residual_sample_size, manifest.node_count)
    rng = np.random.default_rng(args.seed)
    ids = np.sort(rng.choice(manifest.node_count, sample_size, replace=False))
    base = fbin_memmap(manifest.base_fbin)
    reconstruction = fbin_memmap(manifest.reconstruction_fbin)
    cache_path = output / "residual-sample.npy"
    cache = np.lib.format.open_memmap(
        cache_path, mode="w+", dtype=np.float32, shape=(sample_size, manifest.dim)
    )
    for start in range(0, sample_size, args.residual_batch_size):
        end = min(sample_size, start + args.residual_batch_size)
        batch_ids = ids[start:end]
        cache[start:end] = (
            np.asarray(base[batch_ids], dtype=np.float32)
            - np.asarray(reconstruction[batch_ids], dtype=np.float32)
        )
    cache.flush()
    del cache

    metrics = [
        emit_metric(
            output,
            manifest.artifact_id,
            moment,
            args.query_limit,
            "pure",
            0.0,
        )
    ]
    for value in args.lambda_values.split():
        metric_lambda = float(value)
        if metric_lambda < 0.0:
            raise ValueError("lambda values must be nonnegative")
        metrics.append(
            emit_metric(
                output,
                manifest.artifact_id,
                moment,
                args.query_limit,
                "regularized",
                metric_lambda,
            )
        )
    summary = {
        "version": 1,
        "method": "primary_agnostic_joint_innovation_inputs",
        "primary": manifest.name,
        "artifact_id": manifest.artifact_id,
        "query_range": [args.query_start, query_end],
        "query_fbin": str(Path(query_path).resolve()),
        "residual_sample_size": sample_size,
        "residual_sample_ids_sha256": __import__("hashlib").sha256(
            np.ascontiguousarray(ids, dtype="<u8").tobytes()
        ).hexdigest(),
        "residual_cache": str(cache_path.resolve()),
        "metrics": metrics,
        "config": vars(args),
    }
    with open(output / "inputs.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote primary-agnostic JI inputs: {output}")


if __name__ == "__main__":
    main()
