#!/usr/bin/env python3
"""Build OOD-query or decision-event metrics for full-coverage residual PQ."""

from __future__ import annotations

import argparse
import csv
import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_model, transform_vectors
from residual_metric_io import (
    ResidualMetric,
    compute_metric_id,
    write_residual_metric,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument(
        "--mode", choices=("ood-query", "decision-event"), required=True
    )
    parser.add_argument("--events-csv", default="")
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument(
        "--geometry",
        choices=("blockwise", "full"),
        default="blockwise",
        help=(
            "blockwise reproduces the existing per-group metric; full stores "
            "one complete query covariance for a global residual transform"
        ),
    )
    parser.add_argument("--metric-lambda", type=float, default=1.0)
    parser.add_argument(
        "--metric-form",
        choices=("regularized", "pure"),
        default="regularized",
        help=(
            "regularized builds I + lambda * trace_normalized(E[qq^T]); "
            "pure builds the unnormalized E[qq^T] with no task-weight parameter"
        ),
    )
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=10000)
    parser.add_argument("--inversion-weight", type=float, default=1.0)
    parser.add_argument("--protect-weight", type=float, default=1.0)
    parser.add_argument(
        "--normalize-per-query",
        action="store_true",
        help="normalize the total accepted event weight of each query to one",
    )
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--trace-epsilon", type=float, default=1e-12)
    return parser.parse_args()


def decision_query_weights(
    path: str,
    query_start: int,
    query_end: int,
    inversion_weight: float,
    protect_weight: float,
    normalize_per_query: bool,
) -> tuple[dict[int, float], dict[str, int], int]:
    weights: dict[int, float] = defaultdict(float)
    kinds = {"inversion": 0, "protect": 0}
    used_events = 0
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"query_id", "kind", "weight"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"event CSV is missing fields: {sorted(required)}")
        for row in reader:
            query_id = int(row["query_id"])
            if query_id < query_start or query_id >= query_end:
                continue
            kind = str(row["kind"])
            if kind == "inversion":
                multiplier = inversion_weight
            elif kind == "protect":
                multiplier = protect_weight
            else:
                continue
            value = float(row["weight"]) * multiplier
            if not np.isfinite(value) or value < 0.0:
                raise ValueError("event weights must be finite and nonnegative")
            if value == 0.0:
                continue
            weights[query_id] += value
            kinds[kind] += 1
            used_events += 1
    if normalize_per_query:
        weights = {query_id: 1.0 for query_id in weights}
    return dict(weights), kinds, used_events


def accumulate_second_moment(
    queries: np.ndarray,
    pq_model,
    groups: int,
    query_ids: np.ndarray,
    weights: np.ndarray,
    batch_size: int,
    geometry: str = "blockwise",
) -> tuple[np.ndarray, float]:
    group_dim = pq_model.dim // groups
    moment = (
        np.zeros((groups, group_dim, group_dim), dtype=np.float64)
        if geometry == "blockwise"
        else np.zeros((pq_model.dim, pq_model.dim), dtype=np.float64)
    )
    total_weight = 0.0
    for start in range(0, query_ids.size, batch_size):
        end = min(query_ids.size, start + batch_size)
        ids = query_ids[start:end]
        batch_weights = np.asarray(weights[start:end], dtype=np.float64)
        transformed = transform_vectors(
            np.asarray(queries[ids], dtype=np.float32), pq_model
        ).astype(np.float64)
        if geometry == "blockwise":
            grouped = transformed.reshape(-1, groups, group_dim)
            moment += np.einsum(
                "b,bgd,bge->gde",
                batch_weights,
                grouped,
                grouped,
                optimize=True,
            )
        else:
            moment += np.einsum(
                "b,bd,be->de",
                batch_weights,
                transformed,
                transformed,
                optimize=True,
            )
        total_weight += float(np.sum(batch_weights))
    if total_weight <= 0.0:
        raise ValueError("metric construction has zero total query weight")
    return moment / total_weight, total_weight


def main() -> None:
    args = parse_args()
    if (
        args.groups <= 0
        or args.metric_lambda < 0.0
        or args.query_start < 0
        or args.query_limit <= 0
        or args.inversion_weight < 0.0
        or args.protect_weight < 0.0
        or args.batch_size <= 0
        or args.trace_epsilon <= 0.0
    ):
        raise ValueError("invalid residual metric configuration")
    if args.mode == "decision-event" and not args.events_csv:
        raise ValueError("--events-csv is required for decision-event mode")
    if args.mode == "ood-query" and args.events_csv:
        raise ValueError("--events-csv is only valid for decision-event mode")
    if args.metric_form == "pure" and args.mode != "ood-query":
        raise ValueError("pure metric form is only defined for ood-query mode")
    pq_model = read_model(args.pq_model)
    if pq_model.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen R=0 primary model")
    if pq_model.dim % args.groups != 0:
        raise ValueError(
            f"primary dimension {pq_model.dim} is not divisible by {args.groups} groups"
        )
    queries = fbin_memmap(args.query_fbin)
    query_end = args.query_start + args.query_limit
    if queries.shape[1] != pq_model.dim or query_end > queries.shape[0]:
        raise ValueError("requested metric query range is invalid")

    event_counts = {"inversion": 0, "protect": 0}
    used_events = 0
    if args.mode == "ood-query":
        query_ids = np.arange(args.query_start, query_end, dtype=np.int64)
        query_weights = np.ones(query_ids.size, dtype=np.float64)
    else:
        weight_map, event_counts, used_events = decision_query_weights(
            args.events_csv,
            args.query_start,
            query_end,
            args.inversion_weight,
            args.protect_weight,
            args.normalize_per_query,
        )
        if not weight_map:
            raise ValueError("no decision events fall in the requested query range")
        query_ids = np.asarray(sorted(weight_map), dtype=np.int64)
        query_weights = np.asarray(
            [weight_map[int(query_id)] for query_id in query_ids], dtype=np.float64
        )

    second_moment, total_weight = accumulate_second_moment(
        queries,
        pq_model,
        args.groups,
        query_ids,
        query_weights,
        args.batch_size,
        args.geometry,
    )
    group_dim = pq_model.dim // args.groups
    metric_dim = group_dim if args.geometry == "blockwise" else pq_model.dim
    normalized = np.zeros_like(second_moment)
    matrices = np.empty_like(second_moment)
    factors = np.empty_like(second_moment)
    identity = np.eye(metric_dim, dtype=np.float64)
    diagnostics = []
    metric_count = args.groups if args.geometry == "blockwise" else 1
    for group in range(metric_count):
        raw = second_moment[group] if args.geometry == "blockwise" else second_moment
        trace = float(np.trace(raw))
        normalized_matrix = np.zeros_like(raw)
        if trace > args.trace_epsilon:
            normalized_matrix = (metric_dim / trace) * raw
        matrix = (
            raw
            if args.metric_form == "pure"
            else identity + args.metric_lambda * normalized_matrix
        )
        matrix = 0.5 * (matrix + matrix.T)
        if args.metric_form == "pure":
            eigenvalues, eigenvectors = np.linalg.eigh(matrix)
            tolerance = args.trace_epsilon * max(float(eigenvalues[-1]), 1.0)
            if float(eigenvalues[0]) < -tolerance:
                raise ValueError("query second moment is not positive semidefinite")
            clipped = np.maximum(eigenvalues, 0.0)
            factor = (eigenvectors * np.sqrt(clipped)) @ eigenvectors.T
        else:
            factor = np.linalg.cholesky(matrix)
        if args.geometry == "blockwise":
            normalized[group] = normalized_matrix
            matrices[group] = matrix
            factors[group] = factor
        else:
            normalized[:] = normalized_matrix
            matrices[:] = matrix
            factors[:] = factor
        eigenvalues = np.linalg.eigvalsh(matrix)
        diagnostics.append(
            {
                "group": group if args.geometry == "blockwise" else None,
                "scope": args.geometry,
                "raw_trace": trace,
                "normalized_trace": float(np.trace(normalized_matrix)),
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
                "condition_number": (
                    float(eigenvalues[-1] / eigenvalues[0])
                    if eigenvalues[0] > 0.0
                    else None
                ),
            }
        )
    if args.metric_form == "regularized" and args.metric_lambda == 0.0:
        matrices[:] = identity
        factors[:] = identity

    metric_id = compute_metric_id(
        pq_model.artifact_id,
        pq_model.dim,
        args.groups,
        args.mode,
        args.metric_lambda,
        normalized,
        geometry=args.geometry,
        metric_form=args.metric_form,
    )
    metric = ResidualMetric(
        artifact_id=pq_model.artifact_id,
        metric_id=metric_id,
        dim=pq_model.dim,
        groups=args.groups,
        mode=args.mode,
        metric_lambda=args.metric_lambda,
        normalized_second_moment=normalized.astype(np.float32),
        matrices=matrices.astype(np.float32),
        factors=factors.astype(np.float32),
        observations=int(query_ids.size),
        total_weight=total_weight,
        geometry=args.geometry,
        metric_form=args.metric_form,
    )
    write_residual_metric(args.out_model, metric)
    summary_path = (
        Path(args.out_summary)
        if args.out_summary
        else Path(args.out_model).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "method": "full_coverage_residual_quantization_metric",
        "geometry": args.geometry,
        "mode": args.mode,
        "metric_form": args.metric_form,
        "metric_lambda": args.metric_lambda,
        "normalization": (
            "none"
            if args.metric_form == "pure"
            else (
                "group_dim_over_trace"
                if args.geometry == "blockwise"
                else "full_dim_over_trace"
            )
        ),
        "matrix_definition": (
            "W = E[q q^T]"
            if args.metric_form == "pure"
            else (
                "A_g = I + lambda * normalized_second_moment_g"
                if args.geometry == "blockwise"
                else "H = I + lambda * normalized_full_second_moment"
            )
        ),
        "query_range": [args.query_start, query_end],
        "queries_used": int(query_ids.size),
        "events_used": used_events,
        "event_counts": event_counts,
        "event_query_weighting": (
            "per_query_equal" if args.normalize_per_query else "csv_weight_by_kind"
        ),
        "total_weight": total_weight,
        "groups": args.groups,
        "group_dim": group_dim,
        "diagnostics": diagnostics,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "queries": os.path.abspath(args.query_fbin),
            "events_csv": (
                os.path.abspath(args.events_csv) if args.events_csv else None
            ),
            "metric_model": os.path.abspath(args.out_model),
        },
        "config": vars(args),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote residual metric: {args.out_model}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
