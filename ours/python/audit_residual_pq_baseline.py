#!/usr/bin/env python3
"""Audit full-dimensional residual PQ on one fixed ADC candidate pool."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Dict

import numpy as np

from fixed_candidate_io import (
    exact_linear_score_parameters,
    overlap_with_reference,
    read_id_matrix,
    read_visited_csr,
    row_recall,
    write_float_matrix,
    write_id_matrix,
)
from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from residual_pq_baseline import (
    primary_conditioning_lut,
    residual_codebook_lut,
    residual_query_coordinates,
)
from residual_pq_baseline_io import (
    read_residual_pq_model,
    read_residual_pq_side_codes,
)


MODE_NAMES = ("adc", "residual_pq", "exact_residual")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--side-codes", required=True)
    parser.add_argument("--visited-csr", required=True)
    parser.add_argument("--adc-result-ids", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument(
        "--numerics",
        choices=("numpy", "cpp-reference"),
        default="numpy",
        help="cpp-reference reproduces the online scorer's float/double order",
    )
    return parser.parse_args()


def safe_ratio(numerator: float, denominator: float) -> float | None:
    if abs(denominator) <= 1e-15:
        return None
    return float(numerator / denominator)


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    pq_model = read_model(args.pq_model)
    if pq_model.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen R=0 primary model")
    residual_model = read_residual_pq_model(args.residual_model)
    if (
        residual_model.artifact_id != pq_model.artifact_id
        or residual_model.node_count != pq_model.node_count
        or residual_model.dim != pq_model.dim
    ):
        raise ValueError("primary PQ and residual-PQ models are incompatible")
    side_codes, side_width = read_residual_pq_side_codes(
        args.side_codes, residual_model
    )
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    queries = fbin_memmap(args.query_fbin)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("PQ codes do not match the model")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the model")
    if queries.shape[1] != pq_model.dim:
        raise ValueError("query vectors do not match the model")

    query_start, offsets, visited_ids = read_visited_csr(args.visited_csr)
    available = offsets.size - 1
    query_count = available if args.query_limit <= 0 else min(
        available, args.query_limit
    )
    if query_count <= 0 or query_start + query_count > queries.shape[0]:
        raise ValueError("invalid fixed-candidate query range")
    adc_dynamic = read_id_matrix(args.adc_result_ids)
    if adc_dynamic.shape[0] < query_count or adc_dynamic.shape[1] <= 0:
        raise ValueError("ADC results do not cover the residual-PQ audit")
    adc_replay_topk = min(args.topk, int(adc_dynamic.shape[1]))
    adc_dynamic = np.asarray(
        adc_dynamic[:query_count, :adc_replay_topk], dtype=np.uint32
    )
    truth = read_id_matrix(
        args.ground_truth,
        allow_trailing=True,
        row_start=query_start,
        row_count=query_count,
    )[:, : args.topk]

    results: Dict[str, np.ndarray] = {
        name: np.empty((query_count, args.topk), dtype=np.uint32)
        for name in MODE_NAMES
    }
    result_scores: Dict[str, np.ndarray] = {
        name: np.empty((query_count, args.topk), dtype=np.float32)
        for name in MODE_NAMES
    }
    candidate_counts = np.empty(query_count, dtype=np.uint32)
    all_subspaces = np.arange(pq_model.subspaces, dtype=np.int64)[None, :]
    transform64 = np.asarray(pq_model.transform, dtype=np.float64)
    centroid64 = np.asarray(pq_model.centroids, dtype=np.float64)
    prototype64 = np.asarray(residual_model.prototypes, dtype=np.float64)

    for local in range(query_count):
        begin, end = int(offsets[local]), int(offsets[local + 1])
        candidate_ids = np.asarray(visited_ids[begin:end], dtype=np.uint32)
        if candidate_ids.size < args.topk:
            raise ValueError("fixed candidate pool contains fewer nodes than topk")
        if int(np.max(candidate_ids)) >= pq_model.node_count:
            raise ValueError("visited candidate exceeds database range")
        candidate_counts[local] = candidate_ids.size
        candidate_codes = np.asarray(codes[candidate_ids], dtype=np.uint8)
        candidate_side = np.asarray(side_codes[candidate_ids], dtype=np.uint8)
        query = np.asarray(queries[query_start + local], dtype=np.float32)
        if args.numerics == "cpp-reference" and pq_model.has_transform:
            transformed = np.sum(
                transform64 * query.astype(np.float64, copy=False)[None, :],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
            query_t = np.asarray(
                transformed + pq_model.transform_bias, dtype=np.float32
            )
        else:
            query_t = transform_vectors(query[None, :], pq_model)[0]
        query_blocks = query_t.reshape(pq_model.subspaces, pq_model.dsub)
        residual_query = residual_query_coordinates(query_t, residual_model)
        if args.numerics == "cpp-reference":
            adc_lut = np.sum(
                centroid64 * query_blocks.astype(np.float64)[:, None, :],
                axis=2,
                dtype=np.float64,
            ).astype(np.float32)
            if residual_model.quantizer_kind == "additive":
                residual_lut = np.sum(
                    prototype64 * residual_query.astype(np.float64)[None, None, :],
                    axis=2,
                    dtype=np.float64,
                ).astype(np.float32)
            else:
                query_groups = residual_query.reshape(
                    residual_model.groups, residual_model.group_dim
                )
                residual_lut = np.sum(
                    prototype64 * query_groups.astype(np.float64)[:, None, :],
                    axis=2,
                    dtype=np.float64,
                ).astype(np.float32)
            if residual_model.has_primary_conditioning:
                projection = np.sum(
                    np.asarray(
                        residual_model.conditional_basis, dtype=np.float64
                    )
                    * residual_query.astype(np.float64)[:, None],
                    axis=0,
                    dtype=np.float64,
                ).astype(np.float32)
                conditional_lut = np.sum(
                    np.asarray(
                        residual_model.conditional_embeddings, dtype=np.float64
                    )
                    * projection.astype(np.float64)[None, None, :],
                    axis=2,
                    dtype=np.float64,
                ).astype(np.float32)
            else:
                conditional_lut = None
        else:
            adc_lut = np.einsum(
                "md,mkd->mk", query_blocks, pq_model.centroids, optimize=True
            ).astype(np.float32)
            residual_lut = None
            conditional_lut = None
        adc_terms = adc_lut[all_subspaces, candidate_codes]
        adc_scores = np.sum(adc_terms, axis=1, dtype=np.float64).astype(np.float32)
        if conditional_lut is None and residual_model.has_primary_conditioning:
            conditional_lut = primary_conditioning_lut(
                residual_query, residual_model
            )
        conditional_correction = np.zeros(candidate_ids.size, dtype=np.float32)
        if conditional_lut is not None:
            for subspace in range(pq_model.subspaces):
                conditional_correction += conditional_lut[
                    subspace, candidate_codes[:, subspace]
                ]
        if residual_lut is None:
            residual_lut = residual_codebook_lut(
                residual_query, residual_model
            )
        correction = np.zeros(candidate_ids.size, dtype=np.float32)
        for group in range(residual_model.groups):
            correction += residual_lut[group, candidate_side[:, group]]
        residual_scores = adc_scores + conditional_correction + correction
        candidate_base = np.asarray(base[candidate_ids], dtype=np.float32)
        exact_query, exact_bias = exact_linear_score_parameters(query_t, pq_model)
        exact_scores = candidate_base @ exact_query + exact_bias
        mode_scores = {
            "adc": adc_scores,
            "residual_pq": residual_scores,
            "exact_residual": exact_scores,
        }
        for name, scores in mode_scores.items():
            order = np.lexsort(
                (candidate_ids, -np.asarray(scores, dtype=np.float64))
            )[: args.topk]
            results[name][local] = candidate_ids[order]
            result_scores[name][local] = np.asarray(scores, dtype=np.float32)[order]
        if args.log_interval > 0 and (
            (local + 1) % args.log_interval == 0 or local + 1 == query_count
        ):
            print(f"[residual-pq-fixed-audit] {local + 1}/{query_count}", flush=True)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in MODE_NAMES:
        write_id_matrix(str(output / f"{name}.ids.ibin"), results[name])
        write_float_matrix(str(output / f"{name}.scores.fbin"), result_scores[name])
    recalls = {name: row_recall(results[name], truth) for name in MODE_NAMES}
    adc_recall = recalls["adc"]
    metrics: Dict[str, object] = {}
    for name in MODE_NAMES:
        values = recalls[name]
        metrics[name] = {
            "recall": float(np.mean(values)),
            "recall_delta_vs_adc": float(np.mean(values - adc_recall)),
            "rows_improved_vs_adc": int(np.sum(values > adc_recall)),
            "rows_regressed_vs_adc": int(np.sum(values < adc_recall)),
            "mean_topk_overlap_with_adc": float(
                np.mean(
                    overlap_with_reference(results[name], results["adc"], args.topk)
                )
            ),
        }
    exact_gain = float(metrics["exact_residual"]["recall_delta_vs_adc"])
    residual_gain = float(metrics["residual_pq"]["recall_delta_vs_adc"])
    dynamic_overlap = overlap_with_reference(
        results["adc"][:, :adc_replay_topk], adc_dynamic, adc_replay_topk
    )
    summary = {
        "version": 1,
        "method": "fixed_candidate_full_dimensional_residual_pq_audit",
        "numerics": args.numerics,
        "assignment_metric": residual_model.assignment_metric,
        "quantizer_kind": residual_model.quantizer_kind,
        "primary_conditioned": residual_model.has_primary_conditioning,
        "conditional_rank": residual_model.conditional_rank,
        "residual_transform_kind": residual_model.transform_kind,
        "has_residual_transform": residual_model.has_residual_transform,
        "training_objective": residual_model.training_objective,
        "metric_lambda": residual_model.metric_lambda,
        "metric_artifact_id": residual_model.metric_artifact_id,
        "uses_euclidean_assignment": residual_model.uses_euclidean_assignment,
        "query_start": query_start,
        "queries": query_count,
        "topk": args.topk,
        "groups": residual_model.groups,
        "group_dim": residual_model.group_dim,
        "prototype_dim": residual_model.prototype_dim,
        "covered_residual_dimensions": residual_model.dim,
        "codewords_per_group": residual_model.codewords,
        "side_bits_per_node": residual_model.side_bits_per_node,
        "side_bytes_per_node": side_width,
        "candidate_pool": {
            "mean": float(np.mean(candidate_counts)),
            "min": int(np.min(candidate_counts)),
            "max": int(np.max(candidate_counts)),
            "total": int(np.sum(candidate_counts)),
        },
        "adc_replay": {
            "topk": int(adc_replay_topk),
            "exact_order_match_rate": float(
                np.mean(np.all(results["adc"][:, :adc_replay_topk] == adc_dynamic, axis=1))
            ),
            "mean_topk_overlap": float(np.mean(dynamic_overlap)),
        },
        "modes": metrics,
        "decomposition": {
            "residual_pq_retention_of_exact_gain": safe_ratio(
                residual_gain, exact_gain
            )
        },
        "definitions": {
            "residual_pq": (
                "ADC plus a fixed-length residual-code approximation of the complete primary-PQ residual"
            ),
            "exact_residual": "full-precision score on the same fixed ADC pool",
        },
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "residual_model": os.path.abspath(args.residual_model),
            "side_codes": os.path.abspath(args.side_codes),
            "visited_csr": os.path.abspath(args.visited_csr),
            "adc_result_ids": os.path.abspath(args.adc_result_ids),
            "ground_truth": os.path.abspath(args.ground_truth),
        },
    }
    with open(output / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
