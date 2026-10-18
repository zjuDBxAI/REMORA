#!/usr/bin/env python3
"""Audit 864-bit variable-rate primary PQ on a fixed ADC candidate pool."""

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
from variable_primary_pq_io import (
    read_variable_primary_codes,
    read_variable_primary_model,
    unpack_variable_primary_codes,
)


MODE_NAMES = ("adc", "variable_primary_pq", "exact")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--variable-model", required=True)
    parser.add_argument("--variable-codes", required=True)
    parser.add_argument("--visited-csr", required=True)
    parser.add_argument("--adc-result-ids", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.topk <= 0:
        raise ValueError("topk must be positive")
    primary = read_model(args.pq_model)
    model = read_variable_primary_model(args.variable_model)
    if (
        primary.refined_subspaces != 0
        or primary.codewords != 256
        or model.primary_artifact_id != primary.artifact_id
        or model.node_count != primary.node_count
        or model.dim != primary.dim
        or model.subspaces != primary.subspaces
        or model.dsub != primary.dsub
    ):
        raise ValueError("variable and primary PQ models are incompatible")
    primary_codes = read_codes(
        args.codes, expected_artifact_id=primary.artifact_id
    )
    variable_codes, variable_width = read_variable_primary_codes(
        args.variable_codes, model
    )
    base = fbin_memmap(args.base_fbin, primary.node_count)
    queries = fbin_memmap(args.query_fbin)
    if primary_codes.shape != (primary.node_count, primary.subspaces):
        raise ValueError("primary code shape mismatch")
    if base.shape != (primary.node_count, primary.dim):
        raise ValueError("base vector shape mismatch")
    if queries.shape[1] != primary.dim:
        raise ValueError("query vector shape mismatch")

    query_start, offsets, visited_ids = read_visited_csr(args.visited_csr)
    available = offsets.size - 1
    query_count = available if args.query_limit <= 0 else min(
        available, args.query_limit
    )
    if query_count <= 0 or query_start + query_count > queries.shape[0]:
        raise ValueError("invalid fixed-candidate query range")
    adc_dynamic = read_id_matrix(args.adc_result_ids)
    if adc_dynamic.shape[0] < query_count or adc_dynamic.shape[1] < args.topk:
        raise ValueError("ADC result matrix does not cover this audit")
    adc_dynamic = np.asarray(
        adc_dynamic[:query_count, : args.topk], dtype=np.uint32
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
    all_subspaces = np.arange(primary.subspaces, dtype=np.int64)[None, :]
    selected = model.selected_subspaces.astype(np.int64, copy=False)
    selected_mask = np.zeros(primary.subspaces, dtype=bool)
    selected_mask[selected] = True
    unselected = np.flatnonzero(~selected_mask)

    for local in range(query_count):
        begin, end = int(offsets[local]), int(offsets[local + 1])
        candidate_ids = np.asarray(visited_ids[begin:end], dtype=np.uint32)
        if candidate_ids.size < args.topk:
            raise ValueError("fixed candidate pool contains fewer nodes than topk")
        if int(np.max(candidate_ids)) >= primary.node_count:
            raise ValueError("visited candidate exceeds database range")
        candidate_counts[local] = candidate_ids.size
        candidate_primary = np.asarray(
            primary_codes[candidate_ids], dtype=np.uint8
        )
        records = np.asarray(variable_codes[candidate_ids], dtype=np.uint8)
        variable_low, candidate_9bit = unpack_variable_primary_codes(records, model)
        query = np.asarray(queries[query_start + local], dtype=np.float32)
        query_t = transform_vectors(query[None, :], primary)[0]
        query_blocks = query_t.reshape(primary.subspaces, primary.dsub)

        adc_lut = np.einsum(
            "md,mkd->mk", query_blocks, primary.centroids, optimize=True
        ).astype(np.float32)
        adc_terms = adc_lut[all_subspaces, candidate_primary]
        adc_scores = np.sum(adc_terms, axis=1, dtype=np.float64).astype(np.float32)

        variable_scores = np.zeros(candidate_ids.size, dtype=np.float32)
        if unselected.size:
            unselected_centroids = (
                model.centroids_8bit
                if model.has_embedded_8bit_codebooks
                else primary.centroids
            )
            variable_unselected_lut = np.einsum(
                "md,mkd->mk",
                query_blocks[unselected],
                unselected_centroids[unselected],
                optimize=True,
            ).astype(np.float32)
            variable_scores += np.sum(
                variable_unselected_lut[
                    np.arange(unselected.size, dtype=np.int64)[None, :],
                    variable_low[:, unselected],
                ],
                axis=1,
                dtype=np.float64,
            ).astype(np.float32)
        for local_selected, subspace in enumerate(selected):
            lut = (
                model.centroids_9bit[subspace] @ query_blocks[subspace]
            ).astype(np.float32)
            variable_scores += lut[candidate_9bit[:, local_selected]]

        candidate_base = np.asarray(base[candidate_ids], dtype=np.float32)
        exact_query, exact_bias = exact_linear_score_parameters(query_t, primary)
        exact_scores = candidate_base @ exact_query + exact_bias
        scores_by_mode = {
            "adc": adc_scores,
            "variable_primary_pq": variable_scores,
            "exact": exact_scores,
        }
        for name, scores in scores_by_mode.items():
            order = np.lexsort(
                (candidate_ids, -np.asarray(scores, dtype=np.float64))
            )[: args.topk]
            results[name][local] = candidate_ids[order]
            result_scores[name][local] = np.asarray(scores, dtype=np.float32)[order]
        if args.log_interval > 0 and (
            (local + 1) % args.log_interval == 0 or local + 1 == query_count
        ):
            print(
                f"[variable-primary-fixed-audit] {local + 1}/{query_count}",
                flush=True,
            )

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in MODE_NAMES:
        write_id_matrix(str(output / f"{name}.ids.ibin"), results[name])
        write_float_matrix(
            str(output / f"{name}.scores.fbin"), result_scores[name]
        )
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
    dynamic_overlap = overlap_with_reference(
        results["adc"], adc_dynamic, args.topk
    )
    base_codebook_bytes = (
        primary.subspaces * 256 * primary.dsub * np.dtype(np.float32).itemsize
    )
    effective_codebook_bytes = (
        unselected.size * 256 * primary.dsub * np.dtype(np.float32).itemsize
        + selected.size * 512 * primary.dsub * np.dtype(np.float32).itemsize
    )
    summary = {
        "version": 1,
        "method": "fixed_candidate_exact_budget_variable_bit_primary_pq_audit",
        "query_start": query_start,
        "queries": query_count,
        "topk": args.topk,
        "primary_opq_frozen": True,
        "selection_policy": model.selection_policy,
        "training_objective": model.training_objective,
        "assignment_metric": model.assignment_metric,
        "metric_lambda": model.metric_lambda,
        "metric_artifact_id": model.metric_artifact_id,
        "uses_euclidean_assignment": model.uses_euclidean_assignment,
        "selected_subspaces": selected.astype(int).tolist(),
        "selected_subspace_count": model.refined_subspaces,
        "bits_per_selected_subspace": 9,
        "bits_per_unselected_subspace": 8,
        "code_bits_per_node": model.code_bits_per_node,
        "code_bytes_per_node": variable_width,
        "effective_codebook_bytes": int(effective_codebook_bytes),
        "extra_codebook_bytes_vs_800bit_primary": int(
            effective_codebook_bytes - base_codebook_bytes
        ),
        "candidate_pool": {
            "mean": float(np.mean(candidate_counts)),
            "min": int(np.min(candidate_counts)),
            "max": int(np.max(candidate_counts)),
            "total": int(np.sum(candidate_counts)),
        },
        "adc_replay": {
            "exact_order_match_rate": float(
                np.mean(np.all(results["adc"] == adc_dynamic, axis=1))
            ),
            "mean_topk_overlap": float(np.mean(dynamic_overlap)),
        },
        "modes": metrics,
        "definitions": {
            "variable_primary_pq": (
                "the same frozen OPQ transform, 512 primary centroids in the "
                f"{model.refined_subspaces} database-MSE-utility-selected "
                "subspaces and 256 primary centroids in the remaining subspaces"
            ),
            "exact": "full-precision score on the same fixed ADC pool",
        },
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "primary_codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "variable_model": os.path.abspath(args.variable_model),
            "variable_codes": os.path.abspath(args.variable_codes),
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
