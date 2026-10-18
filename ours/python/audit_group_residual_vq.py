#!/usr/bin/env python3
"""Audit selected-subspace residual VQ on one fixed ADC candidate pool."""

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
    selected_linear_score_parameters,
    write_float_matrix,
    write_id_matrix,
)
from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq_io import (
    GroupResidualVQModel,
    read_group_model,
    read_group_side_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--group-model", required=True)
    parser.add_argument("--side-codes", required=True)
    parser.add_argument("--visited-csr", required=True)
    parser.add_argument("--adc-result-ids", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--log-interval", type=int, default=100)
    return parser.parse_args()


def validate_compatibility(group_model, pq_model) -> None:
    if (
        group_model.artifact_id != pq_model.artifact_id
        or group_model.node_count != pq_model.node_count
        or group_model.dim != pq_model.dim
        or group_model.subspaces != pq_model.subspaces
        or group_model.dsub != pq_model.dsub
    ):
        raise ValueError("group residual-VQ model does not match the PQ model")


def group_correction(
    side_codes: np.ndarray,
    query_groups: np.ndarray,
    model: GroupResidualVQModel,
) -> np.ndarray:
    correction = np.zeros(side_codes.shape[0], dtype=np.float32)
    for group in range(model.group_count):
        lut = model.prototypes[group] @ query_groups[group]
        correction += lut[side_codes[:, group]]
    return correction


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
        raise ValueError("--pq-model must be the frozen R=0 model")
    mse_model = read_group_model(args.group_model)
    validate_compatibility(mse_model, pq_model)
    side_codes, side_width = read_group_side_codes(args.side_codes, mse_model)
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
    query_count = available if args.query_limit <= 0 else min(available, args.query_limit)
    if query_count <= 0 or query_start + query_count > queries.shape[0]:
        raise ValueError("invalid fixed-candidate query range")
    adc_dynamic = read_id_matrix(args.adc_result_ids)
    if adc_dynamic.shape[0] < query_count or adc_dynamic.shape[1] < args.topk:
        raise ValueError("ADC results do not cover the group audit")
    adc_dynamic = np.asarray(adc_dynamic[:query_count, : args.topk], dtype=np.uint32)
    truth = read_id_matrix(
        args.ground_truth,
        allow_trailing=True,
        row_start=query_start,
        row_count=query_count,
    )[:, : args.topk]

    quantized_mode = (
        "group_vq_mse"
        if mse_model.uses_euclidean_assignment
        else "group_vq_query_aware"
    )
    modes = [
        "adc",
        quantized_mode,
        "continuous_selected_residual",
        "exact_residual",
    ]
    results: Dict[str, np.ndarray] = {
        name: np.empty((query_count, args.topk), dtype=np.uint32)
        for name in modes
    }
    result_scores: Dict[str, np.ndarray] = {
        name: np.empty((query_count, args.topk), dtype=np.float32)
        for name in modes
    }
    candidate_counts = np.empty(query_count, dtype=np.uint32)
    all_subspaces = np.arange(pq_model.subspaces, dtype=np.int64)[None, :]
    selected = mse_model.selected_subspaces.astype(np.int64)

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
        query_t = transform_vectors(query[None, :], pq_model)[0]
        query_blocks = query_t.reshape(pq_model.subspaces, pq_model.dsub)
        query_groups = np.ascontiguousarray(
            query_blocks[mse_model.groups.astype(np.int64)].reshape(
                mse_model.group_count, mse_model.group_dim
            ),
            dtype=np.float32,
        )
        adc_lut = np.einsum(
            "md,mkd->mk", query_blocks, pq_model.centroids, optimize=True
        ).astype(np.float32)
        adc_terms = adc_lut[all_subspaces, candidate_codes]
        adc_scores = np.sum(adc_terms, axis=1, dtype=np.float64).astype(np.float32)
        mode_scores: Dict[str, np.ndarray] = {
            "adc": adc_scores,
            quantized_mode: adc_scores
            + group_correction(candidate_side, query_groups, mse_model),
        }
        candidate_base = np.asarray(base[candidate_ids], dtype=np.float32)
        selected_query, selected_bias = selected_linear_score_parameters(
            query_t, selected, pq_model
        )
        selected_exact = candidate_base @ selected_query + selected_bias
        selected_adc = np.sum(adc_terms[:, selected], axis=1, dtype=np.float64)
        mode_scores["continuous_selected_residual"] = (
            adc_scores + selected_exact - selected_adc
        )
        exact_query, exact_bias = exact_linear_score_parameters(query_t, pq_model)
        mode_scores["exact_residual"] = candidate_base @ exact_query + exact_bias

        for name, scores in mode_scores.items():
            order = np.lexsort((candidate_ids, -np.asarray(scores, dtype=np.float64)))[
                : args.topk
            ]
            results[name][local] = candidate_ids[order]
            result_scores[name][local] = np.asarray(scores, dtype=np.float32)[order]
        if args.log_interval > 0 and (
            (local + 1) % args.log_interval == 0 or local + 1 == query_count
        ):
            print(f"[group-fixed-audit] {local + 1}/{query_count}", flush=True)

    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for name in modes:
        write_id_matrix(str(output / f"{name}.ids.ibin"), results[name])
        write_float_matrix(
            str(output / f"{name}.scores.fbin"), result_scores[name]
        )
    recalls = {name: row_recall(results[name], truth) for name in modes}
    adc_recall = recalls["adc"]
    metrics: Dict[str, object] = {}
    for name in modes:
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
    selected_gain = float(
        metrics["continuous_selected_residual"]["recall_delta_vs_adc"]
    )
    decomposition: Dict[str, object] = {
        "selected_continuous_coverage_of_exact_gain": safe_ratio(
            selected_gain, exact_gain
        )
    }
    decomposition[f"{quantized_mode}_retention_of_selected_continuous_gain"] = (
        safe_ratio(float(metrics[quantized_mode]["recall_delta_vs_adc"]), selected_gain)
    )
    dynamic_overlap = overlap_with_reference(results["adc"], adc_dynamic, args.topk)
    summary = {
        "version": 1,
        "method": "fixed_candidate_group_residual_vq_audit",
        "query_start": query_start,
        "queries": query_count,
        "topk": args.topk,
        "group_count": mse_model.group_count,
        "subspaces_per_group": mse_model.subspaces_per_group,
        "group_dim": mse_model.group_dim,
        "codewords_per_group": mse_model.codewords,
        "side_bits_per_node": mse_model.side_bits_per_node,
        "side_bytes_per_node": side_width,
        "assignment_metric": mse_model.assignment_metric,
        "metric_lambda": mse_model.metric_lambda,
        "metric_artifact_id": mse_model.metric_artifact_id,
        "uses_euclidean_assignment": mse_model.uses_euclidean_assignment,
        "groups": mse_model.groups.astype(int).tolist(),
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
        "training_objectives": {quantized_mode: mse_model.training_objective},
        "modes": metrics,
        "decomposition": decomposition,
        "definitions": {
            quantized_mode: (
                "ADC plus selected residual prototypes trained and assigned "
                "with an OOD-query metric"
                if quantized_mode == "group_vq_query_aware"
                else "ADC plus database-MSE group residual prototypes"
            ),
            "continuous_selected_residual": (
                "ADC plus exact residual in every group-selected subspace"
            ),
            "exact_residual": "full-precision score on the fixed ADC candidate pool",
        },
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "group_model": os.path.abspath(args.group_model),
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
