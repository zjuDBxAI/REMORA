#!/usr/bin/env python3
"""Encode all database nodes with the full-dimensional residual-PQ baseline."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from additive_residual_quantizer import (
    assign_additive_batch,
    conditional_prediction,
)
from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import nearest_group_codes, resolve_device
from residual_pq_baseline import (
    decode_coded_residual,
    extract_full_residual,
    is_identity_metric,
    nearest_metric_group_codes,
    transform_grouped_residual,
)
from residual_pq_baseline_io import (
    initialize_residual_pq_side_codes,
    read_residual_pq_model,
    read_residual_pq_side_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--out-side-codes", required=True)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument("--additive-coordinate-sweeps", type=int, default=1)
    parser.add_argument("--additive-initial-model", default="")
    parser.add_argument("--additive-initial-side-codes", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.additive_coordinate_sweeps < 0
        or bool(args.additive_initial_model)
        != bool(args.additive_initial_side_codes)
    ):
        raise ValueError("batch-size must be positive")
    device = resolve_device(args.solver_device)
    pq_model = read_model(args.pq_model)
    residual_model = read_residual_pq_model(args.residual_model)
    if (
        pq_model.refined_subspaces != 0
        or residual_model.artifact_id != pq_model.artifact_id
        or residual_model.node_count != pq_model.node_count
        or residual_model.dim != pq_model.dim
    ):
        raise ValueError("primary PQ and residual-PQ models are incompatible")
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("PQ codes do not match the model")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the model")
    initial_side_codes = None
    if args.additive_initial_model:
        initial_model = read_residual_pq_model(args.additive_initial_model)
        if (
            residual_model.quantizer_kind != "additive"
            or initial_model.quantizer_kind != "additive"
            or initial_model.has_primary_conditioning
            or initial_model.artifact_id != residual_model.artifact_id
            or initial_model.groups != residual_model.groups
            or initial_model.codewords != residual_model.codewords
            or not np.array_equal(
                initial_model.prototypes, residual_model.prototypes
            )
            or not np.array_equal(
                initial_model.residual_transform,
                residual_model.residual_transform,
            )
        ):
            raise ValueError("additive initialization model is incompatible")
        initial_side_codes, _ = read_residual_pq_side_codes(
            args.additive_initial_side_codes, initial_model
        )

    side_codes = initialize_residual_pq_side_codes(
        args.out_side_codes, residual_model
    )
    counts = np.zeros(
        (residual_model.groups, residual_model.codewords), dtype=np.int64
    )
    squared_error = np.zeros(residual_model.groups, dtype=np.float64)
    coding_squared_error = 0.0
    for start in range(0, pq_model.node_count, args.batch_size):
        end = min(pq_model.node_count, start + args.batch_size)
        batch_codes = np.asarray(codes[start:end], dtype=np.uint8)
        vectors_t = transform_vectors(
            np.asarray(base[start:end], dtype=np.float32), pq_model
        )
        original_residual = extract_full_residual(
            vectors_t, batch_codes, pq_model, residual_model.groups
        )
        residual = (
            transform_grouped_residual(
                original_residual, residual_model.residual_transform
            )
            if residual_model.has_residual_transform
            else original_residual
        )
        prediction = (
            conditional_prediction(
                batch_codes,
                residual_model.conditional_basis,
                residual_model.conditional_embeddings,
            )
            if residual_model.has_primary_conditioning
            else np.zeros((end - start, residual_model.dim), dtype=np.float32)
        )
        if residual_model.quantizer_kind == "additive":
            innovation = residual.reshape(end - start, residual_model.dim) - prediction
            assigned, side_reconstruction, _ = assign_additive_batch(
                innovation,
                residual_model.prototypes,
                device=device,
                coordinate_sweeps=args.additive_coordinate_sweeps,
                initial_codes=(
                    None
                    if initial_side_codes is None
                    else np.asarray(initial_side_codes[start:end], dtype=np.uint8)
                ),
            )
        elif is_identity_metric(residual_model.metric_matrices):
            assigned = nearest_group_codes(
                residual,
                residual_model.prototypes,
                device=device,
                batch_size=args.batch_size,
            )
            dictionaries = np.arange(residual_model.groups, dtype=np.int64)[None, :]
            side_reconstruction = residual_model.prototypes[
                dictionaries, assigned
            ].reshape(end - start, residual_model.dim)
        else:
            assigned = nearest_metric_group_codes(
                residual,
                residual_model.prototypes,
                residual_model.metric_factors,
                device=device,
                batch_size=args.batch_size,
            )
            dictionaries = np.arange(residual_model.groups, dtype=np.int64)[None, :]
            side_reconstruction = residual_model.prototypes[
                dictionaries, assigned
            ].reshape(end - start, residual_model.dim)
        side_codes[start:end] = assigned
        for group in range(residual_model.groups):
            counts[group] += np.bincount(
                assigned[:, group].astype(np.int64),
                minlength=residual_model.codewords,
            )
        coded_reconstruction = side_reconstruction + prediction
        coding_difference = (
            residual.reshape(end - start, residual_model.dim)
            - coded_reconstruction
        )
        coding_squared_error += float(
            np.sum(coding_difference.astype(np.float64) ** 2)
        )
        decoded = decode_coded_residual(coded_reconstruction, residual_model)
        original_difference = (
            original_residual.reshape(end - start, residual_model.dim) - decoded
        ).reshape(end - start, residual_model.groups, residual_model.group_dim)
        squared_error += np.sum(
            original_difference.astype(np.float64) ** 2, axis=(0, 2)
        )
        if args.log_interval > 0 and (
            end % args.log_interval < args.batch_size or end == pq_model.node_count
        ):
            print(f"[residual-pq-side-codes] {end}/{pq_model.node_count}", flush=True)
    side_codes.flush()
    del side_codes

    summary_path = (
        Path(args.out_summary)
        if args.out_summary
        else Path(args.out_side_codes).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "method": "full_database_residual_pq_encoding",
        "assignment_metric": residual_model.assignment_metric,
        "quantizer_kind": residual_model.quantizer_kind,
        "primary_conditioned": residual_model.has_primary_conditioning,
        "conditional_rank": residual_model.conditional_rank,
        "residual_transform_kind": residual_model.transform_kind,
        "has_residual_transform": residual_model.has_residual_transform,
        "metric_lambda": residual_model.metric_lambda,
        "metric_artifact_id": residual_model.metric_artifact_id,
        "assignment_kernel": (
            "additive_greedy_plus_coordinate_descent"
            if residual_model.quantizer_kind == "additive"
            else "euclidean_exact"
            if is_identity_metric(residual_model.metric_matrices)
            else "mahalanobis"
        ),
        "nodes": pq_model.node_count,
        "groups": residual_model.groups,
        "group_dim": residual_model.group_dim,
        "prototype_dim": residual_model.prototype_dim,
        "covered_residual_dimensions": residual_model.dim,
        "codewords_per_group": residual_model.codewords,
        "side_bits_per_node": residual_model.side_bits_per_node,
        "side_bytes_per_node": residual_model.side_bytes_per_node,
        "empty_database_codes": int(np.sum(counts == 0)),
        "observations_per_group": np.sum(counts, axis=1).astype(int).tolist(),
        "residual_mse_per_group": (
            squared_error / max(pq_model.node_count, 1)
        ).tolist(),
        "residual_mse_total": float(
            np.sum(squared_error) / max(pq_model.node_count, 1)
        ),
        "coding_coordinate_mse_total": float(
            coding_squared_error / max(pq_model.node_count, 1)
        ),
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "residual_model": os.path.abspath(args.residual_model),
            "side_codes": os.path.abspath(args.out_side_codes),
        },
        "config": vars(args),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote residual-PQ side codes: {args.out_side_codes}")
    print(f"wrote encoding summary: {summary_path}")


if __name__ == "__main__":
    main()
