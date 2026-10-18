#!/usr/bin/env python3
"""Encode every database node with fixed-width group residual-VQ side codes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import (
    extract_group_residual,
    nearest_group_codes,
    resolve_device,
)
from group_residual_vq_io import (
    initialize_group_side_codes,
    read_group_model,
)
from residual_pq_baseline import (
    is_identity_metric,
    nearest_metric_group_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--group-model", required=True)
    parser.add_argument("--out-side-codes", required=True)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--log-interval", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch-size must be positive")
    device = resolve_device(args.solver_device)
    pq_model = read_model(args.pq_model)
    group_model = read_group_model(args.group_model)
    if (
        pq_model.refined_subspaces != 0
        or group_model.artifact_id != pq_model.artifact_id
        or group_model.node_count != pq_model.node_count
        or group_model.dim != pq_model.dim
        or group_model.subspaces != pq_model.subspaces
        or group_model.dsub != pq_model.dsub
    ):
        raise ValueError("PQ and group residual-VQ models are incompatible")
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("PQ codes do not match the model")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the model")

    side_codes = initialize_group_side_codes(args.out_side_codes, group_model)
    counts = np.zeros(
        (group_model.group_count, group_model.codewords), dtype=np.int64
    )
    squared_error = np.zeros(group_model.group_count, dtype=np.float64)
    for start in range(0, pq_model.node_count, args.batch_size):
        end = min(pq_model.node_count, start + args.batch_size)
        batch_codes = np.asarray(codes[start:end], dtype=np.uint8)
        vectors_t = transform_vectors(
            np.asarray(base[start:end], dtype=np.float32), pq_model
        )
        residual = extract_group_residual(
            vectors_t, batch_codes, pq_model, group_model.groups
        )
        if is_identity_metric(group_model.resolved_metric_matrices):
            assigned = nearest_group_codes(
                residual,
                group_model.prototypes,
                device=device,
                batch_size=args.batch_size,
            )
        else:
            assigned = nearest_metric_group_codes(
                residual,
                group_model.prototypes,
                group_model.resolved_metric_factors,
                device=device,
                batch_size=args.batch_size,
            )
        side_codes[start:end] = assigned
        for group in range(group_model.group_count):
            counts[group] += np.bincount(
                assigned[:, group].astype(np.int64),
                minlength=group_model.codewords,
            )
            difference = (
                residual[:, group]
                - group_model.prototypes[group, assigned[:, group]]
            )
            squared_error[group] += float(
                np.sum(difference.astype(np.float64) ** 2)
            )
        if args.log_interval > 0 and (
            end % args.log_interval < args.batch_size or end == pq_model.node_count
        ):
            print(f"[group-side-codes] {end}/{pq_model.node_count}", flush=True)
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
        "method": "full_database_group_residual_vq_encoding",
        "nodes": pq_model.node_count,
        "group_count": group_model.group_count,
        "codewords_per_group": group_model.codewords,
        "side_bits_per_node": group_model.side_bits_per_node,
        "side_bytes_per_node": group_model.side_bytes_per_node,
        "assignment_metric": group_model.assignment_metric,
        "metric_lambda": group_model.metric_lambda,
        "metric_artifact_id": group_model.metric_artifact_id,
        "assignment_kernel": (
            "euclidean_exact"
            if group_model.uses_euclidean_assignment
            else "mahalanobis"
        ),
        "empty_database_codes": int(np.sum(counts == 0)),
        "observations_per_group": np.sum(counts, axis=1).astype(int).tolist(),
        "residual_mse_per_group": (
            squared_error / max(pq_model.node_count, 1)
        ).tolist(),
        "residual_mse_total": float(
            np.sum(squared_error) / max(pq_model.node_count, 1)
        ),
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "group_model": os.path.abspath(args.group_model),
            "side_codes": os.path.abspath(args.out_side_codes),
        },
        "config": vars(args),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote group side codes: {args.out_side_codes}")
    print(f"wrote encoding summary: {summary_path}")


if __name__ == "__main__":
    main()
