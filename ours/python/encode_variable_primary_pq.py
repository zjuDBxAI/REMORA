#!/usr/bin/env python3
"""Encode the full database with packed 8/9-bit variable primary-PQ codes."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import resolve_device
from variable_primary_pq import nearest_9bit_codes, nearest_metric_codes
from variable_primary_pq_io import (
    initialize_variable_primary_codes,
    pack_ninth_bits,
    read_variable_primary_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--variable-model", required=True)
    parser.add_argument("--out-codes", required=True)
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--subspace-block-size", type=int, default=8)
    parser.add_argument("--log-interval", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0 or args.subspace_block_size <= 0:
        raise ValueError("batch sizes must be positive")
    device = resolve_device(args.solver_device)
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
    base = fbin_memmap(args.base_fbin, primary.node_count)
    if primary_codes.shape != (primary.node_count, primary.subspaces):
        raise ValueError("primary code shape mismatch")
    if base.shape != (primary.node_count, primary.dim):
        raise ValueError("base vector shape mismatch")
    output_codes = initialize_variable_primary_codes(args.out_codes, model)
    selected = model.selected_subspaces.astype(np.int64, copy=False)
    selected_mask = np.zeros(primary.subspaces, dtype=bool)
    selected_mask[selected] = True
    unselected = np.flatnonzero(~selected_mask)
    try:
        for start in range(0, model.node_count, args.batch_size):
            end = min(model.node_count, start + args.batch_size)
            rows = end - start
            transformed = transform_vectors(
                np.asarray(base[start:end], dtype=np.float32), primary
            ).reshape(rows, primary.subspaces, primary.dsub)
            codes_9bit = np.empty(
                (rows, model.refined_subspaces), dtype=np.uint16
            )
            for local_begin in range(
                0, model.refined_subspaces, args.subspace_block_size
            ):
                local_end = min(
                    model.refined_subspaces,
                    local_begin + args.subspace_block_size,
                )
                block_subspaces = selected[local_begin:local_end]
                if model.uses_euclidean_assignment:
                    assigned = nearest_9bit_codes(
                        transformed[:, block_subspaces],
                        model.centroids_9bit[block_subspaces],
                        device=device,
                        batch_size=args.batch_size,
                    )
                else:
                    assigned = nearest_metric_codes(
                        transformed[:, block_subspaces],
                        model.centroids_9bit[block_subspaces],
                        model.metric_factors[block_subspaces],
                        device=device,
                        batch_size=args.batch_size,
                    )
                codes_9bit[:, local_begin:local_end] = assigned
            records = np.empty(
                (rows, model.code_bytes_per_node), dtype=np.uint8
            )
            records[:, : primary.subspaces] = np.asarray(
                primary_codes[start:end], dtype=np.uint8
            )
            if not model.uses_euclidean_assignment:
                if not model.has_embedded_8bit_codebooks:
                    raise ValueError(
                        "metric model does not contain its 8-bit codebooks"
                    )
                for local_begin in range(
                    0, unselected.size, args.subspace_block_size
                ):
                    local_end = min(
                        unselected.size,
                        local_begin + args.subspace_block_size,
                    )
                    block_subspaces = unselected[local_begin:local_end]
                    assigned8 = nearest_metric_codes(
                        transformed[:, block_subspaces],
                        model.centroids_8bit[block_subspaces],
                        model.metric_factors[block_subspaces],
                        device=device,
                        batch_size=args.batch_size,
                    )
                    records[:, block_subspaces] = assigned8.astype(np.uint8)
            records[:, selected] = (codes_9bit & 255).astype(np.uint8)
            records[:, primary.subspaces :] = pack_ninth_bits(codes_9bit)
            output_codes[start:end] = records
            if args.log_interval > 0 and (
                end % args.log_interval < args.batch_size
                or end == model.node_count
            ):
                print(f"[variable-primary-encode] {end}/{model.node_count}", flush=True)
        output_codes.flush()
    finally:
        del output_codes

    summary_path = (
        Path(args.out_summary)
        if args.out_summary
        else Path(args.out_codes).with_suffix(".summary.json")
    )
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary = {
        "version": 1,
        "method": "full_database_exact_budget_variable_primary_pq_encoding",
        "nodes": model.node_count,
        "subspaces": model.subspaces,
        "selected_subspaces": selected.astype(int).tolist(),
        "selected_subspace_count": model.refined_subspaces,
        "assignment_metric": model.assignment_metric,
        "metric_lambda": model.metric_lambda,
        "metric_artifact_id": model.metric_artifact_id,
        "code_bits_per_node": model.code_bits_per_node,
        "code_bytes_per_node": model.code_bytes_per_node,
        "layout": (
            f"{model.subspaces} low-code bytes followed by "
            f"{model.high_bit_bytes_per_node} packed ninth-bit bytes"
        ),
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "primary_codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "variable_model": os.path.abspath(args.variable_model),
            "variable_codes": os.path.abspath(args.out_codes),
        },
        "config": vars(args),
    }
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote variable-primary-PQ codes: {args.out_codes}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
