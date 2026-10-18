#!/usr/bin/env python3
"""Encode JI residual side codes from an external primary reconstruction."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap
from group_residual_vq import resolve_device
from joint_innovation_rpq import (
    assign_vq,
    initialize_side_codes,
    pack_codes,
    read_model,
    transform_values,
    vq_backend_info,
)
from primary_quantizer_manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--joint-model", required=True)
    parser.add_argument("--out-side-codes", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--log-interval", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch size must be positive")
    primary = read_manifest(args.primary_manifest)
    model = read_model(args.joint_model)
    if (
        model.artifact_id != primary.artifact_id
        or model.node_count != primary.node_count
        or model.dim != primary.dim
    ):
        raise ValueError("primary manifest and JI model are incompatible")
    base = fbin_memmap(primary.base_fbin)
    reconstruction = fbin_memmap(primary.reconstruction_fbin)
    device = resolve_device(args.solver_device)
    side = initialize_side_codes(args.out_side_codes, model)
    counts = [np.zeros(1 << int(bits), dtype=np.int64) for bits in model.bits]
    joint_distortion = 0.0
    residual_squared_error = 0.0
    for start in range(0, primary.node_count, args.batch_size):
        end = min(primary.node_count, start + args.batch_size)
        residual = (
            np.asarray(base[start:end], dtype=np.float32)
            - np.asarray(reconstruction[start:end], dtype=np.float32)
        )
        innovation = transform_values(
            residual, model.residual_mean, model.forward_transform
        )
        decoded_innovation = np.zeros_like(innovation)
        labels = []
        for group in range(model.groups):
            indices = model.indices(group)
            assigned, _ = assign_vq(
                innovation[:, indices],
                model.codebook(group),
                batch_size=args.batch_size,
                device=device,
            )
            labels.append(assigned)
            decoded_innovation[:, indices] = model.codebook(group)[assigned]
            counts[group] += np.bincount(
                assigned.astype(np.int64), minlength=counts[group].size
            )
        difference = innovation.astype(np.float64) - decoded_innovation.astype(
            np.float64
        )
        joint_distortion += float(np.sum(difference * difference))
        decoded_residual = (
            decoded_innovation @ model.inverse_transform + model.residual_mean
        )
        residual_difference = residual.astype(np.float64) - decoded_residual.astype(
            np.float64
        )
        residual_squared_error += float(
            np.sum(residual_difference * residual_difference)
        )
        side[start:end] = pack_codes(labels, model.bits)
        if args.log_interval > 0 and (
            end % args.log_interval < args.batch_size or end == primary.node_count
        ):
            print(f"[primary-agnostic-ji-encode] {end}/{primary.node_count}", flush=True)
    side.flush()
    del side
    rows = max(primary.node_count, 1)
    summary = {
        "version": 1,
        "method": "primary_agnostic_joint_innovation_encoding",
        "primary": primary.name,
        "variant": model.variant,
        "assignment_backend": vq_backend_info(device),
        "nodes": primary.node_count,
        "primary_nominal_bits_per_node": primary.nominal_code_bits_per_node,
        "primary_physical_bytes_per_node": primary.physical_bytes_per_node,
        "side_bits_per_node": model.budget_bits,
        "total_nominal_bits_per_node": (
            primary.nominal_code_bits_per_node + model.budget_bits
        ),
        "joint_metric_mse": joint_distortion / rows,
        "residual_euclidean_mse": residual_squared_error / rows,
        "empty_database_codes": int(sum(np.sum(value == 0) for value in counts)),
        "files": {
            "primary_manifest": os.path.abspath(args.primary_manifest),
            "joint_model": os.path.abspath(args.joint_model),
            "side_codes": os.path.abspath(args.out_side_codes),
        },
        "config": vars(args),
    }
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote primary-agnostic JI side codes: {args.out_side_codes}")


if __name__ == "__main__":
    main()
