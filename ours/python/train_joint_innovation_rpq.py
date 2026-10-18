#!/usr/bin/env python3
"""Train Joint-Innovation residual quantizers and controlled ablations.

One invocation produces four artifacts:

* ``equal-rate``: 8 equal innovation groups with 8 bits each by default;
* ``atomic-equal-rate``: the finer atomic partition with a uniform rate;
* ``rate-adaptive``: the same atomic partition with exact DP allocation;
* ``joint-grouping``: validation-guided merges plus exact DP allocation.

The extra atomic-equal artifact makes the rate-allocation comparison causal:
``atomic-equal-rate`` and ``rate-adaptive`` differ only in their bit allocation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from drb_io import read_model as read_primary_model
from group_residual_vq import resolve_device
from joint_innovation_rpq import (
    assemble_model,
    build_joint_transform,
    fit_vq,
    make_equal_groups,
    optimal_bit_allocation,
    transform_values,
    write_model,
)
from primary_quantizer_manifest import read_manifest
from residual_metric_io import read_residual_metric


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    primary = parser.add_mutually_exclusive_group(required=True)
    primary.add_argument("--pq-model")
    primary.add_argument("--primary-manifest")
    parser.add_argument("--metric-model", required=True)
    parser.add_argument("--training-cache", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--budget-bits", type=int, default=64)
    parser.add_argument("--equal-groups", type=int, default=8)
    parser.add_argument("--atomic-groups", type=int, default=16)
    parser.add_argument("--min-bits", type=int, default=2)
    parser.add_argument("--max-bits", type=int, default=6)
    parser.add_argument("--merge-max-bits", type=int, default=8)
    parser.add_argument("--merge-rounds", type=int, default=4)
    parser.add_argument("--merge-candidates", type=int, default=4)
    parser.add_argument("--fit-sample-size", type=int, default=250000)
    parser.add_argument("--holdout-ratio", type=float, default=0.2)
    parser.add_argument("--iterations", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--eigenvalue-floor-relative", type=float, default=1e-6)
    parser.add_argument(
        "--equal-rate-only",
        action="store_true",
        help="train only the equal-rate variant for a matched control",
    )
    return parser.parse_args()


def json_curve(curve: dict[int, dict]) -> dict[str, dict]:
    return {
        str(bits): dict(record["diagnostics"])
        for bits, record in sorted(curve.items())
    }


def partition_key(groups: list[np.ndarray]) -> list[list[int]]:
    return [np.asarray(group, dtype=np.int64).tolist() for group in groups]


def main() -> None:
    args = parse_args()
    if (
        args.budget_bits <= 0
        or args.budget_bits > 128
        or args.equal_groups <= 0
        or args.atomic_groups <= 0
        or args.min_bits < 0
        or args.max_bits < args.min_bits
        or args.merge_max_bits < args.max_bits
        or args.merge_max_bits > 16
        or args.merge_rounds < 0
        or args.merge_candidates <= 0
        or args.fit_sample_size <= 0
        or not 0.0 < args.holdout_ratio < 1.0
        or args.iterations <= 0
        or args.batch_size <= 0
    ):
        raise ValueError("invalid Joint-Innovation training configuration")
    primary = (
        read_manifest(args.primary_manifest)
        if args.primary_manifest
        else read_primary_model(args.pq_model)
    )
    metric = read_residual_metric(args.metric_model)
    if hasattr(primary, "refined_subspaces") and primary.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen primary quantizer")
    if (
        metric.artifact_id != primary.artifact_id
        or metric.dim != primary.dim
        or metric.geometry != "full"
    ):
        raise ValueError("Joint-Innovation training requires a compatible full metric")
    if primary.dim % args.equal_groups != 0:
        raise ValueError("dimension must divide evenly into --equal-groups")
    if args.budget_bits % args.equal_groups != 0:
        raise ValueError("budget must divide evenly for the equal-rate ablation")
    if not args.equal_rate_only and args.budget_bits % args.atomic_groups != 0:
        raise ValueError("budget must divide evenly for the atomic-equal control")
    atomic_equal_bits = args.budget_bits // args.atomic_groups
    if (not args.equal_rate_only and
            not args.min_bits <= atomic_equal_bits <= args.max_bits):
        raise ValueError("atomic equal rate is outside [min-bits, max-bits]")
    if not args.equal_rate_only and not (
        args.atomic_groups * args.min_bits
        <= args.budget_bits
        <= args.atomic_groups * args.max_bits
    ):
        raise ValueError("atomic rate allocation is infeasible")

    raw_cache = np.load(args.training_cache, mmap_mode="r")
    if raw_cache.dtype != np.float32:
        raise ValueError("training cache must use float32 values")
    if raw_cache.ndim not in (2, 3):
        raise ValueError("training cache must be [rows, dim] or [rows, groups, group_dim]")
    raw_dim = int(np.prod(raw_cache.shape[1:]))
    if raw_dim != primary.dim:
        raise ValueError("training cache dimension does not match primary model")
    sample_size = min(int(args.fit_sample_size), int(raw_cache.shape[0]))
    rng = np.random.default_rng(args.seed)
    sample_ids = np.sort(rng.choice(raw_cache.shape[0], sample_size, replace=False))
    residual = np.asarray(raw_cache[sample_ids], dtype=np.float32).reshape(
        sample_size, primary.dim
    )
    residual_mean, forward, inverse, spectrum, transform_diagnostics = (
        build_joint_transform(
            residual,
            metric.matrices,
            args.eigenvalue_floor_relative,
        )
    )
    innovation_values = transform_values(residual, residual_mean, forward)
    split_rng = np.random.default_rng(args.seed + 1)
    permutation = split_rng.permutation(sample_size)
    holdout_count = max(1, int(round(sample_size * args.holdout_ratio)))
    holdout_ids = permutation[:holdout_count]
    train_ids = permutation[holdout_count:]
    if train_ids.size < (1 << args.merge_max_bits):
        raise ValueError("training split is smaller than the largest requested codebook")
    train = innovation_values[train_ids]
    holdout = innovation_values[holdout_ids]
    device = resolve_device(args.solver_device)
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)

    curve_cache: dict[tuple[int, ...], dict[int, dict]] = {}

    def ensure_curve(group: np.ndarray, bit_values: list[int]) -> dict[int, dict]:
        key = tuple(sorted(int(value) for value in np.asarray(group).tolist()))
        curve = curve_cache.setdefault(key, {})
        group_indices = np.asarray(key, dtype=np.int64)
        for bits in sorted(set(int(value) for value in bit_values)):
            if bits in curve:
                continue
            centers, counts, diagnostics = fit_vq(
                train[:, group_indices],
                holdout[:, group_indices],
                bits=bits,
                iterations=args.iterations,
                batch_size=args.batch_size,
                device=device,
                seed=args.seed + 1009 * len(key) + 9176 * bits + sum(key),
            )
            curve[bits] = {
                "centers": centers,
                "counts": counts,
                "diagnostics": diagnostics,
            }
            print(
                f"[ji-curve] dims={list(key)} bits={bits} "
                f"validation_mse={diagnostics['validation_mse']:.8g}",
                flush=True,
            )
        return curve

    def curves_for(groups: list[np.ndarray], bit_values: list[int]) -> list[dict[int, dict]]:
        return [ensure_curve(group, bit_values) for group in groups]

    equal_groups = make_equal_groups(primary.dim, args.equal_groups)
    equal_bits = [args.budget_bits // args.equal_groups] * args.equal_groups
    equal_curves = [
        ensure_curve(group, [equal_bits[index]])
        for index, group in enumerate(equal_groups)
    ]
    equal_loss = sum(
        equal_curves[index][equal_bits[index]]["diagnostics"]["validation_mse"]
        for index in range(len(equal_groups))
    )

    if args.equal_rate_only:
        model = assemble_model(
            artifact_id=primary.artifact_id,
            node_count=primary.node_count,
            metric_id=metric.metric_id,
            metric_lambda=metric.metric_lambda,
            residual_mean=residual_mean,
            forward_transform=forward,
            inverse_transform=inverse,
            innovation_eigenvalues=spectrum,
            groups=equal_groups,
            bits=equal_bits,
            curves=equal_curves,
            variant="equal-rate",
        )
        model_path = output / "equal-rate" / "model" / "joint_innovation_model.npz"
        write_model(str(model_path), model)
        summary = {
            "version": 1,
            "method": "joint_innovation_equal_rate_residual_quantization",
            "primary_quantizer_frozen": True,
            "metric": {
                "metric_id": int(metric.metric_id),
                "metric_lambda": float(metric.metric_lambda),
                "metric_form": metric.metric_form,
                "geometry": metric.geometry,
            },
            "primary": {
                "name": getattr(primary, "name", "opq-pq"),
                "artifact_id": int(primary.artifact_id),
                "manifest": (
                    str(Path(args.primary_manifest).resolve())
                    if args.primary_manifest
                    else None
                ),
            },
            "training": {
                "available_cache_rows": int(raw_cache.shape[0]),
                "fit_sample_size": int(sample_size),
                "train_rows": int(train.shape[0]),
                "holdout_rows": int(holdout.shape[0]),
                "seed": int(args.seed),
            },
            "joint_transform": transform_diagnostics,
            "innovation_spectrum": {
                "sum": float(np.sum(spectrum)),
                "top_10": np.asarray(spectrum[:10], dtype=float).tolist(),
                "rank_90_percent": int(
                    np.searchsorted(np.cumsum(spectrum), 0.9 * np.sum(spectrum)) + 1
                ),
                "rank_95_percent": int(
                    np.searchsorted(np.cumsum(spectrum), 0.95 * np.sum(spectrum)) + 1
                ),
            },
            "variants": {
                "equal-rate": {
                    "model": str(model_path.resolve()),
                    "groups": partition_key(equal_groups),
                    "group_dims": [int(len(group)) for group in equal_groups],
                    "bits": [int(value) for value in equal_bits],
                    "side_bits_per_node": int(sum(equal_bits)),
                    "codewords": [1 << int(value) for value in equal_bits],
                    "total_lut_entries": int(
                        sum(1 << int(value) for value in equal_bits)
                    ),
                    "validation_joint_distortion": float(equal_loss),
                }
            },
            "merge_history": [],
            "distortion_curves": {
                ",".join(str(value) for value in key): json_curve(curve)
                for key, curve in curve_cache.items()
            },
            "config": vars(args),
        }
        with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"wrote equal-rate Joint-Innovation control: {output}")
        return

    atomic_groups = make_equal_groups(primary.dim, args.atomic_groups)
    candidate_bits = list(range(args.min_bits, args.max_bits + 1))
    atomic_curves = curves_for(atomic_groups, candidate_bits)
    atomic_equal_allocation = [atomic_equal_bits] * args.atomic_groups
    atomic_equal_loss = sum(
        atomic_curves[index][atomic_equal_bits]["diagnostics"]["validation_mse"]
        for index in range(args.atomic_groups)
    )
    adaptive_bits, adaptive_loss = optimal_bit_allocation(
        atomic_curves, args.budget_bits
    )

    current_groups = [group.copy() for group in atomic_groups]
    current_curves = list(atomic_curves)
    current_bits = adaptive_bits
    current_loss = adaptive_loss
    merge_history: list[dict] = []
    for round_index in range(args.merge_rounds):
        energies = np.stack(
            [
                np.mean(holdout[:, np.asarray(group, dtype=np.int64)] ** 2, axis=1)
                for group in current_groups
            ],
            axis=1,
        )
        correlation = np.nan_to_num(np.corrcoef(energies, rowvar=False), nan=0.0)
        candidates = []
        for left in range(len(current_groups)):
            for right in range(left + 1, len(current_groups)):
                candidates.append((abs(float(correlation[left, right])), left, right))
        candidates.sort(reverse=True)
        best: dict | None = None
        for dependence, left, right in candidates[: args.merge_candidates]:
            merged = np.sort(
                np.concatenate((current_groups[left], current_groups[right]))
            ).astype(np.int32)
            merged_curve = ensure_curve(
                merged, list(range(args.min_bits, args.merge_max_bits + 1))
            )
            proposed_groups = [
                group
                for index, group in enumerate(current_groups)
                if index not in (left, right)
            ] + [merged]
            proposed_curves = [
                curve
                for index, curve in enumerate(current_curves)
                if index not in (left, right)
            ] + [merged_curve]
            try:
                proposed_bits, proposed_loss = optimal_bit_allocation(
                    proposed_curves, args.budget_bits
                )
            except ValueError:
                continue
            record = {
                "left": int(left),
                "right": int(right),
                "dependence_score": float(dependence),
                "groups": proposed_groups,
                "curves": proposed_curves,
                "bits": proposed_bits,
                "loss": float(proposed_loss),
            }
            if best is None or proposed_loss < float(best["loss"]):
                best = record
        accepted = best is not None and float(best["loss"]) < current_loss - 1e-12
        merge_history.append(
            {
                "round": round_index + 1,
                "accepted": bool(accepted),
                "loss_before": float(current_loss),
                "loss_after": float(best["loss"]) if best is not None else None,
                "left": int(best["left"]) if best is not None else None,
                "right": int(best["right"]) if best is not None else None,
                "dependence_score": (
                    float(best["dependence_score"]) if best is not None else None
                ),
            }
        )
        if not accepted:
            break
        current_groups = best["groups"]
        current_curves = best["curves"]
        current_bits = best["bits"]
        current_loss = float(best["loss"])

    variants = [
        ("equal-rate", equal_groups, equal_bits, equal_curves, equal_loss),
        (
            "atomic-equal-rate",
            atomic_groups,
            atomic_equal_allocation,
            atomic_curves,
            atomic_equal_loss,
        ),
        ("rate-adaptive", atomic_groups, adaptive_bits, atomic_curves, adaptive_loss),
        ("joint-grouping", current_groups, current_bits, current_curves, current_loss),
    ]
    variant_summaries = {}
    for name, groups, bits, curves, validation_loss in variants:
        model = assemble_model(
            artifact_id=primary.artifact_id,
            node_count=primary.node_count,
            metric_id=metric.metric_id,
            metric_lambda=metric.metric_lambda,
            residual_mean=residual_mean,
            forward_transform=forward,
            inverse_transform=inverse,
            innovation_eigenvalues=spectrum,
            groups=groups,
            bits=bits,
            curves=curves,
            variant=name,
        )
        model_path = output / name / "model" / "joint_innovation_model.npz"
        write_model(str(model_path), model)
        variant_summaries[name] = {
            "model": str(model_path.resolve()),
            "groups": partition_key(groups),
            "group_dims": [int(len(group)) for group in groups],
            "bits": [int(value) for value in bits],
            "side_bits_per_node": int(sum(bits)),
            "codewords": [1 << int(value) for value in bits],
            "total_lut_entries": int(sum(1 << int(value) for value in bits)),
            "validation_joint_distortion": float(validation_loss),
        }

    summary = {
        "version": 1,
        "method": "joint_innovation_rate_adaptive_residual_quantization",
        "primary_quantizer_frozen": True,
        "metric": {
            "metric_id": int(metric.metric_id),
            "metric_lambda": float(metric.metric_lambda),
            "metric_form": metric.metric_form,
            "geometry": metric.geometry,
        },
        "primary": {
            "name": getattr(primary, "name", "opq-pq"),
            "artifact_id": int(primary.artifact_id),
            "manifest": (
                str(Path(args.primary_manifest).resolve())
                if args.primary_manifest
                else None
            ),
        },
        "training": {
            "available_cache_rows": int(raw_cache.shape[0]),
            "fit_sample_size": int(sample_size),
            "train_rows": int(train.shape[0]),
            "holdout_rows": int(holdout.shape[0]),
            "seed": int(args.seed),
        },
        "joint_transform": transform_diagnostics,
        "innovation_spectrum": {
            "sum": float(np.sum(spectrum)),
            "top_10": np.asarray(spectrum[:10], dtype=float).tolist(),
            "rank_90_percent": int(np.searchsorted(np.cumsum(spectrum), 0.9 * np.sum(spectrum)) + 1),
            "rank_95_percent": int(np.searchsorted(np.cumsum(spectrum), 0.95 * np.sum(spectrum)) + 1),
        },
        "variants": variant_summaries,
        "merge_history": merge_history,
        "distortion_curves": {
            ",".join(str(value) for value in key): json_curve(curve)
            for key, curve in curve_cache.items()
        },
        "config": vars(args),
    }
    with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote Joint-Innovation experiment: {output}")


if __name__ == "__main__":
    main()
