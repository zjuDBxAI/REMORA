#!/usr/bin/env python3
"""Train an equal-budget PQ codebook over the complete primary-PQ residual."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import lloyd_group_kmeans, resolve_device
from residual_metric_io import read_residual_metric
from residual_pq_baseline import (
    extract_full_residual,
    is_identity_metric,
    lloyd_metric_group_kmeans,
    lloyd_tail_aware_metric_group_kmeans,
)
from residual_opq import (
    covariance_balanced_rotation,
    materialize_residual_transform,
    random_orthogonal_rotation,
    rotate_transformed_cache_in_place,
    update_opq_rotation,
)
from residual_pq_baseline_io import (
    ResidualPQModel,
    compute_encoding_id,
    read_residual_pq_model,
    write_residual_pq_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--metric-model",
        default="",
        help="optional residual metric artifact; omitted means Euclidean assignment",
    )
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--codewords", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=1000000)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--convergence-tolerance", type=float, default=1e-5)
    parser.add_argument(
        "--tail-objective",
        choices=("mean", "cvar"),
        default="mean",
        help="mean keeps ordinary Lloyd; cvar reweights high-distortion assignments",
    )
    parser.add_argument(
        "--tail-fraction",
        type=float,
        default=0.05,
        help="fraction of largest assigned distances emphasized by --tail-objective cvar",
    )
    parser.add_argument(
        "--tail-cvar-weight",
        type=float,
        default=0.25,
        help="weight beta in mean_metric_mse + beta * tail_mean_metric_mse",
    )
    parser.add_argument(
        "--initial-residual-model",
        default="",
        help="optional residual-PQ model whose prototypes initialize training",
    )
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument(
        "--initialization",
        choices=("sample", "kmeans++"),
        default="sample",
        help=(
            "centroid initialization for metric Lloyd; kmeans++ runs D^2 "
            "sampling in the Mahalanobis-transformed residual space"
        ),
    )
    parser.add_argument(
        "--residual-rotation",
        choices=("none", "opq"),
        default="none",
        help=(
            "opq alternates a metric-preserving global orthogonal residual "
            "rotation with product-codebook Lloyd updates; requires a full metric"
        ),
    )
    parser.add_argument("--rotation-iterations", type=int, default=5)
    parser.add_argument("--rotation-lloyd-iterations", type=int, default=3)
    parser.add_argument(
        "--rotation-initialization",
        choices=("identity", "covariance-balanced", "multistart"),
        default="identity",
        help=(
            "initial OPQ coordinate system; multistart evaluates identity, "
            "covariance-balanced, and --rotation-random-restarts random starts"
        ),
    )
    parser.add_argument(
        "--rotation-random-restarts",
        type=int,
        default=3,
        help="number of random orthogonal starts included by multistart",
    )
    parser.add_argument(
        "--rotation-restart-initialization",
        choices=("sample", "kmeans++"),
        default="sample",
        help="PQ centroid initialization for non-identity OPQ starts",
    )
    parser.add_argument(
        "--rotation-checkpoint",
        choices=("final", "best"),
        default="final",
        help=(
            "final selects the last alternating step from each start; best may "
            "select any non-worsening training-distortion checkpoint"
        ),
    )
    parser.add_argument("--log-interval", type=int, default=100000)
    parser.add_argument("--kmeans-log-interval", type=int, default=1)
    parser.add_argument(
        "--training-cache",
        default="",
        help="optional .npy residual cache to keep; temporary storage is default",
    )
    parser.add_argument(
        "--reuse-training-cache",
        action="store_true",
        help="reuse an existing --training-cache after validating its shape",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.groups <= 0
        or args.codewords != 256
        or args.sample_size < args.codewords
        or args.iterations <= 0
        or args.batch_size <= 0
        or args.convergence_tolerance < 0.0
        or not 0.0 < args.tail_fraction <= 1.0
        or args.tail_cvar_weight < 0.0
        or (args.reuse_training_cache and not args.training_cache)
        or args.rotation_iterations < 0
        or args.rotation_lloyd_iterations <= 0
        or args.rotation_random_restarts < 0
        or (args.residual_rotation == "opq" and args.rotation_iterations == 0)
    ):
        raise ValueError("invalid residual-PQ training configuration")
    device = resolve_device(args.solver_device)
    pq_model = read_model(args.pq_model)
    if pq_model.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen R=0 primary model")
    if pq_model.dim % args.groups != 0:
        raise ValueError(
            f"primary dimension {pq_model.dim} is not divisible by {args.groups} groups"
        )
    group_dim = pq_model.dim // args.groups
    identity = np.broadcast_to(
        np.eye(group_dim, dtype=np.float32),
        (args.groups, group_dim, group_dim),
    ).copy()
    metric = read_residual_metric(args.metric_model) if args.metric_model else None
    if metric is not None and (
        metric.artifact_id != pq_model.artifact_id
        or metric.dim != pq_model.dim
        or metric.groups != args.groups
    ):
        raise ValueError("residual metric is incompatible with the primary PQ model")
    metric_geometry = "blockwise" if metric is None else metric.geometry
    if args.residual_rotation == "opq" and (
        metric is None or metric_geometry != "full"
    ):
        raise ValueError("--residual-rotation opq requires --metric-model geometry=full")
    if args.residual_rotation != "opq" and (
        args.rotation_initialization != "identity"
        or args.rotation_checkpoint != "final"
    ):
        raise ValueError(
            "non-default rotation initialization/checkpoint requires "
            "--residual-rotation opq"
        )
    if metric_geometry == "full" and args.tail_objective != "mean":
        raise ValueError("full residual transforms currently require --tail-objective mean")
    if metric_geometry == "full" and args.initial_residual_model:
        raise ValueError(
            "full residual transforms do not accept --initial-residual-model; "
            "the initialization must use the same learned coordinate system"
        )
    metric_matrices = (
        identity
        if metric is None or metric_geometry == "full"
        else metric.matrices
    )
    metric_factors = (
        identity
        if metric is None or metric_geometry == "full"
        else metric.factors
    )
    initial_prototypes = None
    if args.initial_residual_model:
        initial_model = read_residual_pq_model(args.initial_residual_model)
        if (
            initial_model.artifact_id != pq_model.artifact_id
            or initial_model.node_count != pq_model.node_count
            or initial_model.dim != pq_model.dim
            or initial_model.groups != args.groups
            or initial_model.codewords != args.codewords
        ):
            raise ValueError("initial residual-PQ model is incompatible")
        initial_prototypes = np.asarray(initial_model.prototypes, dtype=np.float32)
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("PQ codes do not match the model")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the model")

    rng = np.random.default_rng(args.seed)
    sample_size = min(int(args.sample_size), pq_model.node_count)
    sample_ids = (
        np.arange(pq_model.node_count, dtype=np.int64)
        if sample_size == pq_model.node_count
        else np.sort(rng.choice(pq_model.node_count, sample_size, replace=False))
    )
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    temporary_cache = not bool(args.training_cache)
    if temporary_cache:
        descriptor, cache_path = tempfile.mkstemp(
            prefix="full_residual_pq_", suffix=".npy"
        )
        os.close(descriptor)
    else:
        cache_path = args.training_cache
    cache_meta_path = f"{cache_path}.meta.json"
    expected_cache_meta = {
        "version": 1,
        "artifact_id": int(pq_model.artifact_id),
        "node_count": int(pq_model.node_count),
        "dim": int(pq_model.dim),
        "groups": int(args.groups),
        "group_dim": int(group_dim),
        "sample_size": int(sample_size),
        "sample_seed": int(args.seed),
    }
    expected_cache_shape = (sample_size, args.groups, group_dim)
    transformed_cache = None
    transformed_cache_path = ""
    residual_transform = np.eye(pq_model.dim, dtype=np.float32)
    residual_inverse_transform = np.eye(pq_model.dim, dtype=np.float32)
    global_metric_matrix = np.eye(pq_model.dim, dtype=np.float32)
    rotation_diagnostics: list[dict] = []
    if args.reuse_training_cache:
        if not os.path.exists(cache_path) or not os.path.exists(cache_meta_path):
            raise ValueError(
                "--reuse-training-cache requested but cache or metadata is missing"
            )
        with open(cache_meta_path, encoding="utf-8") as handle:
            cache_meta = json.load(handle)
        if cache_meta != expected_cache_meta:
            raise ValueError("reused training cache metadata does not match this run")
        cache = np.load(cache_path, mmap_mode="r+")
        if cache.shape != expected_cache_shape or cache.dtype != np.float32:
            raise ValueError("reused training cache has incompatible shape or dtype")
    else:
        cache = np.lib.format.open_memmap(
            cache_path,
            mode="w+",
            dtype=np.float32,
            shape=expected_cache_shape,
        )
    try:
        if not args.reuse_training_cache:
            for start in range(0, sample_size, args.batch_size):
                end = min(sample_size, start + args.batch_size)
                ids = sample_ids[start:end]
                batch_codes = np.asarray(codes[ids], dtype=np.uint8)
                vectors_t = transform_vectors(
                    np.asarray(base[ids], dtype=np.float32), pq_model
                )
                cache[start:end] = extract_full_residual(
                    vectors_t, batch_codes, pq_model, args.groups
                )
                if args.log_interval > 0 and (
                    end % args.log_interval < args.batch_size or end == sample_size
                ):
                    print(f"[residual-pq-cache] {end}/{sample_size}", flush=True)
            cache.flush()
            with open(cache_meta_path, "w", encoding="utf-8") as handle:
                json.dump(expected_cache_meta, handle, indent=2, sort_keys=True)
        else:
            print(
                f"[residual-pq-cache] reusing {cache_path} "
                f"shape={expected_cache_shape}",
                flush=True,
            )
        training_values = cache
        if metric_geometry == "full":
            descriptor, transformed_cache_path = tempfile.mkstemp(
                prefix="transformed_full_residual_pq_", suffix=".npy"
            )
            os.close(descriptor)
            transformed_cache = np.lib.format.open_memmap(
                transformed_cache_path,
                mode="w+",
                dtype=np.float32,
                shape=expected_cache_shape,
            )
            base_metric_transform = np.asarray(metric.factors, dtype=np.float32)
            materialize_residual_transform(
                cache,
                transformed_cache,
                base_metric_transform,
                device=device,
                batch_size=args.batch_size,
                log_interval=args.log_interval,
            )
            transformed_cache.flush()
            training_values = transformed_cache
            global_metric_matrix = np.asarray(metric.matrices, dtype=np.float32)

        if args.tail_objective == "cvar":
            prototypes, counts, diagnostics = lloyd_tail_aware_metric_group_kmeans(
                training_values,
                metric_factors=metric_factors,
                codewords=args.codewords,
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                tail_fraction=args.tail_fraction,
                tail_cvar_weight=args.tail_cvar_weight,
                initial_prototypes=initial_prototypes,
                initialization=args.initialization,
                log_interval=args.kmeans_log_interval,
            )
            diagnostics["assignment_kernel"] = (
                "euclidean_tail_aware"
                if is_identity_metric(metric_matrices)
                else "mahalanobis_tail_aware"
            )
        elif is_identity_metric(metric_matrices) and args.initialization == "sample":
            prototypes, counts, diagnostics = lloyd_group_kmeans(
                training_values,
                codewords=args.codewords,
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                log_interval=args.kmeans_log_interval,
            )
            diagnostics["assignment_kernel"] = "euclidean_exact"
            if initial_prototypes is not None:
                diagnostics["initialization_warning"] = (
                    "--initial-residual-model is currently used only by "
                    "--tail-objective cvar"
                )
        else:
            prototypes, counts, diagnostics = lloyd_metric_group_kmeans(
                training_values,
                metric_factors=metric_factors,
                codewords=args.codewords,
                iterations=args.iterations,
                convergence_tolerance=args.convergence_tolerance,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
                initialization=args.initialization,
                log_interval=args.kmeans_log_interval,
            )
            diagnostics["assignment_kernel"] = (
                "euclidean_metric_exact"
                if is_identity_metric(metric_matrices)
                else "mahalanobis"
            )
            if initial_prototypes is not None:
                diagnostics["initialization_warning"] = (
                    "--initial-residual-model is currently used only by "
                    "--tail-objective cvar"
                )

        if metric_geometry == "full":
            rotation = np.eye(pq_model.dim, dtype=np.float32)
            if args.residual_rotation == "opq":
                identity_rotation = rotation.copy()
                pre_rotation_prototypes = np.asarray(prototypes, dtype=np.float32).copy()
                pre_rotation_counts = np.asarray(counts, dtype=np.int64).copy()
                pre_rotation_diagnostics = dict(diagnostics)
                starts: list[tuple[str, np.ndarray, dict]] = []
                if args.rotation_initialization in {"identity", "multistart"}:
                    starts.append(("identity", identity_rotation.copy(), {}))
                if args.rotation_initialization in {
                    "covariance-balanced", "multistart"
                }:
                    balanced, balanced_diagnostics = covariance_balanced_rotation(
                        training_values,
                        groups=args.groups,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    starts.append(
                        (
                            "covariance-balanced",
                            balanced,
                            balanced_diagnostics,
                        )
                    )
                if args.rotation_initialization == "multistart":
                    for restart in range(args.rotation_random_restarts):
                        starts.append(
                            (
                                f"random-{restart + 1}",
                                random_orthogonal_rotation(
                                    pq_model.dim,
                                    args.seed + 10000 + restart,
                                ),
                                {"seed": args.seed + 10000 + restart},
                            )
                        )
                if not starts:
                    raise RuntimeError("OPQ rotation search has no initial starts")

                cache_rotation = identity_rotation.copy()
                best_checkpoint: dict[str, object] | None = None
                start_summaries: list[dict] = []
                for start_index, (start_name, initial_rotation, start_extra) in enumerate(starts):
                    rotate_transformed_cache_in_place(
                        training_values,
                        cache_rotation,
                        initial_rotation,
                        device=device,
                        batch_size=args.batch_size,
                    )
                    cache_rotation = initial_rotation.copy()
                    start_rotation = initial_rotation.copy()
                    if start_name == "identity":
                        start_prototypes = pre_rotation_prototypes.copy()
                        start_counts = pre_rotation_counts.copy()
                        initial_lloyd = dict(pre_rotation_diagnostics)
                    else:
                        start_prototypes, start_counts, initial_lloyd = (
                            lloyd_metric_group_kmeans(
                                training_values,
                                metric_factors=identity,
                                codewords=args.codewords,
                                iterations=args.iterations,
                                convergence_tolerance=args.convergence_tolerance,
                                device=device,
                                batch_size=args.batch_size,
                                seed=args.seed + 1000 + start_index,
                                initialization=args.rotation_restart_initialization,
                                log_interval=args.kmeans_log_interval,
                            )
                        )
                    start_summary = {
                        "name": start_name,
                        "initialization": start_extra,
                        "initial_lloyd": initial_lloyd,
                        "iterations": [],
                    }
                    initial_loss = float(initial_lloyd["final_metric_mse"])
                    if args.rotation_checkpoint == "best" and (
                        best_checkpoint is None
                        or initial_loss < float(best_checkpoint["metric_mse"])
                    ):
                        best_checkpoint = {
                            "start": start_name,
                            "stage": "initialization",
                            "iteration": 0,
                            "metric_mse": initial_loss,
                            "rotation": start_rotation.copy(),
                            "prototypes": start_prototypes.copy(),
                            "counts": start_counts.copy(),
                            "lloyd": dict(initial_lloyd),
                        }

                    for rotation_iteration in range(args.rotation_iterations):
                        updated_rotation, rotation_step = update_opq_rotation(
                            training_values,
                            start_prototypes,
                            start_rotation,
                            device=device,
                            batch_size=args.batch_size,
                        )
                        rotate_transformed_cache_in_place(
                            training_values,
                            start_rotation,
                            updated_rotation,
                            device=device,
                            batch_size=args.batch_size,
                        )
                        start_rotation = updated_rotation
                        cache_rotation = updated_rotation.copy()
                        start_prototypes, start_counts, lloyd_step = (
                            lloyd_metric_group_kmeans(
                                training_values,
                                metric_factors=identity,
                                codewords=args.codewords,
                                iterations=args.rotation_lloyd_iterations,
                                convergence_tolerance=args.convergence_tolerance,
                                device=device,
                                batch_size=args.batch_size,
                                seed=args.seed + start_index,
                                initialization="sample",
                                initial_prototypes=start_prototypes,
                                log_interval=args.kmeans_log_interval,
                            )
                        )
                        rotation_step.update(
                            {
                                "iteration": rotation_iteration + 1,
                                "lloyd": lloyd_step,
                            }
                        )
                        start_summary["iterations"].append(rotation_step)
                        rotation_diagnostics.append(
                            {"start": start_name, **rotation_step}
                        )
                        loss = float(lloyd_step["final_metric_mse"])
                        if args.rotation_checkpoint == "best" and (
                            best_checkpoint is None
                            or loss < float(best_checkpoint["metric_mse"])
                        ):
                            best_checkpoint = {
                                "start": start_name,
                                "stage": "alternating-opq",
                                "iteration": rotation_iteration + 1,
                                "metric_mse": loss,
                                "rotation": start_rotation.copy(),
                                "prototypes": start_prototypes.copy(),
                                "counts": start_counts.copy(),
                                "lloyd": dict(lloyd_step),
                            }
                        print(
                            f"[residual-opq] start={start_name} "
                            f"iteration={rotation_iteration + 1} "
                            f"metric_mse={loss:.8g}",
                            flush=True,
                        )
                    final_lloyd = (
                        start_summary["iterations"][-1]["lloyd"]
                        if start_summary["iterations"]
                        else initial_lloyd
                    )
                    final_loss = float(final_lloyd["final_metric_mse"])
                    start_summary["final_metric_mse"] = final_loss
                    start_summaries.append(start_summary)
                    if args.rotation_checkpoint == "final" and (
                        best_checkpoint is None
                        or final_loss < float(best_checkpoint["metric_mse"])
                    ):
                        best_checkpoint = {
                            "start": start_name,
                            "stage": "alternating-opq",
                            "iteration": args.rotation_iterations,
                            "metric_mse": final_loss,
                            "rotation": start_rotation.copy(),
                            "prototypes": start_prototypes.copy(),
                            "counts": start_counts.copy(),
                            "lloyd": dict(final_lloyd),
                        }

                if best_checkpoint is None:
                    raise RuntimeError("OPQ rotation search produced no checkpoint")
                rotation = np.asarray(best_checkpoint["rotation"], dtype=np.float32)
                prototypes = np.asarray(
                    best_checkpoint["prototypes"], dtype=np.float32
                )
                counts = np.asarray(best_checkpoint["counts"], dtype=np.int64)
                selected_lloyd = dict(best_checkpoint["lloyd"])
                diagnostics = selected_lloyd
                diagnostics["pre_rotation_kmeans"] = pre_rotation_diagnostics
                diagnostics["rotation"] = rotation_diagnostics
                diagnostics["rotation_search"] = {
                    "initialization": args.rotation_initialization,
                    "checkpoint_policy": args.rotation_checkpoint,
                    "random_restarts": args.rotation_random_restarts,
                    "restart_centroid_initialization": (
                        args.rotation_restart_initialization
                    ),
                    "starts": start_summaries,
                    "selected": {
                        "start": best_checkpoint["start"],
                        "stage": best_checkpoint["stage"],
                        "iteration": best_checkpoint["iteration"],
                        "metric_mse": best_checkpoint["metric_mse"],
                    },
                }
                diagnostics["final_metric_mse"] = float(
                    best_checkpoint["metric_mse"]
                )
                diagnostics["assignment_kernel"] = (
                    "full_metric_transformed_euclidean"
                )

            residual_transform = np.asarray(
                np.asarray(metric.factors, dtype=np.float64)
                @ np.asarray(rotation, dtype=np.float64),
                dtype=np.float32,
            )
            residual_inverse_transform = np.asarray(
                np.linalg.inv(residual_transform.astype(np.float64)),
                dtype=np.float32,
            )
            if args.residual_rotation != "opq":
                diagnostics["rotation"] = rotation_diagnostics
                diagnostics["assignment_kernel"] = (
                    "full_metric_transformed_euclidean"
                )
    finally:
        if transformed_cache is not None:
            transformed_cache.flush()
            del transformed_cache
        if transformed_cache_path and os.path.exists(transformed_cache_path):
            os.unlink(transformed_cache_path)
        del cache
        if temporary_cache and os.path.exists(cache_path):
            os.unlink(cache_path)
        if temporary_cache and os.path.exists(cache_meta_path):
            os.unlink(cache_meta_path)

    encoding_id = compute_encoding_id(
        pq_model.artifact_id,
        pq_model.dim,
        args.groups,
        prototypes,
        metric_matrices=metric_matrices,
        residual_transform=residual_transform,
    )
    metric_mode = "euclidean" if metric is None else metric.mode
    if metric_geometry == "full":
        metric_mode = f"{metric_mode}-full"
    metric_lambda = 0.0 if metric is None else metric.metric_lambda
    metric_artifact_id = 0 if metric is None else metric.metric_id
    model = ResidualPQModel(
        artifact_id=pq_model.artifact_id,
        encoding_id=encoding_id,
        node_count=pq_model.node_count,
        dim=pq_model.dim,
        groups=args.groups,
        codewords=args.codewords,
        prototypes=prototypes,
        prototype_counts=counts,
        training_objective=(
            "full_database_residual_tail_cvar_metric_lloyd"
            if args.tail_objective == "cvar"
            else (
                (
                    "full_database_residual_full_metric_opq_lloyd"
                    if args.residual_rotation == "opq"
                    else "full_database_residual_full_metric_transform_lloyd"
                )
                if metric_geometry == "full"
                else (
                    "full_database_residual_mse_lloyd"
                    if is_identity_metric(metric_matrices)
                    else "full_database_residual_metric_assignment_lloyd"
                )
            )
        ),
        assignment_metric=metric_mode,
        metric_lambda=metric_lambda,
        metric_artifact_id=metric_artifact_id,
        metric_matrices=np.asarray(metric_matrices, dtype=np.float32),
        metric_factors=np.asarray(metric_factors, dtype=np.float32),
        residual_transform=residual_transform,
        residual_inverse_transform=residual_inverse_transform,
        global_metric_matrix=global_metric_matrix,
        transform_kind=(
            "identity"
            if metric_geometry != "full"
            else (
                "full-query-metric-opq"
                if args.residual_rotation == "opq"
                else "full-query-metric"
            )
        ),
    )
    model_path = output / "residual_pq_model.npz"
    write_residual_pq_model(str(model_path), model)
    summary = {
        "version": 1,
        "method": "equal_budget_full_dimensional_residual_pq",
        "training_objective": model.training_objective,
        "tail_objective": args.tail_objective,
        "tail_fraction": args.tail_fraction,
        "tail_cvar_weight": args.tail_cvar_weight,
        "assignment_metric": model.assignment_metric,
        "metric_geometry": metric_geometry,
        "residual_transform_kind": model.transform_kind,
        "has_residual_transform": model.has_residual_transform,
        "metric_lambda": model.metric_lambda,
        "metric_artifact_id": model.metric_artifact_id,
        "uses_euclidean_assignment": model.uses_euclidean_assignment,
        "groups": model.groups,
        "group_dim": model.group_dim,
        "covered_residual_dimensions": model.dim,
        "codewords_per_group": model.codewords,
        "bits_per_group": model.bits_per_group,
        "side_bits_per_node": model.side_bits_per_node,
        "side_bytes_per_node": model.side_bytes_per_node,
        "sample_size": sample_size,
        "kmeans": diagnostics,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "metric_model": (
                os.path.abspath(args.metric_model) if args.metric_model else None
            ),
            "initial_residual_model": (
                os.path.abspath(args.initial_residual_model)
                if args.initial_residual_model
                else None
            ),
            "model": os.path.abspath(model_path),
        },
        "config": vars(args),
    }
    with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote residual-PQ model: {model_path}")
    print(f"wrote summary: {output / 'train_summary.json'}")


if __name__ == "__main__":
    main()
