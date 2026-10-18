#!/usr/bin/env python3
"""Train a full-W additive residual quantizer with optional primary conditioning."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path

import numpy as np

from additive_residual_quantizer import (
    assign_additive_dataset,
    conditional_prediction,
    fit_low_rank_primary_conditioner,
    train_additive_codebooks,
)
from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from group_residual_vq import resolve_device
from residual_metric_io import read_residual_metric
from residual_opq import materialize_residual_transform
from residual_pq_baseline import extract_full_residual
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
    parser.add_argument("--metric-model", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--groups", type=int, default=8)
    parser.add_argument("--codewords", type=int, default=256)
    parser.add_argument("--sample-size", type=int, default=1000000)
    parser.add_argument("--initialization-iterations", type=int, default=5)
    parser.add_argument("--joint-sweeps", type=int, default=2)
    parser.add_argument("--coordinate-iterations", type=int, default=1)
    parser.add_argument("--initial-residual-model", default="")
    parser.add_argument("--freeze-additive-codebooks", action="store_true")
    parser.add_argument("--conditional-assignment-sweeps", type=int, default=1)
    parser.add_argument("--conditional-codebook-sweeps", type=int, default=1)
    parser.add_argument("--conditional-rank", type=int, default=0)
    parser.add_argument("--conditional-rounds", type=int, default=2)
    parser.add_argument("--conditional-epochs", type=int, default=3)
    parser.add_argument("--conditional-learning-rate", type=float, default=3e-4)
    parser.add_argument("--conditional-weight-decay", type=float, default=1e-6)
    parser.add_argument("--solver-device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--training-cache", default="")
    parser.add_argument("--reuse-training-cache", action="store_true")
    parser.add_argument("--log-interval", type=int, default=100000)
    return parser.parse_args()


def _materialize_innovation(
    transformed: np.ndarray,
    primary_codes: np.ndarray,
    basis: np.ndarray,
    embeddings: np.ndarray,
    target: np.ndarray,
    batch_size: int,
) -> None:
    flat = transformed.reshape(transformed.shape[0], -1)
    output = target.reshape(target.shape[0], -1)
    for start in range(0, flat.shape[0], batch_size):
        end = min(flat.shape[0], start + batch_size)
        prediction = conditional_prediction(
            primary_codes[start:end], basis, embeddings
        )
        output[start:end] = np.asarray(flat[start:end], dtype=np.float32) - prediction
    if hasattr(target, "flush"):
        target.flush()


def main() -> None:
    args = parse_args()
    if (
        args.groups <= 0
        or args.codewords != 256
        or args.sample_size < args.codewords
        or args.initialization_iterations <= 0
        or args.joint_sweeps < 0
        or args.coordinate_iterations <= 0
        or args.conditional_assignment_sweeps < 0
        or args.conditional_codebook_sweeps < 0
        or args.conditional_rank < 0
        or args.conditional_rounds < 0
        or args.conditional_epochs <= 0
        or args.conditional_learning_rate <= 0.0
        or args.conditional_weight_decay < 0.0
        or args.batch_size <= 0
        or (args.reuse_training_cache and not args.training_cache)
    ):
        raise ValueError("invalid additive residual training configuration")
    if args.conditional_rank == 0 and args.conditional_rounds != 0:
        args.conditional_rounds = 0
    if args.conditional_rank > 0 and args.conditional_rounds <= 0:
        raise ValueError("conditioned training requires positive conditional rounds")
    if args.freeze_additive_codebooks and args.conditional_rank <= 0:
        raise ValueError("frozen additive conditioning requires conditional-rank > 0")

    device = resolve_device(args.solver_device)
    pq_model = read_model(args.pq_model)
    if args.conditional_rank > pq_model.dim:
        raise ValueError("conditional rank cannot exceed the residual dimension")
    metric = read_residual_metric(args.metric_model)
    if (
        pq_model.refined_subspaces != 0
        or metric.geometry != "full"
        or metric.artifact_id != pq_model.artifact_id
        or metric.dim != pq_model.dim
        or metric.groups != args.groups
        or pq_model.dim % args.groups != 0
    ):
        raise ValueError("additive training requires a compatible full residual metric")
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    if codes.shape != (pq_model.node_count, pq_model.subspaces):
        raise ValueError("primary PQ-code shape mismatch")
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base-vector shape mismatch")

    sample_size = min(args.sample_size, pq_model.node_count)
    rng = np.random.default_rng(args.seed)
    sample_ids = (
        np.arange(pq_model.node_count, dtype=np.int64)
        if sample_size == pq_model.node_count
        else np.sort(rng.choice(pq_model.node_count, sample_size, replace=False))
    )
    group_dim = pq_model.dim // args.groups
    cache_shape = (sample_size, args.groups, group_dim)
    cache_meta = {
        "version": 1,
        "artifact_id": int(pq_model.artifact_id),
        "node_count": int(pq_model.node_count),
        "dim": int(pq_model.dim),
        "groups": int(args.groups),
        "group_dim": int(group_dim),
        "sample_size": int(sample_size),
        "sample_seed": int(args.seed),
    }
    temporary_cache = not bool(args.training_cache)
    if temporary_cache:
        descriptor, cache_path = tempfile.mkstemp(prefix="additive_residual_", suffix=".npy")
        os.close(descriptor)
    else:
        cache_path = args.training_cache
    meta_path = f"{cache_path}.meta.json"
    transformed_descriptor, transformed_path = tempfile.mkstemp(
        prefix="additive_transformed_", suffix=".npy"
    )
    os.close(transformed_descriptor)
    innovation_descriptor, innovation_path = tempfile.mkstemp(
        prefix="additive_innovation_", suffix=".npy"
    )
    os.close(innovation_descriptor)
    cache = None
    transformed = None
    innovation = None
    output = Path(args.out_dir)
    output.mkdir(parents=True, exist_ok=True)
    try:
        if args.reuse_training_cache:
            with open(meta_path, encoding="utf-8") as handle:
                if json.load(handle) != cache_meta:
                    raise ValueError("training-cache metadata mismatch")
            cache = np.load(cache_path, mmap_mode="r")
            if cache.shape != cache_shape or cache.dtype != np.float32:
                raise ValueError("training-cache array mismatch")
        else:
            cache = np.lib.format.open_memmap(
                cache_path, mode="w+", dtype=np.float32, shape=cache_shape
            )
            for start in range(0, sample_size, args.batch_size):
                end = min(sample_size, start + args.batch_size)
                ids = sample_ids[start:end]
                vectors_t = transform_vectors(
                    np.asarray(base[ids], dtype=np.float32), pq_model
                )
                cache[start:end] = extract_full_residual(
                    vectors_t,
                    np.asarray(codes[ids], dtype=np.uint8),
                    pq_model,
                    args.groups,
                )
                if args.log_interval > 0 and (
                    end % args.log_interval < args.batch_size or end == sample_size
                ):
                    print(f"[additive-residual-cache] {end}/{sample_size}", flush=True)
            cache.flush()
            with open(meta_path, "w", encoding="utf-8") as handle:
                json.dump(cache_meta, handle, indent=2, sort_keys=True)

        transformed = np.lib.format.open_memmap(
            transformed_path, mode="w+", dtype=np.float32, shape=cache_shape
        )
        materialize_residual_transform(
            cache,
            transformed,
            np.asarray(metric.factors, dtype=np.float32),
            device=device,
            batch_size=args.batch_size,
            log_interval=args.log_interval,
        )
        transformed.flush()
        flat = transformed.reshape(sample_size, pq_model.dim)
        if args.initial_residual_model:
            initial_model = read_residual_pq_model(args.initial_residual_model)
            if (
                initial_model.artifact_id != pq_model.artifact_id
                or initial_model.dim != pq_model.dim
                or initial_model.groups != args.groups
                or initial_model.codewords != args.codewords
                or initial_model.quantizer_kind != "additive"
                or initial_model.has_primary_conditioning
                or not np.allclose(
                    initial_model.residual_transform,
                    metric.factors,
                    rtol=1e-6,
                    atol=1e-7,
                )
            ):
                raise ValueError("initial additive model is incompatible")
            codebooks = np.asarray(
                initial_model.prototypes, dtype=np.float32
            ).copy()
            side_assignments, counts, initial_mse = assign_additive_dataset(
                flat,
                codebooks,
                device=device,
                batch_size=args.batch_size,
                coordinate_sweeps=args.conditional_assignment_sweeps,
            )
            additive_diagnostics = {
                "method": "fixed_additive_model_initialization",
                "initial_model": os.path.abspath(args.initial_residual_model),
                "final_metric_mse": initial_mse,
                "empty_sample_codewords": int(np.sum(counts == 0)),
            }
        else:
            codebooks, side_assignments, counts, additive_diagnostics = train_additive_codebooks(
                flat,
                dictionaries=args.groups,
                codewords=args.codewords,
                initialization_iterations=args.initialization_iterations,
                joint_sweeps=args.joint_sweeps,
                coordinate_iterations=args.coordinate_iterations,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed,
            )

        basis = np.empty((pq_model.dim, 0), dtype=np.float32)
        embeddings = np.empty((0, 0, 0), dtype=np.float32)
        conditional_diagnostics: list[dict] = []
        final_training_metric_mse = float(
            additive_diagnostics["final_metric_mse"]
        )
        sample_primary_codes = np.asarray(codes[sample_ids], dtype=np.uint8)
        if args.conditional_rank > 0:
            innovation = np.lib.format.open_memmap(
                innovation_path, mode="w+", dtype=np.float32, shape=cache_shape
            )
            for conditional_round in range(args.conditional_rounds):
                basis, embeddings, fit_diagnostics = fit_low_rank_primary_conditioner(
                    flat,
                    sample_primary_codes,
                    side_assignments,
                    codebooks,
                    rank=args.conditional_rank,
                    primary_codewords=pq_model.codewords,
                    epochs=args.conditional_epochs,
                    learning_rate=args.conditional_learning_rate,
                    weight_decay=args.conditional_weight_decay,
                    device=device,
                    batch_size=args.batch_size,
                    seed=args.seed + conditional_round,
                    initial_basis=basis if basis.size else None,
                    initial_embeddings=embeddings if embeddings.size else None,
                )
                _materialize_innovation(
                    transformed,
                    sample_primary_codes,
                    basis,
                    embeddings,
                    innovation,
                    args.batch_size,
                )
                innovation_flat = innovation.reshape(sample_size, pq_model.dim)
                if args.freeze_additive_codebooks:
                    side_assignments, counts, assigned_mse = assign_additive_dataset(
                        innovation_flat,
                        codebooks,
                        device=device,
                        batch_size=args.batch_size,
                        coordinate_sweeps=args.conditional_assignment_sweeps,
                        initial_codes=side_assignments,
                    )
                    refine_diagnostics = {
                        "method": "fixed_codebook_additive_reassignment",
                        "final_metric_mse": assigned_mse,
                        "codebooks_frozen": True,
                    }
                else:
                    codebooks, side_assignments, counts, refine_diagnostics = train_additive_codebooks(
                        innovation_flat,
                        dictionaries=args.groups,
                        codewords=args.codewords,
                        initialization_iterations=args.initialization_iterations,
                        joint_sweeps=args.conditional_codebook_sweeps,
                        coordinate_iterations=args.coordinate_iterations,
                        device=device,
                        batch_size=args.batch_size,
                        seed=args.seed + conditional_round + 1,
                        initial_codebooks=codebooks,
                        initial_codes=side_assignments,
                        log_prefix="conditional-additive",
                    )
                conditional_diagnostics.append(
                    {
                        "round": conditional_round + 1,
                        "conditioner": fit_diagnostics,
                        "additive_refinement": refine_diagnostics,
                    }
                )
            basis, embeddings, final_fit = fit_low_rank_primary_conditioner(
                flat,
                sample_primary_codes,
                side_assignments,
                codebooks,
                rank=args.conditional_rank,
                primary_codewords=pq_model.codewords,
                epochs=args.conditional_epochs,
                learning_rate=args.conditional_learning_rate,
                weight_decay=args.conditional_weight_decay,
                device=device,
                batch_size=args.batch_size,
                seed=args.seed + args.conditional_rounds,
                initial_basis=basis,
                initial_embeddings=embeddings,
            )
            conditional_diagnostics.append(
                {"stage": "final-conditioner-fit", "conditioner": final_fit}
            )
            _materialize_innovation(
                transformed,
                sample_primary_codes,
                basis,
                embeddings,
                innovation,
                args.batch_size,
            )
            side_assignments, counts, final_training_metric_mse = assign_additive_dataset(
                innovation.reshape(sample_size, pq_model.dim),
                codebooks,
                device=device,
                batch_size=args.batch_size,
                coordinate_sweeps=args.conditional_assignment_sweeps,
                initial_codes=side_assignments,
            )

        transform = np.asarray(metric.factors, dtype=np.float32)
        inverse = np.asarray(
            np.linalg.inv(transform.astype(np.float64)), dtype=np.float32
        )
        identity = np.broadcast_to(
            np.eye(group_dim, dtype=np.float32),
            (args.groups, group_dim, group_dim),
        ).copy()
        encoding_id = compute_encoding_id(
            pq_model.artifact_id,
            pq_model.dim,
            args.groups,
            codebooks,
            metric_matrices=identity,
            residual_transform=transform,
            quantizer_kind="additive",
            conditional_basis=basis,
            conditional_embeddings=embeddings,
        )
        conditioned = args.conditional_rank > 0
        model = ResidualPQModel(
            artifact_id=pq_model.artifact_id,
            encoding_id=encoding_id,
            node_count=pq_model.node_count,
            dim=pq_model.dim,
            groups=args.groups,
            codewords=args.codewords,
            prototypes=codebooks,
            prototype_counts=counts,
            training_objective=(
                "full_metric_primary_conditioned_additive_residual_quantization"
                if conditioned
                else "full_metric_additive_residual_quantization"
            ),
            assignment_metric=f"{metric.mode}-full",
            metric_lambda=metric.metric_lambda,
            metric_artifact_id=metric.metric_id,
            metric_matrices=identity,
            metric_factors=identity,
            residual_transform=transform,
            residual_inverse_transform=inverse,
            global_metric_matrix=np.asarray(metric.matrices, dtype=np.float32),
            transform_kind=(
                "full-query-metric-primary-conditioned-additive"
                if conditioned
                else "full-query-metric-additive"
            ),
            quantizer_kind="additive",
            conditional_basis=basis,
            conditional_embeddings=embeddings,
        )
        model_path = output / "residual_pq_model.npz"
        write_residual_pq_model(str(model_path), model)
        summary = {
            "version": 1,
            "method": "full_metric_additive_residual_quantizer",
            "quantizer_kind": model.quantizer_kind,
            "primary_conditioned": model.has_primary_conditioning,
            "conditional_rank": model.conditional_rank,
            "training_objective": model.training_objective,
            "metric_lambda": model.metric_lambda,
            "groups": model.groups,
            "prototype_dim": model.prototype_dim,
            "codewords_per_dictionary": model.codewords,
            "side_bits_per_node": model.side_bits_per_node,
            "sample_size": sample_size,
            "additive_training": additive_diagnostics,
            "conditional_training": conditional_diagnostics,
            "additive_codebooks_frozen": bool(args.freeze_additive_codebooks),
            "final_training_metric_mse": final_training_metric_mse,
            "global_model_bytes": int(
                model.prototypes.nbytes
                + model.conditional_basis.nbytes
                + model.conditional_embeddings.nbytes
                + model.residual_inverse_transform.nbytes
            ),
            "files": {
                "pq_model": os.path.abspath(args.pq_model),
                "codes": os.path.abspath(args.codes),
                "base_fbin": os.path.abspath(args.base_fbin),
                "metric_model": os.path.abspath(args.metric_model),
                "model": os.path.abspath(model_path),
            },
            "config": vars(args),
        }
        with open(output / "train_summary.json", "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2, sort_keys=True)
        print(f"wrote additive residual model: {model_path}")
    finally:
        if innovation is not None:
            innovation.flush()
            del innovation
        if transformed is not None:
            transformed.flush()
            del transformed
        if cache is not None:
            del cache
        for path in (transformed_path, innovation_path):
            if os.path.exists(path):
                os.unlink(path)
        if temporary_cache:
            for path in (cache_path, meta_path):
                if os.path.exists(path):
                    os.unlink(path)


if __name__ == "__main__":
    main()
