#!/usr/bin/env python3
"""Alternate terminal-paired node re-encoding and JI codebook fitting.

The primary quantizer, graph, JI transform, eight 8-bit groups, and total
64-bit/node budget remain fixed.  Nodes observed in primary-only terminal
candidate pools may change their eight code indices.  Nodes absent from the
training pools retain their incoming assignments.

Assignment minimizes the paired terminal score-regression objective over all
query x beam events associated with one database node.  Shared codebooks are
then refit with the assignments frozen.  Both sub-steps are checkpointed by
independent validation fixed-pool Recall, with multi-boundary inversion as the
tie-break, so a bad alternating step can never replace the input model.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

from drb_io import fbin_memmap
from joint_innovation_rpq import read_model, read_side_codes, write_model
from primary_quantizer_manifest import read_manifest
from train_terminal_conditioned_ji import (
    Events,
    active_codebooks,
    make_model,
    predict_events,
    query_features,
    reencode_seen_nodes,
    resolve_device,
    train_codebooks,
    weighted_mse,
    unpack_all_codes,
    write_fixed_side,
)
from train_terminal_ranking_ji import (
    Pools,
    get_or_build_pools,
    is_better,
    predict_all,
    ranking_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--exact-base", required=True)
    parser.add_argument("--exact-queries", required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--initial-side-codes", required=True)
    parser.add_argument("--train-retained", action="append", required=True)
    parser.add_argument("--validation-retained", action="append", required=True)
    parser.add_argument("--pool-cache", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-side-codes", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--sweeps", type=int, default=2)
    parser.add_argument("--retrain-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--max-assignment-score-elements", type=int, default=2_000_000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rank-max", type=int, default=100)
    parser.add_argument("--boundary-window", type=int, default=5)
    parser.add_argument("--validation-boundary-pools", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--force-pool-cache", action="store_true")
    return parser.parse_args()


def pools_to_events(pools: Pools) -> Events:
    """Preserve equal query x beam weighting for paired node assignments."""
    lengths = np.diff(pools.offsets).astype(np.int64, copy=False)
    if np.any(lengths <= 0):
        raise ValueError("terminal pool contains no candidates")
    query_ids = np.repeat(pools.query_ids, lengths)
    # Each query x beam pool receives unit total mass.  Constants shared by all
    # pools do not affect coordinate assignments or weighted-MSE fitting.
    weights = np.repeat(1.0 / lengths.astype(np.float64), lengths).astype(np.float32)
    return Events(query_ids, pools.node_ids, weights, pools.targets)


def evaluate(
    pools: Pools,
    events: Events,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: np.ndarray,
    args: argparse.Namespace,
    device: torch.device,
) -> dict:
    books_t = torch.as_tensor(books, device=device)
    correction = predict_all(pools, codes, phi, books_t, args.batch_size, device)
    metrics = ranking_metrics(
        pools,
        correction,
        args.topk,
        args.rank_max,
        args.boundary_window,
        args.validation_boundary_pools,
    )
    paired = predict_events(events, codes, phi, books_t, args.batch_size, device)
    metrics["validation_paired_mse"] = weighted_mse(events, paired)
    return metrics


def main() -> None:
    args = parse_args()
    if (
        args.sweeps < 0
        or args.retrain_epochs < 0
        or args.batch_size <= 0
        or args.learning_rate <= 0.0
        or args.max_assignment_score_elements <= 0
        or args.topk <= 0
        or args.rank_max <= args.topk
        or args.boundary_window <= 0
    ):
        raise ValueError("invalid alternating terminal-JI configuration")

    device = resolve_device(args.device)
    primary = read_manifest(args.primary_manifest)
    initial = read_model(args.initial_model)
    if (
        initial.artifact_id != primary.artifact_id
        or initial.node_count != primary.node_count
        or initial.dim != primary.dim
        or initial.groups != 8
        or initial.budget_bits != 64
        or np.any(np.asarray(initial.bits) != 8)
    ):
        raise ValueError("alternating trainer requires the deployed 8x8-bit JI model")

    packed = read_side_codes(args.initial_side_codes, initial)
    codes = unpack_all_codes(packed, initial)
    del packed
    primary_queries = fbin_memmap(primary.query_fbin)
    phi = torch.as_tensor(query_features(primary_queries, initial), device=device)

    cache = Path(args.pool_cache)
    train, train_metadata = get_or_build_pools(
        cache / "train",
        args.train_retained,
        args.exact_base,
        args.exact_queries,
        primary.reconstruction_fbin,
        primary.query_fbin,
        initial.residual_mean,
        primary.node_count,
        args.force_pool_cache,
    )
    validation, validation_metadata = get_or_build_pools(
        cache / "validation",
        args.validation_retained,
        args.exact_base,
        args.exact_queries,
        primary.reconstruction_fbin,
        primary.query_fbin,
        initial.residual_mean,
        primary.node_count,
        args.force_pool_cache,
    )
    train_events = pools_to_events(train)
    validation_events = pools_to_events(validation)
    seen_nodes = np.unique(train_events.node_ids)
    initial_seen_codes = codes[seen_nodes].copy()

    books = active_codebooks(initial)
    initial_metrics = {
        "sweep": 0,
        "stage": "input-checkpoint",
        **evaluate(
            validation, validation_events, codes, phi, books, args, device
        ),
    }
    history: list[dict] = [initial_metrics]
    best_metrics = initial_metrics
    best_books = books.copy()
    best_seen_codes = initial_seen_codes.copy()

    train_before = weighted_mse(
        train_events,
        predict_events(
            train_events,
            codes,
            phi,
            torch.as_tensor(books, device=device),
            args.batch_size,
            device,
        ),
    )

    for sweep in range(1, args.sweeps + 1):
        returned_nodes, assignment = reencode_seen_nodes(
            train_events,
            codes,
            phi,
            torch.as_tensor(books, device=device),
            args.batch_size,
            args.max_assignment_score_elements,
            device,
        )
        if not np.array_equal(returned_nodes, seen_nodes):
            raise RuntimeError("terminal re-encoder returned inconsistent node order")

        assignment_metrics = {
            "sweep": sweep,
            "stage": "node-reencoding",
            **assignment,
            **evaluate(
                validation, validation_events, codes, phi, books, args, device
            ),
        }
        selected = is_better(assignment_metrics, best_metrics)
        assignment_metrics["selected"] = selected
        history.append(assignment_metrics)
        print(
            f"[terminal-alternating] {json.dumps(assignment_metrics, sort_keys=True)}",
            flush=True,
        )
        if selected:
            best_metrics = assignment_metrics
            best_books = books.copy()
            best_seen_codes = codes[seen_nodes].copy()

        books, codebook_history = train_codebooks(
            books,
            codes,
            train_events,
            validation_events,
            phi,
            device,
            args.retrain_epochs,
            args.batch_size,
            args.learning_rate,
            args.patience,
            args.seed + sweep,
        )
        codebook_metrics = {
            "sweep": sweep,
            "stage": "codebook-refit",
            "codebook_history": codebook_history,
            **evaluate(
                validation, validation_events, codes, phi, books, args, device
            ),
        }
        selected = is_better(codebook_metrics, best_metrics)
        codebook_metrics["selected"] = selected
        history.append(codebook_metrics)
        print(
            f"[terminal-alternating] {json.dumps(codebook_metrics, sort_keys=True)}",
            flush=True,
        )
        if selected:
            best_metrics = codebook_metrics
            best_books = books.copy()
            best_seen_codes = codes[seen_nodes].copy()

    # Restore the best validation checkpoint.  Nodes absent from the training
    # terminal pools were never touched and retain their incoming assignments.
    codes[seen_nodes] = best_seen_codes
    books = best_books
    final_train_mse = weighted_mse(
        train_events,
        predict_events(
            train_events,
            codes,
            phi,
            torch.as_tensor(books, device=device),
            args.batch_size,
            device,
        ),
    )
    final_metrics = evaluate(
        validation, validation_events, codes, phi, books, args, device
    )
    changed_by_group = np.count_nonzero(
        best_seen_codes != initial_seen_codes, axis=0
    ).astype(np.int64)
    changed_nodes = int(
        np.count_nonzero(np.any(best_seen_codes != initial_seen_codes, axis=1))
    )

    output_model = make_model(
        initial, books, codes, "terminal-paired-alternating-reencode-a1e"
    )
    write_model(args.out_model, output_model)
    write_fixed_side(args.out_side_codes, output_model, codes)
    summary = {
        "version": 1,
        "method": "terminal_paired_alternating_joint_innovation",
        "primary": primary.name,
        "representation": "one fixed 8x8-bit code per database node",
        "budget_bits_per_node": 64,
        "pairing_preserved": True,
        "assignment_objective": (
            "equal-query x equal-beam paired terminal score MSE; "
            "exact coordinate search per group"
        ),
        "checkpoint_selection": (
            "maximum validation fixed-pool Recall; boundary-AUC inversion tie-break; "
            "input model remains a valid fallback"
        ),
        "unseen_node_policy": "retain incoming A1d/A0-compatible code indices",
        "train": train_metadata,
        "validation": validation_metadata,
        "seen_training_nodes": int(seen_nodes.size),
        "seen_training_fraction": float(seen_nodes.size) / float(primary.node_count),
        "changed_seen_nodes": changed_nodes,
        "changed_seen_fraction": changed_nodes / float(max(1, seen_nodes.size)),
        "changed_nodes_by_group": changed_by_group.tolist(),
        "train_paired_mse_before": train_before,
        "train_paired_mse_after": final_train_mse,
        "train_relative_mse_reduction": 1.0 - final_train_mse / train_before,
        "initial_validation": initial_metrics,
        "selected_validation": best_metrics,
        "final_validation": final_metrics,
        "history": history,
        "files": {
            "model": os.path.abspath(args.out_model),
            "side_codes": os.path.abspath(args.out_side_codes),
        },
        "config": vars(args),
    }
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote alternating terminal JI model: {args.out_model}")
    print(f"wrote alternating terminal JI side codes: {args.out_side_codes}")
    print(f"wrote alternating terminal JI summary: {args.out_summary}")


if __name__ == "__main__":
    main()
