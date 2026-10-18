#!/usr/bin/env python3
"""Regularized terminal node re-encoding with a global JI geometry prior.

Codebooks, transform, graph, group layout, and the 64-bit/node budget are
frozen.  A training-seen node may change its eight indices by minimizing a
node-adaptive mixture of:

  * global innovation reconstruction error ||y_i - c(a_i)||^2; and
  * paired terminal score-regression error over its observed queries.

The mixture weight is alpha_i = n_eff / (n_eff + D_eff), where n_eff is the
effective number of distinct training queries for node i and D_eff is the
participation-ratio dimension of the JI spectrum.  This gives sparse nodes a
strong global prior without introducing a tuned task-weight hyperparameter.
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
from train_terminal_alternating_ji import evaluate, pools_to_events
from train_terminal_conditioned_ji import (
    Events,
    active_codebooks,
    make_model,
    predict_events,
    query_features,
    resolve_device,
    unpack_all_codes,
    weighted_mse,
    write_fixed_side,
)
from train_terminal_ranking_ji import get_or_build_pools, is_better


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
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--innovation-batch-size", type=int, default=16384)
    parser.add_argument("--max-assignment-score-elements", type=int, default=2_000_000)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rank-max", type=int, default=100)
    parser.add_argument("--boundary-window", type=int, default=5)
    parser.add_argument("--validation-boundary-pools", type=int, default=12000)
    parser.add_argument("--effective-dimension", type=float, default=0.0)
    parser.add_argument(
        "--alpha-mode", choices=("adaptive", "zero", "one"), default="adaptive"
    )
    parser.add_argument(
        "--checkpoint-policy", choices=("validation", "always"), default="validation"
    )
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--force-pool-cache", action="store_true")
    return parser.parse_args()


def participation_dimension(spectrum: np.ndarray) -> float:
    values = np.maximum(np.asarray(spectrum, dtype=np.float64), 0.0)
    total = float(np.sum(values))
    squared = float(np.sum(values * values))
    if total <= 0.0 or squared <= 0.0:
        raise ValueError("JI spectrum has no positive energy")
    return total * total / squared


def effective_query_counts(events: Events, node_count: int) -> np.ndarray:
    """Kish effective count after merging repeated beam events per query."""
    if events.size == 0:
        raise ValueError("cannot estimate effective query counts from no events")
    order = np.lexsort((events.query_ids, events.node_ids))
    nodes = events.node_ids[order]
    queries = events.query_ids[order]
    weights = np.asarray(events.weights[order], dtype=np.float64)
    boundary = np.empty(nodes.size, dtype=bool)
    boundary[0] = True
    boundary[1:] = (nodes[1:] != nodes[:-1]) | (queries[1:] != queries[:-1])
    starts = np.flatnonzero(boundary)
    pair_nodes = nodes[starts]
    pair_weights = np.add.reduceat(weights, starts)
    total = np.bincount(pair_nodes, weights=pair_weights, minlength=node_count)
    squared = np.bincount(
        pair_nodes, weights=pair_weights * pair_weights, minlength=node_count
    )
    result = np.zeros(node_count, dtype=np.float32)
    valid = squared > 0.0
    result[valid] = np.asarray(
        total[valid] * total[valid] / squared[valid], dtype=np.float32
    )
    return result


def innovation_targets(
    primary,
    model,
    nodes: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    base = fbin_memmap(primary.base_fbin)
    reconstruction = fbin_memmap(primary.reconstruction_fbin)
    if (
        base.shape != reconstruction.shape
        or base.shape != (primary.node_count, primary.dim)
    ):
        raise ValueError("primary base/reconstruction cannot define residual targets")
    output = np.empty((nodes.size, primary.dim), dtype=np.float32)
    mean = torch.as_tensor(model.residual_mean, device=device)
    forward = torch.as_tensor(model.forward_transform, device=device)
    with torch.no_grad():
        for begin in range(0, nodes.size, batch_size):
            end = min(nodes.size, begin + batch_size)
            selected = nodes[begin:end]
            residual = np.asarray(base[selected], dtype=np.float32) - np.asarray(
                reconstruction[selected], dtype=np.float32
            )
            value = torch.as_tensor(residual, device=device)
            output[begin:end] = ((value - mean) @ forward).cpu().numpy()
            if end % 100000 < batch_size or end == nodes.size:
                print(
                    f"[terminal-shrinkage-targets] {end}/{nodes.size}", flush=True
                )
    return output


def reencode_with_shrinkage(
    events: Events,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    seen_nodes: np.ndarray,
    seen_targets: np.ndarray,
    alpha_all: np.ndarray,
    batch_size: int,
    max_score_elements: int,
    device: torch.device,
    group_indices=None,
) -> dict:
    """One exact group-coordinate sweep under the shrinkage objective."""
    predictions = predict_events(events, codes, phi, books, batch_size, device)
    order = np.argsort(events.node_ids, kind="stable")
    sorted_nodes = events.node_ids[order]
    unique_nodes, starts, counts = np.unique(
        sorted_nodes, return_index=True, return_counts=True
    )
    if not np.array_equal(unique_nodes, seen_nodes):
        raise RuntimeError("innovation targets and event node ordering disagree")
    buckets: dict[int, list[int]] = {}
    for record, count in enumerate(counts.tolist()):
        bucket = 1 << max(0, int(count - 1).bit_length())
        buckets.setdefault(bucket, []).append(record)

    changed_by_group: list[int] = []
    for group in (range(books.shape[0]) if group_indices is None else group_indices):
        old_labels = codes[:, group].copy()
        # Equal-width deployment groups are guaranteed by active_codebooks;
        # targets are still gathered through their actual innovation indices.
        for padded_count, records in sorted(buckets.items()):
            records_array = np.asarray(records, dtype=np.int64)
            batch_nodes = max(
                1,
                max_score_elements
                // max(1, padded_count * int(books.shape[1])),
            )
            for begin in range(0, records_array.size, batch_nodes):
                chosen = records_array[begin : begin + batch_nodes]
                rows = chosen.size
                event_matrix = np.full((rows, padded_count), -1, dtype=np.int64)
                node_batch = unique_nodes[chosen]
                for row, record in enumerate(chosen.tolist()):
                    count = int(counts[record])
                    event_matrix[row, :count] = order[
                        int(starts[record]) : int(starts[record]) + count
                    ]
                valid = event_matrix >= 0
                safe_events = np.maximum(event_matrix, 0)
                flat_events = safe_events.reshape(-1)
                qids = torch.as_tensor(
                    events.query_ids[flat_events].reshape(rows, padded_count),
                    device=device,
                )
                weights = np.where(
                    valid, events.weights[safe_events], 0.0
                ).astype(np.float32)
                weight_sum = np.maximum(np.sum(weights, axis=1), 1e-30)
                phi_group = phi[qids, group, :]
                old_vectors = books[
                    group,
                    torch.as_tensor(old_labels[node_batch], device=device),
                ][:, None, :]
                current_group = torch.sum(phi_group * old_vectors, dim=2).cpu().numpy()
                desired = (
                    events.targets[safe_events]
                    - (predictions[safe_events] - current_group)
                ).astype(np.float32)
                candidate = torch.einsum("bmw,kw->bmk", phi_group, books[group])
                desired_t = torch.as_tensor(desired, device=device)[:, :, None]
                weight_t = torch.as_tensor(weights, device=device)[:, :, None]
                terminal = torch.sum(
                    weight_t * torch.square(candidate - desired_t), dim=1
                ) / torch.as_tensor(weight_sum, device=device)[:, None]

                target = torch.as_tensor(
                    seen_targets[chosen, group, :], device=device
                )
                global_loss = torch.sum(
                    torch.square(target[:, None, :] - books[group][None, :, :]),
                    dim=2,
                )
                alpha = torch.as_tensor(alpha_all[node_batch], device=device)[:, None]
                loss = alpha * terminal + (1.0 - alpha) * global_loss
                best = torch.argmin(loss, dim=1).detach().cpu().numpy()
                codes[node_batch, group] = best

        old_score = np.empty(events.size, dtype=np.float32)
        new_score = np.empty(events.size, dtype=np.float32)
        with torch.no_grad():
            for begin in range(0, events.size, batch_size):
                end = min(events.size, begin + batch_size)
                qids = torch.as_tensor(events.query_ids[begin:end], device=device)
                node_ids = events.node_ids[begin:end]
                values = phi[qids, group, :]
                old = books[group, torch.as_tensor(old_labels[node_ids], device=device)]
                new = books[
                    group, torch.as_tensor(codes[node_ids, group], device=device)
                ]
                old_score[begin:end] = torch.sum(values * old, dim=1).cpu().numpy()
                new_score[begin:end] = torch.sum(values * new, dim=1).cpu().numpy()
        predictions += new_score - old_score
        changed = int(
            np.count_nonzero(codes[seen_nodes, group] != old_labels[seen_nodes])
        )
        changed_by_group.append(changed)
        print(
            f"[terminal-shrinkage] group={group + 1}/{books.shape[0]} "
            f"changed_nodes={changed}",
            flush=True,
        )
    return {
        "seen_nodes": int(seen_nodes.size),
        "changed_nodes_by_group": changed_by_group,
        "train_paired_mse_after_assignment": weighted_mse(events, predictions),
    }


def global_distortion(
    targets: np.ndarray, codes: np.ndarray, seen_nodes: np.ndarray, books: np.ndarray
) -> float:
    total = 0.0
    dimensions = 0
    for group in range(books.shape[0]):
        decoded = books[group][codes[seen_nodes, group]]
        difference = np.asarray(targets[:, group, :] - decoded, dtype=np.float64)
        total += float(np.sum(difference * difference))
        dimensions += int(difference.shape[0])
    return total / float(max(1, dimensions))


def main() -> None:
    args = parse_args()
    if (
        args.batch_size <= 0
        or args.innovation_batch_size <= 0
        or args.max_assignment_score_elements <= 0
        or args.topk <= 0
        or args.rank_max <= args.topk
        or args.boundary_window <= 0
        or args.effective_dimension < 0.0
    ):
        raise ValueError("invalid shrinkage terminal-JI configuration")
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
        raise ValueError("shrinkage trainer requires the deployed 8x8-bit JI model")

    packed = read_side_codes(args.initial_side_codes, initial)
    codes = unpack_all_codes(packed, initial)
    del packed
    original_codes = codes.copy()
    primary_queries = fbin_memmap(primary.query_fbin)
    phi = torch.as_tensor(query_features(primary_queries, initial), device=device)
    books = active_codebooks(initial)
    books_t = torch.as_tensor(books, device=device)

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

    d_eff = (
        float(args.effective_dimension)
        if args.effective_dimension > 0.0
        else participation_dimension(initial.innovation_eigenvalues)
    )
    n_eff = effective_query_counts(train_events, primary.node_count)
    if args.alpha_mode == "adaptive":
        alpha_all = n_eff / (n_eff + d_eff)
    elif args.alpha_mode == "zero":
        alpha_all = np.zeros_like(n_eff)
    else:
        alpha_all = np.ones_like(n_eff)
    seen_alpha = alpha_all[seen_nodes]
    seen_n_eff = n_eff[seen_nodes]
    target_full = innovation_targets(
        primary, initial, seen_nodes, args.innovation_batch_size, device
    )
    widths = np.asarray(initial.group_dims, dtype=np.int64)
    if not np.all(widths == widths[0]):
        raise ValueError("shrinkage trainer currently requires equal-width groups")
    seen_targets = np.stack(
        [target_full[:, initial.indices(group)] for group in range(initial.groups)],
        axis=1,
    )
    del target_full

    initial_metrics = {
        "stage": "input-a1d",
        **evaluate(
            validation, validation_events, codes, phi, books, args, device
        ),
    }
    train_before = weighted_mse(
        train_events,
        predict_events(
            train_events, codes, phi, books_t, args.batch_size, device
        ),
    )
    global_before = global_distortion(seen_targets, codes, seen_nodes, books)

    assignment = reencode_with_shrinkage(
        train_events,
        codes,
        phi,
        books_t,
        seen_nodes,
        seen_targets,
        alpha_all,
        args.batch_size,
        args.max_assignment_score_elements,
        device,
    )
    candidate_metrics = {
        "stage": "adaptive-shrinkage-assignment",
        **assignment,
        **evaluate(
            validation, validation_events, codes, phi, books, args, device
        ),
    }
    validation_selected = is_better(candidate_metrics, initial_metrics)
    deployed = validation_selected or args.checkpoint_policy == "always"
    candidate_metrics["selected_by_validation"] = validation_selected
    candidate_codes = codes[seen_nodes].copy()
    global_candidate = global_distortion(seen_targets, codes, seen_nodes, books)
    if not deployed:
        codes[seen_nodes] = original_codes[seen_nodes]

    final_metrics = evaluate(
        validation, validation_events, codes, phi, books, args, device
    )
    changed_by_group = np.count_nonzero(
        codes[seen_nodes] != original_codes[seen_nodes], axis=0
    ).astype(np.int64)
    changed_nodes = int(
        np.count_nonzero(
            np.any(codes[seen_nodes] != original_codes[seen_nodes], axis=1)
        )
    )
    output_model = make_model(
        initial, books, codes, "terminal-adaptive-shrinkage-reencode-a1f"
    )
    write_model(args.out_model, output_model)
    write_fixed_side(args.out_side_codes, output_model, codes)

    summary = {
        "version": 1,
        "method": "terminal_adaptive_shrinkage_joint_innovation",
        "primary": primary.name,
        "representation": "one fixed 8x8-bit code per database node",
        "budget_bits_per_node": 64,
        "codebooks_frozen": True,
        "alpha_rule": args.alpha_mode,
        "effective_query_count": (
            "Kish count after aggregating repeated beam events for each node-query"
        ),
        "effective_dimension": d_eff,
        "checkpoint_selection": (
            "validation fixed-pool Recall; boundary-AUC inversion tie-break; "
            "A1d input remains a valid fallback"
        ),
        "unseen_node_policy": "retain incoming A1d code indices",
        "train": train_metadata,
        "validation": validation_metadata,
        "seen_training_nodes": int(seen_nodes.size),
        "seen_training_fraction": float(seen_nodes.size) / float(primary.node_count),
        "alpha_seen_quantiles": {
            str(q): float(np.quantile(seen_alpha, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "effective_query_count_seen_quantiles": {
            str(q): float(np.quantile(seen_n_eff, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 1.0)
        },
        "candidate_changed_nodes_by_group": assignment["changed_nodes_by_group"],
        "candidate_changed_nodes": int(
            np.count_nonzero(
                np.any(candidate_codes != original_codes[seen_nodes], axis=1)
            )
        ),
        "selected": deployed,
        "selected_by_validation": validation_selected,
        "checkpoint_policy": args.checkpoint_policy,
        "changed_nodes_by_group": changed_by_group.tolist(),
        "changed_nodes": changed_nodes,
        "train_paired_mse_before": train_before,
        "train_paired_mse_candidate": assignment[
            "train_paired_mse_after_assignment"
        ],
        "global_distortion_before": global_before,
        "global_distortion_candidate": global_candidate,
        "initial_validation": initial_metrics,
        "candidate_validation": candidate_metrics,
        "final_validation": final_metrics,
        "files": {
            "model": os.path.abspath(args.out_model),
            "side_codes": os.path.abspath(args.out_side_codes),
        },
        "config": vars(args),
    }
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(
        f"[terminal-shrinkage] selected={deployed} "
        f"validation_selected={validation_selected} D_eff={d_eff:.6g}"
    )
    print(f"wrote shrinkage terminal JI model: {args.out_model}")
    print(f"wrote shrinkage terminal JI side codes: {args.out_side_codes}")
    print(f"wrote shrinkage terminal JI summary: {args.out_summary}")


if __name__ == "__main__":
    main()
