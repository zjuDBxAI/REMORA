#!/usr/bin/env python3
"""Compute exact query-to-base KNN under the deployed PQ+RPQ scorer.

This tool is intended for a sampled RoarGraph construction workload. It scans
all compressed database nodes but never loads FP32 base vectors. Query chunks
are scored against ``x_hat + r_tilde`` and the resulting ID matrix can be fed
directly to ``test_build_roargraph --learn_base_nn_path``.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import time
from pathlib import Path

import numpy as np
import torch

from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from fixed_candidate_io import write_float_matrix, write_id_matrix
from group_residual_vq import resolve_device
from quantized_roargraph import validate_quantized_representation
from residual_pq_baseline_io import (
    read_residual_pq_model,
    read_residual_pq_side_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--side-codes", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--out-ids", required=True)
    parser.add_argument(
        "--out-query-fbin",
        default="",
        help=(
            "optional query slice written for the builder; may be omitted only "
            "when the requested range is the complete input query file"
        ),
    )
    parser.add_argument("--out-scores", default="")
    parser.add_argument("--out-summary", default="")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=10000)
    parser.add_argument("--topk", type=int, default=100)
    parser.add_argument("--query-batch-size", type=int, default=256)
    parser.add_argument("--base-batch-size", type=int, default=65536)
    parser.add_argument("--solver-device", default="cuda")
    parser.add_argument("--allow-tf32", action="store_true")
    parser.add_argument("--log-interval", type=int, default=1000000)
    return parser.parse_args()


def _write_query_slice(path: str, queries: np.ndarray) -> None:
    write_float_matrix(path, np.asarray(queries, dtype=np.float32))


def _torch_reconstruct(
    primary_codes: np.ndarray,
    residual_codes: np.ndarray,
    centroids: torch.Tensor,
    prototypes: torch.Tensor,
    residual_inverse_transform: torch.Tensor | None,
    additive: bool,
    conditional_basis: torch.Tensor | None,
    conditional_embeddings: torch.Tensor | None,
    device: torch.device,
) -> torch.Tensor:
    codes = torch.as_tensor(
        np.array(primary_codes, dtype=np.uint8, copy=True, order="C"),
        dtype=torch.long,
        device=device,
    )
    side = torch.as_tensor(
        np.array(residual_codes, dtype=np.uint8, copy=True, order="C"),
        dtype=torch.long,
        device=device,
    )
    subspaces = torch.arange(centroids.shape[0], device=device)[None, :]
    groups = torch.arange(prototypes.shape[0], device=device)[None, :]
    primary = centroids[subspaces, codes].reshape(codes.shape[0], -1)
    selected = prototypes[groups, side]
    correction = (
        torch.sum(selected, dim=1)
        if additive
        else selected.reshape(side.shape[0], -1)
    )
    if conditional_basis is not None and conditional_embeddings is not None:
        primary_groups = torch.arange(codes.shape[1], device=device)[None, :]
        context = conditional_embeddings[primary_groups, codes].sum(dim=1)
        correction += context @ conditional_basis.T
    if residual_inverse_transform is not None:
        correction = correction @ residual_inverse_transform
    return primary + correction


def exact_quantized_knn(
    queries_t: np.ndarray,
    codes: np.ndarray,
    side_codes: np.ndarray,
    pq_model,
    residual_model,
    topk: int,
    query_batch_size: int,
    base_batch_size: int,
    device: torch.device,
    log_interval: int,
    return_scores: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return exact top-k IDs/scores over all quantized reconstructions."""
    query_tensor = torch.as_tensor(
        np.ascontiguousarray(queries_t, dtype=np.float32), device=device
    )
    centroids = torch.as_tensor(
        np.asarray(pq_model.centroids, dtype=np.float32), device=device
    )
    prototypes = torch.as_tensor(
        np.asarray(residual_model.prototypes, dtype=np.float32), device=device
    )
    residual_inverse_transform = (
        torch.as_tensor(
            np.asarray(
                residual_model.residual_inverse_transform, dtype=np.float32
            ),
            device=device,
        )
        if residual_model.has_residual_transform
        else None
    )
    conditional_basis = (
        torch.as_tensor(
            np.asarray(residual_model.conditional_basis, dtype=np.float32),
            device=device,
        )
        if residual_model.has_primary_conditioning
        else None
    )
    conditional_embeddings = (
        torch.as_tensor(
            np.asarray(residual_model.conditional_embeddings, dtype=np.float32),
            device=device,
        )
        if residual_model.has_primary_conditioning
        else None
    )
    query_count = int(query_tensor.shape[0])
    best_scores = torch.full(
        (query_count, topk), -torch.inf, dtype=torch.float32, device=device
    )
    best_ids = torch.full(
        (query_count, topk), -1, dtype=torch.int64, device=device
    )
    nodes = int(codes.shape[0])
    with torch.no_grad():
        for base_start in range(0, nodes, base_batch_size):
            base_end = min(nodes, base_start + base_batch_size)
            reconstructed = _torch_reconstruct(
                codes[base_start:base_end],
                side_codes[base_start:base_end],
                centroids,
                prototypes,
                residual_inverse_transform,
                residual_model.quantizer_kind == "additive",
                conditional_basis,
                conditional_embeddings,
                device,
            )
            local_k = min(topk, base_end - base_start)
            for query_start in range(0, query_count, query_batch_size):
                query_end = min(query_count, query_start + query_batch_size)
                scores = query_tensor[query_start:query_end] @ reconstructed.T
                local_scores, local_indices = torch.topk(
                    scores, k=local_k, dim=1, largest=True, sorted=True
                )
                local_ids = local_indices.to(torch.int64) + base_start
                combined_scores = torch.cat(
                    (best_scores[query_start:query_end], local_scores), dim=1
                )
                combined_ids = torch.cat(
                    (best_ids[query_start:query_end], local_ids), dim=1
                )
                selected_scores, selected = torch.topk(
                    combined_scores, k=topk, dim=1, largest=True, sorted=True
                )
                best_scores[query_start:query_end] = selected_scores
                best_ids[query_start:query_end] = torch.gather(
                    combined_ids, 1, selected
                )
            if log_interval > 0 and (
                base_end % log_interval < base_batch_size or base_end == nodes
            ):
                print(f"[quantized-knn] scanned {base_end}/{nodes}", flush=True)
    ids = best_ids.cpu().numpy().astype(np.uint32)
    scores = (
        best_scores.cpu().numpy().astype(np.float32)
        if return_scores
        else None
    )
    return ids, scores


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0
        or args.query_limit <= 0
        or args.topk <= 0
        or args.query_batch_size <= 0
        or args.base_batch_size <= 0
        or args.log_interval < 0
    ):
        raise ValueError("invalid query/KNN batch configuration")
    device = resolve_device(args.solver_device)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = bool(args.allow_tf32)
        torch.cuda.reset_peak_memory_stats(device)
    pq_model = read_model(args.pq_model)
    residual_model = read_residual_pq_model(args.residual_model)
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    side_codes, _ = read_residual_pq_side_codes(args.side_codes, residual_model)
    validate_quantized_representation(
        pq_model, residual_model, codes, side_codes
    )
    if args.topk > int(pq_model.node_count):
        raise ValueError("topk exceeds database size")
    all_queries = fbin_memmap(args.query_fbin)
    query_end = args.query_start + args.query_limit
    if query_end > all_queries.shape[0] or all_queries.shape[1] != pq_model.dim:
        raise ValueError("requested query range does not match the model")
    raw_queries = np.asarray(
        all_queries[args.query_start:query_end], dtype=np.float32
    )
    if args.out_query_fbin:
        _write_query_slice(args.out_query_fbin, raw_queries)
        sampled_query_path = args.out_query_fbin
    elif args.query_start == 0 and query_end == all_queries.shape[0]:
        sampled_query_path = args.query_fbin
    else:
        raise ValueError(
            "--out-query-fbin is required unless the complete query file is used"
        )
    queries_t = transform_vectors(raw_queries, pq_model)

    started = time.perf_counter()
    ids, scores = exact_quantized_knn(
        queries_t,
        codes,
        side_codes,
        pq_model,
        residual_model,
        args.topk,
        args.query_batch_size,
        args.base_batch_size,
        device,
        args.log_interval,
        return_scores=bool(args.out_scores),
    )
    seconds = time.perf_counter() - started
    write_id_matrix(args.out_ids, ids)
    if args.out_scores:
        assert scores is not None
        write_float_matrix(args.out_scores, scores)
    summary_path = (
        Path(args.out_summary)
        if args.out_summary
        else Path(args.out_ids).with_suffix(".summary.json")
    )
    summary = {
        "version": 1,
        "method": "exact_all_database_pq_rpq_query_base_knn",
        "score": "q_transformed_dot_(primary_pq_plus_residual_pq_reconstruction)",
        "queries": int(ids.shape[0]),
        "query_range": [args.query_start, query_end],
        "database_nodes": int(pq_model.node_count),
        "topk": args.topk,
        "assignment_metric": str(residual_model.assignment_metric),
        "metric_lambda": float(residual_model.metric_lambda),
        "solver_device": str(device),
        "tf32": bool(args.allow_tf32),
        "seconds": seconds,
        "query_base_scores_per_second": (
            int(ids.shape[0]) * int(pq_model.node_count)
        ) / max(seconds, 1e-12),
        "cpu_peak_rss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        * 1024,
        "gpu_peak_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else 0
        ),
        "materialized_database_matrix": False,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "residual_model": os.path.abspath(args.residual_model),
            "side_codes": os.path.abspath(args.side_codes),
            "queries": os.path.abspath(args.query_fbin),
            "sampled_queries": os.path.abspath(sampled_query_path),
            "knn_ids": os.path.abspath(args.out_ids),
        },
        "config": vars(args),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote quantized query-base KNN: {args.out_ids}")
    print(f"wrote sampled build queries: {args.out_query_fbin}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
