#!/usr/bin/env python3
"""Train fixed-code terminal JI codebooks with rank-invariant objectives.

Two objectives are supported:

* ``centered`` removes the query/pool-common score error before measuring MSE.
* ``multi-boundary-hinge`` samples every possible rank cut in expectation and
  penalizes compressed pair margins that fall below their exact margins.

The primary representation, graph, JI transform, group partition, bit budget,
and database-node assignments are frozen.  Only shared codebook vectors move.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Iterator, Sequence

import numpy as np
import torch

from drb_io import fbin_memmap
from fixed_candidate_io import read_visited_csr
from joint_innovation_rpq import read_model, read_side_codes, write_model
from primary_quantizer_manifest import read_manifest
from train_terminal_conditioned_ji import (
    active_codebooks,
    make_model,
    query_features,
    resolve_device,
    unpack_all_codes,
    write_fixed_side,
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
    parser.add_argument(
        "--objective",
        choices=("centered", "multi-boundary-hinge"),
        required=True,
    )
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-side-codes", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--rank-max", type=int, default=100)
    parser.add_argument("--boundary-window", type=int, default=5)
    parser.add_argument("--boundaries-per-pool", type=int, default=4)
    parser.add_argument("--validation-boundary-pools", type=int, default=12000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--force-pool-cache", action="store_true")
    return parser.parse_args()


def signature(path: str) -> dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


class Pools:
    def __init__(
        self,
        query_ids: np.ndarray,
        beams: np.ndarray,
        offsets: np.ndarray,
        node_ids: np.ndarray,
        targets: np.ndarray,
        exact_scores: np.ndarray,
        primary_scores: np.ndarray,
    ) -> None:
        self.query_ids = np.asarray(query_ids, dtype=np.int64)
        self.beams = np.asarray(beams, dtype=np.int64)
        self.offsets = np.asarray(offsets, dtype=np.int64)
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.targets = np.asarray(targets, dtype=np.float32)
        self.exact_scores = np.asarray(exact_scores, dtype=np.float32)
        self.primary_scores = np.asarray(primary_scores, dtype=np.float32)
        pools = self.query_ids.size
        events = self.node_ids.size
        if not (
            pools > 0
            and self.beams.shape == (pools,)
            and self.offsets.shape == (pools + 1,)
            and int(self.offsets[0]) == 0
            and int(self.offsets[-1]) == events
            and np.all(self.offsets[1:] > self.offsets[:-1])
            and self.targets.shape == (events,)
            and self.exact_scores.shape == (events,)
            and self.primary_scores.shape == (events,)
        ):
            raise ValueError("invalid terminal pool cache")

    @property
    def pool_count(self) -> int:
        return int(self.query_ids.size)

    @property
    def event_count(self) -> int:
        return int(self.node_ids.size)

    def bounds(self, pool: int) -> tuple[int, int]:
        return int(self.offsets[pool]), int(self.offsets[pool + 1])


def cache_fingerprint(
    traces: Sequence[str],
    exact_base: str,
    exact_queries: str,
    reconstruction: str,
    primary_queries: str,
    residual_mean: np.ndarray,
) -> str:
    payload = {
        "version": 2,
        "traces": [signature(value) for value in traces],
        "exact_base": signature(exact_base),
        "exact_queries": signature(exact_queries),
        "reconstruction": signature(reconstruction),
        "primary_queries": signature(primary_queries),
        "residual_mean": hashlib.sha256(
            np.ascontiguousarray(residual_mean, dtype="<f4").tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


def save_pools(path: Path, pools: Pools, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "query_ids.npy", pools.query_ids.astype(np.uint32))
    np.save(path / "beams.npy", pools.beams.astype(np.uint32))
    np.save(path / "offsets.npy", pools.offsets.astype(np.uint64))
    np.save(path / "node_ids.npy", pools.node_ids.astype(np.uint32))
    np.save(path / "targets.npy", pools.targets)
    np.save(path / "exact_scores.npy", pools.exact_scores)
    np.save(path / "primary_scores.npy", pools.primary_scores)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_pools(path: Path) -> tuple[Pools, dict]:
    pools = Pools(
        np.load(path / "query_ids.npy", mmap_mode="r"),
        np.load(path / "beams.npy", mmap_mode="r"),
        np.load(path / "offsets.npy", mmap_mode="r"),
        np.load(path / "node_ids.npy", mmap_mode="r"),
        np.load(path / "targets.npy", mmap_mode="r"),
        np.load(path / "exact_scores.npy", mmap_mode="r"),
        np.load(path / "primary_scores.npy", mmap_mode="r"),
    )
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return pools, metadata


def build_pools(
    traces: Sequence[str],
    exact_base_path: str,
    exact_queries_path: str,
    reconstruction_path: str,
    primary_queries_path: str,
    residual_mean: np.ndarray,
    node_count: int,
) -> tuple[Pools, dict]:
    exact_base = fbin_memmap(exact_base_path)
    exact_queries = fbin_memmap(exact_queries_path)
    reconstruction = fbin_memmap(reconstruction_path)
    primary_queries = fbin_memmap(primary_queries_path)
    loaded = [read_visited_csr(path) for path in traces]
    starts = {value[0] for value in loaded}
    rows = {len(value[1]) - 1 for value in loaded}
    if len(starts) != 1 or len(rows) != 1:
        raise ValueError("all traces in one split must cover identical queries")
    query_start = next(iter(starts))
    query_count = next(iter(rows))
    if query_start + query_count > exact_queries.shape[0]:
        raise ValueError("exact queries do not cover terminal pools")
    if query_start + query_count > primary_queries.shape[0]:
        raise ValueError("primary queries do not cover terminal pools")

    pool_queries: list[int] = []
    pool_beams: list[int] = []
    offsets = [0]
    node_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    exact_chunks: list[np.ndarray] = []
    primary_chunks: list[np.ndarray] = []
    for local in range(query_count):
        raw_pools: list[np.ndarray] = []
        for _, trace_offsets, trace_ids in loaded:
            begin, end = int(trace_offsets[local]), int(trace_offsets[local + 1])
            nodes = np.unique(np.asarray(trace_ids[begin:end], dtype=np.int64))
            if nodes.size == 0 or int(np.max(nodes)) >= node_count:
                raise ValueError("invalid node ID in retained pool")
            raw_pools.append(nodes)
        union = np.unique(np.concatenate(raw_pools))
        qid = query_start + local
        exact_q = np.asarray(exact_queries[qid], dtype=np.float32)
        primary_q = np.asarray(primary_queries[qid], dtype=np.float32)
        union_exact = np.asarray(exact_base[union], dtype=np.float32) @ exact_q
        union_primary = np.asarray(reconstruction[union], dtype=np.float32) @ primary_q
        lookup = {int(node): index for index, node in enumerate(union.tolist())}
        mean_bias = float(primary_q @ residual_mean)
        for nodes in raw_pools:
            positions = np.fromiter(
                (lookup[int(node)] for node in nodes), dtype=np.int64, count=nodes.size
            )
            exact = union_exact[positions]
            primary = union_primary[positions]
            order = np.argsort(-exact, kind="stable")
            nodes = nodes[order]
            exact = exact[order]
            primary = primary[order]
            pool_queries.append(qid)
            pool_beams.append(int(nodes.size))
            node_chunks.append(nodes.astype(np.uint32))
            exact_chunks.append(exact.astype(np.float32))
            primary_chunks.append(primary.astype(np.float32))
            target_chunks.append((exact - primary - mean_bias).astype(np.float32))
            offsets.append(offsets[-1] + int(nodes.size))
        if (local + 1) % 1000 == 0 or local + 1 == query_count:
            print(
                f"[terminal-pools] {local + 1}/{query_count} queries; "
                f"events={offsets[-1]}",
                flush=True,
            )
    pools = Pools(
        np.asarray(pool_queries, dtype=np.int64),
        np.asarray(pool_beams, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        np.concatenate(node_chunks),
        np.concatenate(target_chunks),
        np.concatenate(exact_chunks),
        np.concatenate(primary_chunks),
    )
    metadata = {
        "version": 2,
        "query_start": query_start,
        "queries": query_count,
        "beams": sorted(set(pool_beams)),
        "pools": pools.pool_count,
        "events": pools.event_count,
        "ordering": "exact-score descending within every query x beam pool",
        "pool_weighting": "equal query x equal beam; uniform candidates within pool",
    }
    return pools, metadata


def get_or_build_pools(
    path: Path,
    traces: Sequence[str],
    exact_base: str,
    exact_queries: str,
    reconstruction: str,
    primary_queries: str,
    residual_mean: np.ndarray,
    node_count: int,
    force: bool,
) -> tuple[Pools, dict]:
    fingerprint = cache_fingerprint(
        traces,
        exact_base,
        exact_queries,
        reconstruction,
        primary_queries,
        residual_mean,
    )
    if not force and (path / "metadata.json").exists():
        pools, metadata = load_pools(path)
        if metadata.get("fingerprint") == fingerprint:
            print(f"[reuse] terminal pool cache: {path}")
            return pools, metadata
    pools, metadata = build_pools(
        traces,
        exact_base,
        exact_queries,
        reconstruction,
        primary_queries,
        residual_mean,
        node_count,
    )
    metadata["fingerprint"] = fingerprint
    save_pools(path, pools, metadata)
    return load_pools(path)


def pool_batches(
    pools: Pools, order: np.ndarray, max_events: int
) -> Iterator[np.ndarray]:
    selected: list[int] = []
    events = 0
    for pool in order.tolist():
        begin, end = pools.bounds(pool)
        count = end - begin
        if selected and events + count > max_events:
            yield np.asarray(selected, dtype=np.int64)
            selected = []
            events = 0
        selected.append(pool)
        events += count
    if selected:
        yield np.asarray(selected, dtype=np.int64)


def materialize_batch(pools: Pools, selected: np.ndarray) -> dict[str, np.ndarray]:
    event_indices: list[np.ndarray] = []
    pool_index: list[np.ndarray] = []
    lengths = []
    for local, pool in enumerate(selected.tolist()):
        begin, end = pools.bounds(pool)
        event_indices.append(np.arange(begin, end, dtype=np.int64))
        pool_index.append(np.full(end - begin, local, dtype=np.int64))
        lengths.append(end - begin)
    events = np.concatenate(event_indices)
    return {
        "events": events,
        "pool_index": np.concatenate(pool_index),
        "lengths": np.asarray(lengths, dtype=np.int64),
        "query_ids": pools.query_ids[selected],
    }


def predict_batch(
    batch: dict[str, np.ndarray],
    pools: Pools,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    events = batch["events"]
    qids = torch.as_tensor(
        batch["query_ids"][batch["pool_index"]], device=device
    )
    labels = torch.as_tensor(codes[pools.node_ids[events]], device=device)
    groups = torch.arange(books.shape[0], device=device)[None, :]
    selected = books[groups, labels]
    return torch.sum(selected * phi[qids], dim=(1, 2))


def centered_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    pool_index: torch.Tensor,
    pool_count: int,
) -> torch.Tensor:
    error = prediction - target
    sums = torch.zeros(pool_count, dtype=error.dtype, device=error.device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, pool_index, error)
    counts.index_add_(0, pool_index, torch.ones_like(error))
    centered = error - (sums / counts)[pool_index]
    squares = torch.zeros_like(sums)
    squares.index_add_(0, pool_index, torch.square(centered))
    return torch.mean(squares / counts)


def sampled_boundary_pairs(
    lengths: np.ndarray,
    rank_max: int,
    window: int,
    boundaries_per_pool: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    left: list[int] = []
    right: list[int] = []
    owners: list[int] = []
    offset = 0
    for owner, raw_length in enumerate(lengths.tolist()):
        length = min(int(raw_length), rank_max)
        candidates = np.arange(1, length, dtype=np.int64)
        count = min(boundaries_per_pool, candidates.size)
        boundaries = rng.choice(candidates, size=count, replace=False)
        for boundary in boundaries.tolist():
            inside = np.arange(max(0, boundary - window), boundary, dtype=np.int64)
            outside = np.arange(boundary, min(length, boundary + window), dtype=np.int64)
            l, r = np.meshgrid(inside, outside, indexing="ij")
            left.extend((offset + l.reshape(-1)).tolist())
            right.extend((offset + r.reshape(-1)).tolist())
            owners.extend([owner] * int(l.size))
        offset += int(raw_length)
    return (
        np.asarray(left, dtype=np.int64),
        np.asarray(right, dtype=np.int64),
        np.asarray(owners, dtype=np.int64),
    )


def hinge_loss(
    prediction: torch.Tensor,
    primary_scores: torch.Tensor,
    exact_scores: torch.Tensor,
    lengths: np.ndarray,
    rank_max: int,
    window: int,
    boundaries_per_pool: int,
    rng: np.random.Generator,
    device: torch.device,
) -> torch.Tensor:
    left, right, owners = sampled_boundary_pairs(
        lengths, rank_max, window, boundaries_per_pool, rng
    )
    left_t = torch.as_tensor(left, device=device)
    right_t = torch.as_tensor(right, device=device)
    owner_t = torch.as_tensor(owners, device=device)
    approximate = primary_scores + prediction
    exact_gap = exact_scores[left_t] - exact_scores[right_t]
    predicted_gap = approximate[left_t] - approximate[right_t]
    violation = torch.square(torch.relu(exact_gap - predicted_gap))
    sums = torch.zeros(len(lengths), dtype=violation.dtype, device=device)
    counts = torch.zeros_like(sums)
    sums.index_add_(0, owner_t, violation)
    counts.index_add_(0, owner_t, torch.ones_like(violation))
    return torch.mean(sums / torch.clamp_min(counts, 1.0))


def predict_all(
    pools: Pools,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = np.empty(pools.event_count, dtype=np.float32)
    order = np.arange(pools.pool_count, dtype=np.int64)
    with torch.no_grad():
        for selected in pool_batches(pools, order, batch_size):
            batch = materialize_batch(pools, selected)
            prediction = predict_batch(batch, pools, codes, phi, books, device)
            output[batch["events"]] = prediction.cpu().numpy()
    return output


def boundary_weight_matrix(length: int, window: int) -> np.ndarray:
    weight = np.zeros((length, length), dtype=np.int32)
    for left in range(length):
        for right in range(left + 1, length):
            low = max(1, left + 1, right - window + 1)
            high = min(length - 1, left + window, right)
            if high >= low:
                weight[left, right] = high - low + 1
    return weight


def ranking_metrics(
    pools: Pools,
    correction: np.ndarray,
    topk: int,
    rank_max: int,
    window: int,
    boundary_pool_limit: int,
) -> dict[str, float]:
    hits = 0
    decisions = 0
    boundary_bad = 0
    boundary_total = 0
    matrices: dict[int, np.ndarray] = {}
    # Evenly cover every beam instead of taking the first pools only.
    selected_boundary = np.zeros(pools.pool_count, dtype=bool)
    if boundary_pool_limit > 0:
        unique_beams = np.unique(pools.beams)
        quota = max(1, boundary_pool_limit // max(1, unique_beams.size))
        for beam in unique_beams:
            indices = np.flatnonzero(pools.beams == beam)
            if indices.size > quota:
                positions = np.linspace(0, indices.size - 1, quota, dtype=np.int64)
                indices = indices[positions]
            selected_boundary[indices] = True
    else:
        selected_boundary[:] = True
    for pool in range(pools.pool_count):
        begin, end = pools.bounds(pool)
        length = end - begin
        k = min(topk, length)
        approximate = pools.primary_scores[begin:end] + correction[begin:end]
        predicted_top = np.argpartition(-approximate, k - 1)[:k]
        hits += int(np.sum(predicted_top < k))
        decisions += k
        if selected_boundary[pool]:
            width = min(rank_max, length)
            weights = matrices.setdefault(
                width, boundary_weight_matrix(width, window)
            )
            score = approximate[:width]
            inverted = score[:, None] <= score[None, :]
            boundary_bad += int(np.sum(inverted * weights))
            boundary_total += int(np.sum(weights))
    return {
        "fixed_pool_recall": hits / float(decisions),
        "boundary_auc_inversion": boundary_bad / float(boundary_total),
        "boundary_pairs_weighted": boundary_total,
        "boundary_pools": int(np.sum(selected_boundary)),
    }


def objective_value(
    objective: str,
    pools: Pools,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    batch_size: int,
    rank_max: int,
    window: int,
    boundaries_per_pool: int,
    seed: int,
    device: torch.device,
) -> float:
    rng = np.random.default_rng(seed)
    total = 0.0
    count = 0
    order = np.arange(pools.pool_count, dtype=np.int64)
    with torch.no_grad():
        for selected in pool_batches(pools, order, batch_size):
            batch = materialize_batch(pools, selected)
            prediction = predict_batch(batch, pools, codes, phi, books, device)
            events = batch["events"]
            target = torch.as_tensor(pools.targets[events], device=device)
            pool_index = torch.as_tensor(batch["pool_index"], device=device)
            if objective == "centered":
                loss = centered_loss(prediction, target, pool_index, len(selected))
            else:
                loss = hinge_loss(
                    prediction,
                    torch.as_tensor(pools.primary_scores[events], device=device),
                    torch.as_tensor(pools.exact_scores[events], device=device),
                    batch["lengths"],
                    rank_max,
                    window,
                    boundaries_per_pool,
                    rng,
                    device,
                )
            total += float(loss.item()) * len(selected)
            count += len(selected)
    return total / float(count)


def is_better(candidate: dict, best: dict) -> bool:
    recall_delta = candidate["fixed_pool_recall"] - best["fixed_pool_recall"]
    if recall_delta > 1e-12:
        return True
    if abs(recall_delta) <= 1e-12:
        return candidate["boundary_auc_inversion"] < best["boundary_auc_inversion"]
    return False


def main() -> None:
    args = parse_args()
    if (
        args.epochs < 0
        or args.batch_size <= 0
        or args.learning_rate <= 0.0
        or args.topk <= 0
        or args.rank_max <= args.topk
        or args.boundary_window <= 0
        or args.boundaries_per_pool <= 0
    ):
        raise ValueError("invalid terminal ranking training configuration")
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
        raise ValueError("ranking-objective experiment requires deployed 8x8-bit JI")
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

    initial_books = active_codebooks(initial)
    books = torch.nn.Parameter(torch.as_tensor(initial_books, device=device).clone())
    optimizer = torch.optim.Adam([books], lr=args.learning_rate)
    rng = np.random.default_rng(args.seed)
    initial_correction = predict_all(
        validation, codes, phi, books.detach(), args.batch_size, device
    )
    initial_rank = ranking_metrics(
        validation,
        initial_correction,
        args.topk,
        args.rank_max,
        args.boundary_window,
        args.validation_boundary_pools,
    )
    initial_objective = objective_value(
        args.objective,
        validation,
        codes,
        phi,
        books.detach(),
        args.batch_size,
        args.rank_max,
        args.boundary_window,
        args.boundaries_per_pool,
        args.seed,
        device,
    )
    initial_metrics = {"epoch": 0, "validation_objective": initial_objective, **initial_rank}
    history = [initial_metrics]
    best_metrics = initial_metrics
    best_books = initial_books.copy()
    stale = 0

    for epoch in range(args.epochs):
        order = rng.permutation(train.pool_count)
        total_loss = 0.0
        total_pools = 0
        for selected_pools in pool_batches(train, order, args.batch_size):
            batch = materialize_batch(train, selected_pools)
            prediction = predict_batch(batch, train, codes, phi, books, device)
            events = batch["events"]
            target = torch.as_tensor(train.targets[events], device=device)
            pool_index = torch.as_tensor(batch["pool_index"], device=device)
            if args.objective == "centered":
                loss = centered_loss(
                    prediction, target, pool_index, len(selected_pools)
                )
            else:
                loss = hinge_loss(
                    prediction,
                    torch.as_tensor(train.primary_scores[events], device=device),
                    torch.as_tensor(train.exact_scores[events], device=device),
                    batch["lengths"],
                    args.rank_max,
                    args.boundary_window,
                    args.boundaries_per_pool,
                    rng,
                    device,
                )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(selected_pools)
            total_pools += len(selected_pools)

        correction = predict_all(
            validation, codes, phi, books.detach(), args.batch_size, device
        )
        rank = ranking_metrics(
            validation,
            correction,
            args.topk,
            args.rank_max,
            args.boundary_window,
            args.validation_boundary_pools,
        )
        validation_objective = objective_value(
            args.objective,
            validation,
            codes,
            phi,
            books.detach(),
            args.batch_size,
            args.rank_max,
            args.boundary_window,
            args.boundaries_per_pool,
            args.seed,
            device,
        )
        metrics = {
            "epoch": epoch + 1,
            "train_objective": total_loss / float(total_pools),
            "validation_objective": validation_objective,
            **rank,
        }
        selected = is_better(metrics, best_metrics)
        metrics["selected"] = selected
        history.append(metrics)
        print(f"[terminal-ranking-fit] {json.dumps(metrics, sort_keys=True)}", flush=True)
        if selected:
            best_metrics = metrics
            best_books = books.detach().cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if args.patience > 0 and stale >= args.patience:
                break

    variant = (
        "terminal-centered-fixed-codes-a1c"
        if args.objective == "centered"
        else "terminal-multi-boundary-hinge-fixed-codes-a1d"
    )
    output_model = make_model(initial, best_books, codes, variant)
    write_model(args.out_model, output_model)
    write_fixed_side(args.out_side_codes, output_model, codes)
    summary = {
        "version": 1,
        "method": "terminal_ranking_objective_joint_innovation",
        "objective": args.objective,
        "primary": primary.name,
        "representation": "fixed existing 8x8-bit code per database node",
        "checkpoint_selection": "maximum validation fixed-pool Recall; boundary-AUC inversion tie-break",
        "train": train_metadata,
        "validation": validation_metadata,
        "initial_validation": initial_metrics,
        "selected_validation": best_metrics,
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
    print(f"wrote terminal ranking JI model: {args.out_model}")
    print(f"wrote terminal ranking summary: {args.out_summary}")


if __name__ == "__main__":
    main()
