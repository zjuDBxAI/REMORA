#!/usr/bin/env python3
"""Train a 64-bit JI terminal refiner on real query--candidate events.

The database code remains query independent: every node owns one fixed code.
Only the distribution used to fit the shared codebooks (and, optionally, the
fixed node assignments) changes from the global Cartesian query/base model to
the retained candidate pools produced by primary-only graph traversal.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from dataclasses import replace
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from drb_io import fbin_memmap
from fixed_candidate_io import read_visited_csr
from joint_innovation_rpq import (
    JointInnovationModel,
    compute_model_id,
    initialize_side_codes,
    pack_codes,
    read_model,
    read_side_codes,
    unpack_group,
    write_model,
)
from primary_quantizer_manifest import read_manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--primary-manifest", required=True)
    parser.add_argument("--exact-base", required=True)
    parser.add_argument("--exact-queries", required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument("--initial-side-codes", required=True)
    parser.add_argument("--train-retained", action="append", required=True)
    parser.add_argument("--validation-retained", action="append", required=True)
    parser.add_argument("--event-cache", required=True)
    parser.add_argument("--mode", choices=("fixed-codes", "reencode"), required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-side-codes")
    parser.add_argument("--out-overrides")
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--reencode-sweeps", type=int, default=1)
    parser.add_argument("--retrain-epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=65536)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--patience", type=int, default=3)
    parser.add_argument("--max-assignment-score-elements", type=int, default=2_000_000)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--force-event-cache", action="store_true")
    return parser.parse_args()


def resolve_device(value: str) -> torch.device:
    requested = value.strip().lower()
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(requested)
    if device.type not in ("cpu", "cuda"):
        raise ValueError("terminal-conditioned trainer supports cpu or cuda")
    return device


def file_signature(path: str) -> dict:
    stat = os.stat(path)
    return {
        "path": os.path.abspath(path),
        "bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def event_fingerprint(
    traces: Sequence[str],
    exact_base: str,
    exact_queries: str,
    reconstruction: str,
    primary_queries: str,
    residual_mean: np.ndarray,
) -> str:
    payload = {
        "version": 1,
        "traces": [file_signature(value) for value in traces],
        "exact_base": file_signature(exact_base),
        "exact_queries": file_signature(exact_queries),
        "reconstruction": file_signature(reconstruction),
        "primary_queries": file_signature(primary_queries),
        "residual_mean_sha256": hashlib.sha256(
            np.ascontiguousarray(residual_mean, dtype="<f4").tobytes()
        ).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()


class Events:
    def __init__(
        self,
        query_ids: np.ndarray,
        node_ids: np.ndarray,
        weights: np.ndarray,
        targets: np.ndarray,
    ) -> None:
        self.query_ids = np.asarray(query_ids, dtype=np.int64)
        self.node_ids = np.asarray(node_ids, dtype=np.int64)
        self.weights = np.asarray(weights, dtype=np.float32)
        self.targets = np.asarray(targets, dtype=np.float32)
        size = self.node_ids.size
        if not (
            self.query_ids.shape == (size,)
            and self.weights.shape == (size,)
            and self.targets.shape == (size,)
            and size > 0
            and np.all(self.weights > 0)
        ):
            raise ValueError("invalid terminal event arrays")

    @property
    def size(self) -> int:
        return int(self.node_ids.size)


def save_events(path: Path, events: Events, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.save(path / "query_ids.npy", events.query_ids.astype(np.uint32))
    np.save(path / "node_ids.npy", events.node_ids.astype(np.uint32))
    np.save(path / "weights.npy", events.weights)
    np.save(path / "targets.npy", events.targets)
    (path / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


def load_events(path: Path) -> tuple[Events, dict]:
    metadata = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    return (
        Events(
            np.load(path / "query_ids.npy", mmap_mode="r"),
            np.load(path / "node_ids.npy", mmap_mode="r"),
            np.load(path / "weights.npy", mmap_mode="r"),
            np.load(path / "targets.npy", mmap_mode="r"),
        ),
        metadata,
    )


def build_events(
    traces: Sequence[str],
    exact_base_path: str,
    exact_queries_path: str,
    reconstruction_path: str,
    primary_queries_path: str,
    residual_mean: np.ndarray,
    node_count: int,
) -> tuple[Events, dict]:
    exact_base = fbin_memmap(exact_base_path)
    exact_queries = fbin_memmap(exact_queries_path)
    reconstruction = fbin_memmap(reconstruction_path)
    primary_queries = fbin_memmap(primary_queries_path)
    if exact_base.shape[0] < node_count or reconstruction.shape[0] != node_count:
        raise ValueError("base/reconstruction does not cover the primary database")
    if exact_base.shape[1] != exact_queries.shape[1]:
        raise ValueError("exact query/base dimension mismatch")
    if reconstruction.shape[1] != primary_queries.shape[1]:
        raise ValueError("primary query/reconstruction dimension mismatch")

    loaded = [read_visited_csr(path) for path in traces]
    starts = {value[0] for value in loaded}
    rows = {len(value[1]) - 1 for value in loaded}
    if len(starts) != 1 or len(rows) != 1:
        raise ValueError("all retained traces in a split must cover the same queries")
    query_start = next(iter(starts))
    query_count = next(iter(rows))
    if query_start + query_count > exact_queries.shape[0]:
        raise ValueError("exact query file does not cover retained traces")
    if query_start + query_count > primary_queries.shape[0]:
        raise ValueError("primary query file does not cover retained traces")

    query_chunks: list[np.ndarray] = []
    node_chunks: list[np.ndarray] = []
    weight_chunks: list[np.ndarray] = []
    target_chunks: list[np.ndarray] = []
    trace_count = float(len(loaded))
    raw_events = 0
    for local in range(query_count):
        accumulated: dict[int, float] = {}
        for _, offsets, ids in loaded:
            begin, end = int(offsets[local]), int(offsets[local + 1])
            pool = np.unique(np.asarray(ids[begin:end], dtype=np.uint32))
            if pool.size == 0:
                raise ValueError("retained candidate pool is empty")
            raw_events += int(pool.size)
            contribution = 1.0 / (trace_count * float(pool.size))
            for node in pool.tolist():
                if int(node) >= node_count:
                    raise ValueError("retained candidate ID is outside database")
                accumulated[int(node)] = accumulated.get(int(node), 0.0) + contribution
        nodes = np.fromiter(accumulated.keys(), dtype=np.int64)
        weights = np.fromiter(accumulated.values(), dtype=np.float32)
        query_id = query_start + local
        exact_q = np.asarray(exact_queries[query_id], dtype=np.float32)
        primary_q = np.asarray(primary_queries[query_id], dtype=np.float32)
        exact_score = np.asarray(exact_base[nodes], dtype=np.float32) @ exact_q
        primary_score = np.asarray(reconstruction[nodes], dtype=np.float32) @ primary_q
        mean_bias = float(primary_q @ residual_mean)
        target = exact_score - primary_score - mean_bias
        query_chunks.append(np.full(nodes.size, query_id, dtype=np.int64))
        node_chunks.append(nodes)
        weight_chunks.append(weights)
        target_chunks.append(np.asarray(target, dtype=np.float32))
        if (local + 1) % 1000 == 0 or local + 1 == query_count:
            print(
                f"[terminal-events] {local + 1}/{query_count} queries; "
                f"deduplicated events={sum(value.size for value in node_chunks)}",
                flush=True,
            )
    events = Events(
        np.concatenate(query_chunks),
        np.concatenate(node_chunks),
        np.concatenate(weight_chunks),
        np.concatenate(target_chunks),
    )
    per_query_weight = float(np.sum(events.weights)) / float(query_count)
    if not np.isclose(per_query_weight, 1.0, rtol=2e-5, atol=2e-5):
        raise ValueError("terminal event weights do not sum to one per query")
    metadata = {
        "version": 1,
        "query_start": query_start,
        "queries": query_count,
        "traces": [os.path.abspath(value) for value in traces],
        "raw_events": raw_events,
        "deduplicated_events": events.size,
        "unique_nodes": int(np.unique(events.node_ids).size),
        "weighting": "equal query x equal beam x inverse retained-pool size",
        "target": "exact IP - actual-primary-equivalent reconstruction IP - JI mean bias",
    }
    return events, metadata


def get_or_build_events(
    path: Path,
    traces: Sequence[str],
    exact_base: str,
    exact_queries: str,
    reconstruction: str,
    primary_queries: str,
    residual_mean: np.ndarray,
    node_count: int,
    force: bool,
) -> tuple[Events, dict]:
    fingerprint = event_fingerprint(
        traces,
        exact_base,
        exact_queries,
        reconstruction,
        primary_queries,
        residual_mean,
    )
    metadata_path = path / "metadata.json"
    if not force and metadata_path.exists():
        cached, metadata = load_events(path)
        if metadata.get("fingerprint") == fingerprint:
            print(f"[reuse] terminal event cache: {path}")
            return cached, metadata
    events, metadata = build_events(
        traces,
        exact_base,
        exact_queries,
        reconstruction,
        primary_queries,
        residual_mean,
        node_count,
    )
    metadata["fingerprint"] = fingerprint
    save_events(path, events, metadata)
    return load_events(path)


def unpack_all_codes(packed: np.ndarray, model: JointInnovationModel) -> np.ndarray:
    return np.column_stack(
        [unpack_group(packed, model.bits, group) for group in range(model.groups)]
    ).astype(np.int64)


def query_features(primary_queries: np.ndarray, model: JointInnovationModel) -> np.ndarray:
    transformed = np.asarray(primary_queries, dtype=np.float32) @ np.asarray(
        model.inverse_transform, dtype=np.float32
    ).T
    widths = np.asarray(model.group_dims, dtype=np.int64)
    if not np.all(widths == widths[0]):
        raise ValueError("A1 trainer currently requires equal-width deployment groups")
    return np.stack(
        [transformed[:, model.indices(group)] for group in range(model.groups)], axis=1
    ).astype(np.float32)


def active_codebooks(model: JointInnovationModel) -> np.ndarray:
    widths = np.asarray(model.group_dims, dtype=np.int64)
    counts = 1 << np.asarray(model.bits, dtype=np.int64)
    if not np.all(widths == widths[0]) or not np.all(counts == counts[0]):
        raise ValueError("A1 trainer requires equal-width/equal-rate deployment groups")
    return np.stack([model.codebook(group) for group in range(model.groups)]).astype(
        np.float32
    )


def predict_events(
    events: Events,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    output = np.empty(events.size, dtype=np.float32)
    group_ids = torch.arange(books.shape[0], device=device)[None, :]
    with torch.no_grad():
        for start in range(0, events.size, batch_size):
            end = min(events.size, start + batch_size)
            qids = torch.as_tensor(events.query_ids[start:end], device=device)
            labels = torch.as_tensor(codes[events.node_ids[start:end]], device=device)
            selected = books[group_ids, labels]
            prediction = torch.sum(selected * phi[qids], dim=(1, 2))
            output[start:end] = prediction.detach().cpu().numpy()
    return output


def weighted_mse(events: Events, predictions: np.ndarray) -> float:
    error = np.asarray(predictions, dtype=np.float64) - np.asarray(
        events.targets, dtype=np.float64
    )
    weights = np.asarray(events.weights, dtype=np.float64)
    return float(np.sum(weights * error * error) / np.sum(weights))


def train_codebooks(
    initial: np.ndarray,
    codes: np.ndarray,
    train: Events,
    validation: Events,
    phi: torch.Tensor,
    device: torch.device,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    patience: int,
    seed: int,
) -> tuple[np.ndarray, list[dict]]:
    if epochs <= 0:
        return np.asarray(initial, dtype=np.float32), []
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    books = torch.nn.Parameter(torch.as_tensor(initial, device=device).clone())
    optimizer = torch.optim.Adam([books], lr=learning_rate)
    group_ids = torch.arange(books.shape[0], device=device)[None, :]
    best = books.detach().cpu().numpy().copy()
    initial_validation = weighted_mse(
        validation,
        predict_events(validation, codes, phi, books.detach(), batch_size, device),
    )
    best_validation = initial_validation
    stale = 0
    history: list[dict] = [
        {
            "epoch": 0,
            "train_weighted_mse": None,
            "validation_weighted_mse": initial_validation,
        }
    ]
    for epoch in range(epochs):
        order = rng.permutation(train.size)
        total_loss = 0.0
        total_weight = 0.0
        for begin in range(0, train.size, batch_size):
            selected_indices = order[begin : begin + batch_size]
            qids = torch.as_tensor(train.query_ids[selected_indices], device=device)
            nodes = train.node_ids[selected_indices]
            labels = torch.as_tensor(codes[nodes], device=device)
            target = torch.as_tensor(train.targets[selected_indices], device=device)
            weight = torch.as_tensor(train.weights[selected_indices], device=device)
            selected = books[group_ids, labels]
            prediction = torch.sum(selected * phi[qids], dim=(1, 2))
            loss = torch.sum(weight * torch.square(prediction - target)) / torch.sum(weight)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(torch.sum(weight * torch.square(prediction - target)).item())
            total_weight += float(torch.sum(weight).item())
        current = books.detach()
        val_predictions = predict_events(
            validation, codes, phi, current, batch_size, device
        )
        val_mse = weighted_mse(validation, val_predictions)
        row = {
            "epoch": epoch + 1,
            "train_weighted_mse": total_loss / max(total_weight, 1e-30),
            "validation_weighted_mse": val_mse,
        }
        history.append(row)
        print(f"[terminal-fit] {json.dumps(row, sort_keys=True)}", flush=True)
        if val_mse + 1e-12 < best_validation:
            best_validation = val_mse
            best = current.cpu().numpy().copy()
            stale = 0
        else:
            stale += 1
            if patience > 0 and stale >= patience:
                break
    return best, history


def reencode_seen_nodes(
    events: Events,
    codes: np.ndarray,
    phi: torch.Tensor,
    books: torch.Tensor,
    batch_size: int,
    max_score_elements: int,
    device: torch.device,
) -> tuple[np.ndarray, dict]:
    """Exact group-coordinate assignment under every event of one node."""
    predictions = predict_events(events, codes, phi, books, batch_size, device)
    order = np.argsort(events.node_ids, kind="stable")
    sorted_nodes = events.node_ids[order]
    unique_nodes, starts, counts = np.unique(
        sorted_nodes, return_index=True, return_counts=True
    )
    buckets: dict[int, list[int]] = {}
    for record, count in enumerate(counts.tolist()):
        bucket = 1 << max(0, int(count - 1).bit_length())
        buckets.setdefault(bucket, []).append(record)

    changed_by_group: list[int] = []
    for group in range(books.shape[0]):
        old_labels = codes[:, group].copy()
        for padded_count, records in sorted(buckets.items()):
            records_array = np.asarray(records, dtype=np.int64)
            batch_nodes = max(
                1,
                max_score_elements // max(1, padded_count * int(books.shape[1])),
            )
            for begin in range(0, records_array.size, batch_nodes):
                chosen_records = records_array[begin : begin + batch_nodes]
                rows = chosen_records.size
                event_matrix = np.full((rows, padded_count), -1, dtype=np.int64)
                node_batch = unique_nodes[chosen_records]
                for row, record in enumerate(chosen_records.tolist()):
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
                loss = torch.sum(weight_t * torch.square(candidate - desired_t), dim=1)
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
                new = books[group, torch.as_tensor(codes[node_ids, group], device=device)]
                old_score[begin:end] = torch.sum(values * old, dim=1).cpu().numpy()
                new_score[begin:end] = torch.sum(values * new, dim=1).cpu().numpy()
        predictions += new_score - old_score
        changed = int(np.count_nonzero(codes[unique_nodes, group] != old_labels[unique_nodes]))
        changed_by_group.append(changed)
        print(
            f"[terminal-reencode] group={group + 1}/{books.shape[0]} "
            f"changed_nodes={changed}",
            flush=True,
        )
    return unique_nodes.astype(np.int64), {
        "seen_nodes": int(unique_nodes.size),
        "changed_nodes_by_group": changed_by_group,
        "train_weighted_mse_after_assignment": weighted_mse(events, predictions),
    }


def make_model(
    source: JointInnovationModel, books: np.ndarray, codes: np.ndarray, variant: str
) -> JointInnovationModel:
    prototypes = np.asarray(source.prototypes, dtype=np.float32).copy()
    counts = np.zeros_like(source.prototype_counts, dtype=np.int64)
    for group in range(source.groups):
        count = 1 << int(source.bits[group])
        width = int(source.group_dims[group])
        prototypes[group, :count, :width] = books[group]
        counts[group, :count] = np.bincount(
            codes[:, group], minlength=count
        ).astype(np.int64)
    model_id = compute_model_id(
        source.artifact_id,
        source.metric_id,
        source.forward_transform,
        source.group_indices,
        source.group_dims,
        source.bits,
        prototypes,
    )
    result = replace(
        source,
        model_id=model_id,
        prototypes=prototypes,
        prototype_counts=counts,
        variant=variant,
    )
    result.validate()
    return result


def write_fixed_side(path: str, model: JointInnovationModel, codes: np.ndarray) -> None:
    side = initialize_side_codes(path, model)
    side[:] = pack_codes(
        [codes[:, group] for group in range(model.groups)], model.bits
    )
    side.flush()
    del side


def main() -> None:
    args = parse_args()
    if args.epochs < 0 or args.retrain_epochs < 0 or args.reencode_sweeps < 0:
        raise ValueError("epoch/sweep counts must be non-negative")
    if args.mode == "fixed-codes" and not args.out_side_codes:
        raise ValueError("fixed-codes mode requires --out-side-codes")
    if args.mode == "reencode" and not args.out_overrides:
        raise ValueError("reencode mode requires --out-overrides")
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
        raise ValueError("A1 requires the deployed 8x8-bit JI representation")
    packed = read_side_codes(args.initial_side_codes, initial)
    codes = unpack_all_codes(packed, initial)
    del packed
    exact_base = fbin_memmap(args.exact_base)
    primary_queries = fbin_memmap(primary.query_fbin)
    if exact_base.shape[0] < primary.node_count:
        raise ValueError("exact base does not cover primary nodes")
    phi_np = query_features(primary_queries, initial)
    phi = torch.as_tensor(phi_np, device=device)

    cache = Path(args.event_cache)
    train, train_meta = get_or_build_events(
        cache / "train",
        args.train_retained,
        args.exact_base,
        args.exact_queries,
        primary.reconstruction_fbin,
        primary.query_fbin,
        initial.residual_mean,
        primary.node_count,
        args.force_event_cache,
    )
    validation, validation_meta = get_or_build_events(
        cache / "validation",
        args.validation_retained,
        args.exact_base,
        args.exact_queries,
        primary.reconstruction_fbin,
        primary.query_fbin,
        initial.residual_mean,
        primary.node_count,
        args.force_event_cache,
    )

    books = active_codebooks(initial)
    original_books = torch.as_tensor(books, device=device)
    before_train = weighted_mse(
        train, predict_events(train, codes, phi, original_books, args.batch_size, device)
    )
    before_validation = weighted_mse(
        validation,
        predict_events(validation, codes, phi, original_books, args.batch_size, device),
    )
    books, history = train_codebooks(
        books,
        codes,
        train,
        validation,
        phi,
        device,
        args.epochs,
        args.batch_size,
        args.learning_rate,
        args.patience,
        args.seed,
    )
    reencode_history: list[dict] = []
    seen_nodes = np.unique(train.node_ids)
    if args.mode == "reencode":
        outer_best_books = books.copy()
        outer_best_codes = codes[seen_nodes].copy()
        outer_best_validation = weighted_mse(
            validation,
            predict_events(
                validation,
                codes,
                phi,
                torch.as_tensor(books, device=device),
                args.batch_size,
                device,
            ),
        )
        for sweep in range(args.reencode_sweeps):
            seen_nodes, assignment = reencode_seen_nodes(
                train,
                codes,
                phi,
                torch.as_tensor(books, device=device),
                args.batch_size,
                args.max_assignment_score_elements,
                device,
            )
            books, retrain = train_codebooks(
                books,
                codes,
                train,
                validation,
                phi,
                device,
                args.retrain_epochs,
                args.batch_size,
                args.learning_rate,
                args.patience,
                args.seed + sweep + 1,
            )
            assignment["sweep"] = sweep + 1
            assignment["codebook_history"] = retrain
            sweep_validation = weighted_mse(
                validation,
                predict_events(
                    validation,
                    codes,
                    phi,
                    torch.as_tensor(books, device=device),
                    args.batch_size,
                    device,
                ),
            )
            assignment["validation_weighted_mse"] = sweep_validation
            if sweep_validation + 1e-12 < outer_best_validation:
                outer_best_validation = sweep_validation
                outer_best_books = books.copy()
                outer_best_codes = codes[seen_nodes].copy()
                assignment["selected_by_validation"] = True
            else:
                assignment["selected_by_validation"] = False
            reencode_history.append(assignment)
        books = outer_best_books
        codes[seen_nodes] = outer_best_codes

    final_books = torch.as_tensor(books, device=device)
    after_train = weighted_mse(
        train, predict_events(train, codes, phi, final_books, args.batch_size, device)
    )
    after_validation = weighted_mse(
        validation,
        predict_events(validation, codes, phi, final_books, args.batch_size, device),
    )
    variant = (
        "terminal-conditioned-fixed-codes-a1a"
        if args.mode == "fixed-codes"
        else "terminal-conditioned-fixed-node-code-a1b"
    )
    output_model = make_model(initial, books, codes, variant)
    write_model(args.out_model, output_model)
    if args.mode == "fixed-codes":
        write_fixed_side(args.out_side_codes, output_model, codes)
    else:
        Path(args.out_overrides).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.out_overrides,
            model_id=np.asarray(output_model.model_id, dtype=np.uint64),
            node_ids=np.asarray(seen_nodes, dtype=np.uint32),
            codes=np.asarray(codes[seen_nodes], dtype=np.uint16),
        )

    summary = {
        "version": 1,
        "method": "terminal_conditioned_joint_innovation_training",
        "mode": args.mode,
        "primary": primary.name,
        "representation": "one fixed 8x8-bit code per database node",
        "pairing_preserved": True,
        "train": train_meta,
        "validation": validation_meta,
        "train_weighted_mse_before": before_train,
        "train_weighted_mse_after": after_train,
        "train_relative_mse_reduction": 1.0 - after_train / before_train,
        "validation_weighted_mse_before": before_validation,
        "validation_weighted_mse_after": after_validation,
        "validation_relative_mse_reduction": 1.0
        - after_validation / before_validation,
        "fixed_codebook_history": history,
        "reencode_history": reencode_history,
        "seen_training_nodes": int(seen_nodes.size),
        "unseen_node_policy": (
            "unchanged assignments" if args.mode == "fixed-codes" else "global-JI fallback then paired override"
        ),
        "files": {
            "model": os.path.abspath(args.out_model),
            "side_codes": None
            if not args.out_side_codes
            else os.path.abspath(args.out_side_codes),
            "overrides": None
            if not args.out_overrides
            else os.path.abspath(args.out_overrides),
        },
        "config": vars(args),
    }
    output = Path(args.out_summary)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"wrote terminal-conditioned JI model: {args.out_model}")
    print(f"wrote terminal-conditioned summary: {args.out_summary}")


if __name__ == "__main__":
    main()
