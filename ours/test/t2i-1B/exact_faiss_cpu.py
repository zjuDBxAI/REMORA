"""Exhaustive FP32 inner-product KNN with blockwise Faiss and durable resume.

The database is the outer loop: a block is loaded once and searched by every
query. A k+1 probe detects cutoff ties; those queries receive a full-block
FP32 BLAS selection with ascending IDs. No approximate index is used.
"""
from __future__ import annotations

import os
from pathlib import Path
import time

import numpy as np

from protocol import identity

ALGORITHM = "faiss-cpu-flat-ip-v1-base-outer"
INVALID_ID = np.iinfo(np.uint32).max


def implementation():
    import faiss
    options = faiss.get_compile_options().strip()
    if "AVX512" not in options:
        raise RuntimeError("CPU Faiss exact KNN requires the AVX-512 Faiss build")
    return dict(faiss_version=faiss.__version__, faiss_compile_options=options,
                scoring="FP32 Flat inner product; AVX-512; BLAS threshold=0",
                cutoff_ties="full-block FP32 BLAS, ascending node ID")


def validate_topk(ids, scores, seen, k):
    valid = min(seen, k)
    if ids.dtype != np.uint32 or scores.dtype != np.float32 or ids.shape != scores.shape or ids.shape[1] != k:
        raise ValueError("Invalid Faiss checkpoint array shape/dtype")
    good_ids, good_scores = ids[:, :valid], scores[:, :valid]
    if np.any(good_ids >= seen) or not np.isfinite(good_scores).all():
        raise ValueError("Invalid Faiss checkpoint ID/score")
    if valid > 1:
        if np.any(np.diff(np.sort(good_ids, axis=1), axis=1) == 0):
            raise ValueError("Duplicate Faiss checkpoint IDs")
        if np.any(good_scores[:, 1:] > good_scores[:, :-1]):
            raise ValueError("Faiss checkpoint scores are not descending")
        ties = good_scores[:, 1:] == good_scores[:, :-1]
        if np.any(ties & (good_ids[:, 1:] < good_ids[:, :-1])):
            raise ValueError("Faiss checkpoint tie order differs")
    if valid < k and (np.any(ids[:, valid:] != INVALID_ID) or np.any(scores[:, valid:] != -np.inf)):
        raise ValueError("Invalid unfilled Faiss checkpoint entries")


def _load_checkpoint(path, signature, nq, k, total, block_size):
    with np.load(path, allow_pickle=False) as saved:
        if str(saved["signature"].item()) != identity(signature):
            raise ValueError("Faiss checkpoint inputs/backend differ; use a new output root")
        seen = int(saved["completed_rows"].item())
        if not 0 < seen <= total or (seen != total and seen % block_size):
            raise ValueError("Invalid Faiss checkpoint database boundary")
        ids, scores = saved["ids"].copy(), saved["scores"].copy()
        if ids.shape != (nq, k):
            raise ValueError("Faiss checkpoint query shape differs")
        repaired = int(saved["tie_repair_queries"].item())
    validate_topk(ids, scores, seen, k)
    return seen, ids, scores, repaired


def _save_checkpoint(path, signature, seen, ids, scores, repaired):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as stream:
        np.savez(stream, signature=np.asarray(identity(signature)),
                 completed_rows=np.asarray(seen, dtype=np.uint64), ids=ids, scores=scores,
                 tie_repair_queries=np.asarray(repaired, dtype=np.uint64))
        stream.flush(); os.fsync(stream.fileno())
    tmp.replace(path)


def _repair_ties(queries, base, k):
    ids = np.empty((len(queries), k), dtype=np.int64)
    scores = np.empty((len(queries), k), dtype=np.float32)
    for start in range(0, len(queries), 32):
        block = queries[start:start+32] @ base.T
        if not np.isfinite(block).all():
            raise ValueError("Nonfinite FP32 inner products during tie repair")
        for offset, row in enumerate(block):
            selected = np.argpartition(-row, k-1)[:k]
            cutoff = row[selected].min()
            higher = np.flatnonzero(row > cutoff)
            tied = np.flatnonzero(row == cutoff)[:k-len(higher)]
            local = np.concatenate((higher, tied))
            order = np.lexsort((local, -row[local]))
            ids[start+offset] = local[order]
            scores[start+offset] = row[local[order]]
    return ids, scores


def scan(base, queries, k, *, base_batch_size, query_batch_size, threads,
         signature, checkpoint_path, checkpoint_interval_s=300, progress=None):
    import faiss
    from threadpoolctl import threadpool_limits

    info = implementation()
    if (base.ndim != 2 or queries.ndim != 2 or base.shape[1] != queries.shape[1]
            or not len(queries) or not 0 < k <= len(base) or len(base) > INVALID_ID
            or min(base_batch_size, query_batch_size, threads) <= 0):
        raise ValueError("Invalid Faiss exact KNN dimensions/limits")
    queries = np.ascontiguousarray(queries, dtype=np.float32)
    if not np.isfinite(queries).all():
        raise ValueError("Nonfinite Faiss exact queries")
    nq = len(queries); total = len(base); checkpoint_path = Path(checkpoint_path)
    if checkpoint_path.exists():
        seen, ids, scores, repaired = _load_checkpoint(
            checkpoint_path, signature, nq, k, total, base_batch_size)
    else:
        seen, repaired = 0, 0
        ids = np.full((nq, k), INVALID_ID, dtype=np.uint32)
        scores = np.full((nq, k), -np.inf, dtype=np.float32)
    resume_from = seen; checkpoint_rows = seen
    started = time.monotonic(); last_checkpoint = started
    previous_threads = faiss.omp_get_max_threads()
    previous_threshold = faiss.cvar.distance_compute_blas_threshold
    try:
        faiss.omp_set_num_threads(threads)
        # Keep tail batches and tie probes on the same FP32 GEMM path.
        faiss.cvar.distance_compute_blas_threshold = 0
        with threadpool_limits(limits=threads):
            for start in range(seen, total, base_batch_size):
                values = np.ascontiguousarray(base[start:start+base_batch_size], dtype=np.float32)
                if not np.isfinite(values).all():
                    raise ValueError(f"Nonfinite base rows near {start}")
                count = min(k, len(values)); requested = min(count+1, len(values))
                index = faiss.IndexFlatIP(values.shape[1]); index.add(values)
                for qs in range(0, nq, query_batch_size):
                    q = queries[qs:qs+query_batch_size]
                    local_scores, local_ids = index.search(q, requested)
                    if not np.isfinite(local_scores).all() or np.any(local_ids < 0):
                        raise ValueError("Invalid Faiss FP32 exact results")
                    ambiguous = (np.flatnonzero(local_scores[:, count-1] == local_scores[:, count])
                                 if requested > count else np.empty(0, dtype=np.int64))
                    local_scores = local_scores[:, :count].copy()
                    local_ids = local_ids[:, :count].copy()
                    if len(ambiguous):
                        fixed_ids, fixed_scores = _repair_ties(q[ambiguous], values, count)
                        local_ids[ambiguous] = fixed_ids; local_scores[ambiguous] = fixed_scores
                        repaired += len(ambiguous)
                    local_ids = (local_ids + start).astype(np.uint32)
                    merged_ids = np.concatenate((ids[qs:qs+len(q)], local_ids), axis=1)
                    merged_scores = np.concatenate((scores[qs:qs+len(q)], local_scores), axis=1)
                    order = np.lexsort((merged_ids, -merged_scores), axis=1)[:, :k]
                    ids[qs:qs+len(q)] = np.take_along_axis(merged_ids, order, axis=1)
                    scores[qs:qs+len(q)] = np.take_along_axis(merged_scores, order, axis=1)
                del index, values
                seen = min(start+base_batch_size, total)
                now = time.monotonic()
                checkpoint_written = (checkpoint_rows == 0 or seen == total
                                      or now-last_checkpoint >= checkpoint_interval_s)
                if checkpoint_written:
                    _save_checkpoint(checkpoint_path, signature, seen, ids, scores, repaired)
                    checkpoint_rows = seen; last_checkpoint = time.monotonic()
                if progress:
                    progress(dict(completed_rows=seen, total_rows=total, queries=nq, k=k,
                        elapsed_s=time.monotonic()-started, resume_from=resume_from,
                        checkpoint_rows=checkpoint_rows, checkpoint_written=checkpoint_written,
                        tie_repair_queries=repaired, threads=threads, **info))
    finally:
        faiss.cvar.distance_compute_blas_threshold = previous_threshold
        faiss.omp_set_num_threads(previous_threads)
    validate_topk(ids, scores, total, k)
    return ids, scores, dict(resume_from=resume_from, checkpoint_rows=checkpoint_rows,
                            tie_repair_queries=repaired, **info)
