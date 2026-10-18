"""Exhaustive Faiss CPU KNN over a primary quantizer's reconstructed vectors.

The query and database are in the primary scorer's transformed coordinates.
The existing exact Faiss scanner owns blockwise traversal, deterministic tie
selection, and a durable checkpoint. This command emits the native graph
builder's uint32 KNN matrix only after the full scan is complete.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time

from exact_faiss_cpu import ALGORITHM, implementation, scan
from protocol import CsvLedger, file_id, identity, matrix, write_matrix


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", required=True)
    p.add_argument("--reconstruction-fbin", type=Path, required=True)
    p.add_argument("--query-fbin", type=Path, required=True)
    p.add_argument("--primary-model", type=Path, required=True)
    p.add_argument("--primary-codes", type=Path, required=True)
    p.add_argument("--query-start", type=int, required=True)
    p.add_argument("--query-limit", type=int, required=True)
    p.add_argument("--topk", type=int, required=True)
    p.add_argument("--base-batch-size", type=int, required=True)
    p.add_argument("--query-batch-size", type=int, required=True)
    p.add_argument("--threads", type=int, required=True)
    p.add_argument("--checkpoint-interval-s", type=float, default=300.0)
    p.add_argument("--output-knn", type=Path, required=True)
    p.add_argument("--output-metadata", type=Path, required=True)
    return p


def run(args):
    base = matrix(args.reconstruction_fbin)
    all_queries = matrix(args.query_fbin)
    if (not args.method or min(args.query_limit, args.topk, args.base_batch_size,
                               args.query_batch_size, args.threads) <= 0 or
            args.query_start < 0 or args.checkpoint_interval_s < 0 or
            args.query_start + args.query_limit > len(all_queries) or
            base.shape[1] != all_queries.shape[1] or args.topk > len(base)):
        raise ValueError("Invalid quantized Faiss KNN shapes or limits")
    queries = all_queries[args.query_start:args.query_start + args.query_limit]
    impl = implementation()
    signature = dict(algorithm="quantized-reconstruction-" + ALGORITHM,
                     method=args.method, reconstruction=file_id(args.reconstruction_fbin),
                     transformed_queries=file_id(args.query_fbin),
                     primary_model=file_id(args.primary_model),
                     primary_codes=file_id(args.primary_codes),
                     query_start=args.query_start, query_limit=args.query_limit,
                     rows=len(base), dim=base.shape[1], topk=args.topk,
                     base_batch_size=args.base_batch_size,
                     query_batch_size=args.query_batch_size, threads=args.threads,
                     implementation=impl)
    prefix = args.output_knn.with_suffix("")
    checkpoint = Path(str(prefix) + ".checkpoint.npz")
    progress_path = Path(str(prefix) + ".progress.csv")
    fields = ["algorithm", "method", "completed_rows", "total_rows", "queries", "k",
              "elapsed_s", "resume_from", "checkpoint_rows", "checkpoint_written",
              "tie_repair_queries", "threads", "faiss_version", "faiss_compile_options",
              "scoring", "cutoff_ties"]
    ledger = CsvLedger(progress_path, fields)
    last_log = 0.0

    def on_progress(event):
        nonlocal last_log
        now = time.monotonic()
        if (not event["checkpoint_written"] and event["completed_rows"] < event["total_rows"]
                and now - last_log < 30):
            return
        row = dict(algorithm=signature["algorithm"], method=args.method, **event)
        ledger.append(row)
        save_json(Path(str(prefix) + ".progress.json"), row)
        print(f"[quantized Faiss CPU] {args.method} {event['completed_rows']}/{event['total_rows']} "
              f"database vectors; checkpoint={event['checkpoint_rows']}; "
              f"resume_from={event['resume_from']}", flush=True)
        last_log = now

    started = time.monotonic()
    ids, _, info = scan(base, queries, args.topk,
                        base_batch_size=args.base_batch_size,
                        query_batch_size=args.query_batch_size,
                        threads=args.threads, signature=signature,
                        checkpoint_path=checkpoint,
                        checkpoint_interval_s=args.checkpoint_interval_s,
                        progress=on_progress)
    write_matrix(args.output_knn, ids, "<u4")
    save_json(args.output_metadata, dict(
        signature=signature, signature_id=identity(signature),
        knn_id=file_id(args.output_knn), score="inner product on the primary quantizer's reconstruction",
        full_database=True, completed_base_rows=len(base),
        elapsed_s=time.monotonic() - started, checkpoint_info=info))


if __name__ == "__main__":
    run(parser().parse_args())
