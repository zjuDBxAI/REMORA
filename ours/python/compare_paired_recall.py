#!/usr/bin/env python3
"""Compare two fixed-candidate result matrices with a paired query bootstrap."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from fixed_candidate_io import read_id_matrix, row_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--reference-ids", required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--reference-name", default="reference")
    parser.add_argument("--candidate-name", default="candidate")
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--bootstrap-batch-size", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0
        or args.topk <= 0
        or args.bootstrap_samples <= 0
        or args.bootstrap_batch_size <= 0
    ):
        raise ValueError("invalid paired recall comparison configuration")
    reference = read_id_matrix(args.reference_ids)
    candidate = read_id_matrix(args.candidate_ids)
    if reference.shape != candidate.shape:
        raise ValueError("reference and candidate result shapes differ")
    truth = read_id_matrix(
        args.ground_truth,
        allow_trailing=True,
        row_start=args.query_start,
        row_count=reference.shape[0],
    )
    if args.topk > min(reference.shape[1], truth.shape[1]):
        raise ValueError("topk exceeds result or ground-truth width")
    reference_recall = row_recall(reference, truth, args.topk)
    candidate_recall = row_recall(candidate, truth, args.topk)
    delta = candidate_recall - reference_recall
    rng = np.random.default_rng(args.seed)
    bootstrap = np.empty(args.bootstrap_samples, dtype=np.float64)
    completed = 0
    while completed < args.bootstrap_samples:
        batch = min(args.bootstrap_batch_size, args.bootstrap_samples - completed)
        indices = rng.integers(
            0, delta.size, size=(batch, delta.size), dtype=np.int64
        )
        bootstrap[completed : completed + batch] = np.mean(delta[indices], axis=1)
        completed += batch
    lower, upper = np.quantile(bootstrap, (0.025, 0.975))
    summary = {
        "version": 1,
        "method": "paired_query_bootstrap_recall_comparison",
        "queries": int(delta.size),
        "topk": args.topk,
        "query_start": args.query_start,
        "reference_name": args.reference_name,
        "candidate_name": args.candidate_name,
        "reference_recall": float(np.mean(reference_recall)),
        "candidate_recall": float(np.mean(candidate_recall)),
        "recall_delta": float(np.mean(delta)),
        "recall_delta_pp": float(100.0 * np.mean(delta)),
        "bootstrap_95_ci": [float(lower), float(upper)],
        "bootstrap_95_ci_pp": [float(100.0 * lower), float(100.0 * upper)],
        "rows_improved": int(np.sum(delta > 0.0)),
        "rows_regressed": int(np.sum(delta < 0.0)),
        "rows_tied": int(np.sum(delta == 0.0)),
        "significantly_better_95": bool(lower > 0.0),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "files": {
            "ground_truth": os.path.abspath(args.ground_truth),
            "reference_ids": os.path.abspath(args.reference_ids),
            "candidate_ids": os.path.abspath(args.candidate_ids),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
