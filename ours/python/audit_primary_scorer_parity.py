#!/usr/bin/env python3
"""Verify that a decoded primary representation reproduces its official scorer.

The official values may be similarities or distances.  A query-only additive
constant is removed row by row because it cannot affect ANN rankings.  The
result is intended to be referenced by a PrimaryQuantizerManifest.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap
from fixed_candidate_io import read_id_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reconstruction-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--candidate-ids", required=True)
    parser.add_argument("--official-values", required=True)
    parser.add_argument(
        "--official-value-kind", choices=("similarity", "distance"), required=True
    )
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    parser.add_argument("--out-json", required=True)
    return parser.parse_args()


def read_float_matrix(path: str) -> np.ndarray:
    with open(path, "rb") as handle:
        shape = np.fromfile(handle, dtype="<u4", count=2)
    if shape.size != 2:
        raise ValueError(f"invalid float-matrix header: {path}")
    rows, cols = int(shape[0]), int(shape[1])
    expected = 8 + rows * cols * np.dtype("<f4").itemsize
    if os.path.getsize(path) != expected:
        raise ValueError(f"float-matrix size mismatch: {path}")
    return np.memmap(path, mode="r", dtype="<f4", offset=8, shape=(rows, cols))


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0
        or args.query_limit < 0
        or args.topk <= 0
        or args.atol < 0.0
        or args.rtol < 0.0
    ):
        raise ValueError("invalid scorer-parity configuration")
    reconstruction = fbin_memmap(args.reconstruction_fbin)
    queries = fbin_memmap(args.query_fbin)
    candidate_ids = read_id_matrix(args.candidate_ids)
    official_raw = read_float_matrix(args.official_values)
    if candidate_ids.shape != official_raw.shape:
        raise ValueError("candidate IDs and official values have different shapes")
    rows = candidate_ids.shape[0]
    if args.query_limit > 0:
        rows = min(rows, args.query_limit)
    if args.query_start + rows > queries.shape[0]:
        raise ValueError("requested query rows exceed query matrix")
    if queries.shape[1] != reconstruction.shape[1]:
        raise ValueError("query/reconstruction dimensions differ")
    if int(np.max(candidate_ids[:rows])) >= reconstruction.shape[0]:
        raise ValueError("candidate ID exceeds reconstruction rows")

    total = rows * candidate_ids.shape[1]
    absolute_errors = np.empty(total, dtype=np.float64)
    thresholds = np.empty(total, dtype=np.float64)
    exact_topk_rows = 0
    exact_full_order_rows = 0
    offset_values = np.empty(rows, dtype=np.float64)
    cursor = 0
    sign = 1.0 if args.official_value_kind == "similarity" else -1.0
    for row in range(rows):
        ids = np.asarray(candidate_ids[row], dtype=np.uint32)
        query = np.asarray(queries[args.query_start + row], dtype=np.float64)
        decoded = np.asarray(reconstruction[ids], dtype=np.float64) @ query
        official = sign * np.asarray(official_raw[row], dtype=np.float64)
        # The median is robust to a few malformed candidates and estimates the
        # only affine freedom allowed by this adapter: one constant per query.
        offset = float(np.median(official - decoded))
        offset_values[row] = offset
        aligned = official - offset
        error = np.abs(aligned - decoded)
        threshold = args.atol + args.rtol * np.maximum(
            np.abs(aligned), np.abs(decoded)
        )
        end = cursor + ids.size
        absolute_errors[cursor:end] = error
        thresholds[cursor:end] = threshold
        cursor = end
        official_order = np.lexsort((ids, -official))
        decoded_order = np.lexsort((ids, -decoded))
        exact_full_order_rows += int(np.array_equal(official_order, decoded_order))
        exact_topk_rows += int(
            np.array_equal(
                official_order[: args.topk], decoded_order[: args.topk]
            )
        )

    absolute_errors = absolute_errors[:cursor]
    thresholds = thresholds[:cursor]
    within = absolute_errors <= thresholds
    passed = bool(np.all(within) and exact_topk_rows == rows)
    payload = {
        "version": 1,
        "method": "official_primary_scorer_reconstruction_parity",
        "passed": passed,
        "queries": rows,
        "candidates_per_query": int(candidate_ids.shape[1]),
        "score_elements": int(cursor),
        "query_start": args.query_start,
        "official_value_kind": args.official_value_kind,
        "query_only_offset": {
            "min": float(np.min(offset_values)),
            "max": float(np.max(offset_values)),
            "mean": float(np.mean(offset_values)),
        },
        "max_absolute_score_error_after_offset": float(np.max(absolute_errors)),
        "mean_absolute_score_error_after_offset": float(np.mean(absolute_errors)),
        "p99_absolute_score_error_after_offset": float(
            np.quantile(absolute_errors, 0.99)
        ),
        "score_within_tolerance_rate": float(np.mean(within)),
        "exact_topk_row_match_rate": exact_topk_rows / float(rows),
        "exact_full_order_row_match_rate": exact_full_order_rows / float(rows),
        "atol": args.atol,
        "rtol": args.rtol,
        "files": {
            "reconstruction_fbin": os.path.abspath(args.reconstruction_fbin),
            "query_fbin": os.path.abspath(args.query_fbin),
            "candidate_ids": os.path.abspath(args.candidate_ids),
            "official_values": os.path.abspath(args.official_values),
        },
    }
    output = Path(args.out_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    print(f"wrote scorer parity audit: {output} (passed={passed})")
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
