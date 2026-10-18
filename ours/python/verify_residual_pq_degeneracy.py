#!/usr/bin/env python3
"""Verify lambda=0 metric semantics and report retraining reproducibility.

Two independently trained CUDA Lloyd runs need not be byte-identical because
floating-point ``index_add_`` reductions are order-dependent.  The required
degeneracy property is therefore that lambda=0 stores exact identity metric
matrices/factors and selects the Euclidean assignment kernel.  Byte equality is
reported separately and can be made mandatory for deterministic twin runs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from residual_pq_baseline_io import (
    read_residual_pq_model,
    read_residual_pq_side_codes,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--euclidean-model", required=True)
    parser.add_argument("--euclidean-side-codes", required=True)
    parser.add_argument("--metric-zero-model", required=True)
    parser.add_argument("--metric-zero-side-codes", required=True)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--chunk-size", type=int, default=1000000)
    parser.add_argument(
        "--require-byte-exact",
        action="store_true",
        help=(
            "also require independently trained prototypes/counts/side codes "
            "to be byte-identical; use only for deterministic twin runs"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.chunk_size <= 0:
        raise ValueError("chunk-size must be positive")
    reference = read_residual_pq_model(args.euclidean_model)
    candidate = read_residual_pq_model(args.metric_zero_model)
    if (
        reference.artifact_id != candidate.artifact_id
        or reference.node_count != candidate.node_count
        or reference.dim != candidate.dim
        or reference.groups != candidate.groups
        or reference.codewords != candidate.codewords
    ):
        raise ValueError("Euclidean and lambda=0 models are structurally incompatible")
    reference_codes, reference_width = read_residual_pq_side_codes(
        args.euclidean_side_codes, reference
    )
    candidate_codes, candidate_width = read_residual_pq_side_codes(
        args.metric_zero_side_codes, candidate
    )
    if reference_width != candidate_width:
        raise ValueError("Euclidean and lambda=0 side-code widths differ")
    prototype_equal = bool(np.array_equal(reference.prototypes, candidate.prototypes))
    count_equal = bool(
        np.array_equal(reference.prototype_counts, candidate.prototype_counts)
    )
    prototype_max_abs = float(
        np.max(
            np.abs(
                reference.prototypes.astype(np.float64)
                - candidate.prototypes.astype(np.float64)
            )
        )
    )
    mismatched_bytes = 0
    mismatched_nodes = 0
    for start in range(0, reference.node_count, args.chunk_size):
        end = min(reference.node_count, start + args.chunk_size)
        difference = np.asarray(reference_codes[start:end]) != np.asarray(
            candidate_codes[start:end]
        )
        mismatched_bytes += int(np.sum(difference))
        mismatched_nodes += int(np.sum(np.any(difference, axis=1)))
    reference_identity = np.broadcast_to(
        np.eye(reference.group_dim, dtype=np.float32),
        reference.metric_matrices.shape,
    )
    candidate_identity = np.broadcast_to(
        np.eye(candidate.group_dim, dtype=np.float32),
        candidate.metric_matrices.shape,
    )
    reference_matrix_identity = bool(
        np.array_equal(reference.metric_matrices, reference_identity)
    )
    reference_factor_identity = bool(
        np.array_equal(reference.metric_factors, reference_identity)
    )
    candidate_matrix_identity = bool(
        np.array_equal(candidate.metric_matrices, candidate_identity)
    )
    candidate_factor_identity = bool(
        np.array_equal(candidate.metric_factors, candidate_identity)
    )
    semantic_degeneracy = bool(
        reference.uses_euclidean_assignment
        and reference_matrix_identity
        and reference_factor_identity
        and candidate.uses_euclidean_assignment
        and candidate_matrix_identity
        and candidate_factor_identity
        and candidate.metric_lambda == 0.0
    )
    byte_exact_reproduction = bool(
        prototype_equal
        and count_equal
        and mismatched_bytes == 0
        and reference.encoding_id == candidate.encoding_id
    )
    passed = bool(
        semantic_degeneracy
        and (byte_exact_reproduction or not args.require_byte_exact)
    )
    summary = {
        "version": 2,
        "method": "lambda_zero_semantic_degeneracy_verification",
        "semantic_degeneracy_passed": semantic_degeneracy,
        "byte_exact_reproduction_passed": byte_exact_reproduction,
        "byte_exact_reproduction_required": bool(args.require_byte_exact),
        "reference_metric_matrix_exact_identity": reference_matrix_identity,
        "reference_metric_factor_exact_identity": reference_factor_identity,
        "candidate_metric_matrix_exact_identity": candidate_matrix_identity,
        "candidate_metric_factor_exact_identity": candidate_factor_identity,
        "prototype_exact_match": prototype_equal,
        "prototype_max_abs_difference": prototype_max_abs,
        "prototype_count_exact_match": count_equal,
        "side_code_exact_match": mismatched_bytes == 0,
        "mismatched_side_code_bytes": mismatched_bytes,
        "mismatched_side_code_nodes": mismatched_nodes,
        "mismatched_side_code_node_rate": (
            mismatched_nodes / max(reference.node_count, 1)
        ),
        "encoding_id_match": reference.encoding_id == candidate.encoding_id,
        "reference_uses_euclidean_assignment": (
            reference.uses_euclidean_assignment
        ),
        "candidate_uses_euclidean_assignment": candidate.uses_euclidean_assignment,
        "candidate_metric_lambda": candidate.metric_lambda,
        "independent_retraining_note": (
            "Byte equality is diagnostic only unless --require-byte-exact is set; "
            "independent CUDA Lloyd runs can diverge through nondeterministic "
            "floating-point reductions."
        ),
        "passed": passed,
        "files": {
            "euclidean_model": str(Path(args.euclidean_model).resolve()),
            "euclidean_side_codes": str(
                Path(args.euclidean_side_codes).resolve()
            ),
            "metric_zero_model": str(Path(args.metric_zero_model).resolve()),
            "metric_zero_side_codes": str(
                Path(args.metric_zero_side_codes).resolve()
            ),
        },
    }
    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if not summary["passed"]:
        if args.require_byte_exact and semantic_degeneracy:
            raise SystemExit(
                "lambda=0 semantics pass, but deterministic byte reproduction fails"
            )
        raise SystemExit("lambda=0 residual-PQ semantic degeneracy fails")


if __name__ == "__main__":
    main()
