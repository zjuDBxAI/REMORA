#!/usr/bin/env python3
"""Audit exact/ADC/RPQ inversions on one shared union of search pairs."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
from typing import Iterable

import numpy as np

from drb_io import fbin_memmap, read_codes, read_model
from residual_pq_baseline_io import (
    read_residual_pq_model,
    read_residual_pq_side_codes,
)
from residual_pq_baseline import (
    primary_conditioning_lut,
    residual_query_coordinates,
)


SOURCE_TO_CODE = {"exp": 0, "queue": 1, "stop": 2}
CODE_TO_SOURCE = {value: key for key, value in SOURCE_TO_CODE.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--residual-model", required=True)
    parser.add_argument("--side-codes", required=True)
    parser.add_argument(
        "--rpq-name",
        default="ood-rpq",
        help="label for the residual scorer in JSON/CSV output",
    )
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument(
        "--event-run",
        action="append",
        required=True,
        help="ORIGIN=EVENTS.csv; repeat for exact/adc/ood-rpq trajectories",
    )
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--margin-floor", type=float, default=1e-3)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument("--out-events-npz", default="")
    return parser.parse_args()


def parse_event_spec(spec: str) -> tuple[str, str]:
    if "=" not in spec:
        raise ValueError(f"invalid --event-run {spec!r}; expected ORIGIN=FILE")
    origin, path = (part.strip() for part in spec.split("=", 1))
    if not origin or not path:
        raise ValueError(f"invalid --event-run {spec!r}")
    return origin, path


def load_one_event_file(
    path: str,
    origin_bit: int,
    query_start: int,
    query_end: int,
) -> tuple[np.ndarray, ...]:
    columns: list[list[object]] = [[] for _ in range(6)]
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {
            "query_id", "source", "u_id", "v_id", "exact_gap",
        }
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"{path} must contain {sorted(required)}")
        for row in reader:
            query_id = int(row["query_id"])
            if query_id < query_start or query_id >= query_end:
                continue
            source_name = str(row["source"]).strip().lower()
            if source_name not in SOURCE_TO_CODE:
                continue
            u_id, v_id = int(row["u_id"]), int(row["v_id"])
            exact_gap = float(row["exact_gap"])
            if (
                u_id < 0 or v_id < 0 or u_id == v_id
                or not np.isfinite(exact_gap) or exact_gap <= 0.0
            ):
                continue
            columns[0].append(query_id)
            columns[1].append(SOURCE_TO_CODE[source_name])
            columns[2].append(u_id)
            columns[3].append(v_id)
            columns[4].append(exact_gap)
            columns[5].append(origin_bit)
    if not columns[0]:
        raise ValueError(f"no usable events in {path}")
    return (
        np.asarray(columns[0], dtype=np.int64),
        np.asarray(columns[1], dtype=np.uint8),
        np.asarray(columns[2], dtype=np.uint32),
        np.asarray(columns[3], dtype=np.uint32),
        np.asarray(columns[4], dtype=np.float32),
        np.asarray(columns[5], dtype=np.uint32),
    )


def load_union_events(
    specs: list[tuple[str, str]],
    query_start: int,
    query_end: int,
) -> tuple[tuple[np.ndarray, ...], dict[str, int], dict[str, int]]:
    chunks = []
    raw_counts: dict[str, int] = {}
    origin_bits: dict[str, int] = {}
    for index, (origin, path) in enumerate(specs):
        if origin in origin_bits:
            raise ValueError(f"duplicate event origin: {origin}")
        if index >= 31:
            raise ValueError("at most 31 event origins are supported")
        origin_bit = 1 << index
        origin_bits[origin] = origin_bit
        chunk = load_one_event_file(path, origin_bit, query_start, query_end)
        raw_counts[origin] = int(chunk[0].size)
        chunks.append(chunk)

    concatenated = tuple(
        np.concatenate([chunk[column] for chunk in chunks])
        for column in range(6)
    )
    query_id, source, u_id, v_id, exact_gap, origin_mask = concatenated
    order = np.lexsort((v_id, u_id, source, query_id))
    query_id, source, u_id, v_id, exact_gap, origin_mask = (
        values[order]
        for values in (query_id, source, u_id, v_id, exact_gap, origin_mask)
    )
    first = np.empty(query_id.size, dtype=bool)
    first[0] = True
    first[1:] = (
        (query_id[1:] != query_id[:-1])
        | (source[1:] != source[:-1])
        | (u_id[1:] != u_id[:-1])
        | (v_id[1:] != v_id[:-1])
    )
    starts = np.flatnonzero(first)
    reduced_origin = np.bitwise_or.reduceat(origin_mask, starts)
    events = (
        query_id[starts], source[starts], u_id[starts], v_id[starts],
        exact_gap[starts], reduced_origin,
    )
    return events, raw_counts, origin_bits


def transform_query_cpp(query: np.ndarray, model) -> np.ndarray:
    if not model.has_transform:
        return np.asarray(query, dtype=np.float32)
    transformed = np.sum(
        np.asarray(model.transform, dtype=np.float64)
        * np.asarray(query, dtype=np.float64)[None, :],
        axis=1,
        dtype=np.float64,
    ).astype(np.float32)
    return np.asarray(transformed + model.transform_bias, dtype=np.float32)


def score_nodes(
    node_ids: np.ndarray,
    adc_lut: np.ndarray,
    residual_lut: np.ndarray,
    conditional_lut: np.ndarray | None,
    codes: np.ndarray,
    side_codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_codes = np.asarray(codes[node_ids], dtype=np.uint8)
    subspaces = np.arange(adc_lut.shape[0], dtype=np.int64)[None, :]
    adc = np.sum(
        adc_lut[subspaces, node_codes], axis=1, dtype=np.float64
    ).astype(np.float32)
    refined = np.asarray(adc, dtype=np.float32).copy()
    if conditional_lut is not None:
        for subspace in range(node_codes.shape[1]):
            refined += conditional_lut[subspace, node_codes[:, subspace]]
    node_side = np.asarray(side_codes[node_ids], dtype=np.uint8)
    correction = np.zeros(node_ids.size, dtype=np.float32)
    for group in range(residual_lut.shape[0]):
        correction += residual_lut[group, node_side[:, group]]
    return adc, np.asarray(refined + correction, dtype=np.float32)


def compute_gaps(
    events: tuple[np.ndarray, ...],
    queries: np.ndarray,
    pq_model,
    residual_model,
    codes: np.ndarray,
    side_codes: np.ndarray,
    log_interval: int,
) -> tuple[np.ndarray, np.ndarray]:
    query_id, _source, u_id, v_id, _exact_gap, _origin = events
    adc_gap = np.empty(query_id.size, dtype=np.float32)
    rpq_gap = np.empty(query_id.size, dtype=np.float32)
    centroid64 = np.asarray(pq_model.centroids, dtype=np.float64)
    prototype64 = np.asarray(residual_model.prototypes, dtype=np.float64)
    boundaries = np.flatnonzero(
        np.r_[True, query_id[1:] != query_id[:-1], True]
    )
    query_groups = 0
    for boundary_index in range(boundaries.size - 1):
        begin, end = int(boundaries[boundary_index]), int(boundaries[boundary_index + 1])
        qid = int(query_id[begin])
        query_t = transform_query_cpp(
            np.asarray(queries[qid], dtype=np.float32), pq_model
        )
        primary_blocks = query_t.reshape(pq_model.subspaces, pq_model.dsub)
        residual_query = residual_query_coordinates(query_t, residual_model)
        adc_lut = np.sum(
            centroid64 * primary_blocks.astype(np.float64)[:, None, :],
            axis=2,
            dtype=np.float64,
        ).astype(np.float32)
        if residual_model.quantizer_kind == "additive":
            residual_lut = np.sum(
                prototype64 * residual_query.astype(np.float64)[None, None, :],
                axis=2,
                dtype=np.float64,
            ).astype(np.float32)
        else:
            residual_blocks = residual_query.reshape(
                residual_model.groups, residual_model.group_dim
            )
            residual_lut = np.sum(
                prototype64 * residual_blocks.astype(np.float64)[:, None, :],
                axis=2,
                dtype=np.float64,
            ).astype(np.float32)
        conditional_lut = primary_conditioning_lut(
            residual_query, residual_model
        )
        u_adc, u_rpq = score_nodes(
            u_id[begin:end], adc_lut, residual_lut, conditional_lut,
            codes, side_codes
        )
        v_adc, v_rpq = score_nodes(
            v_id[begin:end], adc_lut, residual_lut, conditional_lut,
            codes, side_codes
        )
        adc_gap[begin:end] = np.asarray(u_adc - v_adc, dtype=np.float32)
        rpq_gap[begin:end] = np.asarray(u_rpq - v_rpq, dtype=np.float32)
        query_groups += 1
        if log_interval > 0 and (
            query_groups % log_interval == 0
            or boundary_index + 1 == boundaries.size - 1
        ):
            print(
                f"[inversion-audit] queries={query_groups} "
                f"events={end}/{query_id.size}",
                flush=True,
            )
    return adc_gap, rpq_gap


def optional_quantile(values: np.ndarray, probability: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, probability))


def score_statistics(
    exact_gap: np.ndarray,
    predicted_gap: np.ndarray,
    mask: np.ndarray,
    margin_floor: float,
    tie_epsilon: float,
) -> dict[str, object]:
    exact = np.asarray(exact_gap[mask], dtype=np.float64)
    gap = np.asarray(predicted_gap[mask], dtype=np.float64)
    count = int(gap.size)
    if count == 0:
        return {"pairs": 0}
    ties = np.abs(gap) <= tie_epsilon
    strict = gap < -tie_epsilon
    nonpositive = gap <= tie_epsilon
    violation = np.maximum(-gap, 0.0)
    normalizer = np.maximum(exact, margin_floor)
    normalized_violation = violation / normalizer
    normalized_error = (gap - exact) / normalizer
    inverted_violation = violation[nonpositive]
    inverted_normalized = normalized_violation[nonpositive]
    return {
        "pairs": count,
        "strict_inversions": int(np.sum(strict)),
        "strict_inversion_rate": float(np.mean(strict)),
        "ties": int(np.sum(ties)),
        "tie_rate": float(np.mean(ties)),
        "nonpositive_inversions": int(np.sum(nonpositive)),
        "nonpositive_inversion_rate": float(np.mean(nonpositive)),
        "pairwise_accuracy": float(np.mean(gap > tie_epsilon)),
        "mean_predicted_gap": float(np.mean(gap)),
        "mean_gap_error": float(np.mean(gap - exact)),
        "rmse_gap_error": float(np.sqrt(np.mean(np.square(gap - exact)))),
        "margin_normalized_mse": float(np.mean(np.square(normalized_error))),
        "mean_violation": float(np.mean(violation)),
        "mean_margin_normalized_violation": float(np.mean(normalized_violation)),
        "inverted_violation_p50": optional_quantile(inverted_violation, 0.50),
        "inverted_violation_p95": optional_quantile(inverted_violation, 0.95),
        "inverted_normalized_violation_p50": optional_quantile(
            inverted_normalized, 0.50
        ),
        "inverted_normalized_violation_p95": optional_quantile(
            inverted_normalized, 0.95
        ),
    }


def comparison_statistics(
    exact_gap: np.ndarray,
    adc_gap: np.ndarray,
    rpq_gap: np.ndarray,
    mask: np.ndarray,
    tie_epsilon: float,
    rpq_name: str = "ood-rpq",
) -> dict[str, object]:
    adc_inverted = adc_gap[mask] <= tie_epsilon
    rpq_inverted = rpq_gap[mask] <= tie_epsilon
    count = int(adc_inverted.size)
    if count == 0:
        return {"pairs": 0}
    recovered = adc_inverted & ~rpq_inverted
    regressed = ~adc_inverted & rpq_inverted
    both = adc_inverted & rpq_inverted
    adc_count = int(np.sum(adc_inverted))
    adc_correct_count = count - adc_count
    recovered_count = int(np.sum(recovered))
    regressed_count = int(np.sum(regressed))
    rpq_count = int(np.sum(rpq_inverted))
    rpq_key = rpq_name.replace("-", "_")
    return {
        "pairs": count,
        "adc_inversions": adc_count,
        "adc_correct_pairs": adc_correct_count,
        f"{rpq_key}_inversions": rpq_count,
        "recovered_adc_inversions": recovered_count,
        "recovery_rate_among_adc_inversions": (
            float(recovered_count / adc_count) if adc_count else None
        ),
        "new_inversions_from_adc_correct": regressed_count,
        "new_inversion_rate_among_adc_correct": (
            float(regressed_count / adc_correct_count)
            if adc_correct_count else None
        ),
        "persistent_inversions": int(np.sum(both)),
        "net_inversions_removed": adc_count - rpq_count,
        "net_inversion_reduction_rate_over_all_pairs": float(
            (adc_count - rpq_count) / count
        ),
        "relative_inversion_reduction": (
            float(1.0 - rpq_count / adc_count)
            if adc_count else None
        ),
        "mean_exact_margin_recovered": (
            float(np.mean(exact_gap[mask][recovered]))
            if np.any(recovered) else None
        ),
        "mean_exact_margin_regressed": (
            float(np.mean(exact_gap[mask][regressed]))
            if np.any(regressed) else None
        ),
    }


def add_scope(
    output: dict[str, object],
    csv_rows: list[dict[str, object]],
    scope_type: str,
    scope_name: str,
    mask: np.ndarray,
    exact_gap: np.ndarray,
    adc_gap: np.ndarray,
    rpq_gap: np.ndarray,
    margin_floor: float,
    tie_epsilon: float,
    rpq_name: str,
) -> None:
    mode_gaps = {
        "full-precision": exact_gap,
        "adc": adc_gap,
        rpq_name: rpq_gap,
    }
    mode_stats = {
        mode: score_statistics(
            exact_gap, gap, mask, margin_floor, tie_epsilon
        )
        for mode, gap in mode_gaps.items()
    }
    comparison = comparison_statistics(
        exact_gap, adc_gap, rpq_gap, mask, tie_epsilon, rpq_name
    )
    comparison_key = f"adc_vs_{rpq_name.replace('-', '_')}"
    output[scope_name] = {"modes": mode_stats, comparison_key: comparison}
    for mode, values in mode_stats.items():
        csv_rows.append(
            {"scope_type": scope_type, "scope": scope_name, "mode": mode, **values}
        )


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0 or args.query_limit < 0
        or args.margin_floor <= 0.0 or args.tie_epsilon < 0.0
    ):
        raise ValueError("invalid query range or margin/tie parameters")
    if not args.rpq_name or args.rpq_name in {"adc", "full-precision"}:
        raise ValueError("--rpq-name must be nonempty and distinct from built-in modes")
    specs = [parse_event_spec(spec) for spec in args.event_run]
    pq_model = read_model(args.pq_model)
    residual_model = read_residual_pq_model(args.residual_model)
    if (
        pq_model.refined_subspaces != 0
        or residual_model.artifact_id != pq_model.artifact_id
        or residual_model.node_count != pq_model.node_count
        or residual_model.dim != pq_model.dim
    ):
        raise ValueError("primary and residual-PQ models are incompatible")
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    side_codes, side_width = read_residual_pq_side_codes(
        args.side_codes, residual_model
    )
    queries = fbin_memmap(args.query_fbin)
    if queries.shape[1] != pq_model.dim:
        raise ValueError("query dimension does not match the PQ model")
    query_end = (
        queries.shape[0]
        if args.query_limit == 0
        else min(queries.shape[0], args.query_start + args.query_limit)
    )
    if args.query_start >= query_end:
        raise ValueError("empty query range")
    events, raw_counts, origin_bits = load_union_events(
        specs, args.query_start, query_end
    )
    query_id, source, u_id, v_id, exact_gap, origin_mask = events
    if max(int(np.max(u_id)), int(np.max(v_id))) >= pq_model.node_count:
        raise ValueError("event node ID exceeds the database")
    adc_gap, rpq_gap = compute_gaps(
        events, queries, pq_model, residual_model, codes, side_codes,
        args.log_interval,
    )

    all_mask = np.ones(query_id.size, dtype=bool)
    scopes: dict[str, object] = {}
    csv_rows: list[dict[str, object]] = []
    add_scope(
        scopes, csv_rows, "overall", "all", all_mask, exact_gap, adc_gap,
        rpq_gap, args.margin_floor, args.tie_epsilon, args.rpq_name,
    )
    by_source: dict[str, object] = {}
    for code, name in CODE_TO_SOURCE.items():
        add_scope(
            by_source, csv_rows, "source", name, source == code,
            exact_gap, adc_gap, rpq_gap, args.margin_floor, args.tie_epsilon,
            args.rpq_name,
        )
    by_origin: dict[str, object] = {}
    for origin, bit in origin_bits.items():
        add_scope(
            by_origin, csv_rows, "origin", origin,
            (origin_mask & bit) != 0, exact_gap, adc_gap, rpq_gap,
            args.margin_floor, args.tie_epsilon, args.rpq_name,
        )

    quantile_edges = np.quantile(exact_gap.astype(np.float64), np.linspace(0, 1, 6))
    by_margin: dict[str, object] = {}
    for index in range(5):
        lower, upper = float(quantile_edges[index]), float(quantile_edges[index + 1])
        mask = exact_gap >= lower
        mask &= exact_gap <= upper if index == 4 else exact_gap < upper
        name = f"q{index * 20:02d}-q{(index + 1) * 20:02d}"
        add_scope(
            by_margin, csv_rows, "exact-margin-quintile", name, mask,
            exact_gap, adc_gap, rpq_gap, args.margin_floor, args.tie_epsilon,
            args.rpq_name,
        )
        by_margin[name]["exact_margin_range"] = [lower, upper]

    summary = {
        "version": 1,
        "method": "shared_dynamic_decision_pair_inversion_audit",
        "rpq_name": args.rpq_name,
        "definition": (
            "u is ordered before v by full precision; a scorer is inverted when "
            "score(u)-score(v) <= tie_epsilon"
        ),
        "sampling_interpretation": (
            "Rates are conditional on the decision-critical pairs emitted by the "
            "input trace runs. Per-source/per-kind trace caps and the protect-margin "
            "filter can enrich inversions, so the overall rate is not a random-pair "
            "database inversion probability. Recovery among ADC-inverted pairs and "
            "new inversions among ADC-correct pairs remain the primary comparisons."
        ),
        "query_range": [args.query_start, query_end],
        "queries_with_events": int(np.unique(query_id).size),
        "raw_event_counts": raw_counts,
        "union_pairs_after_deduplication": int(query_id.size),
        "margin_floor": args.margin_floor,
        "tie_epsilon": args.tie_epsilon,
        "side_bits_per_node": residual_model.side_bits_per_node,
        "side_bytes_per_node": side_width,
        "metric_lambda": residual_model.metric_lambda,
        "overall": scopes["all"],
        "by_source": by_source,
        "by_event_origin": by_origin,
        "by_exact_margin_quintile": by_margin,
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "residual_model": os.path.abspath(args.residual_model),
            "side_codes": os.path.abspath(args.side_codes),
            "queries": os.path.abspath(args.query_fbin),
            "event_runs": {
                origin: os.path.abspath(path) for origin, path in specs
            },
        },
    }
    output_json = Path(args.out_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "scope_type", "scope", "mode", "pairs", "strict_inversions",
        "strict_inversion_rate", "ties", "tie_rate",
        "nonpositive_inversions", "nonpositive_inversion_rate",
        "pairwise_accuracy", "mean_predicted_gap", "mean_gap_error",
        "rmse_gap_error", "margin_normalized_mse", "mean_violation",
        "mean_margin_normalized_violation", "inverted_violation_p50",
        "inverted_violation_p95", "inverted_normalized_violation_p50",
        "inverted_normalized_violation_p95",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(csv_rows)
    if args.out_events_npz:
        output_npz = Path(args.out_events_npz)
        output_npz.parent.mkdir(parents=True, exist_ok=True)
        rpq_gap_key = f"{args.rpq_name.replace('-', '_')}_gap"
        np.savez_compressed(
            output_npz,
            **{
                "query_id": query_id,
                "source": source,
                "u_id": u_id,
                "v_id": v_id,
                "exact_gap": exact_gap,
                "adc_gap": adc_gap,
                rpq_gap_key: rpq_gap,
                "origin_mask": origin_mask,
            },
        )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
