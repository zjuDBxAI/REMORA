#!/usr/bin/env python3
"""Audit deterministic score/rank certificates for residual-PQ scorers."""

from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from audit_dynamic_inversions import (
    CODE_TO_SOURCE,
    load_union_events,
    parse_event_spec,
    transform_query_cpp,
)
from drb_io import fbin_memmap, read_codes, read_model, transform_vectors
from residual_pq_baseline import (
    coded_residual_reconstruction,
    primary_conditioning_lut,
    reconstruct_primary_pq,
    residual_codebook_lut,
    residual_query_coordinates,
    transform_grouped_residual,
)
from residual_pq_baseline_io import (
    ResidualPQModel,
    read_residual_pq_model,
    read_residual_pq_side_codes,
)


@dataclass
class ResidualRun:
    name: str
    model_path: str
    side_codes_path: str
    model: ResidualPQModel
    side_codes: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pq-model", required=True)
    parser.add_argument("--codes", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument(
        "--residual-run",
        action="append",
        required=True,
        help="NAME=MODEL.npz:SIDE_CODES.bin; repeat for Euclidean/OOD RPQ",
    )
    parser.add_argument(
        "--event-run",
        action="append",
        required=True,
        help="ORIGIN=EVENTS.csv; repeat for the traces whose pairs form the audit set",
    )
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument("--query-limit", type=int, default=0)
    parser.add_argument("--node-batch-size", type=int, default=65536)
    parser.add_argument("--tie-epsilon", type=float, default=0.0)
    parser.add_argument(
        "--certificate-atol",
        type=float,
        default=1e-5,
        help="numerical tolerance when checking certificate inequalities",
    )
    parser.add_argument("--log-interval", type=int, default=1000)
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    return parser.parse_args()


def parse_residual_spec(spec: str) -> tuple[str, str, str]:
    if "=" not in spec or ":" not in spec:
        raise ValueError(
            f"invalid --residual-run {spec!r}; expected NAME=MODEL.npz:SIDE.bin"
        )
    name, rest = (part.strip() for part in spec.split("=", 1))
    model_path, side_path = (part.strip() for part in rest.split(":", 1))
    if not name or not model_path or not side_path:
        raise ValueError(f"invalid --residual-run {spec!r}")
    if name in {"full-precision", "adc"}:
        raise ValueError("residual run names must not shadow full-precision/adc")
    return name, model_path, side_path


def load_residual_runs(
    specs: list[str],
    pq_model,
) -> list[ResidualRun]:
    result: list[ResidualRun] = []
    names: set[str] = set()
    for spec in specs:
        name, model_path, side_path = parse_residual_spec(spec)
        if name in names:
            raise ValueError(f"duplicate residual run name: {name}")
        names.add(name)
        model = read_residual_pq_model(model_path)
        if (
            model.artifact_id != pq_model.artifact_id
            or model.node_count != pq_model.node_count
            or model.dim != pq_model.dim
        ):
            raise ValueError(f"{name}: residual model is incompatible with PQ model")
        side_codes, _width = read_residual_pq_side_codes(side_path, model)
        result.append(
            ResidualRun(
                name=name,
                model_path=model_path,
                side_codes_path=side_path,
                model=model,
                side_codes=side_codes,
            )
        )
    return result


def quantile_or_none(values: np.ndarray, probability: float) -> float | None:
    if values.size == 0:
        return None
    return float(np.quantile(values, probability))


def summarize_values(values: np.ndarray) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {
            "mean": None,
            "p50": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "mean": float(np.mean(array)),
        "p50": quantile_or_none(array, 0.50),
        "p90": quantile_or_none(array, 0.90),
        "p95": quantile_or_none(array, 0.95),
        "p99": quantile_or_none(array, 0.99),
        "max": float(np.max(array)),
    }


def metric_quadratic_norm(blocks: np.ndarray, matrices: np.ndarray) -> np.ndarray:
    values = np.asarray(blocks, dtype=np.float64)
    metric = np.asarray(matrices, dtype=np.float64)
    squared = np.einsum("bgd,gde,bge->b", values, metric, values, optimize=True)
    return np.sqrt(np.maximum(squared, 0.0)).astype(np.float32)


def compute_node_etas(
    node_ids: np.ndarray,
    base: np.ndarray,
    pq_model,
    codes: np.ndarray,
    runs: list[ResidualRun],
    batch_size: int,
    log_interval: int,
) -> dict[str, np.ndarray]:
    node_ids = np.asarray(node_ids, dtype=np.uint32)
    result = {
        "adc": np.full(pq_model.node_count, np.nan, dtype=np.float32),
    }
    for run in runs:
        result[run.name] = np.full(pq_model.node_count, np.nan, dtype=np.float32)

    identity_metric = None
    processed = 0
    for start in range(0, node_ids.size, batch_size):
        end = min(node_ids.size, start + batch_size)
        ids = node_ids[start:end]
        primary_codes = np.asarray(codes[ids], dtype=np.uint8)
        base_t = transform_vectors(np.asarray(base[ids], dtype=np.float32), pq_model)
        residual = base_t - reconstruct_primary_pq(primary_codes, pq_model)
        result["adc"][ids] = np.linalg.norm(residual, axis=1).astype(np.float32)

        for run in runs:
            model = run.model
            if identity_metric is None or identity_metric.shape != (
                model.groups,
                model.group_dim,
                model.group_dim,
            ):
                identity_metric = np.broadcast_to(
                    np.eye(model.group_dim, dtype=np.float32),
                    (model.groups, model.group_dim, model.group_dim),
                )
            residual_blocks = residual.reshape(-1, model.groups, model.group_dim)
            coded_residual = (
                transform_grouped_residual(
                    residual_blocks, model.residual_transform
                )
                if model.has_residual_transform
                else residual_blocks
            )
            side = np.asarray(run.side_codes[ids], dtype=np.uint8)
            approx = coded_residual_reconstruction(
                side, primary_codes, model
            ).reshape(-1, model.groups, model.group_dim)
            error = coded_residual - approx
            metric = (
                identity_metric
                if model.metric_matrices is None
                else np.asarray(model.metric_matrices, dtype=np.float32)
            )
            result[run.name][ids] = metric_quadratic_norm(error, metric)

        processed += ids.size
        if log_interval > 0 and (
            processed % log_interval == 0 or processed == node_ids.size
        ):
            print(
                f"[certificate-audit] node etas {processed}/{node_ids.size}",
                flush=True,
            )
    return result


def compute_query_lh(
    query_ids: np.ndarray,
    queries: np.ndarray,
    pq_model,
    runs: list[ResidualRun],
    log_interval: int,
) -> dict[str, np.ndarray]:
    max_query_id = int(np.max(query_ids))
    result = {
        "adc": np.full(max_query_id + 1, np.nan, dtype=np.float32),
    }
    inverse_metrics: dict[str, np.ndarray] = {}
    for run in runs:
        result[run.name] = np.full(max_query_id + 1, np.nan, dtype=np.float32)
        inverse_metrics[run.name] = np.linalg.inv(
            np.asarray(run.model.metric_matrices, dtype=np.float64)
        )

    for index, qid_raw in enumerate(np.asarray(query_ids, dtype=np.int64)):
        qid = int(qid_raw)
        query_t = transform_query_cpp(np.asarray(queries[qid], dtype=np.float32), pq_model)
        result["adc"][qid] = float(np.linalg.norm(query_t.astype(np.float64)))
        for run in runs:
            blocks = residual_query_coordinates(
                query_t, run.model
            ).reshape(run.model.groups, run.model.group_dim)
            inv_metric = inverse_metrics[run.name]
            squared = np.einsum("gd,gde,ge->", blocks, inv_metric, blocks, optimize=True)
            result[run.name][qid] = float(np.sqrt(max(float(squared), 0.0)))
        if log_interval > 0 and (
            (index + 1) % log_interval == 0 or index + 1 == query_ids.size
        ):
            print(
                f"[certificate-audit] query factors {index + 1}/{query_ids.size}",
                flush=True,
            )
    return result


def score_adc_nodes(
    node_ids: np.ndarray,
    adc_lut: np.ndarray,
    codes: np.ndarray,
) -> np.ndarray:
    node_codes = np.asarray(codes[node_ids], dtype=np.uint8)
    subspaces = np.arange(adc_lut.shape[0], dtype=np.int64)[None, :]
    return np.sum(adc_lut[subspaces, node_codes], axis=1, dtype=np.float64).astype(
        np.float32
    )


def score_rpq_nodes(
    node_ids: np.ndarray,
    residual_lut: np.ndarray,
    side_codes: np.ndarray,
) -> np.ndarray:
    side = np.asarray(side_codes[node_ids], dtype=np.uint8)
    scores = np.zeros(node_ids.size, dtype=np.float32)
    for group in range(residual_lut.shape[0]):
        scores += residual_lut[group, side[:, group]]
    return scores


def compute_predicted_gaps(
    events: tuple[np.ndarray, ...],
    queries: np.ndarray,
    pq_model,
    codes: np.ndarray,
    runs: list[ResidualRun],
    log_interval: int,
) -> dict[str, np.ndarray]:
    query_id, _source, u_id, v_id, _exact_gap, _origin = events
    predicted = {
        "full-precision": _exact_gap.astype(np.float32, copy=True),
        "adc": np.empty(query_id.size, dtype=np.float32),
    }
    for run in runs:
        predicted[run.name] = np.empty(query_id.size, dtype=np.float32)

    centroid64 = np.asarray(pq_model.centroids, dtype=np.float64)
    boundaries = np.flatnonzero(np.r_[True, query_id[1:] != query_id[:-1], True])
    for boundary_index in range(boundaries.size - 1):
        begin = int(boundaries[boundary_index])
        end = int(boundaries[boundary_index + 1])
        qid = int(query_id[begin])
        query_t = transform_query_cpp(np.asarray(queries[qid], dtype=np.float32), pq_model)
        primary_blocks = query_t.reshape(pq_model.subspaces, pq_model.dsub)
        adc_lut = np.sum(
            centroid64 * primary_blocks.astype(np.float64)[:, None, :],
            axis=2,
            dtype=np.float64,
        ).astype(np.float32)
        u_adc = score_adc_nodes(u_id[begin:end], adc_lut, codes)
        v_adc = score_adc_nodes(v_id[begin:end], adc_lut, codes)
        predicted["adc"][begin:end] = np.asarray(u_adc - v_adc, dtype=np.float32)

        for run in runs:
            query_coordinates = residual_query_coordinates(query_t, run.model)
            residual_lut = residual_codebook_lut(
                query_coordinates, run.model
            )
            conditional_lut = primary_conditioning_lut(
                query_coordinates, run.model
            )
            u_corr = score_rpq_nodes(u_id[begin:end], residual_lut, run.side_codes)
            v_corr = score_rpq_nodes(v_id[begin:end], residual_lut, run.side_codes)
            if conditional_lut is not None:
                u_corr += score_adc_nodes(
                    u_id[begin:end], conditional_lut, codes
                )
                v_corr += score_adc_nodes(
                    v_id[begin:end], conditional_lut, codes
                )
            predicted[run.name][begin:end] = np.asarray(
                u_adc + u_corr - v_adc - v_corr, dtype=np.float32
            )
        if log_interval > 0 and (
            (boundary_index + 1) % log_interval == 0
            or boundary_index + 1 == boundaries.size - 1
        ):
            print(
                f"[certificate-audit] scored queries "
                f"{boundary_index + 1}/{boundaries.size - 1}",
                flush=True,
            )
    return predicted


def certificate_widths_for_mode(
    mode: str,
    query_id: np.ndarray,
    u_id: np.ndarray,
    v_id: np.ndarray,
    etas: dict[str, np.ndarray],
    lhs: dict[str, np.ndarray],
) -> np.ndarray:
    if mode == "full-precision":
        return np.zeros(query_id.size, dtype=np.float32)
    return np.asarray(
        lhs[mode][query_id] * (etas[mode][u_id] + etas[mode][v_id]),
        dtype=np.float32,
    )


def certificate_statistics(
    exact_gap: np.ndarray,
    predicted_gap: np.ndarray,
    width: np.ndarray,
    mask: np.ndarray,
    tie_epsilon: float,
    certificate_atol: float,
) -> dict[str, object]:
    exact = np.asarray(exact_gap[mask], dtype=np.float64)
    predicted = np.asarray(predicted_gap[mask], dtype=np.float64)
    bound = np.asarray(width[mask], dtype=np.float64)
    count = int(exact.size)
    if count == 0:
        return {"pairs": 0}
    inverted = predicted <= tie_epsilon
    pair_error = np.abs(predicted - exact)
    bound_violated = pair_error > bound + certificate_atol
    certified = exact > bound + certificate_atol
    certified_inverted = certified & inverted
    uncertified = ~certified
    return {
        "pairs": count,
        "inversions": int(np.sum(inverted)),
        "inversion_rate": float(np.mean(inverted)),
        "pairwise_accuracy": float(np.mean(~inverted)),
        "certificate_bound_violations": int(np.sum(bound_violated)),
        "certificate_bound_violation_rate": float(np.mean(bound_violated)),
        "certified_pairs": int(np.sum(certified)),
        "certified_pair_rate": float(np.mean(certified)),
        "certified_inversions": int(np.sum(certified_inverted)),
        "certified_inversion_rate_among_certified": (
            float(np.mean(inverted[certified])) if np.any(certified) else None
        ),
        "uncertified_inversion_rate": (
            float(np.mean(inverted[uncertified])) if np.any(uncertified) else None
        ),
        "mean_pair_error": float(np.mean(pair_error)),
        "pair_error": summarize_values(pair_error),
        "certificate_width": summarize_values(bound),
        "width_to_margin_ratio": summarize_values(
            bound / np.maximum(exact, np.finfo(np.float64).tiny)
        ),
        "certified_margin": summarize_values(exact[certified]),
    }


def write_csv(
    path: str,
    rows: list[dict[str, object]],
) -> None:
    fields = [
        "scope_type",
        "scope",
        "mode",
        "pairs",
        "inversions",
        "inversion_rate",
        "pairwise_accuracy",
        "certificate_bound_violations",
        "certificate_bound_violation_rate",
        "certified_pairs",
        "certified_pair_rate",
        "certified_inversions",
        "certified_inversion_rate_among_certified",
        "uncertified_inversion_rate",
        "mean_pair_error",
        "certificate_width_p50",
        "certificate_width_p95",
        "width_to_margin_ratio_p50",
        "width_to_margin_ratio_p95",
    ]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    if (
        args.query_start < 0
        or args.query_limit < 0
        or args.node_batch_size <= 0
        or args.tie_epsilon < 0.0
        or args.certificate_atol < 0.0
    ):
        raise ValueError("invalid range, batch size, or tolerance")

    pq_model = read_model(args.pq_model)
    if pq_model.refined_subspaces != 0:
        raise ValueError("--pq-model must be the frozen primary model")
    codes = read_codes(args.codes, expected_artifact_id=pq_model.artifact_id)
    base = fbin_memmap(args.base_fbin, pq_model.node_count)
    queries = fbin_memmap(args.query_fbin)
    if base.shape != (pq_model.node_count, pq_model.dim):
        raise ValueError("base vectors do not match the PQ model")
    if queries.shape[1] != pq_model.dim:
        raise ValueError("query vectors do not match the PQ model")

    runs = load_residual_runs(args.residual_run, pq_model)
    event_specs = [parse_event_spec(spec) for spec in args.event_run]
    query_end = (
        queries.shape[0]
        if args.query_limit == 0
        else min(queries.shape[0], args.query_start + args.query_limit)
    )
    if args.query_start >= query_end:
        raise ValueError("empty query range")
    events, raw_counts, origin_bits = load_union_events(
        event_specs, args.query_start, query_end
    )
    query_id, source, u_id, v_id, exact_gap, origin_mask = events
    if int(np.max(query_id)) >= queries.shape[0]:
        raise ValueError("event query ID exceeds query file")
    if max(int(np.max(u_id)), int(np.max(v_id))) >= pq_model.node_count:
        raise ValueError("event node ID exceeds database")

    unique_nodes = np.unique(np.concatenate([u_id, v_id]).astype(np.uint32))
    unique_queries = np.unique(query_id.astype(np.int64))
    print(
        f"[certificate-audit] pairs={query_id.size} "
        f"queries={unique_queries.size} unique_nodes={unique_nodes.size}",
        flush=True,
    )
    etas = compute_node_etas(
        unique_nodes,
        base,
        pq_model,
        codes,
        runs,
        args.node_batch_size,
        args.log_interval,
    )
    lhs = compute_query_lh(unique_queries, queries, pq_model, runs, args.log_interval)
    predicted = compute_predicted_gaps(
        events,
        queries,
        pq_model,
        codes,
        runs,
        args.log_interval,
    )

    modes = ["full-precision", "adc"] + [run.name for run in runs]
    widths = {
        mode: certificate_widths_for_mode(mode, query_id, u_id, v_id, etas, lhs)
        for mode in modes
    }

    summary: dict[str, object] = {
        "version": 1,
        "method": "residual_pq_score_certificate_audit",
        "definition": {
            "score_certificate": (
                "|s_i(q)-s_tilde_i(q)| <= sqrt(q^T H^-1 q) "
                "* sqrt(e_i^T H e_i)"
            ),
            "certified_pair": (
                "exact_gap > certificate_width + certificate_atol; such pairs "
                "are guaranteed not to invert up to the reported numerical tolerance"
            ),
            "certificate_width": (
                "sqrt(q^T H^-1 q) * (eta_u + eta_v), with eta_i=sqrt(e_i^T H e_i)"
            ),
        },
        "query_range": [args.query_start, query_end],
        "queries_with_events": int(unique_queries.size),
        "pairs": int(query_id.size),
        "unique_nodes": int(unique_nodes.size),
        "tie_epsilon": args.tie_epsilon,
        "certificate_atol": args.certificate_atol,
        "raw_event_counts": raw_counts,
        "modes": {},
        "by_source": {},
        "by_event_origin": {},
        "node_eta": {},
        "query_lh": {},
        "files": {
            "pq_model": os.path.abspath(args.pq_model),
            "codes": os.path.abspath(args.codes),
            "base_fbin": os.path.abspath(args.base_fbin),
            "query_fbin": os.path.abspath(args.query_fbin),
            "event_runs": {
                origin: os.path.abspath(path) for origin, path in event_specs
            },
            "residual_runs": {
                run.name: {
                    "model": os.path.abspath(run.model_path),
                    "side_codes": os.path.abspath(run.side_codes_path),
                    "metric_lambda": run.model.metric_lambda,
                    "assignment_metric": run.model.assignment_metric,
                    "side_bits_per_node": run.model.side_bits_per_node,
                }
                for run in runs
            },
        },
    }

    csv_rows: list[dict[str, object]] = []
    all_mask = np.ones(query_id.size, dtype=bool)
    for mode in modes:
        stats = certificate_statistics(
            exact_gap,
            predicted[mode],
            widths[mode],
            all_mask,
            args.tie_epsilon,
            args.certificate_atol,
        )
        summary["modes"][mode] = stats
        csv_rows.append(
            {
                "scope_type": "overall",
                "scope": "all",
                "mode": mode,
                **stats,
                "certificate_width_p50": stats["certificate_width"]["p50"],
                "certificate_width_p95": stats["certificate_width"]["p95"],
                "width_to_margin_ratio_p50": stats["width_to_margin_ratio"]["p50"],
                "width_to_margin_ratio_p95": stats["width_to_margin_ratio"]["p95"],
            }
        )
        if mode != "full-precision":
            eta_values = etas[mode][unique_nodes]
            lh_values = lhs[mode][unique_queries]
            summary["node_eta"][mode] = summarize_values(eta_values)
            summary["query_lh"][mode] = summarize_values(lh_values)

    by_source: dict[str, object] = {}
    for code, source_name in CODE_TO_SOURCE.items():
        mask = source == code
        source_stats = {}
        for mode in modes:
            stats = certificate_statistics(
                exact_gap,
                predicted[mode],
                widths[mode],
                mask,
                args.tie_epsilon,
                args.certificate_atol,
            )
            source_stats[mode] = stats
            csv_rows.append(
                {
                    "scope_type": "source",
                    "scope": source_name,
                    "mode": mode,
                    **stats,
                    "certificate_width_p50": (
                        stats["certificate_width"]["p50"]
                        if stats.get("pairs", 0) else None
                    ),
                    "certificate_width_p95": (
                        stats["certificate_width"]["p95"]
                        if stats.get("pairs", 0) else None
                    ),
                    "width_to_margin_ratio_p50": (
                        stats["width_to_margin_ratio"]["p50"]
                        if stats.get("pairs", 0) else None
                    ),
                    "width_to_margin_ratio_p95": (
                        stats["width_to_margin_ratio"]["p95"]
                        if stats.get("pairs", 0) else None
                    ),
                }
            )
        by_source[source_name] = source_stats
    summary["by_source"] = by_source

    by_origin: dict[str, object] = {}
    for origin, bit in origin_bits.items():
        mask = (origin_mask & bit) != 0
        origin_stats = {}
        for mode in modes:
            origin_stats[mode] = certificate_statistics(
                exact_gap,
                predicted[mode],
                widths[mode],
                mask,
                args.tie_epsilon,
                args.certificate_atol,
            )
        by_origin[origin] = origin_stats
    summary["by_event_origin"] = by_origin

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    write_csv(args.out_csv, csv_rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
