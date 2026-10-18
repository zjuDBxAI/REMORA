#!/usr/bin/env python3
"""Summarize Recall/QPS/latency and traversal metrics from dynamic searches."""

from __future__ import annotations

import argparse
import csv
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np

from fixed_candidate_io import read_id_matrix, row_recall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        help="LABEL=SEARCH_PREFIX; repeat once per method/beam run",
    )
    parser.add_argument("--topk", type=int, default=10)
    parser.add_argument("--query-start", type=int, default=0)
    parser.add_argument(
        "--target-recall", type=float, action="append", default=[]
    )
    parser.add_argument("--out-json", required=True)
    parser.add_argument("--out-csv", required=True)
    parser.add_argument(
        "--out-svg",
        help="optional dependency-free Recall/QPS and Recall/latency plot",
    )
    return parser.parse_args()


def load_run(spec: str, truth: np.ndarray, topk: int) -> dict[str, object]:
    if "=" not in spec:
        raise ValueError(f"invalid --run {spec!r}; expected LABEL=PREFIX")
    label, prefix = spec.split("=", 1)
    label, prefix = label.strip(), prefix.strip()
    if not label or not prefix:
        raise ValueError(f"invalid --run {spec!r}")
    ids = np.asarray(read_id_matrix(prefix + ".ids.ibin"))[:, :topk]
    if ids.shape != truth.shape:
        raise ValueError(
            f"{label} result shape {ids.shape} differs from truth {truth.shape}"
        )
    with open(prefix + ".summary.json", "r", encoding="utf-8") as handle:
        runtime = json.load(handle)
    recall = float(np.mean(row_recall(ids, truth)))
    return {
        "label": label,
        "prefix": os.path.abspath(prefix),
        "mode": str(runtime["mode"]),
        "beam_width": int(runtime["beam_width"]),
        "threads": int(runtime["threads"]),
        "warmup_queries": int(runtime.get("warmup_queries", 0)),
        "recall": recall,
        "qps": float(runtime["qps"]),
        "mean_latency_ms": float(runtime["mean_latency_ms"]),
        "p50_query_latency_ms": float(runtime["p50_query_latency_ms"]),
        "p95_query_latency_ms": float(runtime["p95_query_latency_ms"]),
        "p99_query_latency_ms": float(runtime["p99_query_latency_ms"]),
        "mean_comparisons": float(runtime["mean_comparisons"]),
        "mean_expansions": float(runtime["mean_expansions"]),
        "mean_visited_nodes": float(runtime["mean_visited_nodes"]),
        "mean_final_candidate_pool": float(
            runtime["mean_final_candidate_pool"]
        ),
        "rerank_enabled": bool(runtime.get("rerank_enabled", False)),
        "mean_reranked_candidates": float(
            runtime.get("mean_reranked_candidates", 0.0)
        ),
        "mean_exact_rerank_comparisons": float(
            runtime.get("mean_exact_rerank_comparisons", 0.0)
        ),
        "_id_sha256": hashlib.sha256(
            np.ascontiguousarray(ids, dtype="<u4").tobytes()
        ).hexdigest(),
    }


def aggregate_repetitions(
    raw_rows: list[dict[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[dict[str, object]]] = {}
    for row in raw_rows:
        key = (
            row["label"], row["mode"], row["beam_width"], row["threads"],
            row["warmup_queries"],
        )
        grouped.setdefault(key, []).append(row)

    deterministic_fields = (
        "recall",
        "mean_comparisons",
        "mean_expansions",
        "mean_visited_nodes",
        "mean_final_candidate_pool",
        "mean_reranked_candidates",
        "mean_exact_rerank_comparisons",
    )
    timing_fields = (
        "qps",
        "mean_latency_ms",
        "p50_query_latency_ms",
        "p95_query_latency_ms",
        "p99_query_latency_ms",
    )
    result: list[dict[str, object]] = []
    for key, repetitions in grouped.items():
        reference = repetitions[0]
        for candidate in repetitions[1:]:
            if candidate["_id_sha256"] != reference["_id_sha256"]:
                raise ValueError(
                    f"non-deterministic result IDs for {key}: "
                    f"{reference['prefix']} vs {candidate['prefix']}"
                )
            for field in deterministic_fields:
                if not np.isclose(
                    float(candidate[field]), float(reference[field]),
                    rtol=0.0, atol=1e-9,
                ):
                    raise ValueError(
                        f"non-deterministic {field} for {key}: "
                        f"{reference[field]} vs {candidate[field]}"
                    )

        row = {
            name: value for name, value in reference.items()
            if name != "_id_sha256"
        }
        row["result_id_sha256"] = reference["_id_sha256"]
        row["timing_repetitions"] = len(repetitions)
        row["timing_repetition_prefixes"] = ";".join(
            str(value["prefix"]) for value in repetitions
        )
        for field in timing_fields:
            values = np.asarray(
                [float(value[field]) for value in repetitions], dtype=np.float64
            )
            median = float(np.median(values))
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            row[field] = median
            row[field + "_min"] = float(np.min(values))
            row[field + "_max"] = float(np.max(values))
            row[field + "_std"] = std
            row[field + "_cv_percent"] = (
                100.0 * std / abs(mean) if mean != 0.0 else 0.0
            )
        result.append(row)
    return result


def pareto_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    result = []
    for row in rows:
        dominated = any(
            float(other["recall"]) >= float(row["recall"])
            and float(other["qps"]) >= float(row["qps"])
            and (
                float(other["recall"]) > float(row["recall"])
                or float(other["qps"]) > float(row["qps"])
            )
            for other in rows
            if other is not row
        )
        if not dominated:
            result.append(row)
    return sorted(result, key=lambda value: float(value["recall"]))


def write_frontier_svg(rows: list[dict[str, object]], output: Path) -> None:
    """Write a compact two-panel SVG without requiring matplotlib."""
    width, height = 1160, 470
    panel_width, panel_height = 500, 350
    top, left_first, gap = 55, 75, 80
    palette = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c"]
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)
    for values in by_label.values():
        values.sort(key=lambda value: float(value["recall"]))

    recalls = [float(row["recall"]) * 100.0 for row in rows]
    xmin = max(0.0, min(recalls) - 1.0)
    xmax = min(100.0, max(recalls) + 1.0)
    if xmax <= xmin:
        xmax = xmin + 1.0

    def scale(value: float, low: float, high: float, origin: float, span: float) -> float:
        return origin + (value - low) * span / (high - low)

    chunks = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:Arial,sans-serif;fill:#111827}.tick{font-size:12px;fill:#4b5563}.title{font-size:17px;font-weight:600}.legend{font-size:13px}.grid{stroke:#e5e7eb;stroke-width:1}.axis{stroke:#374151;stroke-width:1.2}</style>',
    ]
    panels = [
        ("qps", "Recall@10--QPS", "QPS", max(float(row["qps"]) for row in rows) * 1.08),
        (
            "mean_latency_ms",
            "Recall@10--Mean latency",
            "Latency (ms/query)",
            max(float(row["mean_latency_ms"]) for row in rows) * 1.08,
        ),
    ]
    for panel_index, (field, title, ylabel, ymax) in enumerate(panels):
        left = left_first + panel_index * (panel_width + gap)
        bottom = top + panel_height
        chunks.append(f'<text class="title" x="{left + panel_width / 2:.1f}" y="27" text-anchor="middle">{html.escape(title)}</text>')
        for tick_index in range(6):
            xvalue = xmin + (xmax - xmin) * tick_index / 5.0
            xpos = left + panel_width * tick_index / 5.0
            chunks.append(f'<line class="grid" x1="{xpos:.1f}" y1="{top}" x2="{xpos:.1f}" y2="{bottom}"/>')
            chunks.append(f'<text class="tick" x="{xpos:.1f}" y="{bottom + 20}" text-anchor="middle">{xvalue:.1f}</text>')
            yvalue = ymax * tick_index / 5.0
            ypos = bottom - panel_height * tick_index / 5.0
            chunks.append(f'<line class="grid" x1="{left}" y1="{ypos:.1f}" x2="{left + panel_width}" y2="{ypos:.1f}"/>')
            chunks.append(f'<text class="tick" x="{left - 10}" y="{ypos + 4:.1f}" text-anchor="end">{yvalue:.2f}</text>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{bottom}" x2="{left + panel_width}" y2="{bottom}"/>')
        chunks.append(f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{bottom}"/>')
        chunks.append(f'<text class="tick" x="{left + panel_width / 2:.1f}" y="{bottom + 43}" text-anchor="middle">Recall@10 (%)</text>')
        chunks.append(f'<text class="tick" transform="translate({left - 55},{top + panel_height / 2}) rotate(-90)" text-anchor="middle">{html.escape(ylabel)}</text>')
        for color_index, (label, values) in enumerate(sorted(by_label.items())):
            color = palette[color_index % len(palette)]
            points = []
            for row in values:
                xpos = scale(float(row["recall"]) * 100.0, xmin, xmax, left, panel_width)
                ypos = bottom - float(row[field]) * panel_height / ymax
                points.append(f"{xpos:.2f},{ypos:.2f}")
            chunks.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{color}" stroke-width="2.5"/>')
            for point in points:
                xpos, ypos = point.split(",")
                chunks.append(f'<circle cx="{xpos}" cy="{ypos}" r="3.6" fill="{color}"/>')

    legend_y = height - 13
    total_legend_width = max(1, len(by_label)) * 180
    legend_x = max(15, (width - total_legend_width) / 2)
    for color_index, label in enumerate(sorted(by_label)):
        color = palette[color_index % len(palette)]
        xpos = legend_x + color_index * 180
        chunks.append(f'<line x1="{xpos:.1f}" y1="{legend_y - 4}" x2="{xpos + 25:.1f}" y2="{legend_y - 4}" stroke="{color}" stroke-width="3"/>')
        chunks.append(f'<circle cx="{xpos + 12.5:.1f}" cy="{legend_y - 4}" r="3.5" fill="{color}"/>')
        chunks.append(f'<text class="legend" x="{xpos + 32:.1f}" y="{legend_y}">{html.escape(label)}</text>')
    chunks.append("</svg>")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(chunks) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.topk <= 0 or args.query_start < 0:
        raise ValueError("topk must be positive and query-start non-negative")
    parsed = []
    for spec in args.run:
        if "=" not in spec:
            raise ValueError(f"invalid --run {spec!r}")
        parsed.append(spec.split("=", 1))
    first_ids = read_id_matrix(parsed[0][1] + ".ids.ibin")
    query_count = int(first_ids.shape[0])
    truth_all = read_id_matrix(args.ground_truth, allow_trailing=True)
    if args.query_start + query_count > truth_all.shape[0]:
        raise ValueError("ground truth does not cover the requested query range")
    truth = np.asarray(
        truth_all[
            args.query_start : args.query_start + query_count, : args.topk
        ]
    )
    raw_rows = [load_run(spec, truth, args.topk) for spec in args.run]
    rows = aggregate_repetitions(raw_rows)
    by_label: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        by_label.setdefault(str(row["label"]), []).append(row)

    target_rows: dict[str, dict[str, object | None]] = {}
    for target in args.target_recall:
        if not 0.0 <= target <= 1.0:
            raise ValueError("target-recall must lie in [0,1]")
        choices: dict[str, object | None] = {}
        for label, label_rows in by_label.items():
            feasible = [
                row for row in label_rows if float(row["recall"]) >= target
            ]
            choices[label] = (
                max(feasible, key=lambda value: float(value["qps"]))
                if feasible
                else None
            )
        target_rows[f"{target:.6g}"] = choices

    summary = {
        "version": 1,
        "method": "dynamic_roargraph_recall_qps_frontier",
        "ground_truth": os.path.abspath(args.ground_truth),
        "query_start": int(args.query_start),
        "queries": query_count,
        "topk": int(args.topk),
        "timing_aggregation": {
            "center": "median_across_independent_process_runs",
            "variation": "min_max_sample_std_and_coefficient_of_variation",
            "deterministic_fields": [
                "result_ids", "recall", "comparisons", "expansions",
                "visited_nodes", "final_candidate_pool",
                "reranked_candidates", "exact_rerank_comparisons",
            ],
        },
        "rows": sorted(
            rows, key=lambda value: (str(value["label"]), int(value["beam_width"]))
        ),
        "pareto_frontier": {
            label: pareto_rows(label_rows)
            for label, label_rows in sorted(by_label.items())
        },
        "best_qps_at_target_recall": target_rows,
    }
    output_json = Path(args.out_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)

    output_csv = Path(args.out_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "label", "mode", "beam_width", "threads", "warmup_queries",
        "recall", "qps",
        "qps_min", "qps_max", "qps_std", "qps_cv_percent",
        "mean_latency_ms", "p50_query_latency_ms", "p95_query_latency_ms",
        "p99_query_latency_ms", "mean_comparisons", "mean_expansions",
        "mean_visited_nodes", "mean_final_candidate_pool",
        "rerank_enabled", "mean_reranked_candidates",
        "mean_exact_rerank_comparisons",
        "timing_repetitions", "result_id_sha256", "prefix",
        "timing_repetition_prefixes",
    ]
    with open(output_csv, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(summary["rows"])
    if args.out_svg:
        write_frontier_svg(rows, Path(args.out_svg))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
