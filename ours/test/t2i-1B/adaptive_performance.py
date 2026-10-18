"""Refine measured Recall gaps without assuming monotone graph search.

Every probe uses the full configured test set. Integer beam/depth settings
are bisected; interpolation is never used as a measured result. Limits and
unreachable portions of the requested Recall range remain visible in CSV.
"""
from __future__ import annotations

import math
import statistics


COVERAGE_FIELDS = ["experiment", "primary", "variant", "side_bits", "mode",
    "recall_min", "recall_max", "target_gap", "measured_points", "attempted_points",
    "min_measured_recall", "max_measured_recall", "max_recall_gap", "pending_points",
    "lower_endpoint_bracketed", "upper_endpoint_reached", "status", "reason"]
GAP_FIELDS = ["experiment", "primary", "variant", "side_bits", "mode",
    "recall_low", "recall_high", "gap", "target_gap", "status", "reason"]
JOB_FIELDS = ["experiment", "primary", "variant", "side_bits", "beam", "rerank_depth", "repeat"]


def curve_specs(config, experiments):
    curves = []
    def add(number, method, variant, bits, mode, domain):
        curves.append(dict(experiment=number, primary=method, variant=variant,
            side_bits=bits, mode=mode, beam_min=min(domain), beam_max=max(domain)))
    if 1 in experiments or 4 in experiments:
        for method in config["methods"]:
            for variant, bits in (("primary", 0), (config["deploy_variant"], 64)):
                for mode in ("no-rerank", "rerank"):
                    add(1, method, variant, bits, mode, config["beams"])
        if config["include_fp32_reference"]:
            add(1, "full-precision", "fp32", 0, "no-rerank", config["beams"])
    if 5 in experiments:
        for method in config["methods"]:
            for bits in config["budgets"]:
                add(5, method, "joint-grouping", bits, "no-rerank", config["bit_budget_beams"])
    return curves


def job(curve, beam, depth, repeat=1):
    return dict(experiment=curve["experiment"], primary=curve["primary"],
        variant=curve["variant"], side_bits=curve["side_bits"], beam=beam,
        rerank_depth=depth, repeat=repeat)


def matches(row, curve):
    return (all(str(row.get(k)) == str(curve[k]) for k in
                ("experiment", "primary", "variant", "side_bits")) and
            bool(int(row.get("rerank_depth", 0))) == (curve["mode"] == "rerank"))


def initial_jobs(config, curve):
    beams = sorted({curve["beam_min"], curve["beam_max"],
        *(max(curve["beam_min"], min(b, curve["beam_max"])) for b in config["adaptive_coarse_beams"])})
    pairs = []
    for beam in beams:
        depths = sorted({beam, *(min(d, beam) for d in config["adaptive_coarse_depths"])}) if curve["mode"] == "rerank" else [0]
        pairs.extend((beam, depth) for depth in depths)
    # Keep both domain endpoints even with a deliberately tiny probe budget.
    if len(pairs) > config["adaptive_max_points_per_curve"]:
        pairs = [pairs[0], pairs[-1], *pairs[1:-1]][:config["adaptive_max_points_per_curve"]]
    return [job(curve, *pair) for pair in pairs]


def initial_performance_jobs(config, experiments):
    return [j for curve in curve_specs(config, experiments) for j in initial_jobs(config, curve)]


def measured_points(rows, curve):
    groups = {}
    for row in rows:
        if matches(row, curve) and row.get("status") == "OK":
            groups.setdefault((int(row["beam"]), int(row["rerank_depth"])), {})[int(row["repeat"])] = row
    points = []
    for (beam, depth), repeats in groups.items():
        recalls = [float(r["recall_at_10"]) for r in repeats.values()]
        if any(not math.isfinite(r) or not 0 <= r <= 1 for r in recalls):
            raise ValueError("Invalid measured Recall")
        if max(recalls)-min(recalls) > 1e-8:
            raise ValueError("Recall changed between deterministic timing repeats")
        points.append(dict(beam=beam, depth=depth, recall=recalls[0],
            qps=statistics.median(float(r["qps"]) for r in repeats.values()), repeats=repeats))
    return sorted(points, key=lambda p: (p["recall"], -p["qps"], p["beam"], p["depth"]))


def midpoint_candidates(left, right, curve):
    b0, b1 = sorted((left["beam"], right["beam"]))
    d0, d1 = sorted((left["depth"], right["depth"]))
    bm, dm = (b0+b1)//2, (d0+d1)//2
    candidates = [(bm, dm), (bm, left["depth"]), (bm, right["depth"]),
                  (left["beam"], dm), (right["beam"], dm)]
    # Extra cuts allow refinement after non-monotone/plateau observations.
    bs = {b0, b1, bm, (3*b0+b1)//4, (b0+3*b1)//4}
    ds = {d0, d1, dm, (3*d0+d1)//4, (d0+3*d1)//4}
    candidates.extend(sorted(((b, d) for b in bs for d in ds),
        key=lambda pair: (abs(pair[0]-bm)+abs(pair[1]-dm), pair)))
    unique = []
    for beam, depth in candidates:
        if curve["mode"] == "no-rerank": depth = 0
        pair = (beam, depth)
        if (curve["beam_min"] <= beam <= curve["beam_max"] and
            (depth == 0 if curve["mode"] == "no-rerank" else 10 <= depth <= beam) and pair not in unique):
            unique.append(pair)
    return unique


def refinement(config, curve, rows, complete_only=False):
    local = [r for r in rows if matches(r, curve)]
    attempted = {(int(r["beam"]), int(r["rerank_depth"])) for r in local}
    points = measured_points(local, curve)
    if complete_only: points = [p for p in points if len(p["repeats"]) >= config["repeats"]]
    low, high = config["recall_range"]; tolerance = config["recall_gap"]
    recalls = sorted({low, high, *(p["recall"] for p in points if low <= p["recall"] <= high)})
    intervals = [(a, b) for a, b in zip(recalls, recalls[1:]) if b-a > tolerance+1e-12]
    base = {k: curve[k] for k in ("experiment", "primary", "variant", "side_bits", "mode")}
    pending, gaps = [], []
    for a, b in intervals:
        reason = "PARAMETER_RESOLUTION_EXHAUSTED"
        candidates = []
        left = [p for p in points if p["recall"] <= a+1e-12]
        right = [p for p in points if p["recall"] >= b-1e-12]
        if left and right:
            candidates = midpoint_candidates(max(left,key=lambda p:(p["recall"],p["qps"])), right[0], curve)
        elif not right:
            maximum = curve["beam_max"]
            candidates = [(maximum, maximum if curve["mode"] == "rerank" else 0)]
            reason = "UPPER_RANGE_NOT_REACHED_WITHIN_BEAM_LIMIT"
        elif not left:
            candidates = [(curve["beam_min"], 10 if curve["mode"] == "rerank" else 0)]
            reason = "LOWER_RANGE_NOT_REACHED_WITHIN_BEAM_LIMIT"
        if not points:
            reason = "NO_SUCCESSFUL_MEASUREMENTS"
            if complete_only and any(r.get("status") == "OK" for r in local): reason = "INCOMPLETE_TIMING_REPEATS"
            if any(r.get("status") == "MEMORY_EXCEEDED" for r in local): reason = "MEMORY_EXCEEDED"
        candidate = next((pair for pair in candidates if pair not in attempted), None)
        if len(attempted) >= config["adaptive_max_points_per_curve"]:
            candidate = None; reason = "POINT_LIMIT_REACHED"
        if candidate is not None:
            pending.append(job(curve, *candidate)); attempted.add(candidate)
            reason = "MEASURE_INTEGER_MIDPOINT"; state = "PENDING"
        else: state = "UNRESOLVED"
        gaps.append(dict(base, recall_low=a, recall_high=b, gap=b-a,
                         target_gap=tolerance, status=state, reason=reason))
    reasons = sorted({g["reason"] for g in gaps if g["status"] == "UNRESOLVED"})
    coverage = dict(base, recall_min=low, recall_max=high, target_gap=tolerance,
        measured_points=len(points), attempted_points=len(attempted)-len(pending),
        min_measured_recall=min((p["recall"] for p in points), default=""),
        max_measured_recall=max((p["recall"] for p in points), default=""),
        max_recall_gap=max((b-a for a,b in zip(recalls,recalls[1:])),default=high-low),
        lower_endpoint_bracketed=any(p["recall"]<=low for p in points),
        upper_endpoint_reached=any(p["recall"]>=high for p in points),
        pending_points=len(pending), status="COVERED" if not intervals else "REFINING" if pending else "UNRESOLVED",
        reason=";".join(reasons))
    return pending, coverage, gaps


def repeat_jobs(config, curve, rows, extra=False):
    jobs = []
    for point in measured_points(rows, curve):
        repeats = point["repeats"]
        target = config["repeats"]
        if extra:
            if len(repeats) < target: continue
            qps = [float(r["qps"]) for r in repeats.values()]
            cv = statistics.pstdev(qps)/statistics.median(qps) if statistics.median(qps) else math.inf
            if cv <= config["adaptive_qps_cv"]: continue
            target = config["adaptive_extra_repeats"]
        jobs.extend(job(curve, point["beam"], point["depth"], repeat)
                    for repeat in range(1,target+1) if repeat not in repeats)
    return jobs
