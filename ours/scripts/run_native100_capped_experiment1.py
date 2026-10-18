"""Resume the unchanged experiment-1 measurements with a bounded scheduler.

Only the pending measurement selection changes. The frozen configuration,
scoring binary, graph, model, CPU affinity, IO policy and measurement IDs remain
unchanged, allowing completed points to be reused without timing them again.
Completed over-budget points remain in the raw CSV/files, but not the main
bounded comparison. The scheduling override is recorded explicitly.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "ours/test/t2i-10M"
sys.path[:0] = [str(HERE), str(ROOT / "ours/scripts"), str(ROOT / "ours/python")]

import adaptive_performance as adaptive
from protocol import load_config, read_csv, write_csv

ORIGINAL_INITIAL = adaptive.initial_jobs
ORIGINAL_REPEATS = adaptive.repeat_jobs
ORIGINAL_MIDPOINTS = adaptive.midpoint_candidates


def eligible(row, cap):
    return int(row.get("rerank_depth", 0)) <= cap


def bounded_initial(config, curve, cap):
    result, seen = [], set()
    for item in ORIGINAL_INITIAL(config, curve):
        item = dict(item, rerank_depth=min(item["rerank_depth"], cap))
        key = tuple(item[k] for k in adaptive.JOB_FIELDS)
        if key not in seen:
            result.append(item); seen.add(key)
    return result


def bounded_refinement(config, curve, rows, cap, complete_only=False):
    # Same Recall-gap rule as the existing scheduler; enforce the cap in
    # midpoint generation AND in the upper-endpoint fallback, not just probes.
    local = [r for r in rows if adaptive.matches(r, curve) and eligible(r, cap)]
    attempted = {(int(r["beam"]), int(r["rerank_depth"])) for r in local}
    points = adaptive.measured_points(local, curve)
    if complete_only:
        points = [p for p in points if len(p["repeats"]) >= config["repeats"]]
    low, high = config["recall_range"]; tolerance = config["recall_gap"]
    recalls = sorted({low, high, *(p["recall"] for p in points if low <= p["recall"] <= high)})
    intervals = [(a, b) for a, b in zip(recalls, recalls[1:]) if b - a > tolerance + 1e-12]
    base = {k: curve[k] for k in ("experiment", "primary", "variant", "side_bits", "mode")}
    pending, gaps = [], []
    for a, b in intervals:
        reason = "PARAMETER_RESOLUTION_EXHAUSTED"
        candidates = []
        left = [p for p in points if p["recall"] <= a + 1e-12]
        right = [p for p in points if p["recall"] >= b - 1e-12]
        if left and right:
            candidates = ORIGINAL_MIDPOINTS(max(left, key=lambda p: (p["recall"], p["qps"])), right[0], curve)
        elif not right:
            maximum = curve["beam_max"]
            candidates = [(maximum, min(maximum, cap) if curve["mode"] == "rerank" else 0)]
            reason = "UPPER_RANGE_NOT_REACHED_WITHIN_BEAM_AND_RERANK_LIMIT"
        elif not left:
            candidates = [(curve["beam_min"], 10 if curve["mode"] == "rerank" else 0)]
            reason = "LOWER_RANGE_NOT_REACHED_WITHIN_BEAM_LIMIT"
        if not points:
            reason = "NO_SUCCESSFUL_MEASUREMENTS"
            if complete_only and any(r.get("status") == "OK" for r in local):
                reason = "INCOMPLETE_TIMING_REPEATS"
            if any(r.get("status") == "MEMORY_EXCEEDED" for r in local):
                reason = "MEMORY_EXCEEDED"
        candidates = [(beam, min(depth, cap)) for beam, depth in candidates]
        candidate = next((pair for pair in candidates if pair not in attempted), None)
        if len(attempted) >= config["adaptive_max_points_per_curve"]:
            candidate = None; reason = "POINT_LIMIT_REACHED"
        if candidate is not None:
            pending.append(adaptive.job(curve, *candidate)); attempted.add(candidate)
            reason = "MEASURE_INTEGER_MIDPOINT"; state = "PENDING"
        else:
            state = "UNRESOLVED"
        gaps.append(dict(base, recall_low=a, recall_high=b, gap=b-a,
                         target_gap=tolerance, status=state, reason=reason))
    reasons = sorted({g["reason"] for g in gaps if g["status"] == "UNRESOLVED"})
    coverage = dict(base, recall_min=low, recall_max=high, target_gap=tolerance,
        measured_points=len(points), attempted_points=len(attempted)-len(pending),
        min_measured_recall=min((p["recall"] for p in points), default=""),
        max_measured_recall=max((p["recall"] for p in points), default=""),
        max_recall_gap=max((b-a for a,b in zip(recalls,recalls[1:])), default=high-low),
        lower_endpoint_bracketed=any(p["recall"] <= low for p in points),
        upper_endpoint_reached=any(p["recall"] >= high for p in points),
        pending_points=len(pending), status="COVERED" if not intervals else "REFINING" if pending else "UNRESOLVED",
        reason=";".join(reasons))
    return pending, coverage, gaps


class ReportLedger:
    def __init__(self, original, cap):
        self.original, self.cap = original, cap

    def rows(self):
        return [r for r in self.original.rows() if eligible(r, self.cap)]


def install_scheduler(runner, cap):
    def initial(config, curve):
        return bounded_initial(config, curve, cap)

    def refine(config, curve, rows, complete_only=False):
        return bounded_refinement(config, curve, rows, cap, complete_only)

    def repeats(config, curve, rows, extra=False):
        return ORIGINAL_REPEATS(config, curve, [r for r in rows if eligible(r, cap)], extra)

    adaptive.initial_jobs = runner.initial_jobs = initial
    adaptive.refinement = runner.refinement = refine
    adaptive.repeat_jobs = runner.repeat_jobs = repeats
    original_init = runner.Experiments.__init__

    def initialize(self, pipeline):
        original_init(self, pipeline)
        recovered = 0
        # A graceful stop can leave a finished native file not yet ingested by
        # Python. Recover it, including over-budget output, without rerunning it.
        for row in read_csv(self.out / "adaptive-plan.csv"):
            item = {k: int(row[k]) if k in ("experiment", "side_bits", "beam", "rerank_depth", "repeat") else row[k]
                    for k in adaptive.JOB_FIELDS}
            _, prefix, fields = self.descriptor(item)
            if fields["point_id"] not in self.points and self.ingest(prefix, fields):
                recovered += 1
        print(f"[resume] {len(self.points)} existing records; recovered {recovered} completed native points; "
              f"pending rerank depth <= {cap}", flush=True)

    runner.Experiments.__init__ = initialize
    original_measure = runner.NativeTimingSession.measure

    def measure(self, prepared):
        if not eligible(prepared[2], cap):
            raise ValueError("Scheduler attempted an over-budget exact rerank")
        return original_measure(self, prepared)

    runner.NativeTimingSession.measure = measure
    original_report = runner.Experiments.report

    def report(self):
        archive = [r for r in self.ledger.rows() if not eligible(r, cap)]
        write_csv(self.out / "experiment1/completed-over-budget.csv", archive, self.ledger.fields)
        original = self.ledger
        try:
            self.ledger = ReportLedger(original, cap)
            original_report(self)
        finally:
            self.ledger = original

    runner.Experiments.report = report


def record_policy(config, cap):
    out = Path(config["output_root"])
    if not (out / "configuration.json").exists():
        from pipeline import Pipeline
        Pipeline(config)
    frozen = json.loads((out / "configuration.json").read_text())
    policy = dict(version=1, experiment=1, max_rerank_depth=cap,
        applies_to="all five primary and REMORA branches; all pending probes, refinements, repeats",
        beam_range_unchanged=True, measurement_configuration_unchanged=True,
        config_id=frozen["config_id"], source_id=frozen["source_id"],
        scheduler_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        completed_over_budget="preserved in raw CSV/files; excluded from bounded comparison")
    path = out / f"scan-policy-l{cap}.json"
    if path.exists() and json.loads(path.read_text()) != policy:
        raise ValueError("Recorded scheduling policy changed; inspect before resuming")
    if not path.exists():
        path.write_text(json.dumps(policy, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("launch", "run"))
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-rerank-depth", type=int, default=200)
    parser.add_argument("--prepared-root", default=str(ROOT / "outputs/t2i-10M/remora-native100-b64-v1"))
    parser.add_argument("--cold-store", default="/tmp/xzl-t2i-10M-cold/base.10M.cold.bin")
    args = parser.parse_args()
    if args.max_rerank_depth < 10:
        parser.error("max-rerank-depth must be >=10")
    cap = args.max_rerank_depth
    c = load_config(args.config)
    record_policy(c, cap)
    if args.command == "run":
        import run as runner
        install_scheduler(runner, cap)
        sys.argv = [str(HERE / "run.py"), "run", "--config", args.config, "--experiments", "1"]
        runner.main()
        return
    import run_native100_experiment1 as launch
    original_checked = launch.run_checked

    def checked(command, **kwargs):
        values = list(map(str, command))
        if len(values) > 3 and values[1] == "-u" and values[2] == str(HERE / "run.py") and values[3] == "run":
            command = [sys.executable, "-u", Path(__file__), "run", "--config", args.config,
                       "--max-rerank-depth", cap]
        return original_checked(command, **kwargs)

    launch.run_checked = checked
    original_pilot = launch.pilot

    def pilot(config, runs, out):
        rows = read_csv(out / "experiment1/storage-parity/summary.csv")
        expected = {(method, variant) for method in config["methods"] for variant in ("primary", "remora")}
        if ({(r["primary"], r["variant"]) for r in rows} == expected and len(rows) == len(expected)
                and all(r["ids_match"] == "True" and int(r["threads"]) == config["search_threads"] for r in rows)):
            if all(r.get('scoring_placement') == ('primary' if r['variant'] == 'primary' else 'traversal') for r in rows):
                print("[reuse] existing traversal-verified mmap/direct storage parity checks", flush=True)
                return
        original_pilot(config, runs, out)

    launch.pilot = pilot
    original_finish = launch.finish_report

    def finish(config, out):
        original_finish(config, out)
        path = out / "experiment1/summary.json"
        summary = json.loads(path.read_text())
        rows = read_csv(out / "experiment1/results.csv")
        summary.update(max_rerank_depth=cap, main_comparison="rerank depth <= cap for every primary/REMORA branch",
            completed_over_budget_measurements=sum(not eligible(r, cap) for r in rows),
            scan_policy=str(out / f"scan-policy-l{cap}.json"))
        path.write_text(json.dumps(summary, indent=2) + "\n")

    launch.finish_report = finish
    sys.argv = [str(launch.__file__), "--config", args.config,
                "--prepared-root", args.prepared_root, "--cold-store", args.cold_store]
    launch.main()


if __name__ == "__main__":
    main()
