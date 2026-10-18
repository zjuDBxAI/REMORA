#!/usr/bin/env python3
"""Existing rate trainer/audit plus one accuracy pass; no QPS report or timing repeats."""
from __future__ import annotations

import argparse
import fcntl
import json
import os
from pathlib import Path
import struct
import subprocess
import sys


def training_command(args):
    command = [sys.executable, "-u", str(Path(args.repo) / "ours/scripts/run_frozen_experiment5.py"),
               "--repo", str(args.repo), "--dataset", args.dataset, "--source", args.source,
               "--prepared", args.prepared, "--audit-source", args.audit_source,
               "--output", args.output, "--budgets", args.budgets, "--beams", args.beams,
               "--queries", str(args.queries), "--repeats", "1", "--phases", "train,quality",
               "--gpu", str(args.gpu)]
    for name in ("trainer", "build_dir"):
        if getattr(args, name):
            command += ["--" + name.replace("_", "-"), getattr(args, name)]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("repo", "dataset", "source", "prepared", "audit-source", "output", "budgets"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--beams", default="100,200")
    parser.add_argument("--queries", type=int, default=10000)
    parser.add_argument("--gpu", type=int, default=2)
    parser.add_argument("--trainer", default="")
    parser.add_argument("--build-dir", default="")
    args = parser.parse_args()
    repo = Path(args.repo).resolve(strict=True)
    out = (repo / args.output).resolve()
    source = (repo / args.source).resolve(strict=True)
    prepared = (repo / args.prepared).resolve(strict=True)
    audit = (repo / args.audit_source).resolve(strict=True)
    for frozen in (source, prepared, audit):
        if out == frozen or frozen in out.parents or out in frozen.parents:
            raise ValueError("Rate output must not overlap a frozen input")
    # Fail before training when truth is incomplete for the unchanged audit.
    with (audit / "experiment3/raw/full-precision-fp32-b0.scores.fbin").open("rb") as stream:
        rows, depth = struct.unpack("<II", stream.read(8))
    if rows != args.queries or depth < 200:
        raise ValueError("Rate quality requires matching full-database exact Top-200 teacher scores")
    out.mkdir(parents=True, exist_ok=True)
    with (out / "accuracy-runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        subprocess.run(training_command(args), cwd=repo, check=True)
        sys.path[:0] = [str(repo / "ours/test/t2i-10M"), str(repo / "ours/scripts"),
                       str(repo / "ours/python"), str(repo / "ours/test/t2i-1M/python")]
        from pipeline import Pipeline
        from protocol import file_id, identity, matrix, read_csv, write_csv
        from run_frozen_experiment5 import rate_plot, save_json
        # These two pure report/validation helpers are already used by the
        # completed RS5M accuracy-only study; their logic is dataset independent.
        sys.path.insert(0, str(repo / "ours/test/rs5m"))
        from run_experiments345 import recall_record, quality_table

        c = json.loads((out / "configuration.json").read_text())["config"]
        if c["dataset"] != args.dataset or c["scoring_placement"] != "traversal":
            raise ValueError("Dataset or traversal contract mismatch")
        p = Pipeline(c)
        budgets = sorted(set(map(int, args.budgets.split(","))))
        beams = sorted(set(map(int, args.beams.split(","))))
        truth = matrix(out / "data/truth.ids.ibin", "<u4")
        methods = c["methods"]
        source_rows = read_csv(source / "experiment1/aggregate.csv")
        quality = read_csv(out / "experiment5/quality/metrics.csv")
        os.sched_setaffinity(0, c["cpus"])
        records = []
        total = len(methods) * (len(budgets) + 1) * len(beams)
        for method in methods:
            for bits in [0, *budgets]:
                variant = "remora" if bits else "primary"
                branch = p.branch(method, variant, bits)
                for beam in beams:
                    dest = out / "experiment5/recall" / method / f"b{bits}" / f"beam{beam}"
                    deps = [branch[k] for k in ("graph", "model", "codes")] + [out / "data/test.fbin"]
                    if bits:
                        deps += [branch["ji_model"], branch["ji_codes"]]
                    stamp = identity(dict(study=file_id(out / "study.json"), method=method, bits=bits,
                                          beam=beam, files={str(v): file_id(v) for v in deps},
                                          runner=file_id(Path(__file__))))
                    cache = dest / "recall.json"
                    if cache.is_file() and json.loads(cache.read_text()).get("fingerprint") == stamp:
                        row = json.loads(cache.read_text())["record"]
                    else:
                        old = next((r for r in source_rows if r["primary"] == method and r["variant"] == variant
                                    and int(r["side_bits"]) == bits and int(r["beam"]) == beam
                                    and int(r["rerank_depth"]) == 0 and r["status"] == "OK"), None)
                        if bits in (0, c["side_bits"]) and old:
                            prefix = Path(old["directory"]) / (old.get("measurement_point_id") or old["point_id"])
                            reused = True
                        else:
                            prefix = dest / "search"
                            reused = False
                            command = [Path(c["build_dir"]) / "roar_primary_ji_search",
                                       *p.native_args(branch, variant), "--graph", branch["graph"],
                                       "--queries", out / "data/test.fbin", "--query-limit", args.queries,
                                       "--topk", 10, "--beam", beam, "--threads", c["search_threads"],
                                       "--warmup-queries", 0, "--packed-graph", "--out-prefix", prefix,
                                       "--memory-budget-bytes", c["memory_budget_bytes"]]
                            p.stage(f"accuracy-{method}-b{bits}-beam{beam}", command,
                                    [Path(str(prefix) + ".ids.ibin"), Path(str(prefix) + ".summary.json")], deps,
                                    budget=c["memory_budget_bytes"])
                        native = json.loads(Path(str(prefix) + ".summary.json").read_text())
                        row = recall_record(native, truth, matrix(str(prefix) + ".ids.ibin", "<u4"), method, bits, beam)
                        row.update(source_reused=reused, native_result_prefix=str(prefix))
                        save_json(cache, dict(fingerprint=stamp, record=row))
                    records.append(row)
                    write_csv(out / "experiment5/recall.csv", records)
                    print(f"[accuracy] {len(records)}/{total} {method}/b{bits}/beam{beam}: "
                          f"Recall={100*float(row['recall_at_10']):.3f}%; one pass; QPS discarded", flush=True)
        tables = quality_table(methods, budgets, beams, quality, records)
        if len(records) != total or any(r["status"] != "COMPLETE" for r in tables):
            raise ValueError("Incomplete quality or Recall results")
        report = out / "experiment5/report"
        write_csv(report / "requested-tables.csv", tables)
        write_csv(report / "quality.csv", quality)
        for method in methods:
            for beam in beams:
                write_csv(report / f"{method}-beam{beam}-table.csv",
                          [r for r in tables if r["primary"] == method and r["beam"] == beam])
        for beam in beams:
            for key in ("DW", "score_rmse", "routing_inv_percent", "boundary_inv_k10_h5_percent", "recall_at_10_percent"):
                rate_plot(report / f"{key}-bits-beam{beam}.svg", [r for r in tables if r["beam"] == beam],
                          key, f"{args.dataset} / REMORA traversal / beam={beam} / no FP rerank", key)
        save_json(report / "summary.json", dict(status="COMPLETE", experiment=5, dataset=args.dataset,
                  side_bits=budgets, methods=methods, beams=beams, recall_points=len(records),
                  qps_measured=False, native_timing_usage="discarded; accuracy only", passes_per_accuracy_point=1,
                  scoring_placement="traversal", allocator="balanced-product; NOT historical DP",
                  requested_table=str(report / "requested-tables.csv")))
        print(f"[complete] {report / 'summary.json'}", flush=True)


if __name__ == "__main__":
    main()
