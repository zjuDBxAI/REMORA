"""Run experiment 1 without retraining or altering the frozen prepared suite.

Separate results, shared immutable graph/codes, NVMe O_DIRECT exact reranking.
Pilot points are correctness checks only and never enter the performance CSV.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
HERE = ROOT / "ours/test/t2i-10M"
sys.path[:0] = [str(HERE), str(ROOT / "ours/python")]

from protocol import load_config, matrix, read_csv
from qvrr import select_runs
from scoring_contract import require_traversal, validate_command, validate_native_summary


def link_once(source, destination):
    source, destination = Path(source).resolve(strict=True), Path(destination)
    if destination.exists() or destination.is_symlink():
        if destination.resolve(strict=True) != source:
            raise ValueError(f"Refusing to replace existing artifact: {destination}")
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.symlink_to(source, target_is_directory=source.is_dir())


def check_refiner(prepared, method, rows, bits=64):
    groups = bits // 8
    dest = prepared / "ji" / method / f"b{bits}/remora"
    summary = json.loads((dest / "training.json").read_text())
    audit = json.loads((dest / "encoding-parity.json").read_text())
    with (dest / "side.bin").open("rb") as f:
        side = struct.unpack("<8sIQIIIQQ", f.read(48))
    with (dest / "online.bin").open("rb") as f:
        online = struct.unpack("<8sIIIIQQQQ", f.read(56))
    if (side[:3] != (b"JIRCODE1", 3, rows) or side[4:6] != (bits, groups)
            or online[:2] != (b"JIONL002", 2) or online[3:6] != (groups, bits, rows)
            or side[6:8] != online[6:8]
            or (dest / "side.bin").stat().st_size != 48 + rows * groups
            or summary["test_queries_used"]
            or audit["faiss_canonical_mismatches"] != 0):
        raise ValueError(f"Prepared refiner validation failed: {method}")


def check_cold_store(path, base_path):
    with path.open("rb") as f:
        header = struct.unpack("<8sIIQIIIIIQQ", f.read(struct.calcsize("<8sIIQIIIIIQQ")))
        magic, version, header_bytes, rows, dim, elem, vector_bytes, per_block, block, offset, blocks = header
        base = matrix(base_path)
        if (magic != b"CDVEC001" or version != 1 or (rows, dim) != base.shape
                or elem != 4 or vector_bytes != dim * 4 or block != 4096
                or per_block != block // vector_bytes or header_bytes != offset
                or blocks != (rows + per_block - 1) // per_block
                or path.stat().st_size != offset + blocks * block):
            raise ValueError("Cold-store header/size mismatch")
        for node in np.unique(np.linspace(0, rows - 1, 1024).astype(np.int64)):
            f.seek(offset + int(node // per_block) * block + int(node % per_block) * vector_bytes)
            value = np.frombuffer(f.read(vector_bytes), dtype="<f4")
            if not np.array_equal(value, base[node]):
                raise ValueError(f"Cold-store/base row-order mismatch at node {node}")
    print(f"[cold-store OK] {rows} nodes; 1024 sampled rows match raw FP32; {path.stat().st_size:,} bytes", flush=True)


def run_checked(command, **kwargs):
    subprocess.run(list(map(str, command)), check=True, **kwargs)


def pilot(c, runs, out):
    """Same graph/model/queries/codes across mmap/direct, compare returned IDs."""
    audit_rows = []
    dest = out / "experiment1/storage-parity"
    dest.mkdir(parents=True, exist_ok=True)
    def setup():
        os.sched_setaffinity(0, c["cpus"])
    for method in c["methods"]:
        run = runs[method]
        for variant in ("primary", "remora"):
            ids = {}
            for storage in ("mmap", "direct"):
                prefix = dest / f"{method}-{variant}-{storage}"
                command = [Path(c["build_dir"]) / "roar_primary_ji_search",
                    "--primary-kind", method, "--primary-model", run["paths"]["model"],
                    "--primary-codes", run["paths"]["codes"], "--graph", run["paths"]["graph"],
                    "--queries", out / "data/test.fbin", "--query-limit", 64, "--topk", 10,
                    "--beam", 50, "--rerank-candidates", 20, "--threads", c["search_threads"],
                    "--warmup-queries", 0, "--packed-graph", "--out-prefix", prefix,
                    "--base", c["base"] if storage == "mmap" else out / "data/base.cold",
                    "--rerank-storage", storage,
                    "--memory-budget-bytes", c["memory_budget_bytes"]]
                if method == "extended-rabitq":
                    command += ["--rabitq-config", c["rabitq_config"]]
                if variant == "primary":
                    command += ["--disable-ji"]
                else:
                    model = out / "ji" / method / f"b{c['side_bits']}/remora"
                    command += ["--ji-query-space", "raw",
                                "--ji-model", model / "online.bin", "--ji-side-codes", model / "side.bin"]
                validate_command(command, variant)
                with Path(str(prefix) + ".log").open("w") as f:
                    run_checked(command, stdout=f, stderr=subprocess.STDOUT, preexec_fn=setup)
                ids[storage] = np.asarray(matrix(str(prefix) + ".ids.ibin", "<u4"))
                metrics = read_csv(str(prefix) + ".metrics.csv")
                summary = json.loads(Path(str(prefix) + ".summary.json").read_text())
                validate_native_summary(summary, variant)
                if len(metrics) != 1 or metrics[0]["rerank_storage"] != storage:
                    raise ValueError(f"Native storage backend mismatch: {method}/{variant}/{storage}")
            matched = bool(np.array_equal(ids["mmap"], ids["direct"]))
            if not matched:
                raise ValueError(f"mmap/direct result mismatch: {method}/{variant}")
            audit_rows.append(dict(primary=method, variant=variant, queries=64, beam=50,
                                   rerank_depth=20, threads=c["search_threads"], ids_match=matched,
                                   scoring_placement="primary" if variant == "primary" else "traversal",
                                   ji_traversal_verified=variant == "remora",
                                   timing_usage="correctness only; excluded from performance results"))
            print(f"[storage-parity OK] {method}/{variant}: mmap/direct Top-10 IDs identical; "
                  f"traversal={'Primary + REMORA' if variant == 'remora' else 'Primary'} verified", flush=True)
    with (dest / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(audit_rows[0]))
        writer.writeheader(); writer.writerows(audit_rows)


def finish_report(c, out):
    from plots import PALETTE, chart, frontier
    rows = read_csv(out / "experiment1/aggregate.csv")
    groups = {}
    for row in rows:
        if row["status"] == "OK" and str(row.get("timing_complete", "")).lower() == "true":
            key = row["primary"], row["variant"], bool(int(row["rerank_depth"]))
            groups.setdefault(key, []).append(row)
    methods = sorted(c["methods"])
    colors = {method: PALETTE[index % len(PALETTE)] for index, method in enumerate(methods)}
    # Keep the main >=80% panels, and provide complete no-rerank curves even
    # when a compressed scorer cannot attain 80%. Never extrapolate Recall.
    dest = out / "experiment1/figures-full-range"
    dest.mkdir(parents=True, exist_ok=True)
    for rerank in (False, True):
        mode = "rerank" if rerank else "no-rerank"
        for metric, label in (("qps", "QPS"), ("mean_service_latency_ms", "Mean service latency (ms)"),
                              ("p50_ms", "P50 service latency (ms)"), ("p95_ms", "P95 service latency (ms)"),
                              ("p99_ms", "P99 service latency (ms)")):
            selected = [(method + (" / Primary" if variant == "primary" else " / REMORA"),
                         colors[method], variant == "primary", frontier(local, metric))
                        for (method, variant, enabled), local in sorted(groups.items()) if enabled == rerank]
            chart(dest / f"{metric}-recall-{mode}.svg", selected,
                  f"{c['dataset']} — {mode} / {c['search_threads']} threads / shared host", label, 0.0)
    latest = {r["point_id"]: r for r in read_csv(out / "experiment1/results.csv")}
    failed = [r for r in latest.values() if r["status"] != "OK"]
    summary = dict(status="COMPLETE" if not failed else "INCOMPLETE", experiment=1,
        methods=c["methods"], queries=c["test_query_limit"], threads=c["search_threads"], cpus=c["cpus"],
        side_bits=c['side_bits'], deployment="REMORA traversal and queue scores; own queue top-L exact reranking",
        scoring_placement="traversal", terminal_only_results_excluded=True,
        fp32_rerank_storage="NVMe O_DIRECT; OS page cache bypassed; device caches not disabled",
        timing_scope="shared host; affinity is not exclusive reservation",
        repeats=c["repeats"], successful_measurements=len(latest)-len(failed), failed_measurements=len(failed),
        adaptive_coverage=read_csv(out / "experiment1/adaptive-coverage.csv"),
        training_reused=True, graphs_reused=True, fp32_curve_included=c["include_fp32_reference"],
        figures=str(out / "experiment1/figures"), full_range_figures=str(dest))
    (out / "experiment1/summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    if failed:
        raise RuntimeError(f"{len(failed)} failed measurements; inspect experiment1/results.csv")
    print(f"[complete] experiment 1: {out / 'experiment1/summary.json'}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--prepared-root", required=True)
    parser.add_argument("--cold-store", required=True)
    args = parser.parse_args()
    c = load_config(args.config)
    require_traversal(c)
    if c["rerank_storage"] != "direct" or c["fp32_storage"] != "direct":
        raise ValueError("This launch requires direct FP32 IO and REMORA traversal")
    prepared = Path(args.prepared_root).resolve(strict=True)
    out = Path(c["output_root"])
    if prepared == out.resolve():
        raise ValueError("Performance results must not alter the frozen training root")
    runs = select_runs(c["graph_package"], c["methods"])
    for method in c["methods"]:
        check_refiner(prepared, method, c["base_rows"], c["side_bits"])
    out.mkdir(parents=True, exist_ok=True)
    link_once(prepared / "ji", out / "ji")
    for name in ("test.fbin", "truth.ids.ibin", "truth.scores.fbin", "truth.metadata.json"):
        link_once(prepared / "data" / name, out / "data" / name)
    run_checked([sys.executable, HERE / "run.py", "preflight", "--config", args.config, "--experiments", "1"])
    cold = Path(args.cold_store).resolve()
    if not cold.exists():
        cold.parent.mkdir(parents=True, exist_ok=True)
        required = 4096 + ((c['base_rows'] + 4096 // (4*c['base_dim']) - 1)
                          // (4096 // (4*c['base_dim']))) * 4096
        if os.statvfs(cold.parent).f_bavail * os.statvfs(cold.parent).f_frsize < required + 1_000_000_000:
            raise RuntimeError(f"Insufficient free space for the {required:,}-byte FP32 cold store")
        run_checked([sys.executable, "-u", ROOT / "ours/python/build_cold_vector_store.py",
                     "--input-fbin", c["base"], "--output", cold])
    check_cold_store(cold, c["base"])
    link_once(cold, out / "data/base.cold")
    pilot(c, runs, out)
    print(f"[performance] 5 primaries x (Primary, REMORA) x (no-rerank, finite rerank); "
          f"REMORA traversal enabled, terminal-only DISABLED; "
          f"{c['search_threads']} pinned CPU threads, {c['test_query_limit']} test queries, "
          f"{c['repeats']} repeats; NVMe O_DIRECT; training is NOT rerun", flush=True)
    started = time.monotonic()
    run_checked([sys.executable, "-u", HERE / "run.py", "run", "--config", args.config, "--experiments", "1"])
    finish_report(c, out)
    print(f"[elapsed] experiment-1 timing/report phase {time.monotonic()-started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
