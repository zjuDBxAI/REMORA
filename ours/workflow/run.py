#!/usr/bin/env python3
"""REMORA front door. Read-only planning by default; existing implementations stay put."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
STAGES = ("preflight", "smoke", "prepare", "exp1", "exp2", "exp3", "exp4", "exp34", "exp5")


def absolute(repo, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def read_json(path):
    return json.loads(Path(path).read_text())


def config(repo, path):
    # Match the existing pipeline's default/override semantics without importing
    # torch, Faiss, NumPy or the scientific implementation during planning.
    c = read_json(repo / "ours/test/t2i-10M/config.example.json")
    c.update(read_json(path))
    return c


def dataset(catalog, name):
    try:
        return catalog["datasets"][name]
    except KeyError:
        raise ValueError(f"Unknown dataset: {name}; use list") from None


def contract(c, spec):
    for key, expected in (("dataset", spec["dataset"]), ("base_dim", spec["dim"]),
                          ("side_bits", spec["side_bits"]), ("scoring_placement", "traversal")):
        if c.get(key) != expected:
            raise ValueError(f"{key}: expected {expected!r}, found {c.get(key)!r}")
    if c.get("budget_allocator") != "balanced-product":
        raise ValueError("The formal workflow requires Balanced Product, not legacy DP/grouping")
    if c.get("search_threads") != 8 or len(c.get("cpus") or []) != 8:
        raise ValueError("The registered performance protocol requires 8 pinned CPU threads")
    if c.get("rerank_storage") != "direct":
        raise ValueError("The registered performance protocol requires disk O_DIRECT reranking")


@dataclass
class Job:
    stage: str
    command: list[str]
    requires: list[Path] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    reuse: bool = False


def plan(repo, catalog, name, stages, python, output=None, gpu=0):
    if type(gpu) is not int or gpu < 0:
        raise ValueError('GPU must be a nonnegative process-visible ordinal')
    spec = dataset(catalog, name)
    out = absolute(repo, output or f"outputs/{name}/workflow-v1")
    prepared = absolute(repo, spec["prepared"])
    performance = absolute(repo, spec["performance"])
    diagnostic_source = absolute(repo, spec.get("diagnostic_source", spec["performance"]))
    # New diagnostic studies must never be placed inside a frozen source.
    for frozen in (prepared, performance):
        if out == frozen or frozen in out.parents or out in frozen.parents:
            raise ValueError("Keep workflow outputs separate from frozen prepared/performance roots")
    teacher = absolute(repo, spec["teacher"]) if spec["teacher"] else out / "exp34"
    jobs = []
    audit_phases = []
    for stage in stages:
        if stage not in STAGES:
            raise ValueError(f"Unknown stage: {stage}")
        if stage in ("exp3", "exp4", "exp34"):
            for phase in (("inversion", "traversal") if stage == "exp34" else
                          ("inversion",) if stage == "exp3" else ("traversal",)):
                if phase not in audit_phases:
                    audit_phases.append(phase)
    audit_added = False
    for stage in stages:
        py = [python, "-u"]
        notes = []
        if stage in ("preflight", "smoke", "prepare"):
            cfg = absolute(repo, spec["prepare_config"])
            command = py + [str(repo / "ours/test/t2i-10M/run.py"), stage,
                            "--config", str(cfg), "--experiments", "1"]
            required = [cfg]
            reuse = stage == "prepare" and prepared_complete(prepared, spec, catalog["methods"])
            if reuse:
                notes.append("Validated full-database refiner artifacts already exist; reuse, no training/encoding")
            elif stage == "prepare":
                notes.append("Reuses imported primary graphs; does not build a graph or retrain primary")
            jobs.append(Job(stage, command, required, notes, reuse))
            continue
        if stage == "exp1":
            cfg = absolute(repo, spec["performance_config"])
            command = py + [str(repo / "ours/scripts/run_native100_capped_experiment1.py"), "launch",
                            "--config", str(cfg), "--prepared-root", str(prepared),
                            "--cold-store", str(absolute(repo, spec["cold_store"])),
                            "--max-rerank-depth", "200"]
            required = [cfg, prepared / "configuration.json"]
            notes += ["CPU8; REMORA traversal; own scorer queue Top-L; NVMe/direct rerank; L<=200",
                      "Uses frozen adaptive performance configuration; never mixes traced QPS into curves"]
        elif stage in ("exp3", "exp4", "exp34"):
            if audit_added:
                continue
            audit_added = True
            command = py + [str(HERE / "adapters/run_frozen_experiments34.py"),
                            "--repo", str(repo), "--source", str(diagnostic_source),
                            "--dataset", spec["dataset"], "--output", str(out / "exp34"),
                            "--beams", "25,50,75,100,150,200,300,500,750,1000",
                            "--query-limit", "10000", "--inversion-ranks", ",".join(map(str, spec["ranks"])),
                            "--phases", ",".join(audit_phases + ["report"])]
            required = [diagnostic_source / "configuration.json", diagnostic_source / "experiment1/summary.json"]
            notes += ["Accuracy-only diagnostic passes; no fresh QPS measurements and no 3-repeat timing scan",
                      "Exp4 matched-Recall tables reuse frozen Exp1 observations at fixed rerank depth"]
            stage = "exp34"
        elif stage == "exp2":
            command = py + [str(repo / "ours/scripts/run_frozen_ablation.py"), "--repo", str(repo),
                            "--dataset", spec["dataset"], "--side-bits", str(spec["side_bits"]),
                            "--prepared", str(prepared), "--performance", str(performance),
                            "--teacher", str(teacher), "--output", str(out / "exp2"),
                            "--beams", "100,200", "--depths", "0,100", "--repeats", "3",
                            "--queries", "10000", "--gpu", str(gpu)]
            required = [prepared / "configuration.json", performance / "configuration.json",
                        teacher / "experiment3/raw/routing-pairs.csv", teacher / "experiment3/raw/common.ids.ibin"]
            if max(spec["ranks"]) < 200:
                notes.append("BLOCKED: old ablation audit requires full-database Top-200; this package only has Top-100")
            if spec.get("ablation_blocked"):
                notes.append("BLOCKED: " + spec["ablation_blocked"])
        else:  # exp5
            command = py + [str(HERE / "adapters/run_accuracy_rate.py"), "--repo", str(repo),
                            "--dataset", spec["dataset"], "--source", str(diagnostic_source),
                            "--prepared", str(prepared), "--audit-source", str(teacher),
                            "--output", str(out / "exp5"), "--budgets", ",".join(map(str, spec["budgets"])),
                            "--beams", "100,200", "--queries", "10000", "--gpu", str(gpu)]
            required = [prepared / "configuration.json", performance / "experiment1/summary.json",
                        teacher / "study.json", teacher / "experiment3/raw/routing-pairs.csv"]
            if max(spec["budgets"]) > 256:
                command += ["--trainer", "ours/experimental/relative-rate-v1/train_remora.py",
                            "--build-dir", "ours/build-relative-rate-v1"]
            if max(spec["ranks"]) < 200:
                notes.append("BLOCKED: rate quality audit requires full-database Top-200; this package only has Top-100")
            notes += ["Relative budgets 0.5/1/1.5/2x default; one Recall pass plus quality; QPS timings discarded"]
        required.insert(0, Path(command[2]))
        jobs.append(Job(stage, command, required, notes))
    return jobs


def prepared_complete(root, spec, methods):
    # A frozen training config can say terminal while the exact same refiner
    # was subsequently deployed during traversal. Never rewrite that history.
    if not (root / "configuration.json").is_file():
        return False
    import struct
    try:
        c = read_json(root / "configuration.json")["config"]
        if (c["dataset"], c["base_dim"], c["side_bits"]) != (spec["dataset"], spec["dim"], spec["side_bits"]):
            return False
        for method in methods:
            dest = root / "ji" / method / f'b{spec["side_bits"]}' / "remora"
            side, online = dest / "side.bin", dest / "online.bin"
            with side.open("rb") as handle:
                h = struct.unpack("<8sIQIIIQQ", handle.read(48))
            with online.open("rb") as handle:
                o = struct.unpack("<8sIIIIQQQQ", handle.read(56))
            if (h[:3] != (b"JIRCODE1", 3, c["base_rows"]) or h[4:6] != (spec["side_bits"], spec["side_bits"] // 8)
                    or o[:2] != (b"JIONL002", 2) or o[3:6] != (h[5], h[4], h[2])
                    or h[6:8] != o[6:8] or side.stat().st_size != 48 + c["base_rows"] * h[5]
                    or not (dest / "training.json").is_file()):
                return False
        return True
    except (OSError, ValueError, KeyError, struct.error):
        return False


def environment(repo, python):
    env = dict(os.environ, PYTHON_BIN=python, PYTHONUNBUFFERED="1", PYTHONDONTWRITEBYTECODE="1", NVIDIA_TF32_OVERRIDE="0")
    paths = [repo / ".runtime/faiss-v1.9.0-build/faiss/python/build/lib", repo / "ours/test/t2i-10M",
             repo / "ours/python", repo / "ours/test/t2i-1M/python"]
    env["PYTHONPATH"] = os.pathsep.join(map(str, paths)) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    env.setdefault("REMORA_NATIVE_PRIMARY_READER", str(repo / "ours/runtime/native100-reader-v1/roar_native_primary_reader"))
    env["REMORA_PRIMARY_DECODE_THREADS"] = "8"
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[name] = "8"
    prefix = Path(python).resolve().parent.parent
    compiler = prefix / "bin/x86_64-conda-linux-gnu-c++"
    if compiler.is_file():
        env.setdefault("CXX", str(compiler))
    env["CMAKE_PREFIX_PATH"] = str(prefix) + (os.pathsep + env["CMAKE_PREFIX_PATH"] if env.get("CMAKE_PREFIX_PATH") else "")
    return env


def display(jobs):
    for index, job in enumerate(jobs, 1):
        print(f"[{index}/{len(jobs)}] {job.stage}" + (" [reuse]" if job.reuse else ""))
        print("  " + shlex.join(job.command))
        for note in job.notes:
            print("  " + note)
        for path in job.requires:
            if not path.exists():
                print(f"  MISSING: {path}")


def execute(repo, jobs, logs, env):
    logs.mkdir(parents=True, exist_ok=True)
    with (logs / "run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        start = time.monotonic()
        for index, job in enumerate(jobs, 1):
            if job.reuse:
                print(f"[reuse] {job.stage}: frozen full-database artifacts", flush=True)
                continue
            missing = [str(p) for p in job.requires if not p.exists()]
            blocked = [note for note in job.notes if note.startswith("BLOCKED:")]
            if missing or blocked:
                raise ValueError("\n".join(blocked + [f"Missing prerequisite: {p}" for p in missing]))
            stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
            with (logs / f"{stamp}-{job.stage}.log").open("x") as log:
                header = json.dumps({"stage": job.stage, "command": job.command, "cwd": str(repo),
                                     "started_utc": stamp, "elapsed_total_estimate": "unknown"})
                log.write(header + "\n"); log.flush()
                print(f"[{index}/{len(jobs)}] {job.stage}; elapsed/total={time.monotonic()-start:.1f}s/unknown", flush=True)
                with subprocess.Popen(job.command, cwd=repo, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                    try:
                        for line in proc.stdout:
                            print(line, end="", flush=True); log.write(line); log.flush()
                        rc = proc.wait()
                    except BaseException:
                        proc.terminate(); proc.wait()
                        raise
                log.write(f"\nexit_code={rc}; elapsed_seconds={time.monotonic()-start:.3f}\n")
                if rc:
                    raise RuntimeError(f"{job.stage} failed with exit code {rc}; see {log.name}")


def report_paths(repo, spec, name, out):
    candidates = [absolute(repo, spec["performance"]) / "experiment1/summary.json",
                  absolute(repo, spec["performance"]) / "experiment1/aggregate.csv",
                  absolute(repo, spec["performance"]) / "experiment1/figures",
                  out / "exp2/report/summary.json", out / "exp34/summary.json",
                  out / "exp5/experiment5/report/summary.json"]
    if spec.get("teacher"):
        candidates.append(absolute(repo, spec["teacher"]) / "summary.json")
    if spec.get("published_rate"):
        candidates.append(absolute(repo, spec["published_rate"]))
    for path in candidates:
        print(("EXISTS " if path.exists() else "MISSING ") + str(path))


def inspect(repo, spec, catalog):
    c = config(repo, absolute(repo, spec["performance_config"]))
    contract(c, spec)
    result = {"dataset": spec["dataset"], "dim": spec["dim"], "primary_target_bits": spec["primary_target_bits"],
              "side_bits": spec["side_bits"], "scoring_placement": c["scoring_placement"],
              "threads": c["search_threads"], "cpus": c["cpus"], "rerank_storage": c["rerank_storage"],
              "prepared_complete": prepared_complete(absolute(repo, spec["prepared"]), spec, catalog["methods"]),
              "primary_budget_note": "target only; actual native/padded bits are recorded per primary"}
    paths = {k: absolute(repo, c[k]) for k in ("base", "train_queries", "test_queries", "graph_package")}
    result["inputs"] = {k: {"path": str(p), "exists": p.exists()} for k, p in paths.items()}
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return all(p.exists() for p in paths.values()) and result["prepared_complete"]


def inventory(repo, stream):
    """Conservative source index, NOT a claim that unlisted files are safe to delete."""
    groups = {
        "graph-library": ("include", "src"),
        "primary-and-kernels": ("ours/include", "ours/src"),
        "refiner-and-io": ("ours/python", "ours/method"),
        "formal-pipeline": ("ours/test/t2i-10M", "ours/test/webvid", "ours/test/rs5m",
                            "ours/test/wordvector-id", "ours/test/t2i-1M-native100"),
        "shared-historical-dependency": ("ours/test/t2i-1M/python", "ours/test/t2i-1B"),
        "native-kernel-overlay": ("ours/experimental/traversal-fast-v1",),
        "relative-rate-overlay": ("ours/experimental/relative-rate-v1",),
        "orchestration": ("ours/scripts", "ours/workflow"),
    }
    allowed = {".py", ".sh", ".cpp", ".cc", ".h", ".hpp", ".json", ".md", ".cmake", ".txt"}
    writer = csv.writer(stream)
    writer.writerow(("role", "path", "bytes", "sha256"))
    seen = set()
    for role, dirs in groups.items():
        for folder in dirs:
            for p in sorted((repo / folder).rglob("*")):
                if (not p.is_file() or p.suffix not in allowed or "__pycache__" in p.parts
                        or p in seen or any(v.startswith("build-") for v in p.relative_to(repo).parts)):
                    continue
                seen.add(p)
                writer.writerow((role, p.relative_to(repo), p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))
    for name in ("CMakeLists.txt", "ours/CMakeLists.txt"):
        p = repo / name
        if p.is_file():
            writer.writerow(("build-definition", name, p.stat().st_size, hashlib.sha256(p.read_bytes()).hexdigest()))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("list", "check", "plan", "run", "reports", "inventory"))
    parser.add_argument("--repo", type=Path, default=REPO)
    parser.add_argument("--catalog", type=Path, default=HERE / "catalog.json")
    parser.add_argument("--dataset", default="t2i-10M")
    parser.add_argument("--stages", default="exp1")
    parser.add_argument("--output", help="New diagnostic/ablation/rate output root; never a frozen source")
    parser.add_argument("--python", default=os.environ.get("PYTHON_BIN", sys.executable))
    parser.add_argument("--gpu", type=int, default=0, help="GPU ordinal for ablation/rate training")
    parser.add_argument("--execute", action="store_true", help="Explicitly launch the printed jobs; otherwise read-only")
    args = parser.parse_args(argv)
    repo = args.repo.resolve(strict=True)
    catalog = read_json(args.catalog)
    if args.command == "list":
        for name, spec in catalog["datasets"].items():
            print(f"{name:14s} D={spec['dim']:3d} primary-target={spec['primary_target_bits']:4d}bit refiner={spec['side_bits']:3d}bit; traversal")
        print("Stages: " + ", ".join(STAGES))
        return 0
    if args.command == "inventory":
        inventory(repo, sys.stdout); return 0
    spec = dataset(catalog, args.dataset)
    out = absolute(repo, args.output or f"outputs/{args.dataset}/workflow-v1")
    if args.command == "reports":
        report_paths(repo, spec, args.dataset, out); return 0
    if args.command == "check":
        return 0 if inspect(repo, spec, catalog) else 2
    python = shutil.which(args.python)
    if not python:
        raise ValueError(f"Python executable not found: {args.python}")
    jobs = plan(repo, catalog, args.dataset, args.stages.split(","), python, args.output, args.gpu)
    display(jobs)
    if args.command == "run" and args.execute:
        # Inspect each current deployment config immediately before execution.
        contract(config(repo, absolute(repo, spec["performance_config"])), spec)
        execute(repo, jobs, out / "logs", environment(repo, python))
    elif args.execute:
        raise ValueError("--execute is only valid with run")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f"workflow: {exc}", file=sys.stderr)
        sys.exit(2)
