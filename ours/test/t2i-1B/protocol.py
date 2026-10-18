"""PDF experiment definitions, strict input validation and durable CSV records.

This module imports neither torch nor faiss, so planning/preflight and report
tests remain usable before installing the training dependencies.
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import struct

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
METHODS = ("opq-pq", "qaq", "leanvec-ood", "extended-rabitq", "saq")
VARIANTS = ("equal-rate", "atomic-equal-rate", "rate-adaptive", "joint-grouping")
BEAMS = list(range(10, 201, 5)) + list(range(225, 1001, 25)) + list(range(1100, 2001, 100))
TARGETS = (.90, .95, .99)
INVERSION_RANKS = (50, 75, 100)
RESULT_FIELDS = [
    "point_id", "config_id", "experiment", "dataset", "base_rows", "base_dim",
    "primary", "implementation", "variant", "side_bits", "beam", "candidate_size",
    "rerank_depth", "rerank_policy", "threads", "cpus", "repeat", "seed", "query_split",
    "queries", "status", "error", "recall_at_10", "qps", "mean_service_latency_ms",
    "p50_ms", "p95_ms", "p99_ms", "visited", "expanded", "comparisons", "reranked",
    "rerank_io_ms", "rerank_physical_bytes", "graph_memory_bytes", "primary_code_bytes",
    "refiner_code_bytes", "codebooks_transform_bytes", "runtime_buffer_bytes", "fp_hot_bytes",
    "accounted_memory_bytes", "peak_rss_bytes", "memory_budget_bytes", "budget_pass",
    "memory_accounting", "representation_id", "graph_id", "ids_id", "directory",
    "primary_scoring_kernel", "exact_rerank_scoring_kernel", "query_preparation_included",
    "fp32_storage", "rerank_storage", "fp32_read_policy", "fp32_store_id",
    "primary_io_ms", "primary_logical_bytes", "primary_physical_bytes", "primary_read_ops",
    "measurement_id", "measurement_point_id", "measurement_reused",
]
INVERSION_FIELDS = ["experiment", "dataset", "primary", "variant", "side_bits", "metric",
    "status", "queries", "queries_with_pairs", "queries_without_pairs", "pairs",
    "inversions", "exact_ties_excluded", "inversion_rate", "pair_weighted_inversion_rate",
    "approximate_tie_rate", "teacher_id", "representation_id", "pair_selection", "note"]


def absolute(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def load_config(path=None):
    config = json.loads((HERE / "config.example.json").read_text())
    config.update({"beams": BEAMS, "bit_budget_beams": BEAMS,
                   "cmake_prefix_path": "", "ground_truth_ids": "",
                   "ground_truth_scores": "", "ground_truth_metadata": ""})
    if path:
        extra = json.loads(Path(path).read_text())
        unknown = set(extra) - set(config)
        if unknown:
            raise ValueError(f"Unknown configuration keys: {sorted(unknown)}")
        config.update(extra)
    for key in ("base", "train_queries", "test_queries", "output_root", "build_dir"):
        config[key] = str(absolute(config[key]))
    for key in ("ground_truth_ids", "ground_truth_scores", "ground_truth_metadata"):
        if config[key]: config[key] = str(absolute(config[key]))
    if not config["methods"] or len(set(config["methods"])) != len(config["methods"]):
        raise ValueError("methods must be nonempty and unique")
    if set(config["methods"]) - set(METHODS):
        raise ValueError("Unconnected baseline requested; see baseline-availability.csv")
    if config["deploy_variant"] not in VARIANTS or set(config["ablation_variants"]) - set(VARIANTS):
        raise ValueError("Invalid JI variant")
    if config["side_bits"] != 64 or config["budgets"] != [16, 32, 48, 64, 96, 128]:
        raise ValueError("PDF suite requires main B=64 and budgets 16/32/48/64/96/128")
    for key in ("base_rows", "base_dim", "test_query_limit", "primary_train_size", "search_threads",
                "build_threads", "compile_jobs", "repeats", "ji_batch_size", "encode_batch_size",
                "primary_iterations", "ji_iterations", "ji_fit_size", "residual_sample_size", "exact_query_batch_size"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["exact_backend"] not in ("numpy", "faiss-cpu"):
        raise ValueError("exact_backend must be numpy or faiss-cpu")
    if config["exact_backend"] == "faiss-cpu" and config["device"] != "cpu":
        raise ValueError("faiss-cpu exact_backend requires device=cpu")
    interval = config["exact_checkpoint_interval_s"]
    if isinstance(interval, bool) or not isinstance(interval, (int, float)) or not math.isfinite(interval) or interval < 0:
        raise ValueError("exact_checkpoint_interval_s must be finite and nonnegative")
    if config["base_rows"] > np.iinfo(np.uint32).max:
        raise ValueError("Node format is uint32; this suite cannot index more than UINT32_MAX nodes")
    if config["memory_budget_bytes"] <= 0 or config["rerank_storage"] not in ("direct", "mmap"):
        raise ValueError("Invalid memory budget or vector backend")
    if config["fp32_storage"] not in ("direct", "mmap") or config["fp32_storage"] != config["rerank_storage"]:
        raise ValueError("FP32 search and rerank must use the same FP32 storage policy")
    ranks = config["inversion_ranks"]
    if not ranks or any(not isinstance(r, int) or isinstance(r, bool) or r < 2 for r in ranks):
        raise ValueError("inversion_ranks must contain integer ranks >= 2")
    config["inversion_ranks"] = sorted(set(ranks))
    if config["base_rows"] < truth_depth(config):
        raise ValueError("Database has fewer nodes than required inversion ground-truth depth")
    for key in ("beams", "bit_budget_beams", "rerank_depths"):
        values = config[key]
        if not values or any(not isinstance(v, int) or v < 10 for v in values):
            raise ValueError(f"Invalid {key}")
        config[key] = sorted(set(values))
    if config["performance_sampling"] not in ("dense", "adaptive"):
        raise ValueError("performance_sampling must be dense or adaptive")
    bounds = config["recall_range"]
    if (not isinstance(bounds, list) or len(bounds) != 2 or
        any(isinstance(x, bool) or not isinstance(x, (float, int)) or not math.isfinite(x) for x in bounds) or
        not 0 <= bounds[0] < bounds[1] <= 1):
        raise ValueError("recall_range must contain two increasing values in [0,1]")
    for key in ("recall_gap", "adaptive_qps_cv"):
        value = config[key]
        if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or not 0 < value < 1:
            raise ValueError(f"{key} must be a fraction between 0 and 1 (0.01 = one percentage point)")
    for key in ("adaptive_coarse_beams", "adaptive_coarse_depths"):
        values = config[key]
        if not values or any(isinstance(v, bool) or not isinstance(v, int) or not 10 <= v <= np.iinfo(np.uint32).max for v in values):
            raise ValueError(f"Invalid {key}")
        config[key] = sorted(set(values))
    for key in ("adaptive_max_points_per_curve", "adaptive_max_rounds", "adaptive_extra_repeats"):
        if isinstance(config[key], bool) or not isinstance(config[key], int) or config[key] <= 0:
            raise ValueError(f"{key} must be a positive integer")
    if config["adaptive_extra_repeats"] < config["repeats"]:
        raise ValueError("adaptive_extra_repeats must be >= repeats")
    if config["ablation_beam"] < 10 or not 0 <= config["ablation_rerank_depth"] <= config["ablation_beam"]:
        raise ValueError("Invalid fixed ablation working point")
    if 0 < config["ablation_rerank_depth"] < 10:
        raise ValueError("Ablation rerank depth must be zero or >=10")
    ranges = []
    for name in ("metric", "select", "build"):
        start, count = config[name + "_query_start"], config[name + "_query_limit"]
        if start < 0 or count <= 0: raise ValueError(f"Invalid {name} query split")
        ranges.append((start, start + count, name))
    for i, (a, b, name) in enumerate(ranges):
        for c, d, other in ranges[i + 1:]:
            if max(a, c) < min(b, d): raise ValueError(f"Overlapping {name}/{other} query splits")
    if Path(config["train_queries"]).resolve() == Path(config["test_queries"]).resolve():
        raise ValueError("Test query file must be independent from training query file")
    allowed = sorted(os.sched_getaffinity(0))
    cpus = config["cpus"] or allowed[:config["search_threads"]]
    if len(set(cpus)) != len(cpus) or len(cpus) < config["search_threads"] or not set(cpus) <= set(allowed):
        raise ValueError(f"Invalid CPU affinity; allowed CPUs: {allowed}")
    config["cpus"] = cpus
    return config


def matrix_header(path, dtype="<f4"):
    path = Path(path)
    with path.open("rb") as f: header = f.read(8)
    if len(header) != 8: raise ValueError(f"Truncated matrix header: {path}")
    rows, dim = struct.unpack("<II", header)
    expected = 8 + rows * dim * np.dtype(dtype).itemsize
    if not rows or not dim or path.stat().st_size != expected:
        raise ValueError(f"Matrix size mismatch: {path}; expected {expected}, actual {path.stat().st_size}")
    return rows, dim


def matrix(path, dtype="<f4"):
    shape = matrix_header(path, dtype)
    return np.memmap(path, mode="r", dtype=dtype, offset=8, shape=shape)


def write_matrix(path, values, dtype="<f4"):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    array = np.ascontiguousarray(values, dtype=dtype)
    if array.ndim != 2: raise ValueError("Expected matrix")
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as f:
        f.write(struct.pack("<II", *array.shape)); array.tofile(f); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def file_id(path):
    """Size, mtime, and first/last MiB fingerprint; not a full-file checksum."""
    path = Path(path); st = path.stat()
    digest = hashlib.sha256(str((st.st_size, st.st_mtime_ns)).encode())
    with path.open("rb") as f:
        digest.update(f.read(1 << 20))
        if st.st_size > 1 << 20:
            f.seek(max(0, st.st_size - (1 << 20))); digest.update(f.read(1 << 20))
    return digest.hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def content_sha256(path):
    """Stable full content hash for the small query/ground-truth files."""
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""): digest.update(block)
    return digest.hexdigest()


def truth_depth(config):
    # Recall@10 needs ten; BoundaryInv K10/h10 needs twenty.
    return max(20, *config["inversion_ranks"])


def read_csv(path):
    if not Path(path).exists(): return []
    with Path(path).open(newline="", encoding="utf-8-sig") as f: return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    rows = list(rows)
    fields = fields or list(dict.fromkeys(k for r in rows for k in r))
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
        f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


class CsvLedger:
    def __init__(self, path, fields=RESULT_FIELDS):
        self.path, self.fields = Path(path), fields
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.path.exists():
            with self.path.open(newline="") as f:
                if next(csv.reader(f), None) != fields: raise ValueError("CSV schema changed; use a new output root")
        else: write_csv(self.path, [], fields)

    def append(self, record):
        with self.path.open("a", newline="", encoding="utf-8") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            w = csv.DictWriter(f, fieldnames=self.fields); w.writerow(record)
            f.flush(); os.fsync(f.fileno())

    def rows(self): return read_csv(self.path)


def performance_jobs(config):
    """Dense beam/depth grid; FP32 shares the vector IO and memory policy."""
    jobs = []
    for b in config["beams"]:
        for m in config["methods"]:
            for variant in ("primary", config["deploy_variant"]):
                for depth in sorted({0, b, *(min(d, b) for d in config["rerank_depths"])}):
                    for repeat in range(1, config["repeats"] + 1):
                        jobs.append(dict(experiment=1, primary=m, variant=variant, beam=b,
                                         rerank_depth=depth, repeat=repeat, side_bits=0 if variant == "primary" else 64))
        if config["include_fp32_reference"]:
            for repeat in range(1, config["repeats"] + 1):
                jobs.append(dict(experiment=1, primary="full-precision", variant="fp32",
                                 beam=b, rerank_depth=0, repeat=repeat, side_bits=0))
    return jobs


def recall_at_10(ids, truth):
    if ids.shape[0] != truth.shape[0] or min(ids.shape[1], truth.shape[1]) < 10:
        raise ValueError("Recall query counts/topk mismatch")
    if any(len(set(map(int, row[:10]))) != 10 for row in ids):
        raise ValueError("Duplicate returned IDs would corrupt Recall")
    return float(np.mean([len(set(map(int, a[:10])) & set(map(int, b[:10]))) / 10
                          for a, b in zip(ids, truth)]))


def pair_statistics(exact, approx, qids, nq):
    e, a, q = [np.asarray(v).reshape(-1) for v in (exact, approx, qids)]
    if e.shape != a.shape or e.shape != q.shape or not np.isfinite(e).all() or not np.isfinite(a).all():
        raise ValueError("Invalid pair score arrays")
    if len(q) and (q.min() < 0 or q.max() >= nq): raise ValueError("Pair query ID out of range")
    q = q.astype(np.int64); valid = e != 0
    den = np.bincount(q[valid], minlength=nq)
    wrong = np.sign(e[valid]) != np.sign(a[valid])
    num = np.bincount(q[valid], weights=wrong, minlength=nq)
    eligible = den > 0
    return dict(queries=nq, queries_with_pairs=int(eligible.sum()),
        queries_without_pairs=int((~eligible).sum()), pairs=int(valid.sum()), inversions=int(wrong.sum()),
        exact_ties_excluded=int((~valid).sum()),
        inversion_rate=float((num[eligible] / den[eligible]).mean()) if eligible.any() else "",
        pair_weighted_inversion_rate=float(num.sum() / den.sum()) if den.sum() else "",
        approximate_tie_rate=float((a[valid] == 0).mean()) if valid.any() else "")


def global_inversions(exact, approx, ranks=INVERSION_RANKS):
    required = max(20, *ranks)
    if exact.ndim != 2 or exact.shape != approx.shape or exact.shape[1] < required:
        raise ValueError(f"Full-database exact Top-{required} score matrices required")
    order = np.argsort(-exact, axis=1, kind="stable")
    e, a = [np.take_along_axis(v, order, axis=1) for v in (exact, approx)]
    result = []
    for r in ranks:
        de, da = e[:, :r - 1] - e[:, 1:r], a[:, :r - 1] - a[:, 1:r]
        q = np.broadcast_to(np.arange(len(e))[:, None], de.shape)
        result.append(dict(metric=f"AdjacentInv-R{r}", **pair_statistics(de, da, q, len(e))))
    for h in (5, 10):
        de = e[:, 10-h:10, None] - e[:, None, 10:10+h]
        da = a[:, 10-h:10, None] - a[:, None, 10:10+h]
        q = np.broadcast_to(np.arange(len(e))[:, None, None], de.shape)
        result.append(dict(metric=f"BoundaryInv-K10-h{h}", **pair_statistics(de, da, q, len(e))))
    return result


def aggregate(rows, repeats):
    groups = {}
    for row in rows:
        if row["status"] != "OK": continue
        key = tuple(row[k] for k in ("experiment", "primary", "variant", "side_bits", "beam", "rerank_depth"))
        groups.setdefault(key, {})[int(row["repeat"])] = row
    out = []
    metrics = ["qps", "mean_service_latency_ms", "p50_ms", "p95_ms", "p99_ms",
               "visited", "expanded", "comparisons", "accounted_memory_bytes", "peak_rss_bytes"]
    for group in groups.values():
        rs = list(group.values()); first = dict(rs[0])
        recalls = [float(r["recall_at_10"]) for r in rs]
        if max(recalls) - min(recalls) > 1e-8: raise ValueError("Recall changed between deterministic timing repeats")
        for k in metrics: first[k] = statistics.median(float(r[k]) for r in rs)
        for k in ("primary_io_ms", "primary_logical_bytes", "primary_physical_bytes", "primary_read_ops",
                  "rerank_io_ms", "rerank_physical_bytes"):
            if all(r.get(k, "") != "" for r in rs):
                first[k] = statistics.median(float(r[k]) for r in rs)
        first.update(repeats=len(rs), timing_complete=len(rs) >= repeats,
                     qps_cv=statistics.pstdev(float(r["qps"]) for r in rs) / first["qps"] if first["qps"] else 0)
        out.append(first)
    return out


def matched_points(rows, targets=TARGETS):
    out = []
    groups = {}
    for r in rows:
        if not r.get("timing_complete"): continue
        key = (r["primary"], r["variant"], "rerank" if int(r["rerank_depth"]) else "no-rerank")
        groups.setdefault(key, []).append(r)
    # Keep failed/unreachable branches visible, including methods with no OK run.
    for key in sorted({(r["primary"], r["variant"], "rerank" if int(r["rerank_depth"]) else "no-rerank") for r in rows}):
        for target in targets:
            feasible = [r for r in groups.get(key, []) if float(r["recall_at_10"]) >= target]
            best = max(feasible, key=lambda r: (float(r["qps"]), -int(r["beam"]), -int(r["rerank_depth"]))) if feasible else None
            record = dict(primary=key[0], variant=key[1], mode=key[2], target_recall=target,
                          achieved=bool(best), selection="fastest fully repeated measured test-frontier point; no interpolation")
            for field in ("beam", "rerank_depth", "recall_at_10", "visited", "expanded", "comparisons",
                          "qps", "mean_service_latency_ms", "p50_ms", "p95_ms", "p99_ms"):
                record[field] = best[field] if best else ""
            out.append(record)
    return out
