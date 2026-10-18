"""Preparation, native builds and per-budget training for the PDF suite."""
from __future__ import annotations

import importlib.util
import importlib.metadata
import json
import math
import os
from pathlib import Path
import resource
import platform
import shutil
import subprocess
import sys
import time

import numpy as np

from protocol import (ROOT, HERE, CsvLedger, file_id, identity, matrix, matrix_header,
                      content_sha256, truth_depth, write_csv, write_matrix)

STAGE_FIELDS = ["stage", "status", "elapsed_s", "peak_rss_bytes", "returncode", "log", "command", "error"]


def save_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(value, f, indent=2, sort_keys=True); f.write("\n"); f.flush(); os.fsync(f.fileno())
    tmp.replace(path)


def child_environment(config):
    env = dict(os.environ)
    # Keep an explicitly supplied dependency environment ahead of repo modules.
    env["PYTHONPATH"] = os.pathsep.join(filter(None, [env.get("PYTHONPATH", ""),
        str(ROOT / "ours/python"), str(HERE)]))
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        env[key] = str(config["build_threads"])
    return env


def process_setup(cpus=None, budget=0):
    def setup():
        if cpus: os.sched_setaffinity(0, cpus)
        if budget: resource.setrlimit(resource.RLIMIT_AS, (budget, budget))
    return setup


class Pipeline:
    def __init__(self, config):
        self.c = config; self.out = Path(config["output_root"]); self.out.mkdir(parents=True, exist_ok=True)
        for number in range(1, 6):
            (self.out / f"experiment{number}").mkdir(exist_ok=True)
        self.env = child_environment(config)
        self.stages = CsvLedger(self.out / "stages.csv", STAGE_FIELDS)
        # A source change must invalidate resume, including scorer/trainer changes.
        sources = list((ROOT / "ours/python").glob("*.py")) + [p for p in HERE.glob("*.py") if not p.name.startswith("test_")]
        sources += list((ROOT / "ours/experimental/traversal-fast-v1").rglob("*.cpp"))
        sources += list((ROOT / "ours/experimental/traversal-fast-v1").rglob("*.h"))
        sources += list((ROOT / "ours/src").glob("*.cpp")) + list((ROOT / "ours/include").rglob("*.h"))
        sources += [ROOT / "ours/CMakeLists.txt", ROOT / "src/index_bipartite.cpp",
                    ROOT / "include/index_bipartite.h",
                    ROOT / "ours/test/t2i-1M/cpp/exp234_audit.cpp",
                    ROOT / "thirdparty/RaBitQ-Library/sample/roar_rabitq_quantize.cpp",
                    ROOT / "thirdparty/SAQ/src/export_roar_primary.cpp"]
        self.source_id = identity({str(p.relative_to(ROOT)): file_id(p) for p in sorted(set(sources))})
        self.config_id = identity(config)
        frozen = self.out / "configuration.json"
        value = dict(config_id=self.config_id, source_id=self.source_id, config=config)
        if frozen.exists() and json.loads(frozen.read_text()) != value:
            raise ValueError("Output root belongs to different config/source; choose a new output_root")
        if not frozen.exists(): save_json(frozen, value)
        versions=[]
        for distribution in ("numpy", "torch", "faiss-cpu", "threadpoolctl"):
            try: version=importlib.metadata.version(distribution)
            except importlib.metadata.PackageNotFoundError: version="MISSING"
            versions.append(dict(component=distribution,value=version))
        versions.extend([dict(component="python",value=platform.python_version()),
            dict(component="executable",value=sys.executable),dict(component="platform",value=platform.platform()),
            dict(component="allowed_cpus",value=json.dumps(sorted(os.sched_getaffinity(0)))),
            dict(component="search_cpus",value=json.dumps(config["cpus"]))])
        write_csv(self.out/"environment.csv",versions)
        availability = [dict(method=m, status="AVAILABLE", implementation=(
            "paper reproduction; see primary export metadata" if m in ("qaq", "leanvec-ood") else "native adapter"))
            for m in ("opq-pq", "qaq", "leanvec-ood", "extended-rabitq", "saq")]
        availability += [dict(method=m, status="UNAVAILABLE", implementation="no connected native adapter")
                         for m in ("scann-avq", "rabitq", "lvq")]
        write_csv(self.out / "baseline-availability.csv", availability)

    def preflight(self):
        rows = []; okay = True
        for key in ("base", "train_queries", "test_queries"):
            try:
                n, d = matrix_header(self.c[key])
                if d != self.c["base_dim"]: raise ValueError("dimension differs from configured raw dimension")
                if key == "base" and n != self.c["base_rows"]: raise ValueError("row count differs from configured full database")
                if key == "test_queries" and n < self.c["test_query_limit"]: raise ValueError("test query count too small")
                if key == "train_queries" and n < max(self.c[k + "_query_start"] + self.c[k + "_query_limit"] for k in ("metric", "select", "build")):
                    raise ValueError("training query splits extend beyond file")
                rows.append(dict(check=key, status="OK", detail=f"{n} x {d}; header/length checked, content checksum not verified"))
            except (OSError, ValueError) as e:
                rows.append(dict(check=key, status="BLOCKED", detail=str(e))); okay = False
        for module in ("torch", "faiss"):
            found = importlib.util.find_spec(module) is not None
            rows.append(dict(check=module, status="OK" if found else "BLOCKED", detail="Python import available" if found else "install training dependency"))
            okay &= found
        if self.c.get("exact_backend") == "faiss-cpu":
            try:
                from exact_faiss_cpu import implementation
                detail = json.dumps(implementation(), sort_keys=True)
                rows.append(dict(check="faiss_cpu_exact", status="OK", detail=detail))
            except (ImportError, RuntimeError) as exc:
                rows.append(dict(check="faiss_cpu_exact", status="BLOCKED", detail=str(exc))); okay = False
        if self.c["device"].startswith("cuda") and importlib.util.find_spec("torch"):
            r = subprocess.run([sys.executable, "-c", "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)"], env=self.env, capture_output=True)
            rows.append(dict(check="cuda", status="OK" if r.returncode == 0 else "BLOCKED", detail=self.c["device"])); okay &= r.returncode == 0
        rows.append(dict(check="search_memory_limit", status="INFO", detail=f"{self.c['memory_budget_bytes']} bytes, index admission + child RLIMIT_AS + measured peak RSS"))
        rows.append(dict(check="uniform_fp32_storage", status="INFO",
            detail=f"FP32 baseline and rerank both use {self.c['fp32_storage']}; same hard search-memory limit"))
        rows.append(dict(check="shared_primary_graph", status="INFO",
            detail=f"{len(self.c['methods'])} primary-only graphs + 1 FP32 teacher; all JI variants/budgets reuse their Q graph"))
        rows.append(dict(check="construction_memory_limit", status="INFO", detail=str(self.c["construction_memory_budget_bytes"]) + " bytes; 0 means unrestricted offline construction"))
        write_csv(self.out / "preflight.csv", rows)
        return okay

    def stage(self, name, command, outputs, inputs=(), budget=None):
        outputs, inputs = [Path(p) for p in outputs], [Path(p) for p in inputs]
        executable=shutil.which(str(command[0]))
        if executable: inputs.append(Path(executable))
        signature = identity(dict(command=list(map(str, command)), inputs={str(p): file_id(p) for p in inputs}, source=self.source_id))
        stamp = self.out / "stamps" / (name + ".json")
        if stamp.exists() and all(p.exists() for p in outputs):
            old = json.loads(stamp.read_text())
            if old.get("signature") == signature and old.get("outputs") == {str(p): file_id(p) for p in outputs}:
                print(f"[resume] {name}", flush=True); return
        log = self.out / "logs" / (name + ".log"); log.parent.mkdir(parents=True, exist_ok=True)
        timing = log.with_suffix(".time.csv")
        for p in outputs: p.parent.mkdir(parents=True, exist_ok=True)
        argv = list(map(str, command)); wrapped = [sys.executable, str(HERE/"measured_subprocess.py"), str(timing), *argv]
        started = time.monotonic(); print(f"[stage] {name}; log={log}", flush=True)
        with log.open("w") as stream:
            process = subprocess.Popen(wrapped, stdout=stream, stderr=subprocess.STDOUT, env=self.env,
                preexec_fn=process_setup(budget=self.c["construction_memory_budget_bytes"] if budget is None else budget))
            while process.poll() is None:
                try: process.wait(timeout=30)
                except subprocess.TimeoutExpired: print(f"[running] {name} {time.monotonic()-started:.0f}s", flush=True)
        peak = ""
        if timing.exists():
            try: peak = int(timing.read_text().strip().splitlines()[-1].split(",")[1]) * 1024
            except (ValueError, IndexError): pass
        error = "" if process.returncode == 0 else f"command exited {process.returncode}; see log"
        if not error and not all(p.is_file() and p.stat().st_size for p in outputs): error = "expected output missing/empty"
        self.stages.append(dict(stage=name, status="FAILED" if error else "OK", elapsed_s=time.monotonic()-started,
            peak_rss_bytes=peak, returncode=process.returncode, log=str(log), command=json.dumps(argv), error=error))
        if error: raise RuntimeError(f"{name}: {error}: {log}")
        save_json(stamp, dict(signature=signature, outputs={str(p): file_id(p) for p in outputs}))

    def python(self, script, **options):
        cmd = [sys.executable, str(ROOT / "ours/python" / script)]
        for key, value in options.items():
            if value is None: continue
            cmd += ["--" + key.replace("_", "-")]
            if value is not True: cmd.append(str(value))
        return cmd

    def data(self):
        c = self.c; dest = self.out / "data"; dest.mkdir(exist_ok=True)
        inputs = {k: file_id(c[k]) for k in ("base", "train_queries", "test_queries")}
        stamp = dest / "inputs.json"
        if stamp.exists():
            previous = json.loads(stamp.read_text())
            if previous["inputs"] != inputs: raise ValueError("Data changed in existing output root")
            for path, fid in previous["outputs"].items():
                if file_id(path) != fid: raise ValueError("Prepared query split changed")
            return
        train, test = matrix(c["train_queries"]), matrix(c["test_queries"])
        for split in ("metric", "select", "build"):
            start, limit = c[split + "_query_start"], c[split + "_query_limit"]
            write_matrix(dest / (split + ".fbin"), train[start:start+limit])
        write_matrix(dest / "test.fbin", test[:c["test_query_limit"]])
        combined = np.concatenate([matrix(dest / (s + ".fbin")) for s in ("metric", "select", "build", "test")])
        write_matrix(dest / "all-queries.fbin", combined)
        write_csv(self.out / "query-splits.csv", [dict(split=s, source=c["test_queries"] if s == "test" else c["train_queries"],
            start=0 if s == "test" else c[s+"_query_start"], count=c["test_query_limit"] if s == "test" else c[s+"_query_limit"],
            usage={"metric":"primary/JI metric fitting", "select":"reserved training validation; never test-fit", "build":"graph construction", "test":"evaluation only"}[s]) for s in ("metric", "select", "build", "test")])
        save_json(stamp, dict(inputs=inputs, outputs={str(p): file_id(p) for p in sorted(dest.glob("*.fbin"))}))

    def compile(self):
        c = self.c; build = Path(c["build_dir"])
        targets = ["roar_primary_ji_search", "roar_exp234_audit", "roar_primary_ji_build", "roar_fp_build",
                   "roar_export_rabitq_primary", "roar_rabitq_quantize", "roar_export_saq_primary"]
        cmd = ["cmake", "-S", str(ROOT / "ours"), "-B", str(build), "-DCMAKE_BUILD_TYPE=Release",
               "-DOURS_JI_KERNEL_ROOT=" + str(ROOT / "ours/experimental/traversal-fast-v1")]
        if c["cmake_prefix_path"]:
            cmd += ["-DCMAKE_PREFIX_PATH=" + c["cmake_prefix_path"]]
            modules = Path(c["cmake_prefix_path"]) / "share/glog/cmake"
            if modules.is_dir(): cmd += ["-DCMAKE_MODULE_PATH=" + str(modules)]
        self.stage("cmake-configure", cmd, [build / "CMakeCache.txt"], budget=0)
        self.stage("cmake-build", ["cmake", "--build", str(build), "--target", *targets, "-j", c["compile_jobs"]],
                   [build / t for t in targets], [build / "CMakeCache.txt"], budget=0)

    def exact(self, query_path, prefix, k):
        """Full-database blocked FP32 IP; no 1M reuse or reduced search domain."""
        ids_path, scores_path = Path(str(prefix)+".ids.ibin"), Path(str(prefix)+".scores.fbin")
        meta_path = Path(str(prefix)+".metadata.json")
        sig = dict(base=file_id(self.c["base"]), queries=file_id(query_path), rows=self.c["base_rows"], k=k,
                   algorithm="blocked-fp32-ip-v2-base-outer", device=self.c["device"])
        backend = self.c.get("exact_backend", "numpy")
        if backend == "faiss-cpu":
            from exact_faiss_cpu import ALGORITHM, implementation
            sig.update(algorithm=ALGORITHM, implementation=implementation(),
                base_batch_size=self.c["encode_batch_size"], query_batch_size=self.c["exact_query_batch_size"])
        if meta_path.exists() and ids_path.exists() and scores_path.exists():
            old = json.loads(meta_path.read_text())
            if old.get("inputs") == sig and old.get("ids_id") == file_id(ids_path) and old.get("scores_id") == file_id(scores_path): return
        if backend == "faiss-cpu":
            return self._exact_faiss_cpu(query_path, prefix, k, sig, ids_path, scores_path, meta_path)
        base, queries = matrix(self.c["base"]), np.asarray(matrix(query_path), dtype=np.float32)
        if base.shape != (self.c["base_rows"], self.c["base_dim"]): raise ValueError("Exact GT must cover configured database")
        gpu = self.c["device"].startswith("cuda")
        if gpu:
            import torch
            if not torch.cuda.is_available(): raise RuntimeError("CUDA requested but unavailable")
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.set_float32_matmul_precision("highest")
        else:
            from threadpoolctl import threadpool_limits
        if not np.isfinite(queries).all(): raise ValueError("Nonfinite exact-search queries")
        nq = len(queries)
        result_ids = np.full((nq, k), np.iinfo(np.uint32).max, dtype=np.uint32)
        result_scores = np.full((nq, k), -np.inf, dtype=np.float32)
        started = time.monotonic(); last_progress=started
        if gpu: all_queries = torch.as_tensor(queries, device=self.c["device"])
        # Read each database block once. Query-outer traversal would rescan and
        # transfer the 800GB base for every 32-query batch.
        for start in range(0, len(base), self.c["encode_batch_size"]):
            values = np.ascontiguousarray(base[start:start+self.c["encode_batch_size"]])
            if not np.isfinite(values).all(): raise ValueError(f"Nonfinite base rows near {start}")
            count=min(k,len(values))
            if gpu: tbase=torch.as_tensor(values,device=self.c["device"])
            for qs in range(0,nq,32):
                q=queries[qs:qs+32]
                local_i=np.empty((len(q),count),dtype=np.uint32)
                local_s=np.empty((len(q),count),dtype=np.float32)
                if gpu:
                    score=all_queries[qs:qs+len(q)] @ tbase.T
                    top_s,top_i=torch.topk(score,count,dim=1,sorted=True)
                    local_s[:]=top_s.cpu().numpy(); local_i[:]=top_i.cpu().numpy()
                    cutoff=top_s[:,-1:]
                    ambiguous=((score==cutoff).sum(dim=1)>(top_s==cutoff).sum(dim=1)).cpu().numpy()
                    # Only a cutoff tie needs the full row copied to CPU.
                    for qi in np.flatnonzero(ambiguous):
                        row=score[int(qi)].cpu().numpy(); threshold=local_s[qi,-1]
                        higher=np.flatnonzero(row>threshold)
                        tied=np.flatnonzero(row==threshold)[:count-len(higher)]
                        local_i[qi]=np.concatenate((higher,tied));local_s[qi]=row[local_i[qi]]
                else:
                    with threadpool_limits(limits=self.c["build_threads"]): score=q @ values.T
                    for qi,row in enumerate(score):
                        selected=np.argpartition(-row,count-1)[:count]
                        threshold=row[selected].min()
                        higher=np.flatnonzero(row>threshold)
                        tied=np.flatnonzero(row==threshold)[:count-len(higher)]
                        local_i[qi]=np.concatenate((higher,tied));local_s[qi]=row[local_i[qi]]
                local_i += start
                ms=np.concatenate((result_scores[qs:qs+len(q)],local_s),axis=1)
                mi=np.concatenate((result_ids[qs:qs+len(q)],local_i),axis=1)
                order=np.lexsort((mi,-ms),axis=1)[:,:k]
                result_scores[qs:qs+len(q)]=np.take_along_axis(ms,order,axis=1)
                result_ids[qs:qs+len(q)]=np.take_along_axis(mi,order,axis=1)
            if start==0 or start+len(values)==len(base) or time.monotonic()-last_progress>=30:
                print(f"[exact] {prefix} database {start+len(values)}/{len(base)}, all {nq} queries",flush=True)
                last_progress=time.monotonic()
        write_matrix(ids_path, result_ids, "<u4"); write_matrix(scores_path, result_scores)
        save_json(meta_path, dict(inputs=sig, ids_id=file_id(ids_path), scores_id=file_id(scores_path),
                                  score="FP32 inner product, TF32 disabled", exact_tie_order="ascending node ID"))
        self.stages.append(dict(stage="exact-"+Path(prefix).name, status="OK", elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, returncode=0, log="", command="blocked FP32 full database", error=""))

    def _exact_faiss_cpu(self, query_path, prefix, k, sig, ids_path, scores_path, meta_path):
        from exact_faiss_cpu import scan
        c = self.c; base, queries = matrix(c["base"]), matrix(query_path)
        if c["device"] != "cpu" or base.shape != (c["base_rows"], c["base_dim"]):
            raise ValueError("Faiss CPU exact GT must cover the configured full database on CPU")
        progress_path = Path(str(prefix)+".progress.csv")
        fields = ["algorithm", "completed_rows", "total_rows", "queries", "k", "elapsed_s",
            "resume_from", "checkpoint_rows", "checkpoint_written", "tie_repair_queries", "threads",
            "faiss_version", "faiss_compile_options", "scoring", "cutoff_ties"]
        ledger = CsvLedger(progress_path, fields)
        started = time.monotonic(); last_progress = 0.
        def progress(row):
            nonlocal last_progress
            now = time.monotonic()
            if not row["checkpoint_written"] and row["completed_rows"] < row["total_rows"] and now-last_progress < 30:
                return
            record = dict(algorithm=sig["algorithm"], **row)
            ledger.append(record)
            save_json(Path(str(prefix)+".progress.json"), record)
            print(f"[exact faiss-cpu] {prefix} database {row['completed_rows']}/{row['total_rows']}, "
                f"all {row['queries']} queries; checkpoint={row['checkpoint_rows']}; "
                f"resume_from={row['resume_from']}", flush=True)
            last_progress = now
        ids, scores, info = scan(base, queries, k, base_batch_size=c["encode_batch_size"],
            query_batch_size=c["exact_query_batch_size"], threads=c["build_threads"], signature=sig,
            checkpoint_path=Path(str(prefix)+".checkpoint.npz"),
            checkpoint_interval_s=c["exact_checkpoint_interval_s"], progress=progress)
        write_matrix(ids_path, ids, "<u4"); write_matrix(scores_path, scores)
        save_json(meta_path, dict(inputs=sig, ids_id=file_id(ids_path), scores_id=file_id(scores_path),
            score="FP32 inner product, TF32 disabled", exact_tie_order="ascending node ID",
            full_database=True, completed_base_rows=c["base_rows"], checkpoint_info=info))
        self.stages.append(dict(stage="exact-"+Path(prefix).name, status="OK", elapsed_s=time.monotonic()-started,
            peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024, returncode=0,
            log=str(progress_path), command="Faiss CPU AVX-512 FP32 Flat full database; blocked; checkpointed", error=""))

    def ground_truth(self):
        dest = self.out / "data"; c = self.c; depth = truth_depth(c)
        if c["ground_truth_ids"]:
            if not all(c[k] for k in ("ground_truth_scores", "ground_truth_metadata")): raise ValueError("External GT needs IDs, scores and provenance metadata")
            meta = json.loads(Path(c["ground_truth_metadata"]).read_text())
            expected = {"base_rows":c["base_rows"], "base_dim":c["base_dim"], "base_id":file_id(c["base"]),
                        "full_database":True, "score":"fp32-inner-product"}
            if any(meta.get(k) != v for k,v in expected.items()): raise ValueError("External exact GT provenance mismatch")
            query_matches = (meta["queries_sha256"] == content_sha256(dest/"test.fbin")) if "queries_sha256" in meta else (meta.get("queries_id") == file_id(dest/"test.fbin"))
            if not query_matches: raise ValueError("External exact GT query provenance mismatch")
            for key, path in (("ids_id", c["ground_truth_ids"]), ("scores_id", c["ground_truth_scores"])):
                if key in meta and meta[key] != file_id(path): raise ValueError("External exact GT fingerprint mismatch")
            ids, scores = matrix(c["ground_truth_ids"], "<u4"), matrix(c["ground_truth_scores"])
            if ids.shape[0] != c["test_query_limit"] or ids.shape[1] < depth or scores.shape != ids.shape or ids.max() >= c["base_rows"] or not np.isfinite(scores).all():
                raise ValueError(f"External exact Top-{depth} shape/IDs/scores invalid")
            if np.any(np.diff(np.sort(ids[:, :depth], axis=1), axis=1) == 0):
                raise ValueError("External exact GT contains duplicate IDs")
            write_matrix(dest / "truth.ids.ibin", ids[:, :depth], "<u4"); write_matrix(dest / "truth.scores.fbin", scores[:, :depth])
            save_json(dest/"truth.metadata.json", dict(source="external-ground-truth", required_topk=depth,
                provenance=meta, ids_id=file_id(dest/"truth.ids.ibin"), scores_id=file_id(dest/"truth.scores.fbin")))
        else: self.exact(dest / "test.fbin", dest / "truth", depth)
        # The common FP32 teacher is needed by experiment 3 even when its
        # separate performance-reference curve is disabled.
        self.exact(dest / "build.fbin", dest / "build-truth", min(100, c["base_rows"]))

    def primary(self, method):
        c = self.c; dest = self.out / "primary" / method; queries = self.out / "data/all-queries.fbin"
        summary = dest / "export_summary.json"; total_queries = sum(c[s+"_query_limit"] for s in ("metric", "select", "build"))+c["test_query_limit"]
        common = [dest / p for p in ("base.transformed.fbin", "primary.reconstruction.fbin", "query.transformed.fbin", "parity.candidate_ids.ibin", "parity.official_scores.fbin")]
        if method in ("opq-pq", "qaq", "leanvec-ood"):
            cmd = self.python("train_generic_primary.py", method=method, base_fbin=c["base"], query_fbin=queries,
                out_dir=dest, train_size=c["primary_train_size"], query_start=0, query_limit=total_queries,
                fit_query_start=0, fit_query_limit=c["metric_query_limit"], seed=c["seed"], iterations=c["primary_iterations"],
                batch_size=c["encode_batch_size"], threads=c["build_threads"], primary_budget_bits=c["primary_budget_bits"],
                opq_dim=c["primary_dim"], pq_groups=c["pq_groups"], leanvec_dim=c["primary_dim"])
            outputs = common + [summary, dest / "primary_model.bin", dest / "primary_codes.bin"]
        elif method == "extended-rabitq":
            quant = dest / "rabitq.codes.bin"
            self.stage("quantize-rabitq", [Path(c["build_dir"])/"roar_rabitq_quantize", c["base"], quant, c["rabitq_bits"], "ip", str(c["rabitq_config"]=="faster").lower()], [quant], [c["base"]])
            cmd = [Path(c["build_dir"])/"roar_export_rabitq_primary", quant, c["base"], queries, dest, 0, total_queries, c["build_threads"]]
            outputs = common + [summary]
        else:
            cmd = [Path(c["build_dir"])/"roar_export_saq_primary", "--base_fbin="+c["base"], "--query_fbin="+str(queries),
                "--out_dir="+str(dest), "--avg_bits="+str(c["saq_avg_bits"]), "--threads="+str(c["build_threads"]),
                "--query_limit="+str(total_queries), "--seed="+str(c["seed"]), "--export_full_codes=true"]
            outputs = common + [summary, dest / "saq_model.bin", dest / "saq_codes.bin"]
        self.stage("primary-"+method, cmd, outputs, [c["base"], queries])
        parity = dest / "scorer-parity.json"
        self.stage("parity-"+method, self.python("audit_primary_scorer_parity.py", reconstruction_fbin=dest/"primary.reconstruction.fbin",
            query_fbin=dest/"query.transformed.fbin", candidate_ids=dest/"parity.candidate_ids.ibin", official_values=dest/"parity.official_scores.fbin",
            official_value_kind="similarity", out_json=parity), [parity], common[1:])
        info = json.loads(summary.read_text())
        if info["nominal_code_bits_per_node"] != c["primary_budget_bits"]: raise ValueError(f"{method}: nominal primary bits differ from common budget")
        manifest = dest / "manifest.json"
        self.stage("manifest-"+method, self.python("create_primary_quantizer_manifest.py", name=method,
            base_fbin=dest/"base.transformed.fbin", reconstruction_fbin=dest/"primary.reconstruction.fbin", query_fbin=dest/"query.transformed.fbin",
            nominal_code_bits_per_node=info["nominal_code_bits_per_node"], physical_bytes_per_node=info["physical_bytes_per_node"],
            implementation=info["implementation"], scorer_parity_json=parity, metadata_json=summary, out_manifest=manifest), [manifest], [summary, parity])
        return self.primary_paths(method)

    def primary_paths(self, method):
        if method == "full-precision":
            path=str(self.exact_vector_path())
            return dict(kind="fp32", model=path, codes=path,
                implementation=f"FP32 graph search; {self.c['fp32_storage']} vector store; common search-memory limit")
        dest = self.out / "primary" / method; info = json.loads((dest/"export_summary.json").read_text())
        names = {"extended-rabitq":("rabitq.codes.bin", "rabitq.codes.bin"), "saq":("saq_model.bin", "saq_codes.bin")}
        model, codes = names.get(method, ("primary_model.bin", "primary_codes.bin"))
        return dict(kind=method, model=str(dest/model), codes=str(dest/codes), implementation=info["implementation"], manifest=str(dest/"manifest.json"))

    def branch(self, method, variant, bits):
        result = self.primary_paths(method)
        dest = self.out / "ji" / method / f"b{bits}" / variant
        # JI changes search scores only. Every ablation and bit budget uses
        # the exact same topology as its primary-only baseline.
        graph_variant = "fp32" if method == "full-precision" else "primary"
        result.update(ji_model=str(dest/"online.bin"), ji_codes=str(dest/"side.codes"),
                      graph=str(self.out/"graphs"/method/f"{graph_variant}-b0.graph"))
        return result

    def ji(self, method, bits):
        c = self.c; primary = self.primary_paths(method); manifest = Path(primary["manifest"])
        inputs = self.out / "ji" / method / "inputs"
        self.stage("ji-input-"+method, self.python("build_primary_agnostic_ji_inputs.py", primary_manifest=manifest,
            query_start=0, query_limit=c["metric_query_limit"], residual_sample_size=c["residual_sample_size"],
            residual_batch_size=c["encode_batch_size"], lambda_values="", seed=c["seed"], out_dir=inputs),
            [inputs/"inputs.json", inputs/"residual-sample.npy", inputs/"metrics/pure-ood.npz"], [manifest])
        dest = self.out / "ji" / method / f"b{bits}"; summary = dest/"train_summary.json"
        max_bits = max(c["ji_max_bits"], math.ceil(bits/c["ji_atomic_groups"])+1)
        # Every budget reruns distortion fitting, DP and grouping.  No truncation
        # of a 64-bit code or shared rate allocation across budgets.
        cmd = self.python("train_joint_innovation_rpq.py", primary_manifest=manifest, metric_model=inputs/"metrics/pure-ood.npz",
            training_cache=inputs/"residual-sample.npy", out_dir=dest, budget_bits=bits, equal_groups=8 if bits<=64 else 16,
            atomic_groups=c["ji_atomic_groups"], min_bits=1, max_bits=max_bits, merge_max_bits=max(c["ji_merge_max_bits"], max_bits),
            merge_rounds=c["ji_merge_rounds"], merge_candidates=c["ji_merge_candidates"], fit_sample_size=c["ji_fit_size"],
            iterations=c["ji_iterations"], batch_size=c["ji_batch_size"], solver_device=c["device"], seed=c["seed"])
        variants = ("equal-rate", "atomic-equal-rate", "rate-adaptive", "joint-grouping")
        self.stage(f"ji-train-{method}-b{bits}", cmd, [summary]+[dest/v/"model/joint_innovation_model.npz" for v in variants],
                   [manifest, inputs/"metrics/pure-ood.npz", inputs/"residual-sample.npy"])
        trained = json.loads(summary.read_text()); curves=[]; losses=[]
        for group, rates in trained["distortion_curves"].items():
            for rate, value in rates.items():
                row=dict(value)
                row.update(primary=method, budget=bits, group=group, bits=rate)
                curves.append(row)
        for variant, info in trained["variants"].items():
            losses.append(dict(primary=method, budget=bits, variant=variant, groups=json.dumps(info["groups"]),
                allocation=json.dumps(info["bits"]), side_bits=info["side_bits_per_node"], distortion=info["validation_joint_distortion"],
                lut_entries=info["total_lut_entries"], training_seed=c["seed"]))
        write_csv(dest/"distortion-curves.csv", curves); write_csv(dest/"training-variants.csv", losses)
        needed = set(c["ablation_variants"]) | {c["deploy_variant"]} if bits==64 else {"joint-grouping"}
        # Experiment 5 always evaluates the DP+grouping model, including at64.
        needed.add("joint-grouping")
        for variant in sorted(needed):
            branch = self.branch(method, variant, bits); vdest = dest/variant; model = vdest/"model/joint_innovation_model.npz"
            self.stage(f"ji-encode-{method}-{variant}-b{bits}", self.python("encode_primary_agnostic_joint_innovation.py",
                primary_manifest=manifest, joint_model=model, out_side_codes=branch["ji_codes"], out_summary=vdest/"encoding.json",
                solver_device=c["device"], batch_size=c["encode_batch_size"]), [branch["ji_codes"], vdest/"encoding.json"], [manifest, model])
            self.stage(f"ji-online-{method}-{variant}-b{bits}", self.python("export_joint_innovation_online.py", joint_model=model,
                side_codes=branch["ji_codes"], out_model=branch["ji_model"], out_summary=vdest/"online.json", format_version=2),
                [branch["ji_model"], vdest/"online.json"], [model, branch["ji_codes"]])

    def exact_vector_path(self):
        return self.out/"data/base.cold" if self.c["rerank_storage"]=="direct" else Path(self.c["base"])

    def native_args(self, branch, variant):
        args = ["--primary-kind", branch["kind"], "--primary-model", branch["model"], "--primary-codes", branch["codes"]]
        if branch["kind"] == "fp32": args += ["--fp32-storage", self.c["fp32_storage"]]
        if branch["kind"] == "extended-rabitq": args += ["--rabitq-config", self.c["rabitq_config"]]
        if variant in ("primary", "fp32"): args += ["--disable-ji"]
        else: args += ["--ji-model", branch["ji_model"], "--ji-side-codes", branch["ji_codes"]]
        return args

    def graph(self, method, variant="primary", bits=0):
        """Build the primary-only graph, also when requested by a JI branch."""
        variant, bits = ("fp32" if method == "full-precision" else "primary"), 0
        c = self.c; branch = self.branch(method, variant, bits); graph = Path(branch["graph"]); knn = graph.with_suffix(".knn.ibin")
        if method == "full-precision":
            cmd = [Path(c["build_dir"])/"roar_fp_build", "--base", c["base"], "--queries", self.out/"data/build.fbin",
                "--knn", self.out/"data/build-truth.ids.ibin", "--graph", graph, "--threads", c["build_threads"],
                "--topk", 100, "--M_pjbp", 32, "--L_pjpq", 100]
            deps = [c["base"], self.out/"data/build.fbin", self.out/"data/build-truth.ids.ibin"]
            outputs=[graph]
        else:
            faiss_quantized = c["device"] == "cpu" and c["exact_backend"] == "faiss-cpu"
            if faiss_quantized:
                # The exporter already produced exactly the reconstruction and
                # transformed queries corresponding to this primary scorer.
                # A separate stage keeps the KNN checkpoint reusable if graph
                # construction is interrupted after the exhaustive scan.
                primary_dir = self.out / "primary" / method
                knn_meta = knn.with_suffix(".metadata.json")
                query_start = c["metric_query_limit"] + c["select_query_limit"]
                command = [sys.executable, str(HERE / "quantized_faiss_cpu.py"),
                           "--method", method,
                           "--reconstruction-fbin", str(primary_dir / "primary.reconstruction.fbin"),
                           "--query-fbin", str(primary_dir / "query.transformed.fbin"),
                           "--primary-model", str(branch["model"]),
                           "--primary-codes", str(branch["codes"]),
                           "--query-start", str(query_start),
                           "--query-limit", str(c["build_query_limit"]),
                           "--topk", "100",
                           "--base-batch-size", str(c["encode_batch_size"]),
                           "--query-batch-size", str(c["exact_query_batch_size"]),
                           "--threads", str(c["build_threads"]),
                           "--checkpoint-interval-s", str(c["exact_checkpoint_interval_s"]),
                           "--output-knn", str(knn),
                           "--output-metadata", str(knn_meta)]
                self.stage("knn-faiss-" + method, command, [knn, knn_meta],
                           [primary_dir / "primary.reconstruction.fbin",
                            primary_dir / "query.transformed.fbin",
                            branch["model"], branch["codes"]])
            cmd = [Path(c["build_dir"])/"roar_primary_ji_build", *self.native_args(branch, variant), "--queries", self.out/"data/build.fbin",
                "--query-start", 0, "--query-limit", c["build_query_limit"], "--knn", knn, "--graph", graph,
                "--summary", graph.with_suffix(".json"), "--topk", 100, "--M_pjbp", 32, "--L_pjpq", 100,
                "--threads", c["build_threads"], "--score-batch", c["encode_batch_size"]]
            if not faiss_quantized: cmd += ["--force-knn"]
            deps=[branch["model"], branch["codes"], self.out/"data/build.fbin"]
            if faiss_quantized: deps += [knn, knn_meta]
            outputs=[graph, graph.with_suffix(".json")] if faiss_quantized else [graph, knn, graph.with_suffix(".json")]
        self.stage(f"graph-{method}-{variant}-b{bits}", cmd, outputs, deps)
        save_json(graph.with_suffix(".provenance.json"), dict(graph_id=file_id(graph), representation_id=identity({str(p):file_id(p) for p in deps}),
            primary=method, variant=variant, bits=bits, graph_policy="shared-primary", ji_used_in_build=False,
            build_queries_id=file_id(self.out/"data/build.fbin"), config_id=self.config_id,
            knn_backend=("faiss-cpu-flat-fp32" if method == "full-precision" and c["exact_backend"] == "faiss-cpu" else
                         "exact-fp32-" + c["device"] if method == "full-precision" else
                         "faiss-cpu-flat-reconstruction" if c["device"] == "cpu" and c["exact_backend"] == "faiss-cpu" else
                         "native")))

    def prepare(self):
        if not self.preflight(): raise RuntimeError(f"Preflight blocked; see {self.out/'preflight.csv'}")
        self.data(); self.compile(); self.ground_truth()
        if self.c["rerank_storage"] == "direct":
            self.stage("cold-vectors", self.python("build_cold_vector_store.py", input_fbin=self.c["base"], output=self.out/"data/base.cold", overwrite=True),
                       [self.out/"data/base.cold"], [self.c["base"]])
        for method in self.c["methods"]:
            self.primary(method); self.graph(method)
            for bits in self.c["budgets"]:
                self.ji(method, bits)
        self.graph("full-precision")
