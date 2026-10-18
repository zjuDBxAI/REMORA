#!/usr/bin/env python3
"""Run PDF experiments 1--5, preserving quantitative results in CSV first."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import struct
import time

import numpy as np

from pipeline import Pipeline, process_setup
from adaptive_performance import (COVERAGE_FIELDS, GAP_FIELDS, JOB_FIELDS, curve_specs,
    initial_jobs, initial_performance_jobs, matches, refinement, repeat_jobs)
from protocol import (CsvLedger, INVERSION_FIELDS, RESULT_FIELDS, TARGETS, aggregate,
                      file_id, global_inversions, identity, load_config, matched_points,
                      matrix, pair_statistics, performance_jobs, read_csv, recall_at_10,
                      truth_depth, write_csv, write_matrix)


def experiment_jobs(c, experiments):
    if c["performance_sampling"] == "adaptive":
        jobs = initial_performance_jobs(c, experiments)
    else:
        jobs = performance_jobs(c) if 1 in experiments or 4 in experiments else []
    if 2 in experiments:
        for method in c["methods"]:
            for variant in c["ablation_variants"]:
                for repeat in range(1, c["repeats"]+1):
                    jobs.append(dict(experiment=2, primary=method, variant=variant, side_bits=64,
                        beam=c["ablation_beam"], rerank_depth=c["ablation_rerank_depth"], repeat=repeat))
    if 5 in experiments and c["performance_sampling"] == "dense":
        for method in c["methods"]:
            for bits in c["budgets"]:
                for beam in c["bit_budget_beams"]:
                    for repeat in range(1, c["repeats"]+1):
                        jobs.append(dict(experiment=5, primary=method, variant="joint-grouping", side_bits=bits,
                                         beam=beam, rerank_depth=0, repeat=repeat))
    return jobs


def experiment_record(row, number):
    """Experiment 4 reuses measured no-rerank experiment-1 points."""
    source = int(row["experiment"])
    if source != number and not (number == 4 and source == 1 and int(row.get("rerank_depth", -1)) == 0):
        return None
    result = dict(row)
    if number == 4:
        result.update(experiment=4, measurement_experiment=source)
    return result


class ExperimentCsvLedger:
    """Durable master plus immediate per-experiment records, repaired on resume."""
    def __init__(self, out, name="results.csv", fields=RESULT_FIELDS):
        out = Path(out)
        self.master = CsvLedger(out/name, fields)
        self.path = self.master.path
        self.fields = fields
        self.split = {}
        master_rows = self.master.rows()
        for number in range(1, 6):
            local_fields = list(fields) + (["measurement_experiment"] if number == 4 and name == "results.csv" else [])
            ledger = CsvLedger(out/f"experiment{number}"/name, local_fields)
            self.split[number] = ledger
            expected = [{key: "" if row.get(key) is None else str(row.get(key, "")) for key in local_fields}
                        for row in self.local_rows(master_rows, number)]
            # An interruption between the two fsyncs leaves the master intact.
            # Repair its per-experiment view before appending new records.
            if ledger.rows() != expected:
                write_csv(ledger.path, expected, local_fields)

    def local_rows(self, rows, number):
        if self.path.name != "results.csv":
            return [dict(r) for r in rows if int(r["experiment"]) == number]
        return [local for row in rows if (local := experiment_record(row, number)) is not None]

    def append(self, row):
        self.master.append(row)
        for number, ledger in self.split.items():
            for local in self.local_rows([row], number):
                ledger.append(local)

    def rows(self):
        return self.master.rows()


def save_experiment_plan(out, jobs):
    fields = ["experiment", "primary", "variant", "side_bits", "beam", "rerank_depth", "repeat"]
    write_csv(Path(out)/"experiment-plan.csv", jobs, fields)
    for number in range(1, 6):
        local = [record for job in jobs if (record := experiment_record(job, number)) is not None]
        write_csv(Path(out)/f"experiment{number}"/"experiment-plan.csv", local,
                  fields + (["measurement_experiment"] if number == 4 else []))


class NativeTimingSession:
    """One loaded index per adaptive curve; stdin waits stay outside QPS."""
    def __init__(self, runner, curve):
        self.runner, self.curve = runner, curve
        self.process = None; self.stream = None; self.failure = None
        label = "-".join(str(curve[k]) for k in ("experiment", "primary", "variant", "side_bits", "mode"))
        self.log = runner.out/"logs"/f"adaptive-{label}-{time.time_ns()}.log"

    def __enter__(self): return self

    def start(self, branch, prefix):
        r = self.runner; c = r.c; budget = c["memory_budget_bytes"]
        cmd = [Path(c["build_dir"])/"roar_primary_ji_search", *r.p.native_args(branch,self.curve["variant"]),
            "--graph",branch["graph"],"--queries",r.out/"data/test.fbin","--query-limit",c["test_query_limit"],
            "--topk",10,"--beam",10,"--threads",c["search_threads"],"--warmup-queries",min(100,c["test_query_limit"]),
            "--out-prefix",prefix,"--stream-plan","--packed-graph","--memory-budget-bytes",budget]
        if self.curve["mode"] == "rerank":
            cmd += ["--rerank-candidates",10,"--base",r.p.exact_vector_path(),"--rerank-storage",c["rerank_storage"]]
        if self.curve["variant"] not in ("primary","fp32") and c["lazy_ji"]: cmd += ["--lazy-ji"]
        env = dict(r.p.env)
        for name in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"): env[name] = str(c["search_threads"])
        self.log.parent.mkdir(parents=True,exist_ok=True); self.stream = self.log.open("w")
        self.process = subprocess.Popen(list(map(str,cmd)),stdin=subprocess.PIPE,stdout=self.stream,
            stderr=subprocess.STDOUT,text=True,env=env,preexec_fn=process_setup(c["cpus"],budget))
        csv.writer(self.process.stdin,lineterminator="\n").writerow(["beam","rerank_candidates","out_prefix"])
        self.process.stdin.flush()

    def measure(self, prepared):
        branch, prefix, fields = prepared
        if self.failure:
            self.runner.failed(fields,self.failure); return
        if self.process is None: self.start(branch,prefix)
        try:
            csv.writer(self.process.stdin,lineterminator="\n").writerow([fields["beam"],fields["rerank_depth"],str(prefix)])
            self.process.stdin.flush()
        except BrokenPipeError:
            pass  # The common failure path records the native diagnostic.
        started = last = time.monotonic()
        while True:
            if self.runner.ingest(prefix,fields): return
            if Path(str(prefix)+".complete").exists():
                self.failure = (-1,"Native stream completed but CSV/IDs failed protocol validation")
                self.runner.failed(fields,self.failure); return
            if self.process.poll() is not None:
                if self.runner.ingest(prefix,fields): return
                tail = self.log.read_text(errors="replace")[-1500:]
                self.failure = (self.process.returncode,tail)
                self.runner.failed(fields,self.failure); return
            if time.monotonic()-last >= 30:
                print(f"[adaptive running] {self.curve['primary']}/{self.curve['variant']} "
                      f"beam={fields['beam']} L={fields['rerank_depth']} {time.monotonic()-started:.0f}s",flush=True)
                last = time.monotonic()
            time.sleep(.2)

    def __exit__(self, kind, value, traceback):
        if self.process is not None:
            if self.process.stdin:
                try: self.process.stdin.close()
                except BrokenPipeError: pass
            if kind is not None and self.process.poll() is None: self.process.terminate()
            try: self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.process.terminate(); self.process.wait(timeout=10)
        if self.stream is not None: self.stream.close()


class Experiments:
    def __init__(self, pipeline):
        self.p = pipeline; self.c = pipeline.c; self.out = pipeline.out
        self.ledger = ExperimentCsvLedger(self.out)
        self.truth = matrix(self.out/"data/truth.ids.ibin", "<u4")
        self._index_results()

    def _index_results(self):
        self.points = {}; self.measurements = {}
        for row in self.ledger.rows(): self._remember(row)

    def _remember(self, row):
        self.points[row["point_id"]] = row
        if row.get("status") == "OK" and row.get("measurement_id"):
            self.measurements.setdefault(row["measurement_id"],row)

    def prepared(self, job, skip_failures=False):
        if not hasattr(self,"points"): self._index_results()
        branch,prefix,fields = self.descriptor(job)
        previous = self.points.get(fields["point_id"])
        if previous and (previous["status"] == "OK" or skip_failures): return None
        source = self.measurements.get(fields["measurement_id"])
        if source:
            record = dict(source,**fields)
            record.update(directory=source["directory"],ids_id=source.get("ids_id",""),status="OK",
                measurement_point_id=source.get("measurement_point_id") or source["point_id"],measurement_reused=True)
            self.ledger.append(record); self._remember(record); return None
        if self.ingest(prefix,fields): return None
        return branch,prefix,fields

    def failed(self, fields, failure):
        code,tail = failure
        status = "MEMORY_EXCEEDED" if any(x in tail for x in ("memory budget admission","bad_alloc","Cannot allocate memory")) else "FAILED"
        record = dict(fields,status=status,error=f"exit={code}; {tail}",queries=self.c["test_query_limit"])
        self.ledger.append(record); self._remember(record)

    def descriptor(self, job):
        branch = self.p.branch(job["primary"], job["variant"], job["side_bits"])
        files = [branch["model"], branch["codes"], branch["graph"], self.out/"data/test.fbin",
                 self.out/"data/truth.ids.ibin", Path(self.c["build_dir"])/"roar_primary_ji_search"]
        if job["variant"] not in ("primary", "fp32"): files += [branch["ji_model"], branch["ji_codes"]]
        exact_path=self.p.exact_vector_path() if job["primary"]=="full-precision" or job["rerank_depth"] else None
        if exact_path: files.append(exact_path)
        fid = {str(p):file_id(p) for p in files}
        pid = identity(dict(config_id=self.p.config_id, source_id=self.p.source_id, job=job, files=fid))
        measurement_id = identity(dict(config_id=self.p.config_id,source_id=self.p.source_id,
            job={k:v for k,v in job.items() if k != "experiment"},files=fid))
        prefix = self.out/f"experiment{job['experiment']}"/"raw"/job["primary"]/f"{job['variant']}-b{job['side_bits']}"/pid
        prefix.parent.mkdir(parents=True, exist_ok=True)
        fields = dict(job, point_id=pid, config_id=self.p.config_id, dataset=self.c["dataset"], base_rows=self.c["base_rows"],
            base_dim=self.c["base_dim"], implementation=branch["implementation"], candidate_size=job["beam"],
            rerank_policy="exact FP32 over top-L retained candidates" if job["rerank_depth"] else "none",
            threads=self.c["search_threads"], cpus=json.dumps(self.c["cpus"]), seed=self.c["seed"], query_split="test",
            memory_budget_bytes=self.c["memory_budget_bytes"],
            fp32_read_policy=self.c["fp32_storage"], fp32_store_id=file_id(exact_path) if exact_path else "",
            memory_accounting="owned graph capacity + full serialized index payloads + recorded owned buffers; peak RSS and hard address-space cap checked separately",
            representation_id=identity({p:v for p,v in fid.items() if p in (branch["model"], branch["codes"], branch["ji_model"], branch["ji_codes"])}),
            graph_id=file_id(branch["graph"]), directory=str(prefix.parent), error="",
            measurement_id=measurement_id,measurement_point_id=pid,measurement_reused=False)
        return branch, prefix, fields

    def ingest(self, prefix, fields):
        path = Path(str(prefix)+".metrics.csv"); ids_path = Path(str(prefix)+".ids.ibin")
        if not path.exists() or not ids_path.exists(): return False
        try:
            rows = read_csv(path)
            if len(rows)!=1: return False
            metrics = rows[0]
            if int(metrics["beam"]) != fields["beam"] or int(metrics["rerank_depth"]) != fields["rerank_depth"]:
                raise ValueError("Native point differs from requested working point")
            ids = matrix(ids_path, "<u4")
            if int(metrics["queries"]) != self.c["test_query_limit"]: raise ValueError("Native test count mismatch")
            if int(metrics["memory_budget_bytes"]) != fields["memory_budget_bytes"]: raise ValueError("Native memory budget differs from common protocol")
            if metrics["fp32_storage"] != (self.c["fp32_storage"] if fields["primary"]=="full-precision" else "none"):
                raise ValueError("Native FP32 storage differs from common protocol")
            if metrics["rerank_storage"] != (self.c["rerank_storage"] if fields["rerank_depth"] else "none"):
                raise ValueError("Native rerank storage differs from common protocol")
            record = dict(fields, **{k:v for k,v in metrics.items() if k in RESULT_FIELDS and k not in ("threads", "beam", "rerank_depth", "memory_budget_bytes")})
            record.update(status="OK" if metrics["budget_pass"]=="true" else "MEMORY_EXCEEDED",
                          recall_at_10=recall_at_10(ids, self.truth), ids_id=file_id(ids_path))
            if record["status"] != "OK":
                record.update(qps="", mean_service_latency_ms="", p50_ms="", p95_ms="", p99_ms="", error="Measured memory exceeds budget; excluded from frontier")
            self.ledger.append(record)
            self._remember(record)
        except (OSError, ValueError, KeyError) as e:
            # A file being written is retried after the process exits.  Atomic
            # IDs and complete two-line metrics are required for ingestion.
            return False
        return True

    def timings(self, jobs):
        groups = {}; scheduled = set(); deferred = []
        for job in jobs:
            prepared = self.prepared(job)
            if prepared is None: continue
            branch,prefix,fields = prepared
            if fields["measurement_id"] in scheduled:
                deferred.append(job); continue
            scheduled.add(fields["measurement_id"])
            # A full FP32 mmap can exceed the limit by itself. Keep its
            # rerank batch separate so feasible no-rerank points still run.
            mode = ("rerank" if job["rerank_depth"] else "no-rerank") if self.c["rerank_storage"]=="mmap" else "combined"
            key = (job["primary"], job["variant"], job["side_bits"], mode)
            groups.setdefault(key, []).append((job, branch, prefix, fields))
        for key, points in groups.items():
            method, variant, bits, mode = key; branch = points[0][1]
            plan = self.out/"plans"/f"{method}-{variant}-b{bits}-{mode}.csv"
            write_csv(plan, [dict(beam=j["beam"], rerank_candidates=j["rerank_depth"], out_prefix=str(prefix)) for j,_,prefix,_ in points],
                      ["beam", "rerank_candidates", "out_prefix"])
            budget=self.c["memory_budget_bytes"]
            cmd = [Path(self.c["build_dir"])/"roar_primary_ji_search", *self.p.native_args(branch, variant),
                "--graph", branch["graph"], "--queries", self.out/"data/test.fbin", "--query-limit", self.c["test_query_limit"],
                "--topk", 10, "--beam", 10, "--threads", self.c["search_threads"], "--warmup-queries", min(100,self.c["test_query_limit"]),
                "--out-prefix", points[0][2], "--batch-plan", plan, "--packed-graph", "--memory-budget-bytes", budget]
            if any(j["rerank_depth"] for j,_,_,_ in points):
                cmd += ["--base", self.p.exact_vector_path(), "--rerank-storage", self.c["rerank_storage"]]
            if variant not in ("primary", "fp32") and self.c["lazy_ji"]: cmd += ["--lazy-ji"]
            log=self.out/"logs"/f"timing-{method}-{variant}-b{bits}-{mode}.log"
            env=dict(self.p.env)
            for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"): env[name]=str(self.c["search_threads"])
            print(f"[timing] {key}, {len(points)} pending points, graph loaded once", flush=True)
            started=time.monotonic(); pending={str(prefix):(prefix,fields) for _,_,prefix,fields in points}
            with log.open("w") as stream:
                process=subprocess.Popen(list(map(str,cmd)), stdout=stream, stderr=subprocess.STDOUT, env=env,
                    preexec_fn=process_setup(self.c["cpus"],budget))
                while process.poll() is None:
                    try: process.wait(timeout=5)
                    except subprocess.TimeoutExpired: pass
                    for token,(prefix,fields) in list(pending.items()):
                        if self.ingest(prefix,fields): del pending[token]
                    if time.monotonic()-started>=30:
                        print(f"[timing] {key} remaining={len(pending)}",flush=True); started=time.monotonic()
            for prefix,fields in pending.values():
                if self.ingest(prefix,fields): continue
                tail=log.read_text(errors="replace")[-1500:]
                self.failed(fields,(process.returncode,tail))
            print(f"[timing done] {key} exit={process.returncode}, CSV={self.ledger.path}",flush=True)
        if deferred:
            # Originals now have durable records; identical cross-experiment
            # points reuse them. Failed originals may be retried by dense mode.
            self.timings(deferred)

    def adaptive_timings(self, experiments):
        c = self.c; curves = curve_specs(c,experiments)
        plan = CsvLedger(self.out/"adaptive-plan.csv",JOB_FIELDS+["phase","round"])
        planned = {tuple(str(r[k]) for k in JOB_FIELDS) for r in plan.rows()}
        coverage, gaps = {}, {}
        def snapshot():
            summaries = list(coverage.values()); details = [r for values in gaps.values() for r in values]
            write_csv(self.out/"adaptive-coverage.csv",summaries,COVERAGE_FIELDS)
            write_csv(self.out/"adaptive-gaps.csv",details,GAP_FIELDS)
            for number in (1,4,5):
                selected = [dict(r,experiment=number) for r in summaries if
                    int(r["experiment"]) == number or number == 4 and int(r["experiment"]) == 1 and r["mode"] == "no-rerank"]
                selected_gaps = [dict(r,experiment=number) for r in details if
                    int(r["experiment"]) == number or number == 4 and int(r["experiment"]) == 1 and r["mode"] == "no-rerank"]
                write_csv(self.out/f"experiment{number}"/"adaptive-coverage.csv",selected,COVERAGE_FIELDS)
                write_csv(self.out/f"experiment{number}"/"adaptive-gaps.csv",selected_gaps,GAP_FIELDS)
        for curve in curves:
            key = tuple(curve[k] for k in ("experiment","primary","variant","side_bits","mode"))
            with NativeTimingSession(self,curve) as session:
                def execute(jobs,phase,round_number):
                    for j in jobs:
                        plan_key = tuple(str(j[k]) for k in JOB_FIELDS)
                        if plan_key not in planned:
                            plan.append(dict(j,phase=phase,round=round_number)); planned.add(plan_key)
                        prepared = self.prepared(j,skip_failures=True)
                        if prepared is not None: session.measure(prepared)
                execute(initial_jobs(c,curve),"coarse",0)
                for round_number in range(1,c["adaptive_max_rounds"]+1):
                    pending,coverage[key],gaps[key] = refinement(c,curve,list(self.points.values()))
                    snapshot()
                    print(f"[adaptive] {key} round={round_number} points={coverage[key]['measured_points']} "
                          f"max Recall gap={coverage[key]['max_recall_gap']:.5f} next={len(pending)}",flush=True)
                    if not pending: break
                    execute(pending,"refine",round_number)
                execute(repeat_jobs(c,curve,list(self.points.values())),"repeat",0)
                execute(repeat_jobs(c,curve,list(self.points.values()),extra=True),"extra-repeat",0)
            pending,coverage[key],gaps[key] = refinement(c,curve,list(self.points.values()),complete_only=True)
            if pending:
                coverage[key].update(status="UNRESOLVED",pending_points=0,reason="ROUND_LIMIT_OR_INCOMPLETE_REPEATS")
                for gap in gaps[key]:
                    if gap["status"] == "PENDING": gap.update(status="UNRESOLVED",reason="ROUND_LIMIT_OR_INCOMPLETE_REPEATS")
            snapshot()
        fixed = [j for j in experiment_jobs(c,experiments) if j["experiment"] == 2]
        self.timings(fixed)
        extra_fixed = []
        fixed_curves = {(j["primary"],j["variant"],j["side_bits"],bool(j["rerank_depth"])) for j in fixed}
        for method,variant,bits,rerank in fixed_curves:
            spec = dict(experiment=2,primary=method,variant=variant,side_bits=bits,mode="rerank" if rerank else "no-rerank")
            extra_fixed.extend(repeat_jobs(c,spec,list(self.points.values()),extra=True))
        self.timings(extra_fixed)
        realized = [{k:r[k] for k in JOB_FIELDS} for r in plan.rows()] + fixed + extra_fixed
        save_experiment_plan(self.out,realized)

    def inversions(self):
        c=self.c; dest=self.out/"experiment3"/"raw"; dest.mkdir(parents=True,exist_ok=True)
        teacher=self.p.branch("full-precision","fp32",0); prefix=dest/"teacher"; pairs_path=dest/"routing-pairs.csv"
        cmd=[Path(c["build_dir"])/"roar_primary_ji_search", *self.p.native_args(teacher,"fp32"),
            "--graph",teacher["graph"],"--queries",self.out/"data/test.fbin","--query-limit",c["test_query_limit"],
            "--beam",c["routing_beam"],"--threads",c["search_threads"],"--out-prefix",prefix,"--packed-graph",
            "--routing-pairs-csv",pairs_path,"--routing-pairs-per-source",c["routing_pairs_per_source"],"--pdf-routing-tags"]
        self.p.stage("routing-teacher",cmd,[pairs_path,Path(str(prefix)+".metrics.csv"),Path(str(prefix)+".ids.ibin")],
                     [teacher["graph"],c["base"],self.out/"data/test.fbin"],budget=0)
        pairs=read_csv(pairs_path); per_query=[[] for _ in range(c["test_query_limit"])]
        for pair in pairs: per_query[int(pair["query_id"])].extend([int(pair["u_id"]),int(pair["v_id"])])
        depth=truth_depth(c)
        ordered=[list(map(int,row[:depth]))+sorted(set(extra)-set(map(int,row[:depth]))) for row,extra in zip(self.truth,per_query)]
        width=max(map(len,ordered)); ids=np.empty((len(ordered),width),dtype=np.uint32); lookups=[]
        for q,row in enumerate(ordered):
            ids[q]=row+[row[0]]*(width-len(row)); lookups.append({node:i for i,node in enumerate(row)})
        audit_ids=dest/"common.ids.ibin"
        if not audit_ids.exists() or not np.array_equal(matrix(audit_ids,"<u4"),ids):
            write_matrix(audit_ids,ids,"<u4")
        teacher_id=identity(dict(graph=file_id(teacher["graph"]),pairs=file_id(pairs_path),ids=file_id(audit_ids)))
        branches=[("full-precision","fp32",0)]
        for method in c["methods"]:
            branches += [(method,"primary",0)] + [(method,v,64) for v in sorted(set(c["ablation_variants"])|{c["deploy_variant"]})]
        exact=None; records=[]
        ledger=ExperimentCsvLedger(self.out,"inversions.csv",INVERSION_FIELDS)
        inv_key=lambda r: tuple(str(r.get(k,"")) for k in ("experiment","primary","variant","side_bits","metric","teacher_id","representation_id"))
        completed={inv_key(r) for r in ledger.rows()}
        for method,variant,bits in branches:
            branch=self.p.branch(method,variant,bits); out=dest/f"{method}-{variant}-b{bits}"
            cmd=[Path(c["build_dir"])/"roar_exp234_audit",*self.p.native_args(branch,variant),"--graph",teacher["graph"],
                "--queries",self.out/"data/test.fbin","--out-prefix",out,"--audit-ids",audit_ids]
            inputs=[branch["model"],branch["codes"],audit_ids,self.out/"data/test.fbin"]
            if bits: inputs += [branch["ji_model"],branch["ji_codes"]]
            self.p.stage(f"inversion-{method}-{variant}",cmd,[Path(str(out)+".scores.fbin")],inputs)
            scores=np.asarray(matrix(str(out)+".scores.fbin"))
            if exact is None: exact=scores
            common=dict(experiment=3,dataset=c["dataset"],primary=method,variant=variant,side_bits=bits,
                status="OK",teacher_id=teacher_id,representation_id=identity({str(p):file_id(p) for p in inputs[:2]+inputs[4:]}),
                pair_selection=f"full database FP32 Top-{depth} + one common FP32 teacher traversal",note="query-balanced; exact ties excluded; approximate ties counted as sign mismatch")
            local=global_inversions(exact[:,:depth],scores[:,:depth],c["inversion_ranks"])
            for event,metric in (("exp","ExpandInv"),("queue","QueueInv"),("stop-boundary-proxy","StopBoundaryProxyInv")):
                subset=[p for p in pairs if p["event_type"]==event]; e=[]; a=[]; qids=[]
                for pair in subset:
                    q=int(pair["query_id"]); u=lookups[q][int(pair["u_id"])]; v=lookups[q][int(pair["v_id"])]
                    e.append(exact[q,u]-exact[q,v]); a.append(scores[q,u]-scores[q,v]); qids.append(q)
                local.append(dict(metric=metric,**pair_statistics(e,a,qids,len(ids))))
            local.append(dict(metric="StopInv",status="NOT_APPLICABLE",queries=len(ids),
                              note="Search terminates on queue exhaustion; no score-based termination decision exists"))
            for experiment in ((2,3) if variant in c["ablation_variants"] else (3,)):
                for record in local:
                    row=dict(common,**record); row["experiment"]=experiment
                    if inv_key(row) not in completed:
                        ledger.append(row); completed.add(inv_key(row))
                    records.append(row)
        # Resume may append the same audit; report always uses latest record per key.
        self.report()

    def report(self):
        c=self.c; raw=self.ledger.rows(); latest={r["point_id"]:r for r in raw}; rows=list(latest.values())
        front=aggregate(rows,c["repeats"])
        write_csv(self.out/"aggregate.csv",front, list(RESULT_FIELDS)+["repeats","timing_complete","qps_cv"])
        performance=[r for r in front if int(r["experiment"])==1]
        perf_raw=[r for r in rows if int(r["experiment"])==1]
        visible=performance+[dict(r,timing_complete=False) for r in perf_raw if r["status"]!="OK"]
        write_csv(self.out/"equal-recall.csv",matched_points(visible,(.80,.85,.90,.95,.99)),
                  ["primary","variant","mode","target_recall","achieved","selection","beam","rerank_depth","recall_at_10","visited","expanded","comparisons","qps","mean_service_latency_ms","p50_ms","p95_ms","p99_ms"])
        fair=visible  # FP32 now shares the IO policy and hard memory limit.
        write_csv(self.out/"equal-recall-256GB.csv",matched_points(fair,(.80,.85,.90,.95,.99)),
                  ["primary","variant","mode","target_recall","achieved","selection","beam","rerank_depth","recall_at_10","visited","expanded","comparisons","qps","mean_service_latency_ms","p50_ms","p95_ms","p99_ms"])
        traversal=matched_points([r for r in visible if int(r["rerank_depth"])==0])
        write_csv(self.out/"traversal.csv",traversal,["primary","variant","mode","target_recall","achieved","selection","beam","rerank_depth","recall_at_10","visited","expanded","comparisons","qps","mean_service_latency_ms","p50_ms","p95_ms","p99_ms"])
        memories=[]
        for row in front:
            mem={k:row[k] for k in ("experiment","primary","variant","side_bits","beam","rerank_depth","graph_memory_bytes","primary_code_bytes",
                 "refiner_code_bytes","codebooks_transform_bytes","runtime_buffer_bytes","fp_hot_bytes","accounted_memory_bytes","peak_rss_bytes")}
            n=c["base_rows"]; per_node=(float(row["graph_memory_bytes"])+float(row["primary_code_bytes"])+float(row["refiner_code_bytes"])+float(row["fp_hot_bytes"]))/n
            fixed=float(row["accounted_memory_bytes"])-per_node*n
            mem.update(memory_budget_bytes=c["memory_budget_bytes"], capacity_at_budget=int(max(0,c["memory_budget_bytes"]-fixed)//per_node),
                capacity_interpretation="linear index-footprint estimate using measured graph degree; not a rebuilt larger graph",
                comparison_group="common FP32 IO policy and hard search-memory limit")
            memories.append(mem)
        write_csv(self.out/"memory.csv",memories,["experiment","primary","variant","side_bits","beam","rerank_depth","graph_memory_bytes","primary_code_bytes","refiner_code_bytes",
            "codebooks_transform_bytes","runtime_buffer_bytes","fp_hot_bytes","accounted_memory_bytes","peak_rss_bytes","memory_budget_bytes","capacity_at_budget","capacity_interpretation","comparison_group"])
        inv=read_csv(self.out/"inversions.csv"); unique={tuple(r[k] for k in ("experiment","primary","variant","side_bits","metric")):r for r in inv}
        write_csv(self.out/"inversions-summary.csv",unique.values(),INVERSION_FIELDS)
        curves=[]; variants=[]
        for path in sorted((self.out/"ji").glob("*/b*/distortion-curves.csv")): curves.extend(read_csv(path))
        for path in sorted((self.out/"ji").glob("*/b*/training-variants.csv")): variants.extend(read_csv(path))
        if curves: write_csv(self.out/"budget-distortion-curves.csv",curves)
        if variants: write_csv(self.out/"budget-training.csv",variants)
        write_csv(self.out/"bit-budget-frontier.csv",[r for r in front if int(r["experiment"])==5],list(RESULT_FIELDS)+["repeats","timing_complete","qps_cv"])
        budgets=[]
        for bits in c["budgets"]:
            selected=[r for r in front if int(r["experiment"])==5 and int(r["side_bits"])==bits]
            failed=[dict(r,timing_complete=False) for r in rows if int(r["experiment"])==5 and int(r["side_bits"])==bits and r["status"]!="OK"]
            budgets.extend(dict(r,side_bits=bits) for r in matched_points(selected+failed))
        write_csv(self.out/"bit-budget-equal-recall.csv",budgets,["primary","variant","mode","target_recall","achieved","selection","beam","rerank_depth","recall_at_10","visited","expanded","comparisons","qps","mean_service_latency_ms","p50_ms","p95_ms","p99_ms","side_bits"])
        self.experiment_reports(front, memories, list(unique.values()), curves, variants)

    def experiment_reports(self, front, memories, inversions, curves, variants):
        """Save directly usable CSV reports beside each experiment's raw output."""
        for number in range(1, 6):
            dest=self.out/f"experiment{number}"
            fields=list(RESULT_FIELDS)+(["measurement_experiment"] if number==4 else [])
            selected=[r for row in front if (r:=experiment_record(row,number)) is not None]
            write_csv(dest/"aggregate.csv",selected,fields+["repeats","timing_complete","qps_cv"])
            memory=[r for row in memories if (r:=experiment_record(row,number)) is not None]
            memory_fields=["experiment","primary","variant","side_bits","beam","rerank_depth","graph_memory_bytes","primary_code_bytes","refiner_code_bytes",
                "codebooks_transform_bytes","runtime_buffer_bytes","fp_hot_bytes","accounted_memory_bytes","peak_rss_bytes","memory_budget_bytes","capacity_at_budget","capacity_interpretation","comparison_group"]
            write_csv(dest/"memory.csv",memory,memory_fields+(["measurement_experiment"] if number==4 else []))
        # Frontier schemas are preserved, including unreachable/over-budget rows.
        reports={1:("equal-recall.csv","equal-recall-256GB.csv"),
                 4:("traversal.csv",),
                 5:("bit-budget-frontier.csv","bit-budget-equal-recall.csv")}
        for number,names in reports.items():
            for name in names:
                path=self.out/name
                # Copy the completed small CSV, never the encoded database.
                with path.open(newline="") as stream:
                    reader=csv.DictReader(stream); data=list(reader)
                    write_csv(self.out/f"experiment{number}"/name,data,reader.fieldnames)
        for number in (2,3):
            write_csv(self.out/f"experiment{number}"/"inversions-summary.csv",
                      [r for r in inversions if int(r["experiment"])==number],INVERSION_FIELDS)
        if curves: write_csv(self.out/"experiment5/budget-distortion-curves.csv",curves)
        if variants:
            write_csv(self.out/"experiment5/budget-training.csv",variants)
            write_csv(self.out/"experiment2/training-variants.csv",[r for r in variants if int(r["budget"])==64])

    def cases(self):
        """One actual query trace per primary/main branch, separate from timing."""
        self.report()
        records=[]; targets=read_csv(self.out/"traversal.csv")
        raw={r["point_id"]:r for r in self.ledger.rows()}
        for method in self.c["methods"]:
            for variant,bits in (("primary",0),(self.c["deploy_variant"],64)):
                candidates=[r for r in targets if r["primary"]==method and r["variant"]==variant and r["achieved"]=="True"]
                selected=next((r for r in candidates if float(r["target_recall"])==.95), candidates[0] if candidates else None)
                if selected is None:
                    okay=[r for r in raw.values() if r["primary"]==method and r["variant"]==variant and r["status"]=="OK" and int(r["rerank_depth"])==0]
                    if not okay: continue
                    selected=max(okay,key=lambda r:float(r["recall_at_10"]))
                beam=int(selected["beam"]); branch=self.p.branch(method,variant,bits)
                dest=self.out/"experiment4"/"raw"/"cases"/f"{method}-{variant}"; dest.mkdir(parents=True,exist_ok=True); prefix=dest/"q0"
                paths={s:dest/(s+".csr") for s in ("visited","retained","expanded")}
                cmd=[Path(self.c["build_dir"])/"roar_primary_ji_search",*self.p.native_args(branch,variant),
                    "--graph",branch["graph"],"--queries",self.out/"data/test.fbin","--query-limit",1,
                    "--beam",beam,"--threads",1,"--out-prefix",prefix,"--packed-graph",
                    "--memory-budget-bytes",self.c["memory_budget_bytes"]]
                if variant!="primary" and self.c["lazy_ji"]: cmd += ["--lazy-ji"]
                for stage,path in paths.items(): cmd += ["--"+stage+"-csr",path]
                deps=[branch["model"],branch["codes"],branch["graph"],self.out/"data/test.fbin"]
                if bits: deps += [branch["ji_model"],branch["ji_codes"]]
                self.p.stage(f"case-{method}-{variant}-beam{beam}",cmd,list(paths.values()),deps,budget=self.c["memory_budget_bytes"])
                returned=set(map(int,matrix(str(prefix)+".ids.ibin","<u4")[0])); true=set(map(int,self.truth[0,:10]))
                base=matrix(self.c["base"]); query=np.asarray(matrix(self.out/"data/test.fbin")[0])
                for stage,path in paths.items():
                    with path.open("rb") as f:
                        magic,version,start,nq,count=struct.unpack("<8sIQIQ",f.read(32))
                        if magic!=b"DRBVST01" or version!=1 or nq!=1 or start!=0: raise ValueError("Invalid case trace")
                        offsets=np.fromfile(f,dtype="<u8",count=2)
                        ids=np.fromfile(f,dtype="<u4",count=count)
                        if offsets[0]!=0 or offsets[1]!=len(ids): raise ValueError("Truncated case trace")
                    for ordinal,node in enumerate(ids):
                        records.append(dict(query_id=0,primary=method,variant=variant,beam=beam,stage=stage,ordinal=ordinal,
                            node_id=int(node),exact_ip=float(np.asarray(base[int(node)])@query),true_top10=int(node) in true,
                            returned_top10=int(node) in returned,working_point_recall=selected["recall_at_10"],
                            timing_usage="diagnostic trace; excluded from QPS aggregates"))
        write_csv(self.out/"traversal-case.csv",records,["query_id","primary","variant","beam","stage","ordinal","node_id","exact_ip","true_top10","returned_top10","working_point_recall","timing_usage"])
        write_csv(self.out/"experiment4/traversal-case.csv",records,["query_id","primary","variant","beam","stage","ordinal","node_id","exact_ip","true_top10","returned_top10","working_point_recall","timing_usage"])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command",choices=("plan","preflight","prepare","run","all","report"))
    parser.add_argument("--config")
    parser.add_argument("--experiments",default="1,2,3,4,5",help="PDF numbering; 4 also measures experiment-1 traversal grid")
    args=parser.parse_args(); c=load_config(args.config); experiments={int(x) for x in args.experiments.split(",")}
    if not experiments or experiments-set(range(1,6)): parser.error("experiments must be a subset of 1,2,3,4,5")
    if args.command=="plan":
        out=Path(c["output_root"]); save_experiment_plan(out,experiment_jobs(c,experiments))
        definitions=[dict(experiment=n,name=name,definition=definition) for n,name,definition in (
            (1,"Performance", f"{c['performance_sampling']} Recall@10/QPS and service P50/P95/P99; adaptive range={c['recall_range']} gap<={c['recall_gap']}; Q and Q+JI share each primary-only graph; common FP32 IO policy and hard memory limit"),
            (2,"Ablation", "B=64 and fixed beam/candidate/rerank; four JI variants share the corresponding Q graph; atomic-equal versus DP isolates allocation"),
            (3,"Inversion", f"full-database Top-{truth_depth(c)}; R={c['inversion_ranks']} and h5/h10; common teacher pairs; true StopInv N/A"),
            (4,"Traversal", "measured Recall .90/.95/.99; beam/visited/expanded; unreachable recorded without extrapolation"),
            (5,"Bit budget", "B16/32/48/64/96/128; independently fit D_G(b), DP and grouping; all budgets share the corresponding Q graph"))]
        write_csv(out/"protocol.csv",definitions)
        for row in definitions: write_csv(out/f"experiment{row['experiment']}"/"protocol.csv",[row])
        label = "initial probes/fixed jobs; refinements and timing repeats depend on measured Recall" if c["performance_sampling"] == "adaptive" else "fixed timing jobs"
        print(f"Plan: {out/'experiment-plan.csv'}; {len(experiment_jobs(c,experiments))} {label}; no benchmarks executed"); return
    pipeline=Pipeline(c)
    if args.command=="preflight":
        okay=pipeline.preflight(); print(f"Preflight {'OK' if okay else 'BLOCKED'}: {pipeline.out/'preflight.csv'}"); raise SystemExit(0 if okay else 2)
    if args.command in ("prepare","all"): pipeline.prepare()
    if args.command in ("run","all","report"):
        runner=Experiments(pipeline)
        if args.command!="report":
            if c["performance_sampling"] == "adaptive": runner.adaptive_timings(experiments)
            else: runner.timings(experiment_jobs(c,experiments))
            if 3 in experiments or 2 in experiments: runner.inversions()
            if 4 in experiments: runner.cases()
        runner.report(); print(f"CSV results: {pipeline.out}")


if __name__=="__main__": main()
