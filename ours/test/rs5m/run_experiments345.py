#!/usr/bin/env python3
"""RS5M experiments 3/4/5: accuracy diagnostics only, one pass, no QPS scan.

Freeze the completed 160-bit experiment-1 graph/primary/model/query artifacts.
Reuse its matched-Recall observations; train only the other relative side rates.
Native executables may emit timer fields, but these are not performance results.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'ours/test/t2i-10M'), str(ROOT/'ours/scripts'),
               str(ROOT/'ours/python'), str(ROOT/'ours/test/t2i-1M/python')]

from protocol import file_id, identity, matrix, read_csv, write_csv
from pipeline import Pipeline
from run_native100_experiment1 import check_refiner, link_once
from run_frozen_experiment5 import (audit_quality, evaluation_nodes, rate_plot,
                                    save_json)
from scoring_contract import require_traversal, validate_native_summary

METHODS = ['opq-pq', 'qaq', 'leanvec-ood', 'extended-rabitq', 'saq']
DEFAULT_SOURCE = 'outputs/rs5m/remora-traversal-exp1-nvme-b160-v1'
DEFAULT_OUTPUT = 'outputs/rs5m/remora-exp345-relative-b80-320-v2'


def relative_budgets(default_bits):
    values = [default_bits//2, default_bits, default_bits*3//2, default_bits*2]
    if any(bits % 8 or bits <= 0 or bits > 512 for bits in values):
        raise ValueError('Relative rates must be legal whole-byte learned groups')
    return values


def frozen_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Protocol changed; use a new output root: {path}')
    else:
        save_json(path, value)


def threshold_rows(rows, methods, depth):
    """Existing accuracy/visits at FIXED L; do not select by QPS."""
    result = []
    for method in methods:
        for variant in ('primary', 'remora'):
            row = dict(primary=method, variant=variant, rerank_depth=depth,
                       selection='minimum observed visited nodes at fixed L and Recall >= target; no interpolation',
                       source='completed experiment-1 observations; no new benchmark')
            local = [r for r in rows if r['primary']==method and r['variant']==variant
                     and int(r['rerank_depth'])==depth and r['status']=='OK']
            for target, label in ((.90, '90'), (.95, '95'), (.99, '99')):
                feasible = [r for r in local if float(r['recall_at_10'])>=target]
                best = min(feasible, key=lambda r:(float(r['visited']), int(r['beam']))) if feasible else None
                row['achieved_'+label] = bool(best)
                row['beam_'+label] = int(best['beam']) if best else 'NOT_REACHED'
                row['visited_'+label] = float(best['visited']) if best else ''
                row['actual_recall_'+label] = float(best['recall_at_10']) if best else ''
            result.append(row)
    return result


def recall_record(native, truth, returned, method, bits, beam):
    variant = 'remora' if bits else 'primary'
    validate_native_summary(native, variant)
    if (returned.shape != (len(truth), 10) or native['queries']!=len(truth)
            or native['beam_width']!=beam or native['rerank_enabled']
            or native.get('base_vectors_loaded') is not False
            or (bits and native.get('ji_active_groups')!=bits//8)):
        raise ValueError('Recall-only search shape/scoring/rerank contract failed')
    if any(len(np.unique(ids))!=10 for ids in returned):
        raise ValueError('Duplicate returned Top-10 IDs')
    hits = sum(int(np.isin(answer, gt[:10]).sum()) for gt, answer in zip(truth, returned))
    return dict(primary=method, variant=variant, side_bits=bits, beam=beam,
                recall_at_10=hits/(10*len(truth)), queries=len(truth), passes=1,
                visited=native['mean_visited_nodes'], expanded=native['mean_expansions'],
                rerank_depth=0, scoring_placement='traversal' if bits else 'primary',
                qps_measured=False, timing_usage='discarded; correctness/Recall only')


def quality_table(methods, budgets, beams, quality, recall):
    qindex = {(r['primary'], int(r['side_bits'])):r for r in quality}
    rindex = {(r['primary'], int(r['side_bits']), int(r['beam'])):r for r in recall}
    tables = []
    for method in methods:
        for beam in beams:
            for bits in [0, *budgets]:
                q = qindex.get((method,bits)); r = rindex.get((method,bits,beam))
                row = dict(primary=method, beam=beam, side_bits=bits,
                           status='COMPLETE' if q and r else 'PENDING',
                           scoring_placement='traversal' if bits else 'primary', rerank_depth=0)
                if q:
                    row.update(DW=float(q['dw']), score_rmse=float(q['score_rmse_top200']),
                        routing_inv_percent=100*float(q['routing_inv']),
                        expand_inv_percent=100*float(q['expand_inv']),
                        queue_inv_percent=100*float(q['queue_inv']),
                        boundary_inv_k10_h5_percent=100*float(q['boundary_inv_h5']),
                        boundary_inv_k10_h10_percent=100*float(q['boundary_inv_h10']),
                        side_payload_MB=float(q['side_payload_bytes'])/1e6,
                        refiner_model_MB=float(q['refiner_model_bytes'])/1e6,
                        hot_index_GB=float(q['hot_index_bytes'])/1e9)
                row.update(stop_inv='N/A: queue exhaustion, not score-based termination',
                           recall_at_10_percent=100*float(r['recall_at_10']) if r else '',
                           visited=float(r['visited']) if r else '')
                tables.append(row)
    return tables


def frozen_sources(repo, source_root, queries):
    source = source_root/'performance'; prepared=source_root/'prepared'
    c = dict(json.loads((source/'configuration.json').read_text())['config'])
    require_traversal(c)
    summary = json.loads((source/'experiment1/summary.json').read_text())
    if (c['dataset']!='RS5M-512' or c['base_dim']!=512 or c['base_rows']!=4828191
            or c['primary_budget_bits']!=2048 or c['side_bits']!=160
            or c['test_query_limit']!=queries or summary['status']!='COMPLETE'
            or set(c['methods'])!=set(METHODS)):
        raise ValueError('Expected complete immutable RS5M experiment 1, 160 bits, identical test query count')
    for method in METHODS:
        check_refiner(source, method, c['base_rows'], c['side_bits'])
        for split in ('metric', 'select'):
            if not (prepared/'pools'/method/split/'pools.npz').is_file():
                raise ValueError('Missing frozen training/validation pools: '+method+'/'+split)
    for directory in (Path(c['build_dir']), repo/'ours/build-relative-rate-v1'):
        for binary in ('roar_primary_ji_search', 'roar_exp234_audit'):
            if not (directory/binary).is_file():
                raise ValueError('Missing precompiled binary: '+str(directory/binary))
    if matrix(source/'data/truth.ids.ibin','<u4').shape!=(queries,100):
        raise ValueError('Expected the frozen official full-database Top-100 truth')
    return c, source, prepared


def reference_view(c, source, out, gpu, status):
    """Diagnostic Top-200 overlay; old experiment-1 measurements stay read-only."""
    status('diagnostic-ground-truth-top200')
    reference=out/'reference'
    subprocess.run([sys.executable,'-u',ROOT/'ours/test/rs5m/extend_ground_truth.py',
                    '--source',source,'--output',reference/'data','--gpu',str(gpu)],check=True)
    for name in ('test.fbin',):
        link_once(source/'data'/name,reference/'data'/name)
    link_once(source/'ji',reference/'ji')
    link_once(source/'experiment1',reference/'experiment1')
    if (source/'data/base.cold').exists():
        link_once(source/'data/base.cold',reference/'data/base.cold')
    frozen=json.loads((source/'configuration.json').read_text())
    config=dict(c,output_root=str(reference))
    frozen_json(reference/'configuration.json',dict(config=config,config_id=identity(config),
        source_id=frozen['source_id'],diagnostic_view=True,original_performance_source=str(source),
        ground_truth='full-database GPU Top-200; original Top-100 unchanged',
        measurements='read-only links to completed experiment 1; NOT a new performance scan'))
    return reference


def run_rate(repo, c, source, prepared, audit, out, args, status):
    audit_meta = json.loads((audit/'study.json').read_text())
    if (audit_meta['source']!=str(source) or audit_meta['queries']!=args.queries
            or audit_meta['side_bits']!=160
            or json.loads((audit/'summary.json').read_text())['status']!='COMPLETE'):
        raise ValueError('Experiment 5 requires completed experiment 3/4 on the same frozen source')
    budgets=relative_budgets(c['side_bits']); beams=[100,200]
    trainer=repo/'ours/experimental/relative-rate-v1/train_remora.py'
    rc=dict(c,output_root=str(out),budgets=budgets,bit_budget_beams=beams,beams=beams,
            repeats=1,build_dir=str(repo/'ours/build-relative-rate-v1'),
            device=f'cuda:{args.gpu}',kmeans_device=f'cuda:{args.gpu}',
            geometric_encoder_device=f'cuda:{args.gpu}',
            fp_graph=audit_meta['teacher_graph'],ablation_variants=['remora'])
    out.mkdir(parents=True,exist_ok=True)
    for name in ('test.fbin','truth.ids.ibin','truth.scores.fbin','truth.metadata.json'):
        link_once(source/'data'/name,out/'data'/name)
    for name in ('metric.fbin','select.fbin'):
        link_once(prepared/'data'/name,out/'data'/name)
    for method in METHODS:
        link_once(prepared/'pools'/method,out/'pools'/method)
        link_once(source/'ji'/method/'b160/remora',out/'ji'/method/'b160/remora')

    class RatePipeline(Pipeline):
        def ji(self,method,bits,pools):
            config=self.out/'trainer-config.json'
            frozen_json(config,self.c)
            dest=self.out/'ji'/method/f'b{bits}/remora'; primary=self.runs[method]['paths']
            self.stage(f'remora-{method}-b{bits}-remora',
                [sys.executable,trainer,'--config',config,'--method',method,
                 '--model',primary['model'],'--codes',primary['codes'],
                 '--train-pool',pools['metric'],'--validation-pool',pools['select'],
                 '--bits',bits,'--variant','remora','--output',dest],
                [dest/'online.bin',dest/'side.bin',dest/'model.npz',dest/'training.json',dest/'encoding-parity.json'],
                [trainer,config,primary['model'],primary['codes'],pools['metric'],pools['select']])

    p=RatePipeline(rc)
    ids=evaluation_nodes(rc['base_rows'],rc['residual_sample_size'],args.distortion_nodes,rc['seed'])
    study=dict(version=1,source=str(source),prepared=str(prepared),audit_source=str(audit),
               config=rc,distortion_nodes=args.distortion_nodes,
               distortion_sample_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
               primary_and_graph_frozen=True,reference_budget=160,
               qps_measured=False,passes_per_accuracy_point=1,
               teacher_graph_role=audit_meta['teacher_graph_role'],
               common_teacher_ids=file_id(audit/'experiment3/raw/common.ids.ibin'),
               trainer=str(trainer),trainer_id=file_id(trainer),
               binary_ids={n:file_id(Path(rc['build_dir'])/n) for n in ('roar_primary_ji_search','roar_exp234_audit')},
               runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    frozen_json(out/'study.json',study)
    if not p.preflight():
        raise RuntimeError('Rate-study preflight failed; inspect '+str(out/'preflight.csv'))
    write_csv(out/'experiment5/distortion-sample.ids.csv',[dict(node_id=int(i)) for i in ids])
    done=0; total=len(METHODS)*len(budgets); training_started=time.monotonic()
    for method in METHODS:
        for bits in budgets:
            status('experiment5-training',primary=method,side_bits=bits,models_done=done,models_total=total)
            try:
                check_refiner(out,method,rc['base_rows'],bits)
                print(f'[reuse] validated model {method}/b{bits}',flush=True)
            except (OSError,ValueError,KeyError):
                p.ji(method,bits,{s:out/'pools'/method/s/'pools.npz' for s in ('metric','select')})
                check_refiner(out,method,rc['base_rows'],bits)
            done+=1; elapsed=time.monotonic()-training_started
            print(f'[progress] models={done}/{total}; elapsed/estimated_training_total='
                  f'{elapsed:.1f}s/{elapsed*total/done:.1f}s (estimate; reference models reused)',flush=True)

    status('experiment5-quality')
    q=matrix(out/'data/test.fbin'); q64=np.asarray(q,dtype=np.float64)
    w=q64.T@q64/len(q64); del q64
    quality_args=SimpleNamespace(audit_source=str(audit),budgets=budgets,queries=args.queries,reference_budget=160)
    quality=audit_quality(p,quality_args,p.runs,ids,w,(matrix,read_csv,write_csv,file_id,identity))

    status('experiment5-recall-only')
    truth=matrix(out/'data/truth.ids.ibin','<u4')
    source_rows=read_csv(source/'experiment1/aggregate.csv')
    recall=[]
    expected=len(METHODS)*(len(budgets)+1)*len(beams)
    for method in METHODS:
        for bits in [0,*budgets]:
            variant='remora' if bits else 'primary'; branch=p.branch(method,variant,bits)
            for beam in beams:
                dest=out/'experiment5/recall'/method/f'b{bits}/beam{beam}'
                deps=[branch['graph'],branch['model'],branch['codes'],out/'data/test.fbin']
                if bits: deps += [branch['ji_model'],branch['ji_codes']]
                stamp=identity(dict(study=study,method=method,bits=bits,beam=beam,
                                    files={str(v):file_id(v) for v in deps}))
                cache=dest/'recall.json'
                if cache.exists() and json.loads(cache.read_text()).get('fingerprint')==stamp:
                    row=json.loads(cache.read_text())['record']
                else:
                    old=next((r for r in source_rows if r['primary']==method and r['variant']==variant
                              and int(r['side_bits'])==bits and int(r['beam'])==beam
                              and int(r['rerank_depth'])==0 and r['status']=='OK'),None)
                    if bits in (0,160) and old:
                        prefix=Path(old['directory'])/(old.get('measurement_point_id') or old['point_id'])
                        reuse=True
                    else:
                        prefix=dest/'search'; reuse=False
                        command=[Path(rc['build_dir'])/'roar_primary_ji_search',*p.native_args(branch,variant),
                                 '--graph',branch['graph'],'--queries',out/'data/test.fbin',
                                 '--query-limit',args.queries,'--topk',10,'--beam',beam,
                                 '--threads',rc['search_threads'],'--warmup-queries',0,
                                 '--packed-graph','--out-prefix',prefix,
                                 '--memory-budget-bytes',rc['memory_budget_bytes']]
                        p.stage(f'recall-only-{method}-b{bits}-beam{beam}',command,
                                [Path(str(prefix)+'.ids.ibin'),Path(str(prefix)+'.summary.json')],deps,
                                budget=rc['memory_budget_bytes'])
                    native=json.loads(Path(str(prefix)+'.summary.json').read_text())
                    row=recall_record(native,truth,matrix(str(prefix)+'.ids.ibin','<u4'),method,bits,beam)
                    row.update(source_reused=reuse,native_result_prefix=str(prefix))
                    save_json(cache,dict(fingerprint=stamp,record=row))
                recall.append(row)
                write_csv(out/'experiment5/recall.csv',recall)
                print(f'[recall-only] {len(recall)}/{expected} {method}/b{bits}/beam{beam}: '
                      f'Recall={100*float(row["recall_at_10"]):.3f}%; reused={row["source_reused"]}; no QPS',flush=True)

    tables=quality_table(METHODS,budgets,beams,quality,recall)
    report=out/'experiment5/report';report.mkdir(parents=True,exist_ok=True)
    write_csv(report/'requested-tables.csv',tables);write_csv(report/'quality.csv',quality)
    for method in METHODS:
        for beam in beams:
            write_csv(report/f'{method}-beam{beam}-table.csv',
                      [r for r in tables if r['primary']==method and r['beam']==beam])
    for metric,label in [('DW','Query-visible distortion'),('score_rmse','Score RMSE, exact Top-200'),
                         ('routing_inv_percent','Routing inversion (%)'),
                         ('boundary_inv_k10_h5_percent','Boundary inversion @10, h=5 (%)'),
                         ('recall_at_10_percent','Recall@10 (%)')]:
        for beam in beams:
            rate_plot(report/f'{metric}-bits-beam{beam}.svg',[r for r in tables if r['beam']==beam],
                      metric,f'RS5M / REMORA traversal / beam={beam} / no FP rerank',label)
    complete=(len(quality)==len(METHODS)*(len(budgets)+1) and len(recall)==expected
              and all(r['status']=='COMPLETE' for r in tables))
    result=dict(status='COMPLETE' if complete else 'INCOMPLETE',experiment=5,dataset='RS5M-512',
                methods=METHODS,side_bits=budgets,beams=beams,queries=args.queries,
                reference_budget=160,reference_models_reused=True,
                primary_nominal_bits=2048,side_budget_fraction={b:b/2048 for b in budgets},
                passes_per_accuracy_point=1,qps_measured=False,performance_repeats=0,
                scoring_placement='traversal',rerank_depth=0,allocator='balanced-product; NOT historical DP',
                quality_rows=len(quality),recall_points=len(recall),
                reused_recall_points=sum(bool(r['source_reused']) for r in recall),
                primary_and_graph_frozen=True,test_used_for_training=False,
                stop_inv='N/A; queue exhaustion, separately labelled proxy in experiment 3',
                requested_table=str(report/'requested-tables.csv'))
    save_json(report/'summary.json',result)
    if not complete: raise RuntimeError('Incomplete accuracy-only rate study')
    print(f'[complete] {report/"summary.json"}',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',default=DEFAULT_SOURCE)
    parser.add_argument('--output',default=DEFAULT_OUTPUT)
    parser.add_argument('--queries',type=int,default=10000)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--distortion-nodes',type=int,default=100000)
    parser.add_argument('--beams',default='50,100,200,300,500,1000')
    parser.add_argument('--phases',default='check,diagnostics,rate')
    args=parser.parse_args();phases=args.phases.split(',')
    if set(phases)-{'check','diagnostics','rate'} or args.queries<=0:
        parser.error('Unknown phase or invalid query count')
    source_root=(ROOT/args.source).resolve(strict=True);out=(ROOT/args.output).resolve()
    if (out==source_root or source_root in out.parents or out in source_root.parents
            or not out.is_relative_to(ROOT/'outputs/rs5m')):
        raise ValueError('Use a separate, non-overlapping RS5M output root')
    c,source,prepared=frozen_sources(ROOT,source_root,args.queries)
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,c['cpus'])
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):
        os.environ[name]=str(c['search_threads'])
    os.environ['REMORA_NATIVE_PRIMARY_READER']=str(ROOT/'ours/runtime/native100-reader-v1/roar_native_primary_reader')
    os.environ['REMORA_PRIMARY_DECODE_THREADS']=str(c['search_threads'])
    started=time.monotonic()
    def status(phase,**extra):
        elapsed=time.monotonic()-started
        save_json(out/'status.json',dict(status='RUNNING',phase=phase,elapsed_s=elapsed,**extra))
        print(f'[phase] {phase}; elapsed={elapsed:.1f}s',flush=True)
    status('preflight')
    frozen_json(out/'workflow.json',dict(source=str(source_root),queries=args.queries,
        experiments=[3,4,5],main_side_bits=160,rate_budgets=relative_budgets(160),
        trajectory_beams=args.beams,rate_beams=[100,200],gpu=args.gpu,
        distortion_nodes=args.distortion_nodes,scoring_placement='traversal',
        qps_measured=False,passes_per_accuracy_point=1,performance_repeats=0,
        experiment4_selection='fixed rerank L and min observed visits; reuse experiment 1',
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
    try:
        if 'diagnostics' in phases or 'rate' in phases:
            source=reference_view(c,source,out,args.gpu,status)
        if 'diagnostics' in phases:
            status('experiments3-and-4')
            subprocess.run([sys.executable,'-u',ROOT/'ours/scripts/run_frozen_experiments34.py',
                '--repo',ROOT,'--dataset','RS5M-512','--source',source,
                '--output',out/'audit','--beams',args.beams,'--query-limit',str(args.queries)],check=True)
            observations=read_csv(source/'experiment1/aggregate.csv')
            for depth in (100,200):
                write_csv(out/'audit/report'/f'traversal-table-l{depth}.csv',threshold_rows(observations,METHODS,depth))
        if 'rate' in phases:
            run_rate(ROOT,c,source,prepared,out/'audit',out/'rate',args,status)
        report=out/'report';report.mkdir(exist_ok=True)
        if (out/'audit/summary.json').is_file():
            for name in ('inversion-table.csv','traversal-table-l100.csv','traversal-table-l200.csv'):
                link_once(out/'audit/report'/name,report/name)
        if (out/'rate/experiment5/report/summary.json').is_file():
            link_once(out/'rate/experiment5/report/requested-tables.csv',report/'rate-table.csv')
        done34=(out/'audit/summary.json').is_file() and json.loads((out/'audit/summary.json').read_text())['status']=='COMPLETE'
        done5=(out/'rate/experiment5/report/summary.json').is_file() and json.loads((out/'rate/experiment5/report/summary.json').read_text())['status']=='COMPLETE'
        state='COMPLETE' if done34 and done5 else 'PARTIAL'
        save_json(out/'status.json',dict(status=state,phase='done',experiments34_complete=done34,
                                       experiment5_complete=done5,qps_measured=False,elapsed_s=time.monotonic()-started))
        print(f'[{state.lower()}] RS5M experiments 3/4/5; reports={report}; no QPS, one accuracy pass',flush=True)
    except BaseException as exc:
        save_json(out/'status.json',dict(status='FAILED',error=str(exc),elapsed_s=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
