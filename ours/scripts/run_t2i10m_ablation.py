#!/usr/bin/env python3
"""A0--A6 frozen-primary, 64-bit, traversal-wide REMORA component ablation.

Training/encoding, timing, and inversion run in separate serial phases. Reuse
the existing Full checkpoint and FP32 teacher pairs, never rebuild graphs or
overwrite prepared/performance studies. Fixed regularization is calibrated
from TRAIN support, with the same total geometric weight as adaptive Full.
"""
import argparse
import fcntl
from html import escape
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np

VARIANTS = (
    ('A0', 'primary', 'Primary only'),
    ('A1', 'euclidean', 'Euclidean residual Product'),
    ('A2', 'alignment', 'Global JI / ordered groups'),
    ('A3', 'balanced', 'Global JI + Balanced Product'),
    ('A4', 'uncentered-fixed', 'Candidate score MSE + fixed regularization'),
    ('A5', 'centered-fixed', 'Centered score MSE + fixed regularization'),
    ('A6', 'remora', 'Full REMORA / support-aware regularization'),
)


def hms(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def jobs(methods, beams, depths, repeats):
    if not methods or not beams or not depths or repeats < 1:
        raise ValueError('Nonempty methods/beams/depths and positive repeats required')
    if min(beams) < 10 or any(d != 0 and (d < 10 or d > min(beams)) for d in depths):
        raise ValueError('Every fixed rerank length must be valid at EVERY beam')
    return [dict(experiment=2, primary=m, variant=v, side_bits=0 if v=='primary' else 64,
                 beam=b, rerank_depth=d, repeat=r)
            for m in methods for _,v,_ in VARIANTS for b in beams for d in depths
            for r in range(1,repeats+1)]


def summarize_rows(rows, inversion, repeats, aggregate):
    result=[]
    lookup={(r['primary'],r['variant'],r['metric']):r for r in inversion}
    names={v:(a,label) for a,v,label in VARIANTS}
    for r in aggregate(rows,repeats):
        a,label=names[r['variant']]
        value=dict(primary=r['primary'],ablation=a,variant=r['variant'],label=label,
            side_bits=r['side_bits'],beam=r['beam'],candidate_size=r['candidate_size'],
            rerank_depth=r['rerank_depth'],rerank_policy='own traversal queue top-L',
            scoring_placement=r['scoring_placement'],threads=r['threads'],repeats=r['repeats'],
            timing_complete=r['timing_complete'],recall_percent=100*float(r['recall_at_10']),
            qps=r['qps'],qps_cv=r['qps_cv'],mean_service_latency_ms=r['mean_service_latency_ms'],
            p50_ms=r['p50_ms'],p95_ms=r['p95_ms'],p99_ms=r['p99_ms'],visited=r['visited'],
            peak_rss_GB=float(r['peak_rss_bytes'])/1e9,
            stop_inv='N/A: queue exhaustion; no score-based stop',shared_host=True)
        for metric in ('AdjacentInv-R50','AdjacentInv-R75','AdjacentInv-R100','AdjacentInv-R200',
                       'BoundaryInv-K10-h5','BoundaryInv-K10-h10','ExpandInv','QueueInv','RoutingInv'):
            inv=lookup.get((r['primary'],r['variant'],metric))
            value[metric+'_percent']=100*float(inv['inversion_rate']) if inv and inv['inversion_rate']!='' else ''
        result.append(value)
    return result


def bar_plot(path, rows, methods, metric, title, ylabel):
    """Measured fixed-operating-point bars, not a smoothed frontier."""
    colors=['#999999','#8dd3c7','#80b1d3','#fdb462','#b3de69','#bc80bd','#0072b2']
    values=[float(r[metric]) for r in rows if r.get(metric,'')!='' and str(r['timing_complete']).lower()=='true']
    if not values: return
    maximum=max(values)*1.12 or 1
    panel=260; width=90+panel*len(methods); bottom=390
    body=[f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="545">',
          '<rect width="100%" height="100%" fill="white"/>',
          '<g font-family="Arial,sans-serif" font-size="12" fill="#222">',
          f'<text x="75" y="25" font-size="17">{escape(title)}</text>',
          f'<text transform="translate(20,245) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']
    index={(r['primary'],r['variant']):r for r in rows}
    for i,m in enumerate(methods):
        left=75+i*panel
        for t in range(6):
            value=maximum*t/5; y=bottom-290*value/maximum
            body.append(f'<path d="M{left},{y:.2f}h225" stroke="#ddd"/>')
            if i==0: body.append(f'<text x="{left-8}" y="{y+4:.2f}" text-anchor="end">{value:.3g}</text>')
        body.append(f'<text x="{left+112}" y="65" text-anchor="middle" font-size="14">{escape(m)}</text>')
        for j,(a,v,_) in enumerate(VARIANTS):
            r=index.get((m,v)); x=left+8+j*31
            if r and r.get(metric,'')!='' and str(r['timing_complete']).lower()=='true':
                height=290*float(r[metric])/maximum
                body.append(f'<rect x="{x}" y="{bottom-height:.2f}" width="23" height="{height:.2f}" fill="{colors[j]}"/>')
            body.append(f'<text x="{x+11}" y="410" text-anchor="middle">{a}</text>')
        body.append(f'<path d="M{left},95V390h225" stroke="#222" fill="none"/>')
    for j,(a,_,label) in enumerate(VARIANTS):
        x=75+(j%4)*330; y=447+(j//4)*27
        body += [f'<rect x="{x}" y="{y-11}" width="13" height="13" fill="{colors[j]}"/>',
                 f'<text x="{x+20}" y="{y}">{escape(a+": "+label)}</text>']
    body += ['<text x="75" y="526">CPU 8 workers; shared host; query preparation included; fixed graph / beam / rerank policy.</text>',
             '</g></svg>']
    Path(path).parent.mkdir(parents=True,exist_ok=True)
    Path(path).write_text('\n'.join(body)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='/home/xzl/REMORA')
    parser.add_argument('--prepared',default='outputs/t2i-10M/remora-native100-b64-v1')
    parser.add_argument('--performance',default='outputs/t2i-10M/remora-native100-traversal-exp1-nvme-b64-v2')
    parser.add_argument('--teacher',default='outputs/t2i-10M/remora-traversal-exp34-b64-v1')
    parser.add_argument('--output',default='outputs/t2i-10M/remora-ablation-b64-v1')
    parser.add_argument('--methods',default='opq-pq,qaq,leanvec-ood,extended-rabitq,saq')
    parser.add_argument('--beams',default='100,200')
    parser.add_argument('--depths',default='0,100')
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--queries',type=int,default=10000)
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--phases',default='train,parity,timing,inversion,report')
    args=parser.parse_args()
    repo=Path(args.repo).resolve(strict=True)
    sys.path[:0]=[str(repo/'ours/test/t2i-10M'),str(repo/'ours/python'),
                  str(repo/'ours/test/t2i-1M/python'),str(repo/'ours/scripts')]
    from protocol import (file_id,identity,matrix,read_csv,write_csv,write_matrix,load_config,
                          global_inversions,pair_statistics,aggregate)
    from pipeline import Pipeline,save_json
    from run import Experiments,save_experiment_plan
    from run_native100_experiment1 import link_once,check_refiner,check_cold_store
    from run_t2i10m_experiment5 import decode_correction
    prepared=(repo/args.prepared).resolve(strict=True)
    performance=(repo/args.performance).resolve(strict=True)
    teacher=(repo/args.teacher).resolve(strict=True)
    out=(repo/args.output).resolve()
    if any(out==p or p in out.parents or out in p.parents for p in (prepared,performance,teacher)):
        raise ValueError('Ablation outputs must be separate from all frozen inputs')
    methods=args.methods.split(','); beams=sorted(set(map(int,args.beams.split(','))))
    depths=sorted(set(map(int,args.depths.split(',')))); plan=jobs(methods,beams,depths,args.repeats)
    phases=args.phases.split(',')
    if set(phases)-{'train','parity','timing','inversion','report'}: parser.error('Invalid phase')
    # Older frozen training configurations predate the explicit GT convention.
    # Supply current defaults first; every frozen setting still takes priority.
    c=load_config()
    c.update(json.loads((prepared/'configuration.json').read_text())['config'])
    if (c['dataset'],c['base_rows'],c['base_dim'],c['side_bits'])!=('t2i-10M',10000000,200,64):
        raise ValueError('This study requires the frozen native-primary T2I-10M b64 configuration')
    if len(set(methods))!=len(methods) or set(methods)-set(c['methods']) or not 0<args.queries<=10000:
        parser.error('Invalid methods/query count')
    c.update(output_root=str(out),methods=methods,ablation_variants=[v for _,v,_ in VARIANTS if v!='primary'],
        device=f'cuda:{args.gpu}',kmeans_device=f'cuda:{args.gpu}',geometric_encoder_device=f'cuda:{args.gpu}',
        scoring_placement='traversal',lazy_ji=False,rerank_storage='direct',fp32_storage='direct',
        search_threads=8,cpus=c['cpus'][:8],test_query_limit=args.queries,repeats=args.repeats,
        inversion_ranks=[50,75,100,200],include_fp32_reference=False)
    if len(c['cpus'])!=8: raise ValueError('Eight allowed CPU cores are required')
    out.mkdir(parents=True,exist_ok=True)
    lock=(out/'runner.lock').open('a')
    fcntl.flock(lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
    os.sched_setaffinity(0,c['cpus'])
    os.environ['REMORA_NATIVE_PRIMARY_READER']=str(repo/'ours/runtime/native100-reader-v1/roar_native_primary_reader')
    os.environ['REMORA_PRIMARY_DECODE_THREADS']='8'
    for name in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'): os.environ[name]='8'
    p=Pipeline(c)
    for method in methods:
        check_refiner(prepared,method,c['base_rows'],64)
        link_once(prepared/'ji'/method/'b64/remora',out/'ji'/method/'b64/remora')
    link_once(prepared/'pools',out/'pools')
    for name in ('metric.fbin','select.fbin','build.fbin'):
        link_once(prepared/'data'/name,out/'data'/name)
    for name,dtype in [('test.fbin','<f4'),('truth.ids.ibin','<u4'),('truth.scores.fbin','<f4')]:
        source=prepared/'data'/name
        if args.queries==len(matrix(source,dtype)): link_once(source,out/'data'/name)
        else:
            values=np.asarray(matrix(source,dtype)[:args.queries]); dest=out/'data'/name
            if dest.exists() and not np.array_equal(matrix(dest,dtype),values): raise ValueError('Query subset changed')
            if not dest.exists(): write_matrix(dest,values,dtype)
    link_once(prepared/'data/truth.metadata.json',out/'data/truth.metadata.json')
    cold=(performance/'data/base.cold').resolve(strict=True)
    check_cold_store(cold,c['base']);link_once(cold,out/'data/base.cold')
    save_experiment_plan(out,plan)
    write_csv(out/'ablation-variants.csv',[dict(ablation=a,variant=v,label=label,
        side_bits=0 if v=='primary' else 64,scoring_placement='primary' if v=='primary' else 'traversal')
        for a,v,label in VARIANTS])
    manifest=dict(version=1,dataset=c['dataset'],methods=methods,side_bits=64,beams=beams,rerank_depths=depths,
        repeats=args.repeats,queries=args.queries,prepared=str(prepared),teacher=str(teacher),
        scoring_placement='traversal',graphs_rebuilt=False,primary_reencoded=False,
        full_checkpoint_reused=True,seed=c['seed'],centered_epochs=c['centered_epochs'],recode_nodes=c['recode_nodes'],
        regularization_control='fixed TRAIN-mass-weighted mean of D_eff/n_eff; equal total prior weight as Full',
        cpu_threads=8,cpus=c['cpus'],gpu=args.gpu,memory_budget_bytes=c['memory_budget_bytes'],shared_host=True,
        fp32_rerank_storage='NVMe O_DIRECT; own scorer queue top-L; query preparation included',
        test_used_for_training=False,teacher_pairs='same frozen experiment-3 FP32 teacher pairs',
        runner_id=file_id(Path(__file__)),source_id=p.source_id,
        frozen_inputs={str(path):file_id(path) for path in
            [prepared/'configuration.json',teacher/'experiment3/raw/common.ids.ibin',
             teacher/'experiment3/raw/routing-pairs.csv',cold]})
    study=out/'study.json'
    if study.exists() and json.loads(study.read_text())!=manifest: raise ValueError('Study changed; choose a new output')
    if not study.exists(): save_json(study,manifest)
    started=time.monotonic()
    def status(phase,**extra):
        save_json(out/'status.json',dict(status='RUNNING',phase=phase,elapsed_s=time.monotonic()-started,**extra))
    trainer_config=out/'trainer-config.json'
    if not trainer_config.exists(): save_json(trainer_config,c)
    def check_model(method,variant):
        dest=out/'ji'/method/f'b64/{variant}'
        summary=json.loads((dest/'training.json').read_text())
        with (dest/'side.bin').open('rb') as stream: side=struct.unpack('<8sIQIIIQQ',stream.read(48))
        with (dest/'online.bin').open('rb') as stream: online=struct.unpack('<8sIIIIQQQQ',stream.read(56))
        if (side[:3]!=(b'JIRCODE1',3,c['base_rows']) or side[4:6]!=(64,8)
            or online[:2]!=(b'JIONL002',2) or online[3:6]!=(8,64,c['base_rows'])
            or side[6:8]!=online[6:8] or (dest/'side.bin').stat().st_size!=48+c['base_rows']*8
            or summary['variant']!=variant or summary['test_queries_used']):
            raise ValueError(f'Invalid frozen ablation model: {method}/{variant}')
        if json.loads((dest/'encoding-parity.json').read_text())['faiss_canonical_mismatches']:
            raise ValueError('Geometric encoder parity failed')
        return dest,summary
    try:
        if 'train' in phases:
            status('train')
            if not p.preflight(): raise RuntimeError(f'Preflight failed; see {out}/preflight.csv')
            work=[(m,v) for m in methods for _,v,_ in VARIANTS if v not in ('primary','remora')]
            for j,(method,variant) in enumerate(work,1):
                dest=out/'ji'/method/f'b64/{variant}'; native=p.runs[method]['paths']
                pools=[out/'pools'/method/split/'pools.npz' for split in ('metric','select')]
                p.stage(f'ablation-train-{method}-{variant}',[sys.executable,repo/'ours/test/t2i-10M/train_remora.py',
                    '--config',trainer_config,'--method',method,'--model',native['model'],'--codes',native['codes'],
                    '--train-pool',pools[0],'--validation-pool',pools[1],'--bits',64,'--variant',variant,'--output',dest],
                    [dest/'online.bin',dest/'side.bin',dest/'training.json',dest/'encoding-parity.json'],
                    [trainer_config,native['model'],native['codes'],*pools])
                check_model(method,variant)
                elapsed=time.monotonic()-started
                print(f'[train progress] {j}/{len(work)} elapsed/estimated_total={hms(elapsed)}/{hms(elapsed*len(work)/j)}',flush=True)
                status('train',models_completed=j,models_to_train=len(work))
        if any(x in phases for x in ('parity','timing','inversion')):
            for m in methods:
                for _,v,_ in VARIANTS:
                    if v!='primary': check_model(m,v)
        if 'parity' in phases:
            status('parity'); records=[]; dest=out/'experiment2/parity';dest.mkdir(parents=True,exist_ok=True)
            count=min(16,args.queries);query=np.asarray(matrix(out/'data/test.fbin')[:count])
            ids=np.asarray(matrix(teacher/'experiment3/raw/common.ids.ibin','<u4')[:count,:32])
            for name,values,dtype in [('query.fbin',query,'<f4'),('ids.ibin',ids,'<u4')]:
                if not (dest/name).exists(): write_matrix(dest/name,values,dtype)
            for m in methods:
                primary_scores=np.asarray(matrix(teacher/f'experiment3/raw/{m}-primary-b0.scores.fbin')[:count,:32])
                for _,v,_ in VARIANTS:
                    if v=='primary': continue
                    branch=p.branch(m,v,64);prefix=dest/f'{m}-{v}'
                    p.stage(f'ablation-parity-{m}-{v}',[Path(c['build_dir'])/'roar_exp234_audit',
                        *p.native_args(branch,v),'--graph',branch['graph'],'--queries',dest/'query.fbin',
                        '--audit-ids',dest/'ids.ibin','--out-prefix',prefix],
                        [Path(str(prefix)+'.scores.fbin')],
                        [branch[k] for k in ('model','codes','ji_model','ji_codes')]+[dest/'query.fbin',dest/'ids.ibin'])
                    model=out/'ji'/m/f'b64/{v}'
                    with np.load(model/'model.npz') as z: books,projection,mean=[z[k].copy() for k in ('books','projection','mean')]
                    labels=np.memmap(model/'side.bin',mode='r',dtype='u1',offset=48,shape=(c['base_rows'],8))
                    rec=decode_correction(books,projection,mean,labels[ids.ravel()],c['base_dim']).reshape(count,32,c['base_dim'])
                    expected=primary_scores+np.einsum('qnd,qd->qn',rec,query)
                    actual=matrix(str(prefix)+'.scores.fbin');error=float(np.max(abs(expected-actual)))
                    if error>2e-5: raise ValueError(f'Native residual score mismatch {m}/{v}: {error}')
                    records.append(dict(primary=m,variant=v,max_abs_score_error=error,passed=True))
                    print(f'[native score parity] {m}/{v}: max_error={error:.3g}',flush=True)
            write_csv(dest/'summary.csv',records)
        if 'timing' in phases:
            status('timing');runner=Experiments(p)
            print(f'[timing] {len(plan)} fixed points; ALL training completed; no tracing during QPS',flush=True)
            runner.timings(plan)
            latest={r['point_id']:r for r in read_csv(out/'experiment2/results.csv')}
            if any(r['status']!='OK' for r in latest.values()): raise RuntimeError('Failed native timing point; inspect results.csv')
        if 'inversion' in phases:
            status('inversion');dest=out/'experiment2/audit';dest.mkdir(parents=True,exist_ok=True)
            common=np.asarray(matrix(teacher/'experiment3/raw/common.ids.ibin','<u4')[:args.queries])
            aid=dest/'common.ids.ibin'
            if not aid.exists(): write_matrix(aid,common,'<u4')
            exact=np.asarray(matrix(teacher/'experiment3/raw/full-precision-fp32-b0.scores.fbin')[:args.queries])
            pairs=[r for r in read_csv(teacher/'experiment3/raw/routing-pairs.csv') if int(r['query_id'])<args.queries]
            lookup=[]
            for ids in common:
                local={}
                for j,node in enumerate(ids): local.setdefault(int(node),j)
                lookup.append(local)
            locations={}
            for name,events in [('ExpandInv',{'exp'}),('QueueInv',{'queue'}),('RoutingInv',{'exp','queue'})]:
                local=[r for r in pairs if r['event_type'] in events]
                q=np.asarray([int(r['query_id']) for r in local],np.int64)
                u=np.asarray([lookup[int(r['query_id'])][int(r['u_id'])] for r in local],np.int64)
                v=np.asarray([lookup[int(r['query_id'])][int(r['v_id'])] for r in local],np.int64)
                locations[name]=(q,u,v)
            inversions=[]
            for m in methods:
                for a,v,_ in VARIANTS:
                    if v in ('primary','remora'):
                        bits=0 if v=='primary' else 64
                        scores=np.asarray(matrix(teacher/f'experiment3/raw/{m}-{v}-b{bits}.scores.fbin')[:args.queries])
                    else:
                        branch=p.branch(m,v,64);prefix=dest/f'{m}-{v}'
                        p.stage(f'ablation-inversion-{m}-{v}',[Path(c['build_dir'])/'roar_exp234_audit',
                            *p.native_args(branch,v),'--graph',branch['graph'],'--queries',out/'data/test.fbin',
                            '--audit-ids',aid,'--out-prefix',prefix],[Path(str(prefix)+'.scores.fbin')],
                            [branch[k] for k in ('model','codes','ji_model','ji_codes')]+[aid,out/'data/test.fbin'])
                        scores=np.asarray(matrix(str(prefix)+'.scores.fbin'))
                    if scores.shape!=exact.shape or not np.isfinite(scores).all(): raise ValueError('Invalid score matrix')
                    metrics=global_inversions(exact[:,:200],scores[:,:200],c['inversion_ranks'])
                    for name,(q,u,w) in locations.items():
                        metrics.append(dict(metric=name,**pair_statistics(exact[q,u]-exact[q,w],scores[q,u]-scores[q,w],q,args.queries)))
                    inversions.extend(dict(primary=m,ablation=a,variant=v,side_bits=0 if v=='primary' else 64,
                        teacher=str(teacher),**r) for r in metrics)
                    write_csv(out/'experiment2/inversions.csv',inversions)
                    print(f'[inversion] {m}/{v} complete; fixed teacher pairs',flush=True)
        if 'report' in phases:
            status('report');report=out/'report';report.mkdir(parents=True,exist_ok=True)
            raw=list({r['point_id']:r for r in read_csv(out/'experiment2/results.csv')}.values())
            inversion=read_csv(out/'experiment2/inversions.csv')
            rows=summarize_rows(raw,inversion,args.repeats,aggregate)
            write_csv(report/'ablation-table.csv',rows)
            write_csv(report/'inversion-table.csv',inversion)
            summaries=[]
            for m in methods:
                for a,v,label in VARIANTS:
                    if v=='primary':continue
                    model,s=check_model(m,v)
                    summaries.append(dict(primary=m,ablation=a,variant=v,label=label,
                        side_bits=64,model_MB=s['model_bytes']/1e6,side_MB=s['side_bytes']/1e6,
                        training_elapsed_s=s['elapsed_s'],best_epoch=s['best_epoch'],
                        initial_validation_recall=s['initial_validation']['fixed_pool_recall'],
                        best_validation_recall=s['best_validation']['fixed_pool_recall'],
                        best_validation_centered_mse=s['best_validation']['centered_mse'],
                        fixed_strength=s.get('fixed_regularization_strength',''),
                        prior_weight_sum=s.get('geometric_penalty_sum',''),reused_checkpoint=v=='remora',
                        seed=c['seed'],test_used_for_training=False))
            write_csv(report/'training-table.csv',summaries)
            for b in beams:
                for d in depths:
                    local=[r for r in rows if int(r['beam'])==b and int(r['rerank_depth'])==d]
                    for metric,label in [('recall_percent','Recall@10 (%)'),('qps','QPS'),
                                         ('BoundaryInv-K10-h5_percent','Boundary inversion@10 / h=5 (%)')]:
                        bar_plot(report/f'{metric}-b{b}-l{d}.svg',local,methods,metric,
                            f'T2I-10M component ablation / B=64 / beam={b} / L={d}',label)
            failed=[r for r in raw if r['status']!='OK']
            complete=(len(raw)==len(plan) and len(rows)==len(plan)//args.repeats
                      and all(str(r['timing_complete']).lower()=='true' for r in rows)
                      and not failed and len(inversion)==len(methods)*len(VARIANTS)*9)
            summary=dict(status='COMPLETE' if complete else 'INCOMPLETE',experiment=2,
                methods=methods,variants=[dict(ablation=a,variant=v,label=l) for a,v,l in VARIANTS],
                expected_timing_runs=len(plan),timing_runs=len(raw),operating_points=len(rows),
                expected_inversion_rows=len(methods)*len(VARIANTS)*9,inversion_rows=len(inversion),
                failed_runs=len(failed),side_bits=64,scoring_placement='traversal',threads=8,
                repeats=args.repeats,queries=args.queries,shared_host=True,graphs_rebuilt=False,
                full_checkpoint_reused=True,elapsed_this_invocation_s=time.monotonic()-started,
                note='Fixed beam/capacity and own-queue rerank policy; candidate IDs may differ. Single training seed, not a multi-seed robustness study.')
            save_json(report/'summary.json',summary);save_json(out/'status.json',dict(summary,phase='done'))
            print(f'[done] ablation report: {report}/summary.json; status={summary["status"]}',flush=True)
            if not complete: raise RuntimeError('Ablation report is incomplete; missing observations are not filled')
    except BaseException as error:
        save_json(out/'status.json',dict(status='FAILED',error=str(error),elapsed_s=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
