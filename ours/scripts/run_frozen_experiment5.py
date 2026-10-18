#!/usr/bin/env python3
"""Dataset-aware frozen-primary REMORA traversal rate study and CSV tables.

Train each side budget independently, then run CPU timings without concurrent
training/audit work. Never rebuild graphs, re-encode primary codes, overwrite a
cold store, or introduce terminal-only scoring. Existing reference models are read-only.
"""
import argparse
from collections import defaultdict
import fcntl
import hashlib
from html import escape
import json
import os
from pathlib import Path
import struct
import sys
import time

import numpy as np


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    temporary.replace(path)


def unique_file_bytes(paths):
    seen=set()
    total=0
    for path in paths:
        st=Path(path).stat()
        key=(st.st_dev,st.st_ino)
        if key not in seen:
            seen.add(key)
            total+=st.st_size
    return total


def evaluation_nodes(rows, fit_size, size, seed):
    """Same independent database sample for every primary and every budget."""
    if size <= 0 or size > rows - fit_size:
        raise ValueError('Distortion sample must fit outside the geometry fit sample')
    fit = np.random.default_rng(seed).choice(rows, fit_size, replace=False)
    allowed = np.ones(rows, dtype=bool)
    allowed[fit] = False
    return np.random.default_rng(seed + 917).choice(np.flatnonzero(allowed), size, replace=False)


def weighted_distortion(error, second_moment):
    e = np.asarray(error, np.float64)
    w = np.asarray(second_moment, np.float64)
    if e.ndim != 2 or w.shape != (e.shape[1], e.shape[1]) or not np.isfinite(e).all():
        raise ValueError('Invalid full-score error / query second moment')
    value = float(np.einsum('ij,ij->', e @ w, e) / len(e))
    if not np.isfinite(value) or value < -1e-12:
        raise ValueError('Invalid weighted distortion')
    return max(0., value)


def decode_correction(books, projection, mean, labels, raw_dim):
    """Transpose of the deployed linear LUT decoder, including its mean bias."""
    groups, _, width = books.shape
    if labels.ndim != 2 or labels.shape[1] != groups or projection.shape != (groups*width,)*2:
        raise ValueError('Refiner product/projection shape mismatch')
    latent = np.concatenate([books[g, labels[:, g]] for g in range(groups)], axis=1)
    return (latent @ projection.T)[:, :raw_dim] + mean[:raw_dim]


def user_table_rows(method, beams, budgets, quality, timings):
    """Do not select the most flattering beam, fill gaps, or invent results."""
    quality = {(r['primary'], int(r['side_bits'])): r for r in quality}
    timings = {(r['primary'], int(r['side_bits']), int(r['beam'])): r for r in timings
               if str(r.get('timing_complete', '')).lower() == 'true'}
    result = []
    for beam in beams:
        for bits in budgets:
            q = quality.get((method, bits), {})
            t = timings.get((method, bits, beam), {})
            row = dict(primary=method, beam=beam, side_bits=bits,
                       status='COMPLETE' if q and t else 'PENDING',
                       DW=q.get('dw', ''), score_rmse=q.get('score_rmse_top200', ''),
                       routing_inv_percent=100*float(q['routing_inv']) if q.get('routing_inv', '') != '' else '',
                       expand_inv_percent=100*float(q['expand_inv']) if q.get('expand_inv', '') != '' else '',
                       queue_inv_percent=100*float(q['queue_inv']) if q.get('queue_inv', '') != '' else '',
                       stop_inv='N/A: queue exhaustion, not score-based termination',
                       boundary_inv_k10_h5_percent=100*float(q['boundary_inv_h5']) if q.get('boundary_inv_h5', '') != '' else '',
                       boundary_inv_k10_h10_percent=100*float(q['boundary_inv_h10']) if q.get('boundary_inv_h10', '') != '' else '',
                       recall_at_10_percent=100*float(t['recall_at_10']) if t else '',
                       side_payload_MB=float(q['side_payload_bytes'])/1e6 if q else '',
                       refiner_model_MB=float(q['refiner_model_bytes'])/1e6 if q else '',
                       hot_index_GB=float(q['hot_index_bytes'])/1e9 if q else '',
                       qps=t.get('qps', ''), mean_latency_ms=t.get('mean_service_latency_ms', ''),
                       peak_rss_GB=float(t['peak_rss_bytes'])/1e9 if t else '',
                       rerank_depth=0, scoring_placement='traversal')
            result.append(row)
    return result


def rate_plot(path, rows, metric, title, ylabel):
    """SVG through measured budget points; no interpolation/extrapolation."""
    colors = ['#0072B2', '#D55E00', '#009E73', '#CC79A7', '#E69F00']
    series = defaultdict(list)
    for r in rows:
        if r.get(metric, '') != '':
            series[r['primary']].append((int(r['side_bits']), float(r[metric])))
    points = [p for values in series.values() for p in values]
    if not points:
        return
    xmax = max(x for x, _ in points)
    ymax = max(y for _, y in points)*1.08 or 1.
    xx = lambda x: 90 + 535*x/max(xmax, 1)
    yy = lambda y: 435 - 375*y/ymax
    body = ['<svg xmlns="http://www.w3.org/2000/svg" width="900" height="520">',
            '<rect width="100%" height="100%" fill="white"/>',
            '<g font-family="Arial,sans-serif" font-size="14" fill="#222">',
            f'<text x="90" y="28" font-size="18">{escape(title)}</text>']
    for i in range(6):
        y = ymax*i/5
        body += [f'<path d="M90,{yy(y):.2f}H625" stroke="#ddd"/>',
                 f'<text x="78" y="{yy(y)+5:.2f}" text-anchor="end">{y:.4g}</text>']
    for x in sorted({x for x, _ in points}):
        body.append(f'<text x="{xx(x):.2f}" y="460" text-anchor="middle">{x}</text>')
    body += ['<path d="M90,60V435H625" stroke="#222" fill="none"/>',
             '<text x="350" y="498" text-anchor="middle">Side bits / vector</text>',
             f'<text transform="translate(20,250) rotate(-90)" text-anchor="middle">{escape(ylabel)}</text>']
    for i, (name, values) in enumerate(sorted(series.items())):
        color = colors[i % len(colors)]
        coords = ' '.join(f'{xx(x):.2f},{yy(y):.2f}' for x, y in sorted(values))
        body += [f'<polyline points="{coords}" fill="none" stroke="{color}" stroke-width="2.2"/>',
                 f'<path d="M650,{75+i*29}h25" stroke="{color}" stroke-width="2.2"/>',
                 f'<text x="685" y="{80+i*29}">{escape(name)}</text>']
    body.append('</g></svg>')
    Path(path).write_text('\n'.join(body)+'\n')


def audit_quality(p, args, imported, ids, w, api):
    """Frozen teacher Top-200 and routing pairs, independent of each refiner."""
    matrix, read_csv, write_csv, file_id, identity = api
    from protocol import global_inversions, pair_statistics
    from qvrr import Primary
    teacher = Path(args.audit_source)/'experiment3/raw'
    exact_path = teacher/'full-precision-fp32-b0.scores.fbin'
    common_path = teacher/'common.ids.ibin'
    exact = np.asarray(matrix(exact_path))
    common = np.asarray(matrix(common_path, '<u4'))
    if exact.shape != common.shape or len(exact) != args.queries or exact.shape[1] < 200:
        raise ValueError('Frozen teacher/query dimensions mismatch')
    pairs = read_csv(teacher/'routing-pairs.csv')
    # Padding repeats ID[0]; retain the first, true column when resolving IDs.
    lookup = []
    for values in common:
        row = {}
        for j, node in enumerate(values):
            row.setdefault(int(node), j)
        lookup.append(row)
    def indices(events):
        local = [v for v in pairs if v['event_type'] in events]
        q = np.asarray([int(v['query_id']) for v in local], dtype=np.int64)
        left = np.asarray([lookup[int(v['query_id'])][int(v['u_id'])] for v in local], dtype=np.int64)
        right = np.asarray([lookup[int(v['query_id'])][int(v['v_id'])] for v in local], dtype=np.int64)
        return q, left, right
    indices_by_name = {'routing_inv': indices({'exp', 'queue'}),
                       'expand_inv': indices({'exp'}), 'queue_inv': indices({'queue'})}
    queries = matrix(p.out/'data/test.fbin')
    base = matrix(p.c['base'])
    records = []
    inv_records = []
    dest = p.out/'experiment5/quality'
    dest.mkdir(parents=True, exist_ok=True)
    total = len(p.c['methods'])*(len(args.budgets)+1)
    completed = 0
    for method in p.c['methods']:
        pri = Primary(imported[method]['paths']['model'], imported[method]['paths']['codes'])
        try:
            residual = np.asarray(base[ids])-pri.reconstruct_raw(ids)
            # Decode once per primary; side models all use this SAME complete error.
            for bits in [0, *args.budgets]:
                variant = 'remora' if bits else 'primary'
                branch = p.branch(method, variant, bits)
                model_dest = p.out/'ji'/method/f'b{bits}/remora'
                dependencies = [branch['model'], branch['codes'], common_path, exact_path,
                                teacher/'routing-pairs.csv', p.out/'data/test.fbin', p.c['base']]
                if bits:
                    dependencies += [model_dest/'model.npz', model_dest/'side.bin', model_dest/'online.bin']
                stamp = identity(dict(sample=hashlib.sha256(ids.tobytes()).hexdigest(),
                    study=json.loads((p.out/'study.json').read_text()),
                    artifacts={str(f):file_id(f) for f in dependencies}))
                cached = dest/f'{method}-b{bits}.json'
                if cached.exists() and json.loads(cached.read_text()).get('fingerprint') == stamp:
                    value = json.loads(cached.read_text())
                    records.append(value['quality']); inv_records.extend(value['inversions'])
                    print(f'[reuse] quality {method}/b{bits}', flush=True)
                    completed += 1
                    continue
                native_scores = teacher/f'{method}-{variant}-b{bits}.scores.fbin'
                if bits not in (0, args.reference_budget):
                    prefix = dest/f'{method}-b{bits}'
                    native_scores = Path(str(prefix)+'.scores.fbin')
                    command = [Path(p.c['build_dir'])/'roar_exp234_audit', *p.native_args(branch,variant),
                        '--graph', p.c['fp_graph'], '--queries', p.out/'data/test.fbin',
                        '--audit-ids', common_path, '--out-prefix', prefix]
                    p.stage(f'audit-{method}-b{bits}', command, [native_scores], dependencies,
                            budget=p.c['memory_budget_bytes'])
                scores = np.asarray(matrix(native_scores))
                if scores.shape != exact.shape or not np.isfinite(scores).all():
                    raise ValueError('Invalid native score audit')
                error = residual
                side_payload = side_bytes = model_bytes = 0
                if bits:
                    with np.load(model_dest/'model.npz') as z:
                        books, projection, mean = [z[k].copy() for k in ('books','projection','mean')]
                    groups = bits//8
                    labels = np.memmap(model_dest/'side.bin', mode='r', dtype='u1', offset=48,
                                       shape=(p.c['base_rows'],groups))
                    correction = decode_correction(books, projection, mean, labels[ids], p.c['base_dim'])
                    error = residual-correction
                    # Test the Python distortion decoder against the actual native scorer.
                    primary_scores = matrix(teacher/f'{method}-primary-b0.scores.fbin')
                    parity_error = 0.
                    for q in np.linspace(0, args.queries-1, min(16,args.queries)).astype(int):
                        nodes = common[q,:32]
                        rec = decode_correction(books,projection,mean,labels[nodes],p.c['base_dim'])
                        parity_error = max(parity_error,float(np.max(abs(
                            rec@queries[q] - (scores[q,:32]-primary_scores[q,:32])))))
                    if parity_error > 2e-5:
                        raise ValueError(f'Native/Python residual decoder mismatch: {parity_error}')
                    side_payload = p.c['base_rows']*groups
                    side_bytes = (model_dest/'side.bin').stat().st_size
                    model_bytes = (model_dest/'online.bin').stat().st_size
                else:
                    parity_error = 0.
                distortion = weighted_distortion(error,w)
                metrics = global_inversions(exact[:,:200],scores[:,:200],(50,75,100,200))
                routing = {}
                for name,(q,left,right) in indices_by_name.items():
                    stat = pair_statistics(exact[q,left]-exact[q,right], scores[q,left]-scores[q,right], q,args.queries)
                    metrics.append(dict(metric=name,**stat))
                    routing[name] = stat['inversion_rate']
                metric_map = {v['metric']:v for v in metrics}
                values = dict(primary=method, side_bits=bits, dw=distortion,
                    global_score_rmse=float(np.sqrt(distortion)),
                    score_rmse_top200=float(np.sqrt(np.mean(np.square(
                        exact[:,:200].astype(np.float64)-scores[:,:200].astype(np.float64))))),
                    boundary_inv_h5=metric_map['BoundaryInv-K10-h5']['inversion_rate'],
                    boundary_inv_h10=metric_map['BoundaryInv-K10-h10']['inversion_rate'],
                    **routing, side_payload_bytes=side_payload, side_file_bytes=side_bytes,
                    refiner_model_bytes=model_bytes, hot_index_bytes=unique_file_bytes(
                        [branch[k] for k in ('graph','codes','model')])+side_bytes+model_bytes,
                    native_python_correction_max_error=parity_error,
                    distortion_nodes=len(ids), score_queries=args.queries,
                    dw_definition='E_e[e.T W_test e]; uncentered complete remaining residual, W_test=E_test[q q.T]',
                    rmse_definition='actual native score error on fixed full-database exact Top-200',
                    routing_definition='query-balanced inversion over common FP32 expansion + queue pairs; excludes stop proxy',
                    storage_definition='unique serialized graph + primary model/codes + refiner model/codes; same inode once; excludes runtime buffers and offline files')
                inversions = [dict(primary=method,side_bits=bits,**v) for v in metrics]
                save_json(cached,dict(fingerprint=stamp,quality=values,inversions=inversions))
                records.append(values); inv_records.extend(inversions)
                write_csv(dest/'metrics.csv',records)
                write_csv(dest/'inversions.csv',inv_records)
                completed += 1
                print(f'[quality] {completed}/{total} {method}/b{bits}: DW={distortion:.8g} '
                      f'Top200 RMSE={values["score_rmse_top200"]:.6g}',flush=True)
        finally:
            pri.close()
    write_csv(dest/'metrics.csv',records)
    write_csv(dest/'inversions.csv',inv_records)
    return records


def report(p, args, quality, api):
    _, read_csv, write_csv, _, _ = api
    from protocol import aggregate
    rows = read_csv(p.out/'experiment5/results.csv')
    latest = {r['point_id']:r for r in rows}
    timings = aggregate(list(latest.values()),p.c['repeats'])
    dest = p.out/'experiment5/report'
    dest.mkdir(parents=True,exist_ok=True)
    write_csv(dest/'aggregate.csv',timings)
    tables=[]
    for method in p.c['methods']:
        local = user_table_rows(method,args.beams,args.budgets,quality,timings)
        tables.extend(local)
        for beam in args.beams:
            chosen = [r for r in local if r['beam']==beam]
            write_csv(dest/f'{method}-beam{beam}.csv',chosen)
            display=[{'Side bits B':r['side_bits'],'DW ↓':r['DW'],
                'Score RMSE ↓':r['score_rmse'],'Routing Inv. ↓ (%)':r['routing_inv_percent'],
                'Boundary Inv.@10 h=5 ↓ (%)':r['boundary_inv_k10_h5_percent'],
                'Boundary Inv.@10 h=10 ↓ (%)':r['boundary_inv_k10_h10_percent'],
                'Recall@10 ↑ (%)':r['recall_at_10_percent'],
                'Storage: side codes (MB)':r['side_payload_MB'],
                'Storage: shared refiner (MB)':r['refiner_model_MB'],
                'Storage: hot index (GB)':r['hot_index_GB'],'Status':r['status']} for r in chosen]
            write_csv(dest/f'{method}-beam{beam}-table.csv',display)
    write_csv(dest/'requested-tables.csv',tables)
    write_csv(dest/'quality.csv',quality)
    for metric,label in [('DW','Query-visible distortion'),('score_rmse','Score RMSE, exact Top-200'),
                         ('routing_inv_percent','Routing inversion (%)'),
                         ('boundary_inv_k10_h5_percent','Boundary inversion @10, h=5 (%)'),
                         ('recall_at_10_percent','Recall@10 (%)'),('qps','QPS')]:
        for beam in args.beams:
            rate_plot(dest/f'{metric}-bits-beam{beam}.svg',[r for r in tables if r['beam']==beam],
                      metric,f'{p.c["dataset"]} / REMORA traversal / beam={beam} / no FP rerank',label)
    expected = len(p.c['methods'])*(len(args.budgets)+1)*len(args.beams)*p.c['repeats']
    complete = (len(latest)==expected and all(r['status']=='OK' for r in latest.values())
                and len(quality)==len(p.c['methods'])*(len(args.budgets)+1)
                and all(r['status']=='COMPLETE' for r in tables))
    summary=dict(status='COMPLETE' if complete else 'INCOMPLETE', experiment=5,
        dataset=p.c['dataset'], methods=p.c['methods'], side_bits=args.budgets, beams=args.beams,
        queries=args.queries, repeats=p.c['repeats'], threads=p.c['search_threads'], cpus=p.c['cpus'],
        scoring_placement='traversal', rerank_depth=0, allocator='balanced-product; NOT historical DP',
        geometry_fit_size=p.c['residual_sample_size'], distortion_sample_nodes=args.distortion_nodes,
        dw_query_geometry='held-out test query second moment, evaluation only; does not change models',
        score_rmse='fixed full-database exact Top-200; differs from sqrt(DW) due to query-node dependence',
        routing_inversion='common FP32 expansion + queue pairs; query-balanced; true StopInv N/A',
        primary_and_graph_frozen=True, reference_budget=args.reference_budget,
        reference_models_reused=args.reference_budget in args.budgets,
        primary_nominal_target_bits=p.c['primary_budget_bits'],
        side_budget_fraction={b:b/p.c['primary_budget_bits'] for b in args.budgets},
        relative_default_multiplier={b:b/args.reference_budget for b in args.budgets},
        timing_scope='shared host; affinity not exclusive',
        training_and_timings_sequential=True, successful_measurements=sum(r['status']=='OK' for r in latest.values()),
        expected_measurements=expected, quality_rows=len(quality), requested_table=str(dest/'requested-tables.csv'))
    save_json(dest/'summary.json',summary)
    print(f'[{summary["status"].lower()}] {dest/"summary.json"}',flush=True)
    return complete


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--repo',default='/home/xzl/REMORA')
    parser.add_argument('--source',default='outputs/t2i-10M/remora-native100-traversal-exp1-nvme-b64-v2')
    parser.add_argument('--source-experiment',type=int,choices=(1,4),default=1)
    parser.add_argument('--prepared',default='outputs/t2i-10M/remora-native100-b64-v1')
    parser.add_argument('--audit-source',default='outputs/t2i-10M/remora-traversal-exp34-b64-v1')
    parser.add_argument('--output',default='outputs/t2i-10M/remora-traversal-exp5-b32-128-v1')
    parser.add_argument('--budgets',default='32,64,96,128')
    parser.add_argument('--beams',default='100,200')
    parser.add_argument('--methods',default='opq-pq,qaq,leanvec-ood,extended-rabitq,saq')
    parser.add_argument('--phases',default='train,timing,quality,report')
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--queries',type=int,default=10000)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--distortion-nodes',type=int,default=100000)
    parser.add_argument('--build-dir',default='')
    parser.add_argument('--trainer',default='')
    args=parser.parse_args()
    repo=Path(args.repo).resolve(strict=True)
    sys.path[:0]=[str(repo/'ours/test/t2i-10M'),str(repo/'ours/scripts'),
                  str(repo/'ours/python'),str(repo/'ours/test/t2i-1M/python')]
    from protocol import file_id,identity,matrix,read_csv,write_csv
    from pipeline import Pipeline
    from run import Experiments,save_experiment_plan
    from run_native100_experiment1 import check_refiner,link_once
    from scoring_contract import require_traversal
    api=matrix,read_csv,write_csv,file_id,identity
    args.budgets=sorted(set(map(int,args.budgets.split(','))))
    args.beams=sorted(set(map(int,args.beams.split(','))))
    methods=args.methods.split(',')
    phases=args.phases.split(',')
    if (not args.budgets or any(b<=0 or b>512 or b%8 for b in args.budgets)
            or not args.beams or min(args.beams)<10 or args.repeats<1 or args.queries<1
            or set(phases)-{'train','timing','quality','report'}):
        parser.error('Invalid budgets, beams, repeat/query counts or phases')
    source=(repo/args.source).resolve(strict=True)
    prepared=(repo/args.prepared).resolve(strict=True)
    args.audit_source=str((repo/args.audit_source).resolve(strict=True))
    out=(repo/args.output).resolve()
    for frozen in (source,prepared,Path(args.audit_source)):
        if out==frozen or frozen in out.parents:
            raise ValueError('New results must not alter a frozen source output root')
    c=dict(json.loads((source/'configuration.json').read_text())['config'])
    # Historical T2I configurations predate the WebVid source-score flag.
    # Here the frozen raw-data truth is already an IP truth, never normalize it.
    c.setdefault('combined_ground_truth_score_kind','inner-product')
    args.reference_budget=c['side_bits']
    require_traversal(c)
    if (c['dataset']!=args.dataset or set(methods)-set(c['methods']) or args.queries!=c['test_query_limit']
            or json.loads((source/f'experiment{args.source_experiment}/summary.json').read_text())['status']!='COMPLETE'):
        raise ValueError('Expected complete matching-dataset traversal source and identical test query count')
    audit_study=json.loads((Path(args.audit_source)/'study.json').read_text())
    if audit_study['source']!=str(source) or audit_study['queries']!=args.queries:
        raise ValueError('Reference pairs must belong to the same frozen source/test queries')
    if audit_study['side_bits']!=args.reference_budget:
        raise ValueError('Reference audit/model budget mismatch')
    # WebVid has no FP32-built graph. Reuse exactly the reference scorer's
    # fixed OPQ topology and label it accurately, never present it as FP search.
    c['fp_graph']=audit_study['teacher_graph']
    if args.build_dir:
        c['build_dir']=str((repo/args.build_dir).resolve(strict=True))
    trainer=(repo/args.trainer).resolve(strict=True) if args.trainer else repo/'ours/test/t2i-10M/train_remora.py'
    c.update(output_root=str(out),methods=methods,budgets=args.budgets,
        bit_budget_beams=args.beams,beams=args.beams,repeats=args.repeats,
        performance_sampling='dense',ablation_variants=['remora'],scoring_placement='traversal',
        device=f'cuda:{args.gpu}',kmeans_device=f'cuda:{args.gpu}',geometric_encoder_device=f'cuda:{args.gpu}',
        build_threads=8,search_threads=8,trainer_threads=8,include_fp32_reference=False)
    out.mkdir(parents=True,exist_ok=True)
    (out/'logs').mkdir(exist_ok=True)
    lock=(out/'study.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
        os.environ[name]='8'
    os.sched_setaffinity(0,c['cpus'])
    os.environ['REMORA_NATIVE_PRIMARY_READER']=str(repo/'ours/runtime/native100-reader-v1/roar_native_primary_reader')
    os.environ['REMORA_PRIMARY_DECODE_THREADS']='8'
    for name in ('test.fbin','truth.ids.ibin','truth.scores.fbin','truth.metadata.json'):
        link_once(source/'data'/name,out/'data'/name)
    for name in ('metric.fbin','select.fbin'):
        link_once(prepared/'data'/name,out/'data'/name)
    for method in methods:
        check_refiner(source,method,c['base_rows'],args.reference_budget)
        link_once(prepared/'pools'/method,out/'pools'/method)
        link_once(source/'ji'/method/f'b{args.reference_budget}/remora',
                  out/'ji'/method/f'b{args.reference_budget}/remora')
    if (source/'data/base.cold').exists():
        link_once(source/'data/base.cold',out/'data/base.cold')
    class RatePipeline(Pipeline):
        def ji(self,method,bits,pools):
            config=self.out/'trainer-config.json'
            if not config.exists():
                save_json(config,self.c)
            dest=self.out/'ji'/method/f'b{bits}/remora'
            primary=self.runs[method]['paths']
            self.stage(f'remora-{method}-b{bits}-remora',
                [sys.executable,trainer,'--config',config,'--method',method,
                 '--model',primary['model'],'--codes',primary['codes'],
                 '--train-pool',pools['metric'],'--validation-pool',pools['select'],
                 '--bits',bits,'--variant','remora','--output',dest],
                [dest/'online.bin',dest/'side.bin',dest/'training.json'],
                [trainer,config,primary['model'],primary['codes'],pools['metric'],pools['select']])
    p=RatePipeline(c)
    for name in ('roar_primary_ji_search','roar_exp234_audit'):
        if not (Path(c['build_dir'])/name).is_file():
            raise ValueError('Missing frozen binary: '+name)
    ids=evaluation_nodes(c['base_rows'],c['residual_sample_size'],args.distortion_nodes,c['seed'])
    manifest=dict(version=1,source=str(source),source_experiment=args.source_experiment,
        prepared=str(prepared),audit_source=args.audit_source,
        config=c,distortion_nodes=args.distortion_nodes,distortion_sample_sha256=hashlib.sha256(ids.tobytes()).hexdigest(),
        raw_primary_dim=c['base_dim'],primary_configuration='unchanged frozen native primary',
        primary_graphs={m:file_id(p.runs[m]['paths']['graph']) for m in methods},
        common_teacher_ids=file_id(Path(args.audit_source)/'experiment3/raw/common.ids.ibin'),
        trainer=str(trainer),trainer_id=file_id(trainer),reference_budget=args.reference_budget,
        teacher_graph_role=audit_study['teacher_graph_role'],
        binary_ids={name:file_id(Path(c['build_dir'])/name)
                    for name in ('roar_primary_ji_search','roar_exp234_audit')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    study=out/'study.json'
    if study.exists() and json.loads(study.read_text())!=manifest:
        raise ValueError('Study configuration/input/runner changed; use a new output root')
    if not study.exists():
        save_json(study,manifest)
    status_path=out/'status.json'
    started=time.monotonic()
    def status(phase,**extra):
        save_json(status_path,dict(status='RUNNING',phase=phase,elapsed_s=time.monotonic()-started,**extra))
        print(f'[phase] {phase}; elapsed={time.monotonic()-started:.1f}s',flush=True)
    status('preflight')
    if not p.preflight():
        raise RuntimeError('Preflight failed; inspect '+str(out/'preflight.csv'))
    write_csv(out/'experiment5/distortion-sample.ids.csv',[dict(node_id=int(i)) for i in ids])
    try:
        if 'train' in phases:
            total=len(methods)*len(args.budgets)
            done=0
            for method in methods:
                for bits in args.budgets:
                    status('training',primary=method,side_bits=bits,models_done=done,models_total=total)
                    try:
                        check_refiner(out,method,c['base_rows'],bits)
                        print(f'[reuse] validated model {method}/b{bits}',flush=True)
                    except (OSError,ValueError,KeyError,struct.error):
                        p.ji(method,bits,{s:out/'pools'/method/s/'pools.npz' for s in ('metric','select')})
                        check_refiner(out,method,c['base_rows'],bits)
                    done+=1
                    elapsed=time.monotonic()-started
                    print(f'[progress] models={done}/{total}; elapsed/estimated_training_total='
                          f'{elapsed:.1f}s/{elapsed*total/done:.1f}s (estimate; timings/audit follow)',flush=True)
        for method in methods:
            for bits in args.budgets:
                check_refiner(out,method,c['base_rows'],bits)
        jobs=[dict(experiment=5,primary=m,variant='remora' if b else 'primary',side_bits=b,
                   beam=beam,rerank_depth=0,repeat=repeat)
              for m in methods for b in [0,*args.budgets] for beam in args.beams
              for repeat in range(1,args.repeats+1)]
        save_experiment_plan(out,jobs)
        if 'timing' in phases:
            status('timing',planned_measurements=len(jobs))
            Experiments(p).timings(jobs)
        metrics_path=out/'experiment5/quality/metrics.csv'
        quality=read_csv(metrics_path)
        if 'quality' in phases:
            status('quality')
            queries=np.asarray(matrix(out/'data/test.fbin'),np.float64)
            w=queries.T@queries/len(queries)
            quality=audit_quality(p,args,p.runs,ids,w,api)
        if 'report' in phases:
            status('report')
            complete=report(p,args,quality,api)
            save_json(status_path,dict(status='COMPLETE' if complete else 'INCOMPLETE',
                                      phase='done',elapsed_s=time.monotonic()-started))
            if not complete:
                raise RuntimeError('Incomplete rate-study results; report exposes missing points')
    except BaseException as exc:
        save_json(status_path,dict(status='FAILED',error=str(exc),elapsed_s=time.monotonic()-started))
        raise


if __name__=='__main__':
    main()
