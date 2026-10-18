#!/usr/bin/env python3
"""Frozen native-primary inversion/trajectory audit; never retrain or rebuild an index.

Traced searches are diagnostics, NOT new QPS measurements. End-to-end matched
Recall tables reuse the completed, uninstrumented experiment-1 measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import time

import numpy as np


def csr(path, queries):
    path = Path(path)
    with path.open('rb') as stream:
        magic, version, start, rows, count = struct.unpack('<8sIQIQ', stream.read(32))
    if (magic, version, start, rows) != (b'DRBVST01', 1, 0, queries):
        raise ValueError(f'Invalid CSR trace header: {path}')
    if path.stat().st_size != 32 + 8 * (rows + 1) + 4 * count:
        raise ValueError(f'Invalid CSR trace size: {path}')
    offsets = np.memmap(path, mode='r', dtype='<u8', offset=32, shape=(rows + 1,))
    if offsets[0] != 0 or offsets[-1] != count or np.any(offsets[1:] < offsets[:-1]):
        raise ValueError(f'Invalid CSR offsets: {path}')
    nodes = np.memmap(path, mode='r', dtype='<u4', offset=32 + 8 * (rows + 1), shape=(count,))
    return offsets, nodes


def coverage_rows(truth, returned, visited, retained, base_rows):
    """GT membership only: no FP32 reads and no approximate score comparison."""
    if truth.shape[0] != returned.shape[0] or min(truth.shape[1], returned.shape[1]) < 10:
        raise ValueError('Coverage/returned query count mismatch')
    bitmap = np.zeros(base_rows, dtype=np.bool_)
    records = []
    for q, (gt, answer) in enumerate(zip(truth[:, :10], returned[:, :10])):
        if len(np.unique(gt)) != 10 or len(np.unique(answer)) != 10:
            raise ValueError('Duplicate Top-10 IDs')
        if max(int(gt.max()), int(answer.max())) >= base_rows:
            raise ValueError('Top-10 ID out of range')
        bitmap[gt] = True
        counts, sizes = [], []
        for offsets, nodes in (visited, retained):
            ids = nodes[int(offsets[q]):int(offsets[q + 1])]
            if len(ids) and int(ids.max()) >= base_rows:
                raise ValueError('Trace node ID out of range')
            counts.append(int(np.count_nonzero(bitmap[ids])))
            sizes.append(len(ids))
        output = int(np.count_nonzero(bitmap[answer]))
        bitmap[gt] = False
        v, c = counts
        if not 0 <= output <= c <= v <= 10:
            raise ValueError('Visited/retained/output coverage containment failed')
        records.append(dict(query_id=q, visited=sizes[0], retained=sizes[1],
                            visited_hits=v, retained_hits=c, output_hits=output,
                            missed_hits=10-v, discarded_hits=v-c, misranked_hits=c-output))
    return records


def mean_coverage(records):
    n = len(records)
    if not n:
        raise ValueError('Empty coverage report')
    mean = lambda key: sum(float(r[key]) for r in records) / n
    return dict(queries=n, visited=mean('visited'), candidate_pool=mean('retained'),
                visited_coverage=mean('visited_hits')/10, retained_coverage=mean('retained_hits')/10,
                recall_at_10=mean('output_hits')/10, missed_loss=mean('missed_hits')/10,
                discarded_loss=mean('discarded_hits')/10, ranking_loss=mean('misranked_hits')/10)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')
    temporary.replace(path)


def hms(seconds):
    seconds = max(0, int(seconds))
    return f'{seconds//3600:02d}:{seconds//60%60:02d}:{seconds%60:02d}'


def draw_figures(out, methods, inversions, coverage):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print('[plots] matplotlib unavailable; CSV reports remain complete',flush=True)
        return
    names={'opq-pq':'OPQ-PQ','qaq':'QAQ','leanvec-ood':'LeanVec-OOD',
           'extended-rabitq':'Extended RaBitQ','saq':'SAQ'}
    plt.rcParams.update({'font.size':10,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
    if inversions:
        indexed={(r['primary'],r['variant'],r['metric']):r for r in inversions}
        metrics=['AdjacentInv-R50','AdjacentInv-R75','AdjacentInv-R100',
                 'BoundaryInv-K10-h5','BoundaryInv-K10-h10','ExpandInv','QueueInv','StopBoundaryProxyInv']
        figure,axes=plt.subplots(2,4,figsize=(16,7),constrained_layout=True)
        for ax,metric in zip(axes.flat,metrics):
            x=np.arange(len(methods))
            for variant,offset,color in [('primary',-.18,'#a3a3a3'),('remora',.18,'#2878b5')]:
                values=[100*float(indexed[m,variant,metric]['inversion_rate'])
                        if indexed[m,variant,metric]['status']=='OK' else np.nan for m in methods]
                ax.bar(x+offset,values,width=.36,label='Primary' if variant=='primary' else 'REMORA')
            ax.set_title(metric);ax.set_ylabel('Query-balanced inversion (%)')
            ax.set_xticks(x,[names[m] for m in methods],rotation=35,ha='right')
            ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
        axes.flat[0].legend(frameon=False)
        dest=out/'experiment3/figures';dest.mkdir(parents=True,exist_ok=True)
        for ext in ('svg','png'): figure.savefig(dest/f'inversions.{ext}',dpi=180)
        plt.close(figure)
    if coverage:
        for score,label in [('retained_coverage','Retained candidate coverage@10'),('recall_at_10','No-rerank Recall@10')]:
            figure,axes=plt.subplots(1,len(methods),figsize=(17,3.6),constrained_layout=True)
            for ax,method in zip(np.atleast_1d(axes),methods):
                for variant,style in [('primary','--'),('remora','-')]:
                    rows=sorted([r for r in coverage if r['primary']==method and r['variant']==variant],key=lambda r:int(r['beam']))
                    ax.plot([float(r[score]) for r in rows],[float(r['visited']) for r in rows],
                            style,marker='o',markersize=3,label='Primary' if variant=='primary' else 'REMORA')
                ax.set_title(names[method]);ax.set_xlabel(label);ax.set_ylabel('Mean visited nodes');ax.grid(alpha=.2)
            np.atleast_1d(axes)[0].legend(frameon=False)
            dest=out/'experiment4/figures';dest.mkdir(parents=True,exist_ok=True)
            for ext in ('svg','png'):figure.savefig(dest/f'visited-{score}.{ext}',dpi=180)
            plt.close(figure)


def user_tables(methods, inversions, fixed_rows):
    """CSV-ready versions of the user's two tables; percentages and actual targets."""
    indexed = {(r['primary'], r['variant'], r['metric']): r for r in inversions}
    keys = [('Adjacent_R50', 'AdjacentInv-R50'), ('Adjacent_R75', 'AdjacentInv-R75'),
            ('Adjacent_R100', 'AdjacentInv-R100'), ('Boundary_K10_h5', 'BoundaryInv-K10-h5'),
            ('Boundary_K10_h10', 'BoundaryInv-K10-h10'), ('ExpandInv', 'ExpandInv'),
            ('QueueInv', 'QueueInv'), ('StopInv', 'StopInv')]
    inv_rows = []
    for method in methods:
        for variant in ('primary', 'remora'):
            if not all((method, variant, key) in indexed for _, key in keys):
                continue
            row = dict(primary=method, variant=variant, unit='percent',
                       averaging='query-balanced; exact ties excluded')
            for field, key in keys:
                stat = indexed[method, variant, key]
                row[field] = 100*float(stat['inversion_rate']) if stat['status']=='OK' else 'N/A'
            inv_rows.append(row)
    lookup = {(r['primary'],r['variant'],float(r['target_recall'])): r for r in fixed_rows
              if int(r['fixed_rerank_depth'])==100}
    traversal = []
    for method in methods:
        for variant in ('primary', 'remora'):
            if not all((method,variant,t) in lookup for t in (.90,.95,.99)):
                continue
            row = dict(primary=method,variant=variant,rerank_depth=100,
                       selection='fastest fully repeated measured point with Recall >= target; no interpolation')
            for target, label in ((.90,'90'),(.95,'95'),(.99,'99')):
                point=lookup[method,variant,target]
                achieved=str(point['achieved']).lower()=='true'
                row['achieved_'+label] = achieved
                row['beam_'+label] = point['beam'] if achieved else 'NOT_REACHED'
                row['visited_'+label] = float(point['visited']) if achieved else ''
                row['actual_recall_'+label] = float(point['recall_at_10']) if achieved else ''
            traversal.append(row)
    return inv_rows, traversal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo', default='/home/xzl/REMORA')
    parser.add_argument('--source', required=True)
    parser.add_argument('--source-experiment', type=int, choices=(1,4), default=1,
                        help='Completed uninstrumented measurements to reuse; 4 supports a dedicated matched-Recall scan')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--teacher-graph', default='', help='Optional fixed topology for the common FP32 scorer teacher')
    parser.add_argument('--output', required=True)
    parser.add_argument('--beams', default='50,100,200,300,500,1000')
    parser.add_argument('--query-limit', type=int, default=10000)
    parser.add_argument('--phases', default='inversion,traversal,report')
    args = parser.parse_args()
    repo = Path(args.repo).resolve(strict=True)
    source = (repo / args.source).resolve(strict=True)
    out = (repo / args.output).resolve()
    if out == source or source in out.parents:
        raise ValueError('Keep completed experiment-1 results immutable; use a separate output root')
    phases = args.phases.split(',')
    if set(phases) - {'inversion', 'traversal', 'report'}:
        parser.error('Unknown phase')
    beams = sorted(set(map(int, args.beams.split(','))))
    if not beams or min(beams) < 10 or args.query_limit <= 0:
        parser.error('Invalid beam/query limit')
    sys.path[:0] = [str(repo/'ours/test/t2i-10M'), str(repo/'ours/scripts'),
                   str(repo/'ours/python'), str(repo/'ours/test/t2i-1M/python')]
    from protocol import (file_id, identity, matrix, read_csv, write_csv,
                          write_matrix, global_inversions, pair_statistics, INVERSION_FIELDS,
                          content_sha256, aggregate, matched_points)
    from pipeline import Pipeline
    from run_native100_experiment1 import check_refiner, link_once
    from scoring_contract import validate_native_summary
    from qvrr import select_runs

    frozen = json.loads((source/'configuration.json').read_text())
    c = dict(frozen['config'])
    if c['dataset'] != args.dataset or c['scoring_placement'] != 'traversal':
        raise ValueError('Dataset mismatch or terminal-only source: expected frozen REMORA traversal')
    bits_budget = c['side_bits']
    if bits_budget <= 0 or bits_budget % 8:
        raise ValueError('The frozen model must use whole-byte learned groups')
    source_query_count = c['test_query_limit']
    if args.query_limit > source_query_count:
        raise ValueError('Requested more queries than the frozen source experiment')
    ranks = [50,75,100,200]
    imported = select_runs(c['graph_package'], c['methods'])
    if args.teacher_graph and not Path(args.teacher_graph).is_file():
        raise ValueError('Explicit teacher graph does not exist')
    candidates = [(args.teacher_graph, 'explicit fixed topology; FP32 teacher scorer'),
                  (c['fp_graph'], 'existing FP32 graph; FP32 teacher scorer')]
    chosen = next(((Path(path).resolve(), role) for path, role in candidates
                   if path and Path(path).is_file()), None)
    if chosen is None:
        chosen = (Path(imported['opq-pq']['paths']['graph']).resolve(strict=True),
                  'existing OPQ-PQ graph; common FP32 teacher scorer, NOT an FP32-built graph')
    c['fp_graph'], teacher_role = str(chosen[0]), chosen[1]
    print(f'[teacher] {teacher_role}: {c["fp_graph"]}', flush=True)
    completed = json.loads((source/f'experiment{args.source_experiment}/summary.json').read_text())
    if completed['status'] != 'COMPLETE':
        raise ValueError('Source uninstrumented measurement scan must be complete before this audit starts')
    c.update(output_root=str(out), test_query_limit=args.query_limit, device='cpu',
             kmeans_device='cpu', geometric_encoder_device='cpu', ablation_variants=['remora'])
    os.sched_setaffinity(0, c['cpus'])
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(c['search_threads'])
    os.environ['REMORA_NATIVE_PRIMARY_READER'] = str(repo/'ours/runtime/native100-reader-v1/roar_native_primary_reader')
    for method in c['methods']:
        check_refiner(source, method, c['base_rows'], c['side_bits'])
    for binary in ('roar_primary_ji_search', 'roar_exp234_audit'):
        if not (Path(c['build_dir'])/binary).is_file():
            raise ValueError(f'Missing precompiled native binary: {binary}')
    with Path(c['fp_graph']).open('rb') as stream:
        entry, nodes = struct.unpack('<II', stream.read(8))
    if nodes != c['base_rows'] or entry >= nodes:
        raise ValueError('FP32 teacher graph header mismatch')
    out.mkdir(parents=True, exist_ok=True)
    for name in ('test.fbin', 'truth.ids.ibin', 'truth.scores.fbin', 'truth.metadata.json'):
        link_once(source/'data'/name, out/'data'/name)
    link_once(source/'ji', out/'ji')
    if (source/'data/base.cold').exists():
        link_once(source/'data/base.cold', out/'data/base.cold')
    truth = np.asarray(matrix(out/'data/truth.ids.ibin', '<u4')[:args.query_limit])
    query = matrix(out/'data/test.fbin')
    base = matrix(c['base'])
    if (truth.shape != (args.query_limit, 200) or query.shape != (source_query_count, c['base_dim'])
            or base.shape != (c['base_rows'], c['base_dim'])):
        raise ValueError('Raw data/full-database Top-200 truth shape mismatch')
    expected = matrix(out/'data/truth.scores.fbin')
    max_error = 0.0
    for q in np.unique(np.linspace(0, args.query_limit-1, min(args.query_limit,16)).astype(int)):
        actual = np.asarray(base[truth[q,:200]]) @ query[q]
        max_error = max(max_error, float(np.max(abs(actual-expected[q,:200]))))
    if max_error > 2e-5:
        raise ValueError('Raw query/GT score mismatch')
    manifest = dict(version=1, dataset=c['dataset'], source=str(source), queries=args.query_limit,
        methods=c['methods'], side_bits=bits_budget, base_rows=c['base_rows'], base_dim=c['base_dim'], beams=beams, threads=c['search_threads'], cpus=c['cpus'],
        memory_budget_bytes=c['memory_budget_bytes'], scoring_placement='traversal',
        teacher_storage='FP32 mmap, diagnostic only; excluded from QPS', teacher_graph=c['fp_graph'],
        teacher_graph_role=teacher_role,
        inversion_ranks=ranks, routing_beam=c['routing_beam'],
        routing_pairs_per_event=c['routing_pairs_per_source'],
        stop_inversion='N/A: queue exhaustion; stop-boundary-proxy is separately labelled',
        trace_timing='diagnostic only; not used as new QPS or latency measurements',
        performance_source=f'completed uninstrumented experiment {args.source_experiment}; NVMe O_DIRECT, L<=200',
        source_experiment=args.source_experiment,
        test_sha256=content_sha256(out/'data/test.fbin'),
        truth_sha256=content_sha256(out/'data/truth.ids.ibin'),
        native_binaries={b:file_id(Path(c['build_dir'])/b) for b in ('roar_primary_ji_search','roar_exp234_audit')},
        runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    manifest_path=out/'study.json'
    if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
        raise ValueError('Study protocol/source changed; choose a new output directory')
    save_json(manifest_path, manifest)

    class DiagnosticPipeline(Pipeline):
        def branch(self, method, variant, bits):
            branch = super().branch(method, variant, bits)
            if method == 'full-precision':
                branch.update(codes=c['base'], implementation=teacher_role+'; diagnostic only, not a performance baseline')
            return branch

        def native_args(self, branch, variant):
            values = super().native_args(branch, variant)
            if branch['kind'] == 'fp32':
                values[values.index('--fp32-storage')+1] = 'mmap'
            return values

    p = DiagnosticPipeline(c)
    source_rows=read_csv(source/f'experiment{args.source_experiment}/results.csv')
    references={}
    for row in source_rows:
        if (row['status']=='OK' and int(row['rerank_depth'])==0
                and int(row['queries'])==args.query_limit):
            references.setdefault((row['primary'],row['variant'],int(row['beam'])),row)
    started = time.monotonic()
    total_steps = (12 if 'inversion' in phases else 0) + (len(beams)*len(c['methods'])*2 if 'traversal' in phases else 0)
    done = 0

    def progress(label):
        nonlocal done
        done += 1
        elapsed = time.monotonic()-started
        estimated = elapsed * total_steps / max(1,done)
        print(f'[progress] {done}/{total_steps} {label}; elapsed/estimated_total={hms(elapsed)}/{hms(estimated)} (estimate)',flush=True)

    if 'inversion' in phases:
        dest=out/'experiment3/raw'; dest.mkdir(parents=True,exist_ok=True)
        teacher=p.branch('full-precision','fp32',0); prefix=dest/'teacher'; pairs_path=dest/'routing-pairs.csv'
        command=[Path(c['build_dir'])/'roar_primary_ji_search',*p.native_args(teacher,'fp32'),
            '--graph',teacher['graph'],'--queries',out/'data/test.fbin','--query-limit',args.query_limit,
            '--topk',10,'--beam',c['routing_beam'],'--threads',c['search_threads'],'--warmup-queries',0,
            '--out-prefix',prefix,'--packed-graph','--routing-pairs-csv',pairs_path,
            '--routing-pairs-per-source',c['routing_pairs_per_source'],'--pdf-routing-tags',
            '--memory-budget-bytes',c['memory_budget_bytes']]
        p.stage('routing-teacher',command,[pairs_path,Path(str(prefix)+'.summary.json')],
                [teacher['graph'],c['base'],out/'data/test.fbin'],budget=c['memory_budget_bytes'])
        progress('common FP32 routing teacher')
        pairs=read_csv(pairs_path); per_query=[set() for _ in range(args.query_limit)]
        for pair in pairs:
            q=int(pair['query_id'])
            if not 0 <= q < args.query_limit: raise ValueError('Routing query ID out of range')
            per_query[q].update((int(pair['u_id']),int(pair['v_id'])))
        ordered=[list(map(int,row[:200]))+sorted(extra-set(map(int,row[:200]))) for row,extra in zip(truth,per_query)]
        width=max(map(len,ordered)); audit_ids=np.empty((args.query_limit,width),dtype=np.uint32)
        lookups=[]
        for q,row in enumerate(ordered):
            audit_ids[q]=row+[row[0]]*(width-len(row))
            lookups.append({node:i for i,node in enumerate(row)})
        id_path=dest/'common.ids.ibin'
        if not id_path.exists() or not np.array_equal(matrix(id_path,'<u4'),audit_ids): write_matrix(id_path,audit_ids,'<u4')
        teacher_id=identity(dict(graph=file_id(teacher['graph']),pairs=file_id(pairs_path),ids=file_id(id_path)))
        indices={}
        for event,name in [('exp','ExpandInv'),('queue','QueueInv'),('stop-boundary-proxy','StopBoundaryProxyInv')]:
            local=[v for v in pairs if v['event_type']==event]
            qids=np.asarray([int(v['query_id']) for v in local],dtype=np.int64)
            left=np.asarray([lookups[int(v['query_id'])][int(v['u_id'])] for v in local],dtype=np.int64)
            right=np.asarray([lookups[int(v['query_id'])][int(v['v_id'])] for v in local],dtype=np.int64)
            indices[name]=(qids,left,right)
        branches=[('full-precision','fp32',0)]+[(m,v,0 if v=='primary' else bits_budget) for m in c['methods'] for v in ('primary','remora')]
        exact=None; records=[]
        for method,variant,bits in branches:
            branch=p.branch(method,variant,bits); prefix=dest/f'{method}-{variant}-b{bits}'
            command=[Path(c['build_dir'])/'roar_exp234_audit',*p.native_args(branch,variant),
                     '--graph',teacher['graph'],'--queries',out/'data/test.fbin','--audit-ids',id_path,'--out-prefix',prefix]
            deps=[branch['model'],branch['codes'],id_path,out/'data/test.fbin']
            if bits: deps += [branch['ji_model'],branch['ji_codes']]
            p.stage(f'inversion-{method}-{variant}',command,[Path(str(prefix)+'.scores.fbin')],deps,budget=c['memory_budget_bytes'])
            scores=np.asarray(matrix(str(prefix)+'.scores.fbin'))
            if exact is None:
                exact=scores
                if not np.allclose(exact[:,:200],expected[:args.query_limit,:200],atol=2e-5,rtol=0):
                    raise ValueError('Native FP32 teacher score/GT mismatch')
            metrics=global_inversions(exact[:,:200],scores[:,:200],ranks)
            for name,(qids,left,right) in indices.items():
                metrics.append(dict(metric=name,**pair_statistics(exact[qids,left]-exact[qids,right],scores[qids,left]-scores[qids,right],qids,args.query_limit)))
            metrics.append(dict(metric='StopInv',status='NOT_APPLICABLE',queries=args.query_limit,
                                note='Queue-exhaustion termination has no score-based stopping comparison'))
            for record in metrics:
                row=dict(experiment=3,dataset=c['dataset'],primary=method,variant=variant,side_bits=bits,
                    status='OK',teacher_id=teacher_id,representation_id=identity({str(v):file_id(v) for v in deps}),
                    pair_selection='full database FP32 Top-200 and common fixed-topology FP32 teacher: '+teacher_role,
                    note='query-balanced; exact ties excluded; approximate ties count as sign mismatch')
                row.update(record); records.append(row)
            write_csv(out/'experiment3/inversions.csv',records,INVERSION_FIELDS)
            progress(f'inversions {method}/{variant}')
        del scores,exact,pairs,audit_ids,lookups,per_query,ordered

    if 'traversal' in phases:
        summaries=[]
        for method in c['methods']:
            for variant in ('primary','remora'):
                bits=0 if variant=='primary' else bits_budget
                branch=p.branch(method,variant,bits)
                for beam in beams:
                    dest=out/'experiment4/raw'/method/variant/f'b{beam}'; dest.mkdir(parents=True,exist_ok=True)
                    completion=dest/'coverage.complete.json'; row_path=dest/'coverage.csv'
                    fingerprint=identity(dict(study=manifest,method=method,variant=variant,beam=beam,
                        files={str(v):file_id(v) for v in [branch['graph'],branch['model'],branch['codes'],
                            *([branch['ji_model'],branch['ji_codes']] if bits else [])]}))
                    if completion.exists() and row_path.exists() and json.loads(completion.read_text()).get('fingerprint')==fingerprint:
                        records=read_csv(row_path)
                        print(f'[reuse] trajectory {method}/{variant}/b{beam}',flush=True)
                    else:
                        prefix=dest/'search'
                        with tempfile.TemporaryDirectory(prefix='remora-exp34-trace-') as scratch:
                            vp=Path(scratch)/'visited.csr'; cp=Path(scratch)/'retained.csr'
                            command=[Path(c['build_dir'])/'roar_primary_ji_search',*p.native_args(branch,variant),
                                '--graph',branch['graph'],'--queries',out/'data/test.fbin','--query-limit',args.query_limit,
                                '--topk',10,'--beam',beam,'--threads',c['search_threads'],'--warmup-queries',0,
                                '--out-prefix',prefix,'--packed-graph','--visited-csr',vp,'--retained-csr',cp,
                                '--memory-budget-bytes',c['memory_budget_bytes']]
                            if bits and c['lazy_ji']: command += ['--lazy-ji']
                            p.stage(f'trace-{method}-{variant}-b{beam}',command,
                                [vp,cp,Path(str(prefix)+'.summary.json'),Path(str(prefix)+'.ids.ibin')],
                                [branch['graph'],branch['model'],branch['codes'],out/'data/test.fbin',
                                 *([branch['ji_model'],branch['ji_codes']] if bits else [])],budget=c['memory_budget_bytes'])
                            native=json.loads(Path(str(prefix)+'.summary.json').read_text())
                            validate_native_summary(native,variant)
                            returned=matrix(str(prefix)+'.ids.ibin','<u4')
                            reference=references.get((method,variant,beam))
                            if reference:
                                reference_path=Path(reference['directory'])/(reference.get('measurement_point_id') or reference['point_id'])
                                reference_ids=matrix(str(reference_path)+'.ids.ibin','<u4')
                                if not np.array_equal(returned,reference_ids):
                                    raise ValueError('Diagnostic trace output differs from frozen experiment-1 output')
                            records=coverage_rows(truth,returned,csr(vp,args.query_limit),csr(cp,args.query_limit),c['base_rows'])
                            write_csv(row_path,records)
                        save_json(completion,dict(fingerprint=fingerprint,
                            scratch_traces='discarded after saving per-query coverage; original data/indexes untouched'))
                    if len(records)!=args.query_limit: raise ValueError('Truncated per-query coverage CSV')
                    summaries.append(dict(primary=method,variant=variant,side_bits=bits,beam=beam,
                        **mean_coverage(records),timing_usage='diagnostic traces; QPS excluded'))
                    write_csv(out/'experiment4/coverage.csv',summaries)
                    progress(f'trajectory {method}/{variant}/b{beam}')

    inversion=read_csv(out/'experiment3/inversions.csv')
    if inversion:
        comparisons=[]
        indexed={(r['primary'],r['variant'],r['metric']):r for r in inversion}
        for method in c['methods']:
            for metric in ['AdjacentInv-R50','AdjacentInv-R75','AdjacentInv-R100','AdjacentInv-R200','BoundaryInv-K10-h5','BoundaryInv-K10-h10','ExpandInv','QueueInv','StopBoundaryProxyInv','StopInv']:
                if (method,'primary',metric) not in indexed or (method,'remora',metric) not in indexed:
                    continue
                a=indexed[(method,'primary',metric)]; b=indexed[(method,'remora',metric)]
                valid=a['status']=='OK' and b['status']=='OK'
                comparisons.append(dict(primary=method,metric=metric,status='OK' if valid else 'NOT_APPLICABLE',
                    primary_inversion_rate=a['inversion_rate'],remora_inversion_rate=b['inversion_rate'],
                    reduction_pp=100*(float(a['inversion_rate'])-float(b['inversion_rate'])) if valid else '',
                    primary_pairs=a['pairs'],remora_pairs=b['pairs'],teacher_id=a['teacher_id']))
        write_csv(out/'experiment3/comparison.csv',comparisons)

    measurements=[r for r in source_rows if r['status']=='OK'
                  and int(r['queries'])==args.query_limit and int(r['rerank_depth'])<=200]
    for r in measurements:
        if r['scoring_placement'] != ('primary' if r['variant']=='primary' else 'traversal'):
            raise ValueError('Terminal-only data cannot enter the traversal report')
    front=aggregate(measurements,c['repeats'])
    fields=['primary','variant','mode','target_recall','achieved','selection','beam','rerank_depth','recall_at_10',
            'visited','expanded','comparisons','qps','mean_service_latency_ms','p50_ms','p95_ms','p99_ms']
    if front:
        matched=matched_points(front,(.90,.95,.99))
        write_csv(out/'experiment4/matched-recall.csv',matched,fields)
        fixed=[]
        for depth in (50,100,200):
            fixed.extend(dict(r,fixed_rerank_depth=depth) for r in matched_points([v for v in front if int(v['rerank_depth'])==depth],(.90,.95,.99)))
        write_csv(out/'experiment4/matched-recall-fixed-depth.csv',fixed,fields+['fixed_rerank_depth'])
    coverage=read_csv(out/'experiment4/coverage.csv')
    candidate_points=[]
    for method in c['methods']:
        for variant in ('primary','remora'):
            local=[r for r in coverage if r['primary']==method and r['variant']==variant]
            for target in (.90,.95,.99):
                feasible=[r for r in local if float(r['retained_coverage'])>=target]
                best=min(feasible,key=lambda r:(float(r['visited']),int(r['beam']))) if feasible else None
                candidate_points.append(dict(primary=method,variant=variant,target_coverage=target,achieved=bool(best),
                    beam=best['beam'] if best else '',visited=best['visited'] if best else '',
                    retained_coverage=best['retained_coverage'] if best else '',
                    selection='minimum observed visits at retained GT coverage; no FP rerank; no extrapolation'))
    write_csv(out/'experiment4/matched-candidate-coverage.csv',candidate_points)
    draw_figures(out,c['methods'],inversion,coverage)
    fixed_rows = read_csv(out/'experiment4/matched-recall-fixed-depth.csv')
    user_inversion, user_traversal = user_tables(c['methods'], inversion, fixed_rows)
    if user_inversion:
        write_csv(out/'report/inversion-table.csv', user_inversion)
    if user_traversal:
        write_csv(out/'report/traversal-table-l100.csv', user_traversal)
    expected_inversions = (1+2*len(c['methods']))*(len(ranks)+6)
    complete=len(inversion)==expected_inversions and len(coverage)==len(beams)*2*len(c['methods'])
    save_json(out/'summary.json',dict(status='COMPLETE' if complete else 'PARTIAL',dataset=c['dataset'],methods=c['methods'],queries=args.query_limit,
        bits=bits_budget,threads=c['search_threads'],scoring_placement='traversal',graphs_rebuilt=False,models_retrained=False,
        inversion_complete=len(inversion)==expected_inversions,inversion_rows=len(inversion),expected_inversion_rows=expected_inversions,
        trajectory_points=len(coverage),expected_trajectory_points=len(beams)*2*len(c['methods']),
        traversal_complete=len(coverage)==len(beams)*2*len(c['methods']),matched_performance_measurements_reused=len(measurements),
        teacher_fp32_storage='mmap for diagnostics only; excluded from QPS', teacher_graph_role=teacher_role,
        matched_performance_fp32_storage=f'NVMe O_DIRECT, own queue Top-L, L<=200; original experiment-{args.source_experiment} measurements',
        elapsed_this_invocation_s=time.monotonic()-started,
        note='No-rerank unreachable targets remain marked unavailable; StopInv is N/A, not the boundary proxy'))
    print(f'[{"complete" if complete else "partial"}] experiment 3/4 reports: {out}; elapsed={hms(time.monotonic()-started)}',flush=True)


if __name__ == '__main__':
    main()
