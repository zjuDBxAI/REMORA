#!/usr/bin/env python3
"""Run current REMORA experiments 3/4/5 on immutable legacy T2I-1M primaries.

The 768-bit historical primary configuration is explicitly labelled. No graph
builder, primary quantizer or historical result writer is invoked. Experiment 4
collects its own uninstrumented finite-rerank measurements, not a fake completed
experiment 1. Training and performance measurements execute sequentially.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import time

import numpy as np


def run(command):
    print('[command] '+ ' '.join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), check=True)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError('Immutable configuration changed: '+str(path))
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix+'.tmp')
    temp.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temp.replace(path)


def import_legacy(args, root):
    from protocol import matrix, file_id
    from qvrr import Primary
    from run_frozen_experiment5 import save_json
    legacy = Path(args.legacy).resolve(strict=True)
    data = Path(args.data).resolve(strict=True)
    base = matrix(data/'base.1M.fbin')
    train = matrix(data/'query.train.30k.fbin')
    test = matrix(data/'query.test.10k.fbin')
    if base.shape != (1000000,200) or train.shape != (30000,200) or test.shape != (10000,200):
        raise ValueError('This entry is for the complete 1M x 200 T2I dataset')
    if not json.loads((data/'dataset.json').read_text())['test_queries_independent']:
        raise ValueError('Test-query independence not declared by source dataset')
    ids = np.unique(np.linspace(0, len(base)-1, 1024).astype(np.int64))
    report = dict(status='files-and-sampled-base-order-verified', dataset='t2i-1M',
                  legacy_primary=True, primary_nominal_bits=768, raw_dim=200,
                  primary_config_note='Historical 768-bit suite; OPQ/QAQ scorer dimension 192; NOT the new 800-bit full-200D setting',
                  sampled_nodes=len(ids), base=str(data/'base.1M.fbin'), methods={})
    for method in args.methods:
        manifest_path = legacy/'primaries/manifests'/f'{method}.json'
        manifest = json.loads(manifest_path.read_text())
        metadata = manifest['metadata']; files = metadata.get('files',{})
        model = metadata.get('model') or files.get('model') or files.get('quantized_codes')
        codes = metadata.get('codes') or files.get('quantized_codes')
        if not model or not codes:
            raise ValueError('Missing legacy primary paths: '+method)
        graph = legacy/'graphs'/method/'code-native/primary/graph.roar.index'
        construction_path = graph.parent/'construction.json'
        construction = json.loads(construction_path.read_text())
        if (construction['nodes'],construction['original_dim'],construction['query_start'],
                construction['build_queries']) != (1000000,200,20000,10000):
            raise ValueError('Unexpected construction dataset/query split: '+method)
        if not construction['code_native_construction'] or construction['representation']!='primary':
            raise ValueError('Expected primary-only code-native graph')
        for key, actual in [('primary_model',model),('primary_codes',codes),('graph',str(graph))]:
            declared = Path(construction['files'][key])
            if not declared.is_absolute(): declared=Path(args.roar_repo)/declared
            if declared.resolve(strict=True) != Path(actual).resolve(strict=True):
                raise ValueError('Graph/primary provenance mismatch: '+method+'/'+key)
        with graph.open('rb') as f:
            entry, nodes = struct.unpack('<II',f.read(8))
        if nodes != len(base) or entry >= nodes or graph.stat().st_size != construction['graph_bytes']:
            raise ValueError('Graph header/size mismatch: '+method)
        primary = Primary(model,codes)
        try:
            if (primary.nodes,primary.dim,primary.record_bytes)!=(len(base),200,int(manifest['physical_bytes_per_node'])):
                raise ValueError('Primary header/storage mismatch: '+method)
            reconstructed=primary.reconstruct_scorer(ids)
            original=matrix(manifest['reconstruction_fbin'])
            reconstruction_error=float(np.max(abs(reconstructed-original[ids])))
            raw=primary.reconstruct_raw(ids)
            linear_error=max(float(np.max(abs(primary.score(test[q],ids)-raw@test[q]))) for q in (0,9999))
            original_query=matrix(manifest['query_fbin'])
            query_map_error=max(float(np.max(abs(test[q]@primary.projection-original_query[30000+q]))) for q in (0,9999))
            transformed=matrix(manifest['base_fbin'])
            base_order_error=float(np.max(abs(base[ids]@primary.projection-transformed[ids])))
            if max(reconstruction_error,linear_error,query_map_error,base_order_error)>2e-5:
                raise ValueError(f'Legacy raw/base-order/scoring parity failed: {method}: '
                                 f'{reconstruction_error}/{linear_error}/{query_map_error}/{base_order_error}')
            if manifest['nominal_code_bits_per_node'] != 768:
                raise ValueError('Do not mix a different legacy primary bitrate')
            report['methods'][method]=dict(graph=str(graph),paths=dict(model=model,codes=codes),
                quantization=dict(physical_bytes_per_node=primary.record_bytes,nominal_code_bits_per_node=768,
                                  implementation=manifest['implementation'],raw_dim=200,scorer_dim=primary.scorer_dim),
                artifacts={str(p):file_id(p) for p in [manifest_path,construction_path,graph,model,codes]},
                parity=dict(reconstruction_max_error=reconstruction_error,raw_linear_max_error=linear_error,
                            test_query_map_max_error=query_map_error,base_order_max_error=base_order_error))
        finally:
            primary.close()
        print(f'[primary verified] {method}: 768 bits, {manifest["physical_bytes_per_node"]} B/node; graph reused',flush=True)
    target=root/'inputs/imported-primary.json'
    frozen_json(target,report)
    gt_ids=matrix(data/'gt.test.top200.ibin','<u4')
    gt_scores=matrix(data/'gt.test.top200.scores.fbin')
    if gt_ids.shape != (10000,200) or gt_scores.shape != gt_ids.shape:
        raise ValueError('Full-database Top-200 ground truth missing')
    if np.any(gt_ids >= len(base)) or np.any(np.diff(gt_scores,axis=1)>1e-6):
        raise ValueError('Invalid ground-truth IDs/score order')
    gt_error=max(float(np.max(abs(base[gt_ids[q]]@test[q]-gt_scores[q])))
                 for q in np.linspace(0,9999,16).astype(int))
    if gt_error>2e-5: raise ValueError('Ground-truth/raw-query score mismatch')
    combined=root/'inputs/gt.test.top200.combined.ibin'
    gt_fingerprint={str(p):file_id(p) for p in [data/'gt.test.top200.ibin',data/'gt.test.top200.scores.fbin']}
    frozen_json(root/'inputs/gt-inputs.json',gt_fingerprint)
    if not combined.exists():
        temp=combined.with_suffix('.tmp')
        with temp.open('wb') as f:
            f.write(struct.pack('<II',*gt_ids.shape))
            f.write(np.asarray(gt_ids,dtype='<u4').tobytes())
            f.write(np.asarray(gt_scores,dtype='<f4').tobytes())
        temp.replace(combined)
    expected_size=8+8*gt_ids.size
    if combined.stat().st_size!=expected_size:
        raise ValueError('Combined truth size mismatch')
    frozen_json(root/'inputs/gt-verified.json',dict(inputs=gt_fingerprint,combined_id=file_id(combined),
        full_database=True,base_rows=len(base),max_sampled_score_error=gt_error,score='raw-inner-product'))
    fp_graph=legacy/'graphs/full-precision/fp32/graph.roar.index'
    if not fp_graph.is_file(): raise ValueError('Existing common FP32 teacher graph missing')
    return target,combined,fp_graph


def native_score_check(p):
    from protocol import matrix,write_matrix,write_csv
    from qvrr import Primary
    ids=np.random.default_rng(p.c['seed']).integers(0,p.c['base_rows'],(8,128),dtype=np.uint32)
    dest=p.out/'smoke';dest.mkdir(exist_ok=True)
    id_path=dest/'ids.ibin'
    if not id_path.exists(): write_matrix(id_path,ids,'<u4')
    if not np.array_equal(matrix(id_path,'<u4'),ids): raise ValueError('Frozen smoke IDs changed')
    query=matrix(p.out/'data/test.fbin'); rows=[]
    for method in p.c['methods']:
        branch=p.branch(method,'primary',0);prefix=dest/method
        score_path=Path(str(prefix)+'.scores.fbin')
        p.stage('native-score-check-'+method,[Path(p.c['build_dir'])/'roar_exp234_audit',
            *p.native_args(branch,'primary'),'--graph',branch['graph'],'--queries',p.out/'data/test.fbin',
            '--audit-ids',id_path,'--out-prefix',prefix],[score_path],
            [branch['model'],branch['codes'],branch['graph'],id_path,p.out/'data/test.fbin'])
        actual=matrix(score_path);primary=Primary(branch['model'],branch['codes'])
        try:
            expected=np.asarray([primary.score(query[q],ids[q]) for q in range(8)])
            error=float(np.max(abs(actual-expected)))
            if error>2e-5: raise ValueError('Native scorer mismatch: '+method)
            rows.append(dict(primary=method,native_decoder_max_score_error=error,status='PASS',
                             timing_usage='correctness only; excluded from performance'))
        finally: primary.close()
    write_csv(dest/'score-parity.csv',rows)
    print('[PASS] all five primary native scorers / complete raw-score decoders',flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repo',default='/home/xzl/REMORA')
    parser.add_argument('--roar-repo',default='/home/xzl/RoarGraph')
    parser.add_argument('--legacy',default='/home/xzl/RoarGraph/ours/outputs/t2i-1M/ji-pure-suite-five-primary-v3')
    parser.add_argument('--data',default='/home/xzl/RoarGraph/data/t2i-1M')
    parser.add_argument('--output',default='outputs/t2i-1M/remora-exp345-b64-v1')
    parser.add_argument('--gpu',type=int,default=2)
    parser.add_argument('--repeats',type=int,default=3)
    parser.add_argument('--methods',default='opq-pq,qaq,leanvec-ood,extended-rabitq,saq')
    parser.add_argument('--beams',default='50,100,200,300,500,1000')
    parser.add_argument('--phases',default='prepare,measure,diagnostics,rate')
    parser.add_argument('--cold-store',default='/tmp/xzl-remora-t2i1m-exp345-v1/base.1M.cold.bin')
    args=parser.parse_args(); args.methods=args.methods.split(',')
    phases=args.phases.split(',');beams=sorted(set(map(int,args.beams.split(','))))
    if (set(phases)-{'check','prepare','measure','diagnostics','rate'} or args.repeats<1
            or min(beams)<10 or len(set(args.methods))!=len(args.methods)):
        parser.error('Invalid phases/repeats/beams/methods')
    repo=Path(args.repo).resolve(strict=True)
    root=(repo/args.output).resolve();root.mkdir(parents=True,exist_ok=True)
    lock=(root/'workflow.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    sys.path[:0]=[str(repo/'ours/test/t2i-10M'),str(repo/'ours/scripts'),str(repo/'ours/python'),
                  str(repo/'ours/test/t2i-1M/python')]
    os.environ['REMORA_NATIVE_PRIMARY_READER']=str(repo/'ours/runtime/native100-reader-v1/roar_native_primary_reader')
    os.environ['REMORA_PRIMARY_DECODE_THREADS']='8'
    from protocol import load_config,read_csv,write_csv,aggregate,file_id
    from pipeline import Pipeline
    from run import Experiments,save_experiment_plan
    from run_native100_experiment1 import check_refiner,link_once,check_cold_store,pilot
    from run_frozen_experiment5 import save_json
    started=time.monotonic()
    def progress(label,**extra):
        elapsed=time.monotonic()-started
        save_json(root/'status.json',dict(status='RUNNING',phase=label,elapsed_s=elapsed,**extra))
        print(f'[phase] {label}; elapsed={elapsed:.1f}s',flush=True)
    progress('verify immutable 1M data/primaries')
    package,combined,fp_graph=import_legacy(args,root)
    config=dict(dataset='t2i-1M',base=str(Path(args.data)/'base.1M.fbin'),
        train_queries=str(Path(args.data)/'query.train.30k.fbin'),test_queries=str(Path(args.data)/'query.test.10k.fbin'),
        base_rows=1000000,base_dim=200,graph_package=str(package),combined_ground_truth=str(combined),
        fp_graph=str(fp_graph),build_dir=str(repo/'ours/build-relative-rate-v1'),
        output_root=str(root/'prepared'),primary_budget_bits=768,primary_dim=192,pq_groups=96,
        metric_query_start=10000,metric_query_limit=8000,select_query_start=18000,select_query_limit=2000,
        build_query_start=20000,build_query_limit=10000,test_query_limit=10000,
        pool_train_queries=4096,pool_validation_queries=1024,methods=args.methods,
        pool_beams=[50,75,100,125,150,200],side_bits=64,budgets=[32,64,96,128],
        centered_epochs=4,recode_nodes=300000,ablation_variants=['remora'],deploy_variant='remora',
        device=f'cuda:{args.gpu}',kmeans_device=f'cuda:{args.gpu}',geometric_encoder_device=f'cuda:{args.gpu}',
        geometric_encoder='faiss',search_threads=8,build_threads=8,trainer_threads=8,compile_jobs=8,
        cpus=list(range(16,24)),scoring_placement='traversal',include_fp32_reference=False,lazy_ji=False,
        performance_sampling='dense',beams=beams,bit_budget_beams=[100,200],rerank_depths=[100,200],
        repeats=args.repeats,rerank_storage='direct',fp32_storage='direct',cold_store=args.cold_store,
        combined_ground_truth_score_kind='inner-product',seed=20261007,
        memory_budget_bytes=256000000000)
    config_path=root/'prepare.config.json';frozen_json(config_path,config)
    c=load_config(config_path);os.sched_setaffinity(0,c['cpus'])
    study=dict(dataset='t2i-1M',experiments=[3,4,5],primary_bits=768,side_bits=[32,64,96,128],
        relative_default_multipliers=[.5,1,1.5,2],nominal_primary_fraction=[b/768 for b in [32,64,96,128]],
        primary_configuration='Historical matched-768-bit native suite, NOT new 800-bit T2I-10M primary',
        scoring_placement='traversal',graph_rebuild=False,primary_reencoding=False,
        config=config,runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        binary_ids={b:file_id(Path(c['build_dir'])/b) for b in ['roar_primary_ji_search','roar_exp234_audit']})
    frozen_json(root/'study.json',study)
    p=Pipeline(c)
    if not p.preflight(): raise RuntimeError('Preflight failed; inspect '+str(p.out/'preflight.csv'))
    p.data();p.ground_truth();native_score_check(p)
    if 'prepare' in phases:
        for index,method in enumerate(c['methods']):
            progress('train 64-bit REMORA',primary=method,models_done=index,models_total=len(c['methods']))
            try:
                check_refiner(p.out,method,c['base_rows'],64)
                print('[reuse] validated 64-bit '+method,flush=True)
            except (OSError,ValueError,KeyError,struct.error):
                pools=p.pools(method);p.ji(method,64,pools)
                check_refiner(p.out,method,c['base_rows'],64)
    if 'measure' in phases:
        for method in c['methods']:check_refiner(p.out,method,c['base_rows'],64)
        progress('uninstrumented experiment-4 matched-Recall measurements')
        cold=Path(args.cold_store)
        if not cold.exists():
            original=Path(args.legacy)/'cold-storage/base.1M.cold.bin'
            check_cold_store(original,c['base'])
            cold.parent.mkdir(parents=True,exist_ok=True)
            run(['cp','--no-clobber',str(original),str(cold)])
        check_cold_store(cold,c['base'])
        performance=root/'performance';performance.mkdir(exist_ok=True)
        pc=dict(c,output_root=str(performance));pp=Pipeline(pc)
        (performance/'logs').mkdir(exist_ok=True)
        for name in ['test.fbin','truth.ids.ibin','truth.scores.fbin','truth.metadata.json']:
            link_once(p.out/'data'/name,performance/'data'/name)
        link_once(p.out/'ji',performance/'ji');link_once(cold,performance/'data/base.cold')
        parity=performance/'experiment4/storage-parity.complete.json'
        if not parity.exists():
            pilot(pc,pp.runs,performance)
            frozen_json(parity,dict(status='PASS',timing_usage='correctness only',
                                   binary=file_id(Path(c['build_dir'])/'roar_primary_ji_search')))
        jobs=[dict(experiment=4,primary=m,variant=v,side_bits=0 if v=='primary' else 64,
            beam=b,rerank_depth=l,repeat=r) for m in c['methods'] for v in ['primary','remora']
            for b in beams for l in [0,100,200] if l<=b for r in range(1,c['repeats']+1)]
        save_experiment_plan(performance,jobs);Experiments(pp).timings(jobs)
        latest={r['point_id']:r for r in read_csv(performance/'experiment4/results.csv')}
        complete=len(latest)==len(jobs) and all(r['status']=='OK' for r in latest.values())
        write_csv(performance/'experiment4/aggregate.csv',aggregate(list(latest.values()),c['repeats']))
        save_json(performance/'experiment4/summary.json',dict(status='COMPLETE' if complete else 'INCOMPLETE',
            experiment=4,role='dedicated uninstrumented matched-Recall grid; NOT full experiment-1 study',
            successful_measurements=sum(r['status']=='OK' for r in latest.values()),expected_measurements=len(jobs),
            scoring_placement='traversal',queries=10000,threads=8,repeats=c['repeats'],beams=beams,
            rerank_depths=[0,100,200],rerank_storage='NVMe O_DIRECT',timing_scope='shared host; affinity not exclusive'))
        if not complete: raise RuntimeError('Experiment-4 measurement grid incomplete')
    if 'diagnostics' in phases:
        progress('experiment 3 inversions and experiment 4 free-running coverage')
        run([sys.executable,'-u',repo/'ours/scripts/run_frozen_experiments34.py',
             '--repo',repo,'--source',root/'performance','--source-experiment',4,'--dataset','t2i-1M',
             '--output',root/'exp34','--beams',','.join(map(str,beams)),'--query-limit',10000])
    if 'rate' in phases:
        progress('experiment 5 independent 32/64/96/128-bit training and testing')
        run([sys.executable,'-u',repo/'ours/scripts/run_frozen_experiment5.py',
            '--repo',repo,'--dataset','t2i-1M','--source',root/'performance','--source-experiment',4,
            '--prepared',p.out,'--audit-source',root/'exp34','--output',root/'exp5',
            '--methods',','.join(args.methods),'--budgets','32,64,96,128','--beams','100,200',
            '--repeats',c['repeats'],'--queries',10000,'--gpu',args.gpu,'--distortion-nodes',100000,
            '--build-dir',c['build_dir'],'--trainer',repo/'ours/experimental/relative-rate-v1/train_remora.py'])
    save_json(root/'status.json',dict(status='COMPLETE' if set(phases)>={'prepare','measure','diagnostics','rate'} else 'PHASES_COMPLETE',
        phases=phases,elapsed_s=time.monotonic()-started))
    print('[done] '+str(root),flush=True)


if __name__=='__main__':
    main()
