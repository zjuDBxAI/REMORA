#!/usr/bin/env python3
"""Match the actual T2I-10M native100 operating points, not a fake all-800 label.

Only OPQ/QAQ change from the legacy 192D/PQ96 setup. Unchanged native codecs
are verified and reused. All five graphs are built at M40/L500. Preparation
and timing wait for RS5M experiments 3/4/5 to complete successfully.
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

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
METHODS = ('opq-pq', 'qaq', 'leanvec-ood', 'extended-rabitq', 'saq')
sys.path[:0] = [str(ROOT/'ours/test/t2i-10M'), str(ROOT/'ours/scripts'),
               str(ROOT/'ours/python'), str(ROOT/'ours/test/t2i-1M/python')]


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')
    temporary.replace(path)


def frozen_json(path, value):
    path = Path(path)
    if path.exists():
        if read_json(path) != value:
            raise ValueError('Immutable configuration changed: '+str(path))
    else:
        write_json(path, value)


def reference_rates(package):
    """Read method-specific rates; do not spread the 800-bit target to codecs."""
    if package.get('raw_dim') != 200:
        raise ValueError('Expected the full-200D T2I-10M reference')
    rates = {}
    for method in METHODS:
        q = package['methods'][method]['quantization']
        bits, dim, size = int(q['nominal_code_bits_per_node']), int(q['scorer_dim']), int(q['physical_bytes_per_node'])
        if method in ('opq-pq', 'qaq'):
            if (bits, dim, size) != (800, 200, 100) or q['pq_groups'] != 100 or q['dimension_reduction']:
                raise ValueError('OPQ/QAQ reference must be full 200D / PQ100x8, without truncation')
        elif method == 'extended-rabitq':
            if (bits, dim, size) != (768, 256, 116) or q['bits_per_dimension'] != 3 or q['quantizer_config'] != 'exact':
                raise ValueError('Unexpected native RaBitQ reference')
        elif method == 'saq':
            if (bits, dim, size) != (768, 256, 216) or q['quantization_plan'] != [[192, 4], [64, 0]]:
                raise ValueError('Unexpected SAQ native budget rounding/plan')
        elif (bits, dim, size) != (768, 192, 104):
            raise ValueError('Unexpected LeanVec native operating point')
        rates[method] = dict(primary_bits=bits, scorer_dim=dim, physical_primary_bytes=size,
                             side_bits=64, total_nominal_bits=bits+64,
                             change_from_legacy=method in ('opq-pq', 'qaq'))
    return rates


def prerequisite_ready(status, audit, rate, pane_dead=None, pane_exit=None):
    """Fail closed; no early release on partial reports or a disappeared tmux."""
    if status.get('status') == 'FAILED':
        raise RuntimeError('RS5M failed: '+status.get('error', 'see its run.log'))
    if pane_dead is True and pane_exit != 0:
        raise RuntimeError('RS5M tmux process exited unsuccessfully')
    if pane_dead is True and status.get('status') != 'COMPLETE':
        raise RuntimeError('RS5M process exited before completing all three experiments')
    return (status.get('status') == 'COMPLETE' and status.get('experiments34_complete') is True
            and status.get('experiment5_complete') is True and audit.get('status') == 'COMPLETE'
            and rate.get('status') == 'COMPLETE' and pane_dead is not False)


def optional_json(path):
    return read_json(path) if Path(path).is_file() else {}


def wait_for_rs5m(args, status):
    while True:
        root = Path(args.wait_root)
        state = optional_json(root/'status.json')
        audit = optional_json(root/'audit/summary.json')
        rate = optional_json(root/'rate/experiment5/report/summary.json')
        pane = subprocess.run(['tmux', 'display-message', '-p', '-t', args.wait_session,
                               '#{pane_dead}:#{pane_dead_status}'], capture_output=True, text=True)
        dead, code = None, None
        if pane.returncode == 0:
            values = pane.stdout.strip().split(':')
            dead = values[0] == '1'
            code = int(values[1]) if dead and values[1] else None
        if prerequisite_ready(state, audit, rate, dead, code):
            print('[queue released] RS5M experiments 3/4/5 completed successfully', flush=True)
            return
        status('QUEUED', 'wait for RS5M experiments 3/4/5', prerequisite_status=state)
        print('[waiting] RS5M '+str(state.get('phase', state.get('status', 'not started'))), flush=True)
        time.sleep(args.poll_seconds)


class Stages:
    def __init__(self, root, status):
        self.root, self.status = Path(root), status
        self.source = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

    def run(self, name, command, outputs, inputs=()):
        from protocol import file_id, identity
        command = list(map(str, command)); outputs = list(map(Path, outputs))
        signature = identity(dict(command=command, source=self.source,
                                  inputs={str(p):file_id(p) for p in inputs}))
        stamp = self.root/'stamps'/f'{name}.json'
        old = optional_json(stamp)
        if old and old.get('signature') != signature:
            raise ValueError('Completed stage inputs/command changed; use a new output root: '+name)
        if old.get('signature') == signature and all(p.is_file() for p in outputs):
            if old.get('outputs') == {str(p):file_id(p) for p in outputs}:
                print('[reuse] '+name, flush=True); return
        for path in outputs:
            path.parent.mkdir(parents=True, exist_ok=True)
        log = self.root/'logs'/f'{name}.log'; log.parent.mkdir(parents=True, exist_ok=True)
        self.status('RUNNING', name, log=str(log)); started = time.monotonic()
        print('[stage] '+name+'; log='+str(log), flush=True)
        with log.open('w') as stream:
            process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
            while process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    print(f'[running] {name}; stage_elapsed={time.monotonic()-started:.0f}s', flush=True)
        if process.returncode or any(not p.is_file() or not p.stat().st_size for p in outputs):
            raise RuntimeError(f'{name} failed ({process.returncode}); see {log}')
        write_json(stamp, dict(signature=signature, command=command,
                              elapsed_s=time.monotonic()-started,
                              outputs={str(p):file_id(p) for p in outputs}))


def make_config(args, root, package, truth, prepared):
    c = dict(read_json(args.performance_reference)['config'])
    c.update(dataset='t2i-1M', base=str(Path(args.data)/'base.1M.fbin'), base_rows=1000000, base_dim=200,
             train_queries=str(Path(args.data)/'query.train.30k.fbin'),
             test_queries=str(Path(args.data)/'query.test.10k.fbin'),
             graph_package=str(package), combined_ground_truth=str(truth), output_root=str(prepared),
             fp_graph=str(Path(args.legacy)/'graphs/full-precision/fp32/graph.roar.index'),
             build_dir=str(ROOT/'ours/build-relative-rate-v1'), primary_budget_bits=800,
             primary_dim=200, pq_groups=100, side_bits=64, budgets=[32,64,96,128],
             metric_query_start=10000, metric_query_limit=8000,
             select_query_start=18000, select_query_limit=2000,
             build_query_start=20000, build_query_limit=10000, test_query_limit=10000,
             device=f'cuda:{args.gpu}', kmeans_device=f'cuda:{args.gpu}',
             geometric_encoder_device=f'cuda:{args.gpu}', search_threads=8, cpus=list(range(16,24)),
             build_threads=8, trainer_threads=8, repeats=3, adaptive_extra_repeats=3,
             scoring_placement='traversal', include_fp32_reference=False,
             cold_store=args.cold_store, rerank_storage='direct', fp32_storage='direct',
             beams=[b for b in c['beams'] if b<=1000],
             adaptive_coarse_beams=[b for b in c['adaptive_coarse_beams'] if b<=1000],
             rerank_depths=[10,20,30,40,50,75,100,150,200],
             adaptive_coarse_depths=[10,20,50,100,200])
    return c


def prepare(args, root, rates, stages, status):
    import numpy as np
    from protocol import matrix, write_matrix, file_id, load_config, read_csv, write_csv
    from run_t2i1m_experiments345 import import_legacy, native_score_check
    from run_native100_experiment1 import check_refiner
    from qvrr import Primary
    # Verify old codec/raw row-order relationships; old OPQ/QAQ are NEVER deployed.
    legacy_args = argparse.Namespace(legacy=args.legacy, data=args.data, methods=list(METHODS),
                                     roar_repo='/home/xzl/RoarGraph')
    status('RUNNING', 'verify raw data and unchanged native primary files')
    old_package, truth, _ = import_legacy(legacy_args, root/'legacy-verification')
    old = read_json(old_package)['methods']
    raw_build = root/'inputs/build.10k.fbin'
    values = matrix(Path(args.data)/'query.train.30k.fbin')[20000:30000]
    if raw_build.exists():
        if not np.array_equal(matrix(raw_build), values): raise ValueError('Frozen build queries changed')
    else:
        write_matrix(raw_build, values)
    result = dict(version=1, status='files-and-sampled-base-order-verified', dataset='t2i-1M',
                  nodes=1000000, raw_dim=200, base=str(Path(args.data)/'base.1M.fbin'),
                  nominal_target_bits=800, actual_rates=rates, methods={},
                  note='Exact T2I-10M native100 layouts; only OPQ/QAQ are exactly 800 bits; independent test queries')
    for method in METHODS:
        if method in ('opq-pq', 'qaq'):
            dest = root/'primaries'/method
            command = [sys.executable, ROOT/'ours/python/train_generic_primary.py', '--method', method,
                       '--base-fbin', Path(args.data)/'base.1M.fbin', '--query-fbin', Path(args.data)/'query.train.30k.fbin',
                       '--out-dir', dest, '--train-size', 100000, '--query-start', 0, '--query-limit', 30000,
                       '--fit-query-start', 0, '--fit-query-limit', 10000, '--iterations', 20,
                       '--seed', 20260831, '--threads', 8, '--batch-size', 65536,
                       '--primary-budget-bits', 800, '--opq-dim', 200, '--pq-groups', 100]
            names = ['primary_model.bin', 'primary_codes.bin', 'primary.reconstruction.fbin', 'export_summary.json']
            stages.run('primary-native100-'+method, command, [dest/n for n in names],
                       [ROOT/'ours/python/train_generic_primary.py', Path(args.data)/'base.1M.fbin',
                        Path(args.data)/'query.train.30k.fbin'])
            q = read_json(dest/'export_summary.json'); paths = dict(model=q['model'], codes=q['codes'])
            recon = q['reconstruction_fbin']
            if (q['nominal_code_bits_per_node'], q['scorer_dim'], q['physical_bytes_per_node']) != (800,200,100):
                raise ValueError('Native100 encoding was not full-200D/PQ100x8')
            primary = Primary(paths['model'], paths['codes'])
            try:
                if primary.projection.shape != (200,200): raise ValueError('Dimension reduction is forbidden')
                orthogonality = float(np.max(abs(primary.projection.T@primary.projection-np.eye(200))))
                if orthogonality > 2e-5: raise ValueError('Expected a full orthogonal OPQ transform')
            finally:
                primary.close()
            q.update(pq_groups=100, pq_bits_per_group=8, dimension_reduction=False,
                     requested_nominal_code_bits_per_node=800, full_input_verified=True,
                     opq_orthogonality_max_error=orthogonality, fit_queries_are_test=False)
        else:
            source = read_json(Path(args.legacy)/'primaries/manifests'/f'{method}.json')
            q = dict(source['metadata']); paths = old[method]['paths']
            recon = source['reconstruction_fbin']
            expected = rates[method]
            if (q['nominal_code_bits_per_node'], q['scorer_dim'], int(q['physical_bytes_per_node'])) != (
                    expected['primary_bits'], expected['scorer_dim'], expected['physical_primary_bytes']):
                raise ValueError('Cannot reuse a codec with different native parameters: '+method)
            if method=='saq' and q['quantization_plan'] != [[192,4],[64,0]]:
                raise ValueError('SAQ segmentation mismatch')
            if method=='extended-rabitq' and (q['bits_per_dimension']!=3 or q['quantizer_config']!='exact'):
                raise ValueError('RaBitQ native configuration mismatch')
            print('[reuse native primary] '+method+f'; actual {expected["primary_bits"]} bits', flush=True)
        graph_dir = root/'graphs'/method
        knn = graph_dir/'query-base.top100.ibin'
        helper = HERE/'gpu_primary_knn.py'
        stages.run('gpu-knn-'+method, [sys.executable, helper, '--model', paths['model'], '--codes', paths['codes'],
                   '--reconstruction', recon, '--queries', raw_build, '--gpu', args.gpu, '--output', knn],
                   [knn, knn.with_suffix('.scores.fbin'), knn.with_suffix('.metadata.json')],
                   [helper, paths['model'], paths['codes'], recon, raw_build])
        graph, construction = graph_dir/'graph.roar.index', graph_dir/'construction.json'
        command = [args.builder, '--primary-kind', method, '--primary-model', paths['model'],
                   '--primary-codes', paths['codes'], '--disable-ji', '--queries', raw_build,
                   '--query-start', 0, '--query-limit', 10000, '--knn', knn, '--graph', graph,
                   '--summary', construction, '--topk', 100, '--M_pjbp', 40, '--L_pjpq', 500,
                   '--threads', args.build_threads, '--score-batch', 65536]
        if method=='extended-rabitq': command += ['--rabitq-config', 'exact']
        stages.run('graph-M40-L500-'+method, command, [graph, construction],
                   [args.builder, paths['model'], paths['codes'], knn, raw_build])
        construction_info = read_json(construction)
        with graph.open('rb') as stream:
            entry, rows = struct.unpack('<II', stream.read(8))
        if (rows!=1000000 or entry>=rows or not construction_info['code_native_construction']
                or construction_info['representation']!='primary' or construction_info['build_queries']!=10000):
            raise ValueError('Graph does not belong to its primary/build query population')
        q['physical_bytes_per_node'] = int(q['physical_bytes_per_node'])
        result['methods'][method] = dict(paths=paths, graph=str(graph), quantization=q,
                                         graph_parameters=dict(M_pjbp=40,L_pjpq=500,topk=100),
                                         artifacts={str(p):file_id(p) for p in [*paths.values(), graph, construction, knn]})
    package = root/'inputs/imported-primary.json'; frozen_json(package, result)
    write_csv(root/'primary-budgets.csv', [dict(primary=m,**rates[m]) for m in METHODS])
    prepared = root/'prepared'
    config = make_config(args, root, package, truth, prepared)
    config_path = root/'prepare.config.json'; frozen_json(config_path, config)
    config = load_config(config_path)
    from pipeline import Pipeline
    pipeline = Pipeline(config)
    if not pipeline.preflight(): raise RuntimeError('Prepared inputs failed preflight')
    pipeline.data(); pipeline.ground_truth(); native_score_check(pipeline)
    splits = read_csv(prepared/'query-splits.csv')
    for split in splits:
        if split['split']=='build':
            split['usage']='10k independent training queries used for native primary M40/L500 graph construction'
    write_csv(prepared/'query-splits.csv', splits)
    for method in METHODS:
        status('RUNNING', 'train 64-bit REMORA', primary=method)
        try:
            check_refiner(prepared, method, 1000000, 64)
            print('[reuse refiner] '+method, flush=True)
        except (OSError, ValueError, KeyError, struct.error):
            pools = pipeline.pools(method); pipeline.ji(method, 64, pools)
            check_refiner(prepared, method, 1000000, 64)
    performance = root/'performance'
    performance_config = dict(config, output_root=str(performance))
    performance_path = root/'performance.config.json'; frozen_json(performance_path, performance_config)
    return prepared, performance_path


def main():
    from protocol import matrix_header, file_id
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=('check','run'))
    parser.add_argument('--output', default=str(ROOT/'outputs/t2i-1M/remora-native100-exp1-nvme-b64-v1'))
    parser.add_argument('--data', default='/home/xzl/RoarGraph/data/t2i-1M')
    parser.add_argument('--legacy', default='/home/xzl/RoarGraph/ours/outputs/t2i-1M/ji-pure-suite-five-primary-v3')
    parser.add_argument('--reference', default='/home/xzl/RoarGraph/ours/transfer/t2i-10M-native100-20261006/imported-primary.jso')
    parser.add_argument('--performance-reference', default=str(ROOT/'outputs/t2i-10M/remora-native100-traversal-exp1-nvme-b64-v2/configuration.json'))
    parser.add_argument('--builder', default='/home/xzl/RoarGraph/ours/build-laion-opq-v1/roar_primary_ji_build')
    parser.add_argument('--build-threads', type=int, default=64)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--cold-store', default='/tmp/xzl-remora-t2i1m-exp345-v1/base.1M.cold.bin')
    parser.add_argument('--wait-root', default=str(ROOT/'outputs/rs5m/remora-exp345-relative-b80-320-v2'))
    parser.add_argument('--wait-session', default='remora-rs5m-exp345')
    parser.add_argument('--poll-seconds', type=float, default=30)
    args = parser.parse_args()
    if args.build_threads<=0 or args.poll_seconds<1 or args.gpu<0:
        parser.error('Invalid threads/GPU/poll interval')
    rates = reference_rates(read_json(args.reference))
    for name, expected in [('base.1M.fbin',(1000000,200)), ('query.train.30k.fbin',(30000,200)),
                           ('query.test.10k.fbin',(10000,200))]:
        if matrix_header(Path(args.data)/name)!=expected:
            raise ValueError('Unexpected complete T2I-1M raw shape: '+name)
    for path in [args.builder, ROOT/'ours/build-relative-rate-v1/roar_primary_ji_search',
                 ROOT/'ours/build-relative-rate-v1/roar_exp234_audit',
                 ROOT/'ours/runtime/native100-reader-v1/roar_native_primary_reader']:
        if not os.access(path, os.X_OK): raise ValueError('Required native binary unavailable: '+str(path))
    if 'avx512f' not in Path('/proc/cpuinfo').read_text(): raise RuntimeError('AVX-512 CPU required')
    if args.command=='check':
        print(json.dumps(dict(status='PLAN_VALIDATED', rates=rates,
                              note='Read-only check; no CUDA allocation, primary encoding, graph build or QPS scan'), indent=2))
        return
    root = Path(args.output).resolve(); root.mkdir(parents=True, exist_ok=True)
    lock = (root/'workflow.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
    started = time.monotonic()
    def status(state, phase, **extra):
        write_json(root/'status.json', dict(status=state,phase=phase,elapsed_s=time.monotonic()-started,**extra))
    policy = dict(version=1, dataset='t2i-1M', primary_target_bits=800, actual_rates=rates, side_bits=64,
                  scoring_placement='traversal', performance_threads=8, performance_repeats=3,
                  max_beam=1000, max_rerank_depth=200, graph_M=40, graph_L=500, build_queries=10000,
                  fp32_curve=False, io_policy='NVMe O_DIRECT, own scorer queue Top-L',
                  prerequisite_root=args.wait_root, prerequisite_session=args.wait_session,
                  reference_id=file_id(args.reference), args=vars(args),
                  runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  binary_ids={str(p):file_id(p) for p in [args.builder,
                      ROOT/'ours/build-relative-rate-v1/roar_primary_ji_search',ROOT/'ours/build-relative-rate-v1/roar_exp234_audit']})
    frozen_json(root/'workflow.json', policy)
    try:
        wait_for_rs5m(args, status)
        stages = Stages(root, status)
        prepared, config = prepare(args, root, rates, stages, status)
        performance_summary = root/'performance/experiment1/summary.json'
        status('RUNNING', 'experiment1 performance scan; REMORA traversal / CPU8 / direct rerank')
        stages.run('experiment1-performance', [sys.executable, '-u', ROOT/'ours/scripts/run_native100_capped_experiment1.py',
                   'launch', '--config', config, '--prepared-root', prepared, '--cold-store', args.cold_store,
                   '--max-rerank-depth', 200], [performance_summary],
                   [config, root/'inputs/imported-primary.json', ROOT/'ours/scripts/run_native100_capped_experiment1.py'])
        if read_json(performance_summary).get('status')!='COMPLETE':
            raise RuntimeError('Performance results incomplete; do not label them finished')
        write_json(root/'experiment1-policy.json',
                   dict(primary_configuration='T2I-10M native100 matched operating points',
                        nominal_target_bits=800, actual_primary_rates=rates,
                        legacy_768bit_opq_qaq_results_reused=False,
                        performance_summary=str(performance_summary)))
        status('COMPLETE', 'done', experiment1_summary=str(performance_summary), actual_rates=rates)
        print('[complete] '+str(performance_summary), flush=True)
    except BaseException as exc:
        status('FAILED', 'stopped', error=str(exc)); raise


if __name__ == '__main__':
    main()
