#!/usr/bin/env python3
"""Unified public CLI: native primary encoding/build, REMORA train, experiments."""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import build as graph_builder
import run as workflow
from artifacts import atomic_json, identity, matrix_shape, sliced_matrix

DATASETS = ('t2i-1M', 't2i-10M', 't2i-1B', 'webvid', 'rs5m', 'wordvector-id', 'laion-400M')
METHODS = ('fp32', 'opq-pq', 'qaq', 'leanvec-ood', 'extended-rabitq', 'saq')


def resolve(repo, value):
    return workflow.absolute(repo, value)


def native_parameters(dim, method, target_bits=None, pq_groups=None,
                      rabitq_bits=None, saq_bits=None, leanvec_dim=None):
    """Native layouts; OPQ/QAQ never truncate dimensions to fit a code size."""
    target = 4 * dim if target_bits is None else target_bits
    if target <= 0:
        raise ValueError('Primary target bits must be positive')
    result = dict(original_dim=dim, target_bits=target)
    if method in ('opq-pq', 'qaq'):
        groups = pq_groups or min((g for g in range(1, dim+1) if dim % g == 0),
                                  key=lambda g: (abs(8*g-target), g))
        if not 0 < groups <= dim or dim % groups:
            raise ValueError('PQ groups must divide the ORIGINAL dimension; no truncation')
        result.update(scorer_dim=dim, pq_groups=groups, bits_per_codeword=8, code_bits=groups*8)
    elif method == 'extended-rabitq':
        if not 64 <= dim < 4096:
            raise ValueError('This native FhtKac adapter supports original dimensions 64..4095')
        padded = math.ceil(dim / 64) * 64
        bits = rabitq_bits or min(range(1, 9), key=lambda b: (abs(padded*b-target), b))
        if bits not in range(1, 9):
            raise ValueError('RaBitQ supports 1..8 coordinate bits')
        result.update(scorer_dim=padded, coordinate_bits=bits, code_bits=padded*bits, config='exact')
    elif method == 'saq':
        # SAQ applies its quota to padded coordinates, not original D.
        bits = saq_bits if saq_bits is not None else target / (math.ceil(dim/64)*64)
        if not 1 <= bits <= 8:
            raise ValueError('SAQ average bits must lie in [1,8]')
        result.update(avg_bits=bits, scorer_dim=math.ceil(dim/64)*64,
                      allocation_quota_bits=bits*math.ceil(dim/64)*64,
                      actual_code_bits='read native segmentation plan in export_summary.json')
    elif method == 'leanvec-ood':
        # 8-bit affine reproduction, reduced D is tunable using validation.
        reduced = leanvec_dim or min(dim, max(1, round(target/8)))
        if not 0 < reduced <= dim:
            raise ValueError('LeanVec reduced dimension must lie in [1,D]')
        result.update(scorer_dim=reduced, coordinate_bits=8, code_bits=8*reduced,
                      implementation='local LeanVec-OOD reproduction')
    elif method == 'fp32':
        result.update(scorer_dim=dim, code_bits=32*dim)
    else:
        raise ValueError(f'Unsupported method: {method}')
    return result


def build_request(args):
    repo = args.repo.resolve(strict=True)
    if args.dataset not in DATASETS:
        raise ValueError('Unsupported dataset; SIFT/GIST and L2 are excluded')
    descriptor = {}
    if args.dataset_config:
        descriptor = json.loads(args.dataset_config.read_text())
        if descriptor.get('dataset') != args.dataset:
            raise ValueError('Dataset descriptor label must match --dataset')
        if descriptor.get('metric') not in ('ip', 'inner_product', 'maximum_inner_product'):
            raise ValueError('Only native IP datasets are supported; no implicit L2 conversion')
    root = resolve(repo, args.data_root or descriptor.get('data_root', f'datasets/{args.dataset}'))
    def source(flag, key, standard):
        value = flag or descriptor.get(key)
        candidate = resolve(repo, value) if value else root / standard
        if not candidate.is_file():
            raise ValueError(f'Missing {key}: {candidate}; supply an explicit path or dataset JSON')
        return candidate.resolve()
    base = source(args.base, 'base', 'base.fbin')
    queries = source(args.build_queries, 'build_queries', 'query.build.fbin')
    nb, dim = matrix_shape(base)
    nq, qdim = matrix_shape(queries)
    if qdim != dim or descriptor.get('dim', dim) != dim:
        raise ValueError('Base/build queries/descriptor dimension mismatch')
    count = args.build_query_limit or nq
    if not 0 < count <= nq or not 0 < args.knn_k <= nb or nb + count > 2**32 - 1:
        raise ValueError('Invalid build query count/Top-K or uint32 graph node count limit')
    for key in ('degree', 'build_beam', 'build_threads', 'iterations', 'train_size',
                'base_block', 'query_batch'):
        if getattr(args, key) < 1:
            raise ValueError(f'{key} must be positive')
    if args.build_beam < args.knn_k:
        raise ValueError('Build beam must be >= build Top-K')
    fit = None
    if args.method in ('qaq', 'leanvec-ood'):
        fit = source(args.fit_queries, 'fit_queries', 'query.train.fbin')
        nf, df = matrix_shape(fit)
        if df != dim or nf < args.fit_query_limit or args.fit_query_limit < 1:
            raise ValueError('Insufficient or dimension-mismatched primary fit queries')
    elif args.fit_queries:
        raise ValueError('--fit-queries applies to QAQ/LeanVec only')
    out = resolve(repo, args.output or f'outputs/{args.dataset}/graphs/{args.method}')
    # Never allow a new build output to contain or overwrite source embeddings.
    for inp in (base, queries, fit):
        if inp and (out == inp or out in inp.parents):
            raise ValueError('Build output must be separate from the input data directory')
    build_dir = resolve(repo, args.build_dir)
    if args.gpu < 0 or args.base_block < args.knn_k:
        raise ValueError('GPU must be nonnegative and base-block must be >=Top-K')
    params = native_parameters(dim, args.method, args.primary_bits, args.pq_groups,
                               args.rabitq_bits, args.saq_avg_bits, args.leanvec_dim)
    if args.method not in ('opq-pq', 'qaq') and args.pq_groups is not None:
        raise ValueError('--pq-groups is only valid for OPQ/QAQ')
    if args.method != 'extended-rabitq' and args.rabitq_bits is not None:
        raise ValueError('--rabitq-bits applies only to Extended-RaBitQ')
    if args.method != 'saq' and args.saq_avg_bits is not None:
        raise ValueError('--saq-avg-bits applies only to SAQ')
    if args.method != 'leanvec-ood' and args.leanvec_dim is not None:
        raise ValueError('--leanvec-dim applies only to LeanVec')
    return dict(version=1, dataset=args.dataset, method=args.method, metric='ip', nodes=nb,
                parameters=params, inputs=[identity(p) for p in (base, queries, fit) if p],
                base=str(base), queries=str(queries), fit_queries=str(fit) if fit else None,
                output=str(out), build_dir=str(build_dir), query_limit=count, topk=args.knn_k,
                degree=args.degree, build_beam=args.build_beam, threads=args.build_threads,
                train_size=args.train_size, iterations=args.iterations, seed=args.seed,
                fit_query_limit=args.fit_query_limit if fit else 0,
                knn_backend=args.knn_backend, gpu=args.gpu, base_block=args.base_block,
                query_batch=args.query_batch, python=args.python)


def build_commands(repo, request):
    r, p = request, request['parameters']
    out, binary = Path(r['output']), Path(r['build_dir'])
    primary_dir = out / 'primary'
    queries = out / 'query.build.fbin'
    model, codes = primary_dir/'primary_model.bin', primary_dir/'primary_codes.bin'
    fit = r['fit_queries'] or str(queries)
    encoder = []
    if r['method'] in ('opq-pq', 'qaq', 'leanvec-ood'):
        encoder = [r['python'], '-u', str(repo/'ours/python/train_generic_primary.py'),
                   '--method', r['method'], '--base-fbin', r['base'], '--query-fbin', fit,
                   '--out-dir', str(primary_dir), '--opq-dim', str(p['original_dim']),
                   '--primary-budget-bits', str(p['code_bits']), '--threads', str(r['threads']),
                   '--train-size', str(r['train_size']), '--iterations', str(r['iterations']),
                   '--seed', str(r['seed']), '--query-limit', str(r['fit_query_limit'] or r['query_limit']),
                   '--fit-query-limit', str(r['fit_query_limit'] or r['query_limit'])]
        if r['method'] == 'leanvec-ood':
            encoder += ['--leanvec-dim', str(p['scorer_dim']), '--leanvec-bits-per-dim', '8']
        else:
            encoder += ['--pq-groups', str(p['pq_groups'])]
    elif r['method'] == 'extended-rabitq':
        model = codes = primary_dir/'base.roarq'
        encoder = [str(binary/'roar_rabitq_quantize'), r['base'], str(model),
                   str(p['coordinate_bits']), 'ip', 'false']
    elif r['method'] == 'saq':
        model, codes = primary_dir/'saq_model.bin', primary_dir/'saq_codes.bin'
        encoder = [str(binary/'roar_export_saq_primary'), f'--base_fbin={r["base"]}',
                   f'--query_fbin={queries}', f'--out_dir={primary_dir}',
                   f'--avg_bits={p["avg_bits"]}', f'--threads={r["threads"]}',
                   f'--query_limit={r["query_limit"]}', f'--seed={r["seed"]}']
    knn = [r['python'], '-u', str(HERE/'knn.py'), '--repo', str(repo), '--queries', str(queries),
           '--output', str(out/'query-base.knn.ibin'), '--topk', str(r['topk']),
           '--backend', r['knn_backend'], '--gpu', str(r['gpu']), '--threads', str(r['threads']),
           '--base-block', str(r['base_block']), '--query-batch', str(r['query_batch'])]
    c = dict(kind='fp32' if r['method'] == 'fp32' else 'native-primary', base=r['base'],
             queries=str(queries), knn=str(out/'query-base.knn.ibin'), graph=str(out/'graph.roar.index'),
             knn_geometry='fp32' if r['method'] == 'fp32' else 'primary', query_limit=r['query_limit'],
             topk=r['topk'], degree=r['degree'], build_beam=r['build_beam'], threads=r['threads'],
             summary=str(out/'construction.json'), primary_kind=r['method'],
             primary_model=str(model), primary_codes=str(codes),
             builder=str(binary/('roar_fp_build' if r['method']=='fp32' else 'roar_primary_ji_build')))
    if r['method'] == 'fp32':
        knn += ['--base', r['base']]
    else:
        knn += ['--model', str(model), '--codes', str(codes),
                '--reader', str(binary/'roar_native_primary_reader')]
    validation, command = graph_builder.commands(repo, c)
    return encoder, knn, validation, command, c


def quantization(config, request):
    """Report actual exported rate, not just the requested operating point."""
    if request['method'] == 'fp32':
        return dict(implementation='fp32', original_dim=request['parameters']['original_dim'],
                    nominal_code_bits_per_node=32*request['parameters']['original_dim'],
                    physical_bytes_per_node=4*request['parameters']['original_dim'])
    model = Path(config['primary_model'])
    if request['method']=='extended-rabitq':
        with model.open('rb') as handle:
            h=struct.unpack('<8s4I6Q',handle.read(72))
        if h[0]!=b'RBQROAR1':
            raise ValueError('Wrong RaBitQ export magic')
        q=dict(implementation='official-rabitq-exact-adapter',base_rows=h[5],original_dim=h[6],
               scorer_dim=h[7],nominal_code_bits_per_node=h[4]*h[7],physical_bytes_per_node=h[8]+h[9])
    else:
        q=json.loads((model.parent/'export_summary.json').read_text())
    if q.get('base_rows',q.get('nodes'))!=request['nodes'] or q['original_dim']!=request['parameters']['original_dim']:
        raise ValueError('Exported primary base row count/original dimension mismatch')
    if q['nominal_code_bits_per_node']<=0:
        raise ValueError('Native exporter produced a zero-bit/degenerate primary; refusing to build a meaningless graph')
    return q


def run_build(args):
    repo = args.repo.resolve(strict=True)
    request = build_request(args)
    encoder, knn, validation, command, config = build_commands(repo, request)
    # Rebuilt binaries or edited encoder/workflow sources are different requests.
    dependencies = [Path(__file__), HERE/'artifacts.py', HERE/'knn.py', HERE/'build.py',
                    Path(command[0])]
    if encoder:
        dependencies.append(Path(encoder[2] if args.method in ('opq-pq','qaq','leanvec-ood') else encoder[0]))
    if args.method != 'fp32':
        dependencies.append(Path(request['build_dir'])/'roar_native_primary_reader')
    request['implementation'] = [identity(p) if p.is_file() else dict(path=str(p), missing=True)
                                 for p in dependencies]
    print(json.dumps(request, indent=2, ensure_ascii=False))
    for name, cmd in (('encode', encoder), ('build KNN', knn), ('native parity', validation), ('graph', command)):
        if cmd:
            print(f'{name}: {shlex.join(cmd)}')
    print('Build KNN uses TRAIN/build queries, never test ground truth. No SIFT/GIST or L2 conversion.')
    if not args.execute:
        print('Plan only; add --execute to run. Repeated identical requests reuse verified completed stages.')
        return 0
    out = Path(request['output'])
    if out.exists() and any(out.iterdir()) and not (out/'request.json').exists():
        raise ValueError('Nonempty output was not created by this interface; choose a new directory')
    # Fail on missing prerequisites before encoding a large database.
    for cmd in (encoder, knn, validation, command):
        if cmd and not Path(cmd[0]).is_file():
            raise ValueError(f'Missing executable: {cmd[0]}; compile/install first')
    if args.method != 'fp32' and not (Path(request['build_dir'])/'roar_native_primary_reader').is_file():
        raise ValueError('Compile roar_native_primary_reader before executing')
    out.mkdir(parents=True, exist_ok=True)
    with (out/'build.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        marker = out/'request.json'
        if marker.exists() and json.loads(marker.read_text()) != request:
            raise ValueError('Build request or inputs changed; choose a NEW --output, no silent reuse')
        atomic_json(marker, request)
        env = workflow.environment(repo, args.python)
        env['REMORA_PRIMARY_DECODE_THREADS'] = str(args.build_threads)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            env[name] = str(args.build_threads)
        # Check Faiss/GPU before an expensive encoder starts. No fallback.
        preflight = [args.python, '-c',
                     'import sys; sys.path.insert(0, '+repr(str(HERE))+'); '
                     'from knn import dependencies; from pathlib import Path; '
                     'f=dependencies(Path('+repr(str(repo))+')); '
                     + ('assert hasattr(f,"StandardGpuResources") and '
                        + str(args.gpu)+' < f.get_num_gpus(), "Requested GPU Faiss is unavailable"'
                        if args.knn_backend == 'gpu' else 'assert hasattr(f,"IndexFlatIP")')]
        subprocess.run(preflight, cwd=repo, env=env, check=True)
        started = time.monotonic()
        def launch(stage, cmd):
            print(f'[{stage}] elapsed/total={time.monotonic()-started:.1f}s/unknown', flush=True)
            with (out/f'{stage}.log').open('a') as log:
                log.write(shlex.join(cmd)+'\n'); log.flush()
                with subprocess.Popen(cmd, cwd=repo, env=env, stdout=subprocess.PIPE,
                                      stderr=subprocess.STDOUT, text=True, bufsize=1) as proc:
                    try:
                        for line in proc.stdout:
                            print(line, end='', flush=True); log.write(line); log.flush()
                        if proc.wait():
                            raise RuntimeError(f'{stage} failed; see {log.name}')
                    except BaseException:
                        if proc.poll() is None:
                            proc.terminate(); proc.wait()
                        raise
        sliced_matrix(request['queries'], config['queries'], request['query_limit'])
        slice_marker = out/'query-slice.complete.json'
        slice_id = identity(config['queries'])
        if slice_marker.exists() and json.loads(slice_marker.read_text()) != slice_id:
            raise ValueError('Existing query slice was modified')
        atomic_json(slice_marker, slice_id)
        if encoder:
            complete = out/'encoding.complete.json'
            outputs = [Path(config['primary_model']), Path(config['primary_codes'])]
            if args.method!='extended-rabitq':
                outputs.append(outputs[0].parent/'export_summary.json')
            if complete.exists():
                if json.loads(complete.read_text()) != [identity(p) for p in outputs]:
                    raise ValueError('Previously encoded primary files were modified')
                print('[reuse] primary encoding', flush=True)
            else:
                outputs[0].parent.mkdir(parents=True, exist_ok=True)
                launch('encode', encoder)
                atomic_json(complete, [identity(p) for p in outputs])
        if request['inputs'] != [identity(p['path']) for p in request['inputs']]:
            raise ValueError('Source vectors changed during encoding')
        q = quantization(config, request)
        # KNN scan checks native decoder/score parity and owns restart metadata.
        launch('knn', knn)
        atomic_json(out/'build.config.json', config)
        validation, command = graph_builder.validate(repo, config)
        graph = Path(config['graph'])
        complete = out/'graph.complete.json'
        if complete.exists():
            if json.loads(complete.read_text()) != identity(graph):
                raise ValueError('Previously built graph was modified')
            print('[reuse] primary graph', flush=True)
        else:
            if graph.exists() or (args.method!='fp32' and Path(config['summary']).exists()):
                raise ValueError('Interrupted/unverified graph output exists; use a new --output')
            if validation:
                launch('validate', validation)
            launch('graph', command)
            atomic_json(complete, identity(graph))
        atomic_json(out/'summary.json', dict(status='COMPLETE', request=request,
                    graph=identity(graph), construction_config=config,
                    quantization=q,
                    elapsed_this_invocation_s=time.monotonic()-started))
        print(f'[complete] {out}/summary.json', flush=True)
    return 0


def training_command(args):
    repo = args.repo.resolve(strict=True)
    if args.config:
        if args.output:
            raise ValueError('With --config, output_root is owned by that config; --output is not an override')
        c = workflow.config(repo, args.config)
        if c.get('scoring_placement') != 'traversal' or c.get('budget_allocator') != 'balanced-product':
            raise ValueError('Training config must declare traversal and balanced-product')
        if c.get('dataset', '').lower().startswith(('sift', 'gist')):
            raise ValueError('SIFT/GIST experiments are excluded')
        labels = json.loads((HERE/'catalog.json').read_text())['datasets']
        expected = labels.get(args.dataset, {}).get('dataset', args.dataset)
        if c['dataset'] != expected:
            raise ValueError('Config dataset does not match --dataset')
        return [args.python, '-u', str(repo/'ours/test/t2i-10M/run.py'), 'prepare',
                '--config', str(args.config.resolve()), '--experiments', '1']
    return None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    build = sub.add_parser('build', help='Primary encoding -> exact Faiss KNN -> native graph')
    build.add_argument('--dataset', choices=DATASETS, required=True)
    build.add_argument('--method', choices=METHODS, required=True)
    build.add_argument('--dataset-config', type=Path, help='JSON dataset/metric/base/build_queries/fit_queries')
    for name in ('data-root', 'base', 'build-queries', 'fit-queries', 'output'):
        build.add_argument('--'+name)
    build.add_argument('--build-dir', default='ours/build-workflow-v1')
    for name, default in (('degree',40), ('build-beam',500), ('knn-k',100), ('build-threads',8),
                          ('build-query-limit',0), ('fit-query-limit',10000), ('train-size',100000),
                          ('iterations',20), ('seed',20261010), ('base-block',1000000), ('query-batch',256)):
        build.add_argument('--'+name, type=int, default=default)
    for name in ('primary-bits', 'pq-groups', 'rabitq-bits', 'leanvec-dim'):
        build.add_argument('--'+name, type=int)
    build.add_argument('--saq-avg-bits', type=float)
    build.add_argument('--knn-backend', choices=('gpu','cpu'), default='gpu')
    build.add_argument('--gpu', type=int, default=0)
    for action in ('train', 'evaluate', 'check', 'reports', 'list'):
        p = sub.add_parser(action)
        p.add_argument('--dataset', choices=DATASETS, default='t2i-10M')
        p.add_argument('--catalog', type=Path, default=HERE/'catalog.json')
        p.add_argument('--output')
        if action == 'train':
            p.add_argument('--config', type=Path, help='Explicit portable prepare config; GPU device lives in this config')
        if action == 'evaluate':
            p.add_argument('--experiments', default='1', help='Comma-separated 1..5; dataset configs from catalog')
            p.add_argument('--gpu', type=int, default=0, help='GPU for ablation/rate training (not QPS)')
    for p in sub.choices.values():
        p.add_argument('--repo', type=Path, default=HERE.parents[1])
        p.add_argument('--python', default=os.environ.get('PYTHON_BIN', sys.executable))
        p.add_argument('--execute', action='store_true', help='Without this flag only print the plan')
    args = parser.parse_args(argv)
    import shutil
    args.python = shutil.which(args.python) or args.python
    if args.action == 'build':
        return run_build(args)
    if args.action == 'train':
        command = training_command(args)
        if command:
            print(shlex.join(command))
            if args.execute:
                subprocess.run(command, cwd=args.repo, env=workflow.environment(args.repo.resolve(), args.python), check=True)
            return 0
        stages = 'prepare'
    elif args.action == 'evaluate':
        numbers = args.experiments.split(',')
        if not numbers or any(n not in ('1','2','3','4','5') for n in numbers):
            raise ValueError('--experiments must be comma-separated 1..5')
        # Routing teacher pairs must exist before ablation/rate consume them.
        stages = ','.join('exp'+n for n in ('1','3','4','2','5') if n in numbers)
    else:
        stages = 'exp1'
    forwarded = [('run' if args.execute else 'plan') if args.action in ('train','evaluate') else args.action,
                 '--repo', str(args.repo), '--catalog', str(args.catalog), '--dataset', args.dataset,
                 '--python', args.python, '--stages', stages]
    if args.output:
        forwarded += ['--output', args.output]
    if args.action == 'evaluate':
        forwarded += ['--gpu', str(args.gpu)]
    if args.execute:
        forwarded += ['--execute']
    return workflow.main(forwarded)


def entry(argv=None):
    try:
        return main(argv)
    except (ValueError, RuntimeError, OSError, KeyError, subprocess.CalledProcessError) as exc:
        print(f'REMORA interface: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    sys.exit(entry())
