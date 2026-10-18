#!/usr/bin/env python3
"""Exact, blockwise Faiss build KNN in the actual native primary geometry.

Database blocks are scanned outside query batches. Two on-disk generations
make a block commit restart-safe; unfinished generations are never resumed.
GPU is explicit and never silently replaced by CPU. Not a QPS benchmark.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from artifacts import atomic_json, identity, matrix_shape


def merge_topk(scores, ids, new_scores, new_ids, k):
    import numpy as np
    values = np.concatenate((scores, new_scores), axis=1)
    labels = np.concatenate((ids, new_ids), axis=1)
    # Full sort of only <=2K values, deterministic score/ID tie breaking.
    order = np.lexsort((labels, -values), axis=1)[:, :k]
    return (np.take_along_axis(values, order, axis=1),
            np.take_along_axis(labels, order, axis=1))


def dependencies(repo):
    sys.path[:0] = [str(repo / 'ours/test/t2i-10M'), str(repo / 'ours/python')]
    # Import torch before a locally built GPU Faiss to load compatible CUDA libs.
    try:
        import torch
        torch.backends.cuda.matmul.allow_tf32 = False
    except ImportError:
        pass
    import faiss
    return faiss


def scan(args):
    import numpy as np
    repo = Path(args.repo).resolve()
    faiss = dependencies(repo)
    output = Path(args.output).resolve()
    nq, raw_dim = matrix_shape(args.queries)
    if args.model:
        if not args.codes or not args.reader:
            raise ValueError('Native primary KNN requires --model, --codes and --reader')
        os.environ['REMORA_NATIVE_PRIMARY_READER'] = str(Path(args.reader).resolve())
        os.environ['REMORA_PRIMARY_DECODE_THREADS'] = str(args.threads)
        from native100_primary import NativePrimary
        primary = NativePrimary(args.model, args.codes)
        nodes, dim = primary.nodes, primary.scorer_dim
        if primary.dim != raw_dim:
            primary.close()
            raise ValueError('Query/native primary original dimension mismatch')
        projection = primary.projection
        def decode(lo, hi):
            return primary.reconstruct_scorer(np.arange(lo, hi, dtype=np.uint32))
    else:
        if not args.base:
            raise ValueError('FP32 KNN requires --base')
        nodes, dim = matrix_shape(args.base)
        if dim != raw_dim:
            raise ValueError('Base/query dimension mismatch')
        primary = None
        projection = None
        base = np.memmap(args.base, '<f4', 'r', offset=8, shape=(nodes, dim))
        def decode(lo, hi):
            return np.ascontiguousarray(base[lo:hi])
    try:
        if not 0 < args.topk <= nodes or nodes + nq > 2**32 - 1:
            raise ValueError('Invalid Top-K or uint32 graph node count')
        if args.base_block < args.topk or args.query_batch < 1 or args.threads < 1:
            raise ValueError('base-block must be >=Top-K; query-batch/threads must be positive')
        if args.backend == 'gpu' and args.topk > 2048:
            raise ValueError('This Faiss GPU Flat search supports Top-K <=2048')
        # Reader protocol has a bounded request; larger GPU blocks are assembled.
        def block_decode(lo, hi):
            if not primary or hi-lo <= 1048576:
                return decode(lo, hi)
            block = np.empty((hi-lo, dim), np.float32)
            for start in range(lo, hi, 1048576):
                end = min(hi, start+1048576)
                block[start-lo:end-lo] = decode(start, end)
            return block
        faiss.omp_set_num_threads(args.threads)
        resources = config = None
        if args.backend == 'gpu':
            if (not hasattr(faiss, 'StandardGpuResources') or
                    not 0 <= args.gpu < faiss.get_num_gpus()):
                raise ValueError('Requested GPU Faiss is unavailable; no CPU fallback')
            resources = faiss.StandardGpuResources()
            resources.setTempMemory(256 * 1024**2)
            config = faiss.GpuIndexFlatConfig()
            config.device = args.gpu
            config.useFloat16 = False
        queries = np.memmap(args.queries, '<f4', 'r', offset=8, shape=(nq, raw_dim))
        def project(lo, hi):
            values = np.ascontiguousarray(queries[lo:hi])
            return np.ascontiguousarray(values @ projection) if projection is not None else values
        # Verify linear geometry before scanning any database block.
        sample = np.unique(np.linspace(0, nodes-1, min(512, nodes)).astype(np.uint32))
        parity = 0.0
        if primary:
            recon = primary.reconstruct_scorer(sample)
            for q in np.unique(np.linspace(0, nq-1, min(8, nq)).astype(int)):
                a = primary.score(queries[q], sample)
                b = recon @ project(q, q+1)[0]
                parity = max(parity, float(np.max(np.abs(a-b))))
                if not np.allclose(a, b, rtol=2e-5, atol=2e-5):
                    raise ValueError('Actual native scorer is not reproduced by decoder/query map')
        inputs = [args.queries] + ([args.model, args.codes, args.reader] if primary else [args.base])
        request = dict(version=1, inputs=[identity(p) for p in inputs], nodes=nodes, dim=dim,
                       queries=nq, topk=args.topk, backend=args.backend, gpu=args.gpu,
                       base_block=args.base_block, query_batch=args.query_batch, threads=args.threads,
                       use_float16=False)
        state_dir = output.parent / (output.name + '.checkpoint')
        marker = state_dir / 'state.json'
        if output.exists() and not marker.exists():
            raise ValueError('Output already exists without this scan\'s checkpoint; refusing overwrite')
        state_dir.mkdir(parents=True, exist_ok=True)
        import fcntl
        with (state_dir / 'scan.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            state = json.loads(marker.read_text()) if marker.exists() else dict(request=request, end=0, generation=0)
            if state['request'] != request:
                raise ValueError('KNN inputs/configuration changed; choose a new output, not this checkpoint')
            final_meta = output.with_suffix('.metadata.json')
            if state.get('complete'):
                metadata = json.loads(final_meta.read_text())
                if metadata['outputs'] != [identity(output), identity(output.with_suffix('.scores.fbin'))]:
                    raise ValueError('Completed KNN outputs were modified')
                print(f'[reuse] complete KNN: {output}', flush=True)
                return metadata
            shape = (nq, args.topk)
            arrays = []
            for generation in (0, 1):
                pair = []
                for name, dtype in (('scores', '<f4'), ('ids', '<u4')):
                    path = state_dir / f'{generation}.{name}'
                    if marker.exists() and path.stat().st_size != nq * args.topk * 4:
                        raise ValueError('Corrupt KNN checkpoint array')
                    pair.append(np.memmap(path, dtype, 'r+' if marker.exists() else 'w+', shape=shape))
                arrays.append(pair)
            if not marker.exists():
                for scores, ids in arrays:
                    scores[:] = -np.inf
                    ids[:] = np.iinfo(np.uint32).max
                    scores.flush(); ids.flush()
                atomic_json(marker, state)
            started = time.monotonic()
            resume_start = state['end']
            for lo in range(state['end'], nodes, args.base_block):
                hi = min(nodes, lo+args.base_block)
                recon = block_decode(lo, hi)
                if not np.isfinite(recon).all():
                    raise ValueError('Nonfinite decoded base block')
                index = (faiss.GpuIndexFlatIP(resources, dim, config) if resources is not None
                         else faiss.IndexFlatIP(dim))
                index.add(recon)
                previous = arrays[state['generation']]
                next_generation = 1-state['generation']
                current = arrays[next_generation]
                for qlo in range(0, nq, args.query_batch):
                    qhi = min(nq, qlo+args.query_batch)
                    q = project(qlo, qhi)
                    if not np.isfinite(q).all():
                        raise ValueError('Nonfinite build query')
                    values, labels = index.search(q, min(args.topk, hi-lo))
                    if np.any(labels < 0) or not np.isfinite(values).all():
                        raise ValueError('Invalid Faiss block result')
                    values, labels = merge_topk(previous[0][qlo:qhi], previous[1][qlo:qhi],
                                               values, (labels+lo).astype(np.uint32), args.topk)
                    current[0][qlo:qhi], current[1][qlo:qhi] = values, labels
                del index, recon
                for array in current:
                    array.flush()
                    # Durable arrays before committing the generation pointer.
                    with Path(array.filename).open('rb') as handle:
                        os.fsync(handle.fileno())
                state.update(end=hi, generation=next_generation)
                atomic_json(marker, state)
                elapsed = time.monotonic()-started
                done = hi-resume_start
                total = elapsed * (nodes-resume_start) / max(1, done)
                print(f'[KNN] base={hi}/{nodes}; elapsed/estimated_total={elapsed:.1f}/{total:.1f}s', flush=True)
            scores, ids = arrays[state['generation']]
            returned_parity = 0.0
            if primary:
                for q in np.unique(np.linspace(0, nq-1, min(16, nq)).astype(int)):
                    actual = primary.score(queries[q], ids[q])
                    returned_parity = max(returned_parity, float(np.max(abs(actual-scores[q]))))
                    if not np.allclose(actual, scores[q], rtol=2e-5, atol=2e-5):
                        raise ValueError('Returned KNN scores differ from native primary scoring')
            for dest, values in ((output, ids), (output.with_suffix('.scores.fbin'), scores)):
                temporary = dest.with_name(dest.name + '.tmp')
                with temporary.open('wb') as handle:
                    handle.write(struct.pack('<II', *shape))
                    for lo in range(0, nq, args.query_batch):
                        handle.write(values[lo:lo+args.query_batch].tobytes())
                    handle.flush(); os.fsync(handle.fileno())
                os.replace(temporary, dest)
            if request['inputs'] != [identity(p['path']) for p in request['inputs']]:
                raise ValueError('KNN inputs changed during the scan; output cannot be marked complete')
            metadata = dict(status='COMPLETE', request=request, full_database=True,
                            geometry='native-primary' if primary else 'fp32',
                            scorer_parity_max_error=parity, returned_score_max_error=returned_parity,
                            outputs=[identity(output), identity(output.with_suffix('.scores.fbin'))])
            atomic_json(final_meta, metadata)
            state['complete'] = True
            atomic_json(marker, state)
            return metadata
    finally:
        if primary:
            primary.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--repo', default=str(HERE.parents[1]))
    p.add_argument('--base')
    p.add_argument('--model')
    p.add_argument('--codes')
    p.add_argument('--reader')
    p.add_argument('--queries', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--topk', type=int, default=100)
    p.add_argument('--backend', choices=('gpu', 'cpu'), default='gpu')
    p.add_argument('--gpu', type=int, default=0)
    p.add_argument('--base-block', type=int, default=1000000)
    p.add_argument('--query-batch', type=int, default=256)
    p.add_argument('--threads', type=int, default=8)
    return p


if __name__ == '__main__':
    try:
        scan(parser().parse_args())
    except (ValueError, RuntimeError, OSError) as exc:
        print(f'build KNN: {exc}', file=sys.stderr)
        sys.exit(2)
