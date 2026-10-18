#!/usr/bin/env python3
"""Exact Faiss-GPU Top-100 in the frozen primary's own scoring space."""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT/'ours/test/t2i-10M'), str(ROOT/'ours/python')]


def main():
    from protocol import matrix, write_matrix, file_id
    from qvrr import Primary
    import faiss
    import numpy as np
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('model', 'codes', 'reconstruction', 'queries', 'output'):
        parser.add_argument('--'+name, required=True)
    parser.add_argument('--gpu', type=int, default=2)
    args = parser.parse_args()
    if not hasattr(faiss, 'StandardGpuResources') or not 0 <= args.gpu < faiss.get_num_gpus():
        raise RuntimeError('Requested GPU Faiss runtime is unavailable; no silent CPU fallback')
    primary = Primary(args.model, args.codes)
    try:
        queries = matrix(args.queries)
        recon = matrix(args.reconstruction)
        if queries.shape[1] != primary.dim or recon.shape != (primary.nodes, primary.scorer_dim):
            raise ValueError('Primary reconstruction/query dimensions do not match the native codec')
        sampled = np.unique(np.linspace(0, primary.nodes-1, 1024).astype(np.uint32))
        decode_error = float(np.max(abs(primary.reconstruct_scorer(sampled)-recon[sampled])))
        if decode_error > 2e-5:
            raise ValueError('Materialized offline reconstruction differs from native decoder')
        projected = np.ascontiguousarray(queries @ primary.projection, dtype=np.float32)
        parity = max(float(np.max(abs(primary.score(queries[q], sampled)-recon[sampled]@projected[q])))
                     for q in np.linspace(0, len(queries)-1, 8).astype(int))
        if parity > 2e-5:
            raise ValueError('Reconstruction-space IP does not match the actual primary scorer')
        started = time.monotonic()
        resource = faiss.StandardGpuResources()
        config = faiss.GpuIndexFlatConfig()
        config.device = args.gpu
        config.useFloat16 = False
        index = faiss.GpuIndexFlatIP(resource, primary.scorer_dim, config)
        for lo in range(0, primary.nodes, 65536):
            hi = min(primary.nodes, lo+65536)
            index.add(np.ascontiguousarray(recon[lo:hi], dtype=np.float32))
            print(f'[GPU upload] {hi}/{primary.nodes}; elapsed={time.monotonic()-started:.1f}s', flush=True)
        ids = np.empty((len(queries), 100), np.uint32)
        scores = np.empty(ids.shape, np.float32)
        for lo in range(0, len(queries), 256):
            hi = min(len(queries), lo+256)
            value, label = index.search(projected[lo:hi], 100)
            if np.any(label < 0) or np.any(label >= primary.nodes) or not np.isfinite(value).all():
                raise ValueError('Invalid full-database GPU KNN result')
            ids[lo:hi], scores[lo:hi] = label, value
            print(f'[GPU Top-100] {hi}/{len(queries)}; elapsed={time.monotonic()-started:.1f}s', flush=True)
        returned_error = max(float(np.max(abs(primary.score(queries[q], ids[q])-scores[q])))
                             for q in np.linspace(0, len(queries)-1, 16).astype(int))
        if returned_error > 2e-5:
            raise ValueError('GPU Top-100 scores differ from native primary scoring')
        output = Path(args.output)
        write_matrix(output, ids, '<u4')
        write_matrix(output.with_suffix('.scores.fbin'), scores)
        metadata = dict(status='COMPLETE', backend='faiss-gpu-flat-ip-fp32', full_database=True,
                        raw_dim=primary.dim, scorer_dim=primary.scorer_dim, nodes=primary.nodes,
                        build_queries=len(queries), topk=100, gpu=args.gpu, use_float16=False,
                        primary_reconstruction_max_error=decode_error, primary_score_max_error=parity,
                        returned_native_score_max_error=returned_error, elapsed_s=time.monotonic()-started,
                        input_ids={p:file_id(p) for p in (args.model, args.codes, args.reconstruction, args.queries)},
                        note='Offline KNN only; graph projection/pruning consumes native primary codes')
        output.with_suffix('.metadata.json').write_text(json.dumps(metadata, indent=2)+'\n')
    finally:
        primary.close()


if __name__ == '__main__':
    main()
