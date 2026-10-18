#!/usr/bin/env python3
"""Add full-database Top-200 diagnostic truth without changing official Top-100."""
import argparse
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[3]
sys.path[:0]=[str(ROOT/'ours/test/t2i-10M'),str(ROOT/'ours/scripts')]
from protocol import content_sha256, file_id, identity, matrix, write_matrix
from run_frozen_experiment5 import save_json


def preserve_top100(old_ids, old_scores, ids, scores):
    """Preserve authoritative Recall/Top-100 tie handling; append the other 100."""
    result_ids=np.empty(ids.shape,dtype=np.uint32)
    result_scores=np.empty(scores.shape,dtype=np.float32)
    if (old_ids.shape!=old_scores.shape or old_ids.shape[1]!=100 or ids.shape!=scores.shape
            or ids.shape!=(len(old_ids),200) or np.any(ids<0) or not np.isfinite(scores).all()):
        raise ValueError('Invalid official/GPU ground truth shape or contents')
    for q in range(len(ids)):
        extra=~np.isin(ids[q],old_ids[q])
        missing=~np.isin(old_ids[q],ids[q])
        if len(np.unique(ids[q]))!=200 or np.count_nonzero(extra)<100:
            raise ValueError('Invalid unique full-database Top-200 candidates')
        # More than 200 database images can have identical embeddings/scores.
        # Keep official IDs, but admit missing ones ONLY at the GPU cutoff tie.
        if np.any(missing) and np.max(abs(old_scores[q,missing]-scores[q,-1]))>1e-6:
            raise ValueError('Official Top-100 missing outside a GPU Top-200 cutoff tie')
        if np.max(np.abs(old_scores[q]-scores[q,:100]))>2e-5:
            raise ValueError('Official Top-100 scores disagree with GPU exact scores')
        tail_ids,tail_scores=ids[q,extra][:100],scores[q,extra][:100]
        if tail_scores[0]>old_scores[q,-1]+2e-5:
            raise ValueError('Ground truth boundary/order disagreement exceeds roundoff tolerance')
        result_ids[q,:100]=old_ids[q];result_ids[q,100:]=tail_ids
        result_scores[q,:100]=old_scores[q];result_scores[q,100:]=tail_scores
    return result_ids,result_scores


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--gpu',type=int,default=2)
    args=parser.parse_args()
    source=Path(args.source).resolve(strict=True);out=Path(args.output).resolve()
    c=json.loads((source/'configuration.json').read_text())['config']
    query_path=source/'data/test.fbin';base=matrix(c['base']);queries=matrix(query_path)
    old_ids=matrix(source/'data/truth.ids.ibin','<u4');old_scores=matrix(source/'data/truth.scores.fbin')
    if base.shape!=(4828191,512) or queries.shape!=(c['test_query_limit'],512) or old_ids.shape!=(len(queries),100):
        raise ValueError('Expected frozen native RS5M 512D and official Top-100')
    stamp=identity(dict(base=file_id(c['base']),queries=content_sha256(query_path),
        official_ids=content_sha256(source/'data/truth.ids.ibin'),
        official_scores=content_sha256(source/'data/truth.scores.fbin'),gpu=args.gpu,k=200,
        runner=file_id(Path(__file__))))
    out.mkdir(parents=True,exist_ok=True)
    meta_path=out/'truth.metadata.json';paths=[out/'truth.ids.ibin',out/'truth.scores.fbin']
    if meta_path.exists():
        meta=json.loads(meta_path.read_text())
        if (meta.get('fingerprint')==stamp and all(path.is_file() for path in paths)
                and meta['outputs']=={str(path):content_sha256(path) for path in paths}):
            print('[reuse] verified full-database Top-200 diagnostic truth',flush=True);return
        raise ValueError('Existing diagnostic truth changed; use a new output directory')
    os.environ['NVIDIA_TF32_OVERRIDE']='0'
    import faiss
    if not hasattr(faiss,'StandardGpuResources') or not 0<=args.gpu<faiss.get_num_gpus():
        raise RuntimeError('Selected GPU Faiss is unavailable')
    faiss.omp_set_num_threads(8)
    resources=faiss.StandardGpuResources();resources.setTempMemory(512<<20)
    config=faiss.GpuIndexFlatConfig();config.device=args.gpu;config.useFloat16=False
    index=faiss.GpuIndexFlatIP(resources,512,config)
    started=time.monotonic();batch=65536
    for lo in range(0,len(base),batch):
        hi=min(len(base),lo+batch);index.add(np.ascontiguousarray(base[lo:hi],dtype=np.float32))
        if hi==len(base) or (lo//batch)%8==0:
            elapsed=time.monotonic()-started
            print(f'[truth-upload] {hi}/{len(base)}; elapsed/estimated_upload_total='
                  f'{elapsed:.1f}s/{elapsed*len(base)/hi:.1f}s',flush=True)
    shape=(len(queries),200)
    part_ids=out/'truth.ids.partial.ibin';part_scores=out/'truth.scores.partial.fbin'
    checkpoint=out/'checkpoint.json';done=0
    if checkpoint.exists():
        old=json.loads(checkpoint.read_text())
        if old['fingerprint']!=stamp or not part_ids.exists() or not part_scores.exists():
            raise ValueError('Diagnostic truth checkpoint inputs changed')
        done=old['done']
    else:
        write_matrix(part_ids,np.zeros(shape,dtype=np.uint32),'<u4')
        write_matrix(part_scores,np.zeros(shape,dtype=np.float32))
    labels=np.memmap(part_ids,mode='r+',dtype='<u4',offset=8,shape=shape)
    values=np.memmap(part_scores,mode='r+',dtype='<f4',offset=8,shape=shape)
    search_started=time.monotonic();initial=done
    for lo in range(done,len(queries),128):
        hi=min(len(queries),lo+128)
        scores,ids=index.search(np.ascontiguousarray(queries[lo:hi],dtype=np.float32),200)
        labels[lo:hi],values[lo:hi]=preserve_top100(old_ids[lo:hi],old_scores[lo:hi],ids,scores)
        labels.flush();values.flush()
        save_json(checkpoint,dict(fingerprint=stamp,done=hi,total=len(queries)))
        elapsed=time.monotonic()-search_started
        print(f'[truth-top200] {hi}/{len(queries)}; elapsed/estimated_search_total='
              f'{elapsed:.1f}s/{elapsed*(len(queries)-initial)/(hi-initial):.1f}s',flush=True)
    max_error=0.
    for q in np.unique(np.linspace(0,len(queries)-1,16).astype(int)):
        actual=np.asarray(base[labels[q]],dtype=np.float64)@np.asarray(queries[q],dtype=np.float64)
        max_error=max(max_error,float(np.max(abs(actual-values[q]))))
    if max_error>2e-5: raise ValueError('Diagnostic Top-200/raw-score validation failed')
    del labels,values,index,resources
    part_ids.replace(paths[0]);part_scores.replace(paths[1])
    save_json(meta_path,dict(fingerprint=stamp,full_database=True,base_rows=len(base),base_dim=512,
        queries=len(queries),depth=200,score='fp32-inner-product',backend='GPU Faiss FlatIP, float32',
        gpu=args.gpu,original_top100_unchanged=True,diagnostic_only=True,qps_measured=False,
        cutoff_ties='official IDs retained; GPU tail chooses arbitrary tied IDs',tf32_disabled=True,
        sampled_raw_fp64_max_error=max_error,queries_sha256=content_sha256(query_path),
        outputs={str(path):content_sha256(path) for path in paths}))
    print(f'[complete] diagnostic Top-200; elapsed={time.monotonic()-started:.1f}s; official Top-100 unchanged',flush=True)


if __name__=='__main__':
    main()
