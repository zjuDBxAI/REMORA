"""Build paired supervision from retained PRIMARY traversal pools, not GT graphs."""
import argparse
from pathlib import Path
import struct
import sys
import numpy as np
from protocol import ROOT, matrix
sys.path.insert(0,str(ROOT/'ours/python'))
from train_terminal_ranking_ji import Pools


def read_trace(path):
    with Path(path).open('rb') as f:
        magic,version,start,nq,count=struct.unpack('<8sIQIQ',f.read(32))
        offsets=np.fromfile(f,dtype='<u8',count=nq+1)
        ids=np.fromfile(f,dtype='<u4',count=count)
        extra=f.read(1)
    if magic!=b'DRBVST01' or version!=1 or start!=0 or len(offsets)!=nq+1 or len(ids)!=count or extra or offsets[0]!=0 or offsets[-1]!=count or np.any(offsets[1:]<=offsets[:-1]):
        raise ValueError('Invalid retained CSR')
    return offsets.astype(np.int64),ids.astype(np.int64)


def save_pools(base_path,query_path,traces,output):
    base,queries=matrix(base_path),matrix(query_path)
    qids=[]; beams=[]; ids=[]; sizes=[]; exact=[]; primary=[]
    for beam,trace,score_path in traces:
        offsets,nodes=read_trace(trace); scores=matrix(score_path)
        if len(offsets)-1!=len(scores) or len(queries)<len(scores): raise ValueError('Pool count mismatch')
        for q in range(len(scores)):
            lo,hi=offsets[q:q+2]; selected=nodes[lo:hi]
            if len(np.unique(selected))!=len(selected) or np.any(selected>=len(base)):
                raise ValueError('Duplicate/invalid retained node')
            if len(selected)>scores.shape[1]: raise ValueError('Primary score matrix too narrow')
            qids.append(q); beams.append(beam); sizes.append(len(selected)); ids.append(selected)
            # Exact labels use RAW database/query vectors, with no renormalization.
            exact.append(np.asarray(base[selected])@queries[q]); primary.append(np.asarray(scores[q,:len(selected)]))
    pool=Pools(qids,beams,np.r_[0,np.cumsum(sizes)],np.concatenate(ids),
               np.concatenate(exact)-np.concatenate(primary),np.concatenate(exact),np.concatenate(primary))
    output=Path(output); output.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(output,**{k:getattr(pool,k) for k in ('query_ids','beams','offsets','node_ids','targets','exact_scores','primary_scores')})
    print(f'[pools] {output}: {pool.pool_count} query/beam pools, {pool.event_count} events',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--base',required=True); p.add_argument('--queries',required=True)
    p.add_argument('--trace',action='append',required=True,help='beam,CSR,score-fbin'); p.add_argument('--output',required=True)
    a=p.parse_args(); save_pools(a.base,a.queries,[(int(b),t,s) for b,t,s in (x.split(',',2) for x in a.trace)],a.output)
