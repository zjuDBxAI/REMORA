"""Small actual-primary geometry audit, not full training or performance timing."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from protocol import ROOT,load_config,matrix,write_csv
sys.path[:0]=[str(ROOT/'ours/python'),str(ROOT/'ours/test/t2i-1M/python')]
from faiss_backend import FaissProductEncoder,resolved_device,train_kmeans
from joint_innovation_rpq import build_joint_transform
from rotation_product import balanced_rotation
from qvrr import Primary,select_runs


def validate(config,output,device):
    import faiss
    import torch
    torch.set_num_threads(1); faiss.omp_set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    device=resolved_device(device,device)
    c=load_config(config); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    raw=matrix(c['base']); queries=matrix(c['train_queries'])[:256]
    w=np.asarray(queries,np.float64).T@np.asarray(queries,np.float64)/len(queries)
    rng=np.random.default_rng(20261004)
    ids=rng.choice(len(raw),2048,replace=False); rows=[]
    for name,r in select_runs(c['graph_package'],c['methods']).items():
        p=Primary(r['paths']['model'],r['paths']['codes'])
        residual=np.asarray(raw[ids])-p.reconstruct_raw(ids)
        mean,forward,_,spectrum,_=build_joint_transform(residual,w,1e-6)
        groups=8; dim=((p.dim+groups-1)//groups)*groups; width=dim//groups
        y=np.zeros((len(residual),dim),np.float32)
        y[:,:p.dim]=(residual-mean)@forward
        rotation=balanced_rotation(np.pad(spectrum,(0,dim-p.dim)),groups)
        rotated=y@rotation
        # Independent holdout nodes; only training queries define the metric.
        books=np.stack([train_kmeans(rotated[:1536,g*width:(g+1)*width],256,3,71+g,device)
                        for g in range(groups)])
        gpu=FaissProductEncoder(rotation,books,device,device)
        audit=gpu.audit(y[1536:],128)
        cpu=FaissProductEncoder(rotation,books,'cpu',device)
        labels=gpu.encode(y[1536:],128)
        mismatch=int(np.count_nonzero(labels!=cpu.encode(y[1536:],128)))
        if mismatch: raise RuntimeError(f'{name}: CPU/GPU frozen-codebook encoding mismatch: {mismatch}')
        row=dict(primary=name,**audit,cpu_gpu_mismatches=mismatch,
                 kmeans_device=device,faiss_version=faiss.__version__,
                 faiss_compile_options=faiss.get_compile_options(),
                 fit_nodes=1536,holdout_nodes=512,side_bits=64,
                 scope='actual-primary residual geometry audit; no centered training or QPS')
        rows.append(row)
        (out/(name+'-faiss.json')).write_text(json.dumps(row,indent=2,sort_keys=True)+'\n')
        print(f'[faiss-audit] {name}: canonical=0; CPU/GPU={mismatch}; '
              f'Torch agreement={audit["faiss_torch_agreement"]:.9f}',flush=True)
    write_csv(out/'faiss-encoding.csv',rows)


if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--config')
    parser.add_argument('--output',required=True); parser.add_argument('--device',default='cuda:2')
    args=parser.parse_args(); validate(args.config,args.output,args.device)
