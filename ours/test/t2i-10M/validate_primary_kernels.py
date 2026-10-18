"""Compare actual five-primary scalar/SIMD scores; exclude from paper QPS."""
import argparse
import json
from pathlib import Path
import subprocess
import struct
import numpy as np
from protocol import load_config,matrix,write_matrix,write_csv
from qvrr import select_runs


def validate(config,output):
    c=load_config(config); out=Path(output); out.mkdir(parents=True,exist_ok=True)
    primary=select_runs(c['graph_package'],c['methods'])
    raw_queries=matrix(c['test_queries']); queries=np.array(raw_queries[:32],copy=True)
    with Path(c['combined_ground_truth']).open('rb') as stream:
        n,k=struct.unpack('<II',stream.read(8))
    if Path(c['combined_ground_truth']).stat().st_size!=8+8*n*k:
        raise ValueError('Expected BigANN IDs+scores')
    gt=np.memmap(c['combined_ground_truth'],mode='r',dtype='<u4',offset=8,shape=(n,k))
    # BigANN's first section is IDs; load the selected block only.
    ids=np.concatenate([np.asarray(gt[:len(queries),:256]),np.full((len(queries),1),c['base_rows']-1,np.uint32)],axis=1)
    write_matrix(out/'queries.fbin',queries); write_matrix(out/'ids.ibin',ids,'<u4')
    results=[]
    for name,r in primary.items():
        print(f'[kernel-audit] {name}',flush=True)
        cmd=[str(Path(c['build_dir'])/'roar_qvrr_kernel_audit'),r['paths']['model'],r['paths']['codes'],str(out/'queries.fbin'),str(out/'ids.ibin')]
        audit=json.loads(subprocess.check_output(cmd,text=True))
        audit['primary']=name
        (out/f'{name}.json').write_text(json.dumps(audit,indent=2,sort_keys=True)+'\n')
        if audit['top10_set_different_queries']:
            raise RuntimeError(f'{name}: scalar/SIMD Top-10 mismatch; see report')
        results.append(audit)
    write_csv(out/'primary-kernels.csv',results)
    print(f'[done] {out/"primary-kernels.csv"}',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--config'); p.add_argument('--output',required=True)
    a=p.parse_args(); validate(a.config,a.output)
