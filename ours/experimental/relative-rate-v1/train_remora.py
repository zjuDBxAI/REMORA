"""Final raw-score/Balanced Product/centered/support-aware REMORA trainer.

One fixed B-bit code per node. Test queries/GT are never read by this program.
The budget study uses fixed 8-bit groups; it is NOT the historical DP allocator.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys
import time
import numpy as np
from protocol import ROOT, matrix, write_csv
sys.path[:0]=[str(ROOT/'ours/python'),str(ROOT/'ours/test/t2i-1M/python')]
from joint_innovation_rpq import build_joint_transform
from rotation_product import balanced_rotation, encode, CenteredObjective, descent_step
from representation_capacity import predict, metrics, mm_reencode
from train_terminal_ranking_ji import Pools
from qvrr import Primary
from faiss_backend import FaissProductEncoder, resolved_device, train_kmeans


def load_pools(path):
    with np.load(path,allow_pickle=False) as z:
        return Pools(*(z[k] for k in ('query_ids','beams','offsets','node_ids','targets','exact_scores','primary_scores')))


def support(pools,nodes):
    lengths=np.diff(pools.offsets); q=np.repeat(pools.query_ids,lengths)
    w=np.repeat(1./lengths,lengths); order=np.lexsort((q,pools.node_ids))
    ids=pools.node_ids[order]; qi=q[order]
    starts=np.r_[0,1+np.flatnonzero((ids[1:]!=ids[:-1])|(qi[1:]!=qi[:-1]))]
    merged=np.add.reduceat(w[order],starts); ni=ids[starts]
    local=np.searchsorted(nodes,ni)
    mass=np.bincount(local,weights=merged,minlength=len(nodes))
    square=np.bincount(local,weights=merged**2,minlength=len(nodes))
    neff=mass**2/np.maximum(square,1e-30)
    return mass,neff


def padded(x,d):
    out=np.zeros((len(x),d),np.float32); out[:,:x.shape[1]]=x; return out


def validation(pools,codes,features,books,device):
    prediction=predict(pools,codes,features,books,'product',device)
    _,rows=metrics(pools,prediction)
    return dict(fixed_pool_recall=float(np.mean([r['fixed_pool_recall'] for r in rows])),
                centered_mse=float(np.mean([r['centered_mse'] for r in rows])))


def export_model(dest,books,projection,mean,codes_path,nodes,bits,model_id):
    groups,_,width=books.shape; d=len(projection)
    artifact=int(model_id[:16],16); mid=int(model_id[16:32],16)
    with (dest/'online.bin.tmp').open('wb') as f:
        f.write(struct.pack('<8sIIIIQQQQ',b'JIONL002',2,d,groups,bits,nodes,artifact,mid,groups*256))
        f.write(np.asarray(mean,dtype='<f4').tobytes())
        f.write(np.ascontiguousarray(projection.T,dtype='<f4').tobytes())
        for g in range(groups): f.write(struct.pack('<IIIIQQ',g*8,8,256,width,g*256*width,g*width))
        f.write(np.arange(d,dtype='<u4').tobytes()); f.write(np.asarray(books,dtype='<f4').tobytes())
    (dest/'online.bin.tmp').replace(dest/'online.bin')
    with (dest/'side.bin.tmp').open('wb') as f,Path(codes_path).open('rb') as src:
        # v3 stores byte-aligned 8-bit groups densely, without 64-bit padding.
        f.write(struct.pack('<8sIQIIIQQ',b'JIRCODE1',3,nodes,d,bits,groups,artifact,mid))
        while True:
            chunk=src.read(8<<20)
            if not chunk: break
            f.write(chunk)
    (dest/'side.bin.tmp').replace(dest/'side.bin')


def train(config,method,model_path,codes_path,train_pool,valid_pool,bits,variant,destination):
    import faiss
    import torch
    c=json.loads(Path(config).read_text()); dest=Path(destination); dest.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(c['trainer_threads']); faiss.omp_set_num_threads(c['trainer_threads'])
    device=c['device']; torch.manual_seed(c['seed'])
    if device.startswith('cuda'):
        if not torch.cuda.is_available(): raise RuntimeError('CUDA requested but unavailable')
        torch.backends.cuda.matmul.allow_tf32=False
    kmeans_device = resolved_device(c.get('kmeans_device', 'auto'), device)
    encoder_device = resolved_device(c.get('geometric_encoder_device', 'auto'), device)
    encoder_backend = c.get('geometric_encoder', 'faiss')
    if encoder_backend not in ('faiss', 'torch'):
        raise ValueError('geometric_encoder must be faiss or torch')
    print(f'[backends] kmeans={kmeans_device} geometry={encoder_backend}/{encoder_device} centered={device}', flush=True)
    if not 0 < bits <= 512 or bits % 8 or variant not in ('euclidean','alignment','balanced','centered','remora'):
        raise ValueError('Unsupported REMORA budget/ablation')
    groups=bits//8; p=Primary(model_path,codes_path); base=matrix(c['base'])
    if base.shape!=(p.nodes,p.dim): raise ValueError('Full base/model mismatch')
    queries=matrix(Path(c['output_root'])/'data/metric.fbin')
    valid_queries=matrix(Path(c['output_root'])/'data/select.fbin')
    pools,vp=load_pools(train_pool),load_pools(valid_pool)
    d=((p.dim+groups-1)//groups)*groups; width=d//groups; started=time.monotonic()
    rng=np.random.default_rng(c['seed']); fit=rng.choice(p.nodes,min(c['residual_sample_size'],p.nodes),replace=False)
    residual=np.asarray(base[fit])-p.reconstruct_raw(fit)
    W=np.eye(p.dim) if variant=='euclidean' else np.asarray(queries,np.float64).T@np.asarray(queries,np.float64)/len(queries)
    mean,forward,inverse,spectrum,diag=build_joint_transform(residual,W,1e-6)
    spec=np.pad(spectrum,(0,d-p.dim)); deff=float(spec.sum()**2/np.square(spec).sum())
    rot=np.eye(d,dtype=np.float32) if variant in ('euclidean','alignment') else balanced_rotation(spec,groups)
    y=padded((residual-mean)@forward,d); rotated=y@rot
    books=np.empty((groups,256,width),np.float32)
    for g in range(groups):
        books[g] = train_kmeans(rotated[:,g*width:(g+1)*width],256,c['ji_iterations'],c['seed']+g,kmeans_device)
        print(f'[global] {method}/b{bits}/{variant} group={g+1}/{groups} elapsed={time.monotonic()-started:.1f}s',flush=True)
    projection=np.zeros((d,d),np.float32); projection[:p.dim,:p.dim]=inverse.T
    def targets(ids):
        result=np.empty((len(ids),d),np.float32)
        for lo in range(0,len(ids),c['ji_batch_size']):
            part=ids[lo:lo+c['ji_batch_size']]
            result[lo:lo+len(part)]=padded((np.asarray(base[part])-p.reconstruct_raw(part)-mean)@forward,d)
        return result
    seen=np.unique(pools.node_ids); val_nodes=np.unique(vp.node_ids)
    union=np.union1d(seen,val_nodes); yy=targets(union); train_y=yy[np.searchsorted(union,seen)]
    codes=np.zeros((p.nodes,groups),np.uint8)
    def geometry(values, rotation, codebooks):
        if encoder_backend == 'torch':
            return encode(values, rotation, codebooks, device, c['ji_batch_size'])
        return FaissProductEncoder(rotation,codebooks,encoder_device,device).encode(values,c['ji_batch_size'])
    codes[union]=geometry(yy,rot,books)
    features=padded(queries,d)@projection; vf=padded(valid_queries,d)@projection
    mass,neff=support(pools,seen)
    penalty=mass*deff/np.maximum(neff,1e-30) if variant=='remora' else np.zeros(len(seen))
    # Keep a validation-only node's code a geometric fallback. A node seen in
    # both splits still has ONE train-fitted code, never a query-specific code.
    unseen_valid=np.setdiff1d(val_nodes,seen); unseen_y=yy[np.searchsorted(union,unseen_valid)]
    def check(b,r):
        if len(unseen_valid): codes[unseen_valid]=geometry(unseen_y,r,b)
        return validation(vp,codes,vf@r,b,device)
    before=check(books,rot); best=dict(books=books.copy(),rot=rot.copy(),codes=codes[seen].copy(),validation=before,epoch=0)
    history=[dict(epoch=0,**before,elapsed_s=time.monotonic()-started)]
    if variant in ('centered','remora'):
        selected=seen[np.argsort(-mass,kind='stable')[:c['recode_nodes']]]
        engine=CenteredObjective(pools,features,seen,train_y,penalty,device,c['ji_batch_size'])
        for epoch in range(1,c['centered_epochs']+1):
            books,rot,bupdate=descent_step(engine,books,rot,codes,'books')
            books,rot,rupdate=descent_step(engine,books,rot,codes,'rotation')
            recode=mm_reencode(pools,codes,features@rot,books,'product',seen,train_y@rot,penalty,selected,device)
            current=check(books,rot)
            key=lambda v:(v['fixed_pool_recall'],-v['centered_mse'])
            if key(current)>key(best['validation']): best=dict(books=books.copy(),rot=rot.copy(),codes=codes[seen].copy(),validation=current,epoch=epoch)
            elapsed=time.monotonic()-started
            history.append(dict(epoch=epoch,**current,best_epoch=best['epoch'],elapsed_s=elapsed,
                                books_update=bupdate['accepted'],rotation_update=rupdate['accepted'],
                                changed=sum(r['changed'] for r in recode if r['accepted'])))
            print(f'[centered] {method}/b{bits}/{variant} epoch={epoch}/{c["centered_epochs"]} validation={current["fixed_pool_recall"]:.6f} best={best["epoch"]} elapsed={elapsed:.1f}s',flush=True)
    books,rot=best['books'],best['rot']; codes[seen]=best['codes']
    final_projection=projection@rot; common=padded(mean[None,:],d)[0]
    frozen_encoder = FaissProductEncoder(rot,books,encoder_device,device) if encoder_backend == 'faiss' else None
    encoder_audit = frozen_encoder.audit(y[:min(len(y),c.get('encoding_audit_size',1024))],c['ji_batch_size']) if frozen_encoder else dict(backend='torch')
    if frozen_encoder:
        (dest/'encoding-parity.json').write_text(json.dumps(encoder_audit,indent=2,sort_keys=True)+'\n')
        print(f'[encoding-audit] Faiss/canonical mismatches={encoder_audit["faiss_canonical_mismatches"]}; Faiss/torch agreement={encoder_audit["faiss_torch_agreement"]:.9f}',flush=True)
    payload=dest/'codes.uint8.tmp'
    # Stream ALL nodes through the frozen model; no N*d reconstructed FP32 file.
    with payload.open('wb') as f:
        for lo in range(0,p.nodes,c['encode_batch_size']):
            hi=min(p.nodes,lo+c['encode_batch_size']); ids=np.arange(lo,hi,dtype=np.int64)
            target_block=targets(ids)
            labels=frozen_encoder.encode(target_block,c['ji_batch_size']) if frozen_encoder else encode(target_block,rot,books,device,c['ji_batch_size'])
            hit=seen[(seen>=lo)&(seen<hi)]; labels[hit-lo]=codes[hit]
            labels.tofile(f)
            elapsed=time.monotonic()-started
            print(f'[encode] {method}/b{bits}/{variant} {hi}/{p.nodes} elapsed={elapsed:.1f}s',flush=True)
    digest=hashlib.sha256(books.tobytes()+final_projection.tobytes()+common.tobytes()+payload.read_bytes()).hexdigest()
    export_model(dest,books,final_projection,common,payload,p.nodes,bits,digest)
    payload.unlink()  # only this newly generated temporary payload; codes live in side.bin
    np.savez_compressed(dest/'model.npz',books=books,projection=final_projection,mean=mean,forward=forward,rotation=rot,spectrum=spec)
    write_csv(dest/'training.csv',history)
    summary=dict(method=method,variant=variant,budget_bits=bits,node_count=p.nodes,raw_dim=p.dim,
        model_dim=d,groups=groups,bits_per_group=8,allocator='balanced-product',
        full_score_residual=True,primary_projection_loss_included=True,
        kmeans_device=kmeans_device,geometric_encoder=encoder_backend,geometric_encoder_device=encoder_device,
        encoding_audit=encoder_audit,encoding_assignments=frozen_encoder.assignments if frozen_encoder else None,
        encoding_ambiguous_assignments=frozen_encoder.ambiguous_assignments if frozen_encoder else None,
        train_pools=pools.pool_count,validation_pools=vp.pool_count,seen_nodes=len(seen),
        effective_dimension=deff,support_neff_quantiles=np.quantile(neff,[0,.25,.5,.75,1]).tolist(),
        support_formula='penalty_i = pool_mass_i * D_eff / n_eff_i; repeated beams merged by query',
        centered=variant in ('centered','remora'),support_regularization=variant=='remora',
        initial_validation=before,best_validation=best['validation'],best_epoch=best['epoch'],
        validation_metric='agreement with exact Top-10 WITHIN retained pool, not global-GT Recall',
        test_queries_used=False,raw_vectors_required_online=False,
        side_bytes=(dest/'side.bin').stat().st_size,model_bytes=(dest/'online.bin').stat().st_size,
        elapsed_s=time.monotonic()-started,geometry_diagnostics=diag)
    (dest/'training.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(f'[done] REMORA {dest}',flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser(); a.add_argument('--config',required=True); a.add_argument('--method',required=True)
    a.add_argument('--model',required=True); a.add_argument('--codes',required=True)
    a.add_argument('--train-pool',required=True); a.add_argument('--validation-pool',required=True)
    a.add_argument('--bits',type=int,required=True); a.add_argument('--variant',required=True); a.add_argument('--output',required=True)
    x=a.parse_args(); train(x.config,x.method,x.model,x.codes,x.train_pool,x.validation_pool,x.bits,x.variant,x.output)
