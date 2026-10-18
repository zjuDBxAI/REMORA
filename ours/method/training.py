"""Final raw-score/Balanced Product/centered/support-aware REMORA trainer.

One fixed B-bit code per node. Test queries/GT are never read by this program.
The budget study uses fixed 8-bit groups; it is NOT the historical DP allocator.
"""
import json
from pathlib import Path
import time
import numpy as np
from protocol import matrix, write_csv
from rotation_product import encode, CenteredObjective, descent_step
from representation_capacity import predict, metrics, mm_reencode
from train_terminal_ranking_ji import Pools
from qvrr import Primary
from faiss_backend import FaissProductEncoder, resolved_device
from .alignment import fit_alignment, load_alignment, padded, signature
from .encoding import encode_database, export_model


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


def ablation_penalty(mass, neff, deff, variant):
    """Equal total prior strength; calibration uses training support only."""
    adaptive = mass * deff / np.maximum(neff, 1e-30)
    strength = float(adaptive.sum() / max(float(mass.sum()), 1e-30))
    if variant == 'remora':
        return adaptive, strength
    if variant in ('uncentered-fixed', 'centered-fixed'):
        return mass * strength, strength
    return np.zeros(len(mass)), strength


def targets_without_decoder_bias(pools, queries, mean):
    bias=np.asarray(queries,dtype=np.float32)@np.asarray(mean,dtype=np.float32)
    return Pools(pools.query_ids,pools.beams,pools.offsets,pools.node_ids,
        pools.targets-np.repeat(bias[pools.query_ids],np.diff(pools.offsets)),
        pools.exact_scores,pools.primary_scores)


def validation(pools,codes,features,books,device):
    prediction=predict(pools,codes,features,books,'product',device)
    _,rows=metrics(pools,prediction)
    return dict(fixed_pool_recall=float(np.mean([r['fixed_pool_recall'] for r in rows])),
                centered_mse=float(np.mean([r['centered_mse'] for r in rows])))


def train(config,method,model_path,codes_path,train_pool,valid_pool,bits,variant,destination,*,alignment_path=None):
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
    if not 0 < bits <= 256 or bits % 8 or variant not in ('euclidean','alignment','balanced','uncentered-fixed','centered-fixed','centered','remora'):
        raise ValueError('Unsupported REMORA budget/ablation')
    groups=bits//8; p=Primary(model_path,codes_path); base=matrix(c['base'])
    if base.shape!=(p.nodes,p.dim): raise ValueError('Full base/model mismatch')
    query_path=c.get('alignment_queries',Path(c['output_root'])/'data/metric.fbin')
    queries=matrix(query_path)
    valid_queries=matrix(c.get('validation_queries',Path(c['output_root'])/'data/select.fbin'))
    pools,vp=load_pools(train_pool),load_pools(valid_pool)
    d=((p.dim+groups-1)//groups)*groups; started=time.monotonic()
    if alignment_path is None:
        state=fit_alignment(c,p,base,queries,bits,variant,method)
    else:
        state=load_alignment(alignment_path,signature(c,method,model_path,codes_path,query_path,bits,variant))
        print(f'[alignment] loaded {alignment_path}',flush=True)
    mean,forward,inverse,spec,rot,books=(state[k] for k in ('mean','forward','inverse','spectrum','rotation','books'))
    deff=float(spec.sum()**2/np.square(spec).sum()); diag=state['diagnostics']
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
    penalty,fixed_strength=ablation_penalty(mass,neff,deff,variant)
    centered_loss=variant!='uncentered-fixed'
    conditioned=variant in ('uncentered-fixed','centered-fixed','centered','remora')
    # The decoder adds q.T mean separately. Centering cancels that pool-wide
    # offset, but uncentered fitting must remove it from its target explicitly.
    fit_pools=pools
    if variant=='uncentered-fixed':
        fit_pools=targets_without_decoder_bias(pools,queries,mean)
    # Keep a validation-only node's code a geometric fallback. A node seen in
    # both splits still has ONE train-fitted code, never a query-specific code.
    unseen_valid=np.setdiff1d(val_nodes,seen); unseen_y=yy[np.searchsorted(union,unseen_valid)]
    def check(b,r):
        if len(unseen_valid): codes[unseen_valid]=geometry(unseen_y,r,b)
        return validation(vp,codes,vf@r,b,device)
    before=check(books,rot); best=dict(books=books.copy(),rot=rot.copy(),codes=codes[seen].copy(),validation=before,epoch=0)
    history=[dict(epoch=0,**before,elapsed_s=time.monotonic()-started)]
    if conditioned:
        selected=seen[np.argsort(-mass,kind='stable')[:c['recode_nodes']]]
        engine=CenteredObjective(fit_pools,features,seen,train_y,penalty,device,c['ji_batch_size'],
                                 centered_loss=centered_loss)
        for epoch in range(1,c['centered_epochs']+1):
            books,rot,bupdate=descent_step(engine,books,rot,codes,'books')
            books,rot,rupdate=descent_step(engine,books,rot,codes,'rotation')
            recode=mm_reencode(fit_pools,codes,features@rot,books,'product',seen,train_y@rot,penalty,selected,device,
                              centered_loss=centered_loss)
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
    encoding_info=encode_database(c,dest,method,bits,variant,p.nodes,books,rot,final_projection,common,
        state['audit_values'],targets,seen,codes,device,encoder_device,encoder_backend,started)
    encoder_audit=encoding_info['audit']
    np.savez_compressed(dest/'model.npz',books=books,projection=final_projection,mean=mean,forward=forward,rotation=rot,spectrum=spec)
    write_csv(dest/'training.csv',history)
    summary=dict(method=method,variant=variant,budget_bits=bits,node_count=p.nodes,raw_dim=p.dim,
        model_dim=d,groups=groups,bits_per_group=8,
        allocator='euclidean-product' if variant=='euclidean' else 'ordered-ji-product' if variant=='alignment' else 'balanced-product',
        full_score_residual=True,primary_projection_loss_included=True,
        alignment_checkpoint=str(Path(alignment_path).resolve()) if alignment_path else None,
        kmeans_device=kmeans_device,geometric_encoder=encoder_backend,geometric_encoder_device=encoder_device,
        encoding_audit=encoder_audit,encoding_assignments=encoding_info['assignments'],
        encoding_ambiguous_assignments=encoding_info['ambiguous_assignments'],
        train_pools=pools.pool_count,validation_pools=vp.pool_count,seen_nodes=len(seen),
        effective_dimension=deff,support_neff_quantiles=np.quantile(neff,[0,.25,.5,.75,1]).tolist(),
        support_formula='penalty_i = pool_mass_i * D_eff / n_eff_i; repeated beams merged by query',
        centered=conditioned and centered_loss,support_regularization=variant=='remora',
        candidate_conditioned=conditioned,
        terminal_loss='centered-score-mse' if conditioned and centered_loss else 'score-mse' if conditioned else 'none',
        regularization='support-adaptive' if variant=='remora' else 'fixed' if variant in ('uncentered-fixed','centered-fixed') else 'none',
        fixed_regularization_strength=fixed_strength,
        geometric_penalty_sum=float(penalty.sum()),
        fixed_regularization_rule='training mass-weighted mean of D_eff/n_eff; same total prior weight as Full',
        initial_validation=before,best_validation=best['validation'],best_epoch=best['epoch'],
        validation_metric='agreement with exact Top-10 WITHIN retained pool, not global-GT Recall',
        test_queries_used=False,raw_vectors_required_online=False,
        side_bytes=(dest/'side.bin').stat().st_size,model_bytes=(dest/'online.bin').stat().st_size,
        elapsed_s=time.monotonic()-started,geometry_diagnostics=diag)
    (dest/'training.json').write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n')
    print(f'[done] REMORA {dest}',flush=True)
