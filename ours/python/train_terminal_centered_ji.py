#!/usr/bin/env python3
"""Pool-centered shrinkage, using a separable tight majorizer per group.

F = sum_p mean((e_p-mean(e_p))**2) + sum_i mass_i*D_eff/n_eff_i*G_i.
Removing centering gives exactly the same node-wise minimizers as A1f.
For group score changes d, ||C(e-d)||^2 <= ||Ce-d||^2: hence independent
node updates on pseudo-target prediction+Ce majorize the fully coupled loss.
Every group recomputes pool means and checks the actual full objective.
"""
import argparse
import json
from pathlib import Path
import numpy as np
import torch
from drb_io import fbin_memmap
from joint_innovation_rpq import read_model, read_side_codes, write_model
from primary_quantizer_manifest import read_manifest
from train_terminal_alternating_ji import pools_to_events, evaluate
from train_terminal_conditioned_ji import (Events, active_codebooks, query_features,
    predict_events, unpack_all_codes, resolve_device, make_model, write_fixed_side)
from train_terminal_ranking_ji import load_pools, is_better
from train_terminal_shrinkage_ji import (effective_query_counts, participation_dimension,
    innovation_targets, reencode_with_shrinkage)
from terminal_centered_refinement import exact_refine


def pool_weights(pools, topk=10, window=5, multiplier=1.0):
    """Training-only exact *within-pool* ranks; every pool has unit mass.

    Ranks K-h+1,...,K+h get multiplier; all other ranks get weight one.
    This is deliberately a weighted variance, not a boundary-pair loss.
    Exact-score ties use node IDs for deterministic ordering.
    """
    if topk < 1 or window < 1 or not np.isfinite(multiplier) or multiplier < 1:
        raise ValueError('Invalid boundary weight settings')
    lengths = np.diff(pools.offsets)
    if np.any(lengths <= 0):
        raise ValueError('Empty candidate pool')
    weights = np.repeat(1.0/lengths, lengths)
    if multiplier == 1:
        return weights
    for begin, end in zip(pools.offsets[:-1], pools.offsets[1:]):
        rank = np.lexsort((pools.node_ids[begin:end], -pools.exact_scores[begin:end]))
        w = np.ones(end-begin, np.float64)
        w[rank[max(0,topk-window):min(end-begin,topk+window)]] = multiplier
        weights[begin:end] = w/w.sum()
    return weights


def centered(error, offsets, weights=None):
    lengths = np.diff(offsets)
    error = np.asarray(error, np.float64)
    if weights is None:
        means = np.add.reduceat(error, offsets[:-1]) / lengths
    else:
        means = (np.add.reduceat(weights*error, offsets[:-1])
                 / np.add.reduceat(weights, offsets[:-1]))
    return error - np.repeat(means, lengths)


def objective(error, offsets, targets, codes, nodes, books, penalty, weights=None):
    ce = centered(error, offsets, weights)
    loss = (float(np.sum(np.add.reduceat(ce**2, offsets[:-1]) / np.diff(offsets)))
            if weights is None else float(np.dot(weights, ce*ce)))
    geom = np.zeros(len(nodes), np.float64)
    for g in range(len(books)):
        diff = targets[:, g].astype(np.float64) - books[g][codes[nodes, g]]
        geom += np.sum(diff**2, axis=1)
    prior = float(np.dot(penalty, geom))
    return dict(total=loss+prior, terminal=loss, geometry=prior)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('primary-manifest','initial-model','initial-side-codes','pool-cache','out-dir'):
        p.add_argument('--'+name,required=True)
    p.add_argument('--device',default='cuda'); p.add_argument('--batch-size',type=int,default=65536)
    p.add_argument('--innovation-batch-size',type=int,default=16384)
    p.add_argument('--max-assignment-score-elements',type=int,default=2000000)
    p.add_argument('--sweeps',type=int,default=1)
    p.add_argument('--patience',type=int,default=0,help='0 evaluates all requested sweeps')
    p.add_argument('--refine-nodes',type=int,default=0)
    p.add_argument('--terminal-guard',action='store_true',
                   help='Require both total and centered terminal training loss not to rise')
    p.add_argument('--boundary-multiplier',type=float,default=1.0)
    p.add_argument('--initial-stage',default='a1d')
    p.add_argument('--topk',type=int,default=10); p.add_argument('--rank-max',type=int,default=100)
    p.add_argument('--boundary-window',type=int,default=5)
    p.add_argument('--validation-boundary-pools',type=int,default=12000)
    args=p.parse_args()
    if args.sweeps<1 or args.patience<0 or args.refine_nodes<0: raise ValueError('Invalid optimization limits')
    device=resolve_device(args.device); primary=read_manifest(args.primary_manifest)
    model=read_model(args.initial_model)
    if model.artifact_id!=primary.artifact_id or model.groups!=8 or model.budget_bits!=64 or np.any(model.bits!=8):
        raise ValueError('Requires matching frozen 8x8-bit representation')
    books=active_codebooks(model); bt=torch.as_tensor(books,device=device)
    codes=unpack_all_codes(read_side_codes(args.initial_side_codes,model),model)
    original=codes.copy()
    phi=torch.as_tensor(query_features(fbin_memmap(primary.query_fbin),model),device=device)
    train,_=load_pools(Path(args.pool_cache)/'train'); valid,_=load_pools(Path(args.pool_cache)/'validation')
    events=pools_to_events(train); ve=pools_to_events(valid)
    if np.intersect1d(train.query_ids,valid.query_ids).size: raise ValueError('Query split leakage')
    nodes=np.unique(events.node_ids)
    neff=effective_query_counts(events,primary.node_count)
    deff=participation_dimension(model.innovation_eigenvalues)
    alpha=neff/(neff+deff)
    mass=np.bincount(events.node_ids,weights=events.weights,minlength=primary.node_count)
    penalty=mass[nodes]*(1-alpha[nodes])/alpha[nodes]
    # Freeze the original uniform-pool shrinkage prior. The boundary ablation
    # changes ONLY the terminal weights, not support counts or geometry penalty.
    weights=pool_weights(train,args.topk,args.boundary_window,args.boundary_multiplier)
    validation_weights=pool_weights(valid,args.topk,args.boundary_window,args.boundary_multiplier)
    if args.boundary_multiplier != 1:
        events=Events(events.query_ids,events.node_ids,weights.astype(np.float32),events.targets)
        weighted_mass=np.bincount(events.node_ids,weights=weights,minlength=primary.node_count)
        alpha=alpha.copy()
        alpha[nodes]=weighted_mass[nodes]/(weighted_mass[nodes]+penalty)
    # Preserve exact backward-compatible arithmetic in the unweighted variant.
    objective_weights=weights if args.boundary_multiplier != 1 else None
    yf=innovation_targets(primary,model,nodes,args.innovation_batch_size,device)
    targets=np.stack([yf[:,model.indices(g)] for g in range(model.groups)],axis=1); del yf
    def metrics():
        result=evaluate(valid,ve,codes,phi,books,args,device)
        pred=predict_events(ve,codes,phi,bt,args.batch_size,device)
        c=centered(ve.targets.astype(np.float64)-pred,valid.offsets)
        result['validation_centered_mse']=float(np.sum(ve.weights*c*c)/np.sum(ve.weights))
        wc=centered(ve.targets.astype(np.float64)-pred,valid.offsets,validation_weights)
        result['validation_weighted_centered_mse']=float(np.dot(validation_weights,wc*wc)/len(valid.query_ids))
        return result
    initial=metrics(); history=[]
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True)
    def save(name, labels):
        dest=out/name; dest.mkdir(parents=True,exist_ok=True)
        artifact=make_model(model,books,labels,'terminal-centered-shrinkage')
        write_model(str(dest/'joint_innovation_model.npz'),artifact)
        write_fixed_side(str(dest/'residual_side_codes.bin'),artifact,labels)
    best_metrics=initial; best_codes=original.copy(); best_stage=args.initial_stage; stale=0
    checkpoints=[]
    pred=predict_events(events,codes,phi,bt,args.batch_size,device)
    for sweep in range(args.sweeps):
        for g in range(model.groups):
            error=events.targets.astype(np.float64)-pred
            before=objective(error,train.offsets,targets,codes,nodes,books,penalty,objective_weights)
            pseudo=Events(events.query_ids,events.node_ids,events.weights,
                np.asarray(pred+centered(error,train.offsets,objective_weights),np.float32))
            old=codes[:,g].copy()
            reencode_with_shrinkage(pseudo,codes,phi,bt,nodes,targets,alpha,args.batch_size,
                args.max_assignment_score_elements,device,group_indices=[g])
            newpred=predict_events(events,codes,phi,bt,args.batch_size,device)
            after=objective(events.targets.astype(np.float64)-newpred,train.offsets,targets,codes,nodes,books,penalty,objective_weights)
            accepted=after['total']<=before['total']
            if args.terminal_guard: accepted=accepted and after['terminal']<=before['terminal']
            changes=int(np.count_nonzero(old!=codes[:,g]))
            if accepted: pred=newpred
            else: codes[:,g]=old
            history.append(dict(sweep=sweep,group=g,before=before,proposal=after,accepted=accepted,changed_nodes=changes,
                                total_nonincreasing=after['total']<=before['total'],
                                terminal_nonincreasing=after['terminal']<=before['terminal']))
            print(f'[centered] {sweep=} {g=} {changes=} {accepted=} objective={before["total"]:.9g}->{after["total"]:.9g}',flush=True)
        current=metrics(); stage=f'sweep-{sweep+1}'
        save(stage,codes)
        improved=is_better(current,best_metrics)
        checkpoints.append(dict(stage=stage,metrics=current,improved_validation=improved))
        if improved:
            best_metrics=current;best_codes=codes.copy();best_stage=stage;stale=0
        else: stale+=1
        (out/'checkpoint_metrics.json').write_text(json.dumps(checkpoints,indent=2,allow_nan=False))
        print(f'[checkpoint] {stage} recall={current["fixed_pool_recall"]} best={best_stage}',flush=True)
        if args.patience and stale>=args.patience: break
    refinement=None
    if args.refine_nodes:
        # A separate controlled branch from the validation-selected MM checkpoint.
        codes=best_codes.copy();save('pre-refine',codes)
        pred=predict_events(events,codes,phi,bt,args.batch_size,device)
        before=objective(events.targets.astype(np.float64)-pred,train.offsets,targets,codes,nodes,books,penalty,objective_weights)
        refinement=exact_refine(events,train.offsets,codes,phi.cpu().numpy(),books,
            nodes,targets,penalty,pred,args.refine_nodes,weights=objective_weights,terminal_guard=args.terminal_guard)
        pred=predict_events(events,codes,phi,bt,args.batch_size,device)
        after=objective(events.targets.astype(np.float64)-pred,train.offsets,targets,codes,nodes,books,penalty,objective_weights)
        if after['total']>before['total']+1e-6*max(1,abs(before['total'])):
            raise RuntimeError('Exact refinement increased full objective')
        if args.terminal_guard and after['terminal']>before['terminal']+1e-6*max(1,abs(before['terminal'])):
            raise RuntimeError('Guarded refinement increased terminal objective')
        refinement.update(before=before,after=after,source=best_stage)
        current=metrics();save('refined',codes)
        improved=is_better(current,best_metrics)
        checkpoints.append(dict(stage='refined',metrics=current,improved_validation=improved))
        if improved:best_metrics=current;best_codes=codes.copy();best_stage='refined'
    candidate=current; selected=best_stage!=args.initial_stage
    if np.any(codes[neff==0]!=original[neff==0]):raise RuntimeError('Unseen nodes changed')
    # Preserve the raw candidate for diagnosis, and the validation-selected model.
    for name, labels in [('candidate',codes),('selected',best_codes)]:save(name,labels)
    summary=dict(method='terminal_centered_shrinkage',effective_dimension=deff,budget_bits=64,
        objective='sum_pool weighted_variance(error) + frozen_uniform_node_geometry_penalty',
        weighting_note='Unit mass per pool; boundary ranks get fixed multiplier. Original uniform-pool support and geometry penalty remain frozen.',
        solver='group-wise separable tight majorization with full-objective acceptance',
        initial_validation=initial,candidate_validation=candidate,selected_by_validation=selected,
        selected_stage=best_stage,selected_validation=best_metrics,checkpoints=checkpoints,refinement=refinement,
        history=history,changed_nodes=int(np.any(codes!=original,axis=1).sum()),config=vars(args))
    (out/'training_summary.json').write_text(json.dumps(summary,indent=2,allow_nan=False))
    print(f'[done] {out} validation_selected={selected}',flush=True)


if __name__=='__main__':main()
