"""Fixed-pool, fixed-rate representation diagnostics (not an ANN replacement).

Product and additive decoders use the SAME reordered JI coordinates. The latter
starts as an exact embedding of the product decoder, not a different subspace.
All nodes retain exactly eight uint8 indices, shared by every query.
"""
import numpy as np
import torch

from train_terminal_ranking_ji import Pools


def subset(pools, indices):
    indices = np.asarray(indices, dtype=np.int64)
    if not len(indices):
        raise ValueError('Empty pool selection')
    rows = [np.arange(*pools.bounds(int(i))) for i in indices]
    events = np.concatenate(rows)
    return Pools(pools.query_ids[indices], pools.beams[indices],
                 np.r_[0, np.cumsum([len(r) for r in rows])], pools.node_ids[events],
                 pools.targets[events], pools.exact_scores[events], pools.primary_scores[events])


def expand_product(books):
    groups, count, width = books.shape
    full = np.zeros((groups, count, groups*width), dtype=np.float32)
    for g in range(groups):
        full[g, :, g*width:(g+1)*width] = books[g]
    return full


def reconstruct(books, labels, kind):
    parts = [books[g][labels[:, g].long()] for g in range(len(books))]
    return torch.cat(parts, dim=1) if kind == 'product' else torch.stack(parts).sum(0)


def predict(pools, codes, features, books, kind, device='cpu', batch=65536):
    """LUT reference, evaluated outside all online timers."""
    bt = torch.as_tensor(books, device=device)
    unique, inverse = np.unique(pools.query_ids, return_inverse=True)
    lut_chunks = []
    with torch.no_grad():
        for start in range(0, len(unique), 512):
            f = torch.as_tensor(features[unique[start:start+512]], device=device)
            if kind == 'product':
                lut = torch.einsum('qgw,gkw->qgk', f.reshape(len(f), len(books), -1), bt)
            else:
                lut = torch.einsum('qd,gkd->qgk', f, bt)
            lut_chunks.append(lut.cpu().numpy())
    luts = np.concatenate(lut_chunks)
    owners = np.repeat(inverse, np.diff(pools.offsets))
    out = np.zeros(pools.event_count, dtype=np.float32)
    for start in range(0, pools.event_count, batch):
        end = min(start+batch, pools.event_count)
        c = codes[pools.node_ids[start:end]]
        for g in range(len(books)):
            out[start:end] += luts[owners[start:end], g, c[:, g]]
    return out


def objective(pools, prediction, codes, books, kind, nodes, targets, penalty, centered_loss=True):
    lengths = np.diff(pools.offsets)
    error = pools.targets.astype(np.float64)-prediction
    means = np.add.reduceat(error, pools.offsets[:-1])/lengths
    ce = error-np.repeat(means, lengths) if centered_loss else error
    terminal = float(np.sum(np.add.reduceat(ce*ce, pools.offsets[:-1])/lengths))
    geometry = 0.
    for start in range(0, len(nodes), 4096):
        sl = slice(start, start+4096)
        rec = reconstruct(torch.from_numpy(books), torch.from_numpy(codes[nodes[sl]]), kind).numpy()
        geometry += float(np.dot(penalty[sl], np.sum((rec.astype(np.float64)-targets[sl])**2, axis=1)))
    return dict(total=terminal+geometry, terminal=terminal, geometry=geometry)


def metrics(pools, prediction, topk=10, truth=None, query_offset=0):
    """Query/beam-balanced metrics. Exact score ties excluded from pair audits.

    fixed_pool_recall = agreement with exact top-k *within the pool*.
    recall_at_k = global-GT recall, only populated if external truth is supplied.
    Equal approximate scores count as half an inversion; top-k uses node ID ties.
    """
    rows = []
    for p in range(pools.pool_count):
        lo, hi = pools.bounds(p)
        ids = pools.node_ids[lo:hi]
        exact = pools.exact_scores[lo:hi]
        scores = pools.primary_scores[lo:hi]+prediction[lo:hi]
        erank = np.lexsort((ids, -exact)); arank = np.lexsort((ids, -scores))
        k = min(topk, hi-lo)
        error = pools.targets[lo:hi].astype(np.float64)-prediction[lo:hi]
        row = dict(query=int(pools.query_ids[p]), beam=int(pools.beams[p]),
                   fixed_pool_recall=len(set(ids[erank[:k]]) & set(ids[arank[:k]]))/k,
                   centered_mse=float(np.mean((error-error.mean())**2)))
        if truth is not None:
            q = int(pools.query_ids[p])-query_offset
            if not 0 <= q < len(truth):
                raise ValueError('Ground truth query offset mismatch')
            gt = set(map(int, truth[q, :topk]))
            row.update(recall_at_k=len(gt & set(ids[arank[:k]]))/topk,
                       retained_coverage=len(gt & set(ids))/topk)
        for window in (5, 10):
            left = erank[max(0, k-window):k]
            right = erank[k:min(hi-lo, k+window)]
            margin = exact[left, None].astype(np.float64)-exact[right][None, :]
            approx = scores[left, None].astype(np.float64)-scores[right][None, :]
            valid = margin > 0
            if not np.any(valid):
                row.update({f'boundary_inv_h{window}': None, f'margin_mse_h{window}': None,
                            f'error_margin_median_h{window}': None})
                continue
            perturb = margin[valid]-approx[valid]
            row[f'boundary_inv_h{window}'] = float(np.mean((approx[valid] < 0)+.5*(approx[valid] == 0)))
            row[f'margin_mse_h{window}'] = float(np.mean(perturb**2))
            row[f'error_margin_median_h{window}'] = float(np.median(np.abs(perturb)/margin[valid]))
        rows.append(row)
    keys = [k for k in rows[0] if k not in ('query', 'beam')]
    result = {k: float(np.mean([r[k] for r in rows if r[k] is not None]))
              if any(r[k] is not None for r in rows) else None for k in keys}
    return result, rows


def pair_refine(pools, codes, features, books, nodes, targets, penalty,
                selected_nodes, shortlist=8):
    """Sequential simultaneous two-group moves with EXACT coupled-loss deltas.

    Shortlists include the old code. Tests cover jointly beneficial moves that
    have no beneficial single-coordinate move. This is NOT a global optimum.
    Unseen nodes are never changed. Pool means are updated after every move.
    """
    if shortlist < 2:
        raise ValueError('Pair shortlist must include at least two choices')
    pred = predict(pools, codes, features, books, 'product')
    error = pools.targets.astype(np.float64)-pred
    lengths = np.diff(pools.offsets)
    owner = np.repeat(np.arange(pools.pool_count), lengths)
    qids = np.repeat(pools.query_ids, lengths)
    weights = 1./lengths[owner]
    means = np.add.reduceat(error, pools.offsets[:-1])/lengths
    order = np.argsort(pools.node_ids, kind='stable')
    unique, starts, counts = np.unique(pools.node_ids[order], return_index=True, return_counts=True)
    if not np.array_equal(unique, nodes):
        raise ValueError('Training target ordering mismatch')
    groups, count, width = books.shape
    changed = 0; simultaneous = 0; total_delta = 0.
    for position, node in enumerate(selected_nodes):
        rec = int(np.searchsorted(nodes, node))
        if rec == len(nodes) or nodes[rec] != node:
            raise ValueError('Selected an unseen training node')
        ev = order[starts[rec]:starts[rec]+counts[rec]]; owners = owner[ev]
        if len(np.unique(owners)) != len(owners):
            raise ValueError('Duplicate node in a pool')
        w = weights[ev]; ce = error[ev]-means[owners]
        alternatives = []
        for g in range(groups):
            old = int(codes[node, g])
            bk = books[g].astype(np.float64)
            f = features[qids[ev], g*width:(g+1)*width].astype(np.float64)
            d = f @ (bk-bk[old]).T
            target = targets[rec, g*width:(g+1)*width]
            geom = np.sum((bk-target)**2, axis=1)
            delta = np.sum(w[:, None]*(-2*ce[:, None]*d+(1-w[:, None])*d*d), axis=0)
            delta += penalty[rec]*(geom-geom[old]); delta[old] = 0
            ranked = np.argsort(delta, kind='stable')
            chosen = np.r_[old, ranked[ranked != old][:min(shortlist, count)-1]]
            alternatives.append((chosen, d[:, chosen], delta[chosen]))
        best = (0., None)
        for g in range(groups):
            kg, dg, sg = alternatives[g]
            for h in range(g+1, groups):
                kh, dh, sh = alternatives[h]
                delta = sg[:, None]+sh[None, :]+2*(dg.T @ ((w*(1-w))[:, None]*dh))
                a, b = np.unravel_index(np.argmin(delta), delta.shape)
                if delta[a, b] < best[0]-1e-12:
                    best = (float(delta[a, b]), (g, h, a, b))
        if best[1] is not None:
            g, h, a, b = best[1]
            kg, dg, _ = alternatives[g]; kh, dh, _ = alternatives[h]
            simultaneous += int(kg[a] != codes[node, g] and kh[b] != codes[node, h])
            codes[node, g] = kg[a]; codes[node, h] = kh[b]
            d = dg[:, a]+dh[:, b]
            error[ev] -= d; means[owners] -= w*d
            changed += 1; total_delta += best[0]
        if (position+1) % 2000 == 0:
            print(f'[joint-pairs] {position+1}/{len(selected_nodes)} changed={changed}', flush=True)
    return dict(changed_nodes=changed, two_group_moves=simultaneous, exact_delta=total_delta,
                shortlist=shortlist, selected_nodes=len(selected_nodes))


def train_books_epoch(pools, codes, features, books, kind, targets_all, strength,
                      learning_rate, device, max_events, seed):
    """Same centered+shrinkage objective and optimizer for both representations.

    Parameter masking is implicit in the compact product shape. Targets use
    train-only nodes. Unseen node indices are unchanged, but book updates can
    change their decoded values; the report states this explicitly.
    """
    from train_terminal_ranking_ji import pool_batches, materialize_batch
    bt = torch.tensor(books, device=device, requires_grad=True)
    ft = torch.as_tensor(features, device=device)
    optimizer = torch.optim.Adam([bt], lr=learning_rate)
    order = np.random.default_rng(seed).permutation(pools.pool_count)
    for number, selected in enumerate(pool_batches(pools, order, max_events)):
        batch = materialize_batch(pools, selected)
        ev = batch['events']
        lengths = np.diff(pools.offsets)[selected]
        owners = torch.as_tensor(np.repeat(np.arange(len(selected)), lengths), device=device)
        ids = pools.node_ids[ev]
        f = ft[torch.as_tensor(np.repeat(pools.query_ids[selected], lengths), device=device)]
        rec = reconstruct(bt, torch.as_tensor(codes[ids], device=device), kind)
        target = torch.as_tensor(pools.targets[ev], device=device)
        error = target-(f*rec).sum(1)
        counts = torch.as_tensor(lengths, dtype=torch.float32, device=device)
        means = torch.zeros(len(selected), device=device).index_add_(0, owners, error)/counts
        y = torch.as_tensor(targets_all[ids], device=device)
        prior = torch.as_tensor(strength[ids], device=device)*((rec-y)**2).sum(1)
        loss = (((error-means[owners])**2+prior)/counts[owners]).sum()/len(selected)
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        if not torch.isfinite(bt).all():
            raise FloatingPointError('Nonfinite codebooks')
        if (number+1) % 200 == 0:
            print(f'[books-{kind}] batches={number+1} loss={float(loss):.6g}', flush=True)
    return bt.detach().cpu().numpy()


def mm_reencode(pools, codes, features, books, kind, nodes, targets, penalty,
                selected_nodes, device, max_elements=2_000_000, centered_loss=True):
    """Full-pool centered majorizer; simultaneous node updates, group acceptance.

    Works for overlapping/full-dimensional books: the geometry target subtracts
    the other seven books. Neither arbitrary query-specific codes nor Euclidean
    greedy RVQ is substituted for the shared centered objective.
    """
    from train_terminal_centered_ji import centered
    pred = predict(pools, codes, features, books, kind, device)
    lengths = np.diff(pools.offsets)
    eq = np.repeat(pools.query_ids, lengths)
    w = np.repeat(1./lengths, lengths)
    order = np.argsort(pools.node_ids, kind='stable')
    unique, starts, counts = np.unique(pools.node_ids[order], return_index=True, return_counts=True)
    if not np.array_equal(unique, nodes):
        raise ValueError('Training target ordering mismatch')
    chosen = np.searchsorted(nodes, selected_nodes)
    if np.any(chosen >= len(nodes)) or not np.array_equal(nodes[chosen], selected_nodes):
        raise ValueError('Selected unseen nodes')
    bt = torch.as_tensor(books, device=device); groups, count, width = books.shape
    ft = torch.as_tensor(features, device=device)
    history = []
    with torch.no_grad():
        for g in range(groups):
            before = objective(pools, pred, codes, books, kind, nodes, targets, penalty, centered_loss)
            desired = (pred+centered(pools.targets.astype(np.float64)-pred, pools.offsets)
                       if centered_loss else pools.targets.astype(np.float64))
            original = codes[:, g].copy()
            for bucket in np.unique(np.ceil(np.log2(counts[chosen])).astype(int)):
                records = chosen[np.ceil(np.log2(counts[chosen])).astype(int) == bucket]
                padded = 1 << int(bucket); step = max(1, max_elements//(padded*count))
                for start in range(0, len(records), step):
                    rr = records[start:start+step]; ids = nodes[rr]
                    em = np.zeros((len(rr), padded), dtype=np.int64)
                    mask = np.zeros_like(em, dtype=np.float32)
                    for j, r in enumerate(rr):
                        n = counts[r]; em[j, :n] = order[starts[r]:starts[r]+n]; mask[j, :n] = 1
                    wt = torch.as_tensor(w[em]*mask, dtype=torch.float32, device=device)
                    f = ft[torch.as_tensor(eq[em], device=device)]
                    labels = torch.as_tensor(codes[ids], device=device)
                    old = bt[g, labels[:, g].long()]
                    if kind == 'product':
                        f = f[:, :, g*width:(g+1)*width]
                        y = torch.as_tensor(targets[rr, g*width:(g+1)*width], device=device)
                    else:
                        rec = reconstruct(bt, labels, kind)
                        y = torch.as_tensor(targets[rr], device=device)-rec+old
                    wanted = torch.as_tensor(desired[em]-pred[em], dtype=torch.float32, device=device)
                    wanted += (f*old[:, None, :]).sum(2)
                    candidate = torch.einsum('bnw,kw->bnk', f, bt[g])
                    loss = (wt[:, :, None]*(candidate-wanted[:, :, None])**2).sum(1)
                    loss += torch.as_tensor(penalty[rr], dtype=torch.float32, device=device)[:, None]*(y[:, None, :]-bt[g][None, :, :]).square().sum(2)
                    best = loss.argmin(1)
                    # Do not change labels on exact/effectively equal minima.
                    old_loss = loss[torch.arange(len(rr), device=device), labels[:, g].long()]
                    best_loss = loss[torch.arange(len(rr), device=device), best]
                    best = torch.where(best_loss < old_loss, best, labels[:, g].long())
                    codes[ids, g] = best.cpu().numpy().astype(np.uint8)
            proposed = predict(pools, codes, features, books, kind, device)
            after = objective(pools, proposed, codes, books, kind, nodes, targets, penalty, centered_loss)
            accepted = after['total'] <= before['total']
            changed = int(np.count_nonzero(codes[:, g] != original))
            if accepted: pred = proposed
            else: codes[:, g] = original
            history.append(dict(group=g, accepted=accepted, changed=changed, before=before, after=after))
            print(f'[mm-{kind}] group={g} changed={changed} accepted={accepted}', flush=True)
    return history
