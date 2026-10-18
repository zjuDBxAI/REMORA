"""Sequential exact discrete updates for a training-selected subset of nodes."""
import numpy as np


def exact_refine(events, offsets, codes, phi, books, nodes, targets, penalty,
                 predictions, max_nodes=20000, weights=None, terminal_guard=False):
    lengths = np.diff(offsets)
    pool_ids = np.repeat(np.arange(len(lengths)), lengths)
    error = events.targets.astype(np.float64) - predictions
    weights = (np.repeat(1.0/lengths, lengths) if weights is None
               else np.asarray(weights, dtype=np.float64))
    if (weights.shape != error.shape or not np.all(np.isfinite(weights))
            or np.any(weights <= 0)
            or not np.allclose(np.add.reduceat(weights, offsets[:-1]), 1, atol=1e-7)):
        raise ValueError('Positive, pool-normalized event weights required')
    weights = weights / np.repeat(np.add.reduceat(weights, offsets[:-1]), lengths)
    means = np.add.reduceat(weights * error, offsets[:-1])
    centered_error = error - means[pool_ids]
    # Selection only uses current training error, never validation/test queries.
    energy = np.bincount(events.node_ids,
        weights=weights*centered_error**2, minlength=len(codes))
    priority = np.lexsort((nodes, -energy[nodes]))[:min(max_nodes, len(nodes))]
    order = np.argsort(events.node_ids, kind='stable')
    unique, starts, counts = np.unique(events.node_ids[order], return_index=True, return_counts=True)
    if not np.array_equal(unique, nodes):
        raise ValueError('Target node ordering mismatch')
    changes = np.zeros(books.shape[0], dtype=np.int64)
    total_delta = 0.0
    terminal_delta = 0.0
    guard_blocked_choices = 0
    books = np.asarray(books, dtype=np.float64)
    for position, record in enumerate(priority):
        node = nodes[record]
        ids = order[starts[record]:starts[record]+counts[record]]
        pools = pool_ids[ids]
        if np.unique(pools).size != len(pools):
            raise ValueError('A node occurs more than once in a candidate pool')
        w = weights[ids]
        features = np.asarray(phi[events.query_ids[ids]], dtype=np.float64)
        for g in range(books.shape[0]):
            old = int(codes[node, g])
            d = features[:, g] @ (books[g]-books[g, old]).T
            ce = error[ids] - means[pools]
            terminal = np.sum(w[:, None]*(-2*ce[:, None]*d+(1-w[:, None])*d*d), axis=0)
            geometry = np.sum((books[g]-targets[record, g])**2, axis=1)
            delta = terminal + penalty[record]*(geometry-geometry[old])
            delta[old] = 0.0
            if terminal_guard:
                # Choose the best feasible code, not merely reject the best
                # unconstrained code. Existing code always remains feasible.
                blocked = (terminal > 0) & (delta < -1e-12)
                guard_blocked_choices += int(np.count_nonzero(blocked))
                delta[terminal > 0] = np.inf
                delta[old] = 0.0
            best = int(np.argmin(delta))
            if delta[best] < -1e-12:
                codes[node, g] = best
                error[ids] -= d[:, best]
                means[pools] -= w*d[:, best]
                total_delta += float(delta[best])
                terminal_delta += float(terminal[best])
                changes[g] += 1
        if (position+1) % 2000 == 0:
            print(f'[exact-refine] {position+1}/{len(priority)} nodes', flush=True)
    return dict(selected_nodes=int(len(priority)),changed_by_group=changes.tolist(),
                summed_exact_delta=total_delta, summed_terminal_delta=terminal_delta,
                terminal_guard=terminal_guard, guard_blocked_choices=guard_blocked_choices,
                selection='largest current training centered-error energy; ties by node id')
