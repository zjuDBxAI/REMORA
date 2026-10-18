"""Orthogonal, fixed-rate Product refinement in existing JI coordinates.

Row convention: v=y@rotation, phi_v=phi@rotation. A decoder stores the
FUSED query projection, never an extra online rotation. No raw vector access
is introduced online. The training objective includes actual pool centering.
"""
import numpy as np
import torch


def balanced_rotation(spectrum, groups=8):
    """Capacity-constrained greedy spectral permutation, not an optimality claim."""
    spectrum = np.asarray(spectrum, np.float64)
    if spectrum.ndim != 1 or len(spectrum) % groups or not np.isfinite(spectrum).all():
        raise ValueError('Invalid spectrum/group layout')
    width = len(spectrum)//groups
    bins = [[] for _ in range(groups)]
    energy = np.zeros(groups)
    for j in np.argsort(-spectrum, kind='stable'):
        g = min((g for g in range(groups) if len(bins[g]) < width),
                key=lambda g: (energy[g], len(bins[g]), g))
        bins[g].append(int(j)); energy[g] += max(0., spectrum[j])
    permutation = np.asarray([j for b in bins for j in b])
    return np.eye(len(spectrum), dtype=np.float32)[:, permutation]


def orthogonal(matrix):
    u, _, vt = np.linalg.svd(np.asarray(matrix, np.float64), full_matrices=False)
    return np.asarray(u@vt, np.float32)


def rotate_rows(values, rotation, device='cpu', batch=16384):
    out = np.empty(values.shape, np.float32)
    rt = torch.as_tensor(rotation, dtype=torch.float32, device=device)
    with torch.no_grad():
        for lo in range(0, len(values), batch):
            chunk = np.asarray(values[lo:lo+batch])
            if not chunk.flags.writeable: chunk = chunk.copy()
            out[lo:lo+batch] = (torch.as_tensor(chunk, device=device)@rt).cpu().numpy()
    return out


def encode(values, rotation, books, device='cpu', batch=4096):
    """Geometric fallback for ALL nodes, including unseen terminal nodes."""
    groups, count, width = books.shape
    out = np.empty((len(values), groups), np.uint8)
    bt = torch.as_tensor(books, device=device)
    rt = torch.as_tensor(rotation, device=device)
    with torch.no_grad():
        for lo in range(0, len(values), batch):
            x = (torch.as_tensor(np.asarray(values[lo:lo+batch]), device=device)@rt).reshape(-1, groups, width)
            for g in range(groups):
                dist = x[:, g].square().sum(1, keepdim=True)+bt[g].square().sum(1)[None, :]-2*x[:, g]@bt[g].T
                out[lo:lo+batch, g] = dist.argmin(1).cpu().numpy().astype(np.uint8)
    return out


def decode(codes, books):
    return np.concatenate([books[g][codes[:, g]] for g in range(len(books))], axis=1)


def lloyd_step(sample, rotation, books, device='cpu'):
    labels = encode(sample, rotation, books, device)
    x = torch.as_tensor(rotate_rows(sample, rotation, device), device=device)
    result = books.copy()
    with torch.no_grad():
        for g in range(len(books)):
            ids = torch.as_tensor(labels[:, g].astype(np.int64), device=device)
            counts = torch.bincount(ids, minlength=len(books[g]))
            sums = torch.zeros_like(torch.as_tensor(books[g], device=device))
            width = books.shape[2]
            sums.index_add_(0, ids, x[:, g*width:(g+1)*width])
            valid = counts > 0
            updated = torch.as_tensor(books[g].copy(), device=device)
            updated[valid] = sums[valid]/counts[valid, None]
            result[g] = updated.cpu().numpy()
    return result


def procrustes(sample, labels, books):
    """Exact fixed-code orthogonal update for GLOBAL geometric MSE only."""
    target = decode(labels, books)
    return orthogonal(sample.astype(np.float64).T@target.astype(np.float64))


def rotation_diagnostics(rotation, original_width):
    d = len(rotation)
    # Column blocks are current groups; row blocks are original groups.
    block = np.arange(d)//original_width
    cross = block[:, None] != block[None, :]
    return dict(orthogonality_error=float(np.max(np.abs(rotation.T@rotation-np.eye(d)))),
                distance_from_identity=float(np.linalg.norm(rotation-np.eye(d))),
                mean_entries_above_1e_4_per_column=float(np.mean(np.sum(abs(rotation)>1e-4, axis=0))),
                off_original_block_energy_fraction=float(np.sum(rotation[cross]**2)/d))


class CenteredObjective:
    """Full-data value/gradient; no stochastic epoch is silently called descent.

    Full LUT prediction avoids per-event dense projection. The geometry term is
    batched over seen nodes and has mass*D_eff/n_eff weights, exactly as the old
    objective. Objective is divided by pool count for numerical convenience.
    """
    def __init__(self, pools, features, nodes, targets, penalty, device='cpu',
                 batch=16384, dtype=torch.float32, centered_loss=True):
        self.pools = pools; self.nodes = np.asarray(nodes)
        self.centered_loss = bool(centered_loss)
        self.targets = targets; self.penalty = penalty
        self.device = device; self.batch = batch; self.dtype = dtype
        unique, inverse = np.unique(pools.query_ids, return_inverse=True)
        lengths = np.diff(pools.offsets)
        self.f = torch.as_tensor(features[unique], dtype=dtype, device=device)
        self.qowner = torch.as_tensor(np.repeat(inverse, lengths), device=device)
        self.owner = torch.as_tensor(np.repeat(np.arange(len(lengths)), lengths), device=device)
        self.lengths = torch.as_tensor(lengths, dtype=dtype, device=device)
        self.target = torch.as_tensor(pools.targets, dtype=dtype, device=device)

    def evaluate(self, books, rotation, codes, gradient=False):
        with torch.set_grad_enabled(gradient):
            bt = torch.tensor(books, dtype=self.dtype, device=self.device, requires_grad=gradient)
            rt = torch.tensor(rotation, dtype=self.dtype, device=self.device, requires_grad=gradient)
            labels = torch.as_tensor(codes[self.pools.node_ids].astype(np.int64), device=self.device)
            f = (self.f@rt).reshape(len(self.f), len(books), -1)
            lut = torch.einsum('qgw,gkw->qgk', f, bt)
            pred = sum(lut[self.qowner, g, labels[:, g]] for g in range(len(books)))
            error = self.target-pred
            means = (torch.zeros(len(self.lengths), dtype=self.dtype, device=self.device).index_add(0, self.owner, error)/self.lengths
                     if self.centered_loss else torch.zeros_like(self.lengths))
            terminal = (((error-means[self.owner])**2)/self.lengths[self.owner]).sum()/self.pools.pool_count
            terminal_value = float(terminal.detach())
            if gradient: terminal.backward()
            del pred, error, means, terminal, lut, f, labels
            geometry_value = 0.
            for lo in range(0, len(self.nodes), self.batch):
                sl = slice(lo, lo+self.batch)
                lab = torch.as_tensor(codes[self.nodes[sl]].astype(np.int64), device=self.device)
                rec = torch.cat([bt[g][lab[:, g]] for g in range(len(books))], 1)
                y = torch.as_tensor(np.asarray(self.targets[sl]), dtype=self.dtype, device=self.device)
                weight = torch.as_tensor(self.penalty[sl], dtype=self.dtype, device=self.device)
                loss = (weight*(y@rt-rec).square().sum(1)).sum()/self.pools.pool_count
                geometry_value += float(loss.detach())
                if gradient: loss.backward()
            result = dict(terminal=terminal_value, geometry=geometry_value,
                          total=terminal_value+geometry_value)
            if not np.isfinite(result['total']): raise FloatingPointError('Nonfinite objective')
            if gradient:
                return result, bt.grad.detach().cpu().numpy(), rt.grad.detach().cpu().numpy()
            return result


def descent_step(engine, books, rotation, codes, parameter, step=.05, backtracks=10):
    """Full-gradient descent with orthogonal polar retraction and true-loss gate."""
    before, gb, gr = engine.evaluate(books, rotation, codes, gradient=True)
    if parameter == 'books':
        grad = gb.astype(np.float64)
        scale = max(float(np.sqrt(np.mean(books.astype(np.float64)**2))), 1e-6)
        direction = grad/max(float(np.sqrt(np.mean(grad**2))), 1e-30)*scale
    elif parameter == 'rotation':
        q = rotation.astype(np.float64); g = gr.astype(np.float64)
        qt = q.T@g
        grad = g-q@((qt+qt.T)/2)
        direction = grad/max(float(np.linalg.norm(grad))/np.sqrt(len(q)), 1e-30)
    else:
        raise ValueError(parameter)
    history = []
    for trial in range(backtracks):
        rate = step*(.5**trial)
        nb, nr = books, rotation
        if parameter == 'books': nb = np.asarray(books-rate*direction, np.float32)
        else: nr = orthogonal(rotation-rate*direction)
        after = engine.evaluate(nb, nr, codes)
        accepted = after['total'] < before['total']-1e-10*max(abs(before['total']), 1e-12)
        history.append(dict(step=rate, after=after, accepted=accepted))
        if accepted:
            return nb, nr, dict(parameter=parameter, accepted=True, before=before, after=after,
                               gradient_norm=float(np.linalg.norm(grad)), trials=history)
    return books, rotation, dict(parameter=parameter, accepted=False, before=before, after=before,
                                gradient_norm=float(np.linalg.norm(grad)), trials=history)
