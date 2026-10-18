"""Exact FlatL2 geometry only; never substitute it for centered code selection.

GPU IDs are explicit process-visible ordinals. Kmeans(gpu=N) is intentionally
not used: that API means 'N GPUs', not 'GPU number N'. Near-tied assignments
are checked using direct FP64 distances and lowest-index tie breaking.
"""
import numpy as np
# Load PyTorch's CUDA runtime before Faiss. The isolated Faiss build uses
# CUDA 12.1 headers while PyTorch supplies a newer compatible CUDA 12 runtime;
# importing Faiss first can pin libcudart to 12.1 and break PyTorch imports.
import torch


def resolved_device(requested, training_device):
    device = training_device if requested == 'auto' else requested
    if device != 'cpu':
        if not device.startswith('cuda:') or not device[5:].isdigit():
            raise ValueError('Faiss device must be cpu, auto, or cuda:N')
        import faiss
        version = tuple(int(v) for v in faiss.__version__.split('.')[:2])
        if version < (1, 9):
            raise RuntimeError('GPU backend requires the tested Faiss >=1.9 runtime; '
                               'run ours/test/t2i-10M/build_faiss_runtime.sh and use run_all.sh')
        gpu = int(device[5:])
        if not hasattr(faiss, 'StandardGpuResources') or gpu >= faiss.get_num_gpus():
            raise RuntimeError(f'Faiss GPU requested but unavailable: {device}')
    return device


def flat_index(width, device, resources=None):
    import faiss
    cpu = faiss.IndexFlatL2(width)
    if device == 'cpu':
        return cpu, None
    # Resource construction must also occur on the requested device, not the
    # process default GPU 0. No multi-GPU Faiss convenience APIs are used.
    import torch
    torch.cuda.set_device(int(device[5:]))
    if resources is None:
        resources = faiss.StandardGpuResources()
        resources.setTempMemory(128 << 20)
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    return faiss.index_cpu_to_gpu(resources, int(device[5:]), cpu, options), resources


def train_kmeans(values, count, iterations, seed, device='cpu'):
    """Same Faiss Clustering parameters, with exactly one selected GPU index."""
    import faiss
    x = np.ascontiguousarray(values, np.float32)
    if x.ndim != 2 or not np.isfinite(x).all() or len(x) < count:
        raise ValueError('K-means requires finite data and at least K samples')
    clustering = faiss.Clustering(x.shape[1], count)
    clustering.niter = iterations
    clustering.nredo = 1
    clustering.seed = seed
    clustering.min_points_per_centroid = 1
    clustering.verbose = False
    index, resources = flat_index(x.shape[1], device)
    clustering.train(x, index)
    books = faiss.vector_to_array(clustering.centroids).reshape(count, x.shape[1]).copy()
    if not np.isfinite(books).all():
        raise FloatingPointError('Nonfinite Faiss centroids')
    return books


def canonical_labels(values, books, batch=128):
    """Stable direct FP64 reference; tiny audit/ambiguous subsets only."""
    out = np.empty(len(values), np.uint8)
    b = np.asarray(books, np.float64)
    for lo in range(0, len(values), batch):
        v = np.asarray(values[lo:lo + batch], np.float64)
        distances = np.square(v[:, None, :] - b[None, :, :]).sum(2)
        out[lo:lo + len(v)] = distances.argmin(1).astype(np.uint8)
    return out


class FaissProductEncoder:
    """Frozen codebooks cached once, batch streams; no N*d GPU allocation."""
    def __init__(self, rotation, books, device='cpu', projection_device='cpu'):
        import torch
        self.books = np.ascontiguousarray(books, np.float32)
        self.groups, count, self.width = self.books.shape
        if count != 256:
            raise ValueError('Expected 256 codewords/group')
        self.rotation = np.ascontiguousarray(rotation, np.float32)
        if self.rotation.shape != (self.groups*self.width,)*2:
            raise ValueError('Rotation/codebook shape mismatch')
        self.device, self.projection_device = device, projection_device
        self.rt = torch.as_tensor(self.rotation, device=projection_device)
        self.resources = None
        self.indices = []
        self.ambiguous_assignments = 0
        self.assignments = 0
        for b in self.books:
            index, self.resources = flat_index(self.width, device, self.resources)
            index.add(b)
            self.indices.append(index)

    def rotate(self, values):
        import torch
        with torch.no_grad():
            x = torch.as_tensor(np.ascontiguousarray(values, np.float32), device=self.projection_device)
            return (x @ self.rt).cpu().numpy()

    def encode(self, values, batch=8192):
        out = np.empty((len(values), self.groups), np.uint8)
        for lo in range(0, len(values), batch):
            rotated = self.rotate(values[lo:lo+batch]).reshape(-1, self.groups, self.width)
            for g, index in enumerate(self.indices):
                x = np.ascontiguousarray(rotated[:, g, :])
                if not np.isfinite(x).all():
                    raise FloatingPointError('Nonfinite geometric encoding input')
                distance, labels = index.search(x, 2)
                # Conservative roundoff window for squared-distance arithmetic.
                scale = np.square(x).sum(1) + np.square(self.books[g][labels[:, 0]]).sum(1)
                scale = np.maximum(scale, np.square(self.books[g]).sum(1).max())
                tolerance = 64*np.finfo(np.float32).eps*self.width*np.maximum(scale, 1e-20)
                ambiguous = distance[:, 1]-distance[:, 0] <= tolerance
                chosen = labels[:, 0].astype(np.uint8)
                if ambiguous.any():
                    chosen[ambiguous] = canonical_labels(x[ambiguous], self.books[g])
                out[lo:lo+len(x), g] = chosen
                self.ambiguous_assignments += int(ambiguous.sum())
                self.assignments += len(x)
        return out

    def audit(self, values, batch=8192):
        """Compare frozen-model assignments with torch and FP64, not kmeans outputs."""
        from rotation_product import encode
        actual = self.encode(values, batch)
        expected = encode(values, self.rotation, self.books, self.projection_device, batch)
        canonical = np.empty_like(actual)
        rotated = self.rotate(values).reshape(-1, self.groups, self.width)
        for g in range(self.groups):
            canonical[:, g] = canonical_labels(rotated[:, g, :], self.books[g])
        faiss_mismatch = int(np.count_nonzero(actual != canonical))
        torch_mismatch = int(np.count_nonzero(expected != canonical))
        if faiss_mismatch:
            raise RuntimeError(f'Faiss geometric assignment audit failed: {faiss_mismatch}')
        # Torch may choose differently on FP32 ties; report, do not hide it.
        examples = []
        for row,group in np.argwhere(expected != canonical)[:10]:
            distances = np.square(np.asarray(rotated[row,group],np.float64)
                                  - np.asarray(self.books[group],np.float64)).sum(1)
            old,new = int(expected[row,group]),int(canonical[row,group])
            examples.append(dict(sample=int(row),group=int(group),torch_code=old,
                                 canonical_code=new,canonical_squared_l2=float(distances[new]),
                                 torch_squared_l2=float(distances[old]),
                                 excess_squared_l2=float(distances[old]-distances[new])))
        return dict(samples=len(values), groups=self.groups,
                    faiss_canonical_mismatches=faiss_mismatch,
                    torch_canonical_mismatches=torch_mismatch,
                    faiss_torch_agreement=float(np.mean(actual == expected)),
                    faiss_device=self.device, projection_device=self.projection_device,
                    torch_near_tie_examples=examples,
                    canonical='direct squared L2 in FP64; lowest code ID on ties')
