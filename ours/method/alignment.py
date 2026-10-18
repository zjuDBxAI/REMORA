"""Fit query-visible residual coordinates and the initial product codebooks."""
import json
from pathlib import Path
import time

import numpy as np

from joint_innovation_rpq import build_joint_transform
from rotation_product import balanced_rotation
from faiss_backend import resolved_device, train_kmeans
from protocol import file_id

VARIANTS = ("euclidean", "alignment", "balanced", "uncentered-fixed",
            "centered-fixed", "centered", "remora")


def padded(values, dim):
    result = np.zeros((len(values), dim), np.float32)
    result[:, :values.shape[1]] = values
    return result


def layout(variant):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown variant: {variant}")
    return variant if variant in ("euclidean", "alignment") else "balanced"


def setup(config):
    import torch
    import faiss
    torch.set_num_threads(config["trainer_threads"])
    faiss.omp_set_num_threads(config["trainer_threads"])
    torch.manual_seed(config["seed"])
    if config["device"].startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but unavailable")
        torch.backends.cuda.matmul.allow_tf32 = False
    return resolved_device(config.get("kmeans_device", "auto"), config["device"])


def fit_alignment(config, primary, base, queries, bits, variant, method):
    """Only training queries and sampled database residuals enter this stage."""
    if isinstance(bits, bool) or not isinstance(bits, int) or not 0 < bits <= 256 or bits % 8:
        raise ValueError("Side bits must be a multiple of 8 in [8, 256]")
    kind = layout(variant)
    if base.shape != (primary.nodes, primary.dim) or queries.shape[1] != primary.dim:
        raise ValueError("Base/query/primary dimension mismatch")
    device = setup(config)
    groups = bits // 8
    dim = ((primary.dim + groups - 1) // groups) * groups
    width = dim // groups
    started = time.monotonic()
    rng = np.random.default_rng(config["seed"])
    fit = rng.choice(primary.nodes, min(config["residual_sample_size"], primary.nodes), replace=False)
    residual = np.asarray(base[fit]) - primary.reconstruct_raw(fit)
    W = np.eye(primary.dim) if kind == "euclidean" else (
        np.asarray(queries, np.float64).T @ np.asarray(queries, np.float64) / len(queries))
    mean, forward, inverse, spectrum, diagnostics = build_joint_transform(residual, W, 1e-6)
    spectrum = np.pad(spectrum, (0, dim - primary.dim))
    rotation = (np.eye(dim, dtype=np.float32) if kind != "balanced"
                else balanced_rotation(spectrum, groups))
    values = padded((residual - mean) @ forward, dim)
    rotated = values @ rotation
    books = np.empty((groups, 256, width), np.float32)
    for group in range(groups):
        books[group] = train_kmeans(rotated[:, group*width:(group+1)*width], 256,
                                    config["ji_iterations"], config["seed"] + group, device)
        print(f"[alignment] {method}/b{bits}/{variant} group={group+1}/{groups} "
              f"elapsed={time.monotonic()-started:.1f}s", flush=True)
    return dict(mean=mean, forward=forward, inverse=inverse, spectrum=spectrum,
                rotation=rotation, books=books,
                audit_values=values[:config.get("encoding_audit_size", 1024)],
                diagnostics=diagnostics)


def signature(config, method, model, codes, queries, bits, variant):
    """Reuse requires the same inputs, layout and initialization settings.

    file_id samples file edges and includes stat information; this is not a
    full-file checksum of large database/code files.
    """
    return dict(version=1, method=method, bits=bits, layout=layout(variant),
                inputs={key: dict(path=str(Path(path).resolve()), identity=file_id(path))
                        for key, path in dict(base=config["base"], model=model,
                                              codes=codes, queries=queries).items()},
                settings={key: config[key] for key in ("seed", "residual_sample_size",
                          "ji_iterations", "encoding_audit_size", "trainer_threads")},
                kmeans_device=resolved_device(config.get("kmeans_device", "auto"), config["device"]))


def save_alignment(path, state, metadata):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Alignment checkpoint already exists; choose a new --output: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **{k: v for k, v in state.items() if k != "diagnostics"},
                            metadata=json.dumps(metadata), diagnostics=json.dumps(state["diagnostics"]))
    temporary.replace(path)


def load_alignment(path, expected):
    with np.load(path, allow_pickle=False) as saved:
        if json.loads(str(saved["metadata"])) != expected:
            raise ValueError("Alignment inputs/settings/layout changed; create a new checkpoint")
        result = {key: saved[key].copy() for key in ("mean", "forward", "inverse", "spectrum",
                                                    "rotation", "books", "audit_values")}
        result["diagnostics"] = json.loads(str(saved["diagnostics"]))
    return result
