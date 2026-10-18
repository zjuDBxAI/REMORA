"""Numerical helpers for group residual vector quantization."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but torch.cuda.is_available() is false; "
            "use --solver-device cpu or run in a CUDA-enabled environment"
        )
    return device


def build_groups(
    selected_subspaces: np.ndarray, group_count: int, policy: str
) -> np.ndarray:
    selected = np.asarray(selected_subspaces, dtype=np.uint32)
    groups = int(group_count)
    if selected.ndim != 1 or groups <= 0 or selected.size % groups != 0:
        raise ValueError("selected subspaces must divide evenly into groups")
    per_group = selected.size // groups
    if policy == "contiguous":
        result = selected.reshape(groups, per_group)
    elif policy == "risk-round-robin":
        result = selected.reshape(per_group, groups).T
    else:
        raise ValueError(f"unknown group policy: {policy}")
    return np.ascontiguousarray(result, dtype=np.uint32)


def extract_group_residual(
    vectors_t: np.ndarray,
    pq_codes: np.ndarray,
    pq_model,
    groups: np.ndarray,
) -> np.ndarray:
    """Return residual blocks with shape [rows, groups, group_dim]."""
    vectors = np.asarray(vectors_t, dtype=np.float32)
    codes = np.asarray(pq_codes, dtype=np.uint8)
    grouping = np.asarray(groups, dtype=np.int64)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != pq_model.dim
        or codes.shape != (vectors.shape[0], pq_model.subspaces)
        or grouping.ndim != 2
    ):
        raise ValueError("group residual input shape mismatch")
    blocks = vectors.reshape(-1, pq_model.subspaces, pq_model.dsub)
    result = np.empty(
        (
            vectors.shape[0],
            grouping.shape[0],
            grouping.shape[1] * pq_model.dsub,
        ),
        dtype=np.float32,
    )
    for group_index, group in enumerate(grouping):
        for local, raw_subspace in enumerate(group):
            subspace = int(raw_subspace)
            begin = local * pq_model.dsub
            end = begin + pq_model.dsub
            result[:, group_index, begin:end] = (
                blocks[:, subspace]
                - pq_model.centroids[subspace, codes[:, subspace]]
            )
    return result


def build_selected_query_metric(
    transformed_queries: np.ndarray,
    groups: np.ndarray,
    dsub: int,
    metric_lambda: float,
    trace_epsilon: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, float | int]]]:
    """Build A_g=I+lambda*Wbar_g in the exact selected-group layout."""
    queries = np.asarray(transformed_queries, dtype=np.float32)
    grouping = np.asarray(groups, dtype=np.int64)
    block_dim = int(dsub)
    if (
        queries.ndim != 2
        or grouping.ndim != 2
        or block_dim <= 0
        or queries.shape[1] % block_dim != 0
        or np.any(grouping < 0)
        or np.any(grouping >= queries.shape[1] // block_dim)
        or not np.isfinite(metric_lambda)
        or metric_lambda < 0.0
        or trace_epsilon <= 0.0
    ):
        raise ValueError("invalid selected query metric configuration")
    blocks = queries.reshape(queries.shape[0], -1, block_dim)
    grouped = np.ascontiguousarray(
        blocks[:, grouping, :].reshape(
            queries.shape[0], grouping.shape[0], grouping.shape[1] * block_dim
        ),
        dtype=np.float64,
    )
    second_moment = np.einsum(
        "bgd,bge->gde", grouped, grouped, optimize=True
    ) / max(queries.shape[0], 1)
    group_dim = grouped.shape[2]
    identity = np.eye(group_dim, dtype=np.float64)
    normalized = np.zeros_like(second_moment)
    matrices = np.empty_like(second_moment)
    factors = np.empty_like(second_moment)
    diagnostics: list[dict[str, float | int]] = []
    for group in range(grouping.shape[0]):
        trace = float(np.trace(second_moment[group]))
        if trace > trace_epsilon:
            normalized[group] = (group_dim / trace) * second_moment[group]
        matrix = identity + metric_lambda * normalized[group]
        matrix = 0.5 * (matrix + matrix.T)
        matrices[group] = matrix
        factors[group] = np.linalg.cholesky(matrix)
        eigenvalues = np.linalg.eigvalsh(matrix)
        diagnostics.append(
            {
                "group": group,
                "raw_trace": trace,
                "normalized_trace": float(np.trace(normalized[group])),
                "min_eigenvalue": float(eigenvalues[0]),
                "max_eigenvalue": float(eigenvalues[-1]),
                "condition_number": float(eigenvalues[-1] / eigenvalues[0]),
            }
        )
    if metric_lambda == 0.0:
        matrices[:] = identity
        factors[:] = identity
    return (
        normalized.astype(np.float32),
        matrices.astype(np.float32),
        factors.astype(np.float32),
        diagnostics,
    )


def nearest_group_codes(
    residual: np.ndarray,
    prototypes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    values = np.asarray(residual, dtype=np.float32)
    centers_np = np.asarray(prototypes, dtype=np.float32)
    if (
        values.ndim != 3
        or centers_np.ndim != 3
        or values.shape[1] != centers_np.shape[0]
        or values.shape[2] != centers_np.shape[2]
        or centers_np.shape[1] > 256
        or batch_size <= 0
    ):
        raise ValueError("nearest group-code shape mismatch")
    centers = torch.as_tensor(centers_np, device=device)
    center_norm = torch.sum(centers * centers, dim=2)
    result = np.empty((values.shape[0], values.shape[1]), dtype=np.uint8)
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            end = min(values.shape[0], start + batch_size)
            batch = torch.as_tensor(
                np.ascontiguousarray(values[start:end]), device=device
            )
            distance = (
                torch.sum(batch * batch, dim=2)[:, :, None]
                + center_norm[None, :, :]
                - 2.0 * torch.einsum("bgd,gkd->bgk", batch, centers)
            )
            result[start:end] = (
                torch.argmin(distance, dim=2).to("cpu").numpy().astype(np.uint8)
            )
    return result


def lloyd_group_kmeans(
    residual: np.ndarray,
    codewords: int,
    iterations: int,
    convergence_tolerance: float,
    device: torch.device,
    batch_size: int,
    seed: int,
    log_interval: int = 1,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Train all groups jointly while streaming batches from a residual cache."""
    values = np.asarray(residual)
    if (
        values.ndim != 3
        or values.shape[0] < codewords
        or codewords <= 1
        or codewords > 256
        or iterations <= 0
        or batch_size <= 0
        or convergence_tolerance < 0.0
    ):
        raise ValueError("invalid group k-means configuration")
    rng = np.random.default_rng(seed)
    groups, dimensions = values.shape[1], values.shape[2]
    initial = np.empty((groups, codewords, dimensions), dtype=np.float32)
    for group in range(groups):
        ids = rng.choice(values.shape[0], codewords, replace=False)
        initial[group] = np.asarray(values[ids, group], dtype=np.float32)
    centers = torch.as_tensor(initial, device=device).clone()
    final_counts = torch.zeros((groups, codewords), dtype=torch.int64, device=device)
    completed = 0
    relative_shift = float("inf")
    empty_clusters = 0
    metric_mse = float("inf")
    with torch.no_grad():
        for iteration in range(iterations):
            sums = torch.zeros(
                (groups, codewords, dimensions), dtype=torch.float32, device=device
            )
            counts = torch.zeros(
                (groups, codewords), dtype=torch.int64, device=device
            )
            center_norm = torch.sum(centers * centers, dim=2)
            distortion = 0.0
            observations = 0
            for start in range(0, values.shape[0], batch_size):
                end = min(values.shape[0], start + batch_size)
                batch = torch.as_tensor(
                    np.ascontiguousarray(values[start:end], dtype=np.float32),
                    device=device,
                )
                distance = (
                    torch.sum(batch * batch, dim=2)[:, :, None]
                    + center_norm[None, :, :]
                    - 2.0 * torch.einsum("bgd,gkd->bgk", batch, centers)
                )
                labels = torch.argmin(distance, dim=2)
                distortion += float(
                    torch.gather(distance, 2, labels[:, :, None]).sum().item()
                )
                observations += int(batch.shape[0] * groups)
                for group in range(groups):
                    counts[group] += torch.bincount(
                        labels[:, group], minlength=codewords
                    )
                    sums[group].index_add_(0, labels[:, group], batch[:, group])
            occupied = counts > 0
            updated = centers.clone()
            updated[occupied] = (
                sums[occupied] / counts[occupied].to(torch.float32)[:, None]
            )
            shift = torch.sum((updated - centers) ** 2)
            scale = torch.sum(centers**2).clamp_min(1e-30)
            relative_shift = float((shift / scale).item())
            centers = updated
            final_counts = counts
            completed = iteration + 1
            empty_clusters = int(torch.sum(~occupied).item())
            metric_mse = distortion / max(observations, 1)
            if log_interval > 0 and (
                completed % log_interval == 0
                or completed == iterations
                or relative_shift <= convergence_tolerance
            ):
                print(
                    f"[group-kmeans] iter={completed} "
                    f"mse={metric_mse:.8g} "
                    f"relative_shift={relative_shift:.8g} "
                    f"empty={empty_clusters}",
                    flush=True,
                )
            if relative_shift <= convergence_tolerance:
                break
    diagnostics = {
        "iterations": completed,
        "relative_centroid_shift": relative_shift,
        "empty_sample_clusters": empty_clusters,
        "final_metric_mse": metric_mse,
    }
    return (
        centers.to("cpu").numpy().astype(np.float32),
        final_counts.to("cpu").numpy().astype(np.int64),
        diagnostics,
    )
