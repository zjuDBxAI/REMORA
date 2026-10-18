"""Global metric transforms and OPQ rotation for residual product quantization."""

from __future__ import annotations

import numpy as np
import torch

from group_residual_vq import nearest_group_codes


def random_orthogonal_rotation(dim: int, seed: int) -> np.ndarray:
    """Return a deterministic Haar-like orthogonal row-coordinate rotation."""
    if dim <= 0:
        raise ValueError("rotation dimension must be positive")
    rng = np.random.default_rng(seed)
    matrix = rng.standard_normal((dim, dim))
    rotation, triangular = np.linalg.qr(matrix)
    signs = np.sign(np.diag(triangular))
    signs[signs == 0.0] = 1.0
    rotation *= signs[None, :]
    return np.asarray(rotation, dtype=np.float32)


def transformed_residual_covariance(
    values: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Estimate the centered covariance of a grouped transformed-residual cache."""
    source = np.asarray(values)
    if source.ndim != 3 or source.shape[0] <= 0 or batch_size <= 0:
        raise ValueError("invalid residual covariance configuration")
    dim = int(source.shape[1] * source.shape[2])
    total = np.zeros(dim, dtype=np.float64)
    second = np.zeros((dim, dim), dtype=np.float64)
    with torch.no_grad():
        for start in range(0, source.shape[0], batch_size):
            end = min(source.shape[0], start + batch_size)
            batch = torch.as_tensor(
                np.ascontiguousarray(source[start:end], dtype=np.float32),
                device=device,
            ).reshape(end - start, dim)
            total += (
                torch.sum(batch, dim=0).to(device="cpu", dtype=torch.float64).numpy()
            )
            second += (
                (batch.T @ batch).to(device="cpu", dtype=torch.float64).numpy()
            )
    mean = total / float(source.shape[0])
    covariance = second / float(source.shape[0]) - np.outer(mean, mean)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance


def eigenvalue_balanced_rotation_from_covariance(
    covariance: np.ndarray,
    groups: int,
    eigenvalue_floor_relative: float = 1e-8,
) -> tuple[np.ndarray, dict]:
    """Build a PCA rotation whose group log-volumes are greedily balanced.

    The eigen-directions are assigned to equal-sized PQ groups using longest-
    processing-time scheduling on shifted log eigenvalues.  Adding a constant
    to every log eigenvalue does not change final group balance because every
    group has the same number of dimensions.
    """
    matrix = np.asarray(covariance, dtype=np.float64)
    if (
        matrix.ndim != 2
        or matrix.shape[0] != matrix.shape[1]
        or groups <= 0
        or matrix.shape[0] % groups != 0
        or not 0.0 < eigenvalue_floor_relative < 1.0
    ):
        raise ValueError("invalid covariance-balanced rotation configuration")
    matrix = 0.5 * (matrix + matrix.T)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    maximum = max(float(eigenvalues[-1]), np.finfo(np.float64).tiny)
    floor = maximum * eigenvalue_floor_relative
    clipped = np.maximum(eigenvalues, floor)
    log_values = np.log(clipped)
    weights = log_values - float(np.min(log_values))
    order = np.argsort(-clipped, kind="stable")
    group_dim = matrix.shape[0] // groups
    members: list[list[int]] = [[] for _ in range(groups)]
    loads = np.zeros(groups, dtype=np.float64)
    for coordinate in order:
        available = [group for group in range(groups) if len(members[group]) < group_dim]
        selected = min(
            available,
            key=lambda group: (loads[group], len(members[group]), group),
        )
        members[selected].append(int(coordinate))
        loads[selected] += float(weights[coordinate])
    permutation = np.asarray(
        [coordinate for member_group in members for coordinate in member_group],
        dtype=np.int64,
    )
    rotation = np.asarray(eigenvectors[:, permutation], dtype=np.float32)
    group_log_determinants = [
        float(np.sum(log_values[np.asarray(member_group, dtype=np.int64)]))
        for member_group in members
    ]
    group_variances = [
        float(np.sum(clipped[np.asarray(member_group, dtype=np.int64)]))
        for member_group in members
    ]
    diagnostics = {
        "method": "pca_eigenvalue_log_volume_balancing",
        "eigenvalue_floor_relative": float(eigenvalue_floor_relative),
        "minimum_eigenvalue": float(eigenvalues[0]),
        "maximum_eigenvalue": float(eigenvalues[-1]),
        "clipped_condition_number": float(np.max(clipped) / np.min(clipped)),
        "group_log_determinants": group_log_determinants,
        "group_variance_sums": group_variances,
        "group_log_determinant_range": float(
            max(group_log_determinants) - min(group_log_determinants)
        ),
        "permutation": permutation.tolist(),
        "orthogonality_max_error": float(
            np.max(
                np.abs(
                    rotation.astype(np.float64).T @ rotation.astype(np.float64)
                    - np.eye(matrix.shape[0])
                )
            )
        ),
    }
    return rotation, diagnostics


def covariance_balanced_rotation(
    values: np.ndarray,
    groups: int,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    """Estimate covariance and return an eigenvalue-balanced OPQ initializer."""
    covariance = transformed_residual_covariance(values, device, batch_size)
    return eigenvalue_balanced_rotation_from_covariance(covariance, groups)


def materialize_residual_transform(
    source: np.ndarray,
    target: np.ndarray,
    transform: np.ndarray,
    device: torch.device,
    batch_size: int,
    log_interval: int = 0,
) -> None:
    """Write ``target = source @ transform`` while keeping grouped storage."""
    values = np.asarray(source)
    output = np.asarray(target)
    matrix_np = np.asarray(transform, dtype=np.float32)
    if (
        values.ndim != 3
        or output.shape != values.shape
        or output.dtype != np.float32
        or matrix_np.shape != (values.shape[1] * values.shape[2],) * 2
        or batch_size <= 0
    ):
        raise ValueError("invalid transformed residual cache configuration")
    matrix = torch.as_tensor(matrix_np, device=device)
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            end = min(values.shape[0], start + batch_size)
            batch = torch.as_tensor(
                np.array(
                    values[start:end],
                    dtype=np.float32,
                    order="C",
                    copy=True,
                ),
                device=device,
            ).reshape(end - start, -1)
            transformed = batch @ matrix
            output[start:end] = (
                transformed.reshape(end - start, values.shape[1], values.shape[2])
                .to("cpu")
                .numpy()
            )
            if log_interval > 0 and (
                end % log_interval < batch_size or end == values.shape[0]
            ):
                print(
                    f"[residual-transform-cache] {end}/{values.shape[0]}",
                    flush=True,
                )


def _cross_covariance_with_pq_reconstruction(
    values: np.ndarray,
    assignments: np.ndarray,
    prototypes: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    rows, groups, group_dim = values.shape
    dim = groups * group_dim
    centers = torch.as_tensor(
        np.asarray(prototypes, dtype=np.float32), device=device
    )
    group_ids = torch.arange(groups, device=device)[None, :]
    cross = torch.zeros((dim, dim), dtype=torch.float64, device="cpu")
    with torch.no_grad():
        for start in range(0, rows, batch_size):
            end = min(rows, start + batch_size)
            batch = torch.as_tensor(
                np.ascontiguousarray(values[start:end], dtype=np.float32),
                device=device,
            ).reshape(end - start, dim)
            labels = torch.as_tensor(
                np.ascontiguousarray(assignments[start:end], dtype=np.uint8),
                dtype=torch.long,
                device=device,
            )
            reconstruction = centers[group_ids, labels].reshape(end - start, dim)
            local = batch.T @ reconstruction
            cross += local.to(device="cpu", dtype=torch.float64)
    return cross.numpy()


def update_opq_rotation(
    transformed_residual: np.ndarray,
    prototypes: np.ndarray,
    current_rotation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> tuple[np.ndarray, dict]:
    """Run one assignment/Procrustes update for ``Y = (rL) @ Q``.

    ``current_rotation`` is the row-coordinate orthogonal matrix ``Q``.  The
    returned matrix minimizes the squared distance from the unrotated metric
    coordinates to the current PQ reconstructions.
    """
    values = np.asarray(transformed_residual)
    centers = np.asarray(prototypes, dtype=np.float32)
    rotation = np.asarray(current_rotation, dtype=np.float64)
    if (
        values.ndim != 3
        or centers.shape[0] != values.shape[1]
        or centers.shape[2] != values.shape[2]
        or rotation.shape != (values.shape[1] * values.shape[2],) * 2
    ):
        raise ValueError("invalid residual OPQ update configuration")

    assignments = nearest_group_codes(
        values, centers, device=device, batch_size=batch_size
    )
    cross_y = _cross_covariance_with_pq_reconstruction(
        values,
        assignments,
        centers,
        device=device,
        batch_size=batch_size,
    )
    # Y = Z Q, hence Z^T Y_hat = Q Y^T Y_hat.
    cross_z = rotation @ cross_y
    left, singular_values, right_t = np.linalg.svd(
        cross_z, full_matrices=False
    )
    updated = left @ right_t
    orthogonality_error = float(
        np.max(np.abs(updated.T @ updated - np.eye(updated.shape[0])))
    )
    diagnostics = {
        "singular_value_sum": float(np.sum(singular_values)),
        "min_singular_value": float(singular_values[-1]),
        "orthogonality_max_error": orthogonality_error,
        "assignment_changes_not_tracked": True,
    }
    return updated.astype(np.float32), diagnostics


def rotate_transformed_cache_in_place(
    values: np.ndarray,
    current_rotation: np.ndarray,
    updated_rotation: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> None:
    """Change cached rows from ``Z Q_old`` to ``Z Q_new`` in place."""
    cache = np.asarray(values)
    old = np.asarray(current_rotation, dtype=np.float64)
    new = np.asarray(updated_rotation, dtype=np.float64)
    if (
        cache.ndim != 3
        or old.shape != new.shape
        or old.shape[0] != cache.shape[1] * cache.shape[2]
    ):
        raise ValueError("invalid in-place residual rotation")
    update = torch.as_tensor(
        np.asarray(old.T @ new, dtype=np.float32), device=device
    )
    dim = old.shape[0]
    with torch.no_grad():
        for start in range(0, cache.shape[0], batch_size):
            end = min(cache.shape[0], start + batch_size)
            batch = torch.as_tensor(
                np.ascontiguousarray(cache[start:end], dtype=np.float32),
                device=device,
            ).reshape(end - start, dim)
            cache[start:end] = (
                (batch @ update)
                .reshape(end - start, cache.shape[1], cache.shape[2])
                .to("cpu")
                .numpy()
            )
