"""Numerical kernels for the 8/9-bit variable-rate primary-PQ baseline."""

from __future__ import annotations

import numpy as np
import torch


def split_initialize_9bit_centroids(
    vectors: np.ndarray,
    primary_codes: np.ndarray,
    primary_centroids: np.ndarray,
    split_scale: float = 0.5,
) -> np.ndarray:
    """Split every existing 8-bit cell along its leading residual direction.

    This gives the 512-centroid Lloyd run a nested, strong initialization: the
    old codebook is not discarded and every old cell starts with two children.
    """
    values = np.asarray(vectors, dtype=np.float32)
    codes = np.asarray(primary_codes, dtype=np.uint8)
    old = np.asarray(primary_centroids, dtype=np.float32)
    if (
        values.ndim != 3
        or codes.shape != values.shape[:2]
        or old.ndim != 3
        or old.shape[0] != values.shape[1]
        or old.shape[1] != 256
        or old.shape[2] != values.shape[2]
        or split_scale < 0.0
    ):
        raise ValueError("invalid 9-bit split initialization inputs")
    rows, subspaces, dsub = values.shape
    result = np.empty((subspaces, 512, dsub), dtype=np.float32)
    for subspace in range(subspaces):
        labels = codes[:, subspace].astype(np.int64, copy=False)
        residual = (
            values[:, subspace]
            - old[subspace, labels]
        ).astype(np.float64, copy=False)
        counts = np.bincount(labels, minlength=256).astype(np.float64)
        safe = np.maximum(counts, 1.0)
        sums = np.stack(
            [
                np.bincount(labels, weights=residual[:, dim], minlength=256)
                for dim in range(dsub)
            ],
            axis=1,
        )
        means = sums / safe[:, None]
        covariances = np.empty((256, dsub, dsub), dtype=np.float64)
        for left in range(dsub):
            for right in range(dsub):
                second = np.bincount(
                    labels,
                    weights=residual[:, left] * residual[:, right],
                    minlength=256,
                ) / safe
                covariances[:, left, right] = (
                    second - means[:, left] * means[:, right]
                )
        covariances = 0.5 * (
            covariances + np.swapaxes(covariances, 1, 2)
        )
        eigenvalues, eigenvectors = np.linalg.eigh(covariances)
        leading_values = np.maximum(eigenvalues[:, -1], 0.0)
        leading_vectors = eigenvectors[:, :, -1]
        delta = (
            float(split_scale)
            * np.sqrt(leading_values)[:, None]
            * leading_vectors
        )
        parent = old[subspace].astype(np.float64) + means
        invalid = counts < 2
        if np.any(invalid):
            parent[invalid] = old[subspace, invalid]
            delta[invalid] = 0.0
        result[subspace, :256] = (parent - delta).astype(np.float32)
        result[subspace, 256:] = (parent + delta).astype(np.float32)
    return result


def assigned_8bit_mse(
    vectors: np.ndarray,
    primary_codes: np.ndarray,
    primary_centroids: np.ndarray,
) -> np.ndarray:
    values = np.asarray(vectors, dtype=np.float32)
    codes = np.asarray(primary_codes, dtype=np.uint8)
    centers = np.asarray(primary_centroids, dtype=np.float32)
    if (
        values.ndim != 3
        or codes.shape != values.shape[:2]
        or centers.shape
        != (values.shape[1], 256, values.shape[2])
    ):
        raise ValueError("invalid 8-bit MSE inputs")
    result = np.zeros(values.shape[1], dtype=np.float64)
    for subspace in range(values.shape[1]):
        labels = codes[:, subspace].astype(np.int64, copy=False)
        error = values[:, subspace] - centers[subspace, labels]
        result[subspace] = float(
            np.mean(np.sum(error.astype(np.float64) ** 2, axis=1))
        )
    return result


def nearest_9bit_codes(
    vectors: np.ndarray,
    centroids: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Assign [rows, groups, dsub] values to 512 centers per group."""
    values = np.asarray(vectors, dtype=np.float32)
    centers_np = np.asarray(centroids, dtype=np.float32)
    if (
        values.ndim != 3
        or centers_np.shape != (values.shape[1], 512, values.shape[2])
        or batch_size <= 0
    ):
        raise ValueError("invalid 9-bit assignment shapes")
    centers = torch.as_tensor(centers_np, device=device)
    center_norm = torch.sum(centers * centers, dim=2)
    result = np.empty(values.shape[:2], dtype=np.uint16)
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
                torch.argmin(distance, dim=2)
                .to("cpu")
                .numpy()
                .astype(np.uint16)
            )
    return result


def nearest_metric_codes(
    vectors: np.ndarray,
    centroids: np.ndarray,
    metric_factors: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Assign values under ||(x-c)L||^2, where A=L L^T."""
    values = np.asarray(vectors, dtype=np.float32)
    centers_np = np.asarray(centroids, dtype=np.float32)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or centers_np.ndim != 3
        or centers_np.shape[0] != values.shape[1]
        or centers_np.shape[2] != values.shape[2]
        or centers_np.shape[1] not in (256, 512)
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or batch_size <= 0
    ):
        raise ValueError("invalid metric primary-PQ assignment shapes")
    factors = torch.as_tensor(factors_np, device=device)
    centers = torch.as_tensor(centers_np, device=device)
    metric_centers = torch.einsum("gkd,gde->gke", centers, factors)
    center_norm = torch.sum(metric_centers * metric_centers, dim=2)
    result = np.empty(values.shape[:2], dtype=np.uint16)
    with torch.no_grad():
        for start in range(0, values.shape[0], batch_size):
            end = min(values.shape[0], start + batch_size)
            batch = torch.as_tensor(
                np.ascontiguousarray(values[start:end]), device=device
            )
            metric_batch = torch.einsum("bgd,gde->bge", batch, factors)
            distance = (
                torch.sum(metric_batch * metric_batch, dim=2)[:, :, None]
                + center_norm[None, :, :]
                - 2.0 * torch.einsum(
                    "bgd,gkd->bgk", metric_batch, metric_centers
                )
            )
            result[start:end] = (
                torch.argmin(distance, dim=2)
                .to("cpu")
                .numpy()
                .astype(np.uint16)
            )
    return result


def lloyd_metric_primary_kmeans(
    vectors: np.ndarray,
    initial_centroids: np.ndarray,
    metric_factors: np.ndarray,
    iterations: int,
    convergence_tolerance: float,
    device: torch.device,
    batch_size: int,
    log_prefix: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Metric-assignment Lloyd for independent 8- or 9-bit subspaces."""
    values = np.asarray(vectors)
    initial = np.asarray(initial_centroids, dtype=np.float32)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or initial.ndim != 3
        or initial.shape[0] != values.shape[1]
        or initial.shape[2] != values.shape[2]
        or initial.shape[1] not in (256, 512)
        or values.shape[0] < initial.shape[1]
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or iterations <= 0
        or convergence_tolerance < 0.0
        or batch_size <= 0
    ):
        raise ValueError("invalid metric primary-PQ Lloyd configuration")
    groups, dimensions = values.shape[1], values.shape[2]
    codewords = initial.shape[1]
    centers = torch.as_tensor(initial, device=device).clone()
    factors = torch.as_tensor(factors_np, device=device)
    relative_shift = float("inf")
    completed = 0
    metric_mse = np.full(groups, np.inf, dtype=np.float64)
    with torch.no_grad():
        for iteration in range(iterations):
            sums = torch.zeros(
                (groups, codewords, dimensions), dtype=torch.float32, device=device
            )
            counts = torch.zeros(
                (groups, codewords), dtype=torch.int64, device=device
            )
            distortion = torch.zeros(groups, dtype=torch.float64, device=device)
            metric_centers = torch.einsum("gkd,gde->gke", centers, factors)
            center_norm = torch.sum(metric_centers * metric_centers, dim=2)
            for start in range(0, values.shape[0], batch_size):
                end = min(values.shape[0], start + batch_size)
                batch = torch.as_tensor(
                    np.ascontiguousarray(values[start:end], dtype=np.float32),
                    device=device,
                )
                metric_batch = torch.einsum("bgd,gde->bge", batch, factors)
                distance = (
                    torch.sum(metric_batch * metric_batch, dim=2)[:, :, None]
                    + center_norm[None, :, :]
                    - 2.0
                    * torch.einsum("bgd,gkd->bgk", metric_batch, metric_centers)
                )
                labels = torch.argmin(distance, dim=2)
                chosen = torch.gather(distance, 2, labels[:, :, None])[:, :, 0]
                distortion += torch.sum(
                    torch.clamp_min(chosen, 0.0).to(torch.float64), dim=0
                )
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
            metric_mse = (distortion / float(values.shape[0])).cpu().numpy()
            completed = iteration + 1
            print(
                f"[{log_prefix}] iter={completed} "
                f"metric_mse={float(np.mean(metric_mse)):.8g} "
                f"relative_shift={relative_shift:.8g} "
                f"empty={int(torch.sum(~occupied).item())}",
                flush=True,
            )
            if relative_shift <= convergence_tolerance:
                break
    final_centers = centers.to("cpu").numpy().astype(np.float32)
    labels = nearest_metric_codes(
        values,
        final_centers,
        factors_np,
        device=device,
        batch_size=batch_size,
    )
    counts_np = np.empty((groups, codewords), dtype=np.int64)
    euclidean_mse = np.empty(groups, dtype=np.float64)
    metric_mse = np.empty(groups, dtype=np.float64)
    for group in range(groups):
        group_labels = labels[:, group].astype(np.int64, copy=False)
        counts_np[group] = np.bincount(group_labels, minlength=codewords)
        error = (
            np.asarray(values[:, group], dtype=np.float32)
            - final_centers[group, group_labels]
        ).astype(np.float64)
        euclidean_mse[group] = float(np.mean(np.sum(error * error, axis=1)))
        metric_error = error @ factors_np[group].astype(np.float64)
        metric_mse[group] = float(
            np.mean(np.sum(metric_error * metric_error, axis=1))
        )
    diagnostics = {
        "iterations": completed,
        "relative_centroid_shift": relative_shift,
        "empty_sample_clusters": int(np.sum(counts_np == 0)),
        "mean_final_metric_mse": float(np.mean(metric_mse)),
        "mean_final_euclidean_mse": float(np.mean(euclidean_mse)),
    }
    return final_centers, counts_np, euclidean_mse, diagnostics


def lloyd_9bit_kmeans(
    vectors: np.ndarray,
    initial_centroids: np.ndarray,
    iterations: int,
    convergence_tolerance: float,
    device: torch.device,
    batch_size: int,
    log_prefix: str = "9bit-kmeans",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """Train several independent 512-centroid subspaces in one streamed run."""
    values = np.asarray(vectors)
    initial = np.asarray(initial_centroids, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < 512
        or initial.shape != (values.shape[1], 512, values.shape[2])
        or iterations <= 0
        or convergence_tolerance < 0.0
        or batch_size <= 0
    ):
        raise ValueError("invalid 9-bit Lloyd configuration")
    groups, dimensions = values.shape[1], values.shape[2]
    centers = torch.as_tensor(initial, device=device).clone()
    final_counts = torch.zeros(
        (groups, 512), dtype=torch.int64, device=device
    )
    mse = np.full(groups, np.inf, dtype=np.float64)
    completed = 0
    relative_shift = float("inf")
    with torch.no_grad():
        for iteration in range(iterations):
            sums = torch.zeros(
                (groups, 512, dimensions), dtype=torch.float32, device=device
            )
            counts = torch.zeros(
                (groups, 512), dtype=torch.int64, device=device
            )
            distortion = torch.zeros(groups, dtype=torch.float64, device=device)
            center_norm = torch.sum(centers * centers, dim=2)
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
                chosen = torch.gather(distance, 2, labels[:, :, None])[:, :, 0]
                distortion += torch.sum(
                    torch.clamp_min(chosen, 0.0).to(torch.float64), dim=0
                )
                for group in range(groups):
                    counts[group] += torch.bincount(
                        labels[:, group], minlength=512
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
            mse = (distortion / float(values.shape[0])).cpu().numpy()
            completed = iteration + 1
            print(
                f"[{log_prefix}] iter={completed} "
                f"mean_mse={float(np.mean(mse)):.8g} "
                f"relative_shift={relative_shift:.8g} "
                f"empty={int(torch.sum(~occupied).item())}",
                flush=True,
            )
            if relative_shift <= convergence_tolerance:
                break
    # Reassign once against the final centers so reported MSE/counts describe
    # the serialized model, rather than the pre-update centers of the last pass.
    final_centers = centers.to("cpu").numpy().astype(np.float32)
    labels = nearest_9bit_codes(
        values, final_centers, device=device, batch_size=batch_size
    )
    counts_np = np.empty((groups, 512), dtype=np.int64)
    mse = np.empty(groups, dtype=np.float64)
    for group in range(groups):
        group_labels = labels[:, group].astype(np.int64, copy=False)
        counts_np[group] = np.bincount(group_labels, minlength=512)
        error = (
            np.asarray(values[:, group], dtype=np.float32)
            - final_centers[group, group_labels]
        )
        mse[group] = float(
            np.mean(np.sum(error.astype(np.float64) ** 2, axis=1))
        )
    diagnostics = {
        "iterations": completed,
        "relative_centroid_shift": relative_shift,
        "empty_sample_clusters": int(np.sum(counts_np == 0)),
    }
    return final_centers, counts_np, mse, diagnostics
