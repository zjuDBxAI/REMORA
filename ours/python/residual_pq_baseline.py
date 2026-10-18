"""Numerical helpers for a full-dimensional residual-PQ baseline."""

from __future__ import annotations

import numpy as np
import torch

from additive_residual_quantizer import (
    additive_reconstruction,
    conditional_prediction,
)


def is_identity_metric(metric_matrices: np.ndarray) -> bool:
    matrices = np.asarray(metric_matrices, dtype=np.float32)
    if matrices.ndim != 3 or matrices.shape[1] != matrices.shape[2]:
        raise ValueError("metric matrices must have shape [groups, dim, dim]")
    identity = np.broadcast_to(
        np.eye(matrices.shape[1], dtype=np.float32), matrices.shape
    )
    return bool(np.array_equal(matrices, identity))


def reconstruct_primary_pq(pq_codes: np.ndarray, pq_model) -> np.ndarray:
    codes = np.asarray(pq_codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[1] != pq_model.subspaces:
        raise ValueError("primary PQ-code shape mismatch")
    subspaces = np.arange(pq_model.subspaces, dtype=np.int64)[None, :]
    reconstructed = pq_model.centroids[subspaces, codes]
    return np.ascontiguousarray(
        reconstructed.reshape(codes.shape[0], pq_model.dim), dtype=np.float32
    )


def extract_full_residual(
    vectors_t: np.ndarray, pq_codes: np.ndarray, pq_model, groups: int
) -> np.ndarray:
    vectors = np.asarray(vectors_t, dtype=np.float32)
    group_count = int(groups)
    if (
        vectors.ndim != 2
        or vectors.shape[1] != pq_model.dim
        or group_count <= 0
        or pq_model.dim % group_count != 0
    ):
        raise ValueError("full residual input shape mismatch")
    residual = vectors - reconstruct_primary_pq(pq_codes, pq_model)
    return np.ascontiguousarray(
        residual.reshape(vectors.shape[0], group_count, pq_model.dim // group_count),
        dtype=np.float32,
    )


def transform_grouped_residual(
    residual: np.ndarray, transform: np.ndarray
) -> np.ndarray:
    """Apply the row-coordinate transform ``y = r @ transform``."""
    values = np.asarray(residual, dtype=np.float32)
    matrix = np.asarray(transform, dtype=np.float32)
    if (
        values.ndim != 3
        or matrix.shape != (values.shape[1] * values.shape[2],) * 2
    ):
        raise ValueError("residual transform shape mismatch")
    flat = values.reshape(values.shape[0], -1)
    transformed = flat @ matrix
    return np.ascontiguousarray(transformed.reshape(values.shape), dtype=np.float32)


def residual_query_groups(query: np.ndarray, model) -> np.ndarray:
    """Map a primary-PQ-space query to the residual code coordinates."""
    if model.quantizer_kind != "product":
        raise ValueError("grouped residual queries require a product quantizer")
    transformed = residual_query_coordinates(query, model)
    return np.ascontiguousarray(
        transformed.reshape(int(model.groups), int(model.group_dim)),
        dtype=np.float32,
    )


def residual_query_coordinates(query: np.ndarray, model) -> np.ndarray:
    """Map a primary-PQ-space query to flat residual code coordinates."""
    value = np.asarray(query, dtype=np.float32)
    if value.shape != (int(model.dim),):
        raise ValueError("residual query shape mismatch")
    inverse = np.asarray(model.residual_inverse_transform, dtype=np.float32)
    transformed = value @ inverse.T
    return np.ascontiguousarray(transformed, dtype=np.float32)


def residual_codebook_lut(query: np.ndarray, model) -> np.ndarray:
    """Build one scalar query LUT entry per side dictionary/codeword."""
    coordinates = np.asarray(query, dtype=np.float32)
    if coordinates.shape != (int(model.dim),):
        raise ValueError("residual code-coordinate query shape mismatch")
    prototypes = np.asarray(model.prototypes, dtype=np.float32)
    if model.quantizer_kind == "additive":
        return np.einsum("gkd,d->gk", prototypes, coordinates, optimize=True).astype(
            np.float32
        )
    grouped = coordinates.reshape(int(model.groups), int(model.group_dim))
    return np.einsum("gkd,gd->gk", prototypes, grouped, optimize=True).astype(
        np.float32
    )


def primary_conditioning_lut(query: np.ndarray, model) -> np.ndarray | None:
    """Return the side-information score table indexed by primary PQ codes."""
    if not model.has_primary_conditioning:
        return None
    coordinates = np.asarray(query, dtype=np.float32)
    projection = coordinates @ np.asarray(model.conditional_basis, dtype=np.float32)
    return np.einsum(
        "mkh,h->mk",
        np.asarray(model.conditional_embeddings, dtype=np.float32),
        projection,
        optimize=True,
    ).astype(np.float32)


def coded_residual_reconstruction(
    side_codes: np.ndarray,
    primary_codes: np.ndarray,
    model,
) -> np.ndarray:
    """Reconstruct residuals in code coordinates before the global inverse."""
    side = np.asarray(side_codes, dtype=np.uint8)
    primary = np.asarray(primary_codes, dtype=np.uint8)
    if side.ndim != 2 or side.shape[1] != int(model.groups):
        raise ValueError("side-code shape mismatch")
    if model.quantizer_kind == "additive":
        reconstruction = additive_reconstruction(side, model.prototypes)
    else:
        dictionaries = np.arange(int(model.groups), dtype=np.int64)[None, :]
        reconstruction = np.asarray(
            model.prototypes[dictionaries, side], dtype=np.float32
        ).reshape(side.shape[0], int(model.dim))
    if model.has_primary_conditioning:
        reconstruction += conditional_prediction(
            primary, model.conditional_basis, model.conditional_embeddings
        )
    return np.ascontiguousarray(reconstruction, dtype=np.float32)


def decode_coded_residual(values: np.ndarray, model) -> np.ndarray:
    """Decode flat residual code coordinates into primary-PQ coordinates."""
    flat = np.asarray(values, dtype=np.float32)
    if flat.ndim != 2 or flat.shape[1] != int(model.dim):
        raise ValueError("coded residual reconstruction shape mismatch")
    decoded = (
        flat
        if not model.has_residual_transform
        else flat @ np.asarray(model.residual_inverse_transform, dtype=np.float32)
    )
    return np.ascontiguousarray(decoded, dtype=np.float32)


def decode_grouped_residual(values: np.ndarray, model) -> np.ndarray:
    """Decode residual code coordinates into the primary-PQ coordinate space."""
    grouped = np.asarray(values, dtype=np.float32)
    if grouped.ndim < 2 or grouped.shape[-2:] != (
        int(model.groups),
        int(model.group_dim),
    ):
        raise ValueError("grouped residual reconstruction shape mismatch")
    leading = grouped.shape[:-2]
    flat = grouped.reshape(-1, int(model.dim))
    decoded = decode_coded_residual(flat, model)
    return np.ascontiguousarray(
        decoded.reshape(*leading, int(model.dim)), dtype=np.float32
    )


def residual_pq_correction(
    side_codes: np.ndarray, query_groups: np.ndarray, prototypes: np.ndarray
) -> np.ndarray:
    codes = np.asarray(side_codes, dtype=np.uint8)
    queries = np.asarray(query_groups, dtype=np.float32)
    centers = np.asarray(prototypes, dtype=np.float32)
    if (
        codes.ndim != 2
        or centers.ndim != 3
        or queries.shape != (centers.shape[0], centers.shape[2])
        or codes.shape[1] != centers.shape[0]
    ):
        raise ValueError("residual-PQ correction shape mismatch")
    result = np.zeros(codes.shape[0], dtype=np.float32)
    for group in range(centers.shape[0]):
        lut = centers[group] @ queries[group]
        result += lut[codes[:, group]]
    return result


def nearest_metric_group_codes(
    residual: np.ndarray,
    prototypes: np.ndarray,
    metric_factors: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    """Assign residuals under ||(r-mu)L||^2 with A=L L^T."""
    values = np.asarray(residual, dtype=np.float32)
    centers_np = np.asarray(prototypes, dtype=np.float32)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or centers_np.ndim != 3
        or values.shape[1] != centers_np.shape[0]
        or values.shape[2] != centers_np.shape[2]
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or centers_np.shape[1] > 256
        or batch_size <= 0
    ):
        raise ValueError("metric residual assignment shape mismatch")
    factors = torch.as_tensor(factors_np, device=device)
    centers = torch.as_tensor(centers_np, device=device)
    metric_centers = torch.einsum("gkd,gde->gke", centers, factors)
    center_norm = torch.sum(metric_centers * metric_centers, dim=2)
    result = np.empty((values.shape[0], values.shape[1]), dtype=np.uint8)
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
                - 2.0
                * torch.einsum(
                    "bgd,gkd->bgk", metric_batch, metric_centers
                )
            )
            result[start:end] = (
                torch.argmin(distance, dim=2).to("cpu").numpy().astype(np.uint8)
            )
    return result


def metric_group_kmeans_plus_plus_initialization(
    residual: np.ndarray,
    metric_factors: np.ndarray,
    codewords: int,
    device: torch.device,
    batch_size: int,
    seed: int,
    log_interval: int = 1,
) -> np.ndarray:
    """Initialize group codebooks with k-means++ under the metric objective.

    For each group, this runs standard D^2 sampling in the transformed space
    ``z_g = A_g^{1/2} r_g`` and returns the selected centers in the original
    residual coordinates.  The subsequent Lloyd objective and online scorer are
    unchanged, but this initialization gives the classical O(log K)
    approximation link to the optimal k-means objective.
    """
    values = np.asarray(residual)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < codewords
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or codewords <= 1
        or codewords > 256
        or batch_size <= 0
    ):
        raise ValueError("invalid metric k-means++ initialization configuration")

    rng = np.random.default_rng(seed)
    rows, groups, dimensions = values.shape
    initial = np.empty((groups, codewords, dimensions), dtype=np.float32)
    progress_step = max(1, codewords // 4)

    with torch.no_grad():
        for group in range(groups):
            factor = torch.as_tensor(factors_np[group], device=device)
            first = int(rng.integers(rows))
            initial[group, 0] = np.asarray(values[first, group], dtype=np.float32)
            center = torch.as_tensor(initial[group, 0], device=device) @ factor
            closest = np.empty(rows, dtype=np.float64)

            for start in range(0, rows, batch_size):
                end = min(rows, start + batch_size)
                batch = torch.as_tensor(
                    np.ascontiguousarray(
                        values[start:end, group], dtype=np.float32
                    ),
                    device=device,
                )
                transformed = batch @ factor
                distance = torch.sum((transformed - center[None, :]) ** 2, dim=1)
                closest[start:end] = (
                    distance.clamp_min_(0.0)
                    .to("cpu")
                    .numpy()
                    .astype(np.float64)
                )

            for codeword in range(1, codewords):
                total = float(np.sum(closest))
                if not np.isfinite(total) or total <= 0.0:
                    chosen = int(rng.integers(rows))
                else:
                    threshold = float(rng.random()) * total
                    chosen = int(np.searchsorted(np.cumsum(closest), threshold))
                    if chosen >= rows:
                        chosen = rows - 1
                initial[group, codeword] = np.asarray(
                    values[chosen, group], dtype=np.float32
                )
                center = torch.as_tensor(initial[group, codeword], device=device) @ factor

                for start in range(0, rows, batch_size):
                    end = min(rows, start + batch_size)
                    batch = torch.as_tensor(
                        np.ascontiguousarray(
                            values[start:end, group], dtype=np.float32
                        ),
                        device=device,
                    )
                    transformed = batch @ factor
                    distance = torch.sum((transformed - center[None, :]) ** 2, dim=1)
                    np.minimum(
                        closest[start:end],
                        distance.clamp_min_(0.0)
                        .to("cpu")
                        .numpy()
                        .astype(np.float64),
                        out=closest[start:end],
                    )

                if log_interval > 0 and (
                    codeword + 1 == codewords
                    or (codeword + 1) % progress_step == 0
                ):
                    print(
                        f"[metric-kmeans++] group={group + 1}/{groups} "
                        f"centers={codeword + 1}/{codewords} "
                        f"mean_closest={np.mean(closest):.8g}",
                        flush=True,
                    )

    return initial


def lloyd_metric_group_kmeans(
    residual: np.ndarray,
    metric_factors: np.ndarray,
    codewords: int,
    iterations: int,
    convergence_tolerance: float,
    device: torch.device,
    batch_size: int,
    seed: int,
    initialization: str = "sample",
    initial_prototypes: np.ndarray | None = None,
    log_interval: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lloyd iterations with metric assignments and original-space means."""
    values = np.asarray(residual)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < codewords
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or codewords <= 1
        or codewords > 256
        or iterations <= 0
        or batch_size <= 0
        or convergence_tolerance < 0.0
        or initialization not in {"sample", "kmeans++"}
    ):
        raise ValueError("invalid metric group k-means configuration")
    groups, dimensions = values.shape[1], values.shape[2]
    if initial_prototypes is not None:
        initial = np.asarray(initial_prototypes, dtype=np.float32)
        if initial.shape != (groups, codewords, dimensions):
            raise ValueError("initial prototype shape mismatch")
        initial = initial.copy()
        initialization_used = "provided"
    elif initialization == "kmeans++":
        initial = metric_group_kmeans_plus_plus_initialization(
            values,
            factors_np,
            codewords=codewords,
            device=device,
            batch_size=batch_size,
            seed=seed,
            log_interval=log_interval,
        )
        initialization_used = initialization
    else:
        rng = np.random.default_rng(seed)
        initial = np.empty((groups, codewords, dimensions), dtype=np.float32)
        for group in range(groups):
            ids = rng.choice(values.shape[0], codewords, replace=False)
            initial[group] = np.asarray(values[ids, group], dtype=np.float32)
        initialization_used = initialization
    centers = torch.as_tensor(initial, device=device).clone()
    factors = torch.as_tensor(factors_np, device=device)
    final_counts = torch.zeros(
        (groups, codewords), dtype=torch.int64, device=device
    )
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
            metric_centers = torch.einsum("gkd,gde->gke", centers, factors)
            center_norm = torch.sum(metric_centers * metric_centers, dim=2)
            distortion = 0.0
            observations = 0
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
                    * torch.einsum(
                        "bgd,gkd->bgk", metric_batch, metric_centers
                    )
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
                    f"[metric-group-kmeans] iter={completed} "
                    f"metric_mse={metric_mse:.8g} "
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
        "initialization": initialization_used,
    }
    return (
        centers.to("cpu").numpy().astype(np.float32),
        final_counts.to("cpu").numpy().astype(np.int64),
        diagnostics,
    )


def lloyd_tail_aware_metric_group_kmeans(
    residual: np.ndarray,
    metric_factors: np.ndarray,
    codewords: int,
    iterations: int,
    convergence_tolerance: float,
    device: torch.device,
    batch_size: int,
    seed: int,
    tail_fraction: float,
    tail_cvar_weight: float,
    initial_prototypes: np.ndarray | None = None,
    initialization: str = "sample",
    log_interval: int = 1,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Lloyd refinement for mean metric MSE plus a CVaR-style tail term.

    The assignment step remains nearest-prototype under the metric.  The update
    step reweights the largest assigned distances in each group, approximating
    ``mean(distance) + beta * CVaR_tail(distance)`` without changing the code
    format or the online scorer.
    """
    values = np.asarray(residual)
    factors_np = np.asarray(metric_factors, dtype=np.float32)
    if (
        values.ndim != 3
        or values.shape[0] < codewords
        or factors_np.shape
        != (values.shape[1], values.shape[2], values.shape[2])
        or codewords <= 1
        or codewords > 256
        or iterations <= 0
        or batch_size <= 0
        or convergence_tolerance < 0.0
        or not 0.0 < tail_fraction <= 1.0
        or tail_cvar_weight < 0.0
        or initialization not in {"sample", "kmeans++"}
    ):
        raise ValueError("invalid tail-aware metric k-means configuration")
    groups, dimensions = values.shape[1], values.shape[2]
    if initial_prototypes is None:
        if initialization == "kmeans++":
            initial = metric_group_kmeans_plus_plus_initialization(
                values,
                factors_np,
                codewords=codewords,
                device=device,
                batch_size=batch_size,
                seed=seed,
                log_interval=log_interval,
            )
        else:
            rng = np.random.default_rng(seed)
            initial = np.empty((groups, codewords, dimensions), dtype=np.float32)
            for group in range(groups):
                ids = rng.choice(values.shape[0], codewords, replace=False)
                initial[group] = np.asarray(values[ids, group], dtype=np.float32)
    else:
        initial = np.asarray(initial_prototypes, dtype=np.float32)
        if initial.shape != (groups, codewords, dimensions):
            raise ValueError("initial prototype shape mismatch")

    centers = torch.as_tensor(initial, device=device).clone()
    factors = torch.as_tensor(factors_np, device=device)
    labels_np = np.empty((values.shape[0], groups), dtype=np.uint8)
    assigned_np = np.empty((values.shape[0], groups), dtype=np.float32)
    final_counts = torch.zeros(
        (groups, codewords), dtype=torch.int64, device=device
    )
    completed = 0
    relative_shift = float("inf")
    empty_clusters = 0
    mean_metric_mse = float("inf")
    tail_metric_mse = float("inf")
    objective = float("inf")
    tail_multiplier = 1.0 + float(tail_cvar_weight) / float(tail_fraction)

    with torch.no_grad():
        for iteration in range(iterations):
            metric_centers = torch.einsum("gkd,gde->gke", centers, factors)
            center_norm = torch.sum(metric_centers * metric_centers, dim=2)
            distortion = 0.0
            observations = 0
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
                    * torch.einsum(
                        "bgd,gkd->bgk", metric_batch, metric_centers
                    )
                )
                labels = torch.argmin(distance, dim=2)
                assigned = torch.gather(distance, 2, labels[:, :, None]).squeeze(2)
                labels_np[start:end] = labels.to("cpu").numpy().astype(np.uint8)
                assigned_np[start:end] = assigned.to("cpu").numpy().astype(np.float32)
                distortion += float(assigned.sum().item())
                observations += int(batch.shape[0] * groups)

            mean_metric_mse = distortion / max(observations, 1)
            tail_thresholds = np.quantile(
                assigned_np.astype(np.float64, copy=False),
                1.0 - float(tail_fraction),
                axis=0,
            ).astype(np.float32)
            tail_mask = assigned_np >= tail_thresholds[None, :]
            tail_count = int(np.sum(tail_mask))
            tail_metric_mse = (
                float(np.mean(assigned_np[tail_mask])) if tail_count else 0.0
            )
            objective = mean_metric_mse + float(tail_cvar_weight) * tail_metric_mse

            sums = torch.zeros(
                (groups, codewords, dimensions), dtype=torch.float32, device=device
            )
            weighted_counts = torch.zeros(
                (groups, codewords), dtype=torch.float32, device=device
            )
            raw_counts = torch.zeros(
                (groups, codewords), dtype=torch.int64, device=device
            )
            for start in range(0, values.shape[0], batch_size):
                end = min(values.shape[0], start + batch_size)
                batch = torch.as_tensor(
                    np.ascontiguousarray(values[start:end], dtype=np.float32),
                    device=device,
                )
                labels = torch.as_tensor(labels_np[start:end], device=device)
                weights_np = np.ones((end - start, groups), dtype=np.float32)
                weights_np[tail_mask[start:end]] = tail_multiplier
                weights = torch.as_tensor(weights_np, device=device)
                for group in range(groups):
                    group_labels = labels[:, group].to(torch.int64)
                    group_weights = weights[:, group]
                    raw_counts[group] += torch.bincount(
                        group_labels, minlength=codewords
                    )
                    weighted_counts[group] += torch.bincount(
                        group_labels,
                        weights=group_weights,
                        minlength=codewords,
                    )
                    sums[group].index_add_(
                        0,
                        group_labels,
                        batch[:, group] * group_weights[:, None],
                    )

            occupied = weighted_counts > 0
            updated = centers.clone()
            updated[occupied] = (
                sums[occupied] / weighted_counts[occupied][:, None]
            )
            shift = torch.sum((updated - centers) ** 2)
            scale = torch.sum(centers**2).clamp_min(1e-30)
            relative_shift = float((shift / scale).item())
            centers = updated
            final_counts = raw_counts
            completed = iteration + 1
            empty_clusters = int(torch.sum(~occupied).item())
            if log_interval > 0 and (
                completed % log_interval == 0
                or completed == iterations
                or relative_shift <= convergence_tolerance
            ):
                print(
                    f"[tail-metric-kmeans] iter={completed} "
                    f"objective={objective:.8g} "
                    f"metric_mse={mean_metric_mse:.8g} "
                    f"tail_mse={tail_metric_mse:.8g} "
                    f"tail_rate={tail_count / max(observations, 1):.8g} "
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
        "final_metric_mse": mean_metric_mse,
        "final_tail_metric_mse": tail_metric_mse,
        "final_tail_aware_objective": objective,
        "tail_fraction": float(tail_fraction),
        "tail_cvar_weight": float(tail_cvar_weight),
        "tail_multiplier": float(tail_multiplier),
        "initialization": (
            "provided_prototypes"
            if initial_prototypes is not None
            else initialization
        ),
    }
    return (
        centers.to("cpu").numpy().astype(np.float32),
        final_counts.to("cpu").numpy().astype(np.int64),
        diagnostics,
    )
