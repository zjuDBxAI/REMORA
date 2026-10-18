"""Core utilities for joint-innovation, variable-rate residual quantization.

The model deliberately has its own artifact format.  The deployed residual-PQ
format assumes equal-dimensional groups with 256 entries each, while this
experiment permits arbitrary innovation-coordinate groups and a different
number of bits per group.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch


MODEL_VERSION = 1
SIDE_MAGIC = b"JIRCODE1"
SIDE_VERSION = 1
SIDE_MULTIWORD_VERSION = 2
SIDE_HEADER = "<IQIIIQQ"


@dataclass
class JointInnovationModel:
    artifact_id: int
    model_id: int
    node_count: int
    dim: int
    budget_bits: int
    metric_id: int
    metric_lambda: float
    residual_mean: np.ndarray
    forward_transform: np.ndarray
    inverse_transform: np.ndarray
    innovation_eigenvalues: np.ndarray
    group_indices: np.ndarray
    group_dims: np.ndarray
    bits: np.ndarray
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    variant: str

    @property
    def groups(self) -> int:
        return int(np.asarray(self.group_dims).size)

    @property
    def max_group_dim(self) -> int:
        return int(np.asarray(self.group_indices).shape[1])

    @property
    def max_codewords(self) -> int:
        return int(np.asarray(self.prototypes).shape[1])

    @property
    def side_bytes_per_node(self) -> int:
        return (int(self.budget_bits) + 7) // 8

    def indices(self, group: int) -> np.ndarray:
        return np.asarray(
            self.group_indices[group, : int(self.group_dims[group])], dtype=np.int64
        )

    def codebook(self, group: int) -> np.ndarray:
        count = 1 << int(self.bits[group])
        width = int(self.group_dims[group])
        return np.asarray(self.prototypes[group, :count, :width], dtype=np.float32)

    def validate(self) -> None:
        d = int(self.dim)
        g = self.groups
        group_dims = np.asarray(self.group_dims, dtype=np.int64)
        bits = np.asarray(self.bits, dtype=np.int64)
        indices = np.asarray(self.group_indices, dtype=np.int64)
        prototypes = np.asarray(self.prototypes)
        counts = np.asarray(self.prototype_counts)
        if (
            int(self.artifact_id) <= 0
            or int(self.model_id) <= 0
            or int(self.node_count) <= 0
            or d <= 0
            or g <= 0
            or int(self.budget_bits) <= 0
            or int(np.sum(bits)) != int(self.budget_bits)
            or np.any(bits < 0)
            or np.any(bits > 16)
            or np.any(group_dims <= 0)
            or indices.ndim != 2
            or indices.shape[0] != g
            or indices.shape[1] < int(np.max(group_dims))
            or prototypes.ndim != 3
            or prototypes.shape[0] != g
            or prototypes.shape[2] != indices.shape[1]
            or prototypes.shape[1] < (1 << int(np.max(bits)))
            or counts.shape != prototypes.shape[:2]
            or not str(self.variant)
        ):
            raise ValueError("invalid joint-innovation model dimensions")
        flattened = []
        for group in range(g):
            group_indices = self.indices(group)
            if np.any(group_indices < 0) or np.any(group_indices >= d):
                raise ValueError("innovation group index is outside the model dimension")
            flattened.extend(group_indices.tolist())
            if not np.all(np.isfinite(self.codebook(group))):
                raise ValueError("joint-innovation codebook contains non-finite values")
        if sorted(flattened) != list(range(d)):
            raise ValueError("innovation groups must partition every coordinate exactly once")
        if np.asarray(self.residual_mean).shape != (d,):
            raise ValueError("residual mean shape mismatch")
        if np.asarray(self.forward_transform).shape != (d, d):
            raise ValueError("forward transform shape mismatch")
        if np.asarray(self.inverse_transform).shape != (d, d):
            raise ValueError("inverse transform shape mismatch")
        if np.asarray(self.innovation_eigenvalues).shape != (d,):
            raise ValueError("innovation spectrum shape mismatch")
        for value in (
            self.residual_mean,
            self.forward_transform,
            self.inverse_transform,
            self.innovation_eigenvalues,
        ):
            if not np.all(np.isfinite(value)):
                raise ValueError("joint-innovation model contains non-finite values")
        identity = np.eye(d, dtype=np.float64)
        if not np.allclose(
            np.asarray(self.forward_transform, dtype=np.float64)
            @ np.asarray(self.inverse_transform, dtype=np.float64),
            identity,
            rtol=3e-4,
            atol=3e-5,
        ):
            raise ValueError("joint-innovation transforms are not inverses")


def compute_model_id(
    artifact_id: int,
    metric_id: int,
    forward_transform: np.ndarray,
    group_indices: np.ndarray,
    group_dims: np.ndarray,
    bits: np.ndarray,
    prototypes: np.ndarray,
) -> int:
    digest = hashlib.sha256()
    digest.update(struct.pack("<QQ", int(artifact_id), int(metric_id)))
    for value, dtype in (
        (forward_transform, "<f4"),
        (group_indices, "<i4"),
        (group_dims, "<u4"),
        (bits, "<u4"),
        (prototypes, "<f4"),
    ):
        digest.update(np.ascontiguousarray(value, dtype=dtype).tobytes())
    result = int.from_bytes(digest.digest()[:8], "little")
    return result if result != 0 else 1


def write_model(path: str, model: JointInnovationModel) -> None:
    model.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(MODEL_VERSION, dtype=np.uint32),
        artifact_id=np.asarray(model.artifact_id, dtype=np.uint64),
        model_id=np.asarray(model.model_id, dtype=np.uint64),
        node_count=np.asarray(model.node_count, dtype=np.uint64),
        dim=np.asarray(model.dim, dtype=np.uint32),
        budget_bits=np.asarray(model.budget_bits, dtype=np.uint32),
        metric_id=np.asarray(model.metric_id, dtype=np.uint64),
        metric_lambda=np.asarray(model.metric_lambda, dtype=np.float64),
        residual_mean=np.asarray(model.residual_mean, dtype=np.float32),
        forward_transform=np.asarray(model.forward_transform, dtype=np.float32),
        inverse_transform=np.asarray(model.inverse_transform, dtype=np.float32),
        innovation_eigenvalues=np.asarray(
            model.innovation_eigenvalues, dtype=np.float32
        ),
        group_indices=np.asarray(model.group_indices, dtype=np.int32),
        group_dims=np.asarray(model.group_dims, dtype=np.uint32),
        bits=np.asarray(model.bits, dtype=np.uint32),
        prototypes=np.asarray(model.prototypes, dtype=np.float32),
        prototype_counts=np.asarray(model.prototype_counts, dtype=np.int64),
        variant=np.asarray(str(model.variant)),
    )


def read_model(path: str) -> JointInnovationModel:
    with np.load(path, allow_pickle=False) as state:
        version = int(state["version"])
        if version != MODEL_VERSION:
            raise ValueError(f"unsupported joint-innovation model version: {version}")
        model = JointInnovationModel(
            artifact_id=int(state["artifact_id"]),
            model_id=int(state["model_id"]),
            node_count=int(state["node_count"]),
            dim=int(state["dim"]),
            budget_bits=int(state["budget_bits"]),
            metric_id=int(state["metric_id"]),
            metric_lambda=float(state["metric_lambda"]),
            residual_mean=np.asarray(state["residual_mean"], dtype=np.float32),
            forward_transform=np.asarray(
                state["forward_transform"], dtype=np.float32
            ),
            inverse_transform=np.asarray(
                state["inverse_transform"], dtype=np.float32
            ),
            innovation_eigenvalues=np.asarray(
                state["innovation_eigenvalues"], dtype=np.float32
            ),
            group_indices=np.asarray(state["group_indices"], dtype=np.int32),
            group_dims=np.asarray(state["group_dims"], dtype=np.uint32),
            bits=np.asarray(state["bits"], dtype=np.uint32),
            prototypes=np.asarray(state["prototypes"], dtype=np.float32),
            prototype_counts=np.asarray(state["prototype_counts"], dtype=np.int64),
            variant=str(state["variant"].item()),
        )
    model.validate()
    return model


def make_equal_groups(dim: int, groups: int) -> list[np.ndarray]:
    if dim <= 0 or groups <= 0 or groups > dim:
        raise ValueError("invalid equal-group configuration")
    return [np.asarray(part, dtype=np.int32) for part in np.array_split(np.arange(dim), groups)]


def pad_groups(groups: Sequence[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    dimensions = np.asarray([len(group) for group in groups], dtype=np.uint32)
    width = int(np.max(dimensions))
    result = np.full((len(groups), width), -1, dtype=np.int32)
    for index, group in enumerate(groups):
        result[index, : len(group)] = np.asarray(group, dtype=np.int32)
    return result, dimensions


def build_joint_transform(
    residual: np.ndarray,
    metric_matrix: np.ndarray,
    eigenvalue_floor_relative: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Return mean, row-forward, row-inverse and innovation spectrum."""
    values = np.asarray(residual, dtype=np.float64)
    metric = np.asarray(metric_matrix, dtype=np.float64)
    if values.ndim != 2 or metric.shape != (values.shape[1], values.shape[1]):
        raise ValueError("joint-transform shape mismatch")
    if not 0.0 < eigenvalue_floor_relative < 1.0:
        raise ValueError("eigenvalue floor must lie in (0, 1)")
    mean = np.mean(values, axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0], 1)
    covariance = 0.5 * (covariance + covariance.T)
    sigma_values, sigma_vectors = np.linalg.eigh(covariance)
    sigma_max = max(float(sigma_values[-1]), np.finfo(np.float64).tiny)
    sigma_floor = eigenvalue_floor_relative * sigma_max
    sigma_clipped = np.maximum(sigma_values, sigma_floor)
    sigma_sqrt = (sigma_vectors * np.sqrt(sigma_clipped)) @ sigma_vectors.T
    sigma_inverse_sqrt = (
        sigma_vectors * (1.0 / np.sqrt(sigma_clipped))
    ) @ sigma_vectors.T
    innovation = sigma_sqrt @ metric @ sigma_sqrt
    innovation = 0.5 * (innovation + innovation.T)
    innovation_values, innovation_vectors = np.linalg.eigh(innovation)
    innovation_raw_values = innovation_values.copy()
    innovation_max = max(
        float(innovation_values[-1]), np.finfo(np.float64).tiny
    )
    innovation_floor = eigenvalue_floor_relative * innovation_max
    innovation_values = np.maximum(innovation_values, innovation_floor)
    order = np.argsort(innovation_values)[::-1]
    innovation_values = innovation_values[order]
    innovation_vectors = innovation_vectors[:, order]
    forward = (
        sigma_inverse_sqrt
        @ innovation_vectors
        @ np.diag(np.sqrt(innovation_values))
    )
    inverse = (
        np.diag(1.0 / np.sqrt(innovation_values))
        @ innovation_vectors.T
        @ sigma_sqrt
    )
    realized = forward @ forward.T
    diagnostics = {
        "covariance_eigenvalue_floor_relative": float(eigenvalue_floor_relative),
        "covariance_eigenvalue_floor_absolute": float(sigma_floor),
        "covariance_raw_min_eigenvalue": float(sigma_values[0]),
        "covariance_raw_max_eigenvalue": float(sigma_values[-1]),
        "covariance_clipped_dimensions": int(np.sum(sigma_values < sigma_floor)),
        "innovation_min_eigenvalue": float(innovation_values[-1]),
        "innovation_max_eigenvalue": float(innovation_values[0]),
        "innovation_raw_min_eigenvalue": float(innovation_raw_values[0]),
        "innovation_eigenvalue_floor_absolute": float(innovation_floor),
        "innovation_clipped_dimensions": int(
            np.sum(innovation_raw_values < innovation_floor)
        ),
        "transform_inverse_max_error": float(
            np.max(np.abs(forward @ inverse - np.eye(values.shape[1])))
        ),
        "metric_realization_max_error": float(np.max(np.abs(realized - metric))),
    }
    return (
        mean.astype(np.float32),
        forward.astype(np.float32),
        inverse.astype(np.float32),
        innovation_values.astype(np.float32),
        diagnostics,
    )


def transform_values(
    residual: np.ndarray, mean: np.ndarray, forward: np.ndarray
) -> np.ndarray:
    return np.asarray(
        (np.asarray(residual, dtype=np.float32) - np.asarray(mean, dtype=np.float32))
        @ np.asarray(forward, dtype=np.float32),
        dtype=np.float32,
    )


def vq_backend_info(device: torch.device) -> dict:
    """Record the native nearest-codeword implementation, outside QPS timing."""
    if torch.device(device).type != "cpu":
        return {"backend": "torch", "device": str(device), "distance": "squared L2"}
    import faiss
    options = faiss.get_compile_options().strip()
    if "AVX512" not in options:
        raise RuntimeError("CPU JI VQ requires the AVX-512 Faiss build")
    return {"backend": "faiss-cpu-flat-l2-v1", "distance": "FP32 exhaustive squared L2",
            "faiss_version": faiss.__version__, "faiss_compile_options": options,
            "threads": faiss.omp_get_max_threads()}


def _cpu_vq_index(centers: np.ndarray):
    import faiss
    vq_backend_info(torch.device("cpu"))
    codebook = np.ascontiguousarray(centers, dtype=np.float32)
    if (codebook.ndim != 2 or not 0 < len(codebook) <= 65536
            or codebook.shape[1] == 0 or not np.isfinite(codebook).all()):
        raise ValueError("invalid finite CPU VQ codebook")
    index = faiss.IndexFlatL2(codebook.shape[1])
    index.add(codebook)
    return index


def _cpu_vq_nearest(index, values: np.ndarray):
    batch = np.ascontiguousarray(values, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[1] != index.d or not np.isfinite(batch).all():
        raise ValueError("invalid finite CPU VQ vectors")
    distance, labels = index.search(batch, 1)
    if not np.isfinite(distance).all() or np.any(labels < 0):
        raise ValueError("invalid CPU Faiss nearest-codeword results")
    return distance[:, 0], labels[:, 0].copy()


def fit_vq(
    train: np.ndarray,
    validation: np.ndarray,
    bits: int,
    iterations: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    initial: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Fit VQ; CPU assignment uses exhaustive Faiss, CUDA uses torch."""
    train_values = np.asarray(train, dtype=np.float32)
    validation_values = np.asarray(validation, dtype=np.float32)
    codewords = 1 << int(bits)
    if (
        train_values.ndim != 2
        or validation_values.ndim != 2
        or validation_values.shape[1] != train_values.shape[1]
        or bits < 0
        or bits > 16
        or train_values.shape[0] < codewords
        or iterations <= 0
        or batch_size <= 0
    ):
        raise ValueError("invalid variable-rate VQ configuration")
    rng = np.random.default_rng(seed)
    if initial is None:
        ids = rng.choice(train_values.shape[0], codewords, replace=False)
        centers_np = np.asarray(train_values[ids], dtype=np.float32)
    else:
        centers_np = np.asarray(initial, dtype=np.float32)
        if centers_np.shape != (codewords, train_values.shape[1]):
            raise ValueError("initial VQ codebook shape mismatch")
    centers = torch.as_tensor(centers_np, device=device).clone()
    cpu = torch.device(device).type == "cpu"
    final_counts = torch.zeros(codewords, dtype=torch.int64, device=device)
    final_train_mse = float("inf")
    completed = 0
    relative_shift = float("inf")
    with torch.no_grad():
        for iteration in range(iterations):
            sums = torch.zeros_like(centers)
            counts = torch.zeros(codewords, dtype=torch.int64, device=device)
            distortion = 0.0
            observations = 0
            cpu_index = _cpu_vq_index(centers.numpy()) if cpu else None
            center_norm = None if cpu else torch.sum(centers * centers, dim=1)
            for start in range(0, train_values.shape[0], batch_size):
                batch_np = np.ascontiguousarray(
                    train_values[start : start + batch_size], dtype=np.float32
                )
                batch = torch.as_tensor(batch_np, device=device)
                if cpu:
                    minimum, label_values = _cpu_vq_nearest(cpu_index, batch_np)
                    labels = torch.from_numpy(label_values)
                    distortion += float(minimum.sum(dtype=np.float64))
                else:
                    distance = (
                        torch.sum(batch * batch, dim=1)[:, None]
                        + center_norm[None, :]
                        - 2.0 * batch @ centers.T
                    )
                    labels = torch.argmin(distance, dim=1)
                    distortion += float(
                        torch.gather(distance, 1, labels[:, None]).sum().item()
                    )
                observations += int(batch.shape[0])
                counts += torch.bincount(labels, minlength=codewords)
                sums.index_add_(0, labels, batch)
            occupied = counts > 0
            updated = centers.clone()
            updated[occupied] = sums[occupied] / counts[occupied, None].float()
            relative_shift = float(
                (torch.sum((updated - centers) ** 2) / torch.sum(centers**2).clamp_min(1e-30)).item()
            )
            centers = updated
            final_counts = counts
            final_train_mse = distortion / max(observations, 1)
            completed = iteration + 1
            if relative_shift <= 1e-6:
                break

    evaluation_index = _cpu_vq_index(centers.numpy()) if cpu else None

    def evaluate(values: np.ndarray) -> float:
        distortion = 0.0
        observations = 0
        with torch.no_grad():
            center_norm = None if cpu else torch.sum(centers * centers, dim=1)
            for start in range(0, values.shape[0], batch_size):
                if cpu:
                    minimum, _ = _cpu_vq_nearest(evaluation_index, values[start:start + batch_size])
                    distortion += float(minimum.sum(dtype=np.float64))
                    observations += len(minimum)
                    continue
                batch = torch.as_tensor(
                    np.ascontiguousarray(values[start : start + batch_size]),
                    device=device,
                )
                distance = (
                    torch.sum(batch * batch, dim=1)[:, None]
                    + center_norm[None, :]
                    - 2.0 * batch @ centers.T
                )
                distortion += float(torch.min(distance, dim=1).values.sum().item())
                observations += int(batch.shape[0])
        return distortion / max(observations, 1)

    diagnostics = {
        "bits": int(bits),
        "codewords": int(codewords),
        "iterations": int(completed),
        "relative_centroid_shift": float(relative_shift),
        "empty_clusters": int(torch.sum(final_counts == 0).item()),
        "train_mse": float(final_train_mse),
        "validation_mse": float(evaluate(validation_values)),
        "assignment_backend": vq_backend_info(device),
    }
    return (
        centers.cpu().numpy().astype(np.float32),
        final_counts.cpu().numpy().astype(np.int64),
        diagnostics,
    )


def assign_vq(
    values: np.ndarray,
    centers: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> tuple[np.ndarray, float]:
    data = np.asarray(values, dtype=np.float32)
    if torch.device(device).type == "cpu":
        if batch_size <= 0 or data.ndim != 2:
            raise ValueError("invalid CPU VQ assignment configuration")
        index = _cpu_vq_index(centers)
        if data.shape[1] != index.d:
            raise ValueError("CPU VQ vector/codebook dimensions differ")
        dtype = np.uint8 if index.ntotal <= 256 else np.uint16
        labels_out = np.empty(data.shape[0], dtype=dtype)
        distortion = 0.0
        for start in range(0, len(data), batch_size):
            minimum, labels = _cpu_vq_nearest(index, data[start:start + batch_size])
            labels_out[start:start + len(labels)] = labels
            distortion += float(minimum.sum(dtype=np.float64))
        return labels_out, distortion / max(len(data), 1)
    codebook = torch.as_tensor(np.asarray(centers, dtype=np.float32), device=device)
    dtype = np.uint8 if codebook.shape[0] <= 256 else np.uint16
    labels_out = np.empty(data.shape[0], dtype=dtype)
    distortion = 0.0
    with torch.no_grad():
        center_norm = torch.sum(codebook * codebook, dim=1)
        for start in range(0, data.shape[0], batch_size):
            batch = torch.as_tensor(
                np.ascontiguousarray(data[start : start + batch_size]), device=device
            )
            distance = (
                torch.sum(batch * batch, dim=1)[:, None]
                + center_norm[None, :]
                - 2.0 * batch @ codebook.T
            )
            minimum, labels = torch.min(distance, dim=1)
            labels_out[start : start + batch.shape[0]] = labels.cpu().numpy().astype(dtype)
            distortion += float(minimum.sum().item())
    return labels_out, distortion / max(data.shape[0], 1)


def optimal_bit_allocation(
    curves: Sequence[dict[int, dict]], budget_bits: int
) -> tuple[list[int], float]:
    """Exact DP allocation for a fixed partition using validation MSE."""
    states: dict[int, tuple[float, list[int]]] = {0: (0.0, [])}
    for curve in curves:
        updated: dict[int, tuple[float, list[int]]] = {}
        for used, (loss, allocation) in states.items():
            for bits, record in curve.items():
                total = used + int(bits)
                if total > budget_bits:
                    continue
                candidate = loss + float(record["diagnostics"]["validation_mse"])
                if total not in updated or candidate < updated[total][0]:
                    updated[total] = (candidate, allocation + [int(bits)])
        states = updated
    if budget_bits not in states:
        raise ValueError("no feasible bit allocation for the requested budget")
    loss, allocation = states[budget_bits]
    return allocation, float(loss)


def assemble_model(
    *,
    artifact_id: int,
    node_count: int,
    metric_id: int,
    metric_lambda: float,
    residual_mean: np.ndarray,
    forward_transform: np.ndarray,
    inverse_transform: np.ndarray,
    innovation_eigenvalues: np.ndarray,
    groups: Sequence[np.ndarray],
    bits: Sequence[int],
    curves: Sequence[dict[int, dict]],
    variant: str,
) -> JointInnovationModel:
    padded_indices, group_dims = pad_groups(groups)
    max_codewords = 1 << max(int(value) for value in bits)
    max_dim = padded_indices.shape[1]
    prototypes = np.zeros(
        (len(groups), max_codewords, max_dim), dtype=np.float32
    )
    counts = np.zeros((len(groups), max_codewords), dtype=np.int64)
    for group, bit_count in enumerate(bits):
        record = curves[group][int(bit_count)]
        codebook = np.asarray(record["centers"], dtype=np.float32)
        code_counts = np.asarray(record["counts"], dtype=np.int64)
        prototypes[group, : codebook.shape[0], : codebook.shape[1]] = codebook
        counts[group, : code_counts.size] = code_counts
    placeholder_id = 1
    model = JointInnovationModel(
        artifact_id=int(artifact_id),
        model_id=placeholder_id,
        node_count=int(node_count),
        dim=int(np.asarray(residual_mean).size),
        budget_bits=int(sum(int(value) for value in bits)),
        metric_id=int(metric_id),
        metric_lambda=float(metric_lambda),
        residual_mean=np.asarray(residual_mean, dtype=np.float32),
        forward_transform=np.asarray(forward_transform, dtype=np.float32),
        inverse_transform=np.asarray(inverse_transform, dtype=np.float32),
        innovation_eigenvalues=np.asarray(
            innovation_eigenvalues, dtype=np.float32
        ),
        group_indices=padded_indices,
        group_dims=group_dims,
        bits=np.asarray(bits, dtype=np.uint32),
        prototypes=prototypes,
        prototype_counts=counts,
        variant=str(variant),
    )
    model.model_id = compute_model_id(
        model.artifact_id,
        model.metric_id,
        model.forward_transform,
        model.group_indices,
        model.group_dims,
        model.bits,
        model.prototypes,
    )
    model.validate()
    return model


def pack_codes(codes: Sequence[np.ndarray], bits: Sequence[int]) -> np.ndarray:
    bit_counts = [int(value) for value in bits]
    budget = sum(bit_counts)
    if budget <= 0 or budget > 128:
        raise ValueError("the compact-code format supports 1..128 bits")
    if len(codes) != len(bit_counts) or not codes:
        raise ValueError("code/group count mismatch")
    rows = int(np.asarray(codes[0]).size)
    words = (budget + 63) // 64
    packed = np.zeros((rows, words), dtype="<u8")
    offset = 0
    for values, width in zip(codes, bit_counts):
        labels = np.asarray(values, dtype=np.uint64)
        if labels.shape != (rows,) or np.any(labels >= (1 << width)):
            raise ValueError("group code does not fit its allocated bits")
        if width == 0:
            continue
        word, shift = divmod(offset, 64)
        packed[:, word] |= labels << np.uint64(shift)
        if shift + width > 64:
            packed[:, word + 1] |= labels >> np.uint64(64 - shift)
        offset += width
    return packed[:, 0] if words == 1 else packed


def unpack_group(packed: np.ndarray, bits: Sequence[int], group: int) -> np.ndarray:
    bit_counts = [int(value) for value in bits]
    offset = sum(bit_counts[: int(group)])
    width = bit_counts[int(group)]
    if width == 0:
        return np.zeros(np.asarray(packed).shape[0], dtype=np.int64)
    mask = (1 << width) - 1
    values = np.asarray(packed, dtype=np.uint64)
    if values.ndim == 1:
        values = values[:, None]
    word, shift = divmod(offset, 64)
    decoded = values[:, word] >> np.uint64(shift)
    if shift + width > 64:
        decoded |= values[:, word + 1] << np.uint64(64 - shift)
    return np.asarray(decoded & np.uint64(mask), dtype=np.int64)


def initialize_side_codes(path: str, model: JointInnovationModel) -> np.memmap:
    model.validate()
    if model.budget_bits > 128:
        raise ValueError("the compact-code format supports at most 128 bits")
    words = (model.budget_bits + 63) // 64
    version = SIDE_VERSION if words == 1 else SIDE_MULTIWORD_VERSION
    header_bytes = 8 + struct.calcsize(SIDE_HEADER)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(SIDE_MAGIC)
        handle.write(
            struct.pack(
                SIDE_HEADER,
                version,
                model.node_count,
                model.dim,
                model.budget_bits,
                model.groups,
                model.artifact_id,
                model.model_id,
            )
        )
        handle.truncate(header_bytes + model.node_count * words * 8)
    shape = (model.node_count,) if words == 1 else (model.node_count, words)
    return np.memmap(
        path,
        mode="r+",
        dtype="<u8",
        offset=header_bytes,
        shape=shape,
    )


def read_side_codes(path: str, model: JointInnovationModel) -> np.memmap:
    model.validate()
    header_bytes = 8 + struct.calcsize(SIDE_HEADER)
    with open(path, "rb") as handle:
        magic = handle.read(8)
        raw = handle.read(struct.calcsize(SIDE_HEADER))
    if magic != SIDE_MAGIC or len(raw) != struct.calcsize(SIDE_HEADER):
        raise ValueError("invalid joint-innovation side-code file")
    version, nodes, dim, budget, groups, artifact_id, model_id = struct.unpack(
        SIDE_HEADER, raw
    )
    if (
        version not in (SIDE_VERSION, SIDE_MULTIWORD_VERSION)
        or nodes != model.node_count
        or dim != model.dim
        or budget != model.budget_bits
        or groups != model.groups
        or artifact_id != model.artifact_id
        or model_id != model.model_id
        or version != (SIDE_VERSION if budget <= 64 else SIDE_MULTIWORD_VERSION)
        or os.path.getsize(path)
        != header_bytes + int(nodes) * ((int(budget) + 63) // 64) * 8
    ):
        raise ValueError("joint-innovation side codes do not match the model")
    words = (int(budget) + 63) // 64
    shape = (int(nodes),) if words == 1 else (int(nodes), words)
    return np.memmap(path, mode="r", dtype="<u8", offset=header_bytes, shape=shape)


def query_coordinates(query: np.ndarray, model: JointInnovationModel) -> np.ndarray:
    value = np.asarray(query, dtype=np.float32)
    if value.shape != (model.dim,):
        raise ValueError("joint-innovation query dimension mismatch")
    return np.asarray(
        value @ np.asarray(model.inverse_transform, dtype=np.float32).T,
        dtype=np.float32,
    )


def residual_score_correction(
    packed: np.ndarray,
    query: np.ndarray,
    model: JointInnovationModel,
) -> np.ndarray:
    phi = query_coordinates(query, model)
    result = np.full(
        np.asarray(packed).shape[0],
        float(np.dot(query, model.residual_mean)),
        dtype=np.float32,
    )
    for group in range(model.groups):
        lut = model.codebook(group) @ phi[model.indices(group)]
        result += lut[unpack_group(packed, model.bits, group)]
    return result
