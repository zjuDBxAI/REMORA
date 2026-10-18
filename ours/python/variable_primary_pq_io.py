"""Artifacts for the exact 864-bit variable-rate primary-PQ baseline.

The on-disk code record is deliberately simple and exact:

* ``M`` bytes store the low eight bits of every subspace code;
* ``ceil(R / 8)`` bytes store the ninth bit for the ``R`` selected subspaces.

For the intended M=100, R=64 experiment this is 108 bytes = 864 bits/node.
"""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


VARIABLE_PRIMARY_MODEL_VERSION = 2
VARIABLE_PRIMARY_CODES_MAGIC = b"VBPQCD01"
VARIABLE_PRIMARY_CODES_VERSION = 1
_CODES_HEADER = "<IQIIIIQQ"


@dataclass
class VariablePrimaryPQModel:
    primary_artifact_id: int
    encoding_id: int
    node_count: int
    dim: int
    subspaces: int
    dsub: int
    selected_subspaces: np.ndarray
    centroids_9bit: np.ndarray
    centroid_counts_9bit: np.ndarray
    mse_8bit: np.ndarray
    mse_9bit: np.ndarray
    utility: np.ndarray
    selection_policy: str
    training_objective: str
    assignment_metric: str = "euclidean"
    metric_lambda: float = 0.0
    metric_artifact_id: int = 0
    centroids_8bit: np.ndarray | None = None
    centroid_counts_8bit: np.ndarray | None = None
    metric_matrices: np.ndarray | None = None
    metric_factors: np.ndarray | None = None

    @property
    def refined_subspaces(self) -> int:
        return int(np.asarray(self.selected_subspaces).size)

    @property
    def code_bits_per_node(self) -> int:
        return 8 * int(self.subspaces) + self.refined_subspaces

    @property
    def code_bytes_per_node(self) -> int:
        return int(self.subspaces) + (self.refined_subspaces + 7) // 8

    @property
    def high_bit_bytes_per_node(self) -> int:
        return (self.refined_subspaces + 7) // 8

    def validate(self) -> None:
        n = int(self.node_count)
        d = int(self.dim)
        m = int(self.subspaces)
        ds = int(self.dsub)
        selected = np.asarray(self.selected_subspaces)
        if (
            int(self.primary_artifact_id) <= 0
            or int(self.encoding_id) <= 0
            or n <= 0
            or d <= 0
            or m <= 0
            or ds <= 0
            or m * ds != d
            or selected.ndim != 1
            or selected.size <= 0
            or selected.size > m
        ):
            raise ValueError("invalid variable-primary-PQ dimensions")
        selected_i = selected.astype(np.int64, copy=False)
        if (
            np.any(selected_i < 0)
            or np.any(selected_i >= m)
            or np.unique(selected_i).size != selected_i.size
        ):
            raise ValueError("selected subspaces are invalid or duplicated")
        expected_centroids = (m, 512, ds)
        if np.asarray(self.centroids_9bit).shape != expected_centroids:
            raise ValueError(
                f"9-bit centroids have shape {np.asarray(self.centroids_9bit).shape}, "
                f"expected {expected_centroids}"
            )
        if np.asarray(self.centroid_counts_9bit).shape != (m, 512):
            raise ValueError("9-bit centroid-count shape mismatch")
        for name in ("mse_8bit", "mse_9bit", "utility"):
            value = np.asarray(getattr(self, name))
            if value.shape != (m,) or not np.all(np.isfinite(value)):
                raise ValueError(f"{name} has invalid shape or values")
        if (
            not np.all(np.isfinite(self.centroids_9bit))
            or np.any(np.asarray(self.centroid_counts_9bit) < 0)
            or not str(self.selection_policy)
            or not str(self.training_objective)
            or not str(self.assignment_metric)
            or not np.isfinite(float(self.metric_lambda))
            or float(self.metric_lambda) < 0.0
        ):
            raise ValueError("invalid variable-primary-PQ model values")
        if self.centroids_8bit is not None:
            if (
                np.asarray(self.centroids_8bit).shape != (m, 256, ds)
                or not np.all(np.isfinite(self.centroids_8bit))
            ):
                raise ValueError("embedded 8-bit centroid shape mismatch")
        if self.centroid_counts_8bit is not None:
            if (
                np.asarray(self.centroid_counts_8bit).shape != (m, 256)
                or np.any(np.asarray(self.centroid_counts_8bit) < 0)
            ):
                raise ValueError("embedded 8-bit centroid-count shape mismatch")
        if (self.centroids_8bit is None) != (self.centroid_counts_8bit is None):
            raise ValueError("8-bit centroids and counts must be supplied together")
        metric_shape = (m, ds, ds)
        if self.metric_matrices is None:
            self.metric_matrices = np.broadcast_to(
                np.eye(ds, dtype=np.float32), metric_shape
            ).copy()
        if self.metric_factors is None:
            self.metric_factors = np.broadcast_to(
                np.eye(ds, dtype=np.float32), metric_shape
            ).copy()
        if (
            np.asarray(self.metric_matrices).shape != metric_shape
            or np.asarray(self.metric_factors).shape != metric_shape
            or not np.all(np.isfinite(self.metric_matrices))
            or not np.all(np.isfinite(self.metric_factors))
        ):
            raise ValueError("variable-primary-PQ metric shape mismatch")
        for subspace in range(m):
            matrix = np.asarray(self.metric_matrices[subspace], dtype=np.float64)
            factor = np.asarray(self.metric_factors[subspace], dtype=np.float64)
            if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-7):
                raise ValueError("assignment metric is not symmetric")
            if float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
                raise ValueError("assignment metric must be positive definite")
            if not np.allclose(factor @ factor.T, matrix, rtol=2e-5, atol=2e-6):
                raise ValueError("assignment metric factor is inconsistent")

    @property
    def has_embedded_8bit_codebooks(self) -> bool:
        return self.centroids_8bit is not None

    @property
    def uses_euclidean_assignment(self) -> bool:
        identity = np.broadcast_to(
            np.eye(self.dsub, dtype=np.float32),
            (self.subspaces, self.dsub, self.dsub),
        )
        return bool(np.array_equal(np.asarray(self.metric_matrices), identity))


def compute_encoding_id(
    primary_artifact_id: int,
    selected_subspaces: np.ndarray,
    centroids_9bit: np.ndarray,
    centroids_8bit: np.ndarray | None = None,
    metric_matrices: np.ndarray | None = None,
) -> int:
    selected = np.ascontiguousarray(selected_subspaces, dtype="<u4")
    centers = np.asarray(centroids_9bit, dtype=np.float32)
    digest = hashlib.sha256()
    digest.update(struct.pack("<QI", int(primary_artifact_id), int(selected.size)))
    digest.update(selected.tobytes())
    digest.update(
        np.ascontiguousarray(centers[selected.astype(np.int64)], dtype="<f4").tobytes()
    )
    if centroids_8bit is not None:
        all_subspaces = np.arange(centers.shape[0], dtype=np.int64)
        mask = np.ones(centers.shape[0], dtype=bool)
        mask[selected.astype(np.int64)] = False
        digest.update(
            np.ascontiguousarray(
                np.asarray(centroids_8bit, dtype=np.float32)[all_subspaces[mask]],
                dtype="<f4",
            ).tobytes()
        )
    if metric_matrices is not None:
        digest.update(
            np.ascontiguousarray(metric_matrices, dtype="<f4").tobytes()
        )
    value = int.from_bytes(digest.digest()[:8], "little")
    return value if value != 0 else 1


def write_variable_primary_model(path: str, model: VariablePrimaryPQModel) -> None:
    model.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(VARIABLE_PRIMARY_MODEL_VERSION, dtype=np.uint32),
        primary_artifact_id=np.asarray(model.primary_artifact_id, dtype=np.uint64),
        encoding_id=np.asarray(model.encoding_id, dtype=np.uint64),
        node_count=np.asarray(model.node_count, dtype=np.uint64),
        dim=np.asarray(model.dim, dtype=np.uint32),
        subspaces=np.asarray(model.subspaces, dtype=np.uint32),
        dsub=np.asarray(model.dsub, dtype=np.uint32),
        selected_subspaces=np.asarray(model.selected_subspaces, dtype=np.uint32),
        centroids_9bit=np.asarray(model.centroids_9bit, dtype=np.float32),
        centroid_counts_9bit=np.asarray(model.centroid_counts_9bit, dtype=np.int64),
        mse_8bit=np.asarray(model.mse_8bit, dtype=np.float64),
        mse_9bit=np.asarray(model.mse_9bit, dtype=np.float64),
        utility=np.asarray(model.utility, dtype=np.float64),
        selection_policy=np.asarray(str(model.selection_policy)),
        training_objective=np.asarray(str(model.training_objective)),
        assignment_metric=np.asarray(str(model.assignment_metric)),
        metric_lambda=np.asarray(model.metric_lambda, dtype=np.float64),
        metric_artifact_id=np.asarray(model.metric_artifact_id, dtype=np.uint64),
        centroids_8bit=(
            np.asarray(model.centroids_8bit, dtype=np.float32)
            if model.centroids_8bit is not None
            else np.empty((0,), dtype=np.float32)
        ),
        centroid_counts_8bit=(
            np.asarray(model.centroid_counts_8bit, dtype=np.int64)
            if model.centroid_counts_8bit is not None
            else np.empty((0,), dtype=np.int64)
        ),
        metric_matrices=np.asarray(model.metric_matrices, dtype=np.float32),
        metric_factors=np.asarray(model.metric_factors, dtype=np.float32),
    )


def read_variable_primary_model(path: str) -> VariablePrimaryPQModel:
    with np.load(path, allow_pickle=False) as state:
        version = int(state["version"])
        if version not in (1, VARIABLE_PRIMARY_MODEL_VERSION):
            raise ValueError(f"unsupported variable-primary-PQ model version: {version}")
        model = VariablePrimaryPQModel(
            primary_artifact_id=int(state["primary_artifact_id"]),
            encoding_id=int(state["encoding_id"]),
            node_count=int(state["node_count"]),
            dim=int(state["dim"]),
            subspaces=int(state["subspaces"]),
            dsub=int(state["dsub"]),
            selected_subspaces=np.asarray(state["selected_subspaces"], dtype=np.uint32),
            centroids_9bit=np.asarray(state["centroids_9bit"], dtype=np.float32),
            centroid_counts_9bit=np.asarray(
                state["centroid_counts_9bit"], dtype=np.int64
            ),
            mse_8bit=np.asarray(state["mse_8bit"], dtype=np.float64),
            mse_9bit=np.asarray(state["mse_9bit"], dtype=np.float64),
            utility=np.asarray(state["utility"], dtype=np.float64),
            selection_policy=str(state["selection_policy"].item()),
            training_objective=str(state["training_objective"].item()),
            assignment_metric=(
                str(state["assignment_metric"].item())
                if version >= 2
                else "euclidean"
            ),
            metric_lambda=(float(state["metric_lambda"]) if version >= 2 else 0.0),
            metric_artifact_id=(
                int(state["metric_artifact_id"]) if version >= 2 else 0
            ),
            centroids_8bit=(
                np.asarray(state["centroids_8bit"], dtype=np.float32)
                if version >= 2 and state["centroids_8bit"].size
                else None
            ),
            centroid_counts_8bit=(
                np.asarray(state["centroid_counts_8bit"], dtype=np.int64)
                if version >= 2 and state["centroid_counts_8bit"].size
                else None
            ),
            metric_matrices=(
                np.asarray(state["metric_matrices"], dtype=np.float32)
                if version >= 2
                else None
            ),
            metric_factors=(
                np.asarray(state["metric_factors"], dtype=np.float32)
                if version >= 2
                else None
            ),
        )
    model.validate()
    return model


def initialize_variable_primary_codes(
    path: str, model: VariablePrimaryPQModel
) -> np.memmap:
    model.validate()
    header_size = 8 + struct.calcsize(_CODES_HEADER)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(VARIABLE_PRIMARY_CODES_MAGIC)
        handle.write(
            struct.pack(
                _CODES_HEADER,
                VARIABLE_PRIMARY_CODES_VERSION,
                model.node_count,
                model.dim,
                model.subspaces,
                model.refined_subspaces,
                model.code_bytes_per_node,
                model.primary_artifact_id,
                model.encoding_id,
            )
        )
        handle.truncate(
            header_size + model.node_count * model.code_bytes_per_node
        )
    return np.memmap(
        path,
        mode="r+",
        dtype=np.uint8,
        offset=header_size,
        shape=(model.node_count, model.code_bytes_per_node),
    )


def read_variable_primary_codes(
    path: str, model: VariablePrimaryPQModel, writable: bool = False
) -> Tuple[np.memmap, int]:
    model.validate()
    header_size = 8 + struct.calcsize(_CODES_HEADER)
    with open(path, "rb") as handle:
        magic = handle.read(8)
        raw = handle.read(struct.calcsize(_CODES_HEADER))
    if magic != VARIABLE_PRIMARY_CODES_MAGIC or len(raw) != struct.calcsize(
        _CODES_HEADER
    ):
        raise ValueError("invalid variable-primary-PQ code file")
    (
        version,
        nodes,
        dim,
        subspaces,
        refined,
        width,
        primary_artifact_id,
        encoding_id,
    ) = struct.unpack(_CODES_HEADER, raw)
    if (
        version != VARIABLE_PRIMARY_CODES_VERSION
        or nodes != model.node_count
        or dim != model.dim
        or subspaces != model.subspaces
        or refined != model.refined_subspaces
        or width != model.code_bytes_per_node
        or primary_artifact_id != model.primary_artifact_id
        or encoding_id != model.encoding_id
    ):
        raise ValueError("variable-primary-PQ codes do not match the model")
    expected = header_size + int(nodes) * int(width)
    if os.path.getsize(path) != expected:
        raise ValueError("variable-primary-PQ code file size mismatch")
    return (
        np.memmap(
            path,
            mode="r+" if writable else "r",
            dtype=np.uint8,
            offset=header_size,
            shape=(int(nodes), int(width)),
        ),
        int(width),
    )


def pack_ninth_bits(codes_9bit: np.ndarray) -> np.ndarray:
    codes = np.asarray(codes_9bit)
    if codes.ndim != 2 or np.any(codes < 0) or np.any(codes >= 512):
        raise ValueError("9-bit codes must have shape [rows, refined] and lie in [0,512)")
    high = ((codes.astype(np.uint16, copy=False) >> 8) & 1).astype(np.uint8)
    return np.packbits(high, axis=1, bitorder="little")


def unpack_variable_primary_codes(
    records: np.ndarray, model: VariablePrimaryPQModel
) -> Tuple[np.ndarray, np.ndarray]:
    values = np.asarray(records, dtype=np.uint8)
    if values.ndim != 2 or values.shape[1] != model.code_bytes_per_node:
        raise ValueError("variable-primary-PQ record shape mismatch")
    low = values[:, : model.subspaces]
    high_packed = values[:, model.subspaces :]
    high = np.unpackbits(
        high_packed, axis=1, count=model.refined_subspaces, bitorder="little"
    ).astype(np.uint16)
    selected = model.selected_subspaces.astype(np.int64, copy=False)
    codes_9bit = low[:, selected].astype(np.uint16) | (high << 8)
    return low, codes_9bit
