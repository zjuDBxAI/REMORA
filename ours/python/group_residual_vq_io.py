"""Artifact I/O shared by the offline group residual-VQ experiments."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


GROUP_MODEL_VERSION = 1
GROUP_CODES_MAGIC = b"GRVQCOD1"
GROUP_CODES_VERSION = 1
_GROUP_CODES_HEADER = "<IQIIQQ"


@dataclass
class GroupResidualVQModel:
    artifact_id: int
    encoding_id: int
    node_count: int
    dim: int
    subspaces: int
    dsub: int
    groups: np.ndarray
    codewords: int
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    training_objective: str
    assignment_metric: str = "euclidean"
    metric_lambda: float = 0.0
    metric_artifact_id: int = 0
    metric_matrices: np.ndarray | None = None
    metric_factors: np.ndarray | None = None

    @property
    def group_count(self) -> int:
        return int(np.asarray(self.groups).shape[0])

    @property
    def subspaces_per_group(self) -> int:
        return int(np.asarray(self.groups).shape[1])

    @property
    def group_dim(self) -> int:
        return self.subspaces_per_group * int(self.dsub)

    @property
    def bits_per_group(self) -> int:
        return int(round(np.log2(int(self.codewords))))

    @property
    def side_bits_per_node(self) -> int:
        return self.group_count * self.bits_per_group

    @property
    def side_bytes_per_node(self) -> int:
        if self.bits_per_group != 8:
            raise ValueError("current group side-code format requires 8-bit groups")
        return self.group_count

    @property
    def selected_subspaces(self) -> np.ndarray:
        return np.asarray(self.groups, dtype=np.uint32).reshape(-1)

    @property
    def resolved_metric_matrices(self) -> np.ndarray:
        if self.metric_matrices is None:
            return np.broadcast_to(
                np.eye(self.group_dim, dtype=np.float32),
                (self.group_count, self.group_dim, self.group_dim),
            ).copy()
        return np.asarray(self.metric_matrices, dtype=np.float32)

    @property
    def resolved_metric_factors(self) -> np.ndarray:
        if self.metric_factors is None:
            return np.broadcast_to(
                np.eye(self.group_dim, dtype=np.float32),
                (self.group_count, self.group_dim, self.group_dim),
            ).copy()
        return np.asarray(self.metric_factors, dtype=np.float32)

    @property
    def uses_euclidean_assignment(self) -> bool:
        identity = np.broadcast_to(
            np.eye(self.group_dim, dtype=np.float32),
            (self.group_count, self.group_dim, self.group_dim),
        )
        return bool(np.array_equal(self.resolved_metric_matrices, identity))

    def validate(self) -> None:
        groups = np.asarray(self.groups)
        codewords = int(self.codewords)
        if (
            int(self.artifact_id) <= 0
            or int(self.encoding_id) <= 0
            or int(self.node_count) <= 0
            or int(self.dim) <= 0
            or int(self.subspaces) <= 0
            or int(self.dsub) <= 0
            or int(self.subspaces) * int(self.dsub) != int(self.dim)
            or groups.ndim != 2
            or groups.shape[0] <= 0
            or groups.shape[1] <= 0
            or codewords != 256
        ):
            raise ValueError("invalid group residual-VQ model dimensions")
        selected = groups.astype(np.int64, copy=False).reshape(-1)
        if (
            np.any(selected < 0)
            or np.any(selected >= int(self.subspaces))
            or np.unique(selected).size != selected.size
        ):
            raise ValueError("group residual-VQ subspaces must be unique and in range")
        expected_prototypes = (
            groups.shape[0],
            codewords,
            groups.shape[1] * int(self.dsub),
        )
        if np.asarray(self.prototypes).shape != expected_prototypes:
            raise ValueError(
                f"prototypes have shape {np.asarray(self.prototypes).shape}, "
                f"expected {expected_prototypes}"
            )
        if np.asarray(self.prototype_counts).shape != (groups.shape[0], codewords):
            raise ValueError("prototype-count shape mismatch")
        if not np.all(np.isfinite(self.prototypes)):
            raise ValueError("group residual prototypes contain non-finite values")
        if np.any(np.asarray(self.prototype_counts) < 0):
            raise ValueError("group residual prototype counts must be nonnegative")
        if not str(self.training_objective):
            raise ValueError("training objective must be non-empty")
        expected_metric = (
            groups.shape[0],
            groups.shape[1] * int(self.dsub),
            groups.shape[1] * int(self.dsub),
        )
        matrices = self.resolved_metric_matrices
        factors = self.resolved_metric_factors
        if (
            matrices.shape != expected_metric
            or factors.shape != expected_metric
            or not np.all(np.isfinite(matrices))
            or not np.all(np.isfinite(factors))
            or not str(self.assignment_metric)
            or not np.isfinite(float(self.metric_lambda))
            or float(self.metric_lambda) < 0.0
            or int(self.metric_artifact_id) < 0
        ):
            raise ValueError("invalid group residual assignment metric")
        reconstructed = np.einsum("gde,gfe->gdf", factors, factors)
        if not np.allclose(matrices, reconstructed, rtol=2e-5, atol=2e-6):
            raise ValueError("group residual metric factors do not match matrices")


def compute_encoding_id(
    artifact_id: int,
    groups: np.ndarray,
    prototypes: np.ndarray,
    metric_matrices: np.ndarray | None = None,
) -> int:
    """Bind side codes to the PQ artifact, grouping, and assignment codebook."""
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", int(artifact_id)))
    digest.update(np.ascontiguousarray(groups, dtype="<u4").tobytes())
    digest.update(np.ascontiguousarray(prototypes, dtype="<f4").tobytes())
    if metric_matrices is not None:
        digest.update(b"metric-assignment-v1")
        digest.update(np.ascontiguousarray(metric_matrices, dtype="<f4").tobytes())
    result = int.from_bytes(digest.digest()[:8], "little")
    return result if result != 0 else 1


def compute_metric_id(
    artifact_id: int,
    groups: np.ndarray,
    metric_lambda: float,
    normalized_second_moment: np.ndarray,
) -> int:
    """Identify one selected-group OOD-query assignment geometry."""
    digest = hashlib.sha256()
    digest.update(b"selected-group-ood-query-metric-v1")
    digest.update(struct.pack("<Qd", int(artifact_id), float(metric_lambda)))
    digest.update(np.ascontiguousarray(groups, dtype="<u4").tobytes())
    digest.update(
        np.ascontiguousarray(normalized_second_moment, dtype="<f4").tobytes()
    )
    result = int.from_bytes(digest.digest()[:8], "little")
    return result if result != 0 else 1


def write_group_model(path: str, model: GroupResidualVQModel) -> None:
    model.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(GROUP_MODEL_VERSION, dtype=np.uint32),
        artifact_id=np.asarray(model.artifact_id, dtype=np.uint64),
        encoding_id=np.asarray(model.encoding_id, dtype=np.uint64),
        node_count=np.asarray(model.node_count, dtype=np.uint64),
        dim=np.asarray(model.dim, dtype=np.uint32),
        subspaces=np.asarray(model.subspaces, dtype=np.uint32),
        dsub=np.asarray(model.dsub, dtype=np.uint32),
        groups=np.asarray(model.groups, dtype=np.uint32),
        codewords=np.asarray(model.codewords, dtype=np.uint32),
        prototypes=np.asarray(model.prototypes, dtype=np.float32),
        prototype_counts=np.asarray(model.prototype_counts, dtype=np.int64),
        training_objective=np.asarray(str(model.training_objective)),
        assignment_metric=np.asarray(str(model.assignment_metric)),
        metric_lambda=np.asarray(model.metric_lambda, dtype=np.float64),
        metric_artifact_id=np.asarray(model.metric_artifact_id, dtype=np.uint64),
        metric_matrices=model.resolved_metric_matrices,
        metric_factors=model.resolved_metric_factors,
    )


def read_group_model(path: str) -> GroupResidualVQModel:
    with np.load(path, allow_pickle=False) as state:
        version = int(state["version"])
        if version != GROUP_MODEL_VERSION:
            raise ValueError(f"unsupported group residual-VQ version: {version}")
        groups = np.asarray(state["groups"], dtype=np.uint32)
        dsub = int(state["dsub"])
        group_dim = int(groups.shape[1]) * dsub
        identity = np.broadcast_to(
            np.eye(group_dim, dtype=np.float32),
            (groups.shape[0], group_dim, group_dim),
        ).copy()
        model = GroupResidualVQModel(
            artifact_id=int(state["artifact_id"]),
            encoding_id=int(state["encoding_id"]),
            node_count=int(state["node_count"]),
            dim=int(state["dim"]),
            subspaces=int(state["subspaces"]),
            dsub=dsub,
            groups=groups,
            codewords=int(state["codewords"]),
            prototypes=np.asarray(state["prototypes"], dtype=np.float32),
            prototype_counts=np.asarray(state["prototype_counts"], dtype=np.int64),
            training_objective=str(state["training_objective"].item()),
            assignment_metric=(
                str(state["assignment_metric"].item())
                if "assignment_metric" in state.files
                else "euclidean"
            ),
            metric_lambda=(
                float(state["metric_lambda"])
                if "metric_lambda" in state.files
                else 0.0
            ),
            metric_artifact_id=(
                int(state["metric_artifact_id"])
                if "metric_artifact_id" in state.files
                else 0
            ),
            metric_matrices=(
                np.asarray(state["metric_matrices"], dtype=np.float32)
                if "metric_matrices" in state.files
                else identity
            ),
            metric_factors=(
                np.asarray(state["metric_factors"], dtype=np.float32)
                if "metric_factors" in state.files
                else identity
            ),
        )
    model.validate()
    return model


def initialize_group_side_codes(
    path: str, model: GroupResidualVQModel
) -> np.memmap:
    model.validate()
    width = model.side_bytes_per_node
    header_size = 8 + struct.calcsize(_GROUP_CODES_HEADER)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(GROUP_CODES_MAGIC)
        handle.write(
            struct.pack(
                _GROUP_CODES_HEADER,
                GROUP_CODES_VERSION,
                model.node_count,
                model.group_count,
                model.codewords,
                model.artifact_id,
                model.encoding_id,
            )
        )
        handle.truncate(header_size + model.node_count * width)
    return np.memmap(
        path,
        mode="r+",
        dtype=np.uint8,
        offset=header_size,
        shape=(model.node_count, width),
    )


def read_group_side_codes(
    path: str, model: GroupResidualVQModel, writable: bool = False
) -> Tuple[np.memmap, int]:
    model.validate()
    header_size = 8 + struct.calcsize(_GROUP_CODES_HEADER)
    with open(path, "rb") as handle:
        magic = handle.read(8)
        raw = handle.read(struct.calcsize(_GROUP_CODES_HEADER))
    if magic != GROUP_CODES_MAGIC or len(raw) != struct.calcsize(_GROUP_CODES_HEADER):
        raise ValueError("invalid group residual-VQ side-code file")
    version, nodes, groups, codewords, artifact_id, encoding_id = struct.unpack(
        _GROUP_CODES_HEADER, raw
    )
    if (
        version != GROUP_CODES_VERSION
        or nodes != model.node_count
        or groups != model.group_count
        or codewords != model.codewords
        or artifact_id != model.artifact_id
        or encoding_id != model.encoding_id
    ):
        raise ValueError("group residual-VQ side codes do not match the model")
    width = model.side_bytes_per_node
    expected = header_size + model.node_count * width
    if os.path.getsize(path) != expected:
        raise ValueError("group residual-VQ side-code file size mismatch")
    return (
        np.memmap(
            path,
            mode="r+" if writable else "r",
            dtype=np.uint8,
            offset=header_size,
            shape=(model.node_count, width),
        ),
        width,
    )

