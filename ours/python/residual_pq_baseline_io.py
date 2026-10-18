"""Artifact I/O for the equal-budget full-dimensional residual-PQ baseline."""

from __future__ import annotations

import hashlib
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np


RESIDUAL_PQ_MODEL_VERSION = 4
RESIDUAL_PQ_CODES_MAGIC = b"RPQCOD01"
RESIDUAL_PQ_CODES_VERSION = 1
_CODES_HEADER = "<IQIIIQQ"
RESIDUAL_PQ_ONLINE_MAGIC = b"RPQONL01"
RESIDUAL_PQ_ONLINE_VERSION = 3
_ONLINE_HEADER = "<IIQIIIIIIIQQd"
_METRIC_INNER_PRODUCT = 1
_FLAG_HAS_TRANSFORM = 1
_FLAG_HAS_RESIDUAL_TRANSFORM = 2
_FLAG_ADDITIVE_QUANTIZER = 4
_FLAG_PRIMARY_CONDITIONED = 8


@dataclass
class ResidualPQModel:
    artifact_id: int
    encoding_id: int
    node_count: int
    dim: int
    groups: int
    codewords: int
    prototypes: np.ndarray
    prototype_counts: np.ndarray
    training_objective: str
    assignment_metric: str = "euclidean"
    metric_lambda: float = 0.0
    metric_artifact_id: int = 0
    metric_matrices: np.ndarray | None = None
    metric_factors: np.ndarray | None = None
    residual_transform: np.ndarray | None = None
    residual_inverse_transform: np.ndarray | None = None
    global_metric_matrix: np.ndarray | None = None
    transform_kind: str = "identity"
    quantizer_kind: str = "product"
    conditional_basis: np.ndarray | None = None
    conditional_embeddings: np.ndarray | None = None

    @property
    def group_dim(self) -> int:
        if int(self.groups) <= 0 or int(self.dim) % int(self.groups) != 0:
            raise ValueError("residual dimension must divide evenly into groups")
        return int(self.dim) // int(self.groups)

    @property
    def prototype_dim(self) -> int:
        return int(self.dim) if self.quantizer_kind == "additive" else self.group_dim

    @property
    def conditional_rank(self) -> int:
        if self.conditional_basis is None:
            return 0
        basis = np.asarray(self.conditional_basis)
        return int(basis.shape[1]) if basis.ndim == 2 else 0

    @property
    def has_primary_conditioning(self) -> bool:
        return self.conditional_rank > 0

    @property
    def bits_per_group(self) -> int:
        value = int(self.codewords)
        bits = int(round(np.log2(value)))
        if (1 << bits) != value:
            raise ValueError("residual-PQ codewords must be a power of two")
        return bits

    @property
    def side_bits_per_node(self) -> int:
        return int(self.groups) * self.bits_per_group

    @property
    def side_bytes_per_node(self) -> int:
        if self.bits_per_group != 8:
            raise ValueError("current residual-PQ side-code format requires 8-bit groups")
        return int(self.groups)

    def validate(self) -> None:
        groups = int(self.groups)
        codewords = int(self.codewords)
        if (
            int(self.artifact_id) <= 0
            or int(self.encoding_id) <= 0
            or int(self.node_count) <= 0
            or int(self.dim) <= 0
            or groups <= 0
            or int(self.dim) % groups != 0
            or codewords != 256
        ):
            raise ValueError("invalid residual-PQ model dimensions")
        if self.quantizer_kind not in {"product", "additive"}:
            raise ValueError("unsupported residual quantizer kind")
        expected_prototypes = (groups, codewords, self.prototype_dim)
        if np.asarray(self.prototypes).shape != expected_prototypes:
            raise ValueError(
                f"prototypes have shape {np.asarray(self.prototypes).shape}, "
                f"expected {expected_prototypes}"
            )
        if np.asarray(self.prototype_counts).shape != (groups, codewords):
            raise ValueError("prototype-count shape mismatch")
        if not np.all(np.isfinite(self.prototypes)):
            raise ValueError("residual-PQ prototypes contain non-finite values")
        if np.any(np.asarray(self.prototype_counts) < 0):
            raise ValueError("residual-PQ prototype counts must be nonnegative")
        if not str(self.training_objective):
            raise ValueError("training objective must be non-empty")
        if not str(self.assignment_metric) or not np.isfinite(float(self.metric_lambda)):
            raise ValueError("invalid residual-PQ assignment metric metadata")
        shape = (groups, self.group_dim, self.group_dim)
        if self.metric_matrices is None:
            self.metric_matrices = np.broadcast_to(
                np.eye(self.group_dim, dtype=np.float32), shape
            ).copy()
        if self.metric_factors is None:
            self.metric_factors = np.broadcast_to(
                np.eye(self.group_dim, dtype=np.float32), shape
            ).copy()
        matrices = np.asarray(self.metric_matrices)
        factors = np.asarray(self.metric_factors)
        if (
            matrices.shape != shape
            or factors.shape != shape
            or not np.all(np.isfinite(matrices))
            or not np.all(np.isfinite(factors))
        ):
            raise ValueError("residual-PQ metric array shape mismatch")
        for group in range(groups):
            matrix = matrices[group].astype(np.float64, copy=False)
            factor = factors[group].astype(np.float64, copy=False)
            if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-7):
                raise ValueError("residual-PQ metric matrix is not symmetric")
            if float(np.linalg.eigvalsh(matrix)[0]) <= 0.0:
                raise ValueError("residual-PQ metric matrix must be positive definite")
            if not np.allclose(
                factor @ factor.T, matrix, rtol=2e-5, atol=2e-6
            ):
                raise ValueError("residual-PQ metric factor is inconsistent")
        identity_full = np.eye(int(self.dim), dtype=np.float32)
        if self.residual_transform is None:
            self.residual_transform = identity_full.copy()
        if self.residual_inverse_transform is None:
            self.residual_inverse_transform = identity_full.copy()
        if self.global_metric_matrix is None:
            self.global_metric_matrix = identity_full.copy()
        transform = np.asarray(self.residual_transform, dtype=np.float64)
        inverse = np.asarray(self.residual_inverse_transform, dtype=np.float64)
        global_metric = np.asarray(self.global_metric_matrix, dtype=np.float64)
        full_shape = (int(self.dim), int(self.dim))
        if (
            transform.shape != full_shape
            or inverse.shape != full_shape
            or global_metric.shape != full_shape
            or not np.all(np.isfinite(transform))
            or not np.all(np.isfinite(inverse))
            or not np.all(np.isfinite(global_metric))
            or not str(self.transform_kind)
        ):
            raise ValueError("invalid residual-PQ coordinate transform")
        identity64 = np.eye(int(self.dim), dtype=np.float64)
        if not np.allclose(transform @ inverse, identity64, rtol=2e-4, atol=2e-5):
            raise ValueError("residual-PQ transforms are not inverses")
        if not np.allclose(global_metric, global_metric.T, rtol=1e-6, atol=1e-7):
            raise ValueError("global residual metric is not symmetric")
        if float(np.linalg.eigvalsh(global_metric)[0]) <= 0.0:
            raise ValueError("global residual metric must be positive definite")
        if not np.allclose(
            transform @ transform.T, global_metric, rtol=2e-4, atol=2e-5
        ):
            raise ValueError("residual transform does not realize the global metric")
        if self.conditional_basis is None:
            self.conditional_basis = np.empty((int(self.dim), 0), dtype=np.float32)
        if self.conditional_embeddings is None:
            self.conditional_embeddings = np.empty((0, 0, 0), dtype=np.float32)
        basis = np.asarray(self.conditional_basis)
        embeddings = np.asarray(self.conditional_embeddings)
        rank = self.conditional_rank
        if rank == 0:
            if basis.shape != (int(self.dim), 0) or embeddings.size != 0:
                raise ValueError("unconditioned model contains conditional parameters")
        elif (
            basis.shape != (int(self.dim), rank)
            or embeddings.ndim != 3
            or embeddings.shape[0] <= 0
            or embeddings.shape[1] <= 0
            or embeddings.shape[2] != rank
            or not np.all(np.isfinite(basis))
            or not np.all(np.isfinite(embeddings))
        ):
            raise ValueError("invalid primary-code conditional parameters")

    @property
    def uses_euclidean_assignment(self) -> bool:
        identity = np.eye(self.group_dim, dtype=np.float32)
        group_identity = bool(
            np.array_equal(
                np.asarray(self.metric_matrices),
                np.broadcast_to(
                    identity,
                    (int(self.groups), self.group_dim, self.group_dim),
                ),
            )
        )
        return bool(
            group_identity
            and np.array_equal(
                np.asarray(self.residual_transform, dtype=np.float32),
                np.eye(int(self.dim), dtype=np.float32),
            )
        )

    @property
    def has_residual_transform(self) -> bool:
        return not np.array_equal(
            np.asarray(self.residual_transform, dtype=np.float32),
            np.eye(int(self.dim), dtype=np.float32),
        )


def compute_encoding_id(
    artifact_id: int,
    dim: int,
    groups: int,
    prototypes: np.ndarray,
    metric_matrices: np.ndarray | None = None,
    residual_transform: np.ndarray | None = None,
    quantizer_kind: str = "product",
    conditional_basis: np.ndarray | None = None,
    conditional_embeddings: np.ndarray | None = None,
) -> int:
    """Bind side codes to the primary PQ artifact and residual codebook."""
    digest = hashlib.sha256()
    digest.update(struct.pack("<QII", int(artifact_id), int(dim), int(groups)))
    digest.update(np.ascontiguousarray(prototypes, dtype="<f4").tobytes())
    if metric_matrices is not None:
        matrices = np.asarray(metric_matrices, dtype=np.float32)
        identity = np.broadcast_to(
            np.eye(int(dim) // int(groups), dtype=np.float32), matrices.shape
        )
        # Identity assignment has exactly the same encoding semantics as v1.
        if not np.array_equal(matrices, identity):
            digest.update(np.ascontiguousarray(matrices, dtype="<f4").tobytes())
    if residual_transform is not None:
        transform = np.asarray(residual_transform, dtype=np.float32)
        if not np.array_equal(transform, np.eye(int(dim), dtype=np.float32)):
            digest.update(np.ascontiguousarray(transform, dtype="<f4").tobytes())
    if quantizer_kind != "product":
        digest.update(str(quantizer_kind).encode("utf-8"))
    if conditional_basis is not None and np.asarray(conditional_basis).size:
        digest.update(
            np.ascontiguousarray(conditional_basis, dtype="<f4").tobytes()
        )
    if conditional_embeddings is not None and np.asarray(conditional_embeddings).size:
        digest.update(
            np.ascontiguousarray(conditional_embeddings, dtype="<f4").tobytes()
        )
    value = int.from_bytes(digest.digest()[:8], "little")
    return value if value != 0 else 1


def clone_residual_pq_score_model(
    model: ResidualPQModel,
    prototypes: np.ndarray,
    training_objective: str,
) -> ResidualPQModel:
    """Create a prototype-only score model that reuses frozen side codes.

    ``encoding_id`` deliberately remains unchanged: it identifies the fixed
    node-to-code assignment produced by ``model``.  Pairwise refinement changes
    only the vectors read by those codes, not any node's stored side code.
    """
    result = ResidualPQModel(
        artifact_id=model.artifact_id,
        encoding_id=model.encoding_id,
        node_count=model.node_count,
        dim=model.dim,
        groups=model.groups,
        codewords=model.codewords,
        prototypes=np.asarray(prototypes, dtype=np.float32).copy(),
        prototype_counts=np.asarray(model.prototype_counts, dtype=np.int64).copy(),
        training_objective=str(training_objective),
        assignment_metric=model.assignment_metric,
        metric_lambda=model.metric_lambda,
        metric_artifact_id=model.metric_artifact_id,
        metric_matrices=np.asarray(model.metric_matrices, dtype=np.float32).copy(),
        metric_factors=np.asarray(model.metric_factors, dtype=np.float32).copy(),
        residual_transform=np.asarray(
            model.residual_transform, dtype=np.float32
        ).copy(),
        residual_inverse_transform=np.asarray(
            model.residual_inverse_transform, dtype=np.float32
        ).copy(),
        global_metric_matrix=np.asarray(
            model.global_metric_matrix, dtype=np.float32
        ).copy(),
        transform_kind=model.transform_kind,
        quantizer_kind=model.quantizer_kind,
        conditional_basis=np.asarray(
            model.conditional_basis, dtype=np.float32
        ).copy(),
        conditional_embeddings=np.asarray(
            model.conditional_embeddings, dtype=np.float32
        ).copy(),
    )
    result.validate()
    return result


def write_residual_pq_model(path: str, model: ResidualPQModel) -> None:
    model.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(RESIDUAL_PQ_MODEL_VERSION, dtype=np.uint32),
        artifact_id=np.asarray(model.artifact_id, dtype=np.uint64),
        encoding_id=np.asarray(model.encoding_id, dtype=np.uint64),
        node_count=np.asarray(model.node_count, dtype=np.uint64),
        dim=np.asarray(model.dim, dtype=np.uint32),
        groups=np.asarray(model.groups, dtype=np.uint32),
        codewords=np.asarray(model.codewords, dtype=np.uint32),
        prototypes=np.asarray(model.prototypes, dtype=np.float32),
        prototype_counts=np.asarray(model.prototype_counts, dtype=np.int64),
        training_objective=np.asarray(str(model.training_objective)),
        assignment_metric=np.asarray(str(model.assignment_metric)),
        metric_lambda=np.asarray(model.metric_lambda, dtype=np.float64),
        metric_artifact_id=np.asarray(model.metric_artifact_id, dtype=np.uint64),
        metric_matrices=np.asarray(model.metric_matrices, dtype=np.float32),
        metric_factors=np.asarray(model.metric_factors, dtype=np.float32),
        residual_transform=np.asarray(model.residual_transform, dtype=np.float32),
        residual_inverse_transform=np.asarray(
            model.residual_inverse_transform, dtype=np.float32
        ),
        global_metric_matrix=np.asarray(
            model.global_metric_matrix, dtype=np.float32
        ),
        transform_kind=np.asarray(str(model.transform_kind)),
        quantizer_kind=np.asarray(str(model.quantizer_kind)),
        conditional_basis=np.asarray(model.conditional_basis, dtype=np.float32),
        conditional_embeddings=np.asarray(
            model.conditional_embeddings, dtype=np.float32
        ),
    )


def read_residual_pq_model(path: str) -> ResidualPQModel:
    with np.load(path, allow_pickle=False) as state:
        version = int(state["version"])
        if version not in (1, 2, 3, RESIDUAL_PQ_MODEL_VERSION):
            raise ValueError(f"unsupported residual-PQ model version: {version}")
        groups = int(state["groups"])
        dim = int(state["dim"])
        group_dim = dim // groups
        identity = np.broadcast_to(
            np.eye(group_dim, dtype=np.float32),
            (groups, group_dim, group_dim),
        ).copy()
        model = ResidualPQModel(
            artifact_id=int(state["artifact_id"]),
            encoding_id=int(state["encoding_id"]),
            node_count=int(state["node_count"]),
            dim=int(state["dim"]),
            groups=int(state["groups"]),
            codewords=int(state["codewords"]),
            prototypes=np.asarray(state["prototypes"], dtype=np.float32),
            prototype_counts=np.asarray(state["prototype_counts"], dtype=np.int64),
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
            metric_matrices=(
                np.asarray(state["metric_matrices"], dtype=np.float32)
                if version >= 2
                else identity.copy()
            ),
            metric_factors=(
                np.asarray(state["metric_factors"], dtype=np.float32)
                if version >= 2
                else identity.copy()
            ),
            residual_transform=(
                np.asarray(state["residual_transform"], dtype=np.float32)
                if version >= 3
                else np.eye(dim, dtype=np.float32)
            ),
            residual_inverse_transform=(
                np.asarray(
                    state["residual_inverse_transform"], dtype=np.float32
                )
                if version >= 3
                else np.eye(dim, dtype=np.float32)
            ),
            global_metric_matrix=(
                np.asarray(state["global_metric_matrix"], dtype=np.float32)
                if version >= 3
                else np.eye(dim, dtype=np.float32)
            ),
            transform_kind=(
                str(state["transform_kind"].item())
                if version >= 3
                else "identity"
            ),
            quantizer_kind=(
                str(state["quantizer_kind"].item())
                if version >= 4
                else "product"
            ),
            conditional_basis=(
                np.asarray(state["conditional_basis"], dtype=np.float32)
                if version >= 4
                else np.empty((dim, 0), dtype=np.float32)
            ),
            conditional_embeddings=(
                np.asarray(state["conditional_embeddings"], dtype=np.float32)
                if version >= 4
                else np.empty((0, 0, 0), dtype=np.float32)
            ),
        )
    model.validate()
    return model


def initialize_residual_pq_side_codes(
    path: str, model: ResidualPQModel
) -> np.memmap:
    model.validate()
    header_size = 8 + struct.calcsize(_CODES_HEADER)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(RESIDUAL_PQ_CODES_MAGIC)
        handle.write(
            struct.pack(
                _CODES_HEADER,
                RESIDUAL_PQ_CODES_VERSION,
                model.node_count,
                model.dim,
                model.groups,
                model.codewords,
                model.artifact_id,
                model.encoding_id,
            )
        )
        handle.truncate(
            header_size + model.node_count * model.side_bytes_per_node
        )
    return np.memmap(
        path,
        mode="r+",
        dtype=np.uint8,
        offset=header_size,
        shape=(model.node_count, model.side_bytes_per_node),
    )


def read_residual_pq_side_codes(
    path: str, model: ResidualPQModel, writable: bool = False
) -> Tuple[np.memmap, int]:
    model.validate()
    header_size = 8 + struct.calcsize(_CODES_HEADER)
    with open(path, "rb") as handle:
        magic = handle.read(8)
        raw = handle.read(struct.calcsize(_CODES_HEADER))
    if magic != RESIDUAL_PQ_CODES_MAGIC or len(raw) != struct.calcsize(
        _CODES_HEADER
    ):
        raise ValueError("invalid residual-PQ side-code file")
    version, nodes, dim, groups, codewords, artifact_id, encoding_id = struct.unpack(
        _CODES_HEADER, raw
    )
    if (
        version != RESIDUAL_PQ_CODES_VERSION
        or nodes != model.node_count
        or dim != model.dim
        or groups != model.groups
        or codewords != model.codewords
        or artifact_id != model.artifact_id
        or encoding_id != model.encoding_id
    ):
        raise ValueError("residual-PQ side codes do not match the model")
    width = model.side_bytes_per_node
    expected = header_size + int(nodes) * width
    if os.path.getsize(path) != expected:
        raise ValueError("residual-PQ side-code file size mismatch")
    return (
        np.memmap(
            path,
            mode="r+" if writable else "r",
            dtype=np.uint8,
            offset=header_size,
            shape=(int(nodes), width),
        ),
        width,
    )


def write_residual_pq_online_model(path: str, pq_model, model: ResidualPQModel) -> None:
    """Export the frozen primary PQ and residual prototypes for the C++ scorer."""
    model.validate()
    pq_model.validate()
    if (
        int(model.artifact_id) != int(pq_model.artifact_id)
        or int(model.node_count) != int(pq_model.node_count)
        or int(model.dim) != int(pq_model.dim)
    ):
        raise ValueError("primary and residual-PQ models do not describe one artifact")
    flags = _FLAG_HAS_TRANSFORM if bool(pq_model.has_transform) else 0
    if model.has_residual_transform:
        flags |= _FLAG_HAS_RESIDUAL_TRANSFORM
    if model.quantizer_kind == "additive":
        flags |= _FLAG_ADDITIVE_QUANTIZER
    if model.has_primary_conditioning:
        if model.conditional_embeddings.shape[:2] != (
            int(pq_model.subspaces),
            int(pq_model.codewords),
        ):
            raise ValueError("conditional embeddings do not match the primary PQ")
        flags |= _FLAG_PRIMARY_CONDITIONED
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(RESIDUAL_PQ_ONLINE_MAGIC)
        handle.write(
            struct.pack(
                _ONLINE_HEADER,
                RESIDUAL_PQ_ONLINE_VERSION,
                _METRIC_INNER_PRODUCT,
                int(pq_model.node_count),
                int(pq_model.dim),
                int(pq_model.subspaces),
                int(pq_model.codewords),
                int(pq_model.dsub),
                int(model.groups),
                int(model.codewords),
                flags,
                int(pq_model.artifact_id),
                int(model.encoding_id),
                float(model.metric_lambda),
            )
        )
        for array in (
            pq_model.transform,
            pq_model.transform_bias,
            pq_model.centroids,
            model.prototypes,
        ):
            handle.write(np.ascontiguousarray(array, dtype="<f4").tobytes())
        if model.has_residual_transform:
            handle.write(
                np.ascontiguousarray(
                    model.residual_inverse_transform, dtype="<f4"
                ).tobytes()
            )
        if model.has_primary_conditioning:
            handle.write(struct.pack("<I", model.conditional_rank))
            handle.write(
                np.ascontiguousarray(model.conditional_basis, dtype="<f4").tobytes()
            )
            handle.write(
                np.ascontiguousarray(
                    model.conditional_embeddings, dtype="<f4"
                ).tobytes()
            )
