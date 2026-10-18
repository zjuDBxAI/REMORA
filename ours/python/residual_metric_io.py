"""Artifact I/O for full-coverage residual quantization metrics."""

from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np


RESIDUAL_METRIC_VERSION = 3


@dataclass
class ResidualMetric:
    artifact_id: int
    metric_id: int
    dim: int
    groups: int
    mode: str
    metric_lambda: float
    normalized_second_moment: np.ndarray
    matrices: np.ndarray
    factors: np.ndarray
    observations: int
    total_weight: float
    geometry: str = "blockwise"
    metric_form: str = "regularized"

    @property
    def group_dim(self) -> int:
        if int(self.groups) <= 0 or int(self.dim) % int(self.groups) != 0:
            raise ValueError("metric dimension must divide evenly into groups")
        return int(self.dim) // int(self.groups)

    def validate(self) -> None:
        groups = int(self.groups)
        if (
            int(self.artifact_id) <= 0
            or int(self.metric_id) <= 0
            or int(self.dim) <= 0
            or groups <= 0
            or int(self.dim) % groups != 0
            or not str(self.mode)
            or not np.isfinite(float(self.metric_lambda))
            or float(self.metric_lambda) < 0.0
            or int(self.observations) <= 0
            or not np.isfinite(float(self.total_weight))
            or float(self.total_weight) <= 0.0
        ):
            raise ValueError("invalid residual metric metadata")
        if self.geometry not in {"blockwise", "full"}:
            raise ValueError("unsupported residual metric geometry")
        if self.metric_form not in {"regularized", "pure"}:
            raise ValueError("unsupported residual metric form")
        shape = (
            (groups, self.group_dim, self.group_dim)
            if self.geometry == "blockwise"
            else (int(self.dim), int(self.dim))
        )
        for name in ("normalized_second_moment", "matrices", "factors"):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.all(np.isfinite(value)):
                raise ValueError(f"invalid residual metric array: {name}")
        matrix_count = groups if self.geometry == "blockwise" else 1
        identity = np.eye(
            self.group_dim if self.geometry == "blockwise" else int(self.dim),
            dtype=np.float64,
        )
        for group in range(matrix_count):
            matrix = np.asarray(
                self.matrices[group]
                if self.geometry == "blockwise"
                else self.matrices,
                dtype=np.float64,
            )
            factor = np.asarray(
                self.factors[group]
                if self.geometry == "blockwise"
                else self.factors,
                dtype=np.float64,
            )
            if not np.allclose(matrix, matrix.T, rtol=1e-6, atol=1e-7):
                raise ValueError("residual metric matrix is not symmetric")
            minimum = float(np.linalg.eigvalsh(matrix)[0])
            if self.metric_form == "regularized" and minimum < 1.0 - 1e-5:
                raise ValueError("regularized residual metric must satisfy A >= I")
            if self.metric_form == "pure" and minimum < -1e-5:
                raise ValueError("pure residual metric must be positive semidefinite")
            if not np.allclose(
                factor @ factor.T, matrix, rtol=2e-5, atol=2e-6
            ):
                raise ValueError("residual metric Cholesky factor is inconsistent")
            if (
                self.metric_form == "regularized"
                and self.metric_lambda == 0.0
                and not np.array_equal(matrix, identity)
            ):
                raise ValueError("lambda=0 metric must be exactly identity")


def compute_metric_id(
    artifact_id: int,
    dim: int,
    groups: int,
    mode: str,
    metric_lambda: float,
    normalized_second_moment: np.ndarray,
    geometry: str = "blockwise",
    metric_form: str = "regularized",
) -> int:
    digest = hashlib.sha256()
    digest.update(
        struct.pack(
            "<QIId",
            int(artifact_id),
            int(dim),
            int(groups),
            float(metric_lambda),
        )
    )
    digest.update(str(mode).encode("utf-8"))
    digest.update(str(geometry).encode("utf-8"))
    digest.update(str(metric_form).encode("utf-8"))
    digest.update(
        np.ascontiguousarray(normalized_second_moment, dtype="<f4").tobytes()
    )
    value = int.from_bytes(digest.digest()[:8], "little")
    return value if value != 0 else 1


def write_residual_metric(path: str, metric: ResidualMetric) -> None:
    metric.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(RESIDUAL_METRIC_VERSION, dtype=np.uint32),
        artifact_id=np.asarray(metric.artifact_id, dtype=np.uint64),
        metric_id=np.asarray(metric.metric_id, dtype=np.uint64),
        dim=np.asarray(metric.dim, dtype=np.uint32),
        groups=np.asarray(metric.groups, dtype=np.uint32),
        mode=np.asarray(str(metric.mode)),
        metric_lambda=np.asarray(metric.metric_lambda, dtype=np.float64),
        normalized_second_moment=np.asarray(
            metric.normalized_second_moment, dtype=np.float32
        ),
        matrices=np.asarray(metric.matrices, dtype=np.float32),
        factors=np.asarray(metric.factors, dtype=np.float32),
        observations=np.asarray(metric.observations, dtype=np.uint64),
        total_weight=np.asarray(metric.total_weight, dtype=np.float64),
        geometry=np.asarray(str(metric.geometry)),
        metric_form=np.asarray(str(metric.metric_form)),
    )


def read_residual_metric(path: str) -> ResidualMetric:
    with np.load(path, allow_pickle=False) as state:
        version = int(state["version"])
        if version not in (1, 2, RESIDUAL_METRIC_VERSION):
            raise ValueError(f"unsupported residual metric version: {version}")
        metric = ResidualMetric(
            artifact_id=int(state["artifact_id"]),
            metric_id=int(state["metric_id"]),
            dim=int(state["dim"]),
            groups=int(state["groups"]),
            mode=str(state["mode"].item()),
            metric_lambda=float(state["metric_lambda"]),
            normalized_second_moment=np.asarray(
                state["normalized_second_moment"], dtype=np.float32
            ),
            matrices=np.asarray(state["matrices"], dtype=np.float32),
            factors=np.asarray(state["factors"], dtype=np.float32),
            observations=int(state["observations"]),
            total_weight=float(state["total_weight"]),
            geometry=(
                str(state["geometry"].item())
                if version >= 2
                else "blockwise"
            ),
            metric_form=(
                str(state["metric_form"].item())
                if version >= 3
                else "regularized"
            ),
        )
    metric.validate()
    return metric
