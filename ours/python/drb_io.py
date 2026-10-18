#!/usr/bin/env python3
"""Binary artifact and fbin helpers shared by the DRB-PQ offline tools."""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import numpy as np


MODEL_MAGIC = b"DRBPQ02\0"
CODES_MAGIC = b"DRBCOD02"
BITS_MAGIC = b"DRBBIT02"
VERSION = 2
METRIC_INNER_PRODUCT = 1
FLAG_HAS_TRANSFORM = 1


@dataclass
class DrbModel:
    artifact_id: int
    node_count: int
    dim: int
    subspaces: int
    codewords: int
    dsub: int
    refined_subspaces: int
    has_transform: bool
    transform: np.ndarray
    transform_bias: np.ndarray
    centroids: np.ndarray
    selected_subspaces: np.ndarray
    directions: np.ndarray
    thresholds: np.ndarray
    beta: np.ndarray

    def validate(self) -> None:
        n, d, m, k, ds, r = (
            int(self.node_count),
            int(self.dim),
            int(self.subspaces),
            int(self.codewords),
            int(self.dsub),
            int(self.refined_subspaces),
        )
        if int(self.artifact_id) <= 0 or n <= 0 or d <= 0 or m <= 0 or k <= 0 or ds <= 0 or m * ds != d:
            raise ValueError("invalid DRB model dimensions")
        if k > 256 or not 0 <= r <= m:
            raise ValueError("byte-code artifact requires K<=256 and 0<=R<=M")
        expected = {
            "transform": (d, d),
            "transform_bias": (d,),
            "centroids": (m, k, ds),
            "selected_subspaces": (r,),
            "directions": (r, ds),
            "thresholds": (r, k),
            "beta": (r, k, 2),
        }
        for name, shape in expected.items():
            if np.asarray(getattr(self, name)).shape != shape:
                raise ValueError(f"{name} has shape {np.asarray(getattr(self, name)).shape}, expected {shape}")
        if np.any(self.selected_subspaces < 0) or np.any(self.selected_subspaces >= m):
            raise ValueError("selected subspace out of range")


def fbin_memmap(path: str, limit: int = 0) -> np.memmap:
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
    if header.size != 2:
        raise ValueError(f"invalid fbin header: {path}")
    stored_rows, dim = int(header[0]), int(header[1])
    expected = 8 + stored_rows * dim * 4
    if os.path.getsize(path) != expected:
        raise ValueError(f"fbin size mismatch: {path}")
    rows = stored_rows if limit <= 0 else min(stored_rows, int(limit))
    return np.memmap(path, mode="r", dtype="<f4", offset=8, shape=(rows, dim))


def write_model(path: str, model: DrbModel) -> None:
    model.validate()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    flags = FLAG_HAS_TRANSFORM if model.has_transform else 0
    with open(path, "wb") as handle:
        handle.write(MODEL_MAGIC)
        handle.write(struct.pack(
            "<IIQIIIIIIQ",
            VERSION,
            METRIC_INNER_PRODUCT,
            model.node_count,
            model.dim,
            model.subspaces,
            model.codewords,
            model.dsub,
            model.refined_subspaces,
            flags,
            model.artifact_id,
        ))
        for array, dtype in (
            (model.transform, "<f4"),
            (model.transform_bias, "<f4"),
            (model.centroids, "<f4"),
            (model.selected_subspaces, "<u4"),
            (model.directions, "<f4"),
            (model.thresholds, "<f4"),
            (model.beta, "<f4"),
        ):
            handle.write(np.ascontiguousarray(array, dtype=dtype).tobytes())


def _read_exact(handle, dtype: str, count: int, name: str) -> np.ndarray:
    result = np.fromfile(handle, dtype=dtype, count=count)
    if result.size != count:
        raise ValueError(f"truncated model while reading {name}")
    return result


def read_model(path: str) -> DrbModel:
    with open(path, "rb") as handle:
        if handle.read(8) != MODEL_MAGIC:
            raise ValueError("invalid DRB model magic")
        raw = handle.read(struct.calcsize("<IIQIIIIIIQ"))
        if len(raw) != struct.calcsize("<IIQIIIIIIQ"):
            raise ValueError("truncated DRB model header")
        version, metric, n, d, m, k, ds, r, flags, artifact_id = struct.unpack("<IIQIIIIIIQ", raw)
        if version != VERSION or metric != METRIC_INNER_PRODUCT:
            raise ValueError("unsupported DRB model version or metric")
        model = DrbModel(
            artifact_id=artifact_id,
            node_count=n,
            dim=d,
            subspaces=m,
            codewords=k,
            dsub=ds,
            refined_subspaces=r,
            has_transform=bool(flags & FLAG_HAS_TRANSFORM),
            transform=_read_exact(handle, "<f4", d * d, "transform").reshape(d, d),
            transform_bias=_read_exact(handle, "<f4", d, "transform bias"),
            centroids=_read_exact(handle, "<f4", m * k * ds, "centroids").reshape(m, k, ds),
            selected_subspaces=_read_exact(handle, "<u4", r, "selected subspaces"),
            directions=_read_exact(handle, "<f4", r * ds, "directions").reshape(r, ds),
            thresholds=_read_exact(handle, "<f4", r * k, "thresholds").reshape(r, k),
            beta=_read_exact(handle, "<f4", r * k * 2, "beta").reshape(r, k, 2),
        )
        if handle.read(1):
            raise ValueError("DRB model has trailing bytes")
    model.validate()
    return model


def write_codes(path: str, codes: np.ndarray, artifact_id: int) -> None:
    codes = np.ascontiguousarray(codes, dtype=np.uint8)
    if codes.ndim != 2 or codes.shape[0] <= 0 or codes.shape[1] <= 0:
        raise ValueError("codes must be a non-empty [N,M] matrix")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(CODES_MAGIC)
        handle.write(struct.pack("<IQIQ", VERSION, codes.shape[0], codes.shape[1], artifact_id))
        handle.write(memoryview(codes))


def read_codes(path: str, mmap: bool = True, expected_artifact_id: int | None = None) -> np.ndarray:
    with open(path, "rb") as handle:
        if handle.read(8) != CODES_MAGIC:
            raise ValueError("invalid PQ-code magic")
        raw = handle.read(struct.calcsize("<IQIQ"))
        if len(raw) != struct.calcsize("<IQIQ"):
            raise ValueError("truncated PQ-code header")
        version, nodes, subspaces, artifact_id = struct.unpack("<IQIQ", raw)
    if version != VERSION:
        raise ValueError("unsupported PQ-code version")
    if expected_artifact_id is not None and artifact_id != expected_artifact_id:
        raise ValueError("PQ codes do not belong to the supplied model")
    expected = 8 + struct.calcsize("<IQIQ") + nodes * subspaces
    if os.path.getsize(path) != expected:
        raise ValueError("PQ-code file size mismatch")
    if mmap:
        return np.memmap(
            path,
            mode="r",
            dtype=np.uint8,
            offset=8 + struct.calcsize("<IQIQ"),
            shape=(nodes, subspaces),
        )
    with open(path, "rb") as handle:
        handle.seek(8 + struct.calcsize("<IQIQ"))
        return np.fromfile(handle, dtype=np.uint8, count=nodes * subspaces).reshape(nodes, subspaces)


def write_side_bits(
    path: str, packed_bits: np.ndarray, refined_subspaces: int, artifact_id: int
) -> None:
    bits = np.ascontiguousarray(packed_bits, dtype=np.uint8)
    width = (int(refined_subspaces) + 7) // 8
    if bits.ndim != 2 or bits.shape[1] != width:
        raise ValueError("packed side-bit width mismatch")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(BITS_MAGIC)
        handle.write(struct.pack("<IQIIQ", VERSION, bits.shape[0], refined_subspaces, width, artifact_id))
        handle.write(memoryview(bits))


def compute_artifact_id(
    node_count: int,
    transform: np.ndarray,
    transform_bias: np.ndarray,
    centroids: np.ndarray,
    codes: np.ndarray,
) -> int:
    """Return a stable nonzero ID binding the codebook and encoded database."""
    digest = hashlib.sha256()
    digest.update(struct.pack("<Q", int(node_count)))
    for array in (transform, transform_bias, centroids):
        digest.update(np.ascontiguousarray(array, dtype="<f4").tobytes())
    digest.update(memoryview(np.ascontiguousarray(codes, dtype=np.uint8)))
    value = int.from_bytes(digest.digest()[:8], "little")
    return value if value != 0 else 1


def transform_vectors(vectors: np.ndarray, model: DrbModel) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2 or vectors.shape[1] != model.dim:
        raise ValueError("vector/model dimension mismatch")
    if not model.has_transform:
        return np.ascontiguousarray(vectors, dtype=np.float32)
    return np.asarray(vectors @ model.transform.T + model.transform_bias, dtype=np.float32)


def save_json(path: str, value: Dict[str, object]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
