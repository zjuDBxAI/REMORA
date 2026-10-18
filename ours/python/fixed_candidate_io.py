"""Shared I/O and metrics for fixed-candidate ANN audits."""

from __future__ import annotations

import os
import struct
from pathlib import Path
from typing import Tuple

import numpy as np


VISITED_MAGIC = b"DRBVST01"
VISITED_VERSION = 1


def read_id_matrix(
    path: str,
    allow_trailing: bool = False,
    row_start: int = 0,
    row_count: int | None = None,
) -> np.ndarray:
    with open(path, "rb") as handle:
        header = np.fromfile(handle, dtype="<u4", count=2)
        if header.size != 2:
            raise ValueError(f"invalid matrix header: {path}")
        rows, cols = int(header[0]), int(header[1])
    required_size = 8 + rows * cols * np.dtype("<u4").itemsize
    actual_size = os.path.getsize(path)
    if actual_size < required_size:
        raise ValueError(f"truncated ID matrix: {path}")
    if not allow_trailing and actual_size != required_size:
        raise ValueError(f"unexpected trailing bytes: {path}")
    if row_start < 0 or row_start > rows:
        raise ValueError(f"row_start is outside ID matrix: {path}")
    selected_rows = rows - row_start if row_count is None else int(row_count)
    if selected_rows < 0 or row_start + selected_rows > rows:
        raise ValueError(f"requested rows exceed ID matrix: {path}")
    offset = 8 + row_start * cols * np.dtype("<u4").itemsize
    return np.memmap(
        path,
        mode="r",
        dtype="<u4",
        offset=offset,
        shape=(selected_rows, cols),
    )


def read_visited_csr(path: str) -> Tuple[int, np.ndarray, np.ndarray]:
    header_format = "<8sIQIQ"
    header_size = struct.calcsize(header_format)
    with open(path, "rb") as handle:
        raw = handle.read(header_size)
    if len(raw) != header_size:
        raise ValueError(f"truncated visited-candidate header: {path}")
    magic, version, query_start, rows, total = struct.unpack(header_format, raw)
    if magic != VISITED_MAGIC or version != VISITED_VERSION:
        raise ValueError(f"invalid visited-candidate format: {path}")
    offsets_bytes = (int(rows) + 1) * np.dtype("<u8").itemsize
    expected = header_size + offsets_bytes + int(total) * np.dtype("<u4").itemsize
    if os.path.getsize(path) != expected:
        raise ValueError(f"visited-candidate file size mismatch: {path}")
    offsets = np.memmap(
        path, mode="r", dtype="<u8", offset=header_size, shape=(int(rows) + 1,)
    )
    ids = np.memmap(
        path,
        mode="r",
        dtype="<u4",
        offset=header_size + offsets_bytes,
        shape=(int(total),),
    )
    if int(offsets[0]) != 0 or int(offsets[-1]) != int(total):
        raise ValueError("invalid visited-candidate offsets")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("visited-candidate offsets are not monotonic")
    return int(query_start), offsets, ids


def write_id_matrix(path: str, values: np.ndarray) -> None:
    payload = np.ascontiguousarray(values, dtype="<u4")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(struct.pack("<II", payload.shape[0], payload.shape[1]))
        handle.write(memoryview(payload))


def write_float_matrix(path: str, values: np.ndarray) -> None:
    payload = np.ascontiguousarray(values, dtype="<f4")
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as handle:
        handle.write(np.asarray(payload.shape, dtype="<u4").tobytes())
        handle.write(memoryview(payload))


def exact_linear_score_parameters(
    query_t: np.ndarray, pq_model
) -> tuple[np.ndarray, float]:
    if pq_model.has_transform:
        return (
            np.asarray(query_t @ pq_model.transform, dtype=np.float32),
            float(query_t @ pq_model.transform_bias),
        )
    return np.asarray(query_t, dtype=np.float32), 0.0


def selected_linear_score_parameters(
    query_t: np.ndarray, selected: np.ndarray, pq_model
) -> tuple[np.ndarray, float]:
    masked = np.zeros(pq_model.dim, dtype=np.float32)
    blocks = masked.reshape(pq_model.subspaces, pq_model.dsub)
    source = query_t.reshape(pq_model.subspaces, pq_model.dsub)
    blocks[selected] = source[selected]
    return exact_linear_score_parameters(masked, pq_model)


def overlap_with_reference(
    values: np.ndarray, reference: np.ndarray, topk: int
) -> np.ndarray:
    return np.asarray(
        [
            len(set(row).intersection(set(other))) / float(topk)
            for row, other in zip(values, reference)
        ],
        dtype=np.float64,
    )


def row_recall(
    ids: np.ndarray, truth: np.ndarray, topk: int | None = None
) -> np.ndarray:
    width = ids.shape[1] if topk is None else int(topk)
    return np.asarray(
        [
            len(set(row[:width]).intersection(set(gt[:width]))) / float(width)
            for row, gt in zip(ids, truth)
        ],
        dtype=np.float64,
    )
