"""Shared helpers for quantization-native RoarGraph construction."""

from __future__ import annotations

import numpy as np

from residual_pq_baseline import (
    coded_residual_reconstruction,
    decode_coded_residual,
)


def validate_quantized_representation(pq_model, residual_model, codes, side_codes) -> None:
    """Validate that primary and residual codes describe one full-coverage index."""
    if (
        int(pq_model.refined_subspaces) != 0
        or int(residual_model.artifact_id) != int(pq_model.artifact_id)
        or int(residual_model.node_count) != int(pq_model.node_count)
        or int(residual_model.dim) != int(pq_model.dim)
        or int(residual_model.groups) * int(residual_model.group_dim)
        != int(pq_model.dim)
        or np.asarray(codes).shape
        != (int(pq_model.node_count), int(pq_model.subspaces))
        or np.asarray(side_codes).shape
        != (int(pq_model.node_count), int(residual_model.groups))
    ):
        raise ValueError("incompatible full-coverage PQ+RPQ representation")


def reconstruct_quantized_batch(
    primary_codes: np.ndarray,
    residual_codes: np.ndarray,
    pq_model,
    residual_model,
) -> np.ndarray:
    """Reconstruct ``x_hat + r_tilde`` in the primary PQ transformed space."""
    codes = np.asarray(primary_codes, dtype=np.uint8)
    side = np.asarray(residual_codes, dtype=np.uint8)
    if (
        codes.ndim != 2
        or side.ndim != 2
        or codes.shape[0] != side.shape[0]
        or codes.shape[1] != int(pq_model.subspaces)
        or side.shape[1] != int(residual_model.groups)
        or int(residual_model.dim) != int(pq_model.dim)
    ):
        raise ValueError("quantized reconstruction batch shape mismatch")
    subspaces = np.arange(int(pq_model.subspaces), dtype=np.int64)[None, :]
    primary = np.asarray(
        pq_model.centroids[subspaces, codes], dtype=np.float32
    ).reshape(codes.shape[0], int(pq_model.dim))
    coded_correction = coded_residual_reconstruction(
        side, codes, residual_model
    )
    correction = decode_coded_residual(coded_correction, residual_model)
    return np.ascontiguousarray(primary + correction, dtype=np.float32)
