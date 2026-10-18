#!/usr/bin/env python3
"""Manifest and reconstruction interface for primary-agnostic JI experiments.

The refinement experiment intentionally consumes decoded primary vectors rather
than reverse engineering a primary method's private code layout.  An exporter
for each primary quantizer must materialize the exact vector whose inner
product reproduces that method's primary score (up to a query-only constant).
"""

from __future__ import annotations

import hashlib
import json
import os
import struct
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from drb_io import fbin_memmap


MANIFEST_VERSION = 1


@dataclass
class PrimaryQuantizerManifest:
    name: str
    artifact_id: int
    node_count: int
    dim: int
    base_fbin: str
    reconstruction_fbin: str
    query_fbin: str
    nominal_code_bits_per_node: int
    physical_bytes_per_node: float
    scorer: str = "inner_product_reconstruction"
    implementation: str = ""
    scorer_parity_json: str = ""
    metadata: dict | None = None

    def validate(self, check_files: bool = True) -> None:
        if (
            not self.name
            or int(self.artifact_id) <= 0
            or int(self.node_count) <= 0
            or int(self.dim) <= 0
            or int(self.nominal_code_bits_per_node) <= 0
            or float(self.physical_bytes_per_node) <= 0.0
            or self.scorer != "inner_product_reconstruction"
        ):
            raise ValueError("invalid primary-quantizer manifest")
        if check_files:
            base = fbin_memmap(self.base_fbin)
            reconstruction = fbin_memmap(self.reconstruction_fbin)
            queries = fbin_memmap(self.query_fbin)
            expected = (int(self.node_count), int(self.dim))
            if base.shape != expected or reconstruction.shape != expected:
                raise ValueError(
                    "base/reconstruction shape does not match primary manifest"
                )
            if queries.shape[1] != int(self.dim):
                raise ValueError("query dimension does not match primary manifest")
            if not self.scorer_parity_json:
                raise ValueError("primary manifest is missing scorer parity evidence")
            with open(self.scorer_parity_json, encoding="utf-8") as handle:
                parity = json.load(handle)
            if not bool(parity.get("passed", False)):
                raise ValueError("primary scorer parity audit did not pass")


def compute_artifact_id(
    name: str,
    node_count: int,
    dim: int,
    reconstruction_fbin: str,
    nominal_code_bits_per_node: int,
) -> int:
    path = os.path.abspath(reconstruction_fbin)
    stat = os.stat(path)
    digest = hashlib.sha256()
    digest.update(str(name).encode("utf-8"))
    digest.update(
        struct.pack(
            "<QQQQ",
            int(node_count),
            int(dim),
            int(nominal_code_bits_per_node),
            int(stat.st_size),
        )
    )
    # Include deterministic samples without reading a multi-GB artifact in full.
    with open(path, "rb") as handle:
        digest.update(handle.read(min(stat.st_size, 1 << 20)))
        if stat.st_size > (1 << 20):
            handle.seek(max(0, stat.st_size - (1 << 20)))
            digest.update(handle.read(1 << 20))
    value = int.from_bytes(digest.digest()[:8], "little")
    return value if value != 0 else 1


def write_manifest(path: str, manifest: PrimaryQuantizerManifest) -> None:
    manifest.validate(check_files=True)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(manifest)
    payload["version"] = MANIFEST_VERSION
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def read_manifest(path: str, check_files: bool = True) -> PrimaryQuantizerManifest:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.pop("version", 0)) != MANIFEST_VERSION:
        raise ValueError("unsupported primary-quantizer manifest version")
    manifest = PrimaryQuantizerManifest(**payload)
    manifest.validate(check_files=check_files)
    return manifest


def write_fbin(path: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values, dtype="<f4")
    if array.ndim != 2:
        raise ValueError("fbin values must be a matrix")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        np.asarray(array.shape, dtype="<u4").tofile(handle)
        array.tofile(handle)
