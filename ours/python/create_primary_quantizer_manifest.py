#!/usr/bin/env python3
"""Create a checked manifest for an externally decoded primary quantizer."""

from __future__ import annotations

import argparse
import json
import os

from drb_io import fbin_memmap
from primary_quantizer_manifest import (
    PrimaryQuantizerManifest,
    compute_artifact_id,
    write_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--name", required=True)
    parser.add_argument("--base-fbin", required=True)
    parser.add_argument("--reconstruction-fbin", required=True)
    parser.add_argument("--query-fbin", required=True)
    parser.add_argument("--nominal-code-bits-per-node", type=int, required=True)
    parser.add_argument("--physical-bytes-per-node", type=float, required=True)
    parser.add_argument("--implementation", default="")
    parser.add_argument("--scorer-parity-json", required=True)
    parser.add_argument("--metadata-json", default="")
    parser.add_argument("--out-manifest", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    base = fbin_memmap(args.base_fbin)
    reconstruction = fbin_memmap(args.reconstruction_fbin)
    if base.shape != reconstruction.shape:
        raise ValueError("primary reconstruction and base shapes differ")
    metadata = {}
    if args.metadata_json:
        with open(args.metadata_json, encoding="utf-8") as handle:
            metadata = json.load(handle)
    artifact_id = compute_artifact_id(
        args.name,
        base.shape[0],
        base.shape[1],
        args.reconstruction_fbin,
        args.nominal_code_bits_per_node,
    )
    manifest = PrimaryQuantizerManifest(
        name=args.name,
        artifact_id=artifact_id,
        node_count=int(base.shape[0]),
        dim=int(base.shape[1]),
        base_fbin=os.path.abspath(args.base_fbin),
        reconstruction_fbin=os.path.abspath(args.reconstruction_fbin),
        query_fbin=os.path.abspath(args.query_fbin),
        nominal_code_bits_per_node=args.nominal_code_bits_per_node,
        physical_bytes_per_node=args.physical_bytes_per_node,
        implementation=args.implementation,
        scorer_parity_json=os.path.abspath(args.scorer_parity_json),
        metadata=metadata,
    )
    write_manifest(args.out_manifest, manifest)
    print(f"wrote primary manifest: {args.out_manifest}")


if __name__ == "__main__":
    main()
