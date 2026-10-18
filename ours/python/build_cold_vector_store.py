#!/usr/bin/env python3
"""Pack an FP32 .fbin into an O_DIRECT-friendly exact-rerank store."""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
from pathlib import Path

import numpy as np


MAGIC = b"CDVEC001"
VERSION = 1
HEADER_FORMAT = "<8sIIQIIIIIQQ"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-fbin", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary")
    parser.add_argument("--block-bytes", type=int, default=4096)
    parser.add_argument("--batch-blocks", type=int, default=1024)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def align_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def main() -> None:
    args = parse_args()
    source = Path(args.input_fbin)
    output = Path(args.output)
    if not source.is_file():
        raise FileNotFoundError(source)
    if args.block_bytes < 512 or args.block_bytes & (args.block_bytes - 1):
        raise ValueError("--block-bytes must be a power of two and at least 512")
    if args.batch_blocks <= 0:
        raise ValueError("--batch-blocks must be positive")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists (use --overwrite): {output}")

    with source.open("rb") as handle:
        raw = handle.read(8)
    if len(raw) != 8:
        raise ValueError(f"truncated fbin header: {source}")
    rows, dim = struct.unpack("<II", raw)
    if rows == 0 or dim == 0:
        raise ValueError("fbin rows and dimension must be positive")
    expected = 8 + rows * dim * 4
    if source.stat().st_size != expected:
        raise ValueError(
            f"fbin size mismatch: expected {expected}, got {source.stat().st_size}"
        )

    vector_bytes = dim * 4
    block_bytes = args.block_bytes
    if vector_bytes > block_bytes:
        block_bytes = align_up(vector_bytes, args.block_bytes)
    vectors_per_block = max(1, block_bytes // vector_bytes)
    data_blocks = math.ceil(rows / vectors_per_block)
    header_bytes = block_bytes
    total_bytes = header_bytes + data_blocks * block_bytes
    vectors = np.memmap(
        source, dtype="<f4", mode="r", offset=8, shape=(rows, dim)
    )

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + f".tmp.{os.getpid()}")
    try:
        with temporary.open("wb") as handle:
            header = struct.pack(
                HEADER_FORMAT,
                MAGIC,
                VERSION,
                header_bytes,
                rows,
                dim,
                4,
                vector_bytes,
                vectors_per_block,
                block_bytes,
                header_bytes,
                data_blocks,
            )
            handle.write(header)
            handle.write(bytes(header_bytes - len(header)))

            for first_block in range(0, data_blocks, args.batch_blocks):
                block_count = min(args.batch_blocks, data_blocks - first_block)
                packed = bytearray(block_count * block_bytes)
                for local_block in range(block_count):
                    block = first_block + local_block
                    first_row = block * vectors_per_block
                    last_row = min(rows, first_row + vectors_per_block)
                    payload = vectors[first_row:last_row].tobytes(order="C")
                    offset = local_block * block_bytes
                    packed[offset : offset + len(payload)] = payload
                handle.write(packed)
                completed = min(rows, (first_block + block_count) * vectors_per_block)
                if first_block == 0 or first_block + block_count == data_blocks or (
                    first_block // args.batch_blocks
                ) % 64 == 0:
                    print(f"[cold-store] {completed}/{rows}")
            handle.flush()
            os.fsync(handle.fileno())
        if temporary.stat().st_size != total_bytes:
            raise RuntimeError("cold-vector output size mismatch")
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()

    summary = {
        "version": VERSION,
        "format": "cold_direct_fp32_block_store",
        "input_fbin": str(source.resolve()),
        "output": str(output.resolve()),
        "rows": rows,
        "dim": dim,
        "vector_bytes": vector_bytes,
        "block_bytes": block_bytes,
        "vectors_per_block": vectors_per_block,
        "data_blocks": data_blocks,
        "header_bytes": header_bytes,
        "input_bytes": expected,
        "output_bytes": total_bytes,
        "padding_overhead_fraction": total_bytes / expected - 1.0,
    }
    summary_path = Path(args.summary) if args.summary else Path(str(output) + ".json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    print(f"wrote cold direct vector store: {output}")
    print(f"wrote summary: {summary_path}")


if __name__ == "__main__":
    main()
