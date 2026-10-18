#!/usr/bin/env python3
"""Export a variable-rate Joint-Innovation model for the C++ online scorer.

Format v1 stores every codeword decoded into the full primary scorer space.
Format v2 (the default) retains the group-local innovation codebooks and the
inverse transform.  The online scorer then computes ``phi=T^T q`` once and
builds each LUT from only the coordinates owned by that group.  Both formats
produce the same residual score; v2 avoids one dense dot product per codeword.
"""

from __future__ import annotations

import argparse
import json
import os
import struct
from pathlib import Path

import numpy as np

from joint_innovation_rpq import read_model, read_side_codes


MAGIC_V1 = b"JIONL001"
MAGIC_V2 = b"JIONL002"
HEADER = "<8sIIIIQQQQ"
GROUP = "<IIIIQQ"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--joint-model", required=True)
    parser.add_argument("--side-codes", required=True)
    parser.add_argument("--out-model", required=True)
    parser.add_argument("--out-summary", required=True)
    parser.add_argument("--format-version", type=int, choices=(1, 2), default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = read_model(args.joint_model)
    side = read_side_codes(args.side_codes, model)
    if side.shape[0] != model.node_count:
        raise ValueError("side-code row count does not match JI model")

    offsets: list[int] = []
    codebooks: list[np.ndarray] = []
    group_indices: list[np.ndarray] = []
    bit_offset = 0
    for group in range(model.groups):
        indices = model.indices(group)
        codebook = np.asarray(model.codebook(group), dtype=np.float32)
        offsets.append(bit_offset)
        codebooks.append(codebook)
        group_indices.append(np.asarray(indices, dtype=np.uint32))
        bit_offset += int(model.bits[group])
    if bit_offset != model.budget_bits:
        raise ValueError("JI group bit widths do not sum to the budget")

    total_codewords = sum(value.shape[0] for value in codebooks)
    output = Path(args.out_model)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "wb") as handle:
        magic = MAGIC_V1 if args.format_version == 1 else MAGIC_V2
        handle.write(
            struct.pack(
                HEADER,
                magic,
                args.format_version,
                model.dim,
                model.groups,
                model.budget_bits,
                model.node_count,
                model.artifact_id,
                model.model_id,
                total_codewords,
            )
        )
        handle.write(np.ascontiguousarray(model.residual_mean, dtype="<f4").tobytes())
        if args.format_version == 1:
            decoded_groups = [
                np.asarray(
                    codebook
                    @ np.asarray(model.inverse_transform[indices, :], dtype=np.float32),
                    dtype=np.float32,
                )
                for codebook, indices in zip(codebooks, group_indices)
            ]
            prototype_start = 0
            for group, decoded in enumerate(decoded_groups):
                handle.write(
                    struct.pack(
                        GROUP,
                        offsets[group],
                        int(model.bits[group]),
                        int(decoded.shape[0]),
                        0,
                        prototype_start,
                        0,
                    )
                )
                prototype_start += int(decoded.shape[0])
            for decoded in decoded_groups:
                handle.write(np.ascontiguousarray(decoded, dtype="<f4").tobytes())
        else:
            # v2 layout after the mean:
            # inverse_transform[d,d], group records, concatenated uint32 group
            # indices, concatenated group-local float32 codebooks.  In a group
            # record reserved0 is group_dim, prototype_offset is measured in
            # float values, and reserved1 is the index-array offset.
            handle.write(
                np.ascontiguousarray(model.inverse_transform, dtype="<f4").tobytes()
            )
            prototype_start = 0
            index_start = 0
            for group, codebook in enumerate(codebooks):
                width = int(codebook.shape[1])
                handle.write(
                    struct.pack(
                        GROUP,
                        offsets[group],
                        int(model.bits[group]),
                        int(codebook.shape[0]),
                        width,
                        prototype_start,
                        index_start,
                    )
                )
                prototype_start += int(codebook.size)
                index_start += width
            for indices in group_indices:
                handle.write(np.ascontiguousarray(indices, dtype="<u4").tobytes())
            for codebook in codebooks:
                handle.write(np.ascontiguousarray(codebook, dtype="<f4").tobytes())

    summary = {
        "version": 2,
        "online_format_version": args.format_version,
        "method": "joint_innovation_code_native_online_export",
        "variant": model.variant,
        "nodes": model.node_count,
        "scorer_dim": model.dim,
        "groups": model.groups,
        "bits_per_group": [int(value) for value in model.bits],
        "bit_offsets": offsets,
        "side_bits_per_node": model.budget_bits,
        "total_codewords": total_codewords,
        "active_groups": int(np.count_nonzero(np.asarray(model.bits) > 0)),
        "online_model_bytes": os.path.getsize(output),
        "prototype_layout": (
            "full-dimensional-decoded" if args.format_version == 1 else "group-local-y-space"
        ),
        "files": {
            "source_joint_model": os.path.abspath(args.joint_model),
            "side_codes": os.path.abspath(args.side_codes),
            "online_model": os.path.abspath(args.out_model),
        },
    }
    summary_path = Path(args.out_summary)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
    print(f"wrote JI online model: {output}")


if __name__ == "__main__":
    main()
