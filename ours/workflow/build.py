#!/usr/bin/env python3
"""Plan/check native RoarGraph construction. Never overwrite an existing graph."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import struct
import subprocess
import sys

HERE = Path(__file__).resolve().parent


def path(repo, value):
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (repo / p).resolve()


def matrix_shape(p):
    with p.open("rb") as stream:
        rows, dim = struct.unpack("<II", stream.read(8))
    if not rows or not dim or p.stat().st_size != 8 + rows * dim * 4:
        raise ValueError(f"Invalid fbin/ibin shape or size: {p}")
    return rows, dim


def commands(repo, c):
    if c["kind"] not in ("native-primary", "fp32"):
        raise ValueError("kind must be native-primary or fp32")
    for key in ("query_limit", "topk", "degree", "build_beam", "threads"):
        if type(c[key]) is not int or c[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if c["knn_geometry"] != ("primary" if c["kind"] == "native-primary" else "fp32"):
        raise ValueError("KNN geometry must match the graph's scorer; test GT is not build KNN")
    target = "roar_primary_ji_build" if c["kind"] == "native-primary" else "roar_fp_build"
    binary = path(repo, c.get("builder", f"ours/build-workflow-v1/{target}"))
    common = ["--queries", str(path(repo, c["queries"])), "--knn", str(path(repo, c["knn"])),
              "--graph", str(path(repo, c["graph"])), "--topk", str(c["topk"]),
              "--M_pjbp", str(c["degree"]), "--L_pjpq", str(c["build_beam"]),
              "--threads", str(c["threads"])]
    if c["kind"] == "fp32":
        return [], [str(binary), "--base", str(path(repo, c["base"])), *common]
    if c["primary_kind"] not in ("opq-pq", "qaq", "leanvec-ood", "extended-rabitq", "saq"):
        raise ValueError("Unsupported primary kind")
    cmd = [str(binary), "--primary-kind", c["primary_kind"],
           "--primary-model", str(path(repo, c["primary_model"])),
           "--primary-codes", str(path(repo, c["primary_codes"])), *common,
           "--summary", str(path(repo, c["summary"])), "--query-limit", str(c["query_limit"]),
           "--disable-ji", "--self-check", "512"]
    return cmd + ["--validate-only"], cmd


def validate(repo, c):
    """Check actual file headers; extension/name is not a format guarantee."""
    validation, command = commands(repo, c)
    binary = Path(command[0])
    if not binary.is_file():
        raise ValueError(f"Build target first: {binary}")
    nq, dq = matrix_shape(path(repo, c["queries"]))
    if c["query_limit"] > nq:
        raise ValueError("query_limit exceeds the build-query file")
    nk, k = matrix_shape(path(repo, c["knn"]))
    if (nk, k) != (c["query_limit"], c["topk"]):
        raise ValueError("KNN rows/Top-K do not match the requested build-query range")
    if c["kind"] == "fp32":
        nb, db = matrix_shape(path(repo, c["base"]))
        if nq != c["query_limit"] or db != dq:
            raise ValueError("FP builder uses all queries: provide a sliced query file with matching base dimension")
    else:
        model, codes = path(repo, c["primary_model"]), path(repo, c["primary_codes"])
        method = c["primary_kind"]
        with model.open("rb") as stream:
            mh = stream.read(72)
        with codes.open("rb") as stream:
            ch = stream.read(64)
        if method in ("opq-pq", "qaq", "leanvec-ood"):
            if mh[:8] != b"GPRIM01\0" or ch[:8] != b"GPCOD01\0":
                raise ValueError("This builder requires GPRIM01/GPCOD01. Imported OPQPQ/ERBQRG/SAQRG/LVOOD files are not interchangeable; reuse their graph or convert and verify parity first")
            h = struct.unpack("<8sIIQIIIIIIQ", mh[:56])
            code = struct.unpack("<8sIIQIIQ", ch[:40])
            nb, db, ds = h[3:6]
            if code[3] != nb or code[4] != h[8] or code[6] != h[10] or codes.stat().st_size != 40 + nb * h[8]:
                raise ValueError("Primary model/code identity or file size mismatch")
            if method in ("opq-pq", "qaq") and ds != db:
                raise ValueError("OPQ/QAQ main configuration must preserve raw dimension, not truncate to fit bits")
        elif method == "extended-rabitq":
            if mh[:8] != b"RBQROAR1":
                raise ValueError("Official Extended-RaBitQ builder requires RBQROAR1, not imported ERBQRG")
            h = struct.unpack("<8sIIIIQQQQQQ", mh)
            nb, db = h[5:7]
            if model != codes:
                raise ValueError("RBQROAR1 embeds codes; primary_model and primary_codes must name the same file")
        else:
            if ch[:8] != b"SAQCODE1":
                raise ValueError("Official SAQ builder requires SAQCODE1 codes, not imported SAQC002")
            h = struct.unpack("<8sIIQQQQQQ", ch)
            nb, db = h[3], dq  # SAQ model dimension is checked by native validate-only.
            if h[5] != model.stat().st_size or codes.stat().st_size != 64 + nb * h[4]:
                raise ValueError("SAQ model/code size mismatch")
        if db != dq:
            raise ValueError("Build query dimension must equal the primary's original dimension")
    if c["topk"] > nb or nb + c["query_limit"] > 2**32 - 1:
        raise ValueError("Top-K/node count exceeds builder limits")
    # Required existing KNN prevents the native builder's implicit CPU scan.
    return validation, command


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("plan", "check", "run"))
    p.add_argument("--repo", type=Path, default=HERE.parents[1])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args(argv)
    repo = args.repo.resolve(strict=True)
    c = json.loads(args.config.read_text())
    validation, command = commands(repo, c)
    if validation:
        print("Score/decoder parity: " + shlex.join(validation))
    print("Graph construction: " + shlex.join(command))
    print("Required KNN is from TRAIN/build queries, using the same primary geometry; NOT test ground truth.")
    if args.command == "plan" and not args.execute:
        return 0
    validation, command = validate(repo, c)
    if args.command == "check" and not args.execute:
        print("Headers, dimensions and KNN shape passed; native score parity is executed by run.")
        return 0
    if args.command != "run" or not args.execute:
        raise ValueError("Use run --execute to launch graph construction")
    graph = path(repo, c["graph"])
    summary = path(repo, c["summary"]) if c["kind"] == "native-primary" else None
    if graph.exists() or (summary and summary.exists()):
        raise ValueError("Refusing to overwrite an existing graph/summary. Reuse it or choose a new config/output path")
    graph.parent.mkdir(parents=True, exist_ok=True)
    if summary:
        summary.parent.mkdir(parents=True, exist_ok=True)
    if validation:
        subprocess.run(validation, cwd=repo, check=True)
    subprocess.run(command, cwd=repo, check=True)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (ValueError, OSError, KeyError, struct.error, subprocess.CalledProcessError) as exc:
        print(f"graph workflow: {exc}", file=sys.stderr)
        sys.exit(2)
