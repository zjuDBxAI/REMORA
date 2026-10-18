#!/usr/bin/env python3
"""Train code-native, rate-matched OPQ-PQ, QAQ-PQ, or LeanVec-OOD.

The common nominal primary budget is configurable and defaults to 768
coordinate-code bits per vector:

* OPQ-PQ: OPQ200->192 followed by PQ96x8;
* QAQ: the same 192-D OPQ coordinate and PQ96x8 layout, with local-shared
  sampled-softmax query-aware metrics used for assignment/codebook updates;
* LeanVec-OOD: a 192-D OOD projection followed by packed locally affine
  4-bit scalar quantization (192x4 bits, with scale/offset reported as
  physical overhead).

QAQ is a documented reproduction of the AAAI'23 algorithm rather than an
official upstream implementation (none is available in this workspace).  It
uses the paper's local-shared sampled-softmax matrices and block-diagonal PQ
optimization; this approximation is recorded in export_summary.json.
LeanVec-OOD uses the paper's eigenvector-search projection variant.  Intel's
proprietary LVQ/LeanVec runtime is not called and the artifact is labeled as a
reproduction, never as the official Intel implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import time

import faiss
import numpy as np

from drb_io import fbin_memmap


MODEL_MAGIC = b"GPRIM01\0"
CODES_MAGIC = b"GPCOD01\0"
VERSION = 1
ENCODING_CODEBOOK8 = 1
ENCODING_AFFINE8 = 2
ENCODING_AFFINE4 = 3
MODEL_HEADER = struct.Struct("<8sIIQIIIIIIQ")
CODES_HEADER = struct.Struct("<8sIIQIIQ")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--method", choices=("opq-pq", "qaq", "leanvec-ood"), required=True)
    p.add_argument("--base-fbin", required=True)
    p.add_argument("--query-fbin", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--train-size", type=int, default=100_000)
    p.add_argument("--query-start", type=int, default=0)
    p.add_argument("--query-limit", type=int, default=30_000)
    p.add_argument("--fit-query-start", type=int, default=0)
    p.add_argument("--fit-query-limit", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=20260831)
    p.add_argument("--iterations", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=65_536)
    p.add_argument("--threads", type=int, default=64)
    p.add_argument("--primary-budget-bits", type=int, default=768)
    p.add_argument("--opq-dim", type=int, default=192)
    p.add_argument("--pq-groups", type=int, default=96)
    p.add_argument("--leanvec-dim", type=int, default=192)
    p.add_argument("--leanvec-bits-per-dim", type=int, choices=(4, 8), default=4)
    p.add_argument("--qaq-clusters", type=int, default=64)
    p.add_argument("--qaq-query-samples", type=int, default=1024)
    p.add_argument("--qaq-iterations", type=int, default=3)
    p.add_argument("--qaq-fit-size", type=int, default=30_000)
    p.add_argument("--qaq-temperature", type=float, default=1.0)
    return p.parse_args()


def sample_rows(matrix: np.ndarray, count: int, seed: int) -> np.ndarray:
    if count >= len(matrix):
        return np.ascontiguousarray(matrix, dtype=np.float32)
    rng = np.random.default_rng(seed)
    ids = np.sort(rng.choice(len(matrix), size=count, replace=False))
    return np.ascontiguousarray(matrix[ids], dtype=np.float32)


def write_fbin(path: Path, rows: int, dim: int, batches) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "wb") as out:
        out.write(struct.pack("<II", rows, dim))
        written = 0
        for batch in batches:
            batch = np.ascontiguousarray(batch, dtype="<f4")
            if batch.ndim != 2 or batch.shape[1] != dim:
                raise ValueError(f"bad fbin batch shape {batch.shape}, expected (*,{dim})")
            out.write(memoryview(batch))
            written += len(batch)
        if written != rows:
            raise ValueError(f"fbin row count mismatch: {written} != {rows}")
        out.flush()
        os.fsync(out.fileno())
    os.replace(temporary, path)


def transform_batches(matrix: np.ndarray, transform: np.ndarray, batch: int):
    for start in range(0, len(matrix), batch):
        yield np.asarray(matrix[start:start + batch] @ transform.T, dtype=np.float32)


def train_opq(train: np.ndarray, groups: int, out_dim: int, iterations: int, seed: int):
    if out_dim % groups:
        raise ValueError("OPQ output dimension must be divisible by PQ groups")
    opq = faiss.OPQMatrix(train.shape[1], groups, out_dim)
    opq.niter = iterations
    opq.niter_pq = min(iterations, 10)
    opq.verbose = False
    # Faiss OPQ uses its own deterministic clustering seed internally.
    opq.train(train)
    transform = faiss.vector_to_array(opq.A).reshape(out_dim, train.shape[1]).astype(np.float32)
    projected = np.ascontiguousarray(train @ transform.T, dtype=np.float32)
    pq = faiss.ProductQuantizer(out_dim, groups, 8)
    pq.cp.niter = iterations
    pq.cp.seed = seed & 0x7FFFFFFF
    pq.train(projected)
    centroids = faiss.vector_to_array(pq.centroids).reshape(groups, 256, out_dim // groups)
    return transform, np.ascontiguousarray(centroids, dtype=np.float32), pq


def make_pq(centroids: np.ndarray) -> faiss.ProductQuantizer:
    groups, codewords, width = centroids.shape
    pq = faiss.ProductQuantizer(groups * width, groups, 8)
    faiss.copy_array_to_vector(np.ascontiguousarray(centroids).reshape(-1), pq.centroids)
    return pq


def cluster_assign(data: np.ndarray, centroids: np.ndarray) -> np.ndarray:
    index = faiss.IndexFlatL2(data.shape[1])
    index.add(np.ascontiguousarray(centroids, dtype=np.float32))
    return index.search(np.ascontiguousarray(data, dtype=np.float32), 1)[1].reshape(-1).astype(np.int32)


def qaq_metrics(cluster_centroids: np.ndarray, queries: np.ndarray, groups: int,
                temperature: float) -> np.ndarray:
    """Return local-shared block metrics [C,M,dsub,dsub]."""
    width = queries.shape[1] // groups
    scores = queries @ cluster_centroids.T
    scores /= max(temperature, 1e-6)
    scores -= scores.max(axis=0, keepdims=True)
    weights = np.exp(scores, dtype=np.float64)
    weights /= np.maximum(weights.sum(axis=0, keepdims=True), 1e-30)
    metrics = np.empty((len(cluster_centroids), groups, width, width), dtype=np.float32)
    for c in range(len(cluster_centroids)):
        w = weights[:, c].astype(np.float32)
        for g in range(groups):
            qg = queries[:, g * width:(g + 1) * width]
            metric = qg.T @ (qg * w[:, None])
            # A tiny data-independent ridge only makes Cholesky/solves stable.
            metric += np.eye(width, dtype=np.float32) * max(np.trace(metric), 1.0) * 1e-7
            metrics[c, g] = metric
    return metrics


def metric_assign(points: np.ndarray, centroids: np.ndarray, metrics: np.ndarray,
                  cluster_ids: np.ndarray) -> np.ndarray:
    labels = np.empty(len(points), dtype=np.int32)
    for c in np.unique(cluster_ids):
        ids = np.flatnonzero(cluster_ids == c)
        metric = metrics[int(c)]
        values, vectors = np.linalg.eigh(metric.astype(np.float64))
        root = (vectors * np.sqrt(np.maximum(values, 1e-12))) @ vectors.T
        transformed_points = np.ascontiguousarray(points[ids] @ root, dtype=np.float32)
        transformed_centroids = np.ascontiguousarray(centroids @ root, dtype=np.float32)
        labels[ids] = cluster_assign(transformed_points, transformed_centroids)
    return labels


def train_qaq(projected: np.ndarray, queries: np.ndarray, initial: np.ndarray,
              cluster_count: int, query_samples: int, fit_size: int,
              iterations: int, temperature: float, seed: int):
    """Local-shared, sampled-softmax, block-diagonal QAQ reproduction."""
    rng = np.random.default_rng(seed)
    fit = sample_rows(projected, min(fit_size, len(projected)), seed + 1)
    sampled_queries = sample_rows(queries, min(query_samples, len(queries)), seed + 2)
    kmeans = faiss.Kmeans(projected.shape[1], cluster_count, niter=15,
                          nredo=1, seed=seed & 0x7FFFFFFF, verbose=False)
    kmeans.train(fit)
    local_centroids = np.asarray(kmeans.centroids, dtype=np.float32)
    cluster_ids = cluster_assign(fit, local_centroids)
    metrics = qaq_metrics(local_centroids, sampled_queries, initial.shape[0], temperature)
    result = initial.copy()
    width = initial.shape[2]
    for iteration in range(iterations):
        for group in range(initial.shape[0]):
            points = fit[:, group * width:(group + 1) * width]
            labels = metric_assign(points, result[group], metrics[:, group], cluster_ids)
            updated = result[group].copy()
            for code in range(256):
                normal = np.zeros((width, width), dtype=np.float64)
                rhs = np.zeros(width, dtype=np.float64)
                selected = labels == code
                if not np.any(selected):
                    continue
                for c in np.unique(cluster_ids[selected]):
                    ids = selected & (cluster_ids == c)
                    count = int(ids.sum())
                    metric = metrics[int(c), group].astype(np.float64)
                    normal += count * metric
                    rhs += metric @ points[ids].sum(axis=0, dtype=np.float64)
                updated[code] = np.linalg.solve(normal + np.eye(width) * 1e-10, rhs)
            result[group] = updated
        print(f"[qaq] iteration {iteration + 1}/{iterations}", flush=True)
    return result, local_centroids, metrics


def encode_qaq(batch: np.ndarray, centroids: np.ndarray, local_centroids: np.ndarray,
               metrics: np.ndarray) -> np.ndarray:
    cluster_ids = cluster_assign(batch, local_centroids)
    groups, _, width = centroids.shape
    codes = np.empty((len(batch), groups), dtype=np.uint8)
    for group in range(groups):
        points = batch[:, group * width:(group + 1) * width]
        codes[:, group] = metric_assign(points, centroids[group], metrics[:, group], cluster_ids)
    return codes


def leanvec_projection(base_train: np.ndarray, query_train: np.ndarray,
                       reduced_dim: int):
    """LeanVec-OOD eigenvector-search variant (paper Algorithm 2).

    It searches beta in [0,1], forms the leading eigenspace of
    (1-beta) Kq + beta Kx, and evaluates the paper's OOD inner-product loss.
    The returned common projection corresponds to A=B=P in that variant.
    """
    kx = base_train.T @ base_train / len(base_train)
    kq = query_train.T @ query_train / len(query_train)
    kx = kx.astype(np.float64)
    kq = kq.astype(np.float64)

    def projection_and_loss(beta: float):
        values, vectors = np.linalg.eigh((1.0 - beta) * kq + beta * kx)
        projection = vectors[:, -reduced_dim:].T
        projected_q = projection @ kq @ projection.T
        projected_x = projection @ kx @ projection.T
        first = np.trace(projected_q @ projected_x)
        second = np.trace(projection @ kx @ kq @ projection.T)
        return projection, float(first - 2.0 * second)

    # Dependency-free bounded golden-section minimization.  The paper reports
    # this one-dimensional objective as smooth and empirically unimodal.
    left, right = 0.0, 1.0
    invphi = (np.sqrt(5.0) - 1.0) / 2.0
    c = right - invphi * (right - left)
    d = left + invphi * (right - left)
    pc, fc = projection_and_loss(c)
    pd, fd = projection_and_loss(d)
    for _ in range(24):
        if fc <= fd:
            right, d, pd, fd = d, c, pc, fc
            c = right - invphi * (right - left)
            pc, fc = projection_and_loss(c)
        else:
            left, c, pc, fc = c, d, pd, fd
            d = left + invphi * (right - left)
            pd, fd = projection_and_loss(d)
    beta = c if fc <= fd else d
    projection = pc if fc <= fd else pd
    loss = min(fc, fd)
    return np.ascontiguousarray(projection, dtype=np.float32), float(beta), float(loss)


def affine_encode(batch: np.ndarray, bits: int):
    levels = (1 << bits) - 1
    minimum = batch.min(axis=1).astype(np.float32)
    maximum = batch.max(axis=1).astype(np.float32)
    scale = (maximum - minimum) / float(levels)
    scale = np.maximum(scale, np.finfo(np.float32).tiny).astype(np.float32)
    codes = np.rint((batch - minimum[:, None]) / scale[:, None]).clip(0, levels).astype(np.uint8)
    reconstruction = minimum[:, None] + scale[:, None] * codes.astype(np.float32)
    return codes, minimum, scale, reconstruction


def pack_affine_codes(codes: np.ndarray, bits: int) -> np.ndarray:
    if bits == 8:
        return np.ascontiguousarray(codes, dtype=np.uint8)
    packed = np.zeros((len(codes), (codes.shape[1] + 1) // 2), dtype=np.uint8)
    packed[:, : codes[:, 0::2].shape[1]] = codes[:, 0::2]
    odd = codes[:, 1::2]
    packed[:, : odd.shape[1]] |= odd << 4
    return packed


def write_artifacts(args, transform, centroids, base, queries, qaq_state=None,
                    extra_summary=None):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scorer_dim = transform.shape[0]
    codebook = centroids is not None
    groups = int(centroids.shape[0]) if codebook else 0
    codewords = 256 if codebook else 0
    affine_bits = args.leanvec_bits_per_dim
    coordinate_bytes = scorer_dim if affine_bits == 8 else (scorer_dim + 1) // 2
    record_bytes = groups if codebook else coordinate_bytes + 8
    encoding = (ENCODING_CODEBOOK8 if codebook else
                ENCODING_AFFINE8 if affine_bits == 8 else ENCODING_AFFINE4)
    digest = hashlib.sha256()
    digest.update(args.method.encode())
    digest.update(np.ascontiguousarray(transform, dtype="<f4").tobytes())
    if codebook:
        digest.update(np.ascontiguousarray(centroids, dtype="<f4").tobytes())
    artifact_id = int.from_bytes(digest.digest()[:8], "little") or 1

    model_path = out / "primary_model.bin"
    with open(model_path, "wb") as f:
        f.write(MODEL_HEADER.pack(MODEL_MAGIC, VERSION, encoding, len(base),
                                  base.shape[1], scorer_dim, groups, codewords,
                                  record_bytes, 0, artifact_id))
        f.write(np.ascontiguousarray(transform, dtype="<f4").tobytes())
        if codebook:
            offsets = np.arange(groups + 1, dtype="<u4") * (scorer_dim // groups)
            f.write(offsets.tobytes())
            f.write(np.ascontiguousarray(centroids, dtype="<f4").tobytes())

    codes_path = out / "primary_codes.bin"
    reconstruction_path = out / "primary.reconstruction.fbin"
    transformed_base_path = out / "base.transformed.fbin"
    temporary_codes = codes_path.with_suffix(".bin.tmp")
    temporary_recon = reconstruction_path.with_suffix(".fbin.tmp")
    temporary_base = transformed_base_path.with_suffix(".fbin.tmp")
    pq = make_pq(centroids) if codebook and args.method == "opq-pq" else None
    with open(temporary_codes, "wb") as codes_out, open(temporary_recon, "wb") as recon_out, open(temporary_base, "wb") as base_out:
        codes_out.write(CODES_HEADER.pack(CODES_MAGIC, VERSION, CODES_HEADER.size,
                                          len(base), record_bytes, 0, artifact_id))
        recon_out.write(struct.pack("<II", len(base), scorer_dim))
        base_out.write(struct.pack("<II", len(base), scorer_dim))
        started = time.monotonic()
        for start in range(0, len(base), args.batch_size):
            end = min(len(base), start + args.batch_size)
            projected = np.ascontiguousarray(base[start:end] @ transform.T, dtype=np.float32)
            base_out.write(memoryview(projected))
            if args.method == "opq-pq":
                codes = np.ascontiguousarray(pq.compute_codes(projected), dtype=np.uint8)
                reconstruction = np.ascontiguousarray(pq.decode(codes), dtype=np.float32)
                codes_out.write(memoryview(codes))
            elif args.method == "qaq":
                local_centroids, metrics = qaq_state
                codes = encode_qaq(projected, centroids, local_centroids, metrics)
                reconstruction = np.empty_like(projected)
                width = scorer_dim // groups
                for g in range(groups):
                    reconstruction[:, g * width:(g + 1) * width] = centroids[g, codes[:, g]]
                codes_out.write(memoryview(codes))
            else:
                codes, minimum, scale, reconstruction = affine_encode(
                    projected, affine_bits)
                packed = pack_affine_codes(codes, affine_bits)
                records = np.empty((len(projected), record_bytes), dtype=np.uint8)
                records[:, :coordinate_bytes] = packed
                records[:, coordinate_bytes:coordinate_bytes + 4] = minimum.view(np.uint8).reshape(-1, 4)
                records[:, coordinate_bytes + 4:coordinate_bytes + 8] = scale.view(np.uint8).reshape(-1, 4)
                codes_out.write(memoryview(records))
            recon_out.write(memoryview(np.ascontiguousarray(reconstruction, dtype="<f4")))
            if end % (10 * args.batch_size) == 0 or end == len(base):
                elapsed = max(time.monotonic() - started, 1e-9)
                print(f"[{args.method}-encode] {end}/{len(base)} ({end/elapsed:.1f}/s)", flush=True)
        for handle in (codes_out, recon_out, base_out):
            handle.flush(); os.fsync(handle.fileno())
    os.replace(temporary_codes, codes_path)
    os.replace(temporary_recon, reconstruction_path)
    os.replace(temporary_base, transformed_base_path)

    selected_queries = np.ascontiguousarray(
        queries[args.query_start:args.query_start + args.query_limit] @ transform.T,
        dtype=np.float32)
    query_path = out / "query.transformed.fbin"
    write_fbin(query_path, len(selected_queries), scorer_dim, [selected_queries])

    # Deterministic scorer parity fixture.
    reconstruction = fbin_memmap(str(reconstruction_path))
    rng = np.random.default_rng(args.seed + 99)
    parity_queries = min(16, len(selected_queries))
    candidate_ids = rng.integers(0, len(base), size=(parity_queries, 256), dtype=np.uint32)
    official = np.empty(candidate_ids.shape, dtype=np.float32)
    for i in range(parity_queries):
        official[i] = reconstruction[candidate_ids[i]] @ selected_queries[i]
    with open(out / "parity.candidate_ids.ibin", "wb") as f:
        f.write(struct.pack("<II", *candidate_ids.shape)); f.write(candidate_ids.astype("<u4").tobytes())
    with open(out / "parity.official_scores.fbin", "wb") as f:
        f.write(struct.pack("<II", *official.shape)); f.write(official.astype("<f4").tobytes())

    implementation = {
        "opq-pq": f"faiss-opq{scorer_dim}-pq{groups}x8",
        "qaq": f"paper-reproduction-qaq-local-shared-block-diagonal-opq{scorer_dim}-pq{groups}x8",
        "leanvec-ood": f"paper-reproduction-leanvec-ood-eigenvector-locally-affine-u{affine_bits}",
    }[args.method]
    summary = {
        "version": 1,
        "method": args.method,
        "implementation": implementation,
        "official_upstream": args.method == "opq-pq",
        "artifact_id": artifact_id,
        "nodes": int(len(base)),
        "original_dim": int(base.shape[1]),
        "scorer_dim": int(scorer_dim),
        "nominal_code_bits_per_node": int(
            groups * 8 if codebook else scorer_dim * affine_bits),
        "physical_bytes_per_node": int(record_bytes),
        "per_vector_metadata_bytes": int(
            0 if codebook else record_bytes - coordinate_bytes),
        "model": str(model_path.resolve()),
        "codes": str(codes_path.resolve()),
        "base_fbin": str(transformed_base_path.resolve()),
        "reconstruction_fbin": str(reconstruction_path.resolve()),
        "query_fbin": str(query_path.resolve()),
        "fit_query_range": [int(args.fit_query_start),
                            int(args.fit_query_start + args.fit_query_limit)],
        "export_query_range": [int(args.query_start),
                               int(args.query_start + args.query_limit)],
        "seed": args.seed,
    }
    if args.method == "qaq":
        summary["reproduction_limits"] = (
            "Uses the paper's sampled-softmax local-shared matrices with their "
            "block-diagonal PQ components; no official QAQ code was available."
        )
    if args.method == "leanvec-ood":
        summary["reproduction_limits"] = (
            "Uses the published eigenvector-search projection and packed locally "
            f"affine {affine_bits}-bit coding; it is not Intel's proprietary "
            "LeanVec/LVQ runtime."
        )
    if extra_summary:
        summary.update(extra_summary)
    with open(out / "export_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, sort_keys=True)
    print(json.dumps(summary, indent=2, sort_keys=True))


def main() -> None:
    args = parse_args()
    faiss.omp_set_num_threads(args.threads)
    base = fbin_memmap(args.base_fbin)
    all_queries = fbin_memmap(args.query_fbin)
    if args.query_start < 0 or args.query_start + args.query_limit > len(all_queries):
        raise ValueError("export query range is outside query fbin")
    if (args.fit_query_start < 0 or args.fit_query_limit <= 0
            or args.fit_query_start + args.fit_query_limit > len(all_queries)):
        raise ValueError("fit query range is outside query fbin")
    fit_queries = np.ascontiguousarray(
        all_queries[args.fit_query_start:args.fit_query_start + args.fit_query_limit],
        dtype=np.float32)
    train = sample_rows(base, min(args.train_size, len(base)), args.seed)
    if args.method in ("opq-pq", "qaq"):
        if args.pq_groups * 8 != args.primary_budget_bits:
            raise ValueError("PQ layout does not match the primary bit budget")
        transform, initial, _ = train_opq(
            train, args.pq_groups, args.opq_dim, args.iterations, args.seed)
        if args.method == "opq-pq":
            write_artifacts(args, transform, initial, base, all_queries)
        else:
            # The common 768-bit budget requires 96 byte codebooks.  Since
            # 200 is not divisible by 96, use the same OPQ200->192 coordinate
            # as the rate-matched OPQ baseline, then replace Euclidean Lloyd
            # with QAQ's local-shared sampled-softmax metric.
            projected = np.ascontiguousarray(train @ transform.T, dtype=np.float32)
            projected_queries = np.ascontiguousarray(
                fit_queries @ transform.T, dtype=np.float32)
            centroids, local, metrics = train_qaq(
                projected, projected_queries, initial, args.qaq_clusters,
                args.qaq_query_samples, args.qaq_fit_size, args.qaq_iterations,
                args.qaq_temperature, args.seed)
            write_artifacts(args, transform, centroids, base, all_queries, (local, metrics))
    else:
        if args.leanvec_dim * args.leanvec_bits_per_dim != args.primary_budget_bits:
            raise ValueError("LeanVec coordinates do not match the primary bit budget")
        transform, beta, objective = leanvec_projection(
            train, fit_queries, args.leanvec_dim)
        write_artifacts(
            args, transform, None, base, all_queries,
            extra_summary={"leanvec_beta": beta,
                           "leanvec_projection_objective": objective})


if __name__ == "__main__":
    main()
