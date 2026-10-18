"""Verify the custom WordVector ID/raw-IP package and configure experiment 1.

The public source originally contains 1M base rows. This content-disjoint ID
split has 899,000 database rows; neither its vectors nor frozen primary graphs
are normalized, re-encoded, or rebuilt here. Source-server paths are provenance
only. Only experiment inputs and REMORA outputs are written locally.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'ours/test/t2i-10M'), str(ROOT / 'ours/python')]
from protocol import content_sha256, file_id, matrix, write_matrix, load_config
from pipeline import save_json
from native100_primary import NativePrimary

METHODS = ['opq-pq', 'qaq', 'leanvec-ood', 'extended-rabitq', 'saq']
DATASET = 'WordVector-ID-custom-raw-IP'


def fixed_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing configuration changed; use a new WORDVECTOR_TAG: {path}')
    else:
        save_json(path, value)


def fixed_matrix(path, values):
    path = Path(path)
    if path.exists():
        if not np.array_equal(matrix(path), values):
            raise ValueError(f'Existing matrix content changed: {path}')
    else:
        write_matrix(path, values)


def row_hashes(values):
    return {hashlib.sha256(np.ascontiguousarray(row, dtype='<f4').tobytes()).digest()
            for row in values}


def check_split_ids(base, build, metric, validation, test, fit):
    roles = {'base': base, 'build': build, 'validation': validation, 'test': test}
    for name, values in {**roles, 'metric': metric, 'fit': fit}.items():
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer):
            raise ValueError(f'Invalid source-ID array: {name}')
        if len(np.unique(values)) != len(values):
            raise ValueError(f'Duplicate source IDs: {name}')
    names = list(roles)
    for a, name in enumerate(names):
        for other in names[a + 1:]:
            if len(np.intersect1d(roles[name], roles[other])):
                raise ValueError(f'Source IDs overlap: {name}/{other}')
    if not np.isin(metric, build).all() or not np.isin(fit, base).all():
        raise ValueError('Metric/quantizer fit rows are not subsets of their authorized roles')
    reserved = np.flatnonzero(~np.isin(build, metric))[:1024]
    if len(reserved) != 1024:
        raise ValueError('Insufficient independent compatibility rows')
    return reserved


def check_truth_ids(ids, queries, nodes):
    if ids.shape != (queries, 100) or np.any(ids >= nodes):
        raise ValueError('Invalid full-database WordVector Top-100 ground truth')
    if np.any(np.diff(np.sort(np.asarray(ids), axis=1), axis=1) == 0):
        raise ValueError('Duplicate ground-truth node IDs')


def pack_truth(base, test, ids_path, output, canonicalize_ties=False):
    """Recover raw-IP scores; optionally order numerical ties consistently.

Canonicalization is allowed only for roundoff-scale discrepancies and must
leave EVERY query's original Top-10 set unchanged. The source file is read-only.
"""
    ids = matrix(ids_path, '<u4')
    check_truth_ids(ids, len(test), len(base))
    scores = np.empty(ids.shape, dtype='<f4')
    for q in range(len(test)):
        scores[q] = np.asarray(base[ids[q]], np.float64) @ np.asarray(test[q], np.float64)
    if not np.isfinite(scores).all():
        raise ValueError('Nonfinite raw-IP ground-truth scores')
    if canonicalize_ties:
        tolerance = 2e-6 * np.maximum(1., np.maximum(abs(scores[:, :-1]), abs(scores[:, 1:])))
        if np.any(np.diff(scores, axis=1) > tolerance):
            raise ValueError('Ground truth has nonnumerical ordering errors')
        order = np.lexsort((np.asarray(ids), -scores), axis=1)
        canonical_ids = np.take_along_axis(ids, order, axis=1)
        if not np.array_equal(np.sort(canonical_ids[:, :10], axis=1), np.sort(ids[:, :10], axis=1)):
            raise ValueError('Numerical canonicalization would change source Top-10 truth')
        ids = canonical_ids
        scores = np.take_along_axis(scores, order, axis=1)
    if np.any(np.diff(scores, axis=1) > 1e-6):
        raise ValueError('Ground-truth IDs are not ordered by raw inner product')
    expected = 8 + 8 * ids.size
    if output.exists():
        with output.open('rb') as handle:
            shape = struct.unpack('<II', handle.read(8))
        if shape != ids.shape or output.stat().st_size != expected:
            raise ValueError('Existing packed ground-truth header/size changed')
        old_ids = np.memmap(output, mode='r', dtype='<u4', offset=8, shape=shape)
        old_scores = np.memmap(output, mode='r', dtype='<f4', offset=8 + 4*ids.size, shape=shape)
        if not np.array_equal(old_ids, ids) or not np.array_equal(old_scores, scores):
            raise ValueError('Existing packed ground-truth contents changed')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        with temporary.open('wb') as handle:
            handle.write(struct.pack('<II', *ids.shape))
            handle.write(np.asarray(ids, dtype='<u4').tobytes())
            handle.write(scores.tobytes())
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(output)
    return ids, scores


def audit_truth_gpu(base, test, ids, gpu):
    """Independent exact full-database search on 16 held-out test queries.

Near-tied alternative IDs are allowed only if direct FP64 inner products
confirm the entire discrepancy is at the numerical boundary. This audit never
changes ground truth and its test queries are not used by the refiner trainer.
"""
    import torch
    import faiss
    torch.cuda.set_device(gpu)
    torch.backends.cuda.matmul.allow_tf32 = False
    resource = faiss.StandardGpuResources()
    resource.setTempMemory(128 << 20)
    options = faiss.GpuClonerOptions()
    options.useFloat16 = False
    index = faiss.index_cpu_to_gpu(resource, gpu, faiss.IndexFlatIP(base.shape[1]), options)
    index.add(np.ascontiguousarray(base, dtype=np.float32))
    selected = np.unique(np.linspace(0, len(test)-1, min(16, len(test))).astype(int))
    _, reference = index.search(np.ascontiguousarray(test[selected]), 100)
    exact, boundary_ties = {10: 0, 100: 0}, {10: 0, 100: 0}
    for row, q in enumerate(selected):
        for depth in (10, 100):
            supplied, expected = ids[q, :depth], reference[row, :depth]
            missing, extra = np.setdiff1d(expected, supplied), np.setdiff1d(supplied, expected)
            if not len(missing):
                exact[depth] += 1
                continue
            sm = np.asarray(base[missing], np.float64) @ np.asarray(test[q], np.float64)
            se = np.asarray(base[extra], np.float64) @ np.asarray(test[q], np.float64)
            tolerance = 2e-6 * max(1., float(np.max(abs(np.r_[sm, se]))))
            if float(sm.max() - se.min()) > tolerance:
                raise ValueError(f'Ground truth fails exact full-database GPU audit: query={q}, K={depth}')
            boundary_ties[depth] += 1
    return dict(queries=len(selected), full_database_rows=len(base),
                backend='Faiss GPU IndexFlatIP FP32; numerical discrepancies checked in FP64',
                gpu=gpu, exact_set_matches=exact, numerical_boundary_alternatives=boundary_ties,
                source_file_modified=False, test_audit_not_used_for_training=True)


def prepare(package, tag, bits=96, queries=10000, gpu=2, repeats=3):
    package = Path(package).resolve(strict=True)
    output = ROOT / 'outputs/wordvector-id' / tag
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((package/'DATASET_INDEX.json').read_text())
    verified = json.loads((package/'TRANSFER_VERIFIED.json').read_text())
    manifest_path = package/'TRANSFER_MANIFEST.json'
    manifest = json.loads(manifest_path.read_text())
    prepared = json.loads((package/'data/prepared.json').read_text())
    if (meta['dataset'] != DATASET or meta['dimension'] != 300
            or meta['metric'] != 'maximum unnormalized inner product'
            or meta['counts'] != {'base': 899000, 'query.build': 89900,
                                  'query.test': 10000, 'query.validation': 1024}
            or not verified['complete'] or not verified['all_sha256_verified']
            or not verified['shapes_verified'] or prepared['per_vector_normalization']
            or prepared['global_scaling'] or not prepared['all_primary_roles_content_disjoint']
            or prepared['official_split_reused'] or prepared['counts'] != meta['counts']
            or set(meta['graphs']) != set(METHODS)):
        raise ValueError('Invalid custom WordVector ID/raw-IP transfer metadata')
    records = {r['path']: r for r in manifest['files']}
    # The transfer verifier counts TRANSFER_MANIFEST.json itself in payload_files;
    # the manifest cannot contain its own hash, and lists only the other files.
    actual_files = {str(p.relative_to(package)) for p in package.rglob('*') if p.is_file()}
    expected_files = set(records) | {'TRANSFER_MANIFEST.json', 'TRANSFER_VERIFIED.json'}
    if (len(records) != len(manifest['files']) or len(records) != manifest['count']
            or verified['payload_files'] != len(records) + 1 or actual_files != expected_files):
        raise ValueError('Transfer inventory count mismatch')
    if sum(r['bytes'] for r in records.values()) != verified['manifest_payload_bytes']:
        raise ValueError('Transfer inventory byte count mismatch')
    for number, (name, record) in enumerate(records.items(), 1):
        path = (package/name).resolve(strict=True)
        if (not path.is_relative_to(package) or path.stat().st_size != record['bytes']
                or content_sha256(path) != record['sha256']):
            raise ValueError(f'Transfer checksum/size changed: {name}')
        if number % 25 == 0 or number == len(records):
            print(f'[transfer verified] {number}/{len(records)} SHA256 payloads', flush=True)

    base, build, metric, valid, test = [matrix(package/f'data/{name}.fbin') for name in
                                       ('base', 'query.build', 'query.metric', 'query.validation', 'query.test')]
    if [a.shape for a in (base, build, metric, valid, test)] != [
            (899000, 300), (89900, 300), (10000, 300), (1024, 300), (10000, 300)]:
        raise ValueError('Unexpected WordVector raw matrix dimensions')
    if not 1 <= queries <= len(test):
        raise ValueError('Requested query limit exceeds the independent test set')
    source = {name: np.load(package/f'data/{name}.source-ids.npy') for name in
              ('base', 'query.build', 'query.metric', 'query.validation', 'query.test', 'base.train')}
    for name, values in source.items():
        if values.shape != (len(matrix(package/f'data/{name}.fbin')),):
            raise ValueError(f'Source-ID/matrix count mismatch: {name}')
    reserved_ids = check_split_ids(*(source[name] for name in
                                   ('base', 'query.build', 'query.metric', 'query.validation', 'query.test', 'base.train')))
    reserved = np.asarray(build[reserved_ids])
    hash_sets = [row_hashes(a) for a in (metric, valid, test, reserved)]
    if any(hash_sets[a] & hash_sets[b] for a in range(4) for b in range(a+1, 4)):
        raise ValueError('Identical vectors overlap refiner/query holdouts')
    packed = np.concatenate((metric, valid, reserved))
    if not np.isfinite(packed).all() or not np.isfinite(test).all():
        raise ValueError('Nonfinite raw queries')
    train_path = inputs/'refiner.train-and-validation.fbin'
    fixed_matrix(train_path, packed)
    fixed_json(inputs/'query-provenance.json', dict(
        dataset=DATASET, actual_base_rows=len(base), original_source_base_rows=1000000,
        official_split_reused=False, metric='raw unnormalized maximum inner product',
        per_vector_normalization=False, global_scaling=False,
        metric_source=str(package/'data/query.metric.fbin'), metric_queries=len(metric),
        validation_source=str(package/'data/query.validation.fbin'), validation_queries=len(valid),
        validation_excluded_from_primary_fit_and_graph_build=True,
        validation_also_used_for_source_leanvec_configuration_selection=True,
        compatibility_build_indices=reserved_ids.tolist(), compatibility_graph_build_used=False,
        test_source=str(package/'data/query.test.fbin'), test_query_limit=queries,
        checks='source ID role separation and selected raw-row SHA256 separation',
        test_queries_used_to_fit_refiner=False))
    truth_path = inputs/'gt.test.combined.ibin'
    truth_ids, _ = pack_truth(base, test, package/'data/gt.test.top100.ibin', truth_path,
                              canonicalize_ties=True)
    original_truth = matrix(package/'data/gt.test.top100.ibin', '<u4')
    reordered_rows = np.flatnonzero(np.any(original_truth != truth_ids, axis=1)).tolist()
    audit_path = inputs/'groundtruth-provenance.json'
    if not audit_path.exists():
        audit = audit_truth_gpu(base, test, truth_ids, gpu)
        fixed_json(audit_path, dict(audit, source=str(package/'data/gt.test.top100.ibin'),
            source_sha256=records['data/gt.test.top100.ibin']['sha256'],
            combined_sha256=content_sha256(truth_path), scores='raw inner product, FP64 accumulation cast to FP32',
            numerical_top100_order_canonicalized=True, reordered_query_rows=reordered_rows,
            original_top10_sets_unchanged=True, original_top100_sets_unchanged=True))
    else:
        audit = json.loads(audit_path.read_text())
        if (audit['source_sha256'] != records['data/gt.test.top100.ibin']['sha256']
                or audit['combined_sha256'] != content_sha256(truth_path)):
            raise ValueError('Existing ground-truth audit no longer matches source/output')
    print(f'[ground truth verified] {len(test)} x Top-100; numerical order adjusted in '
          f'{len(reordered_rows)} rows; ALL source Top-10 sets unchanged', flush=True)

    rng = np.random.default_rng(20261009)
    sample = np.unique(np.r_[0, len(base)-1, rng.integers(0, len(base), 1022)])
    raw = np.asarray(base[sample])
    if not np.isfinite(raw).all():
        raise ValueError('Nonfinite raw base vectors')
    norm_quantiles = np.quantile(np.linalg.norm(raw, axis=1), [0, .25, .5, .75, 1]).tolist()
    imported = {}
    for method in METHODS:
        record = meta['graphs'][method]
        paths = {k: str((package/record['primary_'+k]).resolve(strict=True)) for k in ('model', 'codes')}
        graph = (package/record['graph']).resolve(strict=True)
        if any(not Path(p).is_relative_to(package) for p in [*paths.values(), graph]):
            raise ValueError('Primary path escapes verified package')
        with graph.open('rb') as handle:
            entry, nodes = struct.unpack('<II', handle.read(8))
        construction = json.loads((package/method/'construction.json').read_text())
        if ((entry, nodes) != (record['entry'], len(base))
                or construction['nodes'] != nodes or construction['original_dim'] != 300
                or construction['primary_kind'] != method or construction['representation'] != 'primary'
                or not construction['code_native_construction'] or construction['build_queries'] != len(build)
                or any(Path(construction['files']['primary_'+k]).name != Path(paths[k]).name for k in ('model', 'codes'))):
            raise ValueError(f'{method}: frozen graph/model provenance mismatch')
        model = NativePrimary(paths['model'], paths['codes'])
        try:
            if (model.nodes, model.dim, model.record_bytes) != (len(base), 300, record['record_bytes']):
                raise ValueError(f'{method}: native shape/storage mismatch')
            reconstructed = model.reconstruct_raw(sample)
            error = max(float(np.max(abs(model.score(test[q], sample)-reconstructed@test[q])))
                        for q in (0, min(127, queries-1), queries-1))
            if not np.isfinite(reconstructed).all() or error > 2e-5:
                raise ValueError(f'{method}: raw linear scoring mismatch {error}')
            export = json.loads((package/method/'export_summary.json').read_text())
            imported[method] = dict(paths=paths, graph=str(graph), quantization=dict(
                physical_bytes_per_node=model.record_bytes,
                nominal_code_bits_per_node=record['coordinate_bits'],
                target_primary_bits_per_node=1200,
                actual_refiner_relative_rate=bits/record['coordinate_bits'],
                scorer_dim=model.scorer_dim,
                implementation=export.get('implementation', 'native-'+method)),
                sampled_nodes=len(sample), raw_linear_score_max_error=error,
                reconstruction_mse=float(np.mean(np.square(raw-reconstructed))),
                construction_provenance=str(package/method/'construction.json'))
            print(f'[primary verified] {method}: actual={record["coordinate_bits"]} bits; '
                  f'{model.record_bytes} B/node; scorer_dim={model.scorer_dim}; raw score error={error:.3g}', flush=True)
        finally:
            model.close()
    imported_path = inputs/'imported-primary.json'
    fixed_json(imported_path, dict(status='files-and-sampled-base-order-verified', dataset=DATASET,
        methods=imported, base=str(package/'data/base.fbin'),
        transfer_manifest_sha256=content_sha256(manifest_path),
        payload_verification='all transferred payload SHA256 rechecked locally',
        base_order_proof='immutable payload checksums + native raw reconstruction/score parity',
        sampled_raw_base_norm_quantiles=norm_quantiles,
        sources={name: file_id(package/name) for name in records}))
    common = dict(dataset='WordVector-ID-custom-899K', base=str(package/'data/base.fbin'),
        train_queries=str(train_path), test_queries=str(package/'data/query.test.fbin'),
        test_query_limit=queries, base_rows=len(base), base_dim=300,
        graph_package=str(imported_path), fp_graph='', include_fp32_reference=False,
        combined_ground_truth=str(truth_path), combined_ground_truth_score_kind='inner-product',
        inversion_ranks=[50, 75, 100], build_dir=str(ROOT/'ours/build-wordvector-exp1-v1'),
        primary_budget_bits=1200, primary_dim=300, pq_groups=150,
        rabitq_bits=4, rabitq_config='exact', saq_avg_bits=4.,
        side_bits=bits, budgets=[bits], methods=METHODS,
        device=f'cuda:{gpu}', kmeans_device=f'cuda:{gpu}', geometric_encoder='faiss',
        geometric_encoder_device=f'cuda:{gpu}', ablation_variants=['remora'],
        metric_query_start=0, metric_query_limit=10000,
        select_query_start=10000, select_query_limit=1024,
        build_query_start=11024, build_query_limit=1024,
        pool_train_queries=4096, pool_validation_queries=1024,
        pool_beams=[50, 75, 100, 125, 150, 200], centered_epochs=4, recode_nodes=300000,
        trainer_threads=8, build_threads=8, compile_jobs=8, search_threads=8,
        cpus=list(range(16, 24)), scoring_placement='traversal', lazy_ji=False,
        memory_budget_bytes=256000000000, repeats=repeats, adaptive_extra_repeats=max(5, repeats),
        performance_sampling='adaptive', recall_range=[.80, .99], recall_gap=.01,
        beams=list(range(10, 201, 5))+list(range(225, 1001, 25)),
        adaptive_coarse_beams=[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000],
        adaptive_coarse_depths=[10, 20, 50, 100, 200], rerank_depths=[10, 20, 50, 100, 200],
        rerank_storage='direct', fp32_storage='direct',
        cold_store='/tmp/xzl-wordvector-id-cold/base.899K.cold.bin', seed=20261009)
    for name, directory in [('prepare', 'prepared'), ('performance', 'performance')]:
        config_path = inputs/f'{name}.config.json'
        fixed_json(config_path, dict(common, output_root=str(output/directory)))
        load_config(config_path)
    print(f'[inputs complete] {inputs}; actual database={len(base):,} x 300D; '
          f'primary target 1200 + REMORA {bits} bits; raw IP; graphs/primary read-only', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default=str(ROOT/'datasets/WordVector-ID-native4-M40-L500'))
    parser.add_argument('--tag', default='remora-traversal-exp1-nvme-b96-v1')
    parser.add_argument('--side-bits', type=int, default=96)
    parser.add_argument('--queries', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if '/' in args.tag or args.tag in ('', '.', '..'):
        parser.error('tag must be a single directory name')
    if not 0 < args.side_bits <= 256 or args.side_bits % 8 or args.gpu < 0 or args.repeats < 1:
        parser.error('Invalid side-bits, GPU or repeat count')
    prepare(args.package, args.tag, args.side_bits, args.queries, args.gpu, args.repeats)
