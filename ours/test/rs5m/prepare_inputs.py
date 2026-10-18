"""Verify the native 512D RS5M transfer and configure experiment 1.

The transferred data, five primary encodings and graphs remain read-only.
Only REMORA models and experiment-specific inputs/results are created.
The separate ANN ground-truth IDs/scores are packed without relabelling them;
paired-image metadata is NEVER used as ANN ground truth.
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


def fixed_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing configuration changed; use a new RS5M_TAG: {path}')
    else:
        save_json(path, value)


def fixed_matrix(path, values):
    if path.exists():
        if not np.array_equal(matrix(path), values):
            raise ValueError(f'Existing query split changed: {path}')
    else:
        write_matrix(path, values)


def row_hashes(values):
    return {hashlib.sha256(np.ascontiguousarray(row, dtype='<f4').tobytes()).digest()
            for row in values}


def pack_truth(ids_path, scores_path, output, queries, nodes):
    """BigANN layout: one header, all uint32 IDs, then all FP32 IP scores."""
    ids, scores = matrix(ids_path, '<u4'), matrix(scores_path)
    if (ids.shape != scores.shape or queries > len(ids) or ids.shape[1] < 100
            or queries < 1):
        raise ValueError('Missing/mismatched RS5M ANN Top-100 ground truth')
    selected = np.ascontiguousarray(ids[:queries, :100], dtype='<u4')
    values = np.ascontiguousarray(scores[:queries, :100], dtype='<f4')
    if (np.any(selected >= nodes) or not np.isfinite(values).all()
            or np.any(np.diff(values, axis=1) > 2e-6)):
        raise ValueError('Invalid RS5M inner-product ground truth')
    size = 8 + 8 * selected.size
    if output.exists():
        with output.open('rb') as f:
            shape = struct.unpack('<II', f.read(8))
        if shape != selected.shape or output.stat().st_size != size:
            raise ValueError('Existing packed GT header/size changed')
        old_ids = np.memmap(output, mode='r', dtype='<u4', offset=8, shape=shape)
        old_scores = np.memmap(output, mode='r', dtype='<f4', offset=8+4*selected.size, shape=shape)
        if not np.array_equal(selected, old_ids) or not np.array_equal(values, old_scores):
            raise ValueError('Existing packed GT content changed')
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + '.tmp')
        with temporary.open('wb') as f:
            f.write(struct.pack('<II', *selected.shape))
            f.write(selected.tobytes()); f.write(values.tobytes())
            f.flush(); os.fsync(f.fileno())
        temporary.replace(output)
    return selected, values


def prepare(package, tag, bits=160, queries=10000, gpu=2, repeats=3):
    package = Path(package).resolve(strict=True)
    inputs = ROOT / 'outputs/rs5m' / tag / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((package/'DATASET.json').read_text())
    complete = json.loads((package/'TRANSFER_COMPLETE.json').read_text())
    manifest_path = package/'transfer-manifest.json'
    if (meta['directory'] != 'RS5M-512-native4-M40-L500'
            or meta['metric'] != 'inner_product' or meta['dimension'] != 512
            or meta['nodes'] != 4828191 or not meta['per_vector_normalization']
            or meta['test_queries_used_for_training'] or meta['graph_count'] != 5
            or complete['status'] != 'all_payload_files_size_and_sha256_verified'
            or complete['manifest_sha256'] != content_sha256(manifest_path)):
        raise ValueError('Unverified/incompatible RS5M transfer; do not use the old 1024D package')
    manifest = json.loads(manifest_path.read_text())
    expected = {r['relative_path']: r for r in manifest['files']}
    if len(expected) != manifest['file_count'] or complete['file_count'] != len(expected):
        raise ValueError('Transfer inventory count mismatch')
    for name, record in expected.items():
        path = (package/name).resolve(strict=True)
        if not path.is_relative_to(package) or path.stat().st_size != record['bytes']:
            raise ValueError(f'Transfer payload missing/size changed: {name}')
    # Full payload SHA256 was checked on transfer. Recheck the authoritative
    # metadata, ground truth and query-source IDs used by this experiment.
    checks = ['DATASET.json', 'provenance/data/split.manifest.json',
              'provenance/data/groundtruth.validation.json',
              'provenance/data/groundtruth.summary.json',
              'data/groundtruth.ibin', 'data/groundtruth.scores.fbin',
              'data/query.train.source_ids.npy', 'data/query.validation.source_ids.npy',
              'data/query.test.source_ids.npy']
    for method in METHODS:
        checks += [f'{method}/construction.json', f'{method}/export_summary.json']
    for name in checks:
        if content_sha256(package/name) != expected[name]['sha256']:
            raise ValueError(f'Authoritative transfer artifact checksum changed: {name}')
    split = json.loads((package/'provenance/data/split.manifest.json').read_text())
    gt_audit = json.loads((package/'provenance/data/groundtruth.validation.json').read_text())
    if (not split['complete'] or split['test_in_quantizer_or_graph_training']
            or split['train_test_hash_overlap'] or split['train_validation_hash_overlap']
            or split['train_validation_id_overlap'] or split['validation_test_hash_overlap']
            or not gt_audit['complete'] or gt_audit['base_rows'] != meta['nodes']
            or gt_audit['topk'] != 100
            or gt_audit['ids_sha256'] != expected['data/groundtruth.ibin']['sha256']
            or gt_audit['scores_sha256'] != expected['data/groundtruth.scores.fbin']['sha256']):
        raise ValueError('Invalid training/validation/test separation or ANN truth audit')
    base = matrix(package/'data/base.fbin')
    train = matrix(package/'data/query.train.fbin')
    build = matrix(package/'data/query.build.fbin')
    valid = matrix(package/'data/query.validation.fbin')
    test = matrix(package/'data/query.test.fbin')
    if (base.shape != (4828191, 512) or train.shape != (482819, 512)
            or build.shape != train.shape or valid.shape != (1024, 512)
            or test.shape != (100000, 512)):
        raise ValueError('Unexpected native RS5M shapes')
    if not 1 <= queries <= len(test):
        raise ValueError('Requested query limit exceeds RS5M test set')
    source_ids = [np.load(package/f'data/query.{s}.source_ids.npy')
                  for s in ('train', 'validation', 'test')]
    if any(v.shape != (len(q),) for v, q in zip(source_ids, (train, valid, test))):
        raise ValueError('Query source IDs and embedding rows mismatch')
    if any(len(np.intersect1d(source_ids[a], source_ids[b]))
           for a, b in ((0, 1), (0, 2), (1, 2))):
        raise ValueError('Query source IDs overlap across splits')
    metric, reserved = np.asarray(train[:10000]), np.asarray(train[10000:11024])
    hm, hv, ht, hb = map(row_hashes, (metric, valid, test, reserved))
    if hm & hv or hm & ht or hv & ht or hb & hv or hb & ht or hb & hm:
        raise ValueError('Identical query vectors overlap refiner splits')
    packed_queries = np.concatenate((metric, valid, reserved))
    if not np.isfinite(packed_queries).all() or not np.isfinite(test).all():
        raise ValueError('Nonfinite raw RS5M query vectors')
    train_path = inputs/'refiner.train-and-validation.fbin'
    fixed_matrix(train_path, packed_queries)
    fixed_json(inputs/'query-provenance.json', dict(
        metric_source=str(package/'data/query.train.fbin'), metric_range=[0, 10000],
        metric_source_used_for_primary_graph_training=True,
        validation_source=str(package/'data/query.validation.fbin'), validation_range=[0, 1024],
        validation_excluded_from_primary_fit_and_graph_training=True,
        validation_also_used_for_source_leanvec_configuration_selection=True,
        test_source=str(package/'data/query.test.fbin'), test_range=[0, queries],
        available_test_queries=100000, test_queries_used_to_fit_refiner=False,
        checks='all source IDs disjoint; chosen refiner query vectors disjoint from entire test set',
        compatibility_rows_usage='no graph construction; original primary graphs reused',
        source_split_sha256=expected['provenance/data/split.manifest.json']['sha256']))
    combined = inputs/'gt.test.combined.ibin'
    gt_ids, gt_scores = pack_truth(package/'data/groundtruth.ibin',
        package/'data/groundtruth.scores.fbin', combined, queries, len(base))
    error = max(float(np.max(abs(np.asarray(base[gt_ids[q]], np.float64) @
                                    np.asarray(test[q], np.float64) - gt_scores[q])))
                for q in np.unique(np.linspace(0, queries-1, min(16, queries)).astype(int)))
    if error > 2e-5:
        raise ValueError(f'ANN ground truth/raw base or query order mismatch: {error}')
    fixed_json(inputs/'groundtruth-provenance.json', dict(
        full_database=True, base_rows=len(base), base_dim=512, score='FP32 inner product',
        source_ids=str(package/'data/groundtruth.ibin'),
        source_scores=str(package/'data/groundtruth.scores.fbin'), query_range=[0, queries],
        depth=100, combined_sha256=content_sha256(combined), sampled_max_score_error=error,
        paired_image_metadata_not_used=True, scores_converted=False,
        encoder_holdout=False, ann_query_holdout=True))

    rng = np.random.default_rng(20261007)
    ids = np.unique(np.r_[0, len(base)-1, rng.integers(0, len(base), 1022)])
    raw = np.asarray(base[ids])
    if not np.isfinite(raw).all() or np.max(abs(np.einsum('ij,ij->i', raw, raw)-1)) > 2e-5:
        raise ValueError('RS5M normalized raw-base scoring convention failed')
    records = {r['method']: r for r in meta['graphs']}
    if set(records) != set(METHODS):
        raise ValueError('Expected all five primary graphs')
    imported = {}
    for method in METHODS:
        record = records[method]
        paths = {k: str((package/record[k]).resolve(strict=True)) for k in ('model', 'codes')}
        graph = (package/record['graph']).resolve(strict=True)
        if any(not Path(p).is_relative_to(package) for p in [*paths.values(), graph]):
            raise ValueError('Primary artifact path escapes RS5M package')
        with graph.open('rb') as f:
            entry, nodes = struct.unpack('<II', f.read(8))
        construction = json.loads((package/method/'construction.json').read_text())
        if ((entry, nodes) != (record['entry'], len(base))
                or graph.stat().st_size != record['graph_bytes'] or record['nominal_bits'] != 2048
                or construction['nodes'] != nodes or construction['original_dim'] != 512
                or construction['primary_kind'] != method or construction['representation'] != 'primary'
                or construction['build_queries'] != len(build) or construction['query_start'] != 0
                or any(Path(construction['files']['primary_'+k]).name != Path(paths[k]).name
                       for k in ('model', 'codes'))):
            raise ValueError(f'{method}: graph construction/model provenance mismatch')
        model = NativePrimary(paths['model'], paths['codes'])
        try:
            if (model.nodes, model.dim, model.record_bytes) != (len(base), 512, record['physical_record_bytes']):
                raise ValueError(f'{method}: native shape/storage mismatch')
            reconstruction = model.reconstruct_raw(ids)
            if not np.isfinite(reconstruction).all():
                raise ValueError('Nonfinite native raw reconstruction')
            score_error = max(float(np.max(abs(model.score(test[q], ids)-reconstruction@test[q])))
                              for q in (0, min(127, queries-1), queries-1))
            if score_error > 2e-5:
                raise ValueError(f'{method}: decoded map/actual scorer mismatch {score_error}')
            export = json.loads((package/method/'export_summary.json').read_text())
            imported[method] = dict(paths=paths, graph=str(graph), quantization=dict(
                physical_bytes_per_node=model.record_bytes, nominal_code_bits_per_node=2048,
                implementation=export.get('implementation', 'native-'+method)),
                sampled_nodes=len(ids), raw_linear_score_max_error=score_error,
                reconstruction_mse=float(np.mean(np.square(raw-reconstruction))),
                construction_provenance=str(package/method/'construction.json'))
            print(f'[primary verified] {method}: nominal=2048 bit; physical={model.record_bytes} B/node; '
                  f'scorer_dim={model.scorer_dim}; linear-score error={score_error:.3g}', flush=True)
        finally:
            model.close()
    imported_path = inputs/'imported-primary.json'
    fixed_json(imported_path, dict(status='files-and-sampled-base-order-verified', dataset='rs5m',
        methods=imported, base=str(package/'data/base.fbin'),
        base_order_proof='full source-transfer SHA256 verification plus native raw-score/sample validation; no node conversion',
        transfer_manifest_sha256=complete['manifest_sha256'],
        payload_verification='verified transfer inventory; current sizes, authoritative metadata, IDs/GT rechecked',
        sources={name: file_id(package/name) for name in expected}))
    output = inputs.parent
    common = dict(dataset='RS5M-512', base=str(package/'data/base.fbin'), train_queries=str(train_path),
        test_queries=str(package/'data/query.test.fbin'), test_query_limit=queries,
        base_rows=len(base), base_dim=512, graph_package=str(imported_path),
        fp_graph='', include_fp32_reference=False, combined_ground_truth=str(combined),
        combined_ground_truth_score_kind='inner-product', inversion_ranks=[50, 75, 100],
        build_dir=str(ROOT/'ours/build-rs5m-exp1-v1'), primary_budget_bits=2048,
        primary_dim=512, pq_groups=256, rabitq_bits=4, saq_avg_bits=4.,
        side_bits=bits, budgets=[bits], methods=METHODS,
        device=f'cuda:{gpu}', kmeans_device=f'cuda:{gpu}', geometric_encoder='faiss',
        geometric_encoder_device=f'cuda:{gpu}', ablation_variants=['remora'],
        metric_query_start=0, metric_query_limit=10000, select_query_start=10000, select_query_limit=1024,
        build_query_start=11024, build_query_limit=1024, pool_train_queries=4096,
        pool_validation_queries=1024, pool_beams=[50, 75, 100, 125, 150, 200],
        centered_epochs=4, recode_nodes=300000, trainer_threads=8, build_threads=8, compile_jobs=8,
        search_threads=8, cpus=list(range(16, 24)), scoring_placement='traversal', lazy_ji=False,
        memory_budget_bytes=256000000000, repeats=repeats, adaptive_extra_repeats=max(5, repeats),
        performance_sampling='adaptive', recall_range=[.80, .99], recall_gap=.01,
        beams=list(range(10, 201, 5))+list(range(225, 1001, 25)),
        adaptive_coarse_beams=[10, 25, 50, 75, 100, 150, 200, 300, 500, 750, 1000],
        adaptive_coarse_depths=[10, 20, 50, 100, 200], rerank_depths=[10, 20, 50, 100, 200],
        rerank_storage='direct', fp32_storage='direct',
        cold_store='/tmp/xzl-rs5m512-cold/base.4.8M.cold.bin', seed=20261007)
    for name, directory in [('prepare', 'prepared'), ('performance', 'performance')]:
        config_path = inputs/f'{name}.config.json'
        fixed_json(config_path, dict(common, output_root=str(output/directory)))
        load_config(config_path)
    print(f'[inputs complete] {inputs}; {len(base):,} vectors; {queries}/{len(test)} test queries; '
          f'primary2048 + REMORA{bits} bits; graphs/primary read-only; no rebuilding', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default=str(ROOT/'datasets/RS5M-512-native4-M40-L500'))
    parser.add_argument('--tag', default='remora-traversal-exp1-nvme-b160-v1')
    parser.add_argument('--side-bits', type=int, default=160)
    parser.add_argument('--queries', type=int, default=10000)
    parser.add_argument('--gpu', type=int, default=2)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if '/' in args.tag or args.tag in ('', '.', '..'):
        parser.error('tag must be a single directory name')
    if not 0 < args.side_bits <= 256 or args.side_bits % 8 or args.gpu < 0 or args.repeats < 1:
        parser.error('Invalid side-bits, GPU or repeats')
    prepare(args.package, args.tag, args.side_bits, args.queries, args.gpu, args.repeats)
