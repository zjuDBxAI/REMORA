"""Verify the transferred WebVid package and configure immutable experiment 1.

No graph construction, primary training, normalization or code conversion.
The data package remains read-only; new manifests/splits live under outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
HERE = ROOT / 'ours/test/t2i-10M'
sys.path[:0] = [str(HERE), str(ROOT / 'ours/python')]
from protocol import content_sha256, file_id, matrix, write_matrix
from pipeline import save_json
from native100_primary import NativePrimary


def fixed_json(path, value):
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f'Existing inputs/configuration differ; use a new tag: {path}')
    else:
        save_json(path, value)


def row_hashes(values):
    return {hashlib.sha256(np.ascontiguousarray(row, dtype='<f4').tobytes()).digest()
            for row in values}


def prepare(package, tag, bits):
    package = Path(package).resolve(strict=True)
    output = ROOT / 'outputs/webvid' / tag
    inputs = output / 'inputs'
    inputs.mkdir(parents=True, exist_ok=True)
    meta = json.loads((package / 'DATASET.json').read_text())
    complete = json.loads((package / 'TRANSFER_COMPLETE.json').read_text())
    if (meta['dataset'] != 'webvid' or meta['metric'] != 'inner_product'
            or meta['dimension'] != 512 or meta['nodes'] != 2505000
            or complete['status'] != 'all_payload_files_size_and_sha256_verified'
            or complete['manifest_sha256'] != content_sha256(package / 'transfer-manifest.json')
            or meta['test_queries_used_for_training']):
        raise ValueError('Unverified/incompatible WebVid transfer')
    manifest = json.loads((package / 'transfer-manifest.json').read_text())
    expected = {f['relative_path']: f for f in manifest['files']}
    for name, record in expected.items():
        path = (package / name).resolve(strict=True)
        if not path.is_relative_to(package) or path.stat().st_size != record['bytes']:
            raise ValueError(f'Transfer payload missing/size changed: {name}')
    # Verify small authoritative metadata against the already verified transfer.
    for name in ['provenance/split/split.manifest.json', 'provenance/data/gt.audit.json']:
        if content_sha256(package / name) != expected[name]['sha256']:
            raise ValueError(f'Transfer metadata checksum changed: {name}')
    source_split = json.loads((package / 'provenance/split/split.manifest.json').read_text())
    if (not source_split['complete'] or source_split['test_in_quantizer_or_graph_training']
            or source_split['validation_in_graph_training']):
        raise ValueError('Invalid source train/validation/test separation')

    base = matrix(package / 'data/base.fbin')
    build = matrix(package / 'data/query.build.fbin')
    validation = matrix(package / 'data/query.validation.fbin')
    test = matrix(package / 'data/query.test.fbin')
    if base.shape != (2505000, 512) or test.shape != (10000, 512) or validation.shape != (1024, 512):
        raise ValueError('Unexpected WebVid data dimensions')
    metric = np.asarray(build[:10000])
    reserved = np.asarray(build[10000:11024])
    hm, hv, ht, hb = map(row_hashes, (metric, validation, test, reserved))
    if hm & hv or hm & ht or hv & ht or hb & hv or hb & ht or hb & hm:
        raise ValueError('Identical query rows overlap reserved splits')
    # Validation is the quantizer/graph holdout from the source server, not a
    # random slice of the full train file that could include graph-training rows.
    packed = np.concatenate((metric, validation, reserved))
    train_path = inputs / 'refiner.train-and-validation.fbin'
    if train_path.exists():
        if not np.array_equal(matrix(train_path), packed):
            raise ValueError('Existing refiner split changed')
    else:
        write_matrix(train_path, packed)
    fixed_json(inputs / 'query-provenance.json', dict(
        metric_source=str(package / 'data/query.build.fbin'), metric_range=[0, 10000],
        validation_source=str(package / 'data/query.validation.fbin'), validation_range=[0, 1024],
        validation_excluded_from_primary_and_graph_training=True,
        raw_row_sha256_overlap_checks='metric/validation/test/compatibility rows disjoint',
        compatibility_rows_usage='not used for graph construction; original graphs reused',
        test_queries_used_to_fit_refiner=False,
        source_split_sha256=content_sha256(package / 'provenance/split/split.manifest.json')))

    rng = np.random.default_rng(20261006)
    ids = np.unique(np.r_[0, len(base)-1, rng.integers(0, len(base), 510)])
    raw = np.asarray(base[ids]); norms = np.einsum('ij,ij->i', raw, raw)
    if not np.isfinite(raw).all() or np.max(abs(norms - 1.)) > 2e-5:
        raise ValueError('WebVid unit-base scoring convention failed')
    imported = {}
    for record in meta['graphs']:
        method = record['method']
        paths = {k: str(package / record[k]) for k in ('model', 'codes')}
        graph = package / record['graph']
        with graph.open('rb') as f:
            entry, nodes = struct.unpack('<II', f.read(8))
        if (entry, nodes) != (record['entry'], len(base)):
            raise ValueError(f'{method}: graph header does not match transferred metadata')
        model = NativePrimary(paths['model'], paths['codes'])
        try:
            if (model.nodes, model.dim, model.record_bytes) != (len(base), 512, record['physical_record_bytes']):
                raise ValueError(f'{method}: native model shape/size mismatch')
            reconstructed = model.reconstruct_raw(ids)
            if not np.isfinite(reconstructed).all():
                raise ValueError('Nonfinite primary reconstruction')
            error = max(float(np.max(abs(model.score(test[q], ids) - reconstructed @ test[q])))
                        for q in (0, 127, 9999))
            if error > 2e-5:
                raise ValueError(f'{method}: actual scorer is not the reconstructed linear map: {error}')
            export = json.loads((package / method / 'export_summary.json').read_text())
            imported[method] = dict(paths=paths, graph=str(graph),
                quantization=dict(physical_bytes_per_node=model.record_bytes,
                    nominal_code_bits_per_node=record['nominal_bits'],
                    implementation=export.get('implementation', 'native-' + method)),
                sampled_nodes=len(ids), raw_linear_score_max_error=error,
                reconstruction_mse=float(np.mean(np.square(raw-reconstructed))),
                construction_provenance=str(package / method / 'construction.json'))
            print(f'[primary verified] {method}: {record["nominal_bits"]} bits; '
                  f'{model.record_bytes} B/node; linear-score error={error:.3g}', flush=True)
        finally:
            model.close()
    report = dict(status='files-and-sampled-base-order-verified', dataset='webvid',
        methods=imported, base=str(package / 'data/base.fbin'),
        base_order_proof='source transfer full-payload SHA256 verification + sampled native reconstructions; no row conversion',
        transfer_manifest_sha256=complete['manifest_sha256'],
        payload_verification='transfer completed SHA256 verification; current sizes and small metadata rechecked',
        sources={name:file_id(package / name) for name in expected})
    imported_path = inputs / 'imported-primary.json'
    fixed_json(imported_path, report)
    prepared = output / 'prepared'
    performance = output / 'performance'
    common = dict(dataset='WebVid-2.5M', base=str(package / 'data/base.fbin'),
        train_queries=str(train_path), test_queries=str(package / 'data/query.test.fbin'),
        base_rows=len(base), base_dim=512, graph_package=str(imported_path),
        fp_graph='', include_fp32_reference=False,
        combined_ground_truth=str(package / 'data/groundtruth.ibin'),
        combined_ground_truth_score_kind='normalized-squared-l2',
        build_dir=str(ROOT / 'ours/build-t2i10m-native100-v1'),
        primary_budget_bits=2048, primary_dim=512, pq_groups=256,
        rabitq_bits=4, saq_avg_bits=4., side_bits=bits, budgets=[32,64,128,160,256],
        device='cuda:2', kmeans_device='cuda:2', geometric_encoder='faiss', geometric_encoder_device='cuda:2',
        ablation_variants=['remora'], metric_query_start=0, metric_query_limit=10000,
        select_query_start=10000, select_query_limit=1024,
        build_query_start=11024, build_query_limit=1024,
        pool_train_queries=4096, pool_validation_queries=1024,
        centered_epochs=4, recode_nodes=300000, trainer_threads=8, build_threads=8,
        compile_jobs=8, search_threads=8, cpus=list(range(16,24)),
        scoring_placement='traversal', lazy_ji=False, repeats=3,
        recall_range=[.80,.99], recall_gap=.01, rerank_storage='direct', fp32_storage='direct',
        cold_store='/tmp/xzl-webvid-cold/base.2.5M.cold.bin',
        seed=20261006)
    fixed_json(inputs / 'prepare.config.json', dict(common, output_root=str(prepared)))
    fixed_json(inputs / 'performance.config.json', dict(common, output_root=str(performance)))
    print(f'[inputs complete] {inputs}; primary2048 + REMORA{bits} bits; '
          f'graphs read-only; no primary/graph rebuilding', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package', default=str(ROOT / 'datasets/WebVid-2.5M-native4-M40-L500'))
    parser.add_argument('--tag', default='remora-traversal-exp1-nvme-b160-v1')
    parser.add_argument('--side-bits', type=int, default=160)
    args = parser.parse_args()
    if '/' in args.tag or args.tag in ('', '.', '..'):
        parser.error('tag must be a single directory name')
    if not 0 < args.side_bits <= 256 or args.side_bits % 8:
        parser.error('side-bits must be a multiple of 8 in [8,256]')
    prepare(args.package, args.tag, args.side_bits)
