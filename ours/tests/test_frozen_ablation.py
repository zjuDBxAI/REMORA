import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'ours/test/t2i-10M'), str(ROOT/'ours/scripts')]
from protocol import write_matrix

spec = importlib.util.spec_from_file_location('frozen_ablation', ROOT/'ours/scripts/run_frozen_ablation.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class FrozenAblationTests(unittest.TestCase):
    def test_webvid_native_rate_plan(self):
        plan = study.jobs(['a','b','c','d','e'], [100,200], [0,100], 3, 160)
        self.assertEqual(len(plan), 420)
        for point in plan:
            self.assertEqual(point['side_bits'], 0 if point['variant']=='primary' else 160)
            self.assertEqual(point['experiment'], 2)
            self.assertLessEqual(point['rerank_depth'], point['beam'])
        for bits in (0,64.0,159,264,True):
            with self.assertRaises(ValueError): study.jobs(['a'], [100], [0], 1, bits)
        with self.assertRaises(ValueError): study.jobs(['a'], [50], [100], 1, 160)

    def test_k5_is_not_k10_h5(self):
        exact = np.arange(10,0,-1,dtype='f4')[None,:]
        approximate = exact.copy()
        approximate[0,4], approximate[0,5] = approximate[0,5], approximate[0,4]
        row = study.boundary_at_five(exact, approximate)
        self.assertEqual(row['metric'], 'BoundaryInv-K5-h5')
        self.assertEqual(row['pairs'], 25)
        self.assertEqual(row['inversions'], 1)
        self.assertAlmostEqual(row['inversion_rate'], .04)
        with self.assertRaises(ValueError): study.boundary_at_five(exact[:,:5], approximate[:,:5])

    def test_frozen_inputs_require_same_queries_gt_and_configuration(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            prepared, performance, teacher = [root/name for name in ('prepared','performance','teacher')]
            for folder in (prepared/'data', performance, teacher/'experiment3/raw'):
                folder.mkdir(parents=True)
            c = dict(dataset='WebVid-2.5M',base_rows=2505000,base_dim=512,side_bits=160,
                     base='base',test_queries='test',graph_package='primaries',seed=79,
                     scoring_placement='traversal')
            query = prepared/'data/test.fbin'
            write_matrix(query, np.ones((2,512),dtype='f4'), '<f4')
            ids = np.broadcast_to(np.arange(200,dtype='u4'), (2,200)).copy()
            write_matrix(prepared/'data/truth.ids.ibin', ids, '<u4')
            write_matrix(teacher/'experiment3/raw/common.ids.ibin', ids, '<u4')
            for folder in (performance, teacher):
                (folder/'configuration.json').write_text(json.dumps(dict(config=c)))
            (teacher/'study.json').write_text(json.dumps(dict(dataset=c['dataset'],side_bits=160,
                test_sha256=hashlib.sha256(query.read_bytes()).hexdigest())))
            study.validate_frozen_inputs(prepared, performance, teacher, c)
            ids[0,0] = 201
            write_matrix(teacher/'experiment3/raw/common.ids.ibin', ids, '<u4')
            with self.assertRaisesRegex(ValueError, 'node IDs'):
                study.validate_frozen_inputs(prepared, performance, teacher, c)
            changed = dict(c, side_bits=64)
            (performance/'configuration.json').write_text(json.dumps(dict(config=changed)))
            with self.assertRaisesRegex(ValueError, 'side_bits'):
                study.validate_frozen_inputs(prepared, performance, teacher, c)

    def test_missing_timing_is_not_drawn(self):
        self.assertEqual(study.summarize_rows([], [], 3, lambda rows,repeats: []), [])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'bars.svg'
            study.bar_plot(path, [dict(primary='a',variant='primary',timing_complete=False,qps=100)],
                           ['a'], 'qps', 'Test', 'QPS')
            self.assertFalse(path.exists())


if __name__ == '__main__':
    unittest.main()
