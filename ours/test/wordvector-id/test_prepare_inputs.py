"""Unit tests for ID holdouts and raw (not normalized) IP truth packing."""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from prepare_inputs import check_split_ids, check_truth_ids, pack_truth, fixed_json
from protocol import matrix, write_matrix


class WordVectorInputTests(unittest.TestCase):
    def test_metric_subset_and_independent_reserved_rows(self):
        base = np.arange(2000)
        build = np.arange(2000, 4048)
        metric = build[::2]
        reserved = check_split_ids(base, build, metric, np.arange(5000, 5002),
                                   np.arange(6000, 6002), base[:20])
        self.assertEqual(len(reserved), 1024)
        self.assertFalse(np.isin(build[reserved], metric).any())

    def test_query_and_database_roles_cannot_overlap(self):
        with self.assertRaisesRegex(ValueError, 'overlap'):
            check_split_ids(np.arange(2000), np.arange(1500, 4500), np.arange(2000, 2010),
                            np.arange(5000, 5002), np.arange(6000, 6002), np.arange(10))

    def test_metric_cannot_use_test_queries(self):
        with self.assertRaisesRegex(ValueError, 'subsets'):
            check_split_ids(np.arange(2000), np.arange(2000, 5000), np.array([6000]),
                            np.arange(5000, 5002), np.arange(6000, 6002), np.arange(10))

    def test_truth_rejects_invalid_or_duplicate_ids(self):
        for ids in [np.zeros((1, 100), np.uint32), np.arange(100, dtype=np.uint32).reshape(1, 100)]:
            with self.assertRaises(ValueError):
                check_truth_ids(ids, 1, 99)

    def test_raw_ip_truth_preserves_norm_and_source_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = np.c_[np.arange(1, 102, dtype=np.float32), np.zeros(101)].astype(np.float32)
            query = np.array([[2., 0.]], np.float32)
            ids = np.arange(100, 0, -1, dtype=np.uint32).reshape(1, 100)
            write_matrix(root/'ids.ibin', ids, '<u4')
            old, scores = pack_truth(base, query, root/'ids.ibin', root/'combined.ibin')
            self.assertEqual(float(scores[0, 0]), 202.)
            self.assertTrue(np.array_equal(old, ids))
            self.assertEqual((root/'combined.ibin').stat().st_size, 808)
            # Resumption is stable; a cosine implementation would not yield 202.
            again, other = pack_truth(base, query, root/'ids.ibin', root/'combined.ibin')
            self.assertTrue(np.array_equal(again, ids))
            self.assertTrue(np.array_equal(other, scores))

    def test_truth_is_not_silently_reordered(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            base = np.arange(101, dtype=np.float32).reshape(-1, 1)
            write_matrix(path/'ids.ibin', np.arange(100, dtype=np.uint32).reshape(1, 100), '<u4')
            with self.assertRaisesRegex(ValueError, 'ordered'):
                pack_truth(base, np.ones((1, 1), np.float32), path/'ids.ibin', path/'gt.ibin')

    def test_numerical_ties_can_be_ordered_without_changing_top10(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)
            base = np.arange(101, dtype=np.float32).reshape(-1, 1)
            base[1] = 1.0000005
            base[2] = 1.
            ids = np.arange(100, 0, -1, dtype=np.uint32).reshape(1, 100)
            write_matrix(path/'ids.ibin', ids, '<u4')
            new, scores = pack_truth(base, np.ones((1, 1), np.float32), path/'ids.ibin',
                                     path/'gt.ibin', canonicalize_ties=True)
            self.assertTrue(np.array_equal(new[:, :10], ids[:, :10]))
            self.assertTrue((np.diff(scores, axis=1) <= 0).all())
            self.assertTrue(np.array_equal(matrix(path/'ids.ibin', '<u4'), ids))

    def test_existing_configuration_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary)/'config.json'
            fixed_json(path, {'normalization': False})
            fixed_json(path, {'normalization': False})
            with self.assertRaisesRegex(ValueError, 'changed'):
                fixed_json(path, {'normalization': True})


if __name__ == '__main__':
    unittest.main()
