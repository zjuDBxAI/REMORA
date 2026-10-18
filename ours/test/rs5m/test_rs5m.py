"""Regression tests: preserve IP ANN truth and reject malformed/changed input."""
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from prepare_inputs import pack_truth, fixed_matrix, fixed_json
from protocol import matrix, write_matrix, load_config


class RS5MTests(unittest.TestCase):
    def test_ip_truth_layout_and_resume(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ids = np.tile(np.arange(100, dtype=np.uint32), (5, 1))
            scores = np.tile(np.linspace(.9, .1, 100, dtype=np.float32), (5, 1))
            write_matrix(root/'ids', ids, '<u4'); write_matrix(root/'scores', scores)
            output = root/'combined'
            pack_truth(root/'ids', root/'scores', output, 3, 1000)
            before = output.stat().st_mtime_ns
            pack_truth(root/'ids', root/'scores', output, 3, 1000)
            self.assertEqual(output.stat().st_mtime_ns, before)
            np.testing.assert_array_equal(np.memmap(output, mode='r', dtype='<u4', offset=8, shape=(3, 100)), ids[:3])
            np.testing.assert_array_equal(np.memmap(output, mode='r', dtype='<f4', offset=1208, shape=(3, 100)), scores[:3])
            with self.assertRaisesRegex(ValueError, 'Existing packed GT'):
                pack_truth(root/'ids', root/'scores', output, 4, 1000)

    def test_invalid_truth_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            write_matrix(root/'ids', np.full((2, 100), 100, np.uint32), '<u4')
            write_matrix(root/'scores', np.ones((2, 100), np.float32))
            with self.assertRaisesRegex(ValueError, 'Invalid RS5M'):
                pack_truth(root/'ids', root/'scores', root/'combined', 2, 100)

    def test_query_and_configuration_immutable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            values = np.ones((3, 512), np.float32)
            fixed_matrix(root/'queries', values); fixed_matrix(root/'queries', values)
            np.testing.assert_array_equal(matrix(root/'queries'), values)
            with self.assertRaisesRegex(ValueError, 'query split changed'):
                fixed_matrix(root/'queries', values*2)
            fixed_json(root/'config.json', {'side_bits': 160})
            with self.assertRaisesRegex(ValueError, 'configuration changed'):
                fixed_json(root/'config.json', {'side_bits': 64})

    def test_original_defaults_unchanged(self):
        self.assertEqual(load_config()['side_bits'], 64)
        self.assertEqual(load_config()['combined_ground_truth_score_kind'], 'inner-product')


if __name__ == '__main__':
    unittest.main()
