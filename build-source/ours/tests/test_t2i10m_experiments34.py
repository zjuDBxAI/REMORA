import importlib.util
from pathlib import Path
import struct
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('exp34', Path(__file__).parents[1]/'scripts/run_t2i10m_experiments34.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class CoverageTests(unittest.TestCase):
    def test_loss_decomposition(self):
        truth=np.arange(10,dtype=np.uint32).reshape(1,10)
        output=np.array([[0,1,2,3,4,5,6,20,21,22]],dtype=np.uint32)
        visited=(np.array([0,12]),np.array([0,1,2,3,4,5,6,7,8,20,21,22]))
        retained=(np.array([0,11]),np.array([0,1,2,3,4,5,6,7,20,21,22]))
        rows=study.coverage_rows(truth,output,visited,retained,30)
        summary=study.mean_coverage(rows)
        self.assertAlmostEqual(summary['visited_coverage'],.9)
        self.assertAlmostEqual(summary['retained_coverage'],.8)
        self.assertAlmostEqual(summary['recall_at_10'],.7)
        self.assertAlmostEqual(summary['missed_loss']+summary['discarded_loss']+summary['ranking_loss'],1-summary['recall_at_10'])

    def test_out_of_range_rejected(self):
        truth=np.arange(10,dtype=np.uint32).reshape(1,10)
        trace=(np.array([0,10]),np.arange(10))
        with self.assertRaises(ValueError):
            study.coverage_rows(truth,truth,trace,(np.array([0,1]),np.array([30])),30)

    def test_csr_header_and_size(self):
        with tempfile.TemporaryDirectory(prefix='exp34-unit-') as folder:
            path=Path(folder)/'trace.csr'
            path.write_bytes(struct.pack('<8sIQIQ',b'DRBVST01',1,0,1,2)+struct.pack('<2Q2I',0,2,4,6))
            offsets,nodes=study.csr(path,1)
            np.testing.assert_array_equal(offsets,[0,2])
            np.testing.assert_array_equal(nodes,[4,6])
            with self.assertRaises(ValueError):
                study.csr(path,2)


if __name__ == '__main__':
    unittest.main()
