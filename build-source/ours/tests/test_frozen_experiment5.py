import importlib.util
from pathlib import Path
import tempfile
import unittest
import os

import numpy as np

spec=importlib.util.spec_from_file_location('frozen_exp5',Path(__file__).parents[1]/'scripts/run_frozen_experiment5.py')
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RelativeRateTests(unittest.TestCase):
    def test_high_budget_decoder_matches_lut(self):
        rng=np.random.default_rng(70)
        books=rng.normal(size=(40,256,13)).astype(np.float32)
        projection=np.eye(520,dtype=np.float32)
        mean=rng.normal(size=512).astype(np.float32)
        labels=rng.integers(0,256,size=(7,40),dtype=np.uint8)
        query=rng.normal(size=512).astype(np.float32)
        reconstructed=study.decode_correction(books,projection,mean,labels,512)
        padded=np.pad(query,(0,8)).reshape(40,13)
        lut=np.einsum('gw,gkw->gk',padded,books)
        actual=query@mean+sum(lut[g,labels[:,g]] for g in range(40))
        np.testing.assert_allclose(reconstructed@query,actual,rtol=2e-5,atol=2e-5)

    def test_unique_storage_does_not_count_model_codes_alias_twice(self):
        with tempfile.TemporaryDirectory() as directory:
            first=Path(directory)/'model-code.bin'
            alias=Path(directory)/'alias.bin'
            first.write_bytes(b'12345678')
            os.link(first,alias)
            self.assertEqual(study.unique_file_bytes([first,first,alias]),8)

    def test_scaled_relative_budgets(self):
        multipliers=(.5,1.,1.5,2.)
        self.assertEqual([int(64*m) for m in multipliers],[32,64,96,128])
        self.assertEqual([int(160*m) for m in multipliers],[80,160,240,320])


if __name__=='__main__':
    unittest.main()
