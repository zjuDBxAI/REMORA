"""Regression checks for 512D input, B160, GT labels and traversal jobs."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT / 'ours/test/t2i-10M'), str(ROOT / 'ours/python'),
               str(ROOT / 'ours/test/t2i-1M/python')]
from protocol import load_config, matrix, write_matrix, performance_jobs
from adaptive_performance import curve_specs
from pipeline import Pipeline
from train_remora import export_model
from test_integration import fixture
from qvrr import Primary


class WebVidTests(unittest.TestCase):
    def test_original_t2i_default_remains_b64(self):
        c=load_config()
        self.assertEqual(c['side_bits'],64)
        self.assertEqual(c['budgets'],[16,32,48,64,96,128])
        self.assertEqual(c['combined_ground_truth_score_kind'],'inner-product')

    def test_b160_traversal_plans_and_capped_points(self):
        from importlib.util import spec_from_file_location, module_from_spec
        spec=spec_from_file_location('capped_webvid',ROOT/'ours/scripts/run_native100_capped_experiment1.py')
        cap=module_from_spec(spec); spec.loader.exec_module(cap)
        c=dict(load_config(),side_bits=160,include_fp32_reference=False,beams=[25,100,500])
        curves=curve_specs(c,{1})
        self.assertEqual(len(curves),20)
        self.assertEqual({r['side_bits'] for r in curves if r['variant']=='remora'},{160})
        for curve in curves:
            self.assertTrue(all(j['rerank_depth']<=min(j['beam'],200)
                                for j in cap.bounded_initial(c,curve,200)))
        self.assertEqual({j['side_bits'] for j in performance_jobs(c) if j['variant']=='remora'},{160})

    def test_normalized_gt_converts_without_normalizing_raw_queries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data').mkdir()
            rng=np.random.default_rng(8)
            base=rng.normal(size=(32,512)).astype(np.float32)
            base/=np.linalg.norm(base,axis=1,keepdims=True)
            query=rng.normal(size=(2,512)).astype(np.float32)
            query/=np.linalg.norm(query,axis=1,keepdims=True)
            query*=np.array([.8,1.2],np.float32)[:,None]
            exact=query@base.T
            ids=np.argsort(-exact,axis=1)[:,:20].astype('<u4')
            scores=np.take_along_axis(exact,ids,axis=1)
            dist=2-2*scores/np.linalg.norm(query,axis=1)[:,None]
            source=root/'truth.combined'
            source.write_bytes(struct.pack('<II',2,20)+ids.tobytes()+dist.astype('<f4').tobytes())
            write_matrix(root/'base.fbin',base)
            write_matrix(root/'data/test.fbin',query)
            p=Pipeline.__new__(Pipeline); p.out=root
            p.c=dict(combined_ground_truth=str(source),test_query_limit=2,
                inversion_ranks=[20],base_rows=32,base_dim=512,base=str(root/'base.fbin'),
                combined_ground_truth_score_kind='normalized-squared-l2')
            p.ground_truth()
            np.testing.assert_allclose(matrix(root/'data/truth.scores.fbin'),scores,atol=3e-7)
            np.testing.assert_array_equal(matrix(root/'data/test.fbin'),query)

    @unittest.skipUnless(os.environ.get('QVRR_BUILD_DIR'),'set QVRR_BUILD_DIR for native audit')
    def test_native_b160_512_to_520_padding(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); model,codes,_,queries=fixture(root,n=17,dim=512)
            rng=np.random.default_rng(9); groups=20; dim=520; width=26
            books=rng.normal(scale=.01,size=(groups,256,width)).astype(np.float32)
            labels=rng.integers(0,256,(17,groups),dtype=np.uint8)
            payload=root/'payload'; payload.write_bytes(labels.tobytes())
            output=root/'refiner'; output.mkdir()
            export_model(output,books,np.eye(dim,dtype=np.float32),np.zeros(dim,np.float32),payload,17,160,'1234567890abcdef'*4)
            write_matrix(root/'queries.fbin',queries[:3])
            write_matrix(root/'ids.ibin',np.tile(np.arange(17,dtype=np.uint32),(3,1)),'<u4')
            prefix=root/'audit'
            subprocess.run([str(Path(os.environ['QVRR_BUILD_DIR'])/'roar_exp234_audit'),
                '--primary-kind','qvrr','--primary-model',str(model),'--primary-codes',str(codes),
                '--graph',str(root/'unused.graph'),'--queries',str(root/'queries.fbin'),
                '--out-prefix',str(prefix),'--audit-ids',str(root/'ids.ibin'),
                '--ji-model',str(output/'online.bin'),'--ji-side-codes',str(output/'side.bin'),
                '--ji-query-space','raw'],check=True,capture_output=True)
            residual=np.concatenate([books[g,labels[:,g]] for g in range(groups)],axis=1)[:,:512]
            primary=Primary(model,codes)
            expected=np.asarray([primary.score(q,np.arange(17))+residual@q for q in queries[:3]])
            np.testing.assert_allclose(matrix(str(prefix)+'.scores.fbin'),expected,rtol=3e-5,atol=5e-5)
            self.assertEqual((output/'side.bin').stat().st_size,48+17*20)


if __name__=='__main__': unittest.main()
