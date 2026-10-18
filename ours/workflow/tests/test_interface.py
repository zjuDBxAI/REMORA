import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import artifacts
import interface
import knn
import run


class InterfaceTests(unittest.TestCase):
    def test_native_budget_preserves_dimension(self):
        for dim, groups in ((200,100), (512,256), (300,150)):
            for method in ('opq-pq','qaq'):
                p = interface.native_parameters(dim, method)
                self.assertEqual((p['scorer_dim'], p['pq_groups'], p['code_bits']), (dim,groups,4*dim))

    def test_nearest_legal_pq_not_dimension_truncation(self):
        p = interface.native_parameters(200, 'opq-pq', 768)
        self.assertEqual((p['scorer_dim'], p['pq_groups'], p['code_bits']), (200,100,800))
        with self.assertRaisesRegex(ValueError,'ORIGINAL'):
            interface.native_parameters(200,'opq-pq',pq_groups=96)

    def test_padding_is_counted(self):
        p = interface.native_parameters(200,'extended-rabitq')
        self.assertEqual((p['scorer_dim'],p['coordinate_bits'],p['code_bits']), (256,3,768))
        p = interface.native_parameters(200,'extended-rabitq',rabitq_bits=4)
        self.assertEqual(p['code_bits'],1024)
        p = interface.native_parameters(200,'saq')
        self.assertEqual(p['scorer_dim'],256)
        self.assertEqual(p['avg_bits'],3.125)
        self.assertEqual(p['allocation_quota_bits'],800)
        self.assertIn('actual_code_bits',p)

    def test_zero_budget_rejected(self):
        with self.assertRaises(ValueError):
            interface.native_parameters(200,'opq-pq',0)

    def test_sift_gist_excluded(self):
        self.assertFalse(any('sift' in n.lower() or 'gist' in n.lower() for n in interface.DATASETS))

    def test_gpu_is_not_hardcoded(self):
        catalog = json.loads((HERE/'catalog.json').read_text())
        with tempfile.TemporaryDirectory() as d:
            jobs = run.plan(Path(d),catalog,'t2i-10M',['exp2','exp5'],'/python',gpu=1)
            for job in jobs:
                self.assertEqual(job.command[job.command.index('--gpu')+1],'1')

    def test_evaluation_dependency_order(self):
        with patch.object(run,'main',return_value=0) as forward:
            interface.main(['evaluate','--dataset','t2i-10M','--experiments','1,2,3,4,5'])
            command=forward.call_args[0][0]
            self.assertEqual(command[command.index('--stages')+1],'exp1,exp3,exp4,exp2,exp5')

    def test_build_command_graph_and_encoder_parameters(self):
        r = dict(output='/out',build_dir='/bin',method='opq-pq',parameters=interface.native_parameters(200,'opq-pq'),
                 python='/python',fit_queries=None,base='/data/base',train_size=512,iterations=2,seed=7,
                 fit_query_limit=0,query_limit=32,topk=10,degree=8,build_beam=25,threads=3,
                 knn_backend='gpu',gpu=1,base_block=512,query_batch=8)
        encode, scan, parity, graph, c = interface.build_commands(Path('/repo'),r)
        self.assertEqual(encode[encode.index('--opq-dim')+1],'200')
        self.assertEqual(encode[encode.index('--pq-groups')+1],'100')
        self.assertEqual(graph[graph.index('--M_pjbp')+1],'8')
        self.assertEqual(graph[graph.index('--L_pjpq')+1],'25')
        self.assertIn('--disable-ji', graph)
        self.assertIn('--validate-only',parity)
        self.assertEqual(scan[scan.index('--gpu')+1],'1')

    def test_plan_creates_no_output(self):
        with tempfile.TemporaryDirectory() as folder:
            d = Path(folder)
            (d/'base.fbin').write_bytes(struct.pack('<II',512,8)+b'\0'*(512*8*4))
            (d/'query.build.fbin').write_bytes(struct.pack('<II',16,8)+b'\0'*(16*8*4))
            out = d/'new'
            with redirect_stdout(io.StringIO()):
                self.assertEqual(interface.main(['build','--dataset','t2i-10M','--method','fp32',
                    '--data-root',str(d),'--repo',str(HERE.parents[1]),'--output',str(out),
                    '--knn-k','10','--build-beam','16']),0)
            self.assertFalse(out.exists())

    def test_l2_descriptor_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            descriptor = Path(folder)/'dataset.json'
            descriptor.write_text(json.dumps(dict(dataset='t2i-10M',metric='l2')))
            with self.assertRaisesRegex(ValueError,'L2'):
                interface.main(['build','--dataset','t2i-10M','--method','fp32','--dataset-config',str(descriptor)])

    def test_training_rejects_terminal(self):
        with tempfile.TemporaryDirectory() as folder:
            config = Path(folder)/'config.json'
            config.write_text(json.dumps(dict(scoring_placement='terminal')))
            # Config defaults are only needed for the validation guard.
            with patch.object(run,'config',return_value={'scoring_placement':'terminal'}):
                with self.assertRaisesRegex(ValueError,'traversal'):
                    interface.training_command(argparse.Namespace(repo=HERE.parents[1],config=config,dataset='t2i-10M',output=None))

    def test_bad_matrix_size(self):
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'bad.fbin'
            p.write_bytes(struct.pack('<II',2,3))
            with self.assertRaisesRegex(ValueError,'size'):
                artifacts.matrix_shape(p)

    def test_generic_export_uses_nodes_not_base_rows(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'export_summary.json').write_text(json.dumps(dict(nodes=512,original_dim=200,
                nominal_code_bits_per_node=800,physical_bytes_per_node=100)))
            q=interface.quantization(dict(primary_model=str(path/'primary_model.bin')),
                dict(method='opq-pq',nodes=512,parameters=dict(original_dim=200)))
            self.assertEqual(q['nominal_code_bits_per_node'],800)

    def test_zero_bit_native_export_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            (path/'export_summary.json').write_text(json.dumps(dict(base_rows=512,original_dim=64,
                nominal_code_bits_per_node=0,physical_bytes_per_node=72)))
            with self.assertRaisesRegex(ValueError,'zero-bit'):
                interface.quantization(dict(primary_model=str(path/'saq_model.bin')),
                    dict(method='saq',nodes=512,parameters=dict(original_dim=64)))


try:
    import numpy as np
except ImportError:
    np = None


@unittest.skipIf(np is None,'NumPy needed for block merge tests')
class MergeTests(unittest.TestCase):
    def test_merge_matches_full_scan_and_ties(self):
        scores, ids = knn.merge_topk(np.array([[.5,.1]],np.float32),np.array([[2,1]],np.uint32),
                                   np.array([[.7,.5]],np.float32),np.array([[5,0]],np.uint32),2)
        np.testing.assert_array_equal(ids,[[5,0]])
        np.testing.assert_allclose(scores,[[.7,.5]])

    def test_invalid_padding_does_not_win(self):
        scores, ids = knn.merge_topk(np.full((1,3),-np.inf),np.full((1,3),2**32-1,np.uint32),
                                   np.array([[.2,.1]]),np.array([[4,1]],np.uint32),3)
        np.testing.assert_array_equal(ids[0,:2],[4,1])
        self.assertEqual(float(scores[0,2]),-np.inf)


@unittest.skipUnless(os.environ.get('REMORA_INTERFACE_TEST_REPO'),'Set REMORA_INTERFACE_TEST_REPO for Faiss integration tests')
class ScanTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        rng = np.random.default_rng(87)
        self.base = rng.normal(size=(31,8)).astype(np.float32)
        self.queries = rng.normal(size=(7,8)).astype(np.float32)
        for path, data in ((self.root/'base.fbin',self.base),(self.root/'query.fbin',self.queries)):
            path.write_bytes(struct.pack('<II',*data.shape)+data.tobytes())
        self.args = knn.parser().parse_args(['--repo',os.environ['REMORA_INTERFACE_TEST_REPO'],
            '--base',str(self.root/'base.fbin'),'--queries',str(self.root/'query.fbin'),
            '--output',str(self.root/'knn.ibin'),'--topk','3','--backend','cpu',
            '--base-block','9','--query-batch','2','--threads','2'])

    def tearDown(self):
        self.tmp.cleanup()

    def assert_truth(self):
        ids=np.memmap(self.args.output,'<u4','r',offset=8,shape=(7,3))
        scores=self.queries@self.base.T
        expected=np.argsort(-scores,axis=1)[:,:3]
        np.testing.assert_array_equal(ids,expected)

    def test_block_scan_matches_full_database(self):
        knn.scan(self.args)
        self.assert_truth()
        self.assertEqual(knn.scan(self.args)['status'],'COMPLETE')

    def test_interrupt_after_committed_block_and_resume(self):
        original=knn.merge_topk
        calls=0
        def interrupted(*args):
            nonlocal calls
            calls+=1
            if calls==6:
                raise RuntimeError('simulated interruption during next generation')
            return original(*args)
        with patch.object(knn,'merge_topk',side_effect=interrupted):
            with self.assertRaisesRegex(RuntimeError,'simulated'):
                knn.scan(self.args)
        state=json.loads((self.root/'knn.ibin.checkpoint/state.json').read_text())
        self.assertEqual(state['end'],9)
        knn.scan(self.args)
        self.assert_truth()

    def test_changed_inputs_cannot_reuse(self):
        knn.scan(self.args)
        with (self.root/'query.fbin').open('r+b') as handle:
            handle.seek(8); handle.write(struct.pack('<f',12.3))
        with self.assertRaisesRegex(ValueError,'changed'):
            knn.scan(self.args)

    def test_gpu_scan_matches_full_database(self):
        faiss=knn.dependencies(Path(self.args.repo))
        if not hasattr(faiss,'StandardGpuResources') or faiss.get_num_gpus()<1:
            self.skipTest('GPU Faiss unavailable')
        self.args.backend='gpu'
        knn.scan(self.args)
        self.assert_truth()


if __name__=='__main__':
    unittest.main()
