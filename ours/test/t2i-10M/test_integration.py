"""Small synthetic tests; optional native checks via QVRR_BUILD_DIR."""
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest
import numpy as np
from protocol import ROOT,write_matrix,matrix,load_config,global_inversions
sys.path[:0]=[str(ROOT/'ours/python'),str(ROOT/'ours/test/t2i-1M/python')]
from qvrr import Primary
from train_remora import export_model,support,train
from train_terminal_ranking_ji import Pools
from plots import frontier


def fixture(folder,n=512,dim=4):
    rng=np.random.default_rng(412)
    books=rng.normal(size=(dim,256,1)).astype('<f4'); codes=rng.integers(0,256,(n,dim),dtype=np.uint8)
    projection=np.eye(dim,dtype='<f4')
    model=folder/'primary.model'; code=folder/'primary.codes'
    model.write_bytes(struct.pack('<8sIQ7I',b'OPQPQ001',1,n,dim,dim,dim,256,1,0,1)+projection.tobytes()+books.tobytes())
    code.write_bytes(struct.pack('<8sIQIII',b'OPQC001\0',1,n,dim,dim,4096).ljust(4096,b'\0')+codes.tobytes())
    p=Primary(model,code); x=p.reconstruct_raw(np.arange(n))+rng.normal(scale=.1,size=(n,dim)).astype(np.float32)
    q=rng.normal(size=(20,dim)).astype(np.float32)
    write_matrix(folder/'base.fbin',x); write_matrix(folder/'metric.fbin',q[:12]); write_matrix(folder/'select.fbin',q[12:])
    return model,code,x,q


class IntegrationTests(unittest.TestCase):
    def test_malformed_codes_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp); model,codes,_,_=fixture(f)
            codes.write_bytes(codes.read_bytes()[:-1])
            with self.assertRaisesRegex(ValueError,'Truncated'): Primary(model,codes)

    def test_linear_complete_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            model,codes,x,q=fixture(Path(tmp)); p=Primary(model,codes); ids=np.arange(17)
            np.testing.assert_allclose(p.score(q[0],ids),p.reconstruct_raw(ids)@q[0],rtol=1e-6,atol=1e-6)
            np.testing.assert_allclose(x[ids]@q[0]-p.score(q[0],ids),(x[ids]-p.reconstruct_raw(ids))@q[0],rtol=1e-5,atol=1e-6)

    def test_query_support_merges_beams(self):
        p=Pools([0,0,1],[2,3,2],[0,2,5,7],[0,1,0,1,2,0,2],np.zeros(7),np.zeros(7),np.zeros(7))
        mass,neff=support(p,np.arange(3))
        self.assertAlmostEqual(neff[1],1.)
        self.assertAlmostEqual(neff[0],(5/6+1/2)**2/((5/6)**2+(1/2)**2))

    def test_all_rate_headers_and_native_scores(self):
        native=os.environ.get('QVRR_BUILD_DIR')
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp); model,codes,_,q=fixture(f,n=17)
            write_matrix(f/'ids.ibin',np.tile(np.arange(17,dtype=np.uint32),(3,1)),'<u4')
            write_matrix(f/'queries.fbin',q[:3]); rng=np.random.default_rng(75)
            for bits in (16,32,48,64,96,128):
                g=bits//8; d=((4+g-1)//g)*g; books=rng.normal(size=(g,256,d//g)).astype(np.float32)
                proj=np.eye(d,dtype=np.float32); mean=np.zeros(d,np.float32); mean[:4]=.1
                labels=rng.integers(0,256,(17,g),dtype=np.uint8)
                payload=f/f'payload{bits}'; payload.write_bytes(labels.tobytes()); out=f/f'b{bits}'; out.mkdir()
                export_model(out,books,proj,mean,payload,17,bits,'1234567890abcdef'*4)
                self.assertEqual((out/'side.bin').stat().st_size,48+17*g)
                if native:
                    prefix=out/'audit'
                    subprocess.run([str(Path(native)/'roar_exp234_audit'),'--primary-kind','qvrr','--primary-model',str(model),
                        '--primary-codes',str(codes),'--graph',str(f/'unused.graph'),'--queries',str(f/'queries.fbin'),
                        '--out-prefix',str(prefix),'--audit-ids',str(f/'ids.ibin'),'--ji-model',str(out/'online.bin'),
                        '--ji-side-codes',str(out/'side.bin'),'--ji-query-space','raw'],check=True,capture_output=True)
                    rec=np.concatenate([books[j,labels[:,j]] for j in range(g)],axis=1)
                    expected=np.asarray([Primary(model,codes).score(qq,np.arange(17))+rec[:,:4]@qq+mean[:4]@qq for qq in q[:3]])
                    np.testing.assert_allclose(matrix(str(prefix)+'.scores.fbin'),expected,rtol=3e-5,atol=3e-6)
                    # The same reader must retain the old padded word layouts.
                    legacy=out/'legacy.side.bin'; h=bytearray((out/'side.bin').read_bytes()[:48])
                    stride=((bits+63)//64)*8; struct.pack_into('<I',h,8,1 if stride==8 else 2)
                    padded=np.zeros((17,stride),np.uint8); padded[:,:g]=labels
                    legacy.write_bytes(h+padded.tobytes())
                    subprocess.run([str(Path(native)/'roar_exp234_audit'),'--primary-kind','qvrr','--primary-model',str(model),
                        '--primary-codes',str(codes),'--graph',str(f/'unused.graph'),'--queries',str(f/'queries.fbin'),
                        '--out-prefix',str(out/'legacy-audit'),'--audit-ids',str(f/'ids.ibin'),'--ji-model',str(out/'online.bin'),
                        '--ji-side-codes',str(legacy),'--ji-query-space','raw'],check=True,capture_output=True)
                    np.testing.assert_array_equal(matrix(str(out/'legacy-audit')+'.scores.fbin'),matrix(str(prefix)+'.scores.fbin'))

    def test_new_protocol_and_frontier(self):
        c=load_config(); self.assertEqual(c['inversion_ranks'],[50,100,200]); self.assertEqual(c['deploy_variant'],'remora')
        self.assertEqual(frontier([dict(recall_at_10=.9,qps=100),dict(recall_at_10=.8,qps=90),dict(recall_at_10=.8,qps=110)],'qps'),[(.8,110.),(.9,100.)])

    def test_final_trainer_small_end_to_end(self):
        self.run_small_trainer('cpu')

    def test_fixed_regularization_trainers_small_end_to_end(self):
        for variant in ('uncentered-fixed','centered-fixed'):
            self.run_small_trainer('cpu',variant)

    @unittest.skipUnless(os.environ.get('FAISS_TEST_GPU'),'set FAISS_TEST_GPU for GPU trainer test')
    def test_final_trainer_selected_gpu_end_to_end(self):
        self.run_small_trainer('cuda:'+os.environ['FAISS_TEST_GPU'])

    def run_small_trainer(self,device,variant='remora'):
        with tempfile.TemporaryDirectory() as tmp:
            f=Path(tmp); model,codes,x,q=fixture(f)
            data=f/'data'; data.mkdir()
            write_matrix(data/'metric.fbin',q[:12]); write_matrix(data/'select.fbin',q[12:])
            p=Primary(model,codes); rng=np.random.default_rng(73)
            for split,qq in [('train',q[:12]),('validation',q[12:])]:
                ids=rng.integers(0,len(x),size=(len(qq),20))
                for row in ids: row[:]=rng.choice(len(x),20,replace=False)
                exact=np.array([x[ids[i]]@v for i,v in enumerate(qq)])
                ps=np.array([p.score(v,ids[i]) for i,v in enumerate(qq)])
                np.savez(f/(split+'.npz'),query_ids=np.arange(len(qq)),beams=np.full(len(qq),20),
                    offsets=np.arange(len(qq)+1)*20,node_ids=ids.ravel(),targets=(exact-ps).ravel(),
                    exact_scores=exact.ravel(),primary_scores=ps.ravel())
            c=load_config(); c.update(base=str(f/'base.fbin'),output_root=str(f),residual_sample_size=512,ji_iterations=1,
                device=device,kmeans_device='auto',geometric_encoder='faiss',geometric_encoder_device='auto',
                trainer_threads=1,centered_epochs=1,recode_nodes=20,encode_batch_size=256,ji_batch_size=64)
            config=f/'config.json'; config.write_text(json.dumps(c)); out=f/'trained'
            train(config,'fixture',model,codes,f/'train.npz',f/'validation.npz',16,variant,out)
            result=json.loads((out/'training.json').read_text())
            self.assertFalse(result['test_queries_used']); self.assertTrue(result['full_score_residual'])
            self.assertEqual(result['support_regularization'],variant=='remora'); self.assertEqual(result['side_bytes'],48+512*2)
            self.assertEqual(result['centered'],variant!='uncentered-fixed')
            if variant!='remora':
                self.assertEqual(result['regularization'],'fixed')
                self.assertGreater(result['fixed_regularization_strength'],0)
            self.assertEqual(result['kmeans_device'],device)
            self.assertEqual(result['geometric_encoder_device'],device)
            self.assertEqual(result['encoding_audit']['faiss_canonical_mismatches'],0)


if __name__=='__main__': unittest.main()
