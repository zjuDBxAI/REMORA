"""Full-record, cross-word and masked-tail checks for every QVRR layout."""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest
import numpy as np
from protocol import matrix,write_matrix
from qvrr import Primary


def packed_codes(values,bits):
    return np.packbits(((values[:,:,None]>>np.arange(bits))&1).reshape(len(values),-1),axis=1,bitorder='little')


def layout_fixture(folder,kind):
    rng=np.random.default_rng(548); n=4096; d=19
    model=folder/'model'; codes=folder/'codes'
    if kind in ('opq-pq','qaq'):
        groups=7; width=3; sd=groups*width
        projection=(rng.normal(size=(d,sd))*.1).astype('<f4')
        books=(rng.normal(size=(groups,256,width))*.1).astype('<f4')
        model.write_bytes(struct.pack('<8sIQ7I',b'OPQPQ001',1,n,d,sd,groups,256,width,0,1)+projection.tobytes()+books.tobytes())
        labels=rng.integers(0,256,(n,groups),dtype=np.uint8)
        codes.write_bytes(struct.pack('<8sIQIII',b'OPQC001\0',1,n,groups,groups,4096).ljust(4096,b'\0')+labels.tobytes())
    elif kind.startswith('rabitq'):
        bits=int(kind[-1]); sd=d; packed=(d*bits+7)//8
        projection=(rng.normal(size=(d,d))*.1).astype('<f4')
        model.write_bytes(struct.pack('<8sIQ6I',b'ERBQRG01',1,n,d,bits,packed,0,0,1)+projection.tobytes())
        labels=rng.integers(0,1<<bits,(n,d),dtype=np.uint8)
        record=np.concatenate([packed_codes(labels,bits),rng.uniform(.01,.1,(n,1)).astype('<f4').view('u1')],axis=1)
        codes.write_bytes(struct.pack('<8sIQIII',b'ERBQC001',1,n,packed,packed+4,4096).ljust(4096,b'\0')+record.tobytes())
    elif kind=='saq':
        sd=31; segments=[(0,11,5),(11,7,0),(18,13,3)]; packed=12
        projection=(rng.normal(size=(d,sd))*.1).astype('<f4')
        model.write_bytes(struct.pack('<8sIQ7I',b'SAQRG001',1,n,d,sd,0,3,0,0,1)+b''.join(struct.pack('<3I',*s) for s in segments)+projection.tobytes())
        streams=[]
        for start,width,bits in segments:
            if bits:
                z=rng.integers(0,1<<bits,(n,width),dtype=np.uint8)
                streams.append(((z[:,:,None]>>np.arange(bits))&1).reshape(n,-1))
        payload=np.packbits(np.concatenate(streams,axis=1),axis=1,bitorder='little')
        factors=rng.uniform(.01,.1,(n,3)).astype('<f4').view('u1')
        record=np.concatenate([payload,factors],axis=1)
        codes.write_bytes(struct.pack('<8sIQ5I',b'SAQC002\0',2,n,sd,3,packed,packed+12,4096).ljust(4096,b'\0')+record.tobytes())
    else:
        sd=int(kind.split('-')[-1]); projection=(rng.normal(size=(sd,d))*.1).astype('<f4')
        mean=(rng.normal(size=sd)*.1).astype('<f4')
        model.write_bytes(struct.pack('<8sIQ9I',b'LVOOD001',1,n,d,sd,sd+4,0,0,0,0,0,1)+projection.tobytes()+mean.tobytes())
        lo=rng.uniform(-.5,0,n); hi=rng.uniform(0,.5,n)
        bounds=np.column_stack([lo,hi]).astype('<f2').view('u1')
        labels=rng.integers(0,256,(n,sd),dtype=np.uint8)
        record=np.concatenate([bounds,labels],axis=1)
        codes.write_bytes(struct.pack('<8sIQ4I',b'LVC8001\0',1,n,sd,sd+4,8,4096).ljust(4096,b'\0')+record.tobytes())
    return model,codes


@unittest.skipUnless(os.environ.get('QVRR_BUILD_DIR'),'set QVRR_BUILD_DIR for native tests')
class NativeSimdTests(unittest.TestCase):
    def test_sparse_payload_uses_64bit_offsets(self):
        binary=Path(os.environ['QVRR_BUILD_DIR'])
        for kind in ('opq-pq','rabitq3','saq','leanvec-17'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                f=Path(tmp); model,codes=layout_fixture(f,kind); p=Primary(model,codes)
                stride=p.record_bytes; last=np.asarray(p.codes[-1]).tobytes()
                n=(2**31)//stride+32
                raw=bytearray(model.read_bytes()); struct.pack_into('<Q',raw,12,n); model.write_bytes(raw)
                with codes.open('r+b') as stream:
                    stream.seek(12); stream.write(struct.pack('<Q',n))
                    stream.truncate(4096+n*stride)
                    stream.seek(4096+(n-1)*stride); stream.write(last)
                q=np.random.default_rng(83).normal(size=(1,p.dim)).astype(np.float32)
                ids=np.tile(np.array([0,1,n-1],np.uint32),17)[None,:]
                write_matrix(f/'queries.fbin',q); write_matrix(f/'ids.ibin',ids,'<u4')
                report=json.loads(subprocess.check_output([str(binary/'roar_qvrr_kernel_audit'),str(model),str(codes),str(f/'queries.fbin'),str(f/'ids.ibin')],text=True))
                self.assertLess(report['max_scalar_simd_error'],2e-5)

    def test_all_layouts_last_record_and_masked_tails(self):
        binary=Path(os.environ['QVRR_BUILD_DIR'])
        for kind in ('opq-pq','qaq','rabitq3','rabitq6','saq','leanvec-1','leanvec-17','leanvec-76'):
            with self.subTest(kind=kind),tempfile.TemporaryDirectory() as tmp:
                f=Path(tmp); model,codes=layout_fixture(f,kind); p=Primary(model,codes)
                rng=np.random.default_rng(22); q=rng.normal(size=(8,p.dim)).astype(np.float32)
                ids=rng.integers(0,p.nodes,(len(q),257),dtype=np.uint32); ids[:,-1]=p.nodes-1
                write_matrix(f/'queries.fbin',q); write_matrix(f/'ids.ibin',ids,'<u4')
                report=json.loads(subprocess.check_output([str(binary/'roar_qvrr_kernel_audit'),str(model),str(codes),str(f/'queries.fbin'),str(f/'ids.ibin')],text=True))
                self.assertIn('avx512',report['kernel']); self.assertLess(report['max_scalar_simd_error'],2e-5)
                self.assertEqual(report['top10_set_different_queries'],0)
                subprocess.run([str(binary/'roar_exp234_audit'),'--primary-kind','qvrr','--primary-model',str(model),'--primary-codes',str(codes),
                    '--graph',str(f/'unused'),'--queries',str(f/'queries.fbin'),'--audit-ids',str(f/'ids.ibin'),'--disable-ji','--out-prefix',str(f/'audit')],check=True,capture_output=True)
                expected=np.array([p.score(v,ids[i]) for i,v in enumerate(q)])
                np.testing.assert_allclose(matrix(str(f/'audit')+'.scores.fbin'),expected,rtol=3e-5,atol=2e-6)


if __name__=='__main__': unittest.main()
