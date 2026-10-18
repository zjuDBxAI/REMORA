"""Read the migrated QVRR formats without modifying the original artifacts.

This adapter is for QVRR's reproductions, NOT a claim of official-library parity.
All scores are inner products in the stored model's coordinate system. In
particular LeanVec's raw reconstruction includes the shared reduced-space mean.
"""
import json
from pathlib import Path
import struct

import numpy as np


class Primary:
    def __new__(cls, model, codes):
        from native100_primary import is_native, NativePrimary
        if is_native(model, codes):
            return NativePrimary(model, codes)
        return super().__new__(cls)

    def __init__(self, model, codes):
        self.model, self.code_path = Path(model), Path(codes)
        data = self.model.read_bytes()
        self.magic = data[:8]
        version, self.nodes, self.dim = struct.unpack_from('<IQI', data, 8)
        if version != 1 or not self.nodes or not self.dim:
            raise ValueError('Unsupported QVRR model header')
        self.segments = []
        if self.magic == b'OPQPQ001':
            self.scorer_dim, self.groups, count, self.width = struct.unpack_from('<4I', data, 24)
            if count != 256 or self.groups*self.width != self.scorer_dim:
                raise ValueError('Unsupported PQ layout')
            self.projection = np.frombuffer(data, '<f4', self.dim*self.scorer_dim, 48).reshape(self.dim, self.scorer_dim).copy()
            self.books = np.frombuffer(data, '<f4', self.groups*256*self.width,
                                      48+4*self.dim*self.scorer_dim).reshape(self.groups,256,self.width).copy()
            self.record_bytes = self.groups
            cmagic, cversion, n, groups, record, offset = struct.unpack('<8sIQIII', self._header(32))
            if cmagic != b'OPQC001\0' or cversion != 1 or groups != self.groups:
                raise ValueError('PQ codes/model mismatch')
            expected = 48+4*self.dim*self.scorer_dim+4*self.books.size
        elif self.magic == b'ERBQRG01':
            self.bits, self.packed = struct.unpack_from('<2I', data, 24)
            if not 1 <= self.bits <= 8 or self.packed != (self.dim*self.bits+7)//8:
                raise ValueError('Unsupported RaBitQ packing')
            self.scorer_dim = self.dim
            self.projection = np.frombuffer(data,'<f4',self.dim*self.dim,44).reshape(self.dim,self.dim).copy()
            self.record_bytes = self.packed+4
            self.segments = [(0,self.dim,self.bits,0,self.packed,1.0)]
            cmagic,cversion,n,packed,record,offset = struct.unpack('<8sIQIII',self._header(32))
            if cmagic != b'ERBQC001' or cversion != 1 or packed != self.packed:
                raise ValueError('RaBitQ codes/model mismatch')
            expected = 44+4*self.dim*self.dim
        elif self.magic == b'SAQRG001':
            self.scorer_dim = struct.unpack_from('<I',data,24)[0]
            ns = struct.unpack_from('<I',data,32)[0]
            cursor=48; packed_bits=0
            for g in range(ns):
                start,width,bits = struct.unpack_from('<3I',data,cursor); cursor+=12
                if bits > 8 or start+width > self.scorer_dim:
                    raise ValueError('Unsupported SAQ segment')
                # CAQ midpoint = (code - (2**b-1)/2) / 2**(b-1).
                self.segments.append((start,width,bits,packed_bits,g,2.0**(bits-1) if bits else 1.0))
                packed_bits += width*bits
            self.packed = (packed_bits+7)//8
            self.record_bytes = self.packed+4*ns
            self.segments = [(s,w,b,bit,self.packed+4*g,den) for s,w,b,bit,g,den in self.segments]
            self.projection=np.frombuffer(data,'<f4',self.dim*self.scorer_dim,cursor).reshape(self.dim,self.scorer_dim).copy()
            cmagic,cversion,n,padded,nseg,packed,record,offset=struct.unpack('<8sIQ5I',self._header(40))
            if cmagic != b'SAQC002\0' or cversion != 2 or (padded,nseg,packed)!=(self.scorer_dim,ns,self.packed):
                raise ValueError('SAQ codes/model mismatch')
            expected=cursor+4*self.dim*self.scorer_dim
        elif self.magic == b'LVOOD001':
            self.scorer_dim,self.record_bytes=struct.unpack_from('<2I',data,24)
            self.projection=np.frombuffer(data,'<f4',self.dim*self.scorer_dim,56).reshape(self.scorer_dim,self.dim).T.copy()
            self.mean=np.frombuffer(data,'<f4',self.scorer_dim,56+4*self.dim*self.scorer_dim).copy()
            cmagic,cversion,n,reduced,record,bits,offset=struct.unpack('<8sIQ4I',self._header(36))
            if cmagic != b'LVC8001\0' or cversion != 1 or bits != 8 or reduced != self.scorer_dim or self.record_bytes != self.scorer_dim+4:
                raise ValueError('LeanVec codes/model mismatch')
            expected=56+4*(self.dim*self.scorer_dim+self.scorer_dim)
        else:
            raise ValueError(f'Unconnected QVRR model magic {self.magic!r}')
        if len(data) != expected or n != self.nodes or record != self.record_bytes or offset != 4096:
            raise ValueError('QVRR model/codes size/header mismatch')
        if self.code_path.stat().st_size != offset+self.nodes*record:
            raise ValueError('Truncated/trailing QVRR codes')
        if not np.isfinite(self.projection).all():
            raise ValueError('Nonfinite query projection')
        self.codes=np.memmap(self.code_path,mode='r',dtype='u1',offset=offset,shape=(self.nodes,record))

    def _header(self, count):
        with self.code_path.open('rb') as f: return f.read(count)

    def reconstruct_scorer(self, ids):
        ids=np.asarray(ids,dtype=np.int64)
        if np.any(ids<0) or np.any(ids>=self.nodes): raise ValueError('Node ID out of range')
        record=np.ascontiguousarray(self.codes[ids])
        if self.magic == b'OPQPQ001':
            return np.concatenate([self.books[g,record[:,g]] for g in range(self.groups)],axis=1)
        if self.magic == b'LVOOD001':
            ends=record[:,:4].copy().view('<f2').astype(np.float32)
            if not np.isfinite(ends).all(): raise ValueError('Nonfinite LVQ bounds')
            return self.mean+ends[:,0,None]+record[:,4:]*(ends[:,1,None]-ends[:,0,None])/255.0
        out=np.zeros((len(ids),self.scorer_dim),np.float32)
        # Bit-packed uniform midpoint codes; dropped SAQ segments remain zero.
        unpacked=np.unpackbits(record[:,:self.packed],axis=1,bitorder='little')
        for start,width,bits,bit_offset,factor_offset,denominator in self.segments:
            if not bits: continue
            z=(unpacked[:,bit_offset:bit_offset+width*bits].reshape(len(ids),width,bits)*
               (1<<np.arange(bits))).sum(axis=2)
            factor=record[:,factor_offset:factor_offset+4].copy().view('<f4')[:,0]
            out[:,start:start+width]=(z-(2**bits-1)/2.0)*factor[:,None]/denominator
        if not np.isfinite(out).all(): raise ValueError('Nonfinite reconstruction')
        return out

    def reconstruct_raw(self, ids):
        # NOT an inverse/pseudoinverse: this is the vector representing the
        # actual linear score. It also exposes LeanVec's discarded components.
        return self.reconstruct_scorer(ids)@self.projection.T

    def score(self, query, ids):
        q=np.asarray(query,np.float32)
        if q.shape != (self.dim,): raise ValueError('Raw query dimension mismatch')
        return self.reconstruct_scorer(ids)@(q@self.projection)


def select_runs(root, methods):
    root=Path(root).resolve()
    if root.is_file():
        from native100_primary import select_native_runs
        return select_native_runs(root, methods)
    manifest=json.loads((root/'manifest.json').read_text())
    names={'opq-pq':'OPQ-PQ','qaq':'QAQ','leanvec-ood':'LeanVec-OOD',
           'extended-rabitq':'Extended RaBitQ-3','saq':'SAQ'}
    selected={}
    for name in methods:
        candidates=[r for r in manifest['runs'] if r['dataset']=='t2i-10M' and r['method']==names[name] and r['code_bytes_per_vector']<=80]
        if len(candidates)!=1: raise ValueError(f'Ambiguous/missing matched-budget graph: {name}: {len(candidates)}')
        r=candidates[0]
        selected[name]=dict(r,paths={k:str(root/v) for k,v in r['paths'].items()})
    return selected
