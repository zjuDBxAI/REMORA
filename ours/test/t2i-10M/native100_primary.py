"""Native100 primary decoder using the frozen construction adapters.

The persistent offline reader streams decoded batches. Only a D x scorer-D
query map is shared; raw reconstructions are its transpose applied to decoded
vectors, never a pseudoinverse. Actual score linearity must pass the smoke audit.
"""
import json
import os
from pathlib import Path
import struct
import subprocess
import tempfile

import numpy as np


def is_native(model, codes):
    with Path(model).open('rb') as f:
        magic = f.read(8)
    with Path(codes).open('rb') as f:
        cmagic = f.read(8)
    return magic in (b'GPRIM01\0', b'RBQROAR1') or cmagic == b'SAQCODE1'


class NativePrimary:
    def __init__(self, model, codes):
        self.model, self.code_path = Path(model), Path(codes)
        self.process = None
        self.errors = tempfile.TemporaryFile()
        with self.model.open('rb') as f:
            header = f.read(72)
        self.magic = header[:8]
        if self.magic == b'RBQROAR1':
            fields = struct.unpack('<8s4I6Q', header)
            self.kind = 'extended-rabitq'
            self.record_bytes = fields[8] + fields[9]
        elif self.magic == b'GPRIM01\0':
            fields = struct.unpack('<8sIIQ6IQ', header[:56])
            self.kind = 'generic'
            self.record_bytes = fields[8]
        else:
            with self.code_path.open('rb') as f:
                fields = struct.unpack('<8sIIQQQ3Q', f.read(64))
            if fields[0] != b'SAQCODE1':
                raise ValueError('Unknown native primary model/codes')
            self.magic = b'SAQCODE1'
            self.kind = 'saq'
            self.record_bytes = fields[4]
        reader = os.environ.get('REMORA_NATIVE_PRIMARY_READER')
        if not reader or not Path(reader).is_file():
            raise RuntimeError('Set REMORA_NATIVE_PRIMARY_READER to the compiled offline reader')
        threads = os.environ.get('REMORA_PRIMARY_DECODE_THREADS', '8')
        self.process = subprocess.Popen([reader, self.kind, str(self.model), str(self.code_path), threads],
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.errors)
        raw = self._read(24)
        magic, self.nodes, self.dim, self.scorer_dim = struct.unpack('<8sQII', raw)
        if magic != b'RMPRD01\0' or not self.nodes or not self.dim or not self.scorer_dim:
            raise ValueError('Invalid native decoder protocol header')
        self.projection = np.frombuffer(self._read(4*self.dim*self.scorer_dim), '<f4').reshape(self.dim, self.scorer_dim).copy()
        if not np.isfinite(self.projection).all():
            raise ValueError('Nonfinite native query projection')

    def _read(self, count):
        chunks = bytearray()
        while len(chunks) < count:
            part = self.process.stdout.read(count-len(chunks))
            if not part:
                self.errors.seek(0)
                raise RuntimeError('Native primary decoder failed: ' + self.errors.read().decode(errors='replace')[-2000:])
            chunks.extend(part)
        return bytes(chunks)

    def _ids(self, ids):
        values = np.asarray(ids)
        if values.ndim != 1 or not np.issubdtype(values.dtype, np.integer) or np.any(values<0) or np.any(values>=self.nodes):
            raise ValueError('Invalid native decoder node IDs')
        if len(values)>1048576:
            raise ValueError('Native decoder batch is too large')
        return np.ascontiguousarray(values, dtype='<u4')

    def reconstruct_scorer(self, ids):
        ids = self._ids(ids)
        if not len(ids):
            return np.empty((0,self.scorer_dim),np.float32)
        self.process.stdin.write(struct.pack('<II',1,len(ids)))
        self.process.stdin.write(ids.tobytes())
        self.process.stdin.flush()
        out = np.frombuffer(self._read(4*len(ids)*self.scorer_dim), '<f4').reshape(len(ids), self.scorer_dim).copy()
        if not np.isfinite(out).all():
            raise ValueError('Nonfinite native reconstruction')
        return out

    def reconstruct_raw(self, ids):
        return self.reconstruct_scorer(ids) @ self.projection.T

    def score(self, query, ids):
        q = np.asarray(query,np.float32)
        if q.shape != (self.dim,) or not np.isfinite(q).all():
            raise ValueError('Invalid native query')
        ids = self._ids(ids)
        if not len(ids):
            return np.empty(0,np.float32)
        self.process.stdin.write(struct.pack('<II',2,len(ids)))
        self.process.stdin.write(np.ascontiguousarray(q,dtype='<f4').tobytes())
        self.process.stdin.write(ids.tobytes())
        self.process.stdin.flush()
        return np.frombuffer(self._read(4*len(ids)), '<f4').copy()

    def close(self):
        if self.process is not None:
            self.process.stdin.close()
            self.process.stdout.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                self.process.wait()
            self.process = None
        if getattr(self,'errors',None):
            self.errors.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def select_native_runs(report, methods):
    data = json.loads(Path(report).read_text())
    if data.get('status') != 'files-and-sampled-base-order-verified':
        raise ValueError('Native graph inputs must pass the transfer verification first')
    selected = {}
    for name in methods:
        record = data['methods'][name]
        q = record['quantization']
        selected[name] = dict(method=name, native_format=True,
                             code_bytes_per_vector=q['physical_bytes_per_node'],
                             nominal_code_bits_per_node=q['nominal_code_bits_per_node'],
                             implementation=q['implementation'],
                             paths=dict(record['paths'],graph=record['graph']))
    return selected
