"""Stream database side codes and export the native LUT model formats."""
import hashlib
import json
from pathlib import Path
import struct
import time

import numpy as np
from rotation_product import encode
from faiss_backend import FaissProductEncoder


def export_model(dest,books,projection,mean,codes_path,nodes,bits,model_id):
    groups,_,width=books.shape; d=len(projection)
    artifact=int(model_id[:16],16); mid=int(model_id[16:32],16)
    with (dest/'online.bin.tmp').open('wb') as f:
        f.write(struct.pack('<8sIIIIQQQQ',b'JIONL002',2,d,groups,bits,nodes,artifact,mid,groups*256))
        f.write(np.asarray(mean,dtype='<f4').tobytes())
        f.write(np.ascontiguousarray(projection.T,dtype='<f4').tobytes())
        for g in range(groups): f.write(struct.pack('<IIIIQQ',g*8,8,256,width,g*256*width,g*width))
        f.write(np.arange(d,dtype='<u4').tobytes()); f.write(np.asarray(books,dtype='<f4').tobytes())
    (dest/'online.bin.tmp').replace(dest/'online.bin')
    with (dest/'side.bin.tmp').open('wb') as f,Path(codes_path).open('rb') as src:
        # v3 stores byte-aligned 8-bit groups densely, without 64-bit padding.
        f.write(struct.pack('<8sIQIIIQQ',b'JIRCODE1',3,nodes,d,bits,groups,artifact,mid))
        while True:
            chunk=src.read(8<<20)
            if not chunk: break
            f.write(chunk)
    (dest/'side.bin.tmp').replace(dest/'side.bin')


def encode_database(config, dest, method, bits, variant, nodes, books, rotation,
                    final_projection, common, audit_values, targets, seen, codes,
                    device, encoder_device, encoder_backend, started):
    frozen_encoder = FaissProductEncoder(rotation,books,encoder_device,device) if encoder_backend == 'faiss' else None
    encoder_audit = frozen_encoder.audit(audit_values[:min(len(audit_values),config.get('encoding_audit_size',1024))],config['ji_batch_size']) if frozen_encoder else dict(backend='torch')
    if frozen_encoder:
        (dest/'encoding-parity.json').write_text(json.dumps(encoder_audit,indent=2,sort_keys=True)+'\n')
        print(f'[encoding-audit] Faiss/canonical mismatches={encoder_audit["faiss_canonical_mismatches"]}; Faiss/torch agreement={encoder_audit["faiss_torch_agreement"]:.9f}',flush=True)
    payload=dest/'codes.uint8.tmp'
    # Stream ALL nodes through the frozen model; no N*d reconstructed FP32 file.
    with payload.open('wb') as f:
        for lo in range(0,nodes,config['encode_batch_size']):
            hi=min(nodes,lo+config['encode_batch_size']); ids=np.arange(lo,hi,dtype=np.int64)
            target_block=targets(ids)
            labels=frozen_encoder.encode(target_block,config['ji_batch_size']) if frozen_encoder else encode(target_block,rotation,books,device,config['ji_batch_size'])
            hit=seen[(seen>=lo)&(seen<hi)]; labels[hit-lo]=codes[hit]
            labels.tofile(f)
            elapsed=time.monotonic()-started
            print(f'[encode] {method}/b{bits}/{variant} {hi}/{nodes} elapsed={elapsed:.1f}s',flush=True)
    digest_state = hashlib.sha256(books.tobytes()+final_projection.tobytes()+common.tobytes())
    with payload.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest_state.update(block)
    digest = digest_state.hexdigest()
    export_model(dest,books,final_projection,common,payload,nodes,bits,digest)
    payload.unlink()  # only this newly generated temporary payload; codes live in side.bin
    return dict(audit=encoder_audit,
                assignments=frozen_encoder.assignments if frozen_encoder else None,
                ambiguous_assignments=frozen_encoder.ambiguous_assignments if frozen_encoder else None)
