#!/usr/bin/env python3
"""Reproducible LAION subset -> full-rank OPQ/PQ -> GPU KNN -> native graph.

No raw vectors or decoded N*d files are passed to the graph builder. The GPU
KNN stage temporarily decodes frozen PQ codes into bounded GPU FlatIP shards;
it exhaustively searches the quantized geometry, not the original vectors.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import logging
import os
from pathlib import Path
import struct
import subprocess
import threading
import time

import faiss
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
MODEL_HEADER = struct.Struct('<8sIIQIIIIIIQ')
CODES_HEADER = struct.Struct('<8sIIQIIQ')


def save_json(path, obj):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w') as f:
        json.dump(obj, f, indent=2, sort_keys=True); f.write('\n'); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def save_npz(path, **arrays):
    path = Path(path); temp = path.with_name(path.name + '.tmp')
    with temp.open('wb') as f:
        np.savez(f, **arrays); f.flush(); os.fsync(f.fileno())
    temp.replace(path)


def signature(path):
    p = Path(path); s = p.stat()
    return dict(path=str(p.resolve()), bytes=s.st_size, mtime_ns=s.st_mtime_ns)


def fbin(path, dtype='<f4'):
    with Path(path).open('rb') as f: n, d = struct.unpack('<II', f.read(8))
    if Path(path).stat().st_size != 8 + n*d*np.dtype(dtype).itemsize:
        raise ValueError(f'Invalid matrix size: {path}')
    return np.memmap(path, dtype=dtype, mode='r', offset=8, shape=(n,d))


def new_matrix(path, n, d, dtype='<f4'):
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    with p.open('wb') as f:
        f.write(struct.pack('<II', n, d)); f.truncate(8+n*d*np.dtype(dtype).itemsize)
    return np.memmap(p, dtype=dtype, mode='r+', offset=8, shape=(n,d))


def write_matrix(path, values, dtype='<f4'):
    p = Path(path); tmp = p.with_name(p.name+'.tmp')
    out = new_matrix(tmp, *values.shape, dtype=dtype); out[:] = values; out.flush(); del out
    tmp.replace(p)


def duration(seconds):
    if seconds is None: return 'unknown'
    seconds = max(0, int(seconds)); h, s = divmod(seconds,3600); m,s = divmod(s,60)
    return f'{h:02d}:{m:02d}:{s:02d}'


class Progress:
    def __init__(self, label, total, initial=0):
        self.label, self.total, self.start, self.last = label, total, time.monotonic(), time.monotonic()
        self.initial=initial

    def update(self, done, note=''):
        now=time.monotonic(); elapsed=now-self.start
        estimate=elapsed*(self.total-self.initial)/(done-self.initial) if done>self.initial else None
        logging.info('[%s] %s/%s step=%s elapsed/estimated_total=%s/%s %s',
                     self.label,done,self.total,duration(now-self.last),duration(elapsed),duration(estimate),note)
        self.last=now


def normalize(x):
    x=np.array(x,dtype=np.float32,order='C',copy=True)
    if not np.isfinite(x).all(): raise ValueError('Nonfinite LAION vector')
    norms=np.linalg.norm(x,axis=1)
    if np.any(norms<=0): raise ValueError('Zero LAION vector')
    x/=norms[:,None]
    return x


def extract(root, entries, ids, output, dim, modality):
    """Sorted global IDs yield stable shard/row mapping without full shard loads."""
    p=Path(output); tmp=p.with_name(p.name+'.tmp')
    dest=new_matrix(tmp,len(ids),dim); start=0; done=0
    progress=Progress('prepare-'+modality,len(ids))
    for entry in entries:
        a=np.load(root/entry['path'],mmap_mode='r'); end=start+len(a)
        lo,hi=np.searchsorted(ids,[start,end])
        for i in range(lo,hi,65536):
            selected=ids[i:min(i+65536,hi)]-start
            dest[i:i+len(selected)]=normalize(a[selected]);done+=len(selected)
        if hi>lo: progress.update(done,f"shard={entry['shard']}")
        start=end
    if done!=len(ids): raise ValueError('Requested IDs not covered by shards')
    dest.flush(); del dest; tmp.replace(p)


def prepare(args):
    out=args.out/'data'; out.mkdir(exist_ok=True)
    manifest=json.loads((args.source/'download-manifest.json').read_text())
    if json.loads((args.source/'status.json').read_text())['state']!='complete':
        raise ValueError('Source download is not complete')
    entries={m:sorted([e for e in manifest['files'] if e['modality']==m],key=lambda e:e['shard'])
             for m in ('img_emb','text_emb')}
    if [e['shard'] for e in entries['img_emb']] != [e['shard'] for e in entries['text_emb']]:
        raise ValueError('Unpaired shard IDs')
    total=0; dim=None
    for im,tx in zip(entries['img_emb'],entries['text_emb']):
        arrays=[np.load(args.source/e['path'],mmap_mode='r') for e in (im,tx)]
        for e,a in zip((im,tx),arrays):
            if (args.source/e['path']).stat().st_size!=e['bytes'] or a.ndim!=2 or a.dtype!=np.float16:
                raise ValueError('LAION source size/dtype mismatch')
        if arrays[0].shape!=arrays[1].shape: raise ValueError('Unpaired shard shapes')
        if dim is None: dim=arrays[0].shape[1]
        if arrays[0].shape[1]!=dim:raise ValueError('Mixed dimensions')
        total+=len(arrays[0])
    if args.base_rows>total or dim%2:raise ValueError('Invalid subset size or dimension for full-rank PQ')
    split_counts=dict(metric=args.metric_queries,validation=args.validation_queries,
                      build=args.build_queries,test=args.test_queries)
    requested=sum(split_counts.values())
    if requested>args.base_rows:raise ValueError('Query splits exceed base subset size')
    stamp=out/'prepared.json'
    if stamp.exists():
        meta=json.loads(stamp.read_text())
        for name,sig in meta['files'].items():
            if signature(out/name)!=sig:raise ValueError('Prepared data changed: '+name)
        logging.info('[reuse] %s',stamp);return meta
    rng=np.random.default_rng(args.seed)
    ids=np.sort(rng.choice(total,args.base_rows,replace=False)).astype(np.int64)
    with (out/'base.source-ids.npy').open('wb') as f:np.save(f,ids)
    extract(args.source,entries['img_emb'],ids,out/'base.fbin',dim,'img_emb')
    candidates=np.sort(rng.choice(ids,min(len(ids),3*requested),replace=False))
    extract(args.source,entries['text_emb'],candidates,out/'query.candidates.fbin',dim,'text_emb')
    queries=fbin(out/'query.candidates.fbin');seen=set();unique=[]
    for i in rng.permutation(len(queries)):
        digest=hashlib.sha256(queries[i].tobytes()).digest()
        if digest in seen:continue
        seen.add(digest);unique.append(int(i))
        if len(unique)==requested:break
    if len(unique)<requested:raise ValueError('Insufficient distinct query embeddings; increase candidate oversampling')
    cursor=0;files=['base.fbin','base.source-ids.npy']
    for name,count in split_counts.items():
        selected=np.asarray(unique[cursor:cursor+count]);cursor+=count
        write_matrix(out/f'query.{name}.fbin',queries[selected])
        with (out/f'query.{name}.source-ids.npy').open('wb') as f:np.save(f,candidates[selected])
        files += [f'query.{name}.fbin',f'query.{name}.source-ids.npy']
    # Freeze and export the base training sample, independent of query labels.
    sample_ids=np.sort(rng.choice(len(ids),min(args.train_size,len(ids)),replace=False))
    write_matrix(out/'base.train.fbin',fbin(out/'base.fbin')[sample_ids]);files.append('base.train.fbin')
    meta=dict(source_rows=total,base_rows=len(ids),dim=dim,seed=args.seed,
              subset='uniform image rows without replacement across all shards; sorted source ID',
              query_sampling='unique normalized text embeddings paired with subset images; disjoint query splits',
              normalization='source FP16 -> FP32 -> L2 normalization, once',
              split_counts=split_counts,files={name:signature(out/name) for name in files})
    save_json(stamp,meta);return meta


def gpu_l2(dim,gpu):
    resources=faiss.StandardGpuResources();resources.setTempMemory(256<<20)
    config=faiss.GpuIndexFlatConfig();config.device=gpu;config.useFloat16=False
    index=faiss.GpuIndexFlatL2(resources,dim,config)
    return resources,index


def pq_layout(books):
    if books.ndim!=3:raise ValueError('PQ books must have group/codeword/coordinate axes')
    groups,words,width=books.shape
    if not groups or not width or words<2 or words>256 or words&(words-1):
        raise ValueError('PQ centroid count must be a power of two between 2 and 256')
    bits=words.bit_length()-1
    return groups,words,width,bits,(groups*bits+7)//8


def make_pq(books,bits=None):
    groups,words,width=books.shape
    if bits is None:bits=pq_layout(books)[3]
    if words!=1<<bits:raise ValueError('PQ centroid count mismatch')
    pq=faiss.ProductQuantizer(groups*width,groups,bits)
    faiss.copy_array_to_vector(np.ascontiguousarray(books).ravel(),pq.centroids)
    return pq


def train(args,meta):
    dest=args.out/'primary';dest.mkdir(exist_ok=True);path=dest/'opq-pq.npz'
    bits=args.pq_bits;words=1<<bits;dim=meta['dim'];groups=dim//2
    if path.exists() and (dest/'training.json').exists():
        summary=json.loads((dest/'training.json').read_text())
        if summary['bits_per_group']!=bits or summary['raw_dim']!=dim:
            raise ValueError('Training bitrate/dimension changed; use a new output directory')
        logging.info('[reuse] %s',path)
        with np.load(path) as m:model={k:m[k] for k in m.files}
        if model['books'].shape!=(groups,words,2) or model['rotation'].shape!=(dim,dim):
            raise ValueError('Frozen training arrays do not match requested configuration')
        return model
    x=np.array(fbin(args.out/'data/base.train.fbin'),copy=True)
    resources,assign=gpu_l2(2,args.gpus[-1])
    pq=faiss.ProductQuantizer(dim,groups,bits);pq.assign_index=assign;pq.cp.seed=args.seed & 0x7fffffff
    pq.cp.min_points_per_centroid=1
    opq=faiss.OPQMatrix(dim,groups);opq.pq=pq;opq.niter=1;opq.verbose=False
    opq.max_train_points=len(x)
    first=0
    checkpoint=dest/'rotation.checkpoint.npz'
    if checkpoint.exists():
        with np.load(checkpoint) as ck:
            if ck['books'].shape!=(groups,words,2) or ck['rotation'].shape!=(dim,dim):
                raise ValueError('OPQ checkpoint bitrate/dimension mismatch')
            faiss.copy_array_to_vector(ck['rotation'].ravel(),opq.A)
            faiss.copy_array_to_vector(ck['books'].ravel(),pq.centroids);first=int(ck['step'])
        if not 0<=first<=args.opq_iterations:raise ValueError('Invalid OPQ checkpoint step')
        pq.train_type=faiss.ProductQuantizer.Train_hot_start
    progress=Progress('opq-rotation',args.opq_iterations,first)
    for step in range(first,args.opq_iterations):
        opq.niter_pq_0=args.pq_iterations if step==0 else min(args.pq_iterations,5)
        opq.train(x)
        rotation=faiss.vector_to_array(opq.A).reshape(dim,dim).copy()
        books=faiss.vector_to_array(pq.centroids).reshape(groups,words,2).copy()
        save_npz(checkpoint,rotation=rotation,books=books,step=np.int64(step+1))
        progress.update(step+1)
    rotation=faiss.vector_to_array(opq.A).reshape(dim,dim).copy()
    # Faiss OPQ learns its rotation on centered data; fit the final PQ on the
    # actual, uncentered rotated vectors used by our native score interface.
    z=np.ascontiguousarray(x@rotation.T)
    pq=faiss.ProductQuantizer(dim,groups,bits);pq.assign_index=assign
    pq.cp.seed=args.seed & 0x7fffffff;pq.cp.niter=args.pq_iterations;pq.cp.min_points_per_centroid=1
    pq.cp.max_points_per_centroid=max(1,(len(x)+words-1)//words)
    logging.info('[pq-codebooks] fitting %d groups on GPU %d',groups,args.gpus[-1]);pq.train(z)
    books=faiss.vector_to_array(pq.centroids).reshape(groups,words,2).copy()
    err=float(np.max(np.abs(rotation.astype('float64')@rotation.astype('float64').T-np.eye(dim))))
    if err>1e-4:raise ValueError('OPQ transform is not full-rank orthogonal')
    encoded=pq.compute_codes(z[:4096]);decoded=pq.decode(encoded)
    mse=float(np.square(decoded-z[:4096]).sum(1).mean())
    save_npz(path,rotation=rotation,books=books)
    save_json(dest/'training.json',dict(method=f'Faiss OPQ with PQ{groups}x{bits} for {dim}D',
        faiss_version=faiss.__version__,faiss_compile_options=faiss.get_compile_options(),
        raw_dim=dim,scorer_dim=dim,groups=groups,bits_per_group=bits,codewords=words,
        nominal_bits=groups*bits,record_bytes=(groups*bits+7)//8,dimensionality_reduction=False,
        training_vectors=len(x),opq_iterations=args.opq_iterations,pq_iterations=args.pq_iterations,
        rotation_pq_max_training_vectors_per_group=min(len(x),1000*words),
        training_gpu=args.gpus[-1],threads=args.threads,orthogonality_max_error=err,training_sample_mse=mse))
    return dict(rotation=rotation,books=books)


def export(args,meta,model):
    dest=args.out/'primary';base=fbin(args.out/'data/base.fbin');rotation,books=model['rotation'],model['books']
    dim=meta['dim'];groups,words,width,bits,record=pq_layout(books)
    identity=hashlib.sha256(rotation.tobytes()+books.tobytes()+struct.pack('<Q',len(base))).digest()
    artifact_id=int.from_bytes(identity[:8],'little') or 1
    model_path=dest/'primary_model.bin';codes_path=dest/'primary_codes.bin'
    if (dest/'encoding.json').exists():
        metadata=json.loads((dest/'encoding.json').read_text())
        if metadata['artifact_id']!=artifact_id or codes_path.stat().st_size!=40+len(base)*record:
            raise ValueError('Existing codes do not match model')
        logging.info('[reuse] %s',codes_path);return
    tmp=model_path.with_name(model_path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(MODEL_HEADER.pack(b'GPRIM01\0',1,4,len(base),dim,dim,groups,words,record,0,artifact_id))
        f.write(rotation.astype('<f4').tobytes());f.write(np.arange(0,dim+1,2,dtype='<u4').tobytes())
        f.write(books.astype('<f4').tobytes());f.flush();os.fsync(f.fileno())
    tmp.replace(model_path)
    pq=make_pq(books);resources,assign=gpu_l2(2,args.gpus[-1]);pq.assign_index=assign
    partial=codes_path.with_name(codes_path.name+'.partial');ck=dest/'encode.checkpoint.json';first=0
    header=CODES_HEADER.pack(b'GPCOD01\0',1,40,len(base),record,0,artifact_id)
    if partial.exists() and ck.exists():
        old=json.loads(ck.read_text())
        if old['artifact_id']!=artifact_id:raise ValueError('Encoding checkpoint mismatch')
        first=old['rows']
    mode='r+b' if partial.exists() else 'w+b';progress=Progress('encode',len(base),first)
    with partial.open(mode) as f:
        f.seek(0);f.write(header);f.truncate(40+first*record);f.seek(40+first*record)
        for lo in range(first,len(base),args.batch_size):
            z=np.ascontiguousarray(base[lo:lo+args.batch_size]@rotation.T)
            codes=np.zeros((len(z),record),dtype=np.uint8)
            pq.compute_codes_with_assign_index(faiss.swig_ptr(z),faiss.swig_ptr(codes),len(z))
            if lo==0:
                # Compare with independent CPU PQ assignments. Only numerical
                # nearest-centroid ties may have different labels.
                cpu=pq.compute_codes(z[:256]);a=pq.decode(cpu);b=pq.decode(codes[:256])
                delta=np.square(b-z[:256]).sum(1)-np.square(a-z[:256]).sum(1)
                if np.max(delta)>1e-5:raise ValueError('CPU/GPU PQ encoding parity failed')
                save_json(dest/'encoding-parity.json',dict(samples=len(cpu),
                    differing_code_bytes=int(np.count_nonzero(cpu!=codes[:256])),max_excess_l2=float(np.max(delta))))
            f.write(codes.tobytes());f.flush();os.fsync(f.fileno())
            done=lo+len(z);save_json(ck,dict(rows=done,artifact_id=artifact_id));progress.update(done)
    partial.replace(codes_path)
    save_json(dest/'encoding.json',dict(rows=len(base),dim=dim,groups=groups,bits_per_group=bits,
        record_bytes=record,artifact_id=artifact_id,model=signature(model_path),codes=signature(codes_path)))


def validate_native(args):
    """Fail before the expensive GPU scan if native/packed scorer parity fails."""
    cmd=native_command(args,args.out/'knn/query-base.top100.ibin')+['--validate-only']
    result=subprocess.run(cmd,check=True,text=True,capture_output=True)
    lines=[v for v in result.stdout.splitlines() if v.startswith('{')]
    audit=json.loads(lines[-1])
    if not audit['validated'] or audit['pair_table_legacy_mismatches']:
        raise ValueError('Native score parity failed')
    save_json(args.out/'primary/native-parity.json',audit)
    logging.info('[native-parity] %s',json.dumps(audit))


def gpu_knn(args,meta,model):
    """Sharded FlatIP scans every quantized reconstruction on selected GPUs."""
    dest=args.out/'knn';dest.mkdir(exist_ok=True);path=dest/'query-base.top100.ibin'
    stamp=dest/'summary.json'
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if signature(path)!=old['ids']:raise ValueError('KNN file changed')
        logging.info('[reuse] %s',path);return path
    queries=np.ascontiguousarray(fbin(args.out/'data/query.build.fbin')@model['rotation'].T)
    n=meta['base_rows'];k=min(args.topk,n);record=pq_layout(model['books'])[4]
    codes=np.memmap(args.out/'primary/primary_codes.bin',mode='r',dtype='u1',offset=40,shape=(n,record))
    limits=np.linspace(0,n,len(args.gpus)+1,dtype=np.int64)
    # This runner is for bounded in-GPU shards, not a full 400M materialization.
    # Account for Flat's capacity growth and temporary allocations before add().
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free',
                                   '--format=csv,noheader,nounits'],text=True)
    free={int(row.split(',')[0]):int(row.split(',')[1])*(1<<20) for row in memory.strip().splitlines()}
    for slot,gpu in enumerate(args.gpus):
        required=int(limits[slot+1]-limits[slot])*meta['dim']*4*2+(1<<30)
        if required>free[gpu]:
            raise ValueError(f'GPU {gpu}: bounded Flat shard needs up to {required} bytes, '
                             f'free={free[gpu]}; use a smaller subset or an out-of-core KNN runner')
    def worker(slot,gpu):
        faiss.omp_set_num_threads(max(1,args.threads//len(args.gpus)))
        lo,hi=map(int,limits[slot:slot+2]);kk=min(k,hi-lo);pq=make_pq(model['books'])
        resources=faiss.StandardGpuResources();resources.setTempMemory(256<<20)
        cfg=faiss.GpuIndexFlatConfig();cfg.device=gpu;cfg.useFloat16=False
        index=faiss.GpuIndexFlatIP(resources,meta['dim'],cfg)
        progress=Progress(f'gpu{gpu}-load-quantized',hi-lo)
        for start in range(lo,hi,args.batch_size):
            block=pq.decode(np.ascontiguousarray(codes[start:min(start+args.batch_size,hi)]))
            index.add(block)
            if (start-lo)//args.batch_size%16==0 or start+len(block)==hi:
                progress.update(start+len(block)-lo)
        check=pq.decode(np.ascontiguousarray(codes[lo:min(lo+2048,hi)]))
        small=faiss.GpuIndexFlatIP(resources,meta['dim'],cfg);small.add(check)
        cpu=faiss.IndexFlatIP(meta['dim']);cpu.add(check)
        ds,is_=small.search(queries[:8],min(10,len(check)));dc,ic=cpu.search(queries[:8],min(10,len(check)))
        err=float(np.max(np.abs(ds-dc)))
        if err>1e-4:raise ValueError('GPU quantized-geometry score parity failed')
        save_json(dest/f'gpu{gpu}.parity.json',dict(score_max_error=err,id_mismatches=int(np.count_nonzero(is_!=ic)),
            fp16_storage=False,reference='Faiss CPU exhaustive inner product of the same decoded PQ codes'))
        del small,cpu
        ipath=dest/f'gpu{gpu}.ids.npy';spath=dest/f'gpu{gpu}.scores.npy';checkpoint=dest/f'gpu{gpu}.checkpoint.json'
        first=0;mode='w+'
        if checkpoint.exists():
            old=json.loads(checkpoint.read_text());first=old['queries'];mode='r+'
            if old['base_range']!=[lo,hi]:raise ValueError('GPU partition changed')
        ids=np.lib.format.open_memmap(ipath,mode=mode,dtype=np.int64,shape=(len(queries),kk))
        scores=np.lib.format.open_memmap(spath,mode=mode,dtype=np.float32,shape=(len(queries),kk))
        progress=Progress(f'gpu{gpu}-knn',len(queries),first)
        for start in range(first,len(queries),args.query_batch):
            d,i=index.search(queries[start:start+args.query_batch],kk)
            if (i<0).any():raise ValueError('GPU returned an invalid neighbor')
            end=start+len(i);ids[start:end]=i+lo;scores[start:end]=d;ids.flush();scores.flush()
            save_json(checkpoint,dict(queries=end,base_range=[lo,hi]));progress.update(end)
        del index,resources
        return ipath,spath
    with ThreadPoolExecutor(max_workers=len(args.gpus)) as pool:
        futures=[pool.submit(worker,slot,gpu) for slot,gpu in enumerate(args.gpus)]
        shards=[future.result() for future in futures]
    all_ids=[np.load(i,mmap_mode='r') for i,s in shards];all_scores=[np.load(s,mmap_mode='r') for i,s in shards]
    tmp=path.with_name(path.name+'.tmp');output=new_matrix(tmp,len(queries),k,'<u4')
    score_tmp=dest/'scores.fbin.tmp';sout=new_matrix(score_tmp,len(queries),k)
    for lo in range(0,len(queries),args.query_batch):
        ids=np.concatenate([v[lo:lo+args.query_batch] for v in all_ids],axis=1)
        scores=np.concatenate([v[lo:lo+args.query_batch] for v in all_scores],axis=1)
        order=np.lexsort((ids,-scores),axis=1)[:,:k]
        output[lo:lo+len(order)]=np.take_along_axis(ids,order,axis=1)
        sout[lo:lo+len(order)]=np.take_along_axis(scores,order,axis=1)
    output.flush();sout.flush();del output,sout;tmp.replace(path);score_tmp.replace(dest/'scores.fbin')
    save_json(stamp,dict(method='Faiss GPU FlatIP exhaustive scan of decoded frozen PQ codes',
        raw_base_scoring=False,gpu_float16=False,temporary_gpu_reconstructions=True,
        full_decoded_base_file=False,gpus=args.gpus,base_rows=n,queries=len(queries),topk=k,
        tie_policy='score descending, ID ascending among returned shard candidates; native GPU cutoff ties may differ',ids=signature(path)))
    return path


def native_command(args,knn):
    return [str(args.builder),'--primary-kind','opq-pq','--primary-model',str(args.out/'primary/primary_model.bin'),
        '--primary-codes',str(args.out/'primary/primary_codes.bin'),'--queries',str(args.out/'data/query.build.fbin'),
        '--query-limit',str(args.build_queries),'--knn',str(knn),'--graph',str(args.out/'graph/graph.roar.index'),
        '--summary',str(args.out/'graph/construction.json'),'--disable-ji','--topk',str(args.topk),
        '--M_pjbp',str(args.degree),'--L_pjpq',str(args.build_beam),'--threads',str(args.threads),'--self-check','512']


def build(args,knn):
    stamp=args.out/'graph/construction.json'
    if stamp.exists():
        meta=json.loads(stamp.read_text());path=args.out/'graph/graph.roar.index'
        if not path.exists() or path.stat().st_size!=meta['graph_bytes']:raise ValueError('Graph/summary mismatch')
        logging.info('[reuse] %s',stamp);return
    args.out.joinpath('graph').mkdir(exist_ok=True)
    cmd=native_command(args,knn);start=time.monotonic()
    logging.info('[native-build] threads=%d command=%s',args.threads,' '.join(cmd))
    with subprocess.Popen(['stdbuf','-oL','-eL',*cmd],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1) as proc:
        for line in proc.stdout:
            logging.info('[native-build elapsed/estimated_total=%s/unknown] %s',duration(time.monotonic()-start),line.rstrip())
        result=proc.wait()
    if result:raise RuntimeError(f'Native graph builder failed with exit code {result}')
    graph=json.loads(stamp.read_text())
    if (not graph['code_native_construction'] or graph['base_vectors_loaded']
            or graph['pair_table_bytes']==0 or graph['parity']['pair_table_legacy_mismatches']):
        raise ValueError('Native construction invariants failed')


def finalize(args):
    stamp=args.out/'graph/construction.json'
    graph=json.loads(stamp.read_text())
    save_json(args.out/'summary.json',dict(state='complete',primary='OPQ-PQ',rows=graph['nodes'],
        data=str(args.out/'data/prepared.json'),training=str(args.out/'primary/training.json'),
        encoding=str(args.out/'primary/encoding.json'),knn=str(args.out/'knn/summary.json'),
        construction=str(stamp),graph=str(args.out/'graph/graph.roar.index'),
        native_builder=signature(args.builder)))


def parser():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=ROOT/'data/laion-400M')
    p.add_argument('--out',type=Path,default=ROOT/'ours/outputs/laion-10M/opq-3b-pairtable-v1')
    p.add_argument('--builder',type=Path,default=ROOT/'ours/build-laion-opq-v1/roar_primary_ji_build')
    p.add_argument('--base-rows',type=int,default=10_000_000)
    p.add_argument('--build-queries',type=int,default=100000)
    p.add_argument('--metric-queries',type=int,default=10000)
    p.add_argument('--validation-queries',type=int,default=10000)
    p.add_argument('--test-queries',type=int,default=10000)
    p.add_argument('--train-size',type=int,default=100000)
    p.add_argument('--opq-iterations',type=int,default=20)
    p.add_argument('--pq-iterations',type=int,default=20)
    p.add_argument('--pq-bits',type=int,choices=[6,8],default=6,
                   help='Bits per 2D PQ group: 6 = average 3 bit/dim; 8 = average 4 bit/dim')
    p.add_argument('--threads',type=int,default=len(os.sched_getaffinity(0)))
    p.add_argument('--gpus',default='0,1,2')
    p.add_argument('--batch-size',type=int,default=65536)
    p.add_argument('--query-batch',type=int,default=256)
    p.add_argument('--topk',type=int,default=100)
    p.add_argument('--degree',type=int,default=35)
    p.add_argument('--build-beam',type=int,default=500)
    p.add_argument('--seed',type=int,default=20261005)
    p.add_argument('--stage',choices=['all','prepare','train','encode','knn','build'],default='all')
    return p


def main():
    args=parser().parse_args();args.source=args.source.resolve();args.out=args.out.resolve();args.builder=args.builder.resolve()
    args.gpus=[int(v) for v in args.gpus.split(',')]
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0 or max(args.gpus)>=faiss.get_num_gpus():
        raise ValueError('Requested Faiss GPUs unavailable')
    if min(args.base_rows,args.build_queries,args.metric_queries,args.validation_queries,args.test_queries,
           args.train_size,args.opq_iterations,args.pq_iterations,args.threads,args.batch_size,args.query_batch,args.topk)<1:
        raise ValueError('Counts must be positive')
    if args.topk>args.base_rows:raise ValueError('Top-k exceeds database')
    if args.base_rows<len(args.gpus):raise ValueError('Fewer base vectors than GPUs')
    if not args.builder.is_file():raise ValueError('Native builder is missing; compile it first')
    args.out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s UTC %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'pipeline.log')])
    logging.Formatter.converter=time.gmtime
    import fcntl
    lock=(args.out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'}
    cfg=args.out/'config.json'
    if cfg.exists() and json.loads(cfg.read_text())!=config:raise ValueError('Config changed; use a new output directory')
    save_json(cfg,config);faiss.omp_set_num_threads(args.threads)
    started=time.monotonic();save_json(args.out/'status.json',dict(state='running',stage='prepare',pid=os.getpid()))
    def stopped_after(stage):
        if args.stage!=stage:return False
        save_json(args.out/'status.json',dict(state='stage-complete',stage=stage,
                                             elapsed_seconds=time.monotonic()-started))
        return True
    try:
        meta=prepare(args)
        if stopped_after('prepare'):return
        save_json(args.out/'status.json',dict(state='running',stage='train',pid=os.getpid()))
        model=train(args,meta)
        if stopped_after('train'):return
        save_json(args.out/'status.json',dict(state='running',stage='encode',pid=os.getpid()))
        export(args,meta,model)
        validate_native(args)
        if stopped_after('encode'):return
        save_json(args.out/'status.json',dict(state='running',stage='gpu-knn',pid=os.getpid()))
        knn=gpu_knn(args,meta,model)
        if stopped_after('knn'):return
        save_json(args.out/'status.json',dict(state='running',stage='native-build',pid=os.getpid()))
        build(args,knn)
        finalize(args)
        save_json(args.out/'status.json',dict(state='complete',elapsed_seconds=time.monotonic()-started))
        logging.info('[complete] %s',args.out/'summary.json')
    except BaseException as exc:
        save_json(args.out/'status.json',dict(state='failed',error=str(exc),elapsed_seconds=time.monotonic()-started))
        logging.exception('[failed]');raise


if __name__=='__main__':main()
