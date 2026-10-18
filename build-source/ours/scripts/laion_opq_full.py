#!/usr/bin/env python3
"""All LAION rows, streaming OPQ encoding, out-of-core exhaustive GPU Top-K.

Unlike the small-subset runner, this never writes a decoded full-base fbin.
Each GPU scans disjoint database blocks. Its cumulative Top-K uses two durable
buffers: a checkpoint never overwrites the preceding completed block's input.
Thus restarting a partially scanned block cannot introduce duplicate IDs.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
import contextlib
import hashlib
import json
import logging
import math
import os
from pathlib import Path
import shutil
import struct
import threading
import time

import faiss
import numpy as np

import laion_opq_build as core


def parallel(workers, fn):
    stop=threading.Event()
    pool=ThreadPoolExecutor(max_workers=workers)
    futures=[pool.submit(fn,slot,stop) for slot in range(workers)]
    try:
        for future in as_completed(futures):future.result()
    except BaseException:
        stop.set()
        raise
    finally:
        pool.shutdown(wait=True)


def check_stop(stop):
    if stop.is_set():raise RuntimeError('Another GPU worker stopped; checkpoint retained')


@contextlib.contextmanager
def heartbeat(label):
    done=threading.Event();started=time.monotonic()
    def beat():
        while not done.wait(30):
            logging.info('[%s active] elapsed/estimated_total=%s/unknown',label,core.duration(time.monotonic()-started))
    thread=threading.Thread(target=beat,daemon=True);thread.start()
    try:yield
    finally:done.set();thread.join()


def source_manifest(args):
    if json.loads((args.source/'status.json').read_text())['state']!='complete':
        raise ValueError('LAION download is incomplete')
    manifest=json.loads((args.source/'download-manifest.json').read_text())
    entries={m:sorted([e for e in manifest['files'] if e['modality']==m],key=lambda e:int(e['shard']))
             for m in ('img_emb','text_emb')}
    if [e['shard'] for e in entries['img_emb']]!=[e['shard'] for e in entries['text_emb']]:
        raise ValueError('Unpaired source shards')
    offset=0;dim=None
    for im,tx in zip(entries['img_emb'],entries['text_emb']):
        arrays=[]
        for entry in (im,tx):
            path=args.source/entry['path'];a=np.load(path,mmap_mode='r')
            if path.stat().st_size!=entry['bytes'] or a.ndim!=2 or a.dtype!=np.float16:
                raise ValueError('Source file size/dtype mismatch: '+str(path))
            entry.update(offset=offset,rows=len(a),signature=core.signature(path));arrays.append(a)
        if arrays[0].shape!=arrays[1].shape:raise ValueError('Unpaired source shapes')
        if dim is None:dim=arrays[0].shape[1]
        if arrays[0].shape[1]!=dim:raise ValueError('Mixed dimensions')
        offset+=len(arrays[0])
    if not dim or dim%2 or offset>=2**32:raise ValueError('Unsupported shape for full-rank PQ/native uint32 IDs')
    if args.base_rows not in (0,offset):raise ValueError('Full runner never samples the database; use --base-rows 0')
    return dict(base_rows=offset,source_rows=offset,dim=dim,entries=entries,
                identity=hashlib.sha256(json.dumps(entries,sort_keys=True).encode()).hexdigest())


def extract_cached(args,meta,modality,ids,path):
    """Stage reuse without depending on a partially written matrix."""
    stamp=path.with_name(path.name+'.json')
    identity=hashlib.sha256(meta['identity'].encode()+ids.tobytes()).hexdigest()
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if old['identity']!=identity or old['file']!=core.signature(path):raise ValueError('Sample cache mismatch')
        logging.info('[reuse] %s',path);return
    core.extract(args.source,meta['entries'][modality],ids,path,meta['dim'],modality)
    core.save_json(stamp,dict(identity=identity,file=core.signature(path)))


def prepare(args,meta):
    dest=args.out/'data'
    counts=dict(metric=args.metric_queries,validation=args.validation_queries,build=args.build_queries,test=args.test_queries)
    if args.prepared_from is not None:
        source=args.prepared_from.resolve();data=source/'data'
        if source==args.out:raise ValueError('Prepared source must be a different output directory')
        previous=json.loads((source/'config.json').read_text())
        for key in ('seed','train_size','metric_queries','validation_queries','build_queries','test_queries'):
            if previous[key]!=getattr(args,key):raise ValueError('Prepared source configuration mismatch: '+key)
        old=json.loads((data/'prepared.json').read_text())
        if old['source_identity']!=meta['identity'] or old['query_splits']!=counts:
            raise ValueError('Prepared source identity/splits mismatch')
        if old['base_rows']!=meta['base_rows'] or old['dim']!=meta['dim'] or old['subset'] is not False:
            raise ValueError('Prepared source database shape mismatch')
        for name,sig in old['files'].items():
            if core.signature(data/name)!=sig:raise ValueError('Prepared source file changed: '+name)
        if core.fbin(data/'base.train.fbin').shape!=(min(args.train_size,meta['base_rows']),meta['dim']):
            raise ValueError('Prepared source training sample shape mismatch')
        if dest.exists():
            if not dest.is_symlink() or dest.resolve()!=data.resolve():
                raise ValueError('Existing data directory is not the requested prepared source')
        else:dest.symlink_to(data,target_is_directory=True)
        core.save_json(args.out/'data-reuse.json',dict(source_output=str(source),source_identity=meta['identity'],
                       files=old['files'],read_only_reuse=True))
        logging.info('[reuse-prepared] frozen samples and queries from %s',data)
    dest.mkdir(exist_ok=True);stamp=dest/'prepared.json'
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if old['source_identity']!=meta['identity'] or old['query_splits']!=counts:
            raise ValueError('LAION source/query splits changed')
        for name,sig in old['files'].items():
            if core.signature(dest/name)!=sig:raise ValueError('Prepared file changed: '+name)
        logging.info('[reuse] %s',stamp);return
    requested=sum(counts.values())
    if requested>meta['base_rows']:raise ValueError('Too many disjoint queries requested')
    rng=np.random.default_rng(args.seed)
    sample=np.sort(rng.choice(meta['base_rows'],min(args.train_size,meta['base_rows']),replace=False))
    extract_cached(args,meta,'img_emb',sample,dest/'base.train.fbin')
    # Oversampling is only for deduplicating query embeddings; all image rows
    # remain in the database. Persist source IDs for every final query split.
    candidates=np.sort(rng.choice(meta['base_rows'],min(meta['base_rows'],3*requested),replace=False))
    extract_cached(args,meta,'text_emb',candidates,dest/'query.candidates.fbin')
    query=core.fbin(dest/'query.candidates.fbin');seen=set();selected=np.empty(requested,dtype=np.int64);count=0
    progress=core.Progress('deduplicate-query-splits',requested)
    for i in rng.permutation(len(query)):
        digest=hashlib.sha256(query[i].tobytes()).digest()
        if digest in seen:continue
        seen.add(digest);selected[count]=i;count+=1
        if count%100000==0:progress.update(count)
        if count==requested:break
    if count!=requested:raise ValueError('Not enough unique normalized queries; no smaller query set substituted')
    del seen
    files=['base.train.fbin'];cursor=0
    for name,n in counts.items():
        ids=selected[cursor:cursor+n];cursor+=n
        path=dest/f'query.{name}.fbin';tmp=path.with_name(path.name+'.tmp')
        output=core.new_matrix(tmp,n,meta['dim'])
        for lo in range(0,n,args.batch_size):output[lo:lo+args.batch_size]=query[ids[lo:lo+args.batch_size]]
        output.flush();del output;tmp.replace(path)
        idpath=dest/f'query.{name}.source-ids.npy'
        with idpath.open('wb') as f:np.save(f,candidates[ids])
        files.extend([path.name,idpath.name]);logging.info('[query-split] %s=%d',name,n)
    core.save_json(stamp,dict(source_identity=meta['identity'],base_rows=meta['base_rows'],dim=meta['dim'],
        subset=False,node_id='concatenate numeric source shard order, then row order; every image retained',
        base_fp32_materialized=False,normalization='FP16 to FP32 then L2 normalization',
        query_splits=counts,query_deduplication='SHA256 of normalized FP32 embedding across all splits',
        files={name:core.signature(dest/name) for name in files}))


def export_streaming(args,meta,model):
    dest=args.out/'primary';dest.mkdir(exist_ok=True);dim=meta['dim'];n=meta['base_rows']
    rotation,books=model['rotation'],model['books'];groups,words,width,bits,record=core.pq_layout(books)
    identity=hashlib.sha256(rotation.tobytes()+books.tobytes()+meta['identity'].encode()).digest()
    artifact=int.from_bytes(identity[:8],'little') or 1
    model_path=dest/'primary_model.bin';codes_path=dest/'primary_codes.bin';partial=dest/'primary_codes.bin.partial'
    stamp=dest/'encoding.json'
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if old['artifact_id']!=artifact or old['codes']!=core.signature(codes_path):raise ValueError('Codes changed')
        logging.info('[reuse] %s',codes_path);return
    tmp=model_path.with_name(model_path.name+'.tmp')
    with tmp.open('wb') as f:
        f.write(core.MODEL_HEADER.pack(b'GPRIM01\0',1,4,n,dim,dim,groups,words,record,0,artifact))
        f.write(rotation.astype('<f4').tobytes());f.write(np.arange(0,dim+1,2,dtype='<u4').tobytes())
        f.write(books.astype('<f4').tobytes());f.flush();os.fsync(f.fileno())
    tmp.replace(model_path)
    header=core.CODES_HEADER.pack(b'GPCOD01\0',1,40,n,record,0,artifact)
    if not partial.exists() and codes_path.exists():
        # Encoding file was published just before a process interruption.
        if codes_path.stat().st_size!=40+n*record:raise ValueError('Unexpected published codes length')
        codes_path.replace(partial)
    if partial.exists():
        with partial.open('rb') as f:old=f.read(40)
        if old!=header or partial.stat().st_size!=40+n*record:raise ValueError('Encoding partial/header mismatch')
    else:
        with partial.open('wb') as f:f.write(header);f.truncate(40+n*record);f.flush();os.fsync(f.fileno())
    def worker(slot,stop):
        gpu=args.gpus[slot];faiss.omp_set_num_threads(max(1,args.threads//len(args.gpus)))
        resources,assign=core.gpu_l2(2,gpu);pq=core.make_pq(books);pq.assign_index=assign
        entries=meta['entries']['img_emb'][slot::len(args.gpus)]
        for entry in entries:
            check_stop(stop);checkpoint=dest/f'encode-shard-{entry["shard"]}.json';first=0
            if checkpoint.exists():
                old=json.loads(checkpoint.read_text())
                if old['artifact_id']!=artifact:raise ValueError('Shard encoder checkpoint mismatch')
                first=old['rows']
            if first==entry['rows']:continue
            source=np.load(args.source/entry['path'],mmap_mode='r')
            progress=core.Progress(f'gpu{gpu}-encode-shard{entry["shard"]}',len(source),first)
            with partial.open('r+b',buffering=0) as output:
                for lo in range(first,len(source),args.batch_size):
                    check_stop(stop)
                    x=core.normalize(source[lo:lo+args.batch_size]);z=np.ascontiguousarray(x@rotation.T)
                    codes=np.zeros((len(z),record),dtype=np.uint8)
                    pq.compute_codes_with_assign_index(faiss.swig_ptr(z),faiss.swig_ptr(codes),len(z))
                    if lo==0:
                        cpu=pq.compute_codes(z[:128]);a=pq.decode(cpu);b=pq.decode(codes[:128])
                        excess=float(np.max(np.square(b-z[:128]).sum(1)-np.square(a-z[:128]).sum(1)))
                        if excess>1e-5:raise ValueError('GPU encoding differs from CPU nearest-centroid assignment')
                        core.save_json(dest/f'encoding-parity-shard{entry["shard"]}.json',
                            dict(samples=len(cpu),differing_code_bytes=int(np.count_nonzero(cpu!=codes[:128])),max_excess_l2=excess))
                    output.seek(40+(entry['offset']+lo)*record);data=memoryview(codes).cast('B')
                    while data:
                        written=output.write(data)
                        if not written:raise OSError('Short code write')
                        data=data[written:]
                    os.fsync(output.fileno());end=lo+len(z)
                    core.save_json(checkpoint,dict(rows=end,artifact_id=artifact));progress.update(end)
        del pq,assign,resources
    parallel(len(args.gpus),worker)
    partial.replace(codes_path)
    core.save_json(stamp,dict(rows=n,dim=dim,groups=groups,bits_per_group=bits,record_bytes=record,
        artifact_id=artifact,source_identity=meta['identity'],base_fp32_materialized=False,
        model=core.signature(model_path),codes=core.signature(codes_path)))


def transformed_queries(args,model):
    dest=args.out/'knn';dest.mkdir(exist_ok=True);path=dest/'queries.rotated.fbin';stamp=dest/'queries.rotated.json'
    source=args.out/'data/query.build.fbin'
    identity=hashlib.sha256(model['rotation'].tobytes()+json.dumps(core.signature(source),sort_keys=True).encode()).hexdigest()
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if old['identity']!=identity or old['file']!=core.signature(path):raise ValueError('Rotated queries changed')
        return path
    queries=core.fbin(source);tmp=path.with_name(path.name+'.tmp');output=core.new_matrix(tmp,*queries.shape)
    progress=core.Progress('rotate-build-queries',len(queries))
    for lo in range(0,len(queries),args.batch_size):
        end=min(len(queries),lo+args.batch_size)
        output[lo:end]=queries[lo:end]@model['rotation'].T;progress.update(end)
    output.flush();del output;tmp.replace(path)
    core.save_json(stamp,dict(identity=identity,file=core.signature(path)));return path


def merge_topk(ids,scores,k):
    order=np.lexsort((ids,-scores),axis=1)[:,:k]
    return np.take_along_axis(ids,order,axis=1),np.take_along_axis(scores,order,axis=1)


def flush_buffers(arrays,paths):
    for array,path in zip(arrays,paths):
        array.flush()
        with path.open('rb') as f:os.fsync(f.fileno())


def blocked_knn(args,meta,model):
    dest=args.out/'knn';dest.mkdir(exist_ok=True);final=dest/'query-base.top100.ibin';stamp=dest/'summary.json'
    query_path=transformed_queries(args,model);query=core.fbin(query_path)
    n=meta['base_rows'];k=args.topk;record=core.pq_layout(model['books'])[4]
    codepath=args.out/'primary/primary_codes.bin'
    fingerprint=hashlib.sha256(json.dumps(dict(codes=core.signature(codepath),queries=core.signature(query_path),
        block_rows=args.gpu_block_rows,gpus=args.gpus,k=k,query_batch=args.query_batch),sort_keys=True).encode()).hexdigest()
    if stamp.exists():
        old=json.loads(stamp.read_text())
        if old['fingerprint']!=fingerprint or old['ids']!=core.signature(final):raise ValueError('KNN output changed')
        logging.info('[reuse] %s',final);return final
    codes=np.memmap(codepath,dtype='u1',mode='r',offset=40,shape=(n,record))
    blocks=[(lo,min(n,lo+args.gpu_block_rows)) for lo in range(0,n,args.gpu_block_rows)]
    if len(blocks)<len(args.gpus):raise ValueError('Need at least one database block per GPU')
    import subprocess
    memory=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.free','--format=csv,noheader,nounits'],text=True)
    free={int(row.split(',')[0]):int(row.split(',')[1])*(1<<20) for row in memory.strip().splitlines()}
    needed=min(n,args.gpu_block_rows)*meta['dim']*4*2+(2<<30)
    for gpu in args.gpus:
        if free[gpu]<needed:raise ValueError(f'GPU {gpu} needs conservative {needed} bytes for block, free={free[gpu]}')
    logging.info('[knn-plan] base=%d queries=%d k=%d blocks=%d block_rows=%d GPUs=%s pair_evaluations=%d',
                 n,len(query),k,len(blocks),args.gpu_block_rows,args.gpus,n*len(query))
    def worker(slot,stop):
        gpu=args.gpus[slot];faiss.omp_set_num_threads(max(1,args.threads//len(args.gpus)))
        jobs=list(range(slot,len(blocks),len(args.gpus)));checkpoint=dest/f'gpu{gpu}.state.json'
        state=dict(fingerprint=fingerprint,completed_blocks=0,queries_done=0)
        if checkpoint.exists():state=json.loads(checkpoint.read_text())
        if state['fingerprint']!=fingerprint:raise ValueError('Incompatible KNN checkpoint')
        if state['completed_blocks']>len(jobs):raise ValueError('Invalid completed block count')
        # Two fixed-size buffers per GPU, irrespective of the number of blocks.
        buffers=[];paths=[]
        for parity in range(2):
            pp=[dest/f'gpu{gpu}.buffer{parity}.ids.npy',dest/f'gpu{gpu}.buffer{parity}.scores.npy']
            arrays=[]
            for path,dtype in zip(pp,[np.uint32,np.float32]):
                if not path.exists() and (state['completed_blocks'] or state['queries_done']):
                    raise ValueError('Checkpoint references missing Top-K buffer')
                arrays.append(np.lib.format.open_memmap(path,mode='r+' if path.exists() else 'w+',
                                                       dtype=dtype,shape=(len(query),k)))
                if arrays[-1].shape!=(len(query),k) or arrays[-1].dtype!=dtype:raise ValueError('Buffer shape/dtype mismatch')
            buffers.append(arrays);paths.append(pp)
        resources=faiss.StandardGpuResources();resources.setTempMemory(512<<20)
        config=faiss.GpuIndexFlatConfig();config.device=gpu;config.useFloat16=False
        pq=core.make_pq(model['books']);total_work=sum(blocks[j][1]-blocks[j][0] for j in jobs)*len(query)
        done_work=sum(blocks[j][1]-blocks[j][0] for j in jobs[:state['completed_blocks']])*len(query)
        if state['completed_blocks']<len(jobs):
            b=blocks[jobs[state['completed_blocks']]];done_work+=(b[1]-b[0])*state['queries_done']
        progress=core.Progress(f'gpu{gpu}-knn-pairs',total_work,done_work)
        for sequence in range(state['completed_blocks'],len(jobs)):
            check_stop(stop);block_id=jobs[sequence];lo,hi=blocks[block_id];width=hi-lo
            index=faiss.GpuIndexFlatIP(resources,meta['dim'],config)
            loading=core.Progress(f'gpu{gpu}-load-block{block_id}',width)
            for start in range(lo,hi,args.batch_size):
                check_stop(stop);end=min(hi,start+args.batch_size)
                index.add(pq.decode(np.ascontiguousarray(codes[start:end])))
                if (start-lo)//args.batch_size%16==0 or end==hi:loading.update(end-lo)
            kk=min(k,width);first=state['queries_done'];last_saved=first
            out_ids,out_scores=buffers[sequence%2];previous=buffers[(sequence-1)%2]
            for start in range(first,len(query),args.query_batch):
                check_stop(stop);end=min(len(query),start+args.query_batch)
                scores,ids=index.search(np.ascontiguousarray(query[start:end]),kk)
                if (ids<0).any() or not np.isfinite(scores).all():raise ValueError('Invalid GPU Flat result')
                ids=(ids+lo).astype(np.uint32)
                if sequence:
                    ids,scores=merge_topk(np.concatenate([previous[0][start:end],ids],axis=1),
                        np.concatenate([previous[1][start:end],scores],axis=1),k)
                elif kk<k:
                    ids=np.pad(ids,((0,0),(0,k-kk)),constant_values=np.iinfo(np.uint32).max)
                    scores=np.pad(scores,((0,0),(0,k-kk)),constant_values=-np.inf)
                out_ids[start:end]=ids;out_scores[start:end]=scores
                if end-last_saved>=args.checkpoint_queries or end==len(query):
                    flush_buffers(buffers[sequence%2],paths[sequence%2])
                    state.update(queries_done=end,active_base_block=block_id)
                    core.save_json(checkpoint,state);done_work+=width*(end-last_saved);last_saved=end
                    progress.update(done_work,f'block={block_id+1}/{len(blocks)} queries={end}/{len(query)}')
                    # Fault injection is used only by the small integration test.
                    if getattr(args,'_test_interrupt_after_checkpoint',False):
                        raise RuntimeError('TEST interruption after durable query checkpoint')
            del index
            state.update(completed_blocks=sequence+1,queries_done=0,active_base_block=None)
            core.save_json(checkpoint,state)
        del resources
    parallel(len(args.gpus),worker)
    inputs=[]
    for slot,gpu in enumerate(args.gpus):
        state=json.loads((dest/f'gpu{gpu}.state.json').read_text());parity=(state['completed_blocks']-1)%2
        inputs.append([np.load(dest/f'gpu{gpu}.buffer{parity}.{name}.npy',mmap_mode='r') for name in ('ids','scores')])
    temp=final.with_name(final.name+'.tmp');ids_out=core.new_matrix(temp,len(query),k,'<u4')
    scorepath=dest/'scores.fbin';scoretemp=dest/'scores.fbin.tmp';scores_out=core.new_matrix(scoretemp,len(query),k)
    progress=core.Progress('merge-gpu-topk',len(query))
    for lo in range(0,len(query),args.batch_size):
        hi=min(len(query),lo+args.batch_size)
        ids,scores=merge_topk(np.concatenate([a[0][lo:hi] for a in inputs],axis=1),
                             np.concatenate([a[1][lo:hi] for a in inputs],axis=1),k)
        if (ids>=n).any() or not np.isfinite(scores).all():raise ValueError('Incomplete final Top-K')
        ids_out[lo:hi]=ids;scores_out[lo:hi]=scores;progress.update(hi)
    flush_buffers([ids_out,scores_out],[temp,scoretemp]);del ids_out,scores_out
    temp.replace(final);scoretemp.replace(scorepath)
    core.save_json(stamp,dict(method='out-of-core exhaustive Faiss GPU FlatIP on frozen OPQ reconstructions',
        fingerprint=fingerprint,base_rows=n,queries=len(query),topk=k,gpus=args.gpus,blocks=len(blocks),
        gpu_block_rows=args.gpu_block_rows,raw_base_scoring=False,gpu_float16=False,
        full_decoded_base_file=False,cumulative_buffer_bytes=2*len(args.gpus)*len(query)*k*8,
        tie_policy='score descending then ID among GPU-returned ties; GPU cutoff ties may choose equivalent IDs',
        ids=core.signature(final),scores=core.signature(scorepath)))
    return final


def graph_preflight(args,meta):
    # Conservative estimate, not a measured RSS. Shared codes, both adjacency
    # arrays, allocator/locks, KNN, and one reusable bitset per build thread.
    n=meta['base_rows'];record=(meta['dim']//2*args.pq_bits+7)//8
    estimate=n*(record+args.degree*4*6+48+40+32+4)+args.threads*((n+7)//8)+args.build_queries*424
    available=int(next(line.split()[1] for line in Path('/proc/meminfo').read_text().splitlines()
                       if line.startswith('MemAvailable:')))*1024
    core.save_json(args.out/'graph-memory-preflight.json',dict(estimated_bytes=estimate,available_bytes=available,
        estimate_only=True,threads=args.threads,base_rows=n,raw_base_vectors_loaded=False))
    if estimate>available*0.95:
        raise RuntimeError('KNN complete, but conservative graph memory estimate exceeds safe available RAM; '
                           'retained all artifacts; graph not started')


def parser():
    p=core.parser()
    p.description=__doc__
    p.set_defaults(base_rows=0,build_queries=10_000_000,query_batch=1024,pq_bits=8,
        out=core.ROOT/'ours/outputs/laion-400M/opq-4b-q10M-top100-v1')
    p.add_argument('--prepared-from',type=Path,
                   help='Reuse frozen prepared data from this previous full-run output; validate and symlink read-only')
    p.add_argument('--gpu-block-rows',type=int,default=4_000_000)
    p.add_argument('--checkpoint-queries',type=int,default=32768)
    return p


def main():
    args=parser().parse_args()
    for key in ('source','out','builder'):setattr(args,key,getattr(args,key).resolve())
    if args.prepared_from is not None:args.prepared_from=args.prepared_from.resolve()
    args.gpus=[int(v) for v in args.gpus.split(',')]
    if not args.gpus or len(set(args.gpus))!=len(args.gpus) or min(args.gpus)<0 or max(args.gpus)>=faiss.get_num_gpus():
        raise ValueError('Requested GPUs unavailable')
    if min(args.build_queries,args.metric_queries,args.validation_queries,args.test_queries,args.train_size,
           args.opq_iterations,args.pq_iterations,args.threads,args.batch_size,args.query_batch,args.topk,
           args.gpu_block_rows,args.checkpoint_queries)<1:raise ValueError('Counts must be positive')
    if not args.builder.is_file():raise ValueError('Native builder missing')
    args.out.mkdir(parents=True,exist_ok=True)
    logging.basicConfig(level=logging.INFO,format='%(asctime)s UTC %(message)s',
        handlers=[logging.StreamHandler(),logging.FileHandler(args.out/'pipeline.log')])
    logging.Formatter.converter=time.gmtime
    import fcntl
    lock=(args.out/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    config={k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items() if k!='stage'}
    cfg=args.out/'config.json'
    if cfg.exists():
        previous=json.loads(cfg.read_text())
        previous.setdefault('pq_bits',6);previous.setdefault('prepared_from',None)
        if previous!=config:raise ValueError('Config changed; use new output directory')
    core.save_json(cfg,config);faiss.omp_set_num_threads(args.threads);started=time.monotonic()
    def stage(name):
        core.save_json(args.out/'status.json',dict(state='running',stage=name,pid=os.getpid(),
                                                 elapsed_seconds=time.monotonic()-started))
        logging.info('[stage] %s elapsed=%s',name,core.duration(time.monotonic()-started))
    def stop_after(name):
        if args.stage!=name:return False
        core.save_json(args.out/'status.json',dict(state='stage-complete',stage=name,
                                                 elapsed_seconds=time.monotonic()-started));return True
    try:
        stage('source-validation');meta=source_manifest(args)
        if args.topk>meta['base_rows']:raise ValueError('Top-K exceeds database')
        core.save_json(args.out/'source-manifest.json',meta)
        # Includes retained query dedup staging, transformed queries, all Top-K
        # buffers, packed codes and a conservative final graph file allocation.
        needed=meta['base_rows']*((meta['dim']//2*args.pq_bits+7)//8+args.degree*8+4)+sum(
            (args.build_queries,args.metric_queries,args.validation_queries,args.test_queries))*meta['dim']*4*5
        needed+=2*len(args.gpus)*args.build_queries*args.topk*8
        if not (args.out/'primary/encoding.json').exists() and shutil.disk_usage(args.out).free<needed:
            raise ValueError(f'Insufficient disk for conservative pipeline estimate: {needed} bytes')
        logging.info('[full-scale] base=%d dim=%d build_queries=%d topk=%d CPU_threads=%d GPUs=%s',
                     meta['base_rows'],meta['dim'],args.build_queries,args.topk,args.threads,args.gpus)
        stage('prepare')
        with heartbeat('prepare'):prepare(args,meta)
        if stop_after('prepare'):return
        stage('train')
        with heartbeat('opq-training'):model=core.train(args,meta)
        if stop_after('train'):return
        stage('encode')
        with heartbeat('streaming-encode'):export_streaming(args,meta,model)
        core.validate_native(args)
        if stop_after('encode'):return
        stage('gpu-knn')
        with heartbeat('blocked-gpu-knn'):knn=blocked_knn(args,meta,model)
        if stop_after('knn'):return
        stage('native-build');graph_preflight(args,meta)
        with heartbeat('native-build'):core.build(args,knn)
        core.finalize(args)
        core.save_json(args.out/'status.json',dict(state='complete',elapsed_seconds=time.monotonic()-started))
        logging.info('[complete] %s',args.out/'summary.json')
    except BaseException as exc:
        core.save_json(args.out/'status.json',dict(state='interrupted' if isinstance(exc,KeyboardInterrupt) else 'failed',
            error=repr(exc),elapsed_seconds=time.monotonic()-started));logging.exception('[failed]');raise


if __name__=='__main__':main()
