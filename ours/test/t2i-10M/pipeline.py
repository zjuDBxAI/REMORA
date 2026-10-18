"""Reuse migrated T2I-10M graphs; train the final REMORA, never rebuild graphs."""
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import numpy as np
from protocol import ROOT,HERE,CsvLedger,file_id,identity,matrix,matrix_header,write_csv,write_matrix,truth_depth,content_sha256
from qvrr import Primary,select_runs
from scoring_contract import require_traversal,validate_command

spec=importlib.util.spec_from_file_location('_pdf_pipeline',ROOT/'ours/test/t2i-1B/pipeline.py')
legacy=importlib.util.module_from_spec(spec); spec.loader.exec_module(legacy)
process_setup=legacy.process_setup
save_json=legacy.save_json


class Pipeline(legacy.Pipeline):
    def __init__(self,c):
        require_traversal(c)
        self.c=c; self.out=Path(c['output_root']); self.out.mkdir(parents=True,exist_ok=True)
        for n in range(1,6): (self.out/f'experiment{n}').mkdir(exist_ok=True)
        self.env=legacy.child_environment(c)
        self.env['PYTHONPATH']=os.pathsep.join([str(HERE),str(ROOT/'ours/python'),str(ROOT/'ours/test/t2i-1M/python'),self.env.get('PYTHONPATH','')])
        self.stages=CsvLedger(self.out/'stages.csv',legacy.STAGE_FIELDS)
        sources=list(HERE.glob('*.py'))+list((ROOT/'ours/python').glob('*.py'))
        sources+=list((ROOT/'ours/method').glob('*.py'))
        sources+=list((ROOT/'ours/experimental/traversal-fast-v1').rglob('*.cpp'))
        sources+=list((ROOT/'ours/experimental/traversal-fast-v1').rglob('*.h'))
        sources+=list((ROOT/'ours/src').glob('*.cpp'))+list((ROOT/'ours/include').rglob('*.h'))
        sources+=[ROOT/'ours/CMakeLists.txt',ROOT/'ours/test/t2i-1B/pipeline.py',ROOT/'ours/test/t2i-1M/cpp/exp234_audit.cpp',ROOT/'ours/tests/qvrr_kernel_audit.cpp']
        sources+=[ROOT/'ours/test/t2i-1M/python'/s for s in ('rotation_product.py','representation_capacity.py')]
        self.source_id=identity({str(p.relative_to(ROOT)):file_id(p) for p in sorted(set(sources))})
        self.config_id=identity(c); self.runs=select_runs(c['graph_package'],c['methods'])
        frozen=self.out/'configuration.json'; value=dict(config_id=self.config_id,source_id=self.source_id,config=c)
        if frozen.exists() and json.loads(frozen.read_text())!=value:
            raise ValueError('Configuration/source changed; choose a new output_root')
        if not frozen.exists(): save_json(frozen,value)
        environment=[dict(component='python',value=sys.version),dict(component='executable',value=sys.executable),
                     dict(component='allowed_cpus',value=json.dumps(sorted(os.sched_getaffinity(0)))),
                     dict(component='search_cpus',value=json.dumps(c['cpus'])),
                     dict(component='timing_scope',value='shared host; CPU affinity is not exclusive resource reservation')]
        for name in ('numpy','torch','faiss-cpu'):
            try: version=importlib.metadata.version(name)
            except importlib.metadata.PackageNotFoundError: version='distribution metadata unavailable'
            environment.append(dict(component=name,value=version))
        # Distribution metadata can describe an older installed wheel, not
        # the isolated runtime selected by PYTHONPATH. Record what is loaded.
        try:
            from faiss_backend import resolved_device
            import faiss
            environment.extend([dict(component='faiss_runtime',value=faiss.__version__),
                                dict(component='faiss_runtime_path',value=faiss.__file__),
                                dict(component='faiss_compile_options',value=faiss.get_compile_options())])
        except ImportError as error:
            # Planning may run before training dependencies are installed.
            # Preflight, not construction of a plan, reports the blocker.
            environment.append(dict(component='faiss_runtime',value='unavailable: '+str(error)))
        write_csv(self.out/'environment.csv',environment)
        write_csv(self.out/'baseline-availability.csv',[dict(method=m,status='AVAILABLE',
            implementation=self.runs[m]['implementation'] if self.runs[m].get('native_format')
            else 'QVRR migrated reproduction; not official library adapter') for m in c['methods']])

    def preflight(self):
        rows=[]; okay=True
        cpuinfo=Path('/proc/cpuinfo')
        avx512=cpuinfo.exists() and 'avx512f' in cpuinfo.read_text()
        rows.append(dict(check='native AVX-512',status='OK' if avx512 else 'BLOCKED',detail='FP32 and residual native kernels; offline Faiss need not be AVX-512'))
        okay &= avx512
        for key in ('base','train_queries','test_queries'):
            try:
                n,d=matrix_header(self.c[key])
                if d!=self.c['base_dim'] or (key=='base' and n!=self.c['base_rows']): raise ValueError('Raw matrix shape mismatch')
                if key=='train_queries' and n<max(self.c[s+'_query_start']+self.c[s+'_query_limit'] for s in ('metric','select','build')): raise ValueError('Training split out of range')
                if key=='test_queries' and n<self.c['test_query_limit']: raise ValueError('Not enough test queries')
                rows.append(dict(check=key,status='OK',detail=f'{n} x {d}'))
            except (OSError,ValueError) as e: rows.append(dict(check=key,status='BLOCKED',detail=str(e))); okay=False
        for method,r in self.runs.items():
            try:
                p=Primary(r['paths']['model'],r['paths']['codes'])
                if (p.nodes,p.dim)!=(self.c['base_rows'],self.c['base_dim']): raise ValueError('Primary/base mismatch')
                with Path(r['paths']['graph']).open('rb') as f: ep,n=struct.unpack('<II',f.read(8))
                if n!=p.nodes or ep>=n: raise ValueError('Graph header mismatch')
                if not r.get('native_format') and (not Path(r['paths']['graph']+'.complete').is_file() or not Path(r['paths']['codes']+'.complete').is_file()): raise ValueError('Missing construction completion marker')
                rows.append(dict(check=method,status='OK',detail=f'reuse graph; {p.record_bytes} physical bytes/node; magic={p.magic.decode()}'))
            except (OSError,ValueError) as e: rows.append(dict(check=method,status='BLOCKED',detail=str(e))); okay=False
        if self.c['include_fp32_reference']:
            try:
                with Path(self.c['fp_graph']).open('rb') as f: ep,n=struct.unpack('<II',f.read(8))
                if n!=self.c['base_rows'] or ep>=n: raise ValueError('FP graph/base mismatch')
                rows.append(dict(check='FP graph',status='OK',detail=f'{n} nodes'))
            except (OSError,ValueError) as e:
                rows.append(dict(check='FP graph',status='BLOCKED',detail=str(e))); okay=False
        else:
            rows.append(dict(check='FP graph',status='NOT_REQUESTED',detail='FP32 search curve excluded; exact ground truth still required'))
        try:
            n,k=self._combined_header()
            if n<self.c['test_query_limit'] or k<truth_depth(self.c): raise ValueError('Combined GT too small')
            rows.append(dict(check='combined GT',status='OK',detail=f'{n} queries x {k}; {self.c["combined_ground_truth_score_kind"]}'))
        except (OSError,ValueError) as e: rows.append(dict(check='combined GT',status='BLOCKED',detail=str(e))); okay=False
        for module in ('torch','faiss'):
            found=importlib.util.find_spec(module) is not None
            rows.append(dict(check=module,status='OK' if found else 'BLOCKED',detail='training dependency')); okay &= found
        if self.c['device'].startswith('cuda'):
            import torch
            found=torch.cuda.is_available() and int(self.c['device'].split(':')[1])<torch.cuda.device_count()
            rows.append(dict(check='training device',status='OK' if found else 'BLOCKED',detail=self.c['device'])); okay &= found
        try:
            from faiss_backend import resolved_device
            for key in ('kmeans_device','geometric_encoder_device'):
                device=resolved_device(self.c[key],self.c['device'])
                rows.append(dict(check=key,status='OK',detail=device))
        except (ImportError,RuntimeError,ValueError) as e:
            rows.append(dict(check='Faiss backends',status='BLOCKED',detail=str(e))); okay=False
        rows.append(dict(check='experiment5 allocator',status='INFO',detail='Final Balanced Product; NOT old D_G(b)/DP/grouping'))
        write_csv(self.out/'preflight.csv',rows)
        return okay

    def _combined_header(self):
        path=Path(self.c['combined_ground_truth'])
        with path.open('rb') as f: n,k=struct.unpack('<II',f.read(8))
        if not n or not k or path.stat().st_size!=8+8*n*k: raise ValueError('Expected BigANN IDs followed by FP32 scores')
        return n,k

    def compile(self):
        build=Path(self.c['build_dir']); prefix=self.c['cmake_prefix_path'] or os.environ.get('CONDA_PREFIX','')
        cmd=['cmake','-S',ROOT/'ours','-B',build,'-DCMAKE_BUILD_TYPE=Release',
             '-DOURS_JI_KERNEL_ROOT='+str(ROOT/'ours/experimental/traversal-fast-v1'),
             '-DRABITQ_ROOT='+str(ROOT/'thirdparty/RaBitQ-Library'),'-DSAQ_ROOT='+str(ROOT/'thirdparty/SAQ')]
        if prefix: cmd+=['-DCMAKE_PREFIX_PATH='+prefix,'-DCMAKE_MODULE_PATH='+str(Path(prefix)/'share/glog/cmake')]
        self.stage('cmake-configure',cmd,[build/'CMakeCache.txt'],budget=0)
        targets=['roar_primary_ji_search','roar_exp234_audit','roar_qvrr_kernel_audit']
        self.stage('cmake-build',['cmake','--build',build,'--target',*targets,'-j',self.c['compile_jobs']],
                   [build/t for t in targets],[build/'CMakeCache.txt'],budget=0)

    def data(self):
        super().data()
        # Reserve data/build.fbin only for legacy compatibility; NO graph builder
        # or query-base KNN construction is ever called by this pipeline.
        from protocol import read_csv
        rows=read_csv(self.out/'query-splits.csv')
        for row in rows:
            if row['split']=='build': row['usage']='compatibility reservation only; NO graph construction; original primary graphs reused'
        write_csv(self.out/'query-splits.csv',rows)

    def ground_truth(self):
        n,k=self._combined_header(); path=Path(self.c['combined_ground_truth'])
        ids=np.memmap(path,mode='r',dtype='<u4',offset=8,shape=(n,k))
        scores=np.memmap(path,mode='r',dtype='<f4',offset=8+4*n*k,shape=(n,k))
        nq=self.c['test_query_limit']; depth=truth_depth(self.c)
        selected=np.asarray(ids[:nq,:depth]); sc=np.asarray(scores[:nq,:depth])
        if self.c['combined_ground_truth_score_kind']=='normalized-squared-l2':
            # WebVid's base is unit-normalized, while raw query rows retain
            # their original norms. Its source GT uses normalized queries:
            # dist = 2 - 2 * IP(raw_query / norm, unit_base).
            # Scaling EACH QUERY by its positive norm preserves the ID order;
            # do not modify either the queries or the frozen primary codes.
            query=matrix(self.out/'data/test.fbin')
            norms=np.sqrt(np.einsum('ij,ij->i',query,query,dtype=np.float64))
            if not np.isfinite(norms).all() or np.any(norms<=0): raise ValueError('Invalid query norms')
            sc=np.asarray((1.-.5*sc)*norms[:,None],dtype=np.float32)
        if np.any(selected>=self.c['base_rows']) or not np.isfinite(sc).all() or np.any(np.diff(sc,axis=1)>1e-6): raise ValueError('Invalid IP ground truth')
        # Check scores against the actual raw files. Provenance names in the
        # migrated package point to another machine and are not trusted paths.
        base,query=matrix(self.c['base']),matrix(self.out/'data/test.fbin')
        error=0.
        for q in np.unique(np.linspace(0,nq-1,min(nq,16)).astype(int)):
            error=max(error,float(np.max(abs(np.asarray(base[selected[q]])@query[q]-sc[q]))))
        if error>2e-5: raise ValueError(f'GT/raw score mismatch {error}; do not normalize or relabel the data')
        for suffix,values,dtype in [('ids.ibin',selected,'<u4'),('scores.fbin',sc,'<f4')]:
            out=self.out/'data'/('truth.'+suffix)
            if not out.exists() or not np.array_equal(matrix(out,dtype),values): write_matrix(out,values,dtype)
        save_json(self.out/'data/truth.metadata.json',dict(full_database=True,base_rows=self.c['base_rows'],base_dim=self.c['base_dim'],
            source=str(path),source_id=file_id(path),queries_sha256=content_sha256(self.out/'data/test.fbin'),
            score='fp32-inner-product',source_score_kind=self.c['combined_ground_truth_score_kind'],
            sampled_max_score_error=error,graph_build=False))

    def branch(self,method,variant,bits):
        if method=='full-precision':
            return dict(kind='fp32',model=self.c['base'],codes=str(self.exact_vector_path()),graph=self.c['fp_graph'],
                        ji_model='',ji_codes='',implementation='FP32 reference on existing FP graph; common IO policy')
        r=self.runs[method]; dest=self.out/'ji'/method/f'b{bits}'/variant
        return dict(kind=method if r.get('native_format') else 'qvrr',model=r['paths']['model'],codes=r['paths']['codes'],graph=r['paths']['graph'],
            ji_model=str(dest/'online.bin'),ji_codes=str(dest/'side.bin'),
            implementation=(r['implementation'] if r.get('native_format') else 'QVRR migrated reproduction') + (' + final raw-score REMORA' if bits else ''))

    def native_args(self,branch,variant):
        require_traversal(self.c)
        args=['--primary-kind',branch['kind'],'--primary-model',branch['model'],'--primary-codes',branch['codes']]
        if branch['kind']=='extended-rabitq': args+=['--rabitq-config',self.c['rabitq_config']]
        if branch['kind']=='fp32': args+=['--fp32-storage',self.c['fp32_storage']]
        if variant in ('primary','fp32'): args+=['--disable-ji']
        else:
            args+=['--ji-model',branch['ji_model'],'--ji-side-codes',branch['ji_codes'],'--ji-query-space','raw']
        validate_command(args,variant)
        return args

    def pools(self,method):
        from collect_pools import read_trace
        branch=self.branch(method,'primary',0); paths={}
        for split,nq in [('metric',self.c['pool_train_queries']),('select',self.c['pool_validation_queries'])]:
            dest=self.out/'pools'/method/split; dest.mkdir(parents=True,exist_ok=True)
            traces=[]; dependencies=[self.c['base'],self.out/'data'/f'{split}.fbin']
            for beam in self.c['pool_beams']:
                prefix=dest/f'b{beam}'; csr=Path(str(prefix)+'.retained.csr')
                cmd=[Path(self.c['build_dir'])/'roar_primary_ji_search',*self.native_args(branch,'primary'),
                    '--graph',branch['graph'],'--queries',self.out/'data'/f'{split}.fbin','--query-limit',nq,
                    '--beam',beam,'--threads',self.c['build_threads'],'--out-prefix',prefix,'--retained-csr',csr,'--packed-graph']
                self.stage(f'pool-trace-{method}-{split}-b{beam}',cmd,[csr],[branch['model'],branch['codes'],branch['graph'],self.out/'data'/f'{split}.fbin'])
                offsets,nodes=read_trace(csr); width=int(np.max(np.diff(offsets))); ids=np.empty((nq,width),np.uint32)
                for q in range(nq):
                    lo,hi=offsets[q:q+2]; ids[q]=nodes[lo]; ids[q,:hi-lo]=nodes[lo:hi]
                aid=Path(str(prefix)+'.audit.ids.ibin')
                if not aid.exists() or not np.array_equal(matrix(aid,'<u4'),ids): write_matrix(aid,ids,'<u4')
                audit=Path(str(prefix)+'.audit'); score=Path(str(audit)+'.scores.fbin')
                self.stage(f'pool-score-{method}-{split}-b{beam}',[Path(self.c['build_dir'])/'roar_exp234_audit',*self.native_args(branch,'primary'),
                    '--graph',branch['graph'],'--queries',self.out/'data'/f'{split}.fbin','--out-prefix',audit,'--audit-ids',aid],
                    [score],[aid,branch['model'],branch['codes'],self.out/'data'/f'{split}.fbin'])
                traces.append(f'{beam},{csr},{score}'); dependencies.extend([csr,score])
            output=dest/'pools.npz'; cmd=[sys.executable,HERE/'collect_pools.py','--base',self.c['base'],
                '--queries',self.out/'data'/f'{split}.fbin','--output',output]
            for trace in traces: cmd+=['--trace',trace]
            self.stage(f'pools-{method}-{split}',cmd,[output],dependencies); paths[split]=output
        return paths

    def ji(self,method,bits,pools):
        needed=set(self.c['ablation_variants'])|{self.c['deploy_variant']} if bits==self.c['side_bits'] else {self.c['deploy_variant']}
        model=self.runs[method]['paths']; config=self.out/'trainer-config.json'
        if not config.exists(): save_json(config,self.c)
        for variant in sorted(needed):
            dest=self.out/'ji'/method/f'b{bits}'/variant
            self.stage(f'remora-{method}-b{bits}-{variant}',[sys.executable,HERE/'train_remora.py','--config',config,
                '--method',method,'--model',model['model'],'--codes',model['codes'],
                '--train-pool',pools['metric'],'--validation-pool',pools['select'],
                '--bits',bits,'--variant',variant,'--output',dest],
                [dest/'online.bin',dest/'side.bin',dest/'training.json'],
                [config,model['model'],model['codes'],pools['metric'],pools['select']])
        rows=[]
        for variant in sorted(needed):
            dest=self.out/'ji'/method/f'b{bits}'/variant
            summary=json.loads((dest/'training.json').read_text())
            rows.append(dict(primary=method,budget=bits,variant=variant,groups=summary['groups'],
                allocation='8 bits/group',side_bits=bits,physical_side_payload_bytes_per_node=bits//8,
                distortion=summary['best_validation']['centered_mse'],
                validation_fixed_pool_recall=summary['best_validation']['fixed_pool_recall'],
                allocator='balanced-product; no legacy DP',training_seed=self.c['seed']))
        write_csv(self.out/'ji'/method/f'b{bits}'/'training-variants.csv',rows)

    def prepare(self,experiments=None):
        if not self.preflight(): raise RuntimeError(f'Preflight blocked; see {self.out/"preflight.csv"}')
        self.data(); self.compile(); self.ground_truth()
        if self.c['rerank_storage']=='direct':
            cold=Path(self.c['cold_store']) if self.c['cold_store'] else self.out/'data/base.cold'
            self.stage('cold-vectors',self.python('build_cold_vector_store.py',input_fbin=self.c['base'],output=cold,overwrite=True),
                       [cold],[self.c['base']])
            link=self.out/'data/base.cold'
            if cold != link:
                if link.exists() or link.is_symlink():
                    if link.resolve()!=cold.resolve(): raise ValueError('Refusing to replace another cold store')
                else: link.symlink_to(cold)
        budgets=self.c['budgets'] if experiments is None or 5 in experiments else [self.c['side_bits']]
        for method in self.c['methods']:
            paths=self.pools(method)
            for bits in budgets: self.ji(method,bits,paths)
        write_csv(self.out/'reused-graphs.csv',[dict(primary=m,graph=r['paths']['graph'],graph_id=file_id(r['paths']['graph']),
            model=r['paths']['model'],codes=r['paths']['codes'],physical_bytes_per_node=r['code_bytes_per_vector'],rebuilt=False)
            for m,r in self.runs.items()])

    def smoke(self):
        """Score parity only: does not encode 10M nodes or train a final model."""
        if not self.preflight(): raise RuntimeError('Preflight blocked')
        self.data(); self.compile(); self.ground_truth()
        base=matrix(self.c['base']); query=matrix(self.out/'data/test.fbin'); nq=min(8,len(query))
        dest=self.out/'smoke'; dest.mkdir(exist_ok=True)
        ids=np.random.default_rng(self.c['seed']).integers(0,len(base),(nq,100),dtype=np.uint32)
        write_matrix(dest/'ids.ibin',ids,'<u4'); rows=[]
        for method in self.c['methods']:
            branch=self.branch(method,'primary',0); prefix=dest/method
            self.stage('smoke-score-'+method,[Path(self.c['build_dir'])/'roar_exp234_audit',*self.native_args(branch,'primary'),
                '--graph',branch['graph'],'--queries',self.out/'data/test.fbin','--out-prefix',prefix,'--audit-ids',dest/'ids.ibin'],
                [Path(str(prefix)+'.scores.fbin')],[dest/'ids.ibin',branch['model'],branch['codes'],self.out/'data/test.fbin'])
            actual=matrix(str(prefix)+'.scores.fbin'); p=Primary(branch['model'],branch['codes'])
            expected=np.array([p.score(query[q],ids[q]) for q in range(nq)])
            error=float(np.max(abs(actual-expected)))
            raw=np.array([p.reconstruct_raw(ids[q])@query[q] for q in range(nq)])
            linear=float(np.max(abs(expected-raw)))
            if error>self.c['query_map_tolerance'] or linear>self.c['query_map_tolerance']: raise ValueError(f'{method}: score parity failed {error}/{linear}')
            rows.append(dict(primary=method,native_python_max_error=error,raw_linear_score_max_error=linear,
                physical_bytes_per_node=p.record_bytes,status='OK',official_library_parity='not claimed'))
        write_csv(dest/'score-parity.csv',rows); print(f'[smoke OK] {dest/"score-parity.csv"}',flush=True)
