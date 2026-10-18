import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[2]
sys.path[:0]=[str(ROOT/'ours/test/t2i-10M'),str(ROOT/'ours/python'),str(ROOT/'ours/test/t2i-1M/python')]
from train_remora import ablation_penalty,targets_without_decoder_bias
from train_terminal_ranking_ji import Pools
from rotation_product import CenteredObjective
from representation_capacity import objective,predict,mm_reencode

spec=importlib.util.spec_from_file_location('ablation_study',ROOT/'ours/scripts/run_t2i10m_ablation.py')
study=importlib.util.module_from_spec(spec);spec.loader.exec_module(study)


class AblationTests(unittest.TestCase):
    def setUp(self):
        rng=np.random.default_rng(79)
        self.pools=Pools(np.array([0,1]),np.array([3,3]),np.array([0,3,6]),
            np.array([0,1,2,0,2,3]),rng.normal(size=6).astype('f4'),
            rng.normal(size=6).astype('f4'),rng.normal(size=6).astype('f4'))
        self.features=rng.normal(size=(2,4))
        self.books=rng.normal(size=(2,4,2))
        self.codes=rng.integers(0,4,size=(4,2),dtype=np.uint8)
        self.nodes=np.arange(4);self.targets=rng.normal(size=(4,4))
        self.penalty=rng.uniform(.1,1,size=4)
        self.rotation=np.eye(4)

    def test_fixed_controls_identical_and_match_adaptive_total_mass(self):
        mass=np.array([1.,2.,5.]);neff=np.array([1.,3.,8.])
        a,s=ablation_penalty(mass,neff,24.,'remora')
        b,t=ablation_penalty(mass,neff,24.,'uncentered-fixed')
        c,u=ablation_penalty(mass,neff,24.,'centered-fixed')
        np.testing.assert_array_equal(b,c)
        self.assertAlmostEqual(a.sum(),b.sum())
        np.testing.assert_allclose(b/mass,t)
        self.assertEqual((s,t),(t,u))
        z,_=ablation_penalty(mass,neff,24.,'balanced')
        np.testing.assert_array_equal(z,np.zeros(3))

    def test_objective_matches_explicit_scores_in_both_modes(self):
        for centered_loss in (False,True):
            engine=CenteredObjective(self.pools,self.features,self.nodes,self.targets,
                self.penalty,dtype=torch.float64,centered_loss=centered_loss)
            rec=np.concatenate([self.books[g,self.codes[:,g]] for g in range(2)],axis=1)
            events=self.features[np.repeat(self.pools.query_ids,3)]
            error=self.pools.targets-np.einsum('ed,ed->e',events,rec[self.pools.node_ids])
            errors=error.reshape(2,3)
            if centered_loss: errors=errors-errors.mean(axis=1,keepdims=True)
            terminal=np.square(errors).mean()
            geometric=np.dot(self.penalty,np.square(self.targets-rec).sum(1))/2
            value=engine.evaluate(self.books,self.rotation,self.codes)
            self.assertAlmostEqual(value['terminal'],terminal,places=12)
            self.assertAlmostEqual(value['geometry'],geometric,places=12)
            prediction=predict(self.pools,self.codes,self.features,self.books,'product')
            manual=objective(self.pools,prediction,self.codes,self.books,'product',
                self.nodes,self.targets,self.penalty,centered_loss)
            self.assertAlmostEqual(value['total'],manual['total']/2,places=6)

    def test_gradient_matches_finite_difference_in_both_modes(self):
        for centered_loss in (False,True):
            engine=CenteredObjective(self.pools,self.features,self.nodes,self.targets,
                self.penalty,dtype=torch.float64,centered_loss=centered_loss)
            _,gradient,_=engine.evaluate(self.books,self.rotation,self.codes,gradient=True)
            low=self.books.copy();high=self.books.copy();low[0,0,0]-=1e-5;high[0,0,0]+=1e-5
            numeric=(engine.evaluate(high,self.rotation,self.codes)['total']-
                     engine.evaluate(low,self.rotation,self.codes)['total'])/2e-5
            self.assertAlmostEqual(gradient[0,0,0],numeric,places=7)

    def test_uncentered_training_removes_mean_bias_exactly_once(self):
        queries=np.array([[1.,2.],[3.,4.]],dtype='f4');mean=np.array([.5,-.25],dtype='f4')
        original=self.pools.targets.copy()
        fixed=targets_without_decoder_bias(self.pools,queries,mean)
        bias=np.repeat(queries@mean,3)
        np.testing.assert_allclose(fixed.targets+bias,original,rtol=1e-6,atol=1e-7)
        np.testing.assert_array_equal(self.pools.targets,original)

    def test_code_updates_do_not_increase_corresponding_true_loss(self):
        for centered_loss in (False,True):
            codes=self.codes.copy()
            history=mm_reencode(self.pools,codes,self.features.astype('f4'),self.books.astype('f4'),
                'product',self.nodes,self.targets.astype('f4'),self.penalty,self.nodes,'cpu',
                centered_loss=centered_loss)
            for row in history:
                if row['accepted']: self.assertLessEqual(row['after']['total'],row['before']['total'])

    def test_existing_centered_default_is_unchanged(self):
        first=self.codes.copy();second=self.codes.copy()
        args=(self.pools,self.features.astype('f4'),self.books.astype('f4'),'product',
              self.nodes,self.targets.astype('f4'),self.penalty,self.nodes,'cpu')
        mm_reencode(args[0],first,*args[1:])
        mm_reencode(args[0],second,*args[1:],centered_loss=True)
        np.testing.assert_array_equal(first,second)

    def test_plan_fixed_rate_and_own_queue_points(self):
        plan=study.jobs(['a','b','c','d','e'],[100,200],[0,100],3)
        self.assertEqual(len(plan),420)
        for job in plan:
            self.assertEqual(job['side_bits'],0 if job['variant']=='primary' else 64)
            self.assertEqual(job['experiment'],2)
            self.assertLessEqual(job['rerank_depth'],job['beam'])
        with self.assertRaises(ValueError): study.jobs(['a'],[50],[100],1)

    def test_missing_observations_stay_missing(self):
        self.assertEqual(study.summarize_rows([],[],3,lambda rows,repeats:[]),[])

    def test_plot_does_not_display_incomplete_timing(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bars.svg'
            study.bar_plot(path,[dict(primary='a',variant='primary',timing_complete=False,qps=100)],
                ['a'],'qps','Test','QPS')
            self.assertFalse(path.exists())


if __name__=='__main__': unittest.main()
