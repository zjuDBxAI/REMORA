"""Regression tests for single-pass, no-QPS RS5M experiments 3/4/5."""
import ast
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments345 import (frozen_json, quality_table, recall_record,
                               relative_budgets, threshold_rows)
from extend_ground_truth import preserve_top100


class AccuracyOnlyTests(unittest.TestCase):
    def test_extended_truth_preserves_authoritative_top100(self):
        old=np.arange(100,dtype=np.uint32)[None,:]
        ids=np.arange(200,dtype=np.int64)[None,:]
        scores=np.linspace(1.,0.,200,dtype=np.float32)[None,:]
        old_scores=scores[:,:100].copy()
        # Different tie ordering must not change the original recall truth.
        ids[0,0],ids[0,1]=ids[0,1],ids[0,0]
        labels,values=preserve_top100(old,old_scores,ids,scores)
        np.testing.assert_array_equal(labels[:,:100],old)
        np.testing.assert_array_equal(values[:,:100],old_scores)
        self.assertEqual(len(np.unique(labels)),200)
        ids[0,0]=999
        with self.assertRaises(ValueError): preserve_top100(old,old_scores,ids,scores)

    def test_relative_rate_matches_160_bit_default(self):
        self.assertEqual(relative_budgets(160), [80,160,240,320])
        self.assertEqual(relative_budgets(64), [32,64,96,128])
        with self.assertRaises(ValueError): relative_budgets(24)

    def test_cutoff_ties_do_not_relabel_official_truth(self):
        old=np.arange(100,dtype=np.uint32)[None,:]
        old_scores=np.full((1,100),.5,np.float32)
        ids=np.arange(50,250,dtype=np.int64)[None,:]
        scores=np.full((1,200),.5,np.float32)
        labels,values=preserve_top100(old,old_scores,ids,scores)
        np.testing.assert_array_equal(labels[:,:100],old)
        self.assertEqual(len(np.unique(labels)),200)
        old_scores[0,0]=.6
        with self.assertRaisesRegex(ValueError,'cutoff tie'):
            preserve_top100(old,old_scores,ids,scores)

    def test_fixed_depth_selection_ignores_qps_and_marks_unreachable(self):
        common=dict(primary='opq-pq',variant='remora',status='OK',rerank_depth='100')
        rows=[dict(common,beam='100',visited='1000',recall_at_10='.96',qps='1'),
              dict(common,beam='200',visited='2000',recall_at_10='.97',qps='100000'),
              dict(common,beam='50',visited='500',recall_at_10='.98',qps='10000',rerank_depth='20')]
        ours=threshold_rows(rows,['opq-pq'],100)[1]
        self.assertEqual(ours['beam_95'],100)
        self.assertFalse(ours['achieved_99'])
        self.assertEqual(ours['beam_99'],'NOT_REACHED')
        self.assertNotIn('qps',ours)

    def test_single_pass_recall_and_fail_closed_traversal(self):
        truth=np.tile(np.arange(200,dtype=np.uint32),(2,1))
        returned=np.tile(np.arange(10,dtype=np.uint32),(2,1))
        native=dict(ji_traversal_enabled=True,terminal_ji={'enabled':False},ji_query_space='raw',
                    queries=2,beam_width=100,rerank_enabled=False,base_vectors_loaded=False,
                    ji_active_groups=20,mean_visited_nodes=1000,mean_expansions=100)
        row=recall_record(native,truth,returned,'opq-pq',160,100)
        self.assertEqual(row['recall_at_10'],1.)
        self.assertEqual(row['passes'],1)
        self.assertFalse(row['qps_measured'])
        self.assertNotIn('qps',row)
        for bad in [dict(native,ji_traversal_enabled=False),dict(native,rerank_enabled=True),
                    dict(native,ji_active_groups=8),dict(native,terminal_ji={'enabled':True})]:
            with self.assertRaises(ValueError): recall_record(bad,truth,returned,'opq-pq',160,100)
        with self.assertRaises(ValueError):
            recall_record(native,truth,np.zeros((2,10),np.uint32),'opq-pq',160,100)

    def test_report_does_not_need_timer_results(self):
        quality=dict(primary='opq-pq',side_bits=160,dw=.001,score_rmse_top200=.002,
                     routing_inv=.2,expand_inv=.1,queue_inv=.3,boundary_inv_h5=.4,
                     boundary_inv_h10=.5,side_payload_bytes=2000,refiner_model_bytes=100,
                     hot_index_bytes=10000)
        recall=dict(primary='opq-pq',side_bits=160,beam=100,recall_at_10=.9,visited=1000)
        rows=quality_table(['opq-pq'],[160],[100],[quality],[recall])
        self.assertEqual(rows[1]['status'],'COMPLETE')
        self.assertEqual(rows[1]['recall_at_10_percent'],90.)
        self.assertEqual(rows[0]['status'],'PENDING')
        self.assertNotIn('qps',rows[1]);self.assertNotIn('mean_latency_ms',rows[1])

    def test_immutable_workflow_and_no_benchmark_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'study.json'
            frozen_json(path,{'passes':1});frozen_json(path,{'passes':1})
            with self.assertRaises(ValueError): frozen_json(path,{'passes':3})
        tree=ast.parse((Path(__file__).parent/'run_experiments345.py').read_text())
        for node in ast.walk(tree):
            if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
                self.assertNotEqual(node.func.attr,'timings')
        shell=(Path(__file__).parent/'run_experiments345.sh').read_text()
        self.assertNotIn('REPEATS:-3',shell)


if __name__=='__main__':
    unittest.main()
