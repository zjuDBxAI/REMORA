import importlib.util
from pathlib import Path
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('frozen_exp34', Path(__file__).parents[1]/'scripts/run_frozen_experiments34.py')
study = importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class FrozenAuditTests(unittest.TestCase):
    def test_coverage_decomposition(self):
        gt=np.arange(10,dtype=np.uint32).reshape(1,10)
        answer=np.array([[0,1,2,3,4,5,6,20,21,22]],dtype=np.uint32)
        visited=(np.array([0,12]),np.array([0,1,2,3,4,5,6,7,8,20,21,22]))
        retained=(np.array([0,11]),np.array([0,1,2,3,4,5,6,7,20,21,22]))
        summary=study.mean_coverage(study.coverage_rows(gt,answer,visited,retained,30))
        self.assertAlmostEqual(summary['retained_coverage'],.8)
        self.assertAlmostEqual(summary['recall_at_10'],.7)
        self.assertAlmostEqual(sum(summary[k] for k in ('missed_loss','discarded_loss','ranking_loss')),.3)

    def test_tables_preserve_unavailable_stop_and_targets(self):
        metrics=['AdjacentInv-R50','AdjacentInv-R75','AdjacentInv-R100','BoundaryInv-K10-h5',
                 'BoundaryInv-K10-h10','ExpandInv','QueueInv','StopInv']
        inv=[dict(primary='opq-pq',variant=v,metric=m,status='NOT_APPLICABLE' if m=='StopInv' else 'OK',
                  inversion_rate='' if m=='StopInv' else .25) for v in ('primary','remora') for m in metrics]
        fixed=[dict(primary='opq-pq',variant=v,target_recall=t,fixed_rerank_depth=100,
                    achieved='True' if t<.99 else 'False',beam=100,visited=2000,recall_at_10=.96)
               for v in ('primary','remora') for t in (.9,.95,.99)]
        rows, traversal=study.user_tables(['opq-pq'],inv,fixed)
        self.assertEqual(len(rows),2)
        self.assertEqual(rows[0]['Adjacent_R75'],25)
        self.assertEqual(rows[0]['StopInv'],'N/A')
        self.assertEqual(traversal[0]['beam_99'],'NOT_REACHED')
        self.assertEqual(traversal[0]['actual_recall_99'],'')
        self.assertEqual(traversal[0]['actual_recall_95'],.96)
        self.assertEqual(traversal[0]['rerank_depth'],100)

    def test_incomplete_tables_not_fabricated(self):
        self.assertEqual(study.user_tables(['saq'],[],[]),([],[]))

    def test_top100_package_and_configurable_inversion_ranks(self):
        ranks=study.inversion_ranks('100,50,75,50')
        self.assertEqual(ranks,[50,75,100])
        truth=np.arange(100,dtype=np.uint32).reshape(1,100)
        scores=np.ones((1,100),np.float32)
        query=np.ones((1,300),np.float32)
        base=np.ones((1000,300),np.float32)
        self.assertEqual(study.validate_truth_shapes(truth,scores,query,base,1,1,1000,300,ranks),100)
        with self.assertRaisesRegex(ValueError,'Top-200'):
            study.validate_truth_shapes(truth,scores,query,base,1,1,1000,300,[50,100,200])
        with self.assertRaises(ValueError): study.inversion_ranks('1,50')

    def test_fixed_depth_accuracy_selection_does_not_use_qps(self):
        common=dict(primary='opq-pq',variant='remora',status='OK',expanded='50',comparisons='999')
        rows=[dict(common,beam='100',visited='1000',recall_at_10='.96',rerank_depth='100',qps='1'),
              dict(common,beam='200',visited='2000',recall_at_10='.97',rerank_depth='100',qps='100000'),
              dict(common,beam='50',visited='500',recall_at_10='.98',rerank_depth='20',qps='100000')]
        selected=study.accuracy_matched_points(rows,100)
        self.assertEqual(selected[1]['beam'],'100')
        self.assertFalse(selected[2]['achieved'])
        self.assertEqual(selected[2]['visited'],'')
        self.assertNotIn('qps',selected[1])
        self.assertNotIn('mean_service_latency_ms',selected[1])

    def test_no_rerank_table_keeps_unreachable_targets(self):
        metrics=['AdjacentInv-R50','AdjacentInv-R75','AdjacentInv-R100','BoundaryInv-K10-h5',
                 'BoundaryInv-K10-h10','ExpandInv','QueueInv','StopInv']
        inv=[dict(primary='saq',variant=v,metric=m,status='NOT_APPLICABLE' if m=='StopInv' else 'OK',
                  inversion_rate='' if m=='StopInv' else .2) for v in ('primary','remora') for m in metrics]
        rows=[dict(primary='saq',variant=v,status='OK',beam='100',visited='2000',
                   recall_at_10='.91',rerank_depth='0',expanded='100',comparisons='1999')
              for v in ('primary','remora')]
        _,tables=study.user_tables(['saq'],inv,study.accuracy_matched_points(rows,0),0)
        self.assertEqual(tables[0]['rerank_depth'],0)
        self.assertEqual(tables[0]['beam_90'],'100')
        self.assertEqual(tables[0]['beam_95'],'NOT_REACHED')


if __name__ == '__main__':
    unittest.main()
