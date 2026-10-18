"""Read-only budget and queue regressions; no GPU or heavy preparation."""
import argparse
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiment1 import reference_rates, prerequisite_ready, frozen_json, make_config


def reference():
    methods={}
    values=[('opq-pq',800,200,100),('qaq',800,200,100),('extended-rabitq',768,256,116),
            ('saq',768,256,216),('leanvec-ood',768,192,104)]
    for name,bits,dim,size in values:
        q=dict(nominal_code_bits_per_node=bits,scorer_dim=dim,physical_bytes_per_node=size)
        if name in ('opq-pq','qaq'):q.update(pq_groups=100,dimension_reduction=False)
        if name=='extended-rabitq':q.update(bits_per_dimension=3,quantizer_config='exact')
        if name=='saq':q['quantization_plan']=[[192,4],[64,0]]
        methods[name]=dict(quantization=q)
    return dict(raw_dim=200,methods=methods)


class Native100Tests(unittest.TestCase):
    def test_same_as_10m_is_not_false_all800(self):
        rates=reference_rates(reference())
        self.assertEqual(rates['opq-pq']['primary_bits'],800)
        self.assertEqual(rates['qaq']['scorer_dim'],200)
        self.assertEqual(rates['opq-pq']['total_nominal_bits'],864)
        self.assertEqual(rates['extended-rabitq']['total_nominal_bits'],832)
        self.assertEqual(sum(r['change_from_legacy'] for r in rates.values()),2)

    def test_old_pq96_projection_cannot_pass(self):
        value=reference();value['methods']['opq-pq']['quantization'].update(
            nominal_code_bits_per_node=768,scorer_dim=192,pq_groups=96,dimension_reduction=True)
        with self.assertRaises(ValueError):reference_rates(value)

    def test_padding_and_saq_plan_are_real_budget(self):
        for name in ('extended-rabitq','saq','leanvec-ood'):
            value=reference();value['methods'][name]['quantization']['nominal_code_bits_per_node']=800
            with self.assertRaises(ValueError):reference_rates(value)

    def test_partial_or_running_parent_never_releases(self):
        complete=dict(status='COMPLETE',experiments34_complete=True,experiment5_complete=True)
        report=dict(status='COMPLETE')
        self.assertFalse(prerequisite_ready(dict(status='RUNNING'),report,report,False))
        self.assertFalse(prerequisite_ready(complete,report,{},True,0))
        self.assertFalse(prerequisite_ready(complete,report,report,False))
        self.assertTrue(prerequisite_ready(complete,report,report,True,0))
        self.assertTrue(prerequisite_ready(complete,report,report))

    def test_failed_parent_stops_queue(self):
        with self.assertRaises(RuntimeError):prerequisite_ready(dict(status='FAILED'),{}, {})
        with self.assertRaises(RuntimeError):prerequisite_ready(dict(status='COMPLETE'),{}, {},True,1)
        with self.assertRaises(RuntimeError):prerequisite_ready(dict(status='PARTIAL'),{}, {},True,0)

    def test_cannot_mutate_previous_workflow(self):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/'workflow.json'
            frozen_json(path,dict(primary_dim=200));frozen_json(path,dict(primary_dim=200))
            with self.assertRaises(ValueError):frozen_json(path,dict(primary_dim=192))

    def test_performance_contract_is_traversal_800_target_64_side(self):
        import json
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary)
            source=root/'reference.json'
            source.write_text(json.dumps(dict(config=dict(beams=[50,100,200,1000,2000],
                adaptive_coarse_beams=[50,100,1000,2000],scoring_placement='terminal',
                primary_dim=192,pq_groups=96,primary_budget_bits=768))))
            args=argparse.Namespace(performance_reference=source,data=str(root/'data'),
                legacy=str(root/'legacy'),gpu=2,cold_store=str(root/'cold'))
            c=make_config(args,root,root/'package',root/'truth',root/'prepared')
            self.assertEqual((c['primary_dim'],c['pq_groups'],c['primary_budget_bits'],c['side_bits']),
                             (200,100,800,64))
            self.assertEqual(c['scoring_placement'],'traversal')
            self.assertEqual(c['search_threads'],8)
            self.assertEqual(c['repeats'],3)
            self.assertEqual(c['rerank_storage'],'direct')
            self.assertLessEqual(max(c['beams']),1000)
            self.assertLessEqual(max(c['rerank_depths']),200)
            self.assertFalse(c['include_fp32_reference'])


if __name__=='__main__':unittest.main()
