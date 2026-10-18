import importlib.util
from pathlib import Path
import unittest
import tempfile

import numpy as np

spec=importlib.util.spec_from_file_location('exp5',Path(__file__).parents[1]/'scripts/run_t2i10m_experiment5.py')
study=importlib.util.module_from_spec(spec)
spec.loader.exec_module(study)


class RateStudyTests(unittest.TestCase):
    def test_distortion_equals_explicit_query_score_error(self):
        rng=np.random.default_rng(21)
        error=rng.normal(size=(7,4));query=rng.normal(size=(5,4))
        value=study.weighted_distortion(error,query.T@query/len(query))
        self.assertAlmostEqual(value,float(np.square(query@error.T).mean()),places=12)

    def test_complete_decoder_includes_shared_mean_and_raw_projection(self):
        rng=np.random.default_rng(22)
        books=rng.normal(size=(2,256,3));projection=rng.normal(size=(6,6))
        labels=np.array([[2,8],[3,9]],dtype=np.uint8);mean=rng.normal(size=4)
        correction=study.decode_correction(books,projection,mean,labels,4)
        query=rng.normal(size=4);padded=np.r_[query,0,0]
        features=padded@projection
        expected=np.array([query@mean+sum(features[g*3:(g+1)*3]@books[g,a[g]]
                          for g in range(2)) for a in labels])
        np.testing.assert_allclose(correction@query,expected,rtol=1e-12,atol=1e-12)

    def test_same_sample_is_disjoint_from_fit(self):
        sample=study.evaluation_nodes(1000,100,200,7)
        fit=np.random.default_rng(7).choice(1000,100,replace=False)
        self.assertEqual(len(np.intersect1d(sample,fit)),0)
        np.testing.assert_array_equal(sample,study.evaluation_nodes(1000,100,200,7))

    def test_missing_measurement_never_filled_from_other_beam(self):
        quality=[dict(primary='opq-pq',side_bits=16,dw=.1,score_rmse_top200=.2,
                      routing_inv=.3,boundary_inv_h5=.4,boundary_inv_h10=.5,
                      side_payload_bytes=20000000,refiner_model_bytes=400000,hot_index_bytes=2100000000)]
        timings=[dict(primary='opq-pq',side_bits=16,beam=100,timing_complete=True,
                      recall_at_10=.8,qps=1000,mean_service_latency_ms=1,peak_rss_bytes=2200000000)]
        rows=study.user_table_rows('opq-pq',[100,200],[16,32],quality,timings)
        self.assertEqual(rows[0]['recall_at_10_percent'],80)
        self.assertEqual(rows[0]['side_payload_MB'],20)
        self.assertEqual(rows[0]['routing_inv_percent'],30)
        self.assertEqual(rows[0]['status'],'COMPLETE')
        self.assertEqual(rows[2]['recall_at_10_percent'],'')
        self.assertEqual(rows[2]['status'],'PENDING')
        self.assertEqual(rows[1]['DW'],'')

    def test_rate_plot_uses_only_measured_bits(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'rate.svg'
            study.rate_plot(path,[dict(primary='opq-pq',side_bits=16,DW=.1),
                                  dict(primary='opq-pq',side_bits=64,DW=.04)],
                            'DW','Measured rate study','Distortion')
            text=path.read_text()
            self.assertIn('Side bits / vector',text)
            self.assertIn('opq-pq',text)


if __name__=='__main__':
    unittest.main()
