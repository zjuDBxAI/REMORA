import os
import unittest
import numpy as np
from faiss_backend import FaissProductEncoder, canonical_labels, flat_index, resolved_device, train_kmeans


class FaissBackendTests(unittest.TestCase):
    def test_cpu_kmeans_matches_existing_faiss(self):
        import faiss
        faiss.omp_set_num_threads(1)
        x=np.random.default_rng(32).normal(size=(512,9)).astype(np.float32)
        original=faiss.Kmeans(9,16,niter=3,nredo=1,seed=20,min_points_per_centroid=1)
        original.train(x)
        np.testing.assert_array_equal(train_kmeans(x,16,3,20,'cpu'),original.centroids)

    def test_faiss_all_rates_match_torch_and_reference(self):
        import faiss
        import torch
        faiss.omp_set_num_threads(1); torch.set_num_threads(1)
        rng=np.random.default_rng(812)
        for bits in (16,32,48,64,96,128):
            groups=bits//8; width=13; d=groups*width
            rotation=np.linalg.qr(rng.normal(size=(d,d)))[0].astype(np.float32)
            books=rng.normal(size=(groups,256,width)).astype(np.float32)
            values=rng.normal(size=(512,d)).astype(np.float32)
            encoder=FaissProductEncoder(rotation,books,'cpu','cpu')
            audit=encoder.audit(values,128)
            self.assertEqual(audit['faiss_canonical_mismatches'],0)
            self.assertEqual(audit['faiss_torch_agreement'],1.)
            np.testing.assert_array_equal(encoder.encode(values,17),encoder.encode(values,512))

    def test_duplicate_codewords_choose_lowest_id(self):
        books=np.zeros((2,256,3),np.float32)
        x=np.ones((27,6),np.float32)
        encoder=FaissProductEncoder(np.eye(6,dtype=np.float32),books)
        np.testing.assert_array_equal(encoder.encode(x,7),np.zeros((27,2),np.uint8))
        self.assertEqual(encoder.ambiguous_assignments,54)

    def test_empty_stream_and_bad_device(self):
        books=np.zeros((1,256,3),np.float32)
        encoder=FaissProductEncoder(np.eye(3,dtype=np.float32),books)
        self.assertEqual(encoder.encode(np.empty((0,3),np.float32)).shape,(0,1))
        with self.assertRaises(ValueError): resolved_device('cuda:','cpu')
        with self.assertRaises(ValueError): resolved_device('cuda:-1','cpu')

    @unittest.skipUnless(os.environ.get('FAISS_TEST_GPU'),'set FAISS_TEST_GPU to run selected-device tests')
    def test_gpu_kmeans_selected_device_and_encoding(self):
        import faiss
        gpu=int(os.environ['FAISS_TEST_GPU']); device=f'cuda:{gpu}'
        self.assertEqual(resolved_device('auto',device),device)
        index,resource=flat_index(13,device)
        self.assertEqual(index.getDevice(),gpu)
        rng=np.random.default_rng(36)
        x=rng.normal(size=(1024,13)).astype(np.float32)
        books=train_kmeans(x,256,2,71,device)
        self.assertEqual(books.shape,(256,13))
        encoder=FaissProductEncoder(np.eye(13,dtype=np.float32),books[None,:,:],device,device)
        audit=encoder.audit(x[:512],128)
        self.assertEqual(audit['faiss_canonical_mismatches'],0)
        self.assertEqual(audit['faiss_torch_agreement'],1.)
        zero=FaissProductEncoder(np.eye(13,dtype=np.float32),np.zeros((1,256,13),np.float32),device,device)
        np.testing.assert_array_equal(zero.encode(x[:31]),np.zeros((31,1),np.uint8))
        for bits in (16,32,48,64,96,128):
            groups=bits//8; dim=groups*13
            rotation=np.linalg.qr(rng.normal(size=(dim,dim)))[0].astype(np.float32)
            books=rng.normal(size=(groups,256,13)).astype(np.float32)
            values=rng.normal(size=(128,dim)).astype(np.float32)
            encoder=FaissProductEncoder(rotation,books,device,device)
            audit=encoder.audit(values,32)
            self.assertEqual(audit['faiss_canonical_mismatches'],0)
            self.assertEqual(audit['faiss_torch_agreement'],1.)
            np.testing.assert_array_equal(encoder.encode(values,17),encoder.encode(values,128))


if __name__=='__main__': unittest.main()
