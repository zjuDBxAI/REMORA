# T2I-1M experiment 1 — matching T2I-10M native100

This entry does not deploy the historical 192D/PQ96 OPQ or QAQ files. It
re-trains those two primaries in full 200D with PQ100×8 and builds new primary
graphs at M40/L500. The other three native codecs are reused only after raw
base order and score/reconstruction parity checks. They also receive new
M40/L500 graphs, not renamed historical graphs.

| Primary | Native representation | Actual nominal bits | Refiner bits | Total nominal bits |
| --- | --- | ---: | ---: | ---: |
| OPQ-PQ | Full 200D, PQ100×8 | 800 | 64 | 864 |
| QAQ reproduction | Full 200D, PQ100×8 | 800 | 64 | 864 |
| Extended-RaBitQ | Padded 256D, 3-bit exact | 768 | 64 | 832 |
| SAQ | Official plan 192×4 + 64×0 | 768 | 64 | 832 |
| LeanVec-OOD reproduction | 192D, packed 4-bit affine | 768 | 64 | 832 |

These are the actual T2I-10M operating points. The 800-bit target is not a
claim that all five codecs have an exactly 800-bit payload. Actual physical
bytes also include native per-vector metadata and are reported separately.

Training/query splits are specific to the available T2I-1M 30k training queries:

- Primary query-aware fitting: train queries [0,10000).
- REMORA fitting: [10000,18000); candidate pools use its first 4096 queries.
- REMORA validation: [18000,20000); pools use its first 1024 queries.
- Graph construction: [20000,30000), Top-100 over all one million vectors.
- Evaluation: separate 10k test-query file, never used for fitting or selection.

GPU Faiss computes exact primary-space KNN; AVX-512 native code performs graph
construction. REMORA learns complete raw score residuals, Balanced Product,
centered candidate-pool loss and support-aware reencoding. Every REMORA branch
uses REMORA **during traversal**, not just terminal scoring.

Experiment 1 uses CPU 8 threads on CPUs 16–23, 3 timing repetitions, no-rerank
and NVMe O_DIRECT finite rerank. Each branch reranks its own scorer queue Top-L;
L≤200 and beam≤1000. The host is shared, not exclusively reserved. Performance
outputs and primaries are separate from all historical 768-bit results.

The workflow waits for RS5M experiments 3/4/5: root status, audit and rate
reports must all be COMPLETE, and its retained tmux pane must exit successfully.
A failed prerequisite stops the queue; partial reports never release it.

```bash
cd /home/xzl/REMORA
COMMAND=check bash ours/test/t2i-1M-native100/run_experiment1.sh
bash ours/test/t2i-1M-native100/run_experiment1.sh
```

Default output:
`outputs/t2i-1M/remora-native100-exp1-nvme-b64-v1/`

`status.json` records QUEUED/RUNNING/COMPLETE/FAILED. Stage logs are under
`logs/`; `performance/experiment1/summary.json` and its figure directories hold
the final experimental report. Resume by rerunning the same command; immutable
workflow/stage fingerprints reject changed input or parameter configurations.
