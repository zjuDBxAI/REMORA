# Opt-in traversal scoring kernel

This source overlay is deliberate: the active `traversal-rerank-grid-b64-v1`
suite freezes binaries separately for its 1/8-thread phases and fingerprints
the original sources. Those sources and binaries must remain unchanged until
that suite finishes. Default builds still use `ours/src` and `ours/include`.

Enable this variant with:

```bash
cmake -S ours -B ours/build-traversal-fast -DCMAKE_BUILD_TYPE=Release \
  -DOURS_JI_KERNEL_ROOT="$PWD/ours/experimental/traversal-fast-v1" \
  -DRABITQ_ROOT=/home/xzl/RaBitQ-Library \
  -DSAQ_ROOT="$PWD/thirdparty/SAQ"
cmake --build ours/build-traversal-fast -j2 --target roar_primary_ji_search
```

## Changes

- AVX-512 8x8-bit specialization gathers one 64-bit word per candidate and
  extracts byte codes in registers, rather than invoking generic `CodeAt`
  inside every group/lane loop.
- `AddCorrectionsBatch` writes Primary + JI directly into score buffers. It
  preserves `primary + (bias + ordered_group_sum)`, never reassociating the
  primary score into the residual accumulation. Generic/portable fallbacks
  retain this rule, including 256-bit codes and zero-bit groups.
- `PrepareQueryReused` reuses worker-local padding, projection and LUT storage.
  Only lazy traversal builds an upper bound; eager traversal omits it.
- Worker-local neighbor, lazy, terminal, output and rerank buffers are reused
  for all optimized search placements (also the primary-only control).
- FP32 rerank selects the sorted Top-K prefix rather than fully sorting all
  reranked candidates. The existing total ordering / node-ID tie break is kept.
- Disabled component profiling no longer calls the inner scoring timer.

The visited set and candidate-pool algorithm are unchanged. There is no new
node representation, model, graph, additional node bits or per-node cache.
Primary preparation and residual LUT preparation still occur once per query;
query preparation remains included in end-to-end timings.

Search JSON exposes `search_kernel_version`, `search_workspace_reused`,
`ji_traversal_upper_bound_built`, and `ji_batch_add_inplace`. The direct kernel
is reported as `avx512-direct8x8-register-decode`.

## Verification and benchmark

```bash
bash ours/test/t2i-1M/scripts/53_benchmark_traversal_fast.sh
```

Generated tests check 192 old/new search cases: scores/IDs plus visited,
expanded and retained CSR, with primary-only/terminal/eager/lazy placement,
threads 1/2, 64/256-bit and 0-bit groups, v1/v2 models, padded queries, ties,
partial/full/no exact rerank. Low-level tests also cover register-path batch
tails, last-node accesses, repeated IDs, ordered in-place addition, invalid
IDs and repeated bound-enabled/disabled preparation.

The benchmark freezes the new executable and source snapshot. It waits for
**both** phases of `traversal-rerank-grid-b64-v1` to finish and for that suite's
lock to be released. It then uses the exact same frozen artifacts and measures
five primaries, primary-only/terminal/eager/lazy, 1/8 threads, 64-bit residual,
beam 100, finite FP32 rerank 20, one repeat, 10,000 queries. Separate single-
thread 1,000-query instrumented runs attribute costs. Old/new execution order
alternates. Use uninstrumented QPS, not the instrumented QPS, for speedups.

Outputs: `$OUTPUT_ROOT/traversal-fast-v1/report/summary.json`,
`paired-speedup.csv`, `measurements.csv`. While queued, inspect `progress.json`.
The queue does not stop the active sweep or other services. CPU affinity is
inherited from the reference experiment; all results remain shared-load.

Overrides: `TRAVERSAL_FAST_TAG`, `TRAVERSAL_FAST_REFERENCE_TAG`,
`TRAVERSAL_FAST_BUILD`, `TRAVERSAL_FAST_QUERY_LIMIT`,
`TRAVERSAL_FAST_PROFILE_QUERIES`, `TRAVERSAL_FAST_CHECK_CPUS` (default 16,17),
`BUILD_THREADS`. Use a new tag for any changed kernel/configuration.

This is an implementation candidate, not a claim of a measured speedup.
Promote it to the default only after numerical and paired performance checks.
