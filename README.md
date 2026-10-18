# REMORA

REMORA learns compact residual codes for an existing primary quantizer. It corrects query-dependent score errors to improve ranking accuracy in compressed graph search.

The method implementation is organized in [`ours/method/`](ours/method/), with [`method.py`](method.py) as the public entry point. This guide covers candidate pool preparation, Alignment, and full REMORA training. The original graph construction, Experiments 1-5, and dataset adaptation notes are preserved in [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md).

## Quick start

Run all commands from the repository root. First configure `configs/method.json` as described below:

```bash
# 1. Collect training and validation candidate pools using an existing primary and graph.
#    Skip this step if matching pools.npz files are already available.
python method.py prepare --config configs/method.json --execute

# 2. Run the existing Alignment variant, fit initial codebooks, and encode the full database.
python method.py alignment --config configs/method.json --execute

# 3. Run full REMORA: Balanced Product + centered training + support-aware regularization + encoding.
python method.py train --config configs/method.json --variant remora --execute
```

**Alignment and full REMORA training are independent method configurations.** Full training does not require running the Alignment variant first. Both reuse the prepared primary, graph, and candidate pools, and write their results to `alignment/` and `remora/`, respectively.

Omit `--execute` from any `method.py` command to print its plan without training or creating output directories:

```bash
python method.py alignment --config configs/method.json
python method.py train --config configs/method.json
python method.py --help
```

## 1. Environment setup

The Python environment requires NumPy, PyTorch, SciPy, Matplotlib, and Faiss:

```bash
python -m pip install -r requirements.txt

# Install CPU Faiss for CPU training.
python -m pip install faiss-cpu
```

GPU training requires PyTorch and GPU Faiss builds compatible with the local driver. See the [original environment instructions](docs/EXPERIMENTS.md#3-build-gpu-faiss) and [`build_faiss_runtime.sh`](ours/test/t2i-10M/build_faiss_runtime.sh) for the repository's GPU Faiss build recipe. Set the corresponding `PYTHONPATH` before using an isolated Faiss runtime.

Candidate collection, native primary decoding, and online search also require C++ programs. Build dependencies include a C++20 compiler, CMake >= 3.17, Boost program_options, OpenMP, glog, gflags, fmt, BLAS, and Eigen. System package installation commands are available in the [original environment instructions](docs/EXPERIMENTS.md#1-system-and-python-environment). Build from the repository root:

```bash
cmake -S ours -B ours/build-workflow-v1 -DCMAKE_BUILD_TYPE=Release \
  -DOURS_JI_KERNEL_ROOT="$PWD/ours/experimental/traversal-fast-v1" \
  -DRABITQ_ROOT="$PWD/thirdparty/RaBitQ-Library" \
  -DSAQ_ROOT="$PWD/thirdparty/SAQ"

cmake --build ours/build-workflow-v1 -j 8 --target \
  roar_primary_ji_search roar_exp234_audit roar_native_primary_reader

export REMORA_NATIVE_PRIMARY_READER="$PWD/ours/build-workflow-v1/roar_native_primary_reader"
```

The current native search recipe targets AVX-512 CPUs. Rebuild when moving to another machine. Existing candidate pools can be reused without collecting them again.

## 2. Inputs and configuration

| Input | Purpose |
|---|---|
| `base.fbin` | Original database vectors for complete score residuals and full-database encoding |
| Primary model / codes | Quantization model and database codes that exactly match the existing graph |
| Primary graph | RoarGraph used to collect candidate pools through Primary traversal |
| `query.train.fbin` | Query-visible alignment, training candidate pools, and candidate-conditioned training |
| `query.validation.fbin` | Validation candidate pools and checkpoint selection |

Prepare independent training, validation, and test query splits in advance. The method entry point uses inner product scoring; test queries and test ground truth are not used for method training. An `.fbin` file contains a little-endian `uint32 rows, uint32 dim` header followed by `rows * dim` FP32 values. Database node order must match the primary codes and graph.

Copy and edit the configuration template:

```bash
mkdir -p configs
cp ours/method/config.example.json configs/method.json
```

See [`config.example.json`](ours/method/config.example.json) for the full template. **All relative paths are resolved from the repository root.** Absolute paths are also supported.

| Field | Description |
|---|---|
| `method` | `opq-pq` / `qaq` / `leanvec-ood` / `extended-rabitq` / `saq` |
| `base` | Original database `.fbin` file |
| `primary_model`, `primary_codes`, `graph` | Matching primary model, codes, and graph |
| `alignment_queries`, `validation_queries` | Independently prepared training and validation query `.fbin` files |
| `output_root` | Output root for this run, such as `outputs/method/opq-pq-b64` |
| `build_dir` | Directory containing the local native programs |
| `bits` | Residual budget: multiples of 8 from 8 to 256 bits; the T2I example uses 64 |
| `variant` | Defaults to `remora`; the `alignment` command selects the existing Alignment variant |
| `device` | `cpu` or a process-visible GPU device such as `cuda:0` |
| `kmeans_device`, `geometric_encoder_device` | `auto` follows `device`; explicit devices are also supported |
| `pool_train_queries`, `pool_validation_queries` | Number of queries to collect from the start of each file; must not exceed its row count |
| `pool_beams` | Beam sizes for candidate collection, such as `[50, 100, 200]` |
| `centered_epochs`, `recode_nodes` | Candidate-conditioned training epochs and the maximum number of nodes selected for recoding per epoch |

Append `--device cpu` for CPU execution. Both Faiss devices in the template use `auto`, so they also switch to CPU. Each product codebook contains 256 codewords, requiring at least 256 database vectors and residual training samples.

To reuse existing candidate pools, set `train_pool` and `validation_pool` to the corresponding `.npz` files and skip `prepare`. For outputs from the original workflow, use `<prepared>/data/metric.fbin`, `data/select.fbin`, and `pools/<method>/{metric,select}/pools.npz`. Query file row indices must match the query IDs in the pools. Each pool file contains `query_ids, beams, offsets, node_ids, targets, exact_scores, primary_scores`, with `targets = exact_scores - primary_scores`.

If a primary or graph is not yet available, first use [`build_graph.py`](build_graph.py) for encoding and graph construction; see the [graph construction instructions](docs/EXPERIMENTS.md#primary-encoding-and-graph-construction). The method entry point reuses an existing primary and graph.

## 3. Prepare candidate pools

```bash
python method.py prepare --config configs/method.json --execute
```

This stage runs **Primary traversal** separately on training and validation queries, collects retained candidates, and computes exact and primary scores. Default outputs:

```text
<output_root>/pools/
  train/pools.npz
  validation/pools.npz
```

Retained CSR files, scores, and logs for each beam remain in the corresponding split directory. Candidate pools provide offline supervision. During REMORA deployment, residual correction participates in traversal scoring.

## 4. Run Alignment

```bash
python method.py alignment --config configs/method.json --execute
```

Alignment corresponds to the existing `variant=alignment`. It constructs query-visible / JI coordinates from complete score residuals and the training query second moment, fits product codebooks in ordered groups, and then encodes the full database and exports the online model. Balanced Product reordering, candidate-conditioned centered optimization, and support-aware regularization are disabled for this variant.

The equivalent explicit command is:

```bash
python method.py train --config configs/method.json --variant alignment --execute
```

Results are written to `<output_root>/alignment/` by default.

## 5. Run full REMORA training

```bash
python method.py train --config configs/method.json --variant remora --execute
```

Full training performs query-visible alignment, Balanced Product initialization, candidate-conditioned centered optimization, support-aware regularization, node recoding, and full-database encoding. Checkpoints are selected using fixed-pool Recall and centered MSE on the validation candidate pools. Results are written to `<output_root>/remora/` by default.

Override the device, budget, or output directory as needed:

```bash
python method.py train --config configs/method.json \
  --variant remora --bits 128 --device cuda:0 \
  --output outputs/method/opq-pq-b128/remora --execute
```

### Optional: initialize first, then continue training

To inspect or reuse the alignment and initial codebooks before candidate-conditioned training, run these stages separately:

```bash
# Save the query-visible + Balanced Product initialization for REMORA.
python method.py initialize --config configs/method.json --variant remora --execute

# Load the initialization, then optimize, encode the full database, and export.
python method.py train --config configs/method.json --variant remora \
  --alignment outputs/method/opq-pq-b64/alignment-balanced-b64.npz --execute
```

The checkpoint path above matches `output_root` in the template. `initialize` reads only the database, primary model/codes, and training queries; candidate pools, validation queries, and a graph are not required. The inputs, budget, layout, and initialization settings must match the subsequent training run and are checked when loading. This checkpoint is distinct from the final `model.npz`.

To save an initialization for the Alignment variant, use `initialize --variant alignment`. It produces `alignment-ordered-b64.npz` by default, which can be loaded with `alignment --alignment <checkpoint>`.

### Other existing variants

```bash
python method.py train --config configs/method.json --variant balanced --execute
python method.py train --config configs/method.json --variant centered-fixed --execute
```

| `variant` | Method configuration |
|---|---|
| `euclidean` | Euclidean residual product quantization |
| `alignment` | Query-visible / JI alignment with ordered groups |
| `balanced` | Alignment + Balanced Product; geometric initialization only |
| `uncentered-fixed` | Balanced Product + uncentered candidate MSE + fixed regularization |
| `centered-fixed` | Balanced Product + centered candidate MSE + fixed regularization |
| `centered` | Balanced Product + centered candidate MSE, without support regularization |
| `remora` | Full method: centered training + support-aware regularization |

The standalone method commands refuse to overwrite existing results. Use a new `--output` to rerun training. Skip `prepare` when candidate pools already exist; choose a new `output_root` to collect them again.

## 6. Outputs and online use

Both Alignment and full training produce:

```text
<output_root>/<variant>/
  trainer-config.json   Resolved method configuration for this run
  model.npz             Offline codebooks, projection, mean, transform, and spectrum
  online.bin            C++ online scoring model
  side.bin              One fixed residual code per database node
  training.csv          Training history
  training.json         Devices, budget, validation results, and model sizes
  encoding-parity.json  Encoding consistency audit when using Faiss encoding
```

Use `online.bin` and `side.bin` with the same primary and graph:

```text
score(q, i) = primary_score(q, i) + residual_correction(q, code_i)
```

The corresponding search arguments are `--ji-model <online.bin> --ji-side-codes <side.bin> --ji-query-space raw`. See [docs/EXPERIMENTS.md](docs/EXPERIMENTS.md) for the complete search arguments and experiment protocols.

## Method source map

| File | Responsibility |
|---|---|
| [`method.py`](method.py) | Standalone method entry point |
| [`ours/method/cli.py`](ours/method/cli.py) | Configuration, arguments, execution plans, and stage dispatch |
| [`ours/method/pools.py`](ours/method/pools.py) | Training and validation candidate collection on an existing graph |
| [`ours/method/alignment.py`](ours/method/alignment.py) | Query-visible alignment, initial codebooks, and initialization checkpoints |
| [`ours/method/training.py`](ours/method/training.py) | Centered optimization, support-aware regularization, node recoding, and validation selection |
| [`ours/method/encoding.py`](ours/method/encoding.py) | Batched full-database encoding and online format export |
| [`ours/method/config.example.json`](ours/method/config.example.json) | Method configuration template |

Shared numerical helpers, primary format readers, and C++ scoring kernels remain in their original directories. See [`ours/method/README.md`](ours/method/README.md) for the dependencies. The original `ours/test/t2i-10M/train_remora.py` remains as a compatibility entry point and calls the same method implementation. `remora.py` continues to manage the original prepare and experiment workflow. Use new output directories when running the original experiments after this refactor to satisfy source identity checks.

Run the small CPU checks with:

```bash
python -m unittest discover -s ours/method/tests -v
python -m unittest discover -s ours/test/t2i-10M -p test_integration.py -v
```

## Publish to GitHub

The publication helper targets `https://github.com/zjuDBxAI/REMORA.git`. Run it on a machine with GitHub access and an account that can push to the repository:

```bash
gh auth login --hostname github.com
bash ours/workflow/publish_github.sh
```

The helper clones the target repository into a temporary directory, copies the source files selected by `.gitignore`, commits the changes, and pushes to the repository's default branch. Existing remote history and files outside the source snapshot are preserved. The source directory's Git metadata is left untouched.

To update an existing clean clone, provide its path. The helper updates it from the remote with a fast-forward merge before copying the source and committing:

```bash
bash ours/workflow/publish_github.sh --checkout /home/xzl/REMORA
```
