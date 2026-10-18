# REMORA method implementation

See the [root README](../../README.md) for setup, configuration, and commands. Run `python method.py --help` from the repository root to inspect the standalone entry points.

This directory groups the method stages and reuses existing numerical helpers and native format support. It is not a self-contained dependency bundle that can be copied and run on its own.

| File | Main functions |
|---|---|
| `cli.py` | `main`: prepare / alignment / initialize / train |
| `pools.py` | `prepare_pools`: Primary traversal candidates and score labels |
| `alignment.py` | `fit_alignment`: complete residuals, query second moment, JI transform, grouping, and clustering; `save_alignment` / `load_alignment`: initialization state |
| `training.py` | `train`: candidate-conditioned optimization, support penalties, node recoding, and validation selection |
| `encoding.py` | `encode_database`: batched encoding; `export_model`: native online formats |

Shared dependencies remain at their original locations:

| Location | Components used by the method |
|---|---|
| `ours/python/joint_innovation_rpq.py` | `build_joint_transform` |
| `ours/test/t2i-1M/python/rotation_product.py` | Balanced Product, geometric encoding, `CenteredObjective`, and gradient updates |
| `ours/test/t2i-1M/python/representation_capacity.py` | LUT predictions, validation metrics, and `mm_reencode` |
| `ours/python/train_terminal_ranking_ji.py` | The `Pools` data structure and its transitive dependencies |
| `ours/test/t2i-10M/{qvrr,native100_primary}.py` | Primary format support and complete reconstruction in the original vector space |
| `ours/test/t2i-10M/faiss_backend.py` | Faiss clustering, geometric encoding, and encoding consistency audits |
| `ours/test/t2i-10M/{protocol,collect_pools}.py` | Matrix / CSV I/O, CSR reading, and candidate labels |
| `ours/experimental/traversal-fast-v1/`, `ours/src/`, `ours/include/` | C++ online scoring and graph search |

The `alignment` command runs the existing Alignment variant, including full-database encoding. `initialize` saves only a variant's state before candidate-conditioned training. Full REMORA uses Balanced Product initialization by default.

Existing rate experiments above 256 bits continue to use `ours/experimental/relative-rate-v1/` and are outside the budget range of this standalone entry point.
