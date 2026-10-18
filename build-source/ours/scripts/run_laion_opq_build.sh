#!/usr/bin/env bash
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
faiss_python="${FAISS_PYTHON_PATH:-/home/xzl/REMORA/.runtime/faiss-v1.9.0-build/faiss/python/build/lib}"
export PYTHONPATH="$faiss_python${PYTHONPATH:+:$PYTHONPATH}"
export OMP_NUM_THREADS="${BUILD_THREADS:-$(nproc)}"
export OPENBLAS_NUM_THREADS="$OMP_NUM_THREADS"
export MKL_NUM_THREADS="$OMP_NUM_THREADS"
export PYTHONUNBUFFERED=1
exec "$python_bin" -u ours/scripts/laion_opq_build.py --threads "$OMP_NUM_THREADS" "$@"
