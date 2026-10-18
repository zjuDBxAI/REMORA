#!/usr/bin/env bash
# Reuse completed native-primary REMORA codes; run performance experiment only.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$script_dir/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
export REMORA_NATIVE_PRIMARY_READER="${REMORA_NATIVE_PRIMARY_READER:-$repo_dir/ours/runtime/native100-reader-v1/roar_native_primary_reader}"
export REMORA_PRIMARY_DECODE_THREADS="${REMORA_PRIMARY_DECODE_THREADS:-8}"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
faiss_python="${FAISS_PYTHON_PATH:-$repo_dir/.runtime/faiss-v1.9.0-build/faiss/python/build/lib}"
export PYTHONPATH="$faiss_python:$script_dir:$repo_dir/ours/python:$repo_dir/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
cd "$repo_dir"
exec "$python_bin" -u ours/scripts/run_native100_experiment1.py \
  --config "${CONFIG:-$script_dir/config.native100-exp1-nvme.json}" \
  --prepared-root "${PREPARED_ROOT:-$repo_dir/outputs/t2i-10M/remora-native100-b64-v1}" \
  --cold-store "${COLD_STORE:-/tmp/xzl-t2i-10M-cold/base.10M.cold.bin}"
