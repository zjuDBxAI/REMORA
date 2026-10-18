#!/usr/bin/env bash
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remora_repo="$(cd -- "$suite_dir/../../.." && pwd)"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
python_prefix="$(dirname -- "$(dirname -- "$python_bin")")"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export LD_LIBRARY_PATH="$python_prefix/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
export REMORA_NATIVE_PRIMARY_READER="$remora_repo/ours/runtime/native100-reader-v1/roar_native_primary_reader"
export REMORA_PRIMARY_DECODE_THREADS=8
cd "$remora_repo"
exec "$python_bin" -u "$suite_dir/run_experiment1.py" "${COMMAND:-run}" \
  --gpu "${GPU:-2}" --build-threads "${BUILD_THREADS:-64}" \
  --output "${T2I1M_EXP1_OUTPUT:-$remora_repo/outputs/t2i-1M/remora-native100-exp1-nvme-b64-v1}" \
  --wait-root "${WAIT_ROOT:-$remora_repo/outputs/rs5m/remora-exp345-relative-b80-320-v2}" \
  --wait-session "${WAIT_SESSION:-remora-rs5m-exp345}" "$@"
