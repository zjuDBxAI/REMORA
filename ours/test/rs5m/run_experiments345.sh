#!/usr/bin/env bash
# Accuracy only: no performance scan, one pass, frozen primary/graph/model.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remora_repo="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$remora_repo"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
exec "$python_bin" -u "$suite_dir/run_experiments345.py" \
  --source "${RS5M_SOURCE:-outputs/rs5m/remora-traversal-exp1-nvme-b160-v1}" \
  --output "${RS5M_EXP345_OUTPUT:-outputs/rs5m/remora-exp345-relative-b80-320-v2}" \
  --queries "${RS5M_QUERY_LIMIT:-10000}" --gpu "${GPU:-2}" \
  --distortion-nodes "${DISTORTION_NODES:-100000}" \
  --beams "${RS5M_TRAJECTORY_BEAMS:-50,100,200,300,500,1000}" \
  --phases "${RS5M_EXP345_PHASES:-check,diagnostics,rate}"
