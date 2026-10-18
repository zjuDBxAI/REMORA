#!/usr/bin/env bash
# Frozen WordVector 96-bit diagnostics: one pass per beam, no QPS scan.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
runner_repo="$(cd -- "$suite_dir/../../.." && pwd)"
remora_repo="${REMORA_REPO:-/home/xzl/REMORA}"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
export PYTHONUNBUFFERED=1 PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export REMORA_PRIMARY_DECODE_THREADS=8
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
printf '[protocol] frozen primary graphs and 96-bit REMORA; REMORA traversal; one accuracy pass; no new QPS\n'
exec "$python_bin" -u "$runner_repo/ours/scripts/run_frozen_experiments34.py" \
  --repo "$remora_repo" \
  --dataset WordVector-ID-custom-899K \
  --source "${WORDVECTOR_SOURCE:-outputs/wordvector-id/remora-traversal-exp1-nvme-b96-v1/performance}" \
  --output "${WORDVECTOR_EXP34_OUTPUT:-outputs/wordvector-id/remora-exp34-b96-v1}" \
  --inversion-ranks 50,75,100 \
  --beams "${WORDVECTOR_TRAJECTORY_BEAMS:-25,50,75,100,150,200,300,500,750,1000}" \
  --query-limit "${WORDVECTOR_QUERY_LIMIT:-10000}" \
  --phases "${WORDVECTOR_EXP34_PHASES:-inversion,traversal,report}"
