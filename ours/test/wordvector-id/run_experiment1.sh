#!/usr/bin/env bash
# Frozen raw-IP primary graphs -> GPU REMORA training -> CPU8 timing.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remora_repo="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$remora_repo"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
wordvector_tag="${WORDVECTOR_TAG:-remora-traversal-exp1-nvme-b96-v1}"
wordvector_phase="${WORDVECTOR_PHASE:-all}"
output="$remora_repo/outputs/wordvector-id/$wordvector_tag"
mkdir -p "$output"
exec 9>"$output/run.lock"
if ! flock -n 9; then
  printf 'This WordVector run is already active: %s\n' "$output" >&2
  exit 73
fi
export REMORA_NATIVE_PRIMARY_READER="${REMORA_NATIVE_PRIMARY_READER:-$remora_repo/ours/runtime/native100-reader-v1/roar_native_primary_reader}"
export REMORA_PRIMARY_DECODE_THREADS=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export NVIDIA_TF32_OVERRIDE=0
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
compiler="$(dirname -- "$python_bin")/x86_64-conda-linux-gnu-c++"
if [[ -x "$compiler" ]]; then export CXX="${CXX:-$compiler}"; fi
export CMAKE_PREFIX_PATH="$(dirname -- "$(dirname -- "$python_bin")")${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
case "$wordvector_phase" in preflight|smoke|prepare|performance|all) ;; *) printf 'Invalid WORDVECTOR_PHASE\n' >&2; exit 2 ;; esac
"$python_bin" -u "$suite_dir/prepare_inputs.py" \
  --package "${WORDVECTOR_PACKAGE:-$remora_repo/datasets/WordVector-ID-native4-M40-L500}" \
  --tag "$wordvector_tag" --side-bits "${WORDVECTOR_SIDE_BITS:-96}" \
  --gpu "${GPU:-2}" --queries "${WORDVECTOR_QUERY_LIMIT:-10000}" --repeats "${REPEATS:-3}"
prepare_config="$output/inputs/prepare.config.json"
performance_config="$output/inputs/performance.config.json"
if [[ "$wordvector_phase" == preflight || "$wordvector_phase" == smoke ]]; then
  exec "$python_bin" -u ours/test/t2i-10M/run.py "$wordvector_phase" --config "$prepare_config" --experiments 1
fi
if [[ "$wordvector_phase" == all || "$wordvector_phase" == prepare ]]; then
  printf '[phase] verify raw-IP native score parity and REMORA traversal contract\n'
  "$python_bin" -u ours/test/t2i-10M/run.py smoke --config "$prepare_config" --experiments 1
  printf '[phase] GPU REMORA training/full-database encoding; CPU timing NOT started\n'
  "$python_bin" -u ours/test/t2i-10M/run.py prepare --config "$prepare_config" --experiments 1
fi
if [[ "$wordvector_phase" == all || "$wordvector_phase" == performance ]]; then
  printf '[phase] CPU8 AVX-512; REMORA TRAVERSAL; own queue Top-L NVMe O_DIRECT rerank; L<=200\n'
  exec "$python_bin" -u ours/scripts/run_native100_capped_experiment1.py launch \
    --config "$performance_config" --prepared-root "$output/prepared" \
    --cold-store /tmp/xzl-wordvector-id-cold/base.899K.cold.bin --max-rerank-depth 200
fi
