#!/usr/bin/env bash
# RS5M primary graphs stay read-only; GPU REMORA training precedes CPU timing.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remora_repo="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$remora_repo"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
rs5m_tag="${RS5M_TAG:-remora-traversal-exp1-nvme-b160-v1}"
rs5m_bits="${RS5M_SIDE_BITS:-160}"
rs5m_phase="${RS5M_PHASE:-all}"
output="$remora_repo/outputs/rs5m/$rs5m_tag"
export REMORA_NATIVE_PRIMARY_READER="${REMORA_NATIVE_PRIMARY_READER:-$remora_repo/ours/runtime/native100-reader-v1/roar_native_primary_reader}"
export REMORA_PRIMARY_DECODE_THREADS=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
compiler="$(dirname -- "$python_bin")/x86_64-conda-linux-gnu-c++"
if [[ -x "$compiler" ]]; then export CXX="${CXX:-$compiler}"; fi
export CMAKE_PREFIX_PATH="$(dirname -- "$(dirname -- "$python_bin")")${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
case "$rs5m_phase" in preflight|smoke|prepare|performance|all) ;; *) printf 'Invalid RS5M_PHASE\n' >&2; exit 2 ;; esac
"$python_bin" -u "$suite_dir/prepare_inputs.py" \
  --package "${RS5M_PACKAGE:-$remora_repo/datasets/RS5M-512-native4-M40-L500}" \
  --tag "$rs5m_tag" --side-bits "$rs5m_bits" --gpu "${GPU:-2}" \
  --queries "${RS5M_QUERY_LIMIT:-10000}" --repeats "${REPEATS:-3}"
prepare_config="$output/inputs/prepare.config.json"
performance_config="$output/inputs/performance.config.json"
if [[ "$rs5m_phase" == preflight || "$rs5m_phase" == smoke ]]; then
  exec "$python_bin" -u ours/test/t2i-10M/run.py "$rs5m_phase" --config "$prepare_config" --experiments 1
fi
if [[ "$rs5m_phase" == all || "$rs5m_phase" == prepare ]]; then
  printf '[phase] native scorer parity; REMORA traversal contract; no graph rebuilding\n'
  "$python_bin" -u ours/test/t2i-10M/run.py smoke --config "$prepare_config" --experiments 1
  printf '[phase] GPU REMORA training and full-database encoding; CPU timing has NOT started\n'
  "$python_bin" -u ours/test/t2i-10M/run.py prepare --config "$prepare_config" --experiments 1
fi
if [[ "$rs5m_phase" == all || "$rs5m_phase" == performance ]]; then
  printf '[phase] CPU8 AVX-512 performance; REMORA TRAVERSAL; queue Top-L NVMe direct rerank; L<=200\n'
  exec "$python_bin" -u ours/scripts/run_native100_capped_experiment1.py launch \
    --config "$performance_config" --prepared-root "$output/prepared" \
    --cold-store "$output/prepared/data/base.cold" --max-rerank-depth 200
fi
