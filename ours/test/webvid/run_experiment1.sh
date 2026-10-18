#!/usr/bin/env bash
# Read-only primary graphs -> GPU REMORA preparation -> parity -> CPU timing.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
remora_repo="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$remora_repo"
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
webvid_tag="${WEBVID_TAG:-remora-traversal-exp1-nvme-b160-v1}"
webvid_bits="${WEBVID_SIDE_BITS:-160}"
webvid_phase="${WEBVID_PHASE:-all}"
webvid_package="${WEBVID_PACKAGE:-$remora_repo/datasets/WebVid-2.5M-native4-M40-L500}"
output="$remora_repo/outputs/webvid/$webvid_tag"
export REMORA_NATIVE_PRIMARY_READER="${REMORA_NATIVE_PRIMARY_READER:-$remora_repo/ours/runtime/native100-reader-v1/roar_native_primary_reader}"
export REMORA_PRIMARY_DECODE_THREADS=8
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
export PYTHONPATH="$remora_repo/.runtime/faiss-v1.9.0-build/faiss/python/build/lib:$remora_repo/ours/test/t2i-10M:$remora_repo/ours/python:$remora_repo/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
compiler="$(dirname -- "$python_bin")/x86_64-conda-linux-gnu-c++"
if [[ -x "$compiler" ]]; then export CXX="${CXX:-$compiler}"; fi
export CMAKE_PREFIX_PATH="$(dirname -- "$(dirname -- "$python_bin")")${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
case "$webvid_phase" in preflight|smoke|prepare|performance|all) ;; *) printf 'Invalid WEBVID_PHASE\n' >&2; exit 2 ;; esac
"$python_bin" -u "$suite_dir/prepare_inputs.py" --package "$webvid_package" --tag "$webvid_tag" --side-bits "$webvid_bits"
prepare_config="$output/inputs/prepare.config.json"
performance_config="$output/inputs/performance.config.json"
if [[ "$webvid_phase" == preflight || "$webvid_phase" == smoke ]]; then
  exec "$python_bin" -u ours/test/t2i-10M/run.py "$webvid_phase" --config "$prepare_config" --experiments 1
fi
if [[ "$webvid_phase" == all || "$webvid_phase" == prepare ]]; then
  "$python_bin" -u ours/test/t2i-10M/run.py smoke --config "$prepare_config" --experiments 1
  "$python_bin" -u ours/test/t2i-10M/run.py prepare --config "$prepare_config" --experiments 1
fi
if [[ "$webvid_phase" == all || "$webvid_phase" == performance ]]; then
  # Offline training completes before timing; never run the two concurrently.
  exec "$python_bin" -u ours/scripts/run_native100_capped_experiment1.py launch \
    --config "$performance_config" --prepared-root "$output/prepared" \
    --cold-store "$output/prepared/data/base.cold" --max-rerank-depth 200
fi
