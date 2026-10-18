#!/usr/bin/env bash
# Preparation ONLY: no graph build, primary retraining, or QPS timing.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
roar_repo="$(cd -- "$script_dir/../../.." && pwd)"
if [[ -f "$roar_repo/ours/test/t2i-10M/run_all.sh" ]]; then
  remora_repo="${REMORA_REPO:-$roar_repo}"
else
  remora_repo="${REMORA_REPO:-/home/xzl/REMORA}"
fi
python_bin="${PYTHON_BIN:-/home/xzl/miniconda3/envs/roargraph/bin/python}"
config="${CONFIG:-$script_dir/config.native100-gpu.json}"
local_reader="$remora_repo/ours/runtime/native100-reader-v1/roar_native_primary_reader"
if [[ ! -x "$local_reader" ]]; then
  local_reader="$roar_repo/ours/build-native100-reader-v1/roar_native_primary_reader"
fi
export REMORA_NATIVE_PRIMARY_READER="${REMORA_NATIVE_PRIMARY_READER:-$local_reader}"
export REMORA_PRIMARY_DECODE_THREADS="${REMORA_PRIMARY_DECODE_THREADS:-8}"
compiler="$(dirname -- "$python_bin")/x86_64-conda-linux-gnu-c++"
if [[ -x "$compiler" ]]; then export CXX="${CXX:-$compiler}"; fi
export CMAKE_PREFIX_PATH="$(dirname -- "$(dirname -- "$python_bin")")${CMAKE_PREFIX_PATH:+:$CMAKE_PREFIX_PATH}"
command="${COMMAND:-prepare}"
case "$command" in preflight|smoke|prepare) ;; *) printf 'COMMAND must be preflight, smoke, or prepare\n' >&2; exit 2 ;; esac
test -x "$REMORA_NATIVE_PRIMARY_READER"
test -f "$remora_repo/ours/test/t2i-10M/native100_primary.py"
cd "$remora_repo"
PYTHON_BIN="$python_bin" CONFIG="$config" COMMAND="$command" EXPERIMENTS=1 \
  bash ours/test/t2i-10M/run_all.sh
