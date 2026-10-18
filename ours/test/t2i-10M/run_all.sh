#!/usr/bin/env bash
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
export PYTHONPATH="$repo_dir/ours/python:$repo_dir/ours/test/t2i-1M/python${PYTHONPATH:+:$PYTHONPATH}"
faiss_python="${FAISS_PYTHON_PATH:-$repo_dir/.runtime/faiss-v1.9.0-build/faiss/python/build/lib}"
if [[ -d "$faiss_python/faiss" ]]; then
  faiss_cache="$repo_dir/.runtime/faiss-v1.9.0-build/CMakeCache.txt"
  if [[ -f "$faiss_cache" ]]; then
    build_python="$(sed -n 's/^_Python_EXECUTABLE:INTERNAL=//p' "$faiss_cache")"
    if [[ -n "$build_python" ]]; then
      expected_abi="$("$build_python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      current_abi="$("$python_bin" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
      if [[ "$expected_abi" != "$current_abi" ]]; then
        printf 'Faiss Python ABI mismatch: built for %s, requested %s. Rebuild the isolated runtime.\n' "$expected_abi" "$current_abi" >&2
        exit 1
      fi
    fi
  fi
  export PYTHONPATH="$faiss_python:$PYTHONPATH"
fi
exec "$python_bin" "$suite_dir/run.py" "${COMMAND:-all}" \
  --config "${CONFIG:-$suite_dir/config.example.json}" \
  --experiments "${EXPERIMENTS:-1,2,3,4,5}"
