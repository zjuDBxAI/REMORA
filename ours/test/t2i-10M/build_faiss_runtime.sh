#!/usr/bin/env bash
# Isolated source build; never changes the user's existing conda environment.
set -euo pipefail
suite_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
repo_dir="$(cd -- "$suite_dir/../../.." && pwd)"
cd "$repo_dir"
python_bin="${PYTHON_BIN:-python3}"
faiss_source="$repo_dir/.runtime/src/faiss-v1.9.0"
faiss_build="$repo_dir/.runtime/faiss-v1.9.0-build"
tools_dir="$repo_dir/.runtime/build-tools"
cuda_toolkit="${FAISS_CUDA_TOOLKIT:-/usr/local/cuda-12.1}"
if [[ ! -d "$faiss_source/.git" ]]; then
  git clone --depth 1 --branch v1.9.0 https://github.com/facebookresearch/faiss.git "$faiss_source"
fi
if [[ ! -x "$tools_dir/swig/data/bin/swig" ]]; then
  "$python_bin" -m pip install --target "$tools_dir" swig==4.3.0
fi
export SWIG_LIB="$tools_dir/swig/data/share/swig/4.3.0"
cmake -S "$faiss_source" -B "$faiss_build" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_COMPILER="${FAISS_CXX_COMPILER:-/usr/bin/g++}" \
  -DCMAKE_CUDA_COMPILER="$cuda_toolkit/bin/nvcc" -DCUDAToolkit_ROOT="$cuda_toolkit" \
  -DCMAKE_CUDA_ARCHITECTURES="${FAISS_CUDA_ARCH:-89}" \
  -DFAISS_ENABLE_GPU=ON -DFAISS_ENABLE_PYTHON=ON -DFAISS_ENABLE_C_API=OFF \
  -DBUILD_TESTING=OFF -DBUILD_SHARED_LIBS=ON -DFAISS_OPT_LEVEL=avx512 \
  -DBLA_VENDOR=Generic -DPython_EXECUTABLE="$python_bin" \
  -DSWIG_EXECUTABLE="$tools_dir/swig/data/bin/swig" -DSWIG_DIR="$SWIG_LIB"
cmake --build "$faiss_build" --target swigfaiss_avx512 -j "${FAISS_BUILD_JOBS:-12}"
# Upstream packaging operates only on its own generated build directory.
cd "$faiss_build/faiss/python"
"$python_bin" setup.py build
printf '%s\n' '[done] Isolated Faiss runtime built. Run run_all.sh with the same Python ABI.'
