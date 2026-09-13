#!/bin/bash
# Build puffer_ptcg/_C.so: PTCG C env statically linked into the pufferlib-4.0
# native CUDA trainer. Run from repo root on the CUDA host.
#   bash native/build_native.sh [--debug]
# No raylib, no cudnn. fp32 only (PRECISION_FLOAT). Requires: nvcc, gcc,
# pybind11 + numpy (pip), nccl (nvidia-nccl-cu12 wheel or system), libcg.
set -euo pipefail
cd "$(dirname "$0")"

OPT="-O2"
NVCC_OPT="-O2 --threads 0"
if [[ "${1:-}" == "--debug" ]]; then OPT="-O0 -g"; NVCC_OPT="-O0 -g -lineinfo"; fi

# GPU arch from the actual device (sm_86 was hardcoded and won't run on
# newer parts — a 5090 is sm_120). Override with PTCG_SM if needed.
# --id=0 rather than `| head -1`: on a multi-GPU host nvidia-smi writes one line
# per GPU, head exits after the first, and nvidia-smi dies of SIGPIPE (141).
# Under `set -euo pipefail` that status propagates and kills this script AFTER
# SM was already assigned -- a silent exit 141 with no output, and the
# SM:-86 fallback below never runs. `|| true` keeps set -e off the fallback.
SM="${PTCG_SM:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader --id=0 2>/dev/null | tr -d '.' || true)}"
SM="${SM:-86}"

CUDA_HOME=${CUDA_HOME:-$(dirname "$(dirname "$(which nvcc)")")}
PY=${PY:-python}
PYTHON_INCLUDE=$($PY -c "import sysconfig; print(sysconfig.get_path('include'))")
PYBIND_INCLUDE=$($PY -c "import pybind11; print(pybind11.get_include())")
NUMPY_INCLUDE=$($PY -c "import numpy; print(numpy.get_include())")
EXT_SUFFIX=$($PY -c "import sysconfig; print(sysconfig.get_config_var('EXT_SUFFIX'))")

# NCCL from NCCL_HOME, else the system, else the nvidia-nccl wheel.
# NCCL_HOME=<dir holding include/nccl.h and lib/libnccl.so.2> pins one when
# the system NCCL was built for a newer CUDA than the driver supports: the
# run then dies in ncclCommInitRank with "CUDA driver version is insufficient
# for CUDA runtime version" (NCCL_DEBUG=WARN shows it). The wheel matching
# torch's CUDA is
#   NCCL_HOME=$(python -c "import nvidia.nccl;print(nvidia.nccl.__path__[0])")
NCCL_IFLAG=""; NCCL_LFLAG=""
if [[ -n "${NCCL_HOME:-}" ]]; then
    NCCL_IFLAG="-I$NCCL_HOME/include"; NCCL_LFLAG="-L$NCCL_HOME/lib"
fi
for dir in /usr/include /usr/local/cuda/include; do
    [[ -z "$NCCL_IFLAG" && -f "$dir/nccl.h" ]] && NCCL_IFLAG="-I$dir" && break
done
for dir in /usr/lib/x86_64-linux-gnu /usr/local/cuda/lib64; do
    [[ -z "$NCCL_LFLAG" ]] && { [[ -f "$dir/libnccl.so" ]] || [[ -f "$dir/libnccl.so.2" ]]; } && NCCL_LFLAG="-L$dir" && break
done
if [[ -z "$NCCL_IFLAG" ]]; then
    NCCL_IFLAG=$($PY -c "import nvidia.nccl, os; print('-I' + os.path.join(nvidia.nccl.__path__[0], 'include'))" 2>/dev/null || true)
fi
if [[ -z "$NCCL_LFLAG" ]]; then
    NCCL_LFLAG=$($PY -c "import nvidia.nccl, os; print('-L' + os.path.join(nvidia.nccl.__path__[0], 'lib'))" 2>/dev/null || true)
fi
[[ -z "$NCCL_IFLAG" ]] && { echo "ERROR: nccl.h not found (pip install nvidia-nccl-cu12/-cu13)"; exit 1; }
RPATH_FLAGS=()
[[ "$NCCL_LFLAG" == -L* ]] && RPATH_FLAGS+=("-Wl,-rpath,${NCCL_LFLAG#-L}")
# pip wheels ship only libnccl.so.2 (no unversioned symlink for -lnccl)
NCCL_LINK="-lnccl"
if [[ "$NCCL_LFLAG" == -L* ]] && [[ ! -e "${NCCL_LFLAG#-L}/libnccl.so" ]] \
        && [[ -e "${NCCL_LFLAG#-L}/libnccl.so.2" ]]; then
    NCCL_LINK="-l:libnccl.so.2"
fi

CG_DIR="$(pwd)/../data/cg"
mkdir -p build puffer_ptcg
[[ -f puffer_ptcg/__init__.py ]] || echo "from . import _C  # noqa" > puffer_ptcg/__init__.py

echo "== C env static lib =="
CSRC=(ptcg/obs.c ptcg/tables.c ptcg/tracker.c ptcg/encoder.c ptcg/env.c vendor/cJSON.c ptcg/binding.c)
OBJS=()
for f in "${CSRC[@]}"; do
    o="build/$(basename "$f" .c).o"
    gcc -c $OPT -fopenmp -std=gnu11 -fPIC -fvisibility=hidden -fno-semantic-interposition \
        -Wall -Wno-unused-parameter -Iptcg -Ivendor -Isrc "$f" -o "$o"
    OBJS+=("$o")
done
ar rcs build/libstatic_ptcg.a "${OBJS[@]}"

echo "== CUDA trainer (sm_${SM}, fp32) =="
nvcc -c -arch=sm_${SM} -std=c++17 $NVCC_OPT \
    -Xcompiler -fPIC -Xcompiler=-fopenmp \
    -Xcompiler=-D_GLIBCXX_USE_CXX11_ABI=1 \
    -Xcompiler=-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION \
    -Isrc -Imodel -Iptcg -Ivendor \
    -I"$PYTHON_INCLUDE" -I"$PYBIND_INCLUDE" -I"$NUMPY_INCLUDE" \
    -I"$CUDA_HOME/include" $NCCL_IFLAG \
    -DOBS_TENSOR_T=FloatTensor -DENV_NAME=ptcg \
    -DPTCG_NATIVE -DPRECISION_FLOAT \
    src/bindings.cu -o build/bindings.o

# libnvidia-ml: the driver ships libnvidia-ml.so.1 but the .so dev symlink
# comes from cuda-nvml-dev, which is not installed everywhere (without that
# package the link fails with "cannot find -lnvidia-ml"). The
# CUDA stub is the supported way to link it -- the real driver library is
# picked up at runtime either way.
NVML_LFLAG=""
if ! [[ -e /usr/lib/x86_64-linux-gnu/libnvidia-ml.so || -e "$CUDA_HOME/lib64/libnvidia-ml.so" ]]; then
    [[ -e "$CUDA_HOME/lib64/stubs/libnvidia-ml.so" ]] \
        && NVML_LFLAG="-L$CUDA_HOME/lib64/stubs" \
        || echo "WARN: no libnvidia-ml.so and no CUDA stub; link may fail"
fi

echo "== link =="
g++ -shared -fPIC -fopenmp $OPT \
    build/bindings.o build/libstatic_ptcg.a \
    -L"$CUDA_HOME/lib64" $NVML_LFLAG $NCCL_LFLAG -L"$CG_DIR" \
    "${RPATH_FLAGS[@]}" -Wl,-rpath,"$CG_DIR" -Wl,-rpath,"$CUDA_HOME/lib64" \
    -lcudart -lcublas -lcublasLt -lcurand $NCCL_LINK -lnvidia-ml -lcg -lm -lpthread \
    -Wl,-Bsymbolic-functions \
    -o "puffer_ptcg/_C${EXT_SUFFIX}"
echo "Built: native/puffer_ptcg/_C${EXT_SUFFIX}"
