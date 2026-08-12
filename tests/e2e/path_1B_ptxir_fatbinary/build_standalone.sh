#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTXEMU_ROOT="${PTXEMU_ROOT:-$(cd "$HERE/../../.." && pwd)}"
CUDART_SO="${PTXEMU_ROOT}/build/lib/libcudart.so"
PTXIR_EMBED="${PTXEMU_ROOT}/build/bin/ptxir_embed"

# Step 1: nvcc -> .cubin (raw kernel binary, no PTXIR yet)
nvcc -arch=sm_100 -O2 -c "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_kernels.cubin" --cubin

# Step 2: ptxir_embed --in-cubin X.cubin --in-ptx X.ptx --kernel-name K --out X.embedded
# NOTE: Actual CLI is --in-cubin / --in-ptx / --out / --kernel-name (NOT --input/--output/--manifest)
"$PTXIR_EMBED" \
    --in-cubin "$HERE/path_1B_kernels.cubin" \
    --in-ptx "$HERE/path_1B_kernels.ptx" \
    --kernel-name vector_add \
    --out "$HERE/path_1B_standalone.cubin"

# Step 3: Build standalone executable that loads via PTX-EMU libcudart.so
nvcc -arch=sm_100 "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_standalone" \
    -L"$PTXEMU_ROOT/build/lib" -lcudart -Xlinker -rpath -Xlinker "$PTXEMU_ROOT/build/lib"
