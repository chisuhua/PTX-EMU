#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTXEMU_ROOT="${PTXEMU_ROOT:-$(cd "$HERE/../../.." && pwd)}"
CUDART_SO="${PTXEMU_ROOT}/build/lib/libcudart.so"
PTXIR_EMBED="${PTXEMU_ROOT}/build/bin/ptxir_embed"

# Step 1: nvcc -c -> .o (object file, NOT .cubin -- ptxir_embed needs .o)
nvcc -c -arch=sm_100 -O2 "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_kernels.o"

# Step 2: cuobjdump -ptx -> .ptx (then clean to remove prefix bytes)
cuobjdump -ptx "$HERE/path_1B_kernels.o" > "$HERE/path_1B_kernels.raw.ptx"
# Strip everything before ".version" line (cuobjdump emits ELF prefix)
sed -n '/^\.version/,$p' "$HERE/path_1B_kernels.raw.ptx" > "$HERE/path_1B_kernels.ptx"

# Step 3: ptxir_embed --in-cubin X.o --in-ptx X.ptx --kernel-name K --out X.embedded
"$PTXIR_EMBED" \
    --in-cubin "$HERE/path_1B_kernels.o" \
    --in-ptx "$HERE/path_1B_kernels.ptx" \
    --kernel-name vector_add \
    --out "$HERE/path_1B_standalone.cubin"

# Step 4: Build standalone executable that loads via PTX-EMU libcudart.so
nvcc -arch=sm_100 "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_standalone" \
    -L"$PTXEMU_ROOT/build/lib" -lcudart -Xlinker -rpath -Xlinker "$PTXEMU_ROOT/build/lib"
