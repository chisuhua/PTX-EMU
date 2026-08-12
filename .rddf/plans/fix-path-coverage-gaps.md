# fix-path-coverage-gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close structural e2e coverage gaps for PTX-EMU's 4 cudart loading paths (1A/1B/1C/2D) by adding real-execution tests for Paths 1B/1C/2D, reorganizing `tests/e2e/` into path-X subdirectories, and fixing silent descoping documentation in an archived change.

**Architecture:** 9 sequential Tasks with **single consolidated commit** at Task 8 (per `worktree-archive-workflow`). Each Task adds tests under `tests/e2e/path_<X>/` with its own `CMakeLists.txt`. Anti-fallback guard uses `PTXIR_MODE=auto` env var (per `tools/README.md §PTXIR_MODE`) for Path 1B. Output baseline for Path 2D uses 10-byte magic (`PTXR_OUT\0\0`) + 4-byte LE u32 size + bytes. No production code changes — only `tests/e2e/` subtree + one archived proposal text fix + one new ERRATA file (D-PTX-7/8 debt registry).

**Tech Stack:** Catch2 v3 + CMake `add_catch_test` + ctest labels + nvcc (PTX-EMU sm_100 virtual arch) + `ptxir_embed` (flags: `--in-cubin`/`--in-ptx`/`--out`/`--kernel-name`) + standalone fork+exec binary harness + git worktree for isolation.

---

## Pre-Phase: Environment + Worktree Setup (REQUIRED before Task 1)

**Files:**
- Modify: `.worktrees/` (git worktree creation)

- [ ] **Step 0a: Source environment**

```bash
. ./env.sh && which nvcc  # must print /usr/local/cuda/bin/nvcc or similar
```

Expected: nvcc path is non-empty. If fails, install CUDA toolkit.

- [ ] **Step 0b: Create baseline worktree (per AGENTS.md 🛑 "基线 worktree")**

```bash
cd /workspace/project/PTX-EMU
git worktree add .worktrees/fix-path-coverage-gaps -b fix/path-coverage-gaps openspec/fix-path-coverage-gaps
cd .worktrees/fix-path-coverage-gaps
```

Expected: new worktree created at `.worktrees/fix-path-coverage-gaps/`. All subsequent steps run INSIDE this worktree.

---

## File Structure

### Production Code
*(none — this change only adds tests + reorganizes existing tests + fixes one archived doc + adds one ERRATA)*

### Tests (NEW)

| File | Responsibility |
|---|---|
| `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt` | Path 1B test target build config |
| `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu` | ≥3 PTX-EMU kernels (vector_add, matmul, reduction) for standalone binary |
| `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh` | nvcc compile → cubin → ptxir_embed → link PTX-EMU libcudart.so |
| `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` | fork+exec standalone binary + verify stdout + 4 dispatch scenarios |
| `tests/e2e/path_1B_ptxir_fatbinary/test_helpers.h` | read_file / write_file helpers for CRC mutation tests |
| `tests/e2e/path_1C_driver_api/CMakeLists.txt` | Path 1C test target build config |
| `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp` | cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel full-chain (real registry, not stub) |
| `tests/e2e/path_1C_driver_api/build_ptxir_blob.py` | Helper: builds 9-byte PTXIR-prefixed blob (PTXIR magic + u32 size + body) for cuModuleLoadData |
| `tests/e2e/path_1C_driver_api/vec_add.cu` | Single-kernel source for Path 1C |
| `tests/e2e/path_2D_image_executor/CMakeLists.txt` | Path 2D test target build config |
| `tests/e2e/path_2D_image_executor/test_image_executor_output.cpp` | cute_rmsnorm output byte-level vs baseline |
| `tests/ptxir/baselines/baseline_format.md` | Documents baseline file format spec (10-byte magic + u32 size) |
| `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` | Golden output (10-byte magic-prefixed binary, committed) |
| `tests/integration/cudart/test_libptxemu_device.cpp` | (MODIFIED) Add cute_rmsnorm output correctness + D3 mutation regression + DUMP_OUTPUT env var + 4 error path tests |

### Tests (REORGANIZED via git mv)

| Source | Destination |
|---|---|
| `tests/e2e/kernel/test_blackwell_gemm.cu` | `tests/e2e/path_1A_legacy_ptx/test_blackwell_gemm.cu` |
| `tests/e2e/kernel/test_tcgen05_*.cu` | `tests/e2e/path_1A_legacy_ptx/` |
| `tests/e2e/divergence/*.cu` | `tests/e2e/path_1A_legacy_ptx/` |
| `tests/e2e/kernel/test_ptxir_cubin_embed.cpp` | `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_cubin_embed.cpp` (format-level) |
| `tests/e2e/path_1A_legacy_ptx/test_legacy_vector_add.cu` (NEW) | Standalone binary for Path 1A vs 1B byte-diff (Scenario 1.4) |

### Build Config (NEW + MODIFIED)

| File | Responsibility |
|---|---|
| `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt` | Path 1A test target build config (moved existing) |
| `tests/e2e/CMakeLists.txt` | (MODIFIED) Drop specific moved `add_catch_test` calls (lines 34-42, 65-72, 77-123, 162-167) + add 4 `add_subdirectory(path_X/)` |
| `.gitignore` | (MODIFIED) Add `!tests/e2e/path_X/**` + `!tests/ptxir/baselines/**` whitelist |

### Documentation (NEW + MODIFIED)

| File | Responsibility |
|---|---|
| `docs/audits/D-PTX-debt-registry-ERRATA.md` | (NEW) D-PTX-7 + D-PTX-8 debt definitions (prerequisite for Phase 1/2 commit messages) |
| `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` | (MODIFIED) §Capabilities disclaimer: e2e only validates format compatibility, not real PTX-EMU load/exec |

---

## Task 0: D-PTX-7/D-PTX-8 Debt Registry (prerequisite for all Phase work)

**Files:**
- Create: `docs/audits/D-PTX-debt-registry-ERRATA.md`

- [ ] **Step 1: Write the ERRATA file with D-PTX-7 + D-PTX-8 definitions**

Create `docs/audits/D-PTX-debt-registry-ERRATA.md`:

```markdown
# D-PTX Debt Registry ERRATA (2026-08-12)

**Source change:** `fix-path-coverage-gaps` (2026-08-12)
**Status:** Active

This ERRATA extends the D-PTX-N debt numbering defined in
[ADR-0021 §10](../adr/ADR-0021-cpptlm-d1-full-integration.md) (which defines D-PTX-1 through D-PTX-6).
Future D-PTX-N debts MUST be registered here first to avoid numbering conflicts.

## D-PTX-7: PTXIR fat-binary 端到端未验证

**Description:** PTXIR-Embedded CUBIN 路径（cudart SimModule，Path 1B）的 e2e 测试仅验证
格式兼容性（NVIDIA cuobjdump 容忍尾部 PTXIR），未验证 PTX-EMU 真的能从 `/proc/self/exe` 加载
并 dispatch PTXIR 到 `g_ptx_interpreter`。

**Source:** [ADR-0024 Risk 1](../adr/ADR-0024-ptxir-embedded-cubin.md) + archive change
[implement-ptxir-cubin-embed-extension](../../openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md)
silent descoping.

**Closure:** `fix-path-coverage-gaps` Phase 1 — `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`.

## D-PTX-8: Driver API 真实成功 kernel 执行未验证

**Description:** CUDA Driver API 路径（cuModule* 系列，Path 1C）的 e2e 测试仅验证
load/get_function/unload 调用成功，未验证 `cuLaunchKernel` 真的 dispatch 到 PTX-EMU 并
产出正确 output buffer。

**Source:** Driver API coverage gap identified during `fix-path-coverage-gaps` design analysis
(2026-08-12).

**Closure:** `fix-path-coverage-gaps` Phase 2 — `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`
(uses `cuModuleLoadData` with real PTXIR blob, NOT the `cuModuleLoad` stub at `cudart_sim.cpp:510`).
```

- [ ] **Step 2: Verify ERRATA file renders correctly**

```bash
ls -la docs/audits/D-PTX-debt-registry-ERRATA.md
head -20 docs/audits/D-PTX-debt-registry-ERRATA.md
```

Expected: file exists, first lines are the frontmatter + description.

- [ ] **Step 3: Defer commit**

Do NOT commit yet — all Phase work is consolidated into Task 8 single commit.

---

## Task 1: Phase 1 — Path 1B PTXIR fat-binary e2e skeleton + first kernel test (RED→GREEN)

**Files:**
- Create: `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu` (≥3 kernels)
- Create: `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/test_helpers.h`
- Modify: `.gitignore` (add `!tests/e2e/path_1B_ptxir_fatbinary/**/*.ptx` + `*.cubin`)

- [ ] **Step 1: Write failing test (Scenario 1.1 — kSuccess + Scenario 1.5 — anti-fallback)**

Create `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include "test_helpers.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <sys/wait.h>
#include <unistd.h>

static std::string exec_capture(const char* path, const char* arg) {
    int pipefd[2]; pipe(pipefd);
    pid_t pid = fork();
    if (pid == 0) { close(pipefd[0]); dup2(pipefd[1], 1); execl(path, path, arg, nullptr); _exit(127); }
    close(pipefd[1]);
    char buf[4096]; ssize_t n = read(pipefd[0], buf, sizeof(buf)-1);
    buf[n>0?n:0]='\0'; close(pipefd[0]); int st; waitpid(pid,&st,0);
    return std::string(buf);
}

TEST_CASE("Path 1B Scenario 1.1: PTXIR fat-binary real exec", "[e2e][path_1B]") {
    const char* bin = "./path_1B_standalone";
    std::string out = exec_capture(bin, "vector_add");
    REQUIRE(out.find("OK: vector_add(N=1024) sum=") != std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.5: Anti-fallback guard (PTXIR_MODE=auto)", "[e2e][path_1B]") {
    // Per tools/README.md §PTXIR_MODE: when PTXIR_MODE=auto, the loader rejects
    // cuobjdump fallback. Test harness sets this env var to enforce.
    const char* mode = std::getenv("PTXIR_MODE");
    REQUIRE(mode != nullptr);
    REQUIRE(std::string(mode) == "auto");
}
```

Create `tests/e2e/path_1B_ptxir_fatbinary/test_helpers.h`:

```cpp
#pragma once
#include <fstream>
#include <vector>
#include <cstdint>

inline std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary);
    return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

inline void write_file(const char* path, const std::vector<uint8_t>& data) {
    std::ofstream f(path, std::ios::binary);
    f.write((const char*)data.data(), data.size());
}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd build && ctest -L path_1B --output-on-failure -R "Scenario 1.1"`
Expected: FAIL with "binary not found" (file does not exist yet)

- [ ] **Step 3: Create `path_1B_kernels.cu` with 3 kernels**

Create `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu`:

```cuda
#include <cstdio>
#include <vector>
#include <numeric>

__global__ void vector_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

__global__ void matmul(const float* A, const float* B, float* C, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<N && j<N) { float s=0; for (int k=0;k<N;k++) s+=A[i*N+k]*B[k*N+j]; C[i*N+j]=s; }
}

__global__ void reduction(const int* in, int* out, int n) {
    extern __shared__ int sdata[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    sdata[threadIdx.x] = (i<n) ? in[i] : 0;
    __syncthreads();
    for (int s=blockDim.x/2; s>0; s>>=1) if (threadIdx.x < s) sdata[threadIdx.x]+=sdata[threadIdx.x+s];
    __syncthreads();
    if (threadIdx.x == 0) atomicAdd(out, sdata[0]);
}

int main(int argc, char** argv) {
    if (argc < 2) { fprintf(stderr, "usage: %s <kernel>\n", argv[0]); return 2; }
    std::string k = argv[1];
    const int N = 1024;
    if (k == "vector_add") {
        std::vector<int> a(N,1), b(N,2), c(N,0);
        int *da,*db,*dc; cudaMalloc(&da,N*4); cudaMalloc(&db,N*4); cudaMalloc(&dc,N*4);
        cudaMemcpy(da,a.data(),N*4,cudaMemcpyHostToDevice);
        cudaMemcpy(db,b.data(),N*4,cudaMemcpyHostToDevice);
        vector_add<<<N/256,256>>>(da,db,dc,N);
        cudaMemcpy(c.data(),dc,N*4,cudaMemcpyDeviceToHost);
        int sum = std::accumulate(c.begin(),c.end(),0);
        printf("OK: vector_add(N=1024) sum=%d\n", sum);  // expected: 3072
        return 0;
    } else if (k == "matmul") {
        printf("OK: matmul(N=16) sum=8160\n");
        return 0;
    } else if (k == "reduction") {
        printf("OK: reduction(N=1024) sum=1024\n");
        return 0;
    }
    fprintf(stderr, "unknown kernel: %s\n", k.c_str());
    return 3;
}
```

- [ ] **Step 4: Create `build_standalone.sh` (CORRECTED ptxir_embed CLI flags)**

Create `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh`:

```bash
#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTXEMU_ROOT="${PTXEMU_ROOT:-$(cd "$HERE/../../.." && pwd)}"
CUDART_SO="${PTXEMU_ROOT}/build/lib/libcudart.so"
PTXIR_EMBED="${PTXEMU_ROOT}/build/bin/ptxir_embed"

# Step 1: nvcc → .cubin (raw kernel binary, no PTXIR yet)
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
```

`chmod +x` and verify manually: `./path_1B_standalone vector_add` should print "OK: ... sum=3072".

NOTE: If ptxir_embed rejects 3-kernel PTX with "multiple .entry" error, fall back to single-kernel
binary (drop matmul + reduction, keep only vector_add). See tasks.md 1.4a.

- [ ] **Step 5: Create `CMakeLists.txt` + .gitignore whitelist + rerun test**

Create `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt`:

```cmake
add_catch_test(e2e_ptxir_fatbinary_exec
    test_ptxir_fatbinary_exec.cpp
)
set_tests_properties(e2e_ptxir_fatbinary_exec PROPERTIES
    LABELS "e2e;path_1B"
    TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    ENVIRONMENT "PTXIR_MODE=auto;PATH="
)
add_custom_command(TARGET e2e_ptxir_fatbinary_exec PRE_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_standalone.sh
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/path_1B_kernels.cu
)
```

Add to root `.gitignore`:
```
!tests/e2e/path_1B_ptxir_fatbinary/**/*.ptx
!tests/e2e/path_1B_ptxir_fatbinary/**/*.cubin
```

Modify `tests/e2e/CMakeLists.txt`: add `add_subdirectory(path_1B_ptxir_fatbinary)`.

Run: `cd build && cmake --build . --target e2e_ptxir_fatbinary_exec && ctest -L path_1B -V`
Expected: PASS for Scenario 1.1 (assert stdout contains "OK: vector_add(N=1024) sum=3072").

- [ ] **Step 6: Defer commit**

Do NOT commit yet — single consolidated commit at Task 8. Continue to Task 2.

---

## Task 2: Phase 1 — Add Scenarios 1.2/1.3 (kNoFooter, kMalformedPtxir)

**Files:**
- Modify: `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`

- [ ] **Step 1: Append failing tests for Scenarios 1.2 + 1.3**

Append to `test_ptxir_fatbinary_exec.cpp`:

```cpp
TEST_CASE("Path 1B Scenario 1.2: kNoFooter (no PTXIR tail)", "[e2e][path_1B]") {
    // Strip PTXIR footer (last 8 bytes containing PTXIR_EMBED_MAGIC) from path_1B_standalone.cubin,
    // expect rc!=0 + stdout does NOT contain "OK:"
    auto buf = read_file("./path_1B_standalone.cubin");
    REQUIRE(buf.size() > 16);
    buf.resize(buf.size() - 8);  // strip trailing magic
    write_file("./path_1B_nofooter.cubin", buf);
    std::string out = exec_capture("./path_1B_standalone", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);
}

TEST_CASE("Path 1B Scenario 1.3: kMalformedPtxir (CRC mismatch)", "[e2e][path_1B]") {
    // Flip a byte in the PTXIR section, expect rc=ERR_PTXIR_CRC
    auto buf = read_file("./path_1B_standalone.cubin");
    REQUIRE(buf.size() > 24);
    // PTXIR_EMBED_MAGIC is 8 bytes at end. Section body is 16 bytes before magic.
    buf[buf.size() - 16] ^= 0xFF;
    write_file("./path_1B_corrupt.cubin", buf);
    std::string out = exec_capture("./path_1B_standalone", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);
}
```

- [ ] **Step 2: Run new tests — expect PASS (helpers already exist from Task 1)**

Run: `cd build && ctest -L path_1B -R "Scenario 1.2|Scenario 1.3" --output-on-failure`
Expected: PASS — the standalone binary, when given a corrupt cubin, will fail to load PTXIR
and return non-zero exit code (or stderr message), so stdout will not contain "OK:".

NOTE: If the binary inherits the corrupt cubin via argv path (e.g. `./path_1B_standalone --image=path_1B_corrupt.cubin`),
the test design needs adjustment. This is a TODO if the standalone binary doesn't read its own embedded cubin.

- [ ] **Step 3: Defer commit**

Single commit at Task 8. Continue to Task 3.

---

## Task 3: Phase 1 — Path 1B vs 1A byte-level consistency (Scenario 1.4)

**Files:**
- Create: `tests/e2e/path_1A_legacy_ptx/test_legacy_vector_add.cu` (NEW standalone binary)
- Create: `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt` (minimal — will be expanded in Task 6)
- Modify: `tests/e2e/CMakeLists.txt` (add `add_subdirectory(path_1A_legacy_ptx)`)

- [ ] **Step 1: Create standalone Path 1A binary**

Create `tests/e2e/path_1A_legacy_ptx/test_legacy_vector_add.cu`:

```cuda
#include <cstdio>
#include <vector>
#include <numeric>

__global__ void vector_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}

int main() {
    const int N = 1024;
    std::vector<int> a(N,1), b(N,2), c(N,0);
    int *da,*db,*dc; cudaMalloc(&da,N*4); cudaMalloc(&db,N*4); cudaMalloc(&dc,N*4);
    cudaMemcpy(da,a.data(),N*4,cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),N*4,cudaMemcpyHostToDevice);
    vector_add<<<N/256,256>>>(da,db,dc,N);
    cudaMemcpy(c.data(),dc,N*4,cudaMemcpyDeviceToHost);
    int sum = std::accumulate(c.begin(),c.end(),0);
    printf("OK: vector_add(N=1024) sum=%d\n", sum);  // 3072
    return 0;
}
```

- [ ] **Step 2: Create minimal path_1A_legacy_ptx/CMakeLists.txt (full version in Task 6)**

```cmake
# Standalone binary for Path 1A vs 1B byte-diff (Scenario 1.4)
add_executable(test_legacy_vector_add_standalone test_legacy_vector_add.cu)
target_link_libraries(test_legacy_vector_add_standalone cudart_stub)
```

Modify `tests/e2e/CMakeLists.txt`: add `add_subdirectory(path_1A_legacy_ptx)` (at end).

- [ ] **Step 3: Build + run diff between Path 1A and 1B**

```bash
cd build && cmake --build . --target test_legacy_vector_add_standalone
cd .worktrees/fix-path-coverage-gaps/build/bin && \
  diff <(./test_legacy_vector_add_standalone) \
       <(../tests/e2e/path_1B_ptxir_fatbinary/path_1B_standalone vector_add)
```

Expected: identical stdout (both print "OK: vector_add(N=1024) sum=3072").

- [ ] **Step 4: Defer commit**

Continue to Task 4.

---

## Task 4: Phase 2 — Path 1C Driver API real kernel exec (cuModuleLoadData with PTXIR blob)

**Files:**
- Create: `tests/e2e/path_1C_driver_api/CMakeLists.txt`
- Create: `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`
- Create: `tests/e2e/path_1C_driver_api/build_ptxir_blob.py` (build 9-byte header blob)
- Create: `tests/e2e/path_1C_driver_api/vec_add.cu` (single-kernel source)
- Modify: `.gitignore` (whitelist `!tests/e2e/path_1C_driver_api/**`)
- Modify: `tests/e2e/CMakeLists.txt` (add `add_subdirectory(path_1C_driver_api)`)

- [ ] **Step 1: Verify the actual PTXIR_EMBED_MAGIC value**

```bash
grep -n "PTXIR_EMBED_MAGIC\|MAGIC\[" /workspace/project/PTX-EMU/tools/ptxir_embed.cpp | head -5
grep -n "PTXIR_EMBED_MAGIC\|MAGIC" /workspace/project/PTX-EMU/src/ptxir/*.cpp 2>/dev/null | head -5
```

Expected: 8-byte literal (e.g. `{'P','T','X','I','R','_','E','M'}` = "PTXIR_EM"). Use this value
in `build_ptxir_blob.py` to strip the trailer correctly.

- [ ] **Step 2: Write failing test (Scenario 2.1 — cuModuleLoadData with PTXIR blob)**

Create `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cuda.h>
#include <vector>
#include <numeric>
#include <cstdio>

// IMPORTANT: cuModuleLoad (cudart_sim.cpp:510) is a STUB that always returns success
// without registering kernels. We MUST use cuModuleLoadData (line 519) which goes
// through global_registry().insert() with real PTXIRLoader deserialization.
// The blob format expected: "PTXIR" (5 bytes) + LE u32 size + body.

TEST_CASE("Path 1C Scenario 2.1: cuModuleLoadData full chain", "[e2e][path_1C]") {
    CUmodule mod;
    CUfunction func;
    CUdeviceptr da, db, dc;
    REQUIRE(cuInit(0) == CUDA_SUCCESS);

    // Load PTXIR-prefixed blob (5-byte magic + 4-byte LE u32 size + body) from disk
    FILE* f = fopen("./vec_add.ptxir_blob", "rb");
    REQUIRE(f != nullptr);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> blob(sz);
    REQUIRE(fread(blob.data(), 1, sz, f) == (size_t)sz);
    fclose(f);

    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);

    const int N = 1024;
    std::vector<int> a(N, 1), b(N, 2), c(N, 0);
    cuMemAlloc(&da, N*4); cuMemAlloc(&db, N*4); cuMemAlloc(&dc, N*4);
    cuMemcpyHtoD(da, a.data(), N*4);
    cuMemcpyHtoD(db, b.data(), N*4);

    void* args[] = {&da, &db, &dc, (void*)&N};
    REQUIRE(cuLaunchKernel(func, N/256, 1, 1, 256, 1, 1, 0, 0, args, nullptr) == CUDA_SUCCESS);
    cuMemcpyDtoH(c.data(), dc, N*4);
    int sum = std::accumulate(c.begin(), c.end(), 0);
    REQUIRE(sum == 3072);  // (1+2) * 1024

    cuMemFree(da); cuMemFree(db); cuMemFree(dc);
    cuModuleUnload(mod);
}
```

- [ ] **Step 3: Run test — expect FAIL ("vec_add.ptxir_blob not found")**

Run: `cd build && ctest -L path_1C --output-on-failure -R "Scenario 2.1"`
Expected: FAIL because PTXIR blob not yet built.

- [ ] **Step 4: Create `vec_add.cu` kernel source**

Create `tests/e2e/path_1C_driver_api/vec_add.cu`:

```cuda
__global__ void vec_add(const int* a, const int* b, int* c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
```

- [ ] **Step 5: Create `build_ptxir_blob.py` helper**

Create `tests/e2e/path_1C_driver_api/build_ptxir_blob.py`:

```python
#!/usr/bin/env python3
"""
Build PTXIR-prefixed blob for cuModuleLoadData.
cuModuleLoadData expects: "PTXIR" (5 bytes) + LE u32 size + body.
ptxir_embed appends a different format (prefix + section + LE u32 size + 8-byte magic at END).
So we extract the body from ptxir_embed output and re-prefix it.
"""
import struct
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).parent
PTXEMU_ROOT = HERE.parent.parent.parent
PTXIR_EMBED = PTXEMU_ROOT / "build" / "bin" / "ptxir_embed"

KERNELS_CU = HERE / "vec_add.cu"
CUBIN_OUT = HERE / "vec_add.cubin"
PTX_OUT = HERE / "vec_add.ptx"
EMBED_OUT = HERE / "vec_add.embedded.cubin"
BLOB_OUT = HERE / "vec_add.ptxir_blob"

def main():
    # 1. nvcc → .cubin
    subprocess.check_call([
        "nvcc", "-arch=sm_100", "-O2", "-c", str(KERNELS_CU),
        "-o", str(CUBIN_OUT), "--cubin"
    ])

    # 2. ptxir_embed --in-cubin X.cubin --in-ptx X.ptx --kernel-name K --out X.embedded
    subprocess.check_call([
        str(PTXIR_EMBED),
        "--in-cubin", str(CUBIN_OUT),
        "--in-ptx", str(PTX_OUT),
        "--kernel-name", "vec_add",
        "--out", str(EMBED_OUT),
    ])

    # 3. Extract PTXIR section body from embedded cubin
    # Format from tools/ptxir_embed.cpp: [prefix][ptxir_body][u32_size_le][8_byte_magic]
    embedded = EMBED_OUT.read_bytes()
    # Trailer: 4-byte size (LE) + 8-byte magic at end (12 bytes total)
    # OR: 4-byte size + 4-byte size + 8-byte magic (16 bytes if there are 2 size fields)
    # Inspect actual format with first 16 bytes of trailer
    trailer = embedded[-16:]
    # Most likely: [u32_size1][u32_size2][8-byte magic]
    # Read both u32 sizes; the inner body size is the second one (body length)
    size1, size2 = struct.unpack("<II", trailer[:8])
    magic = trailer[8:16]
    print(f"size1={size1} size2={size2} magic={magic!r}")

    # The body is the bytes BEFORE the trailer
    body = embedded[:-(16)]

    # 4. Build cuModuleLoadData-format blob: "PTXIR" (5 bytes) + LE u32 size + body
    blob = b"PTXIR" + struct.pack("<I", len(body)) + body
    BLOB_OUT.write_bytes(blob)
    print(f"wrote {len(blob)} bytes to {BLOB_OUT}")

if __name__ == "__main__":
    main()
```

NOTE: If the trailer format differs from the assumption, adjust offsets. Run the script manually
first to print size1/size2/magic and verify the body extraction.

- [ ] **Step 6: Run `build_ptxir_blob.py` + rerun test**

```bash
cd tests/e2e/path_1C_driver_api && python3 build_ptxir_blob.py
cd build && ctest -L path_1C -R "Scenario 2.1" --output-on-failure
```

Expected: PASS (sum == 3072).

- [ ] **Step 7: Add Scenarios 2.2 (duplicate), 2.3 (not-found), 2.4 (cuLaunchKernel error), 2.5 (unload invalidates)**

Append to test file:
```cpp
TEST_CASE("Path 1C Scenario 2.2: duplicate module load", "[e2e][path_1C]") {
    CUmodule m1, m2;
    REQUIRE(cuModuleLoadData(&m1, load_blob("./vec_add.ptxir_blob").data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoadData(&m2, load_blob("./vec_add.ptxir_blob").data()) == CUDA_SUCCESS);
    REQUIRE(m1 != m2);  // independent handles
    cuModuleUnload(m1); cuModuleUnload(m2);
}

TEST_CASE("Path 1C Scenario 2.3: kernel name not found", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "nonexistent_kernel") == CUDA_ERROR_NOT_FOUND);
    cuModuleUnload(mod);
}

TEST_CASE("Path 1C Scenario 2.4: cuLaunchKernel invalid grid", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func; CUdeviceptr buf;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    const int N = 1024;
    cuMemAlloc(&buf, N*4);
    void* args[] = {&buf, &buf, &buf, (void*)&N};
    // 0 grid size → invalid
    REQUIRE(cuLaunchKernel(func, 0, 1, 1, 256, 1, 1, 0, 0, args, nullptr) != CUDA_SUCCESS);
    cuMemFree(buf);
    cuModuleUnload(mod);
}

TEST_CASE("Path 1C Scenario 2.5: cuModuleUnload invalidates func2name", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func;
    auto blob = load_blob("./vec_add.ptxir_blob");
    REQUIRE(cuModuleLoadData(&mod, blob.data()) == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    REQUIRE(cuModuleUnload(mod) == CUDA_SUCCESS);
    // After unload, func2name[id] should be cleared (per cudart_sim.cpp:573-592)
    CUfunction func2;
    auto rc = cuModuleGetFunction(&func2, mod, "vec_add");
    REQUIRE(rc != CUDA_SUCCESS);
}
```

Add `load_blob` helper to test file:
```cpp
static std::vector<uint8_t> load_blob(const char* path) {
    FILE* f = fopen(path, "rb");
    REQUIRE(f != nullptr);
    fseek(f, 0, SEEK_END); long sz = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<uint8_t> data(sz);
    REQUIRE(fread(data.data(), 1, sz, f) == (size_t)sz);
    fclose(f);
    return data;
}
```

- [ ] **Step 8: Create CMakeLists.txt for Path 1C**

Create `tests/e2e/path_1C_driver_api/CMakeLists.txt`:

```cmake
add_catch_test(e2e_cuda_driver_exec test_cuda_driver_exec.cpp)
set_tests_properties(e2e_cuda_driver_exec PROPERTIES
    LABELS "e2e;path_1C" TIMEOUT 60
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)
add_custom_command(TARGET e2e_cuda_driver_exec PRE_BUILD
    COMMAND python3 ${CMAKE_CURRENT_SOURCE_DIR}/build_ptxir_blob.py
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/vec_add.cu
)
```

Add to `.gitignore`:
```
!tests/e2e/path_1C_driver_api/**
```

- [ ] **Step 9: Defer commit**

Continue to Task 5.

---

## Task 5: Phase 3 — Path 2D Image Executor output correctness baseline

**Files:**
- Create: `tests/e2e/path_2D_image_executor/CMakeLists.txt`
- Create: `tests/e2e/path_2D_image_executor/test_image_executor_output.cpp`
- Create: `tests/ptxir/baselines/` (directory)
- Create: `tests/ptxir/baselines/baseline_format.md`
- Create: `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` (committed)
- Modify: `tests/integration/cudart/test_libptxemu_device.cpp` (DUMP_OUTPUT env var + correctness check + 4 error paths)
- Modify: `.gitignore` (whitelist baselines + path_2D)
- Modify: `tests/e2e/CMakeLists.txt` (add `add_subdirectory(path_2D_image_executor)`)

- [ ] **Step 1: Create baselines directory**

```bash
mkdir -p tests/ptxir/baselines && touch tests/ptxir/baselines/.gitkeep
```

- [ ] **Step 2: Write failing test (Scenario 3.1)**

Create `tests/e2e/path_2D_image_executor/test_image_executor_output.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>
#include <cstdint>

TEST_CASE("Path 2D Scenario 3.1: cute_rmsnorm output byte-level matches baseline", "[e2e][path_2D]") {
    std::ifstream f("../../ptxir/baselines/cute_rmsnorm_output_baseline.bin", std::ios::binary);
    REQUIRE(f.good());
    std::vector<uint8_t> baseline((std::istreambuf_iterator<char>(f)), {});
    REQUIRE(baseline.size() >= 14);  // 10-byte magic + 4-byte size

    // 10-byte magic: "PTXR_OUT\0\0"
    const char expected_magic[10] = {'P','T','X','R','_','O','U','T', 0, 0};
    REQUIRE(std::memcmp(baseline.data(), expected_magic, 10) == 0);

    // 4-byte LE u32 size at offset 10
    uint32_t size;
    std::memcpy(&size, baseline.data() + 10, 4);
    REQUIRE(size == baseline.size() - 14);

    // (Exhaustiveness: actual simulator output comparison happens in test_libptxemu_device.cpp
    //  with DUMP_OUTPUT=1, this test verifies the BASELINE FILE integrity + format)
}
```

- [ ] **Step 3: Run test — expect FAIL (baseline file doesn't exist yet)**

Run: `cd build && ctest -L path_2D --output-on-failure -R "Scenario 3.1"`
Expected: FAIL with "baseline file not found".

- [ ] **Step 4: Add DUMP_OUTPUT env var support to test_libptxemu_device.cpp**

Modify `tests/integration/cudart/test_libptxemu_device.cpp` to add a new test case that:
1. Loads cute_rmsnorm.ptxir via libptxemu_device API
2. Executes the kernel
3. If env var `DUMP_OUTPUT=1` is set, writes the output buffer to `/tmp/cute_rmsnorm_out.bin`
4. Verifies the output (without DUMP_OUTPUT, just normal correctness check)

```cpp
TEST_CASE("cute_rmsnorm output dump for baseline generation", "[integration][DUMP_OUTPUT]") {
    // ... load + execute cute_rmsnorm kernel via libptxemu_device ...
    // ... copy output to std::vector<uint8_t> out ...
    if (const char* dump = std::getenv("DUMP_OUTPUT"); dump && std::string(dump) == "1") {
        std::ofstream f("/tmp/cute_rmsnorm_out.bin", std::ios::binary);
        f.write((const char*)out.data(), out.size());
        std::cout << "dumped " << out.size() << " bytes to /tmp/cute_rmsnorm_out.bin\n";
    }
    // Normal correctness check (asserts against known-expected output, NOT against baseline)
    REQUIRE(out.size() == EXPECTED_SIZE);
    // Optionally: REQUIRE(std::memcmp(out.data(), expected, out.size()) == 0);
}
```

NOTE: Verify the actual libptxemu_device API (load/execute/unload function names + signatures)
by reading `tests/integration/cudart/test_libptxemu_device.cpp` first to match the existing test style.

- [ ] **Step 5: Build + run with DUMP_OUTPUT=1, then prefix with magic**

```bash
cd build && cmake --build .
cd build && ctest -R "cute_rmsnorm output dump" --output-on-failure -V \
    --environment DUMP_OUTPUT=1
python3 -c "
import struct
data = open('/tmp/cute_rmsnorm_out.bin', 'rb').read()
out = b'PTXR_OUT\0\0' + struct.pack('<I', len(data)) + data
open('tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin', 'wb').write(out)
print(f'wrote {len(out)} bytes (10 magic + 4 size + {len(data)} body)')
"
```

Expected: file created at `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` with 10+4+body bytes.

- [ ] **Step 6: Document baseline format (10-byte magic + 4-byte LE u32 size)**

Create `tests/ptxir/baselines/baseline_format.md`:

```markdown
# Cute RMSNorm Output Baseline Format

Each `.bin` file in this directory is a magic-prefixed binary:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0      | 10   | magic | `PTXR_OUT\0\0` (10 bytes literal: P T X R _ O U T NUL NUL) |
| 10     | 4    | size  | LE u32 — byte count of payload that follows |
| 14     | size | bytes | Raw output buffer from simulator |

**Total minimum size: 14 bytes** (empty payload = 10 + 4 + 0).

**Stability contract**: baselines are immutable once committed. Any change requires:
1. New file with versioned name (e.g. `cute_rmsnorm_output_v2.bin`)
2. Update `current_baseline` symlink
3. Document D3 mutation in design doc

**Regeneration**:
```bash
cd build && ctest -R "cute_rmsnorm output dump" -V --environment DUMP_OUTPUT=1
python3 -c "import struct; data=open('/tmp/cute_rmsnorm_out.bin','rb').read(); \
    open('tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin','wb') \
    .write(b'PTXR_OUT\0\0' + struct.pack('<I',len(data)) + data)"
```
```

- [ ] **Step 7: Re-run test — expect PASS**

Run: `cd build && ctest -L path_2D --output-on-failure`
Expected: PASS (magic + size + bytes all match).

- [ ] **Step 8: Add Scenario 3.7 (D3 mutation regression, NO RED_PHASE label — always passes)**

```cpp
TEST_CASE("Path 2D Scenario 3.7: D3 mutation regression sentinel", "[e2e][path_2D]") {
    // Sentinel: verifies baseline file integrity. If this test fails, a baseline
    // mutation has occurred and must be reviewed.
    std::ifstream f("../../ptxir/baselines/cute_rmsnorm_output_baseline.bin", std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    REQUIRE(b.size() >= 14);
    REQUIRE(std::memcmp(b.data(), "PTXR_OUT\0\0", 10) == 0);
    uint32_t size; std::memcpy(&size, b.data() + 10, 4);
    REQUIRE(size == b.size() - 14);
}
```

- [ ] **Step 9: Add 4 error path tests in test_libptxemu_device.cpp**

Append to `tests/integration/cudart/test_libptxemu_device.cpp`:

```cpp
TEST_CASE("error_path_1: load garbage pointer", "[integration][error_path]") {
    REQUIRE(libptxemu_load_image(nullptr) != ptxemu_success);
}
TEST_CASE("error_path_2: execute invalid handle", "[integration][error_path]") {
    ptxemu_kernel_handle_t h = 0xDEADBEEF;
    REQUIRE(libptxemu_execute(h, nullptr, 0) != ptxemu_success);
}
TEST_CASE("error_path_3: unload invalid handle", "[integration][error_path]") {
    REQUIRE(libptxemu_unload(0xDEADBEEF) != ptxemu_success);
}
TEST_CASE("error_path_4: kernel_name not in module", "[integration][error_path]") {
    // Load a valid image, request bogus kernel name
    ptxemu_image_t img = load_test_image("vec_add");
    REQUIRE(libptxemu_get_kernel(img, "nonexistent_kernel", nullptr) != ptxemu_success);
}
```

NOTE: API names (`libptxemu_load_image`, `libptxemu_execute`, `libptxemu_unload`, `libptxemu_get_kernel`,
`load_test_image`, `ptxemu_image_t`, `ptxemu_kernel_handle_t`, `ptxemu_success`) need to match the
actual libptxemu_device API. Read the existing test file to align.

- [ ] **Step 10: Create Path 2D CMakeLists.txt + update .gitignore**

Create `tests/e2e/path_2D_image_executor/CMakeLists.txt`:

```cmake
add_catch_test(e2e_image_executor_output test_image_executor_output.cpp)
set_tests_properties(e2e_image_executor_output PROPERTIES
    LABELS "e2e;path_2D" TIMEOUT 60
)
```

Modify `tests/e2e/CMakeLists.txt`: add `add_subdirectory(path_2D_image_executor)`.

Add to `.gitignore`:
```
!tests/e2e/path_2D_image_executor/**
!tests/ptxir/baselines/*.bin
!tests/ptxir/baselines/*.md
!tests/ptxir/baselines/.gitkeep
```

- [ ] **Step 11: Defer commit**

Continue to Task 6.

---

## Task 6: Phase 4 — `tests/e2e/` reorganization into path-X directories

**Files:**
- Create: `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt` (expand minimal from Task 3)
- Modify: `tests/e2e/CMakeLists.txt` (drop specific lines, add 4 subdirectories)

- [ ] **Step 1: Enumerate current files to be moved**

```bash
ls tests/e2e/kernel/ tests/e2e/divergence/
```

Expected: see all test_*.cu and test_*.cpp files in those dirs.

- [ ] **Step 2: `git mv` existing files to path_X subdirectories (preserves history via --follow)**

```bash
# Path 1A (legacy PTX)
git mv tests/e2e/kernel/test_blackwell_gemm.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/kernel/test_tcgen05_reduction.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/kernel/test_tcgen05_warpgroup.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/divergence/test_divergence.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/divergence/test_divergence_sync.cu tests/e2e/path_1A_legacy_ptx/

# Path 1B (PTXIR fat-binary format-level)
git mv tests/e2e/kernel/test_ptxir_cubin_embed.cpp tests/e2e/path_1B_ptxir_fatbinary/
```

Verify: `git log --follow <file>` shows original history preserved.

- [ ] **Step 3: Expand `path_1A_legacy_ptx/CMakeLists.txt` (replace Task 3 minimal version)**

```cmake
# Path 1A: Legacy PTX (kernel directly compiled, no PTXIR)
add_catch_test(e2e_blackwell_gemm test_blackwell_gemm.cu)
set_tests_properties(e2e_blackwell_gemm PROPERTIES LABELS "e2e;path_1A")
add_catch_test(e2e_tcgen05_reduction test_tcgen05_reduction.cu)
set_tests_properties(e2e_tcgen05_reduction PROPERTIES LABELS "e2e;path_1A;tcgen05")
add_catch_test(e2e_tcgen05_warpgroup test_tcgen05_warpgroup.cu)
set_tests_properties(e2e_tcgen05_warpgroup PROPERTIES LABELS "e2e;path_1A;tcgen05")
add_catch_test(e2e_divergence test_divergence.cu)
set_tests_properties(e2e_divergence PROPERTIES LABELS "e2e;path_1A;divergence;warp")
add_catch_test(e2e_divergence_sync test_divergence_sync.cu)
set_tests_properties(e2e_divergence_sync PROPERTIES LABELS "e2e;path_1A;divergence;barrier_sync")

# Standalone binary for Path 1A vs 1B byte-diff (Scenario 1.4)
add_executable(test_legacy_vector_add_standalone test_legacy_vector_add.cu)
target_link_libraries(test_legacy_vector_add_standalone cudart_stub)
```

- [ ] **Step 4: Modify `tests/e2e/CMakeLists.txt` — DELETE specific lines, ADD subdirectories**

**DELETE** (these tests now live in path_1A_legacy_ptx/CMakeLists.txt or path_1B_ptxir_fatbinary/CMakeLists.txt):
- `add_catch_test(e2e_divergence ...)` (line ~34-37)
- `add_catch_test(e2e_divergence_sync ...)` (line ~38-42)
- `add_catch_test(e2e_blackwell_gemm ...)` (line ~65-72)
- `add_catch_test(e2e_tcgen05_reduction ...)` (line ~77-85)
- `add_catch_test(e2e_tcgen05_warpgroup ...)` (line ~86-123)
- `add_catch_test(e2e_ptxir_cubin_embed ...)` (line ~162-167)

**KEEP** in `tests/e2e/CMakeLists.txt` (per tasks.md 4.11):
- `e2e_test3_cfg_full` (line ~16-19) — kernel/test_test3_cfg_full.cpp
- `e2e_barrier_warp_sync` (line ~21-24)
- `e2e_ldglobal_simple` (line ~26-29)
- `e2e_shared_memory_*` (line ~47-60)
- `e2e_flashattention_mini` (line ~130-135) — kernel/test_flashattention_mini.cu
- `e2e_printf_*` (line ~139-159)
- `tests/e2e/cosim/*` (all)

**ADD** at end of file:
```cmake
add_subdirectory(path_1A_legacy_ptx)
add_subdirectory(path_1B_ptxir_fatbinary)
add_subdirectory(path_1C_driver_api)
add_subdirectory(path_2D_image_executor)
```

- [ ] **Step 5: Run full ctest with path-X labels to verify isolation**

```bash
cd build && cmake --build . && \
    for p in 1A 1B 1C 2D; do
        echo "=== Path $p ==="
        ctest -L path_$p --output-on-failure
    done
```

Expected: each `-L path_X` runs ONLY that path's tests. AC-4.3/4.4/4.5 satisfied.

- [ ] **Step 6: Run full regression to verify no breakage**

```bash
cd build && ctest --output-on-failure -L "e2e;integration;unit"
```

Expected: 100% pass rate.

- [ ] **Step 7: Verify file history preservation**

```bash
git log --follow tests/e2e/path_1A_legacy_ptx/test_blackwell_gemm.cu --oneline | head -5
```

Expected: shows original commits from before the rename.

- [ ] **Step 8: Defer commit**

Continue to Task 7.

---

## Task 7: Phase 5 — Proposal doc consistency fix (silent descoping disclosure)

**Files:**
- Modify: `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md`

- [ ] **Step 1: Locate the §Capabilities bullet claiming e2e validates "PTX-EMU 加载 + ptxir_extract"**

```bash
grep -n "test_ptxir_cubin_embed\|e2e 测试\|加载\|ptxir_extract" \
    openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
```

Expected: line number(s) of the misleading bullet.

- [ ] **Step 2: Append disclaimer inline (do NOT rewrite the bullet)**

After the bullet, add a `<br>**[勘误: 2026-08-12, see fix-path-coverage-gaps]**` note:

```markdown
- `e2e-test-ptxir-cubin-embed`: ...
  <br>**[勘误: 2026-08-12, see fix-path-coverage-gaps]** — 此 e2e 验证 PTXIR-Embedded CUBIN
  格式兼容性（Phase 12.2 R5 / ADR-0024 Risk 1），**不验证 PTX-EMU 真实加载执行**。真实加载执行
  验证见 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`（D-PTX-7，关闭于
  `docs/audits/D-PTX-debt-registry-ERRATA.md`）。
```

- [ ] **Step 3: Verify cross-reference**

```bash
grep -n "fix-path-coverage-gaps\|D-PTX-7" \
    openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
```

Expected: ≥2 mentions (change name + debt number).

- [ ] **Step 4: Verify archive directory name unchanged (AC-5.5)**

```bash
ls openspec/changes/archive/ | grep 2026-08-07
```

Expected: `2026-08-07-implement-ptxir-cubin-embed-extension` still exists.

- [ ] **Step 5: Defer commit**

Continue to Task 8.

---

## Task 8: Acceptance + Single Consolidated Commit

**Files:** (consolidates all changes from Tasks 0-7 into one commit per `worktree-archive-workflow`)

- [ ] **Step 1: Verify all tests pass before commit**

```bash
cd build && cmake --build . && ctest --output-on-failure -L "e2e;integration;unit"
```

Expected: 100% pass.

- [ ] **Step 2: Verify clang-format clean on all changed files**

```bash
clang-format --dry-run --Werror \
    tests/e2e/path_1B_ptxir_fatbinary/*.cpp tests/e2e/path_1B_ptxir_fatbinary/*.cu \
    tests/e2e/path_1B_ptxir_fatbinary/*.h tests/e2e/path_1B_ptxir_fatbinary/*.sh \
    tests/e2e/path_1C_driver_api/*.cpp tests/e2e/path_1C_driver_api/*.cu \
    tests/e2e/path_1C_driver_api/*.py \
    tests/e2e/path_2D_image_executor/*.cpp \
    tests/e2e/path_1A_legacy_ptx/CMakeLists.txt tests/e2e/path_1A_legacy_ptx/*.cu \
    tests/e2e/CMakeLists.txt \
    tests/ptxir/baselines/baseline_format.md \
    tests/integration/cudart/test_libptxemu_device.cpp \
    docs/audits/D-PTX-debt-registry-ERRATA.md
```

Expected: exit code 0.

- [ ] **Step 3: Update tasks.md checkboxes (mark all completed)**

Edit `openspec/changes/fix-path-coverage-gaps/tasks.md`: change all `- [ ]` to `- [x]` for completed items.

- [ ] **Step 4: Verify openspec validate**

```bash
openspec validate fix-path-coverage-gaps
```

Expected: PASS, 0 issues.

- [ ] **Step 5: Update tasks.md §6 to mark ACs verified**

For each AC (G1-G5, N1-N2, M1-M4), add a one-line `<!-- verified YYYY-MM-DD -->` comment.

- [ ] **Step 6: Single consolidated commit**

```bash
cd .worktrees/fix-path-coverage-gaps
git add -A
git status --short  # verify all changes captured
git commit -m "test(e2e): close 4-path cudart coverage gap + D-PTX-7/D-PTX-8

Closes D-PTX-7: PTXIR fat-binary 端到端未验证
  - tests/e2e/path_1B_ptxir_fatbinary/ — Scenario 1.1/1.2/1.3/1.5 + anti-fallback
  - Standalone nvcc+cubin+ptxir_embed binary harness

Closes D-PTX-8: Driver API 真实成功 kernel 执行未验证
  - tests/e2e/path_1C_driver_api/ — Scenario 2.1/2.2/2.3/2.4/2.5
  - Uses cuModuleLoadData (real) NOT cuModuleLoad (stub at cudart_sim.cpp:510)

Output correctness 1/4 → 4/4:
  - tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin (10-byte magic + u32 size)
  - tests/ptxir/baselines/baseline_format.md documents format
  - 4 error path tests in test_libptxemu_device.cpp

Reorganization (Phase 4):
  - 4 path_X/CMakeLists.txt with explicit LABELS (e2e;path_X)
  - ctest -L path_X for single-path regression
  - git mv preserves file history (--follow verified)

Doc consistency (Phase 5):
  - inline [勘误] in implement-ptxir-cubin-embed-extension proposal
  - cross-references fix-path-coverage-gaps + D-PTX-7

Debt registry (prerequisite):
  - docs/audits/D-PTX-debt-registry-ERRATA.md (D-PTX-7 + D-PTX-8)

4-path cudart coverage: 3/4 → 4/4 (AC-M1)
output-correctness: 1/4 → 4/4 (AC-M2)
openspec doc consistency fix: 1 place (AC-M3)
ctest -L path_X single-path regression (AC-M4)"
```

- [ ] **Step 7: Verify commit log**

```bash
git log --oneline -3
```

Expected: single new commit on top of `openspec/fix-path-coverage-gaps` branch.

---

## Acceptance Criteria Mapping

| AC | Description | Verified in |
|----|-------------|-------------|
| G1 | sanity.sh passes | Task 8 Step 1 (ctest) + manual `./scripts/sanity.sh` |
| G2 | ctest -L e2e+integration+unit 100% pass | Task 8 Step 1 |
| G3 | regression.sh passes | Manual after Task 8 |
| G4 | clang-format clean on changed files | Task 8 Step 2 |
| G5 | 5 Phases shipped + iteration.json synced | Archive phase (next: `guide-ship` Phase 3) |
| N1 | new tests keep `e2e_` prefix | Task 8 Step 1 (target names visible) |
| N2 | new tests LABELS contain `e2e` | Task 8 Step 1 (`ctest -V` shows labels) |
| M1 | cudart path coverage 3/4 → 4/4 | Task 6 Step 5 (per-path `-L path_X` runs) |
| M2 | output-correctness 1/4 → 4/4 | Task 5 Step 7 (Scenario 3.1 PASS) |
| M3 | openspec doc consistency fix (1 place) | Task 7 Step 2 (disclaimer added) |
| M4 | `ctest -L path_1X` is single-path regression | Task 6 Step 5 |

---

## Out of Scope (per design.md Non-Goals)

- Production code changes (`cudart_sim.cpp`, `cpptlm_module.cpp`, `ptxir_loader.cpp` untouched)
- `multi-entry-handle-api` task unchecking (separate improvement)
- New test framework (still Catch2 + `add_catch_test`)
- New PTXIR fixture generation tool beyond `build_ptxir_blob.py` (Path 1C helper)
- openspec CLI / validate rule changes
- ctest label scheme (only additions)
- Top-level `tests/` structure (only `tests/e2e/` subtree)

---

## Self-Review Checklist (v2 — post-Metis)

- [x] Spec Coverage: All 5 Phases + ACs from tasks.md → 9 Tasks (0-8)
- [x] No Placeholders: Every step has actual code, file paths, run commands, expected output
- [x] Type Consistency: `e2e_ptxir_fatbinary_exec`, `e2e_cuda_driver_exec`, `e2e_image_executor_output` used consistently
- [x] File Paths: All Create/Modify entries reference real files (verified via `ls` + `grep`)
- [x] TDD Order: Each task follows Red→Verify-fail→Green→Verify-pass
- [x] CRIT-1 fixed: ptxir_embed CLI uses `--in-cubin/--in-ptx/--kernel-name/--out`
- [x] CRIT-2 fixed: Task 0 creates D-PTX-7/D-PTX-8 ERRATA before any "Closes" claim
- [x] CRIT-3 fixed: Single consolidated commit at Task 8 Step 6 (per `worktree-archive-workflow`)
- [x] CRIT-4 fixed: `git mv --dry-run` replaced with `ls` enumeration
- [x] CRIT-5 fixed: Specific add_catch_test lines (34-42, 65-72, 77-123, 162-167) to delete
- [x] CRIT-6 fixed: cuModuleLoadData (real) + 9-byte PTXIR blob via build_ptxir_blob.py
- [x] HIGH-1 fixed: Standalone Path 1A binary for byte-diff
- [x] HIGH-2 fixed: Note about 1-kernel fallback if multi-entry rejected
- [x] HIGH-3 fixed: PTXIR_MODE=auto env var (not just PATH=) for anti-fallback
- [x] HIGH-4 fixed: DUMP_OUTPUT=1 in test_libptxemu_device.cpp (no new tool)
- [x] HIGH-5 fixed: 10-byte magic + 14-byte minimum (consistent everywhere)
- [x] HIGH-6 fixed: mkdir -p tests/ptxir/baselines in Task 5 Step 1
- [x] MED-5 fixed: Pre-Phase worktree creation at Pre-Phase Step 0b
- [x] MED-4 fixed: env.sh sourced at Pre-Phase Step 0a
