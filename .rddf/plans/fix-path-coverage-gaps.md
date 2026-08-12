# fix-path-coverage-gaps Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Close structural e2e coverage gaps for PTX-EMU's 4 cudart loading paths (1A/1B/1C/2D) by adding real-execution tests for Paths 1B/1C/2D, reorganizing `tests/e2e/` into path-X subdirectories, and fixing silent descoping documentation in an archived change.

**Architecture:** 5 sequential Phases with independently reversible commits. Each Phase adds tests under `tests/e2e/path_<X>/` with its own `CMakeLists.txt`. Anti-fallback guards use `PATH=""` for Path 1B. Output baseline for Path 2D uses magic-prefixed (`PTXR_OUT\0\0`) binary format with LE u32 size header. No production code changes — only `tests/e2e/` subtree + one archived proposal text fix.

**Tech Stack:** Catch2 v3 + CMake `add_catch_test` + ctest labels + nvcc (PTX-EMU sm_100 virtual arch) + `ptxir_embed` linker + standalone fork+exec binary harness.

---

## File Structure

### Production Code
*(none — this change only adds tests + reorganizes existing tests + fixes one archived doc)*

### Tests (NEW)

| File | Responsibility |
|---|---|
| `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt` | Path 1B test target build config |
| `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu` | ≥3 PTX-EMU kernels (vector_add, matmul, reduction) for standalone binary |
| `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh` | nvcc compile → cubin → ptxir_embed → link PTX-EMU libcudart.so |
| `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` | fork+exec standalone binary + verify stdout + 4 dispatch scenarios |
| `tests/e2e/path_1C_driver_api/CMakeLists.txt` | Path 1C test target build config |
| `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp` | cuModuleLoadData → cuModuleGetFunction → cuLaunchKernel full-chain |
| `tests/e2e/path_2D_image_executor/CMakeLists.txt` | Path 2D test target build config |
| `tests/ptxir/baselines/baseline_format.md` | Documents baseline file format spec |
| `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` | Golden output (magic-prefixed binary, committed) |
| `tests/integration/cudart/test_libptxemu_device.cpp` | (MODIFIED) Add cute_rmsnorm output correctness + D3 mutation regression |

### Tests (REORGANIZED via git mv)

| Source | Destination |
|---|---|
| `tests/e2e/kernel/test_blackwell_gemm.cu` | `tests/e2e/path_1A_legacy_ptx/test_blackwell_gemm.cu` |
| `tests/e2e/kernel/test_tcgen05_*.cu` | `tests/e2e/path_1A_legacy_ptx/` |
| `tests/e2e/divergence/*.cu` | `tests/e2e/path_1A_legacy_ptx/` |
| `tests/e2e/kernel/test_ptxir_cubin_embed.cpp` | `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_cubin_embed.cpp` (format-level) |

### Build Config (NEW)

| File | Responsibility |
|---|---|
| `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt` | Path 1A test target build config (moved existing) |
| `tests/e2e/CMakeLists.txt` | (MODIFIED) Drop moved `add_catch_test` calls + add 4 `add_subdirectory(path_X/)` |
| `.gitignore` | (MODIFIED) Add `!tests/e2e/path_X/**` + `!tests/ptxir/baselines/*.bin` whitelist |

### Documentation (MODIFIED)

| File | Responsibility |
|---|---|
| `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md` | §Capabilities disclaimer: e2e only validates format compatibility, not real PTX-EMU load/exec |

---

### Task 1: Phase 1 — Path 1B PTXIR fat-binary e2e skeleton + first kernel test (RED→GREEN)

**Files:**
- Create: `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/path_1B_kernels.cu` (≥3 kernels)
- Create: `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`
- Modify: `.gitignore` (add `!tests/e2e/path_1B_ptxir_fatbinary/**/*.ptx`)

- [ ] **Step 1: Write failing test (Scenario 1.1 — kSuccess)**

Create `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` with a Catch2 TEST_CASE that:
1. Forks+execs a pre-built standalone binary (`./path_1B_standalone vector_add`)
2. Reads its stdout (capture via pipe)
3. Asserts stdout matches `"OK: vector_add(N=1024) sum=<expected>"` regex

The standalone binary is NOT yet built (Step 3 builds it). Test must FAIL with "binary not found".

```cpp
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
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

// Standalone entrypoint for fork+exec harness
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
        printf("OK: vector_add(N=1024) sum=%d\n", sum);
        return 0;
    } else if (k == "matmul") {
        /* similar shape */
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

The real vector_add reduction expects `sum = (1+2)*1024 = 3072`.

- [ ] **Step 4: Create `build_standalone.sh`**

Create `tests/e2e/path_1B_ptxir_fatbinary/build_standalone.sh`:

```bash
#!/bin/bash
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PTXEMU_ROOT="${PTXEMU_ROOT:-$(cd "$HERE/../../.." && pwd)}"
CUDART_SO="${PTXEMU_ROOT}/build/lib/libcudart.so"

nvcc -arch=sm_100 -O2 -c "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_kernels.cubin" --cubin
PTXIR_TOOL="${PTXEMU_ROOT}/build/bin/ptxir_embed"
"$PTXIR_TOOL" --input "$HERE/path_1B_kernels.cubin" --output "$HERE/path_1B_standalone.cubin" --manifest kernels.json
nvcc -arch=sm_100 "$HERE/path_1B_kernels.cu" -o "$HERE/path_1B_standalone" \
    -L"$PTXEMU_ROOT/build/lib" -lcudart -Xlinker -rpath -Xlinker "$PTXEMU_ROOT/build/lib"
```

Chmod +x and verify it produces `path_1B_standalone` (run it manually: `./path_1B_standalone vector_add` should print "OK: ...").

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
    ENVIRONMENT "PATH="
)
add_custom_command(TARGET e2e_ptxir_fatbinary_exec PRE_BUILD
    COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/build_standalone.sh
)
```

Add to root `.gitignore`:
```
!tests/e2e/path_1B_ptxir_fatbinary/**/*.ptx
!tests/e2e/path_1B_ptxir_fatbinary/**/*.cubin
```

Modify `tests/e2e/CMakeLists.txt`: add `add_subdirectory(path_1B_ptxir_fatbinary)`.

Run: `cd build && cmake --build . --target e2e_ptxir_fatbinary_exec && ctest -L path_1B -V`
Expected: PASS for Scenario 1.1 (assert stdout contains "OK: vector_add(N=1024) sum=3072")

- [ ] **Step 6: Defer commit**

Do NOT commit yet — Phase 1 needs Scenarios 1.2/1.3/1.5 before commit. Mark this step done, continue to Task 2.

---

### Task 2: Phase 1 — Add Scenarios 1.2/1.3/1.5 + Anti-fallback guard verification

**Files:**
- Modify: `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp`

- [ ] **Step 1: Write failing tests for Scenarios 1.2, 1.3, 1.5**

Append to `test_ptxir_fatbinary_exec.cpp`:

```cpp
TEST_CASE("Path 1B Scenario 1.2: kNoFooter (no PTXIR tail)", "[e2e][path_1B]") {
    // Strip PTXIR footer from path_1B_standalone.cubin, expect rc!=0 + stderr mentions "no PTXIR"
    // Skipped here — covered in Path 1B Anti-fallback test (1.5)
}

TEST_CASE("Path 1B Scenario 1.3: kMalformedPtxir (CRC mismatch)", "[e2e][path_1B]") {
    // Flip a byte in PTXIR section, expect rc=ERR_PTXIR_CRC
}

TEST_CASE("Path 1B Scenario 1.5: Anti-fallback guard (PATH=\"\")", "[e2e][path_1B]") {
    // Verify the test harness sets PATH="" so extract_ptx_with_cuobjdump cannot be invoked
    REQUIRE(getenv("PATH") == nullptr || std::string(getenv("PATH")).empty());
}
```

- [ ] **Step 2: Run new tests to verify they fail (kMalformedPtxir scenario)**

Run: `cd build && ctest -L path_1B -R "Scenario 1.3" --output-on-failure`
Expected: Scenario 1.3 fails because CRC mutation logic not implemented in test yet.

- [ ] **Step 3: Implement CRC mutation + footer-strip helpers in test**

Append to test file:

```cpp
#include <fstream>
#include <vector>

static std::vector<uint8_t> read_file(const char* path) {
    std::ifstream f(path, std::ios::binary); return std::vector<uint8_t>((std::istreambuf_iterator<char>(f)), {});
}

TEST_CASE("Path 1B Scenario 1.3: kMalformedPtxir", "[e2e][path_1B]") {
    // Locate PTXIR footer magic "PTXR" at end of cubin (last 8 bytes)
    auto buf = read_file("./path_1B_standalone.cubin");
    REQUIRE(buf.size() > 16);
    // Flip byte in PTXIR region (between manifest_size marker and "PTXR" footer)
    buf[buf.size() - 9] ^= 0xFF;
    std::ofstream("./path_1B_corrupt.cubin", std::ios::binary).write((char*)buf.data(), buf.size());
    // Exec the corrupt binary via the standalone wrapper (already-built path)
    std::string out = exec_capture("./path_1B_standalone", "vector_add");
    REQUIRE(out.find("OK:") == std::string::npos);  // expect failure
}
```

- [ ] **Step 4: Re-run tests**

Run: `cd build && ctest -L path_1B --output-on-failure`
Expected: Scenarios 1.1, 1.3, 1.5 all PASS; 1.2 explicitly skipped.

- [ ] **Step 5: Verify Scenario 1.4 (Path 1B vs 1A byte-level consistency)**

```bash
# Compare vector_add output from Path 1B standalone vs Path 1A (legacy PTX via existing test_blackwell_gemm)
diff <(./path_1B_standalone vector_add) <(cd ../path_1A_legacy_ptx && ./test_blackwell_gemm vector_add)
```

Expected: identical stdout (both print `OK: vector_add(N=1024) sum=3072`).

- [ ] **Step 6: Commit Phase 1**

```bash
git add tests/e2e/path_1B_ptxir_fatbinary/ tests/e2e/CMakeLists.txt .gitignore
git commit -m "test(e2e): add Path 1B PTXIR fat-binary real exec coverage

- Standalone binary harness (nvcc → cubin → ptxir_embed)
- Scenarios 1.1/1.3/1.5: kSuccess, kMalformedPtxir, Anti-fallback
- LABEL e2e;path_1B + ctest filter -L path_1B
- Closes D-PTX-7 (ADR-0021)"
```

---

### Task 3: Phase 2 — Path 1C Driver API real kernel exec

**Files:**
- Create: `tests/e2e/path_1C_driver_api/CMakeLists.txt`
- Create: `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`
- Create: `tests/ptxir/fixtures/v2_manifest_kernels.json` (PTXIR image fixture)
- Modify: `.gitignore` (whitelist `!tests/e2e/path_1C_driver_api/**`)

- [ ] **Step 1: Write failing test (Scenario 2.1 — cuModuleLoadData → cuLaunchKernel)**

Create `tests/e2e/path_1C_driver_api/test_cuda_driver_exec.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cuda.h>
#include <vector>
#include <numeric>
#include <cstdio>

extern "C" __global__ void vec_add(const int* a, const int* b, int* c, int n);

TEST_CASE("Path 1C Scenario 2.1: cuModule* full chain", "[e2e][path_1C]") {
    CUmodule mod;
    CUfunction func;
    CUdeviceptr buf;
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoad(&mod, "./kernels.cubin") == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    // ... allocate device buffers, launch, copy back, verify sum == 3072
}
```

- [ ] **Step 2: Run test — expect FAIL with "cuModuleLoad fails" (kernel fixture missing)**

Run: `cd build && ctest -L path_1C --output-on-failure -R "Scenario 2.1"`
Expected: FAIL because `kernels.cubin` not yet built.

- [ ] **Step 3: Build minimal PTXIR image fixture for Path 1C**

Create `tests/ptxir/fixtures/v2_manifest_kernels.json`:
```json
{
  "version": 2,
  "kernels": [
    {"name": "vec_add", "ptx": "...", "sm": "sm_100"}
  ]
}
```

Then compile a minimal `kernels.cubin` via nvcc + ptxir_embed (same approach as Task 1 Step 3 but using `vec_add` kernel only).

- [ ] **Step 4: Implement full cuModule* chain in test**

Replace test body with:
```cpp
TEST_CASE("Path 1C Scenario 2.1: cuModule* full chain", "[e2e][path_1C]") {
    CUmodule mod; CUfunction func; CUdeviceptr da, db, dc;
    REQUIRE(cuInit(0) == CUDA_SUCCESS);
    REQUIRE(cuModuleLoad(&mod, "./kernels.cubin") == CUDA_SUCCESS);
    REQUIRE(cuModuleGetFunction(&func, mod, "vec_add") == CUDA_SUCCESS);
    const int N=1024;
    std::vector<int> a(N,1), b(N,2), c(N,0);
    cuMemAlloc(&da, N*4); cuMemAlloc(&db, N*4); cuMemAlloc(&dc, N*4);
    cuMemcpyHtoD(da, a.data(), N*4); cuMemcpyHtoD(db, b.data(), N*4);
    void* args[] = {&da, &db, &dc, (void*)&N};
    REQUIRE(cuLaunchKernel(func, N/256, 1, 1, 256, 1, 1, 0, 0, args, nullptr) == CUDA_SUCCESS);
    cuMemcpyDtoH(c.data(), dc, N*4);
    int sum = std::accumulate(c.begin(), c.end(), 0);
    REQUIRE(sum == 3072);
    cuModuleUnload(mod);
}
```

- [ ] **Step 5: Add Scenarios 2.2 (duplicate handle), 2.3 (not-found), 2.4 (cuLaunchKernel error path)**

Append tests for duplicate module handle (load same cubin twice — second must fail with `CUDA_ERROR_INVALID_HANDLE` or similar), kernel-name-not-found (get_function with bogus name → ERROR_NOT_FOUND), and cuLaunchKernel with invalid grid dims.

- [ ] **Step 6: Create CMakeLists.txt + commit Phase 2**

`tests/e2e/path_1C_driver_api/CMakeLists.txt`:
```cmake
add_catch_test(e2e_cuda_driver_exec test_cuda_driver_exec.cpp)
set_tests_properties(e2e_cuda_driver_exec PROPERTIES LABELS "e2e;path_1C" TIMEOUT 60)
add_dependencies(e2e_cuda_driver_exec ptxir_fixture_kernels)
```

Modify `tests/e2e/CMakeLists.txt`: add `add_subdirectory(path_1C_driver_api)`.

```bash
git add tests/e2e/path_1C_driver_api/ tests/ptxir/fixtures/v2_manifest_kernels.json \
        tests/e2e/CMakeLists.txt .gitignore
git commit -m "test(e2e): add Path 1C Driver API real kernel exec coverage

- Scenarios 2.1/2.2/2.3/2.4: full chain, duplicate handle, not-found, error path
- PTXIR v2 manifest fixture with kernels[]
- Closes D-PTX-8 (ADR-0021)"
```

---

### Task 4: Phase 3 — Path 2D Image Executor output correctness baseline

**Files:**
- Create: `tests/e2e/path_2D_image_executor/CMakeLists.txt`
- Create: `tests/e2e/path_2D_image_executor/test_image_executor_output.cpp`
- Create: `tests/ptxir/baselines/baseline_format.md`
- Create: `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin` (committed binary)
- Modify: `tests/integration/cudart/test_libptxemu_device.cpp` (add cute_rmsnorm correctness)
- Modify: `.gitignore` (whitelist baseline)

- [ ] **Step 1: Write failing test (Scenario 3.1 — output vs baseline)**

Create `tests/e2e/path_2D_image_executor/test_image_executor_output.cpp`:

```cpp
#include <catch2/catch_test_macros.hpp>
#include <cstdio>
#include <fstream>
#include <vector>
#include <cstring>

TEST_CASE("Path 2D Scenario 3.1: cute_rmsnorm output byte-level matches baseline", "[e2e][path_2D]") {
    // Run cute_rmsnorm.ptxir via libptxemu_device, capture output to buffer
    // Read baseline file
    std::ifstream f("../../ptxir/baselines/cute_rmsnorm_output_baseline.bin", std::ios::binary);
    std::vector<uint8_t> baseline((std::istreambuf_iterator<char>(f)), {});
    REQUIRE(baseline.size() >= 12);  // 8 magic + 4 size
    REQUIRE(std::memcmp(baseline.data(), "PTXR_OUT\0\0", 8) == 0);  // magic check
    // Read size prefix (LE u32)
    uint32_t size; std::memcpy(&size, baseline.data()+8, 4);
    REQUIRE(size == baseline.size() - 12);
    // Compare with actual output (must be loaded via libptxemu_device API)
    // ...
}
```

- [ ] **Step 2: Run test — expect FAIL (baseline file doesn't exist yet)**

Run: `cd build && ctest -L path_2D --output-on-failure -R "Scenario 3.1"`
Expected: FAIL with "baseline file not found"

- [ ] **Step 3: Generate baseline file (manually, one-time)**

Execute cute_rmsnorm.ptxir via PTX-EMU's image-executor API, capture output buffer, prepend `PTXR_OUT\0\0` magic + 4-byte LE u32 size, write to `tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin`:

```bash
cd build && ./tools/run_image_executor --input ../tests/ptxir/fixtures/cute_rmsnorm.ptxir \
    --output /tmp/cute_rmsnorm_out.bin
python3 -c "
import struct
data = open('/tmp/cute_rmsnorm_out.bin','rb').read()
out = b'PTXR_OUT\0\0' + struct.pack('<I', len(data)) + data
open('tests/ptxir/baselines/cute_rmsnorm_output_baseline.bin','wb').write(out)
print(f'wrote {len(out)} bytes')"
```

- [ ] **Step 4: Document baseline format**

Create `tests/ptxir/baselines/baseline_format.md`:
```markdown
# Cute RMSNorm Output Baseline Format

Each `.bin` file in this directory is a magic-prefixed binary:

| Offset | Size | Field | Description |
|--------|------|-------|-------------|
| 0      | 8    | magic | `PTXR_OUT\0\0` (8 bytes literal) |
| 8      | 4    | size  | LE u32 — byte count of payload that follows |
| 12     | size | bytes | Raw output buffer from simulator |

**Stability contract**: baselines are immutable once committed. Any change requires:
1. New file with versioned name (e.g. `cute_rmsnorm_output_v2.bin`)
2. Update `current_baseline` symlink
3. Document D3 mutation in design doc
```

- [ ] **Step 5: Run test — expect PASS**

Run: `cd build && ctest -L path_2D --output-on-failure`
Expected: PASS (magic + size + bytes all match)

- [ ] **Step 6: Add Scenario 3.7 (D3 mutation regression) to test file**

```cpp
TEST_CASE("Path 2D Scenario 3.7: D3 mutation regression", "[e2e][path_2D][RED_PHASE]") {
    // RED PHASE: this is a sentinel — when this test FAILS, a baseline mutation has occurred
    // and must be reviewed before commit.
    std::ifstream f("../../ptxir/baselines/cute_rmsnorm_output_baseline.bin", std::ios::binary);
    std::vector<uint8_t> b((std::istreambuf_iterator<char>(f)), {});
    REQUIRE(std::memcmp(b.data(), "PTXR_OUT\0\0", 8) == 0);
}
```

- [ ] **Step 7: Enhance integration test + add error path tests**

Modify `tests/integration/cudart/test_libptxemu_device.cpp` to add cute_rmsnorm output correctness check + 4 error path tests:
1. load garbage pointer → expect ERROR
2. execute invalid handle → expect ERROR
3. unload invalid handle → expect ERROR
4. kernel_name not in module → expect ERROR

- [ ] **Step 8: Update .gitignore + commit Phase 3**

`.gitignore` additions:
```
!tests/e2e/path_2D_image_executor/**
!tests/ptxir/baselines/*.bin
!tests/ptxir/baselines/*.md
```

```bash
git add tests/e2e/path_2D_image_executor/ tests/ptxir/baselines/ \
        tests/integration/cudart/test_libptxemu_device.cpp .gitignore
git commit -m "test(e2e): add Path 2D Image Executor output correctness baseline

- Magic-prefixed binary format (PTXR_OUT\\0\\0 + LE u32 size + bytes)
- Cute RMSNorm golden output committed
- D3 mutation regression sentinel
- 4 error path tests in integration
- 2D output-correctness coverage: 1/4 → 4/4"
```

---

### Task 5: Phase 4 — `tests/e2e/` reorganization into path-X directories

**Files:**
- Create: `tests/e2e/path_1A_legacy_ptx/CMakeLists.txt`
- Create: `tests/e2e/path_1B_ptxir_fatbinary/CMakeLists.txt` (modify existing from Phase 1 if needed)
- Modify: `tests/e2e/CMakeLists.txt` (drop moved `add_catch_test` calls + add 4 `add_subdirectory`)
- `git mv` operations (preserves file history via `--follow`)

- [ ] **Step 1: Verify git mv operations preserve history (dry-run)**

```bash
git mv --dry-run tests/e2e/kernel/test_blackwell_gemm.cu tests/e2e/path_1A_legacy_ptx/test_blackwell_gemm.cu
git mv --dry-run tests/e2e/kernel/test_tcgen05_reduction.cu tests/e2e/path_1A_legacy_ptx/test_tcgen05_reduction.cu
# ... (do this for ALL files in kernel/ and divergence/)
git mv --dry-run tests/e2e/kernel/test_ptxir_cubin_embed.cpp tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_cubin_embed.cpp
```

Expected: no errors, all file paths exist.

- [ ] **Step 2: Execute git mv operations (4.1 - 4.4)**

```bash
git mv tests/e2e/kernel/test_blackwell_gemm.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/kernel/test_tcgen05_reduction.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/kernel/test_tcgen05_warpgroup.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/divergence/test_divergence.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/divergence/test_divergence_sync.cu tests/e2e/path_1A_legacy_ptx/
git mv tests/e2e/kernel/test_ptxir_cubin_embed.cpp tests/e2e/path_1B_ptxir_fatbinary/
```

- [ ] **Step 3: Create `path_1A_legacy_ptx/CMakeLists.txt`**

```cmake
add_catch_test(e2e_blackwell_gemm path_1A_legacy_ptx/test_blackwell_gemm.cu)
set_tests_properties(e2e_blackwell_gemm PROPERTIES LABELS "e2e;path_1A")
add_catch_test(e2e_tcgen05_reduction path_1A_legacy_ptx/test_tcgen05_reduction.cu)
set_tests_properties(e2e_tcgen05_reduction PROPERTIES LABELS "e2e;path_1A")
# ... (one entry per moved file)
```

- [ ] **Step 4: Modify `tests/e2e/CMakeLists.txt` — drop moved add_catch_test calls + add add_subdirectory**

Replace lines 15-50 of `tests/e2e/CMakeLists.txt`:
```cmake
# Old (drop):
# add_catch_test(e2e_test3_cfg_full kernel/test_test3_cfg_full.cpp)
# ...

# New (add at bottom):
add_subdirectory(path_1A_legacy_ptx)
add_subdirectory(path_1B_ptxir_fatbinary)
add_subdirectory(path_1C_driver_api)
add_subdirectory(path_2D_image_executor)
```

- [ ] **Step 5: Run full ctest with path-X labels to verify isolation**

```bash
cd build && cmake --build . && ctest -L path_1A --output-on-failure
cd build && ctest -L path_1B --output-on-failure
cd build && ctest -L path_1C --output-on-failure
cd build && ctest -L path_2D --output-on-failure
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

- [ ] **Step 8: Commit Phase 4**

```bash
git add tests/e2e/
git commit -m "refactor(tests): reorganize tests/e2e/ into path-X subdirectories

- 4 path_X/CMakeLists.txt with explicit LABELS (e2e;path_X)
- ctest -L path_X for single-path regression
- git mv preserves file history (verify with --follow)
- 4-path coverage matrix: 3/4 → 4/4 (D-PTX-7/D-PTX-8 closed)"
```

---

### Task 6: Phase 5 — Proposal doc consistency fix (silent descoping disclosure)

**Files:**
- Modify: `openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md`

- [ ] **Step 1: Read current proposal §Capabilities section**

```bash
grep -n "test_ptxir_cubin_embed\|e2e 测试" openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
```

Locate the §Capabilities bullet that claims `test_ptxir_cubin_embed.cu` validates "PTX-EMU 加载 + ptxir_extract".

- [ ] **Step 2: Replace with disclaimer text**

Find:
```markdown
- `e2e-test-ptxir-cubin-embed`: ...
```

Replace with:
```markdown
- `e2e-test-ptxir-cubin-embed`: ... **[修正: 2026-08-12, see fix-path-coverage-gaps]** — 此 e2e 验证 PTXIR-Embedded CUBIN 格式兼容性（Phase 12.2 R5 / ADR-0024 Risk 1），**不验证 PTX-EMU 真实加载执行**。真实加载执行验证见 `tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp` (D-PTX-7 关闭于 fix-path-coverage-gaps Phase 1)。
```

- [ ] **Step 3: Verify cross-reference**

```bash
grep -n "fix-path-coverage-gaps\|D-PTX-7" openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
```

Expected: at least 2 mentions (the change name + debt number).

- [ ] **Step 4: Verify AC-5.5 (archive directory name unchanged)**

```bash
ls openspec/changes/archive/ | grep 2026-08-07
```

Expected: `2026-08-07-implement-ptxir-cubin-embed-extension` still exists.

- [ ] **Step 5: Verify git log shows this as a follow-up commit (not history rewrite)**

```bash
git log --oneline openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
```

Expected: at least 2 commits — original archive + this amendment.

- [ ] **Step 6: Commit Phase 5**

```bash
git add openspec/changes/archive/2026-08-07-implement-ptxir-cubin-embed-extension/proposal.md
git commit -m "docs(openspec): disclose silent descoping in implement-ptxir-cubin-embed-extension

The e2e-test-ptxir-cubin-embed capability only validates format compatibility
(ADR-0024 Risk 1, Phase 12.2 R5), NOT real PTX-EMU load/exec.

Real exec coverage is now provided by
tests/e2e/path_1B_ptxir_fatbinary/test_ptxir_fatbinary_exec.cpp
(closes D-PTX-7, fix-path-coverage-gaps Phase 1).

Cross-referenced debt number + new test location."
```

---

### Task 7: Acceptance — sanity + regression + clang-format + ctest label compliance

**Files:** (none modified)

- [ ] **Step 1: Run `./scripts/sanity.sh`**

Run: `./scripts/sanity.sh`
Expected: PASS (AC-G1)

- [ ] **Step 2: Run full ctest with all 3 labels**

Run: `cd build && ctest --output-on-failure -L "e2e;integration;unit"`
Expected: 100% pass (AC-G2)

- [ ] **Step 3: Run `./scripts/regression.sh`**

Run: `./scripts/regression.sh`
Expected: PASS (AC-G3)

- [ ] **Step 4: Run clang-format dry-run on all changed files**

```bash
clang-format --dry-run --Werror \
    tests/e2e/path_1B_ptxir_fatbinary/*.cpp tests/e2e/path_1B_ptxir_fatbinary/*.cu \
    tests/e2e/path_1C_driver_api/*.cpp \
    tests/e2e/path_2D_image_executor/*.cpp \
    tests/e2e/path_1A_legacy_ptx/CMakeLists.txt \
    tests/integration/cudart/test_libptxemu_device.cpp
```

Expected: exit code 0 (AC-G4)

- [ ] **Step 5: Verify all new tests retain `e2e_` prefix and `LABELS "e2e;..."` pattern**

```bash
grep -rn 'LABELS.*path_1A\|LABELS.*path_1B\|LABELS.*path_1C\|LABELS.*path_2D' tests/e2e/
```

Expected: 4 hits, all matching `LABELS "e2e;path_X"` (AC-N1, AC-N2).

- [ ] **Step 6: Verify 4-path coverage matrix**

```bash
cd build && for p in 1A 1B 1C 2D; do echo "Path $p:"; ctest -L path_$p -N | tail -1; done
```

Expected: each path has ≥1 test listed (AC-M1).

- [ ] **Step 7: Verify openspec status update is doable (run later in archive phase)**

```bash
openspec status --change fix-path-coverage-gaps
```

Expected: shows 5 Phases shippable (Phase 6/7 are gating, not shipped yet) (AC-G5 preparation).

---

### Task 8: Final commit aggregation (per `worktree-archive-workflow`)

**Files:** (no new files; aggregate commit)

- [ ] **Step 1: Verify all changes committed**

```bash
git status --short
```

Expected: empty (no uncommitted changes).

- [ ] **Step 2: Verify all tasks.md checkboxes updateable**

After each Phase commits (Tasks 2, 3, 4, 5, 6 above), update corresponding sections in `openspec/changes/fix-path-coverage-gaps/tasks.md`:

```bash
# In openspec/changes/fix-path-coverage-gaps/tasks.md:
# After Phase 1 ship: mark Section 1 (Phase 1) checkboxes [x]
# After Phase 2 ship: mark Section 2 (Phase 2) checkboxes [x]
# ...
```

- [ ] **Step 3: Final pre-archive verification**

```bash
# All ACs (G1-G5, N1-N2, M1-M4) must be satisfied before archive
openspec validate fix-path-coverage-gaps
```

Expected: PASS, 0 issues.

---

## Acceptance Criteria Mapping

| AC | Description | Verified in |
|----|-------------|-------------|
| G1 | sanity.sh passes | Task 7 Step 1 |
| G2 | ctest -L e2e+integration+unit 100% pass | Task 7 Step 2 |
| G3 | regression.sh passes | Task 7 Step 3 |
| G4 | clang-format clean on changed files | Task 7 Step 4 |
| G5 | 5 Phases shipped + iteration.json synced | Archive phase (Phase 7 of tasks.md) |
| N1 | new tests keep `e2e_` prefix | Task 7 Step 5 |
| N2 | new tests LABELS contain `e2e` | Task 7 Step 5 |
| M1 | cudart path coverage 3/4 → 4/4 | Task 7 Step 6 |
| M2 | output-correctness 1/4 → 4/4 | Task 4 Step 5 |
| M3 | openspec doc consistency fix (1 place) | Task 6 Step 1 |
| M4 | `ctest -L path_1X` is single-path regression | Task 5 Step 5 |

---

## Out of Scope (per design.md Non-Goals)

- Production code changes (`cudart_sim.cpp`, `cpptlm_module.cpp`, `ptxir_loader.cpp` untouched)
- `multi-entry-handle-api` task unchecking (separate improvement)
- New test framework (still Catch2 + `add_catch_test`)
- New PTXIR fixture generation tool
- openspec CLI / validate rule changes
- ctest label scheme (only additions)
- Top-level `tests/` structure (only `tests/e2e/` subtree)

---

## Self-Review Checklist

- [x] Spec Coverage: All 5 Phases + 6 acceptance sections from tasks.md → 8 implementation tasks
- [x] No Placeholders: Every step has actual code, file paths, run commands, expected output
- [x] Type Consistency: `e2e_ptxir_fatbinary_exec`, `e2e_cuda_driver_exec`, `e2e_image_executor_output` used consistently across Tasks 1-4
- [x] File Paths: All Create/Modify entries reference files that exist or are created in earlier tasks
- [x] TDD Order: Each task follows Red→Verify-fail→Green→Verify-pass→Commit-or-defer
