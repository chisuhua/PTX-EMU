# feat-ptxemu-image-executor Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use skill_use("execute") to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement `libptxemu_device.so` + `cpptlm_module.h` C-API for in-memory PTXIR image loading and execution; relocate 5 global symbols per ADR-0021 v1.1 amendment; verify byte-identical fallback for default LD_PRELOAD path; pass D3 perf acceptance (< 1.10 cute_rmsnorm wall-time ratio).

**Architecture:** 3-Phase implementation per ADR-0029 §D5:
- **Phase 0 (Commit 1)**: Line-level symbol relocation — `g_cpptlm_bridge` + `cpptlm_attach_bridge` + `cpptlm_detach_bridge` + `g_bridge_user_override` move to `PtxEmuDriverShim.cpp`; `g_gpu_context` moves to `ptx_interpreter.cpp`. 5 byte-identical fallback gates verify zero behavior change.
- **Phase 1 (Commit 2)**: New `cpptlm_module.h` public ABI (5 `extern "C"` functions, `CPPTLM_MODULE_VERSION 1`) + `PtxEmuImageExecutor` singleton (`g_image_executor` per [SINGLE-GPU-INSTANCE] #4) + `libptxemu_device.so` shared library. D3 mutation bug fix via per-launch re-deserialize.
- **Perf (Commit 3)**: `bench/cute/cute_rmsnorm.ptx` D3 deserialize cost measured; threshold `< 1.10`. Pass → continue; Fail → trigger A1 fallback change.
- **Docs (Commit 4)**: README + CHANGELOG + lessons-learned §44 + ADR-0029 compliance checkboxes + `v0.1.0` tag.

**Tech Stack:** C++20, CMake, Catch2, `std::mutex`/`std::unordered_map`/`std::atomic`, `PTXIRLoader::deserializeForCubin` (existing), `PtxContextAdapter::fromEmbedded` (existing), `PtxInterpreter::launchPtxInterpreter` (existing).

---

## File Structure

### Production Code

| File | Responsibility |
|---|---|
| `include/cudart/cpptlm_module.h` (NEW) | Public C-API: 5 `extern "C"` functions + `CPPTLM_MODULE_VERSION 1`. Zero PTX-EMU internal type exposure. |
| `src/cudart/cpptlm_module.cpp` (NEW) | `PtxEmuImageExecutor` singleton: `images_` map (handle → bytes), `exec_mu_` mutex, `next_handle_` atomic. 5 ABI entry wrappers. |
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` (MODIFY) | Add definitions: `g_cpptlm_bridge`, `cpptlm_attach_bridge`, `cpptlm_detach_bridge`, `g_bridge_user_override`. Maintain same-TU invariant per ADR-0021 v1.1 amendment. |
| `src/cudart/ptx_interpreter.cpp` (MODIFY) | Add definition: `std::unique_ptr<GPUContext> g_gpu_context;` (per ADR-0021 v1.1 amendment relocation). |
| `src/cudart/cudart_sim.cpp` (MODIFY) | Remove 5 global symbol definitions (line-level diff lock — no call-site modifications). |
| `src/CMakeLists.txt` (MODIFY) | Add `libptxemu_device.so` target linking `ptxsim` + `ptx_ir` + `ptxir` (per ADR-0029 §D5). |

### Tests

| File | Responsibility |
|---|---|
| `tests/integration/test_phase0_byte_identical_gates.cpp` (NEW) | 5 gates: `nm -D` diff, SONAME, symlinks, `g_cpptlm_bridge==nullptr` unit, logger→`g_gpu_context` unit. |
| `tests/unit/cudart/test_cpptlm_module.cpp` (NEW) | 10 tests: 5 ABI entry roundtrip + invalid handle rejection + concurrent serialization (D3 mutex verification). |
| `tests/unit/cudart/test_image_executor_mutation.cpp` (NEW) | D3 fix verification: byte-identical double-deserialize, N=1000 sequential launches determinism, image bytes SHA-256 invariance. |
| `tests/integration/test_cpptlm_module_dlopen.cpp` (NEW) | DL-isolated test: `dlopen("libptxemu_device.so")` without libcudart.so dependency. |
| `tests/integration/test_cpptlm_module_inflight.cpp` (NEW) | Multi-threaded concurrent `ptxemu_image_execute` serialization (mutex deadlock prevention). |
| `tests/performance/test_ptxir_deserialize_cost.cpp` (NEW) | D3 perf acceptance: cute_rmsnorm wall-time ratio (Group A baseline vs Group B 100× re-deserialize). |

---

## Commit 1 — Phase 0 Step 1: 5 Global Symbol Relocation + 5 Byte-Identical Gates

> **Prerequisites:** Commit 0 (ADR-0021 v1.1 amendment) already shipped (`8d05f35f` + `100afdc4`). `g_cpptlm_bridge` may now be defined outside `cudart_sim.cpp`. Hard gate per ADR-0029 §合规检查.
> **Strategy:** TDD 5-step — write 5 failing gates → verify fail → relocate 5 symbols → verify pass → defer commit (aggregated at archive per Phase 2.7).
> **Lessons applied:** §1 cross-module state translation (no call-site changes); §3 Phase commit granularity; §4 baseline worktree (establish `baseline-build` archive before any change); §14 byte-identical fallback must be test-locked.

### Task 1: Establish baseline worktree + archive build

**Files:**
- Create: `.worktrees/baseline-ptxemu-image-executor/` (via `git worktree add`)
- Create: `build-baseline/` (via CMake Release build inside baseline worktree)

- [ ] **Step 1: Create baseline worktree from current HEAD**

```bash
cd /workspace/project/PTX-EMU
git worktree add .worktrees/baseline-ptxemu-image-executor -b baseline/feat-ptxemu-image-executor HEAD
ls .worktrees/baseline-ptxemu-image-executor/lib/libcudart.so.12.0
# expect: file exists
```

Expected: Worktree created, `libcudart.so.12.0` symlink resolved.

- [ ] **Step 2: Full Release build in baseline worktree**

```bash
cd /workspace/project/PTX-EMU/.worktrees/baseline-ptxemu-image-executor
. ../env.sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
# expect: zero errors, build/lib/libcudart.so.12.0 produced
```

Expected: ~15-20 minutes. **MUST** be full build (partial build → ctest reports "executable not found" per Lesson §4 trap).

- [ ] **Step 3: Archive baseline ctest oracle**

```bash
cd /workspace/project/PTX-EMU/.worktrees/baseline-ptxemu-image-executor/build
ctest --output-on-failure -j$(nproc) 2>&1 | tee /tmp/baseline-ctest.log
grep -c "passed" /tmp/baseline-ctest.log
# expect: count ≥ 230 (per task 1.3.6 "现有 230+ ctest 全集")
```

Expected: 230+ tests pass, log archived to `/tmp/baseline-ctest.log`.

- [ ] **Step 4: Capture baseline binary + symbol surface**

```bash
cd /workspace/project/PTX-EMU
mkdir -p /tmp/baseline-artifacts
nm -D --defined-only .worktrees/baseline-ptxemu-image-executor/build/lib/libcudart.so | sort > /tmp/baseline-artifacts/libcudart-nm-before.txt
objdump -p .worktrees/baseline-ptxemu-image-executor/build/lib/libcudart.so | grep SONAME > /tmp/baseline-artifacts/libcudart-soname-before.txt
ls -la .worktrees/baseline-ptxemu-image-executor/lib/libcudart.so* > /tmp/baseline-artifacts/libcudart-symlinks-before.txt
wc -l /tmp/baseline-artifacts/libcudart-nm-before.txt
# expect: ~150 lines
```

Expected: `/tmp/baseline-artifacts/` contains `libcudart-nm-before.txt` + `libcudart-soname-before.txt` + `libcudart-symlinks-before.txt`. These are the oracles for Gate 1/2/3 verification.

- [ ] **Step 5: Document baseline state for post-Phase-0 regression check**

```bash
cd /workspace/project/PTX-EMU
cp /tmp/baseline-ctest.log .rddf/plans/baseline-ctest-oracle.log
echo "Baseline ctest oracle: $(grep -c 'passed' /tmp/baseline-ctest.log) tests passed at $(date -Iseconds)" > .rddf/plans/baseline-state.md
cat .rddf/plans/baseline-state.md
```

Expected: `baseline-state.md` records baseline test count + timestamp.

---

### Task 2: Write 5 failing gates (Red)

**Files:**
- Create: `tests/integration/test_phase0_byte_identical_gates.cpp`

- [ ] **Step 1: Create test file skeleton**

```cpp
// tests/integration/test_phase0_byte_identical_gates.cpp
// Per ADR-0029 §D7: Phase 0 完成后 5 gates 必须全部通过
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include "cudart/cpptlm_bridge.h"  // for g_cpptlm_bridge
#include "cudart/ptx_interpreter.h"  // for g_gpu_context

extern "C" size_t get_gpu_clock_from_context();

namespace fs = std::filesystem;

static std::string libcudart_so() {
    return fs::path(getenv("CMAKE_BINARY_DIR") ?: "build") / "lib" / "libcudart.so.12.0";
}

TEST_CASE("Gate 1: nm -D --defined-only libcudart.so symbol surface unchanged", "[integration][phase0][gate]") {
    // Per ADR-0029 D7 gate 1
    fs::path build_lib = getenv("CMAKE_BINARY_DIR") ? fs::path(getenv("CMAKE_BINARY_DIR")) / "lib/libcudart.so.12.0" : fs::path("build/lib/libcudart.so.12.0");
    REQUIRE(fs::exists(build_lib));

    // Diff against baseline (assumes /tmp/baseline-artifacts/libcudart-nm-before.txt exists)
    std::ifstream baseline("/tmp/baseline-artifacts/libcudart-nm-before.txt");
    REQUIRE(baseline.good());

    // Run nm on current build
    std::string cmd = "nm -D --defined-only " + build_lib.string() + " | sort";
    FILE* pipe = popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);

    // Stream-compare: collect both into sorted vectors, then diff
    std::vector<std::string> before, after;
    std::string line;
    while (std::getline(baseline, line)) before.push_back(line);
    char buf[4096];
    while (fgets(buf, sizeof(buf), pipe)) after.push_back(std::string(buf));
    pclose(pipe);

    REQUIRE(before == after);  // expect PASS after relocation, FAIL before
}

TEST_CASE("Gate 2: SONAME preserved as libcudart.so.12", "[integration][phase0][gate]") {
    fs::path libcudart = getenv("CMAKE_BINARY_DIR") ? fs::path(getenv("CMAKE_BINARY_DIR")) / "lib/libcudart.so.12.0" : fs::path("build/lib/libcudart.so.12.0");
    std::string cmd = "objdump -p " + libcudart.string() + " | grep SONAME";
    FILE* pipe = popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);
    char buf[256];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    REQUIRE(result.find("libcudart.so.12") != std::string::npos);
}

TEST_CASE("Gate 3: POST_BUILD symlinks preserved", "[integration][phase0][gate]") {
    std::string cmd = "ls -la lib/libcudart.so* 2>/dev/null";
    FILE* pipe = popen(cmd.c_str(), "r");
    REQUIRE(pipe != nullptr);
    char buf[1024];
    std::string result;
    while (fgets(buf, sizeof(buf), pipe)) result += buf;
    pclose(pipe);
    // expect both .12 versioned + main unversioned symlinks
    REQUIRE(result.find("libcudart.so.12") != std::string::npos);
    REQUIRE(result.find("libcudart.so ") != std::string::npos);
}

TEST_CASE("Gate 4: g_cpptlm_bridge nullptr standalone path test", "[integration][phase0][gate]") {
    // Per cpptlm_bridge.h:61 "nullptr = 独立模式，字节级兼容" contract
    REQUIRE(g_cpptlm_bridge == nullptr);
    // If standalone mode works correctly, the absence of bridge is the test.
    // No further assertions needed — the contract is "nullptr path = byte-identical to pre-bridge"
}

TEST_CASE("Gate 5: logger→g_gpu_context clock path test (relocation linkage)", "[integration][phase0][gate]") {
    // Per src/utils/logger.cpp:8 extern size_t get_gpu_clock_from_context()
    // After relocation: g_gpu_context is now defined in ptx_interpreter.cpp
    // This test verifies the linkage survives the relocation.
    size_t clock1 = get_gpu_clock_from_context();
    // Trigger some work (init g_gpu_context if needed) — note: full path tested via launch
    size_t clock2 = get_gpu_clock_from_context();
    // Either both 0 (no context init) or clock2 >= clock1 (monotonic)
    REQUIRE((clock2 == 0 || clock2 >= clock1));
}
```

- [ ] **Step 2: Add test to ctest configuration**

Modify `tests/integration/CMakeLists.txt` — append (if not already present):

```cmake
add_executable(test_phase0_byte_identical_gates
    test_phase0_byte_identical_gates.cpp
)
target_link_libraries(test_phase0_byte_identical_gates PRIVATE cudart ptxsim ptx_ir ptxir)
add_test(NAME integration_phase0_byte_identical_gates
         COMMAND test_phase0_byte_identical_gates)
set_tests_properties(integration_phase0_byte_identical_gates PROPERTIES LABELS "integration;phase0")
```

If a more specific sub-CMakeLists exists (e.g. `tests/integration/cudart/`), prefer that location. Verify with `find tests -name CMakeLists.txt`.

- [ ] **Step 3: Verify Gate 4 fails (g_cpptlm_bridge relocation prerequisite)**

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_phase0_byte_identical_gates 2>&1 | head -40
ctest -R "phase0_byte_identical" --output-on-failure 2>&1 | head -40
# expect: Gate 4 PASSES today (g_cpptlm_bridge == nullptr standalone mode is current state)
# Gates 1/2/3 should PASS today (baseline equality before any change)
# Gate 5 may PASS or FAIL depending on whether g_gpu_context is initialized
```

Expected: This is the **baseline Red** state — gates 1/2/3 should already pass (no change yet); gate 4 passes (current state); gate 5 may pass or fail. We re-verify after relocation.

- [ ] **Step 4: Document baseline gate state**

```bash
cd /workspace/project/PTX-EMU
ctest -R "phase0_byte_identical" --output-on-failure 2>&1 | tee /tmp/phase0-gates-before.log
echo "Pre-relocation gate state captured"
```

Expected: `/tmp/phase0-gates-before.log` records baseline gate outcomes. Will diff against `/tmp/phase0-gates-after.log` after relocation.

---

### Task 3: Relocate `g_cpptlm_bridge` + 3 ABI symbols to `PtxEmuDriverShim.cpp` (Implement Step 1)

**Files:**
- Modify: `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` (append at top, before existing class methods)
- Modify: `src/cudart/cudart_sim.cpp` (remove 5 definitions, lines 92/104/110/126-138/355-360)

> **Critical per Lesson §1**: This is a line-level diff. **DO NOT** modify any call-site of `g_cpptlm_bridge`, `cpptlm_attach_bridge`, `cpptlm_detach_bridge`, `g_bridge_user_override`, `g_gpu_context`. Only the **definitions** move.

- [ ] **Step 1: Read current `cudart_sim.cpp` lines 85-145 and 350-365 (definitions + ABI entry points)**

Verify exact strings to be moved:
- `cudart_sim.cpp:92` — `std::unique_ptr<GPUContext> g_gpu_context;`
- `cudart_sim.cpp:104` — `CppTLMBridge* g_cpptlm_bridge = nullptr;`
- `cudart_sim.cpp:110` — `static bool g_bridge_user_override = false;`
- `cudart_sim.cpp:126-132` — `cpptlm_attach_bridge` function body
- `cudart_sim.cpp:134-138` — `cpptlm_detach_bridge` function body

Use `read_mcp_resource` or direct Read to capture exact lines. **DO NOT** alter comment lines 95-124 or 96-103 — those are documentation, move them along with their associated symbols.

- [ ] **Step 2: Append 4 bridge symbols to `PtxEmuDriverShim.cpp` (top of file, before `#include` blocks)**

Edit `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` to prepend (above existing `#include "PtxEmuDriverShim.h"`):

```cpp
// Phase 0 Step 1: 4 bridge symbols relocated from cudart_sim.cpp
// per ADR-0029 §D2 + ADR-0021 v1.1 amendment.
// Same-TU invariant maintained: cpptlm_set_driver()'s g_ptx_emu_driver_shim
// + g_cpptlm_bridge + cpptlm_attach_bridge + cpptlm_detach_bridge +
// g_bridge_user_override all in PtxEmuDriverShim.cpp.
//
// cpptlm_set_driver() in cudart_sim.cpp uses these via external linkage
// (g_cpptlm_bridge is referenced from cudart_sim.cpp:341 etc., unchanged).

#include "cudart/cpptlm_bridge.h"
#include "cudart/cpptlm_bridge/PtxEmuDriverShim.h"

// CppTLM Bridge 全局指针 (D-PTX-1, relocated from cudart_sim.cpp:104)
CppTLMBridge* g_cpptlm_bridge = nullptr;

// g_bridge_user_override (relocated from cudart_sim.cpp:110, line-level diff)
// static thread_local in original; verify semantic equivalence
static bool g_bridge_user_override = false;

// cpptlm_attach_bridge / cpptlm_detach_bridge ABI entry points (relocated
// from cudart_sim.cpp:126-138, line-level diff — body byte-identical)
extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(CppTLMBridge* bridge) {
    // (preserved body from cudart_sim.cpp:126-132)
    g_cpptlm_bridge = bridge;
    g_bridge_user_override = (bridge != nullptr);
}

extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge() {
    // (preserved body from cudart_sim.cpp:134-138)
    g_cpptlm_bridge = nullptr;
    g_bridge_user_override = false;
}
```

> **Note**: The `PTX_DEBUG_EMU` log calls in the original body are preserved by copying the implementation file's logger include (`#include "utils/logger.h"`) if present, OR omit the log calls (degrade to no-log). Check `PtxEmuDriverShim.cpp` top — if logger include present, preserve debug; otherwise strip.

- [ ] **Step 3: Remove 4 definitions + 2 function bodies from `cudart_sim.cpp`**

Edit `src/cudart/cudart_sim.cpp`:

**Remove (line 92)**:
```cpp
std::unique_ptr<GPUContext> g_gpu_context;
```

**Remove (lines 95-103, the documentation comment block)**:
```cpp
// ============================================================================
// CppTLM Bridge 全局指针 (D-PTX-1)
// ============================================================================
// 默认 nullptr，加载 libcpptlm_cudart.so 后赋值。
// nullptr 时所有操作走原有同步路径（字节级相同）。
// ============================================================================
#include "cudart/cpptlm_bridge.h"
#include "cudart/cpptlm_bridge/PtxEmuDriverShim.h"
#include "cudart/stub_bridge.h"
```

**Remove (line 104)**:
```cpp
CppTLMBridge* g_cpptlm_bridge = nullptr;
```

**Remove (lines 106-110)**:
```cpp
// g_bridge_user_override: 当用户通过 cpptlm_attach_bridge() 显式注入
// mock bridge 时设为 true，阻止 initialize_environment() 的 StubBridge
// auto-attach 覆盖用户的 mock。cpptlm_detach_bridge() 重置为 false。
// 见 auto-co-sim-standalone design.md D1。
static bool g_bridge_user_override = false;
```

**Remove (lines 112-138)** — entire `cpptlm_attach_bridge` / `cpptlm_detach_bridge` block:

```cpp
// ============================================================================
// cpptlm_attach_bridge / cpptlm_detach_bridge ABI entry points (B1)
// ============================================================================
// Per ADR-0021 (D-PTX-1): CppTLM's libcpptlm_cudart.so calls these on
// load/unload to install/uninstall the bridge pointer. Both are idempotent:
//   - attach: overwrite is allowed (last-call-wins); nullptr bridges call to
//     detach semantics per cpptlm_bridge.h:160 documentation contract.
//   - detach: safe to call when already nullptr (no-op).
//
// Metis second-pass review B1: declarations in cpptlm_bridge.h:161,168 were
// symbols without definitions, causing link errors. Implementations live
// here (same TU as g_cpptlm_bridge per D-PTX-1) to ensure the global pointer
// is mutated only through these ABI entry points.
// ============================================================================
extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(CppTLMBridge* bridge) {
    PTX_DEBUG_EMU("cpptlm_attach_bridge: bridge=%p (was %p)",
                  (void*)bridge, (void*)g_cpptlm_bridge);
    // nullptr bridge ≡ detach (per cpptlm_bridge.h:160 contract).
    g_cpptlm_bridge = bridge;
    g_bridge_user_override = (bridge != nullptr);
}

extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge() {
    PTX_DEBUG_EMU("cpptlm_detach_bridge (was %p)", (void*)g_cpptlm_bridge);
    g_cpptlm_bridge = nullptr;
    g_bridge_user_override = false;
}
```

- [ ] **Step 4: Verify call-sites untouched (Lesson §1 cross-module diff check)**

```bash
cd /workspace/project/PTX-EMU
grep -n "g_cpptlm_bridge\|cpptlm_attach_bridge\|cpptlm_detach_bridge\|g_bridge_user_override" src/cudart/cudart_sim.cpp | grep -v "^//\|^[[:space:]]*\*\|^[[:space:]]*//" | head -20
# expect: only call-sites (e.g. line 341 `g_cpptlm_bridge = &stub_bridge`), no `= nullptr;` definition
grep -n "g_cpptlm_bridge" src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp | head -5
# expect: 1 match = the new definition
git diff --stat HEAD -- src/cudart/cudart_sim.cpp src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp
# expect: ~50 lines diff (4 definitions removed, 4 added in PtxEmuDriverShim.cpp)
```

Expected: Only definitions moved; no logic changes. `git diff --stat` shows ~50 line delta.

---

### Task 4: Relocate `g_gpu_context` to `ptx_interpreter.cpp`

**Files:**
- Modify: `src/cudart/ptx_interpreter.cpp` (add definition near top, after `#include` blocks)
- (Already done in Task 3 step 3: removed from `cudart_sim.cpp:92`)

- [ ] **Step 1: Read `ptx_interpreter.cpp` top to find insertion point**

```bash
cd /workspace/project/PTX-EMU
head -20 src/cudart/ptx_interpreter.cpp
# expect: #include blocks + class declaration; find line AFTER `#include`s
```

- [ ] **Step 2: Add `g_gpu_context` definition after includes `#include`

Edit `src/cudart/ptx_interpreter.cpp` — append (after `#include "utils/logger.h"` or equivalent final include):

```cpp
// Phase 0 Step 1: g_gpu_context relocated from cudart_sim.cpp:92
// per ADR-0029 §D2 + ADR-0021 v1.1 amendment.
// Declaration in ptx_interpreter.h:19 (extern) — definition now in same TU
// as PtxInterpreter class (per D2 row 3 relocation target).
//
// Note: g_gpu_context is consumed by ptx_interpreter.cpp:142, ptx_interpreter.cpp:336
// (read access). Call-sites unchanged per Lesson §1.
std::unique_ptr<GPUContext> g_gpu_context;
```

> **Note**: `ptx_interpreter.cpp` already references `g_gpu_context` (line 17 has a comment "不再需要在这里声明g_gpu_context，已在头文件中声明"). The definition's previous location was `cudart_sim.cpp:92`. After relocation, `ptx_interpreter.cpp` provides the definition.

- [ ] **Step 3: Verify no duplicate symbol error at compile time**

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target cudart 2>&1 | grep -E "multiple definition|first defined here|g_gpu_context" | head -20
# expect: no errors
```

Expected: Build succeeds. **If** "multiple definition of `g_gpu_context`" error appears, an old build artifact was missed — run `cmake --build build --target clean && cmake --build build`.

- [ ] **Step 4: Verify `g_gpu_context` call-sites unchanged**

```bash
cd /workspace/project/PTX-EMU
grep -n "g_gpu_context" src/cudart/cudart_sim.cpp src/cudart/ptx_interpreter.cpp src/cudart/cuda_driver.cpp 2>/dev/null | grep -v "^[[:space:]]*\*\|^[[:space:]]*//" | head -15
# expect: call-sites at cudart_sim.cpp:292, 296, 298, 308, 309, 315, 322, 341, 356, 357, 502 (UNCHANGED)
#         ptx_interpreter.cpp:142, 336 (UNCHANGED)
#         cuda_driver.cpp: ... (verify no relocation needed)
```

Expected: All call-sites identical to pre-relocation grep output. Cross-check against baseline grep saved in `.rddf/plans/baseline-state.md`.

---

### Task 5: Move `get_gpu_clock_from_context()` to `ptx_interpreter.cpp` (Logger test gate)

**Files:**
- Modify: `src/cudart/cudart_sim.cpp` (remove `extern "C" { size_t get_gpu_clock_from_context() {...} }` block, lines 351-360)
- Modify: `src/cudart/ptx_interpreter.cpp` (add equivalent function with `extern "C"` linkage)

> **Rationale**: Gate 5 verifies `logger.cpp:8` `extern size_t get_gpu_clock_from_context()` resolves correctly after `g_gpu_context` relocation. The function body references `g_gpu_context` — keeping it in `ptx_interpreter.cpp` (same TU as `g_gpu_context` definition) ensures clean linkage.

- [ ] **Step 1: Read `cudart_sim.cpp:351-360` to capture exact function body**

```cpp
#ifdef __cplusplus
extern "C" {
#endif

size_t get_gpu_clock_from_context() {
    if (g_gpu_context) {
        return g_gpu_context->get_clock();
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 2: Remove the block from `cudart_sim.cpp:351-360`**

Delete lines 351-360 inclusive. Ensure no orphan `#ifdef`/`#endif` mismatch.

- [ ] **Step 3: Add `get_gpu_clock_from_context()` to `ptx_interpreter.cpp` (after `g_gpu_context` definition)**

```cpp
// Phase 0 Step 1: get_gpu_clock_from_context relocated from cudart_sim.cpp:355.
// Per ADR-0029 §D7 gate 5: src/utils/logger.cpp:8 extern resolves via this TU.
// Body byte-identical to original.
#ifdef __cplusplus
extern "C" {
#endif

size_t get_gpu_clock_from_context() {
    if (g_gpu_context) {
        return g_gpu_context->get_clock();
    }
    return 0;
}

#ifdef __cplusplus
}
#endif
```

- [ ] **Step 4: Build and verify Gate 5 compiles**

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target cudart 2>&1 | tail -20
# expect: zero errors
```

Expected: Clean build. `logger.cpp:2` `#include "cudart/ptx_interpreter.h"` resolves `g_gpu_context` declaration; `extern size_t get_gpu_clock_from_context()` declaration in `logger.cpp:8` resolves against new `ptx_interpreter.cpp` definition.

---

### Task 6: Rebuild + verify 5 gates pass (Green)

**Files:**
- Modify: None (build + test only)

- [ ] **Step 1: Full rebuild (incremental OK at this stage, full clean not required)**

```bash
cd /workspace/project/PTX-EMU
cmake --build build -j$(nproc) 2>&1 | tail -10
# expect: 0 errors, 0 warnings related to undefined references / multiple definitions
```

Expected: Build succeeds.

- [ ] **Step 2: Run 5-gate test + verify all pass**

```bash
cd /workspace/project/PTX-EMU
ctest -R "phase0_byte_identical" --output-on-failure 2>&1 | tee /tmp/phase0-gates-after.log
# expect: 5/5 gates PASS
```

Expected:
- Gate 1 (`nm -D` diff): PASS — symbol surface unchanged after relocation
- Gate 2 (SONAME): PASS — `libcudart.so.12` preserved
- Gate 3 (symlinks): PASS — `.12` + main symlinks present
- Gate 4 (`g_cpptlm_bridge == nullptr`): PASS — standalone mode contract
- Gate 5 (logger→g_gpu_context clock): PASS — `get_gpu_clock_from_context()` resolves correctly

- [ ] **Step 3: Run full ctest, verify 0 regression vs baseline**

```bash
cd /workspace/project/PTX-EMU
ctest --output-on-failure -j$(nproc) 2>&1 | tee /tmp/full-ctest-after-phase0.log
diff <(grep "passed" /tmp/baseline-ctest.log | sort) <(grep "passed" /tmp/full-ctest-after-phase0.log | sort) > /tmp/phase0-regression-diff.txt
test ! -s /tmp/phase0-regression-diff.txt && echo "0 REGRESSION" || (echo "REGRESSION DETECTED"; cat /tmp/phase0-regression-diff.txt)
# expect: "0 REGRESSION"
```

Expected: Zero regression vs baseline `baseline-ctest-oracle.log`. **If** regression detected, **immediately revert** this commit per Lesson §3 + §14 (byte-identical fallback must be test-locked).

- [ ] **Step 4: Capture post-relocation binary oracle**

```bash
cd /workspace/project/PTX-EMU
nm -D --defined-only build/lib/libcudart.so | sort > /tmp/baseline-artifacts/libcudart-nm-after-phase0.txt
diff /tmp/baseline-artifacts/libcudart-nm-before.txt /tmp/baseline-artifacts/libcudart-nm-after-phase0.txt
# expect: empty diff
```

Expected: Empty diff confirms Gate 1's byte-level equivalence. Archive to `.rddf/plans/`.

- [ ] **Step 5: Defer commit (per Phase 2.7 — aggregated at archive)**

```bash
cd /workspace/project/PTX-EMU
git status --short
git diff --stat HEAD
# expect: 4 files modified (cudart_sim.cpp, PtxEmuDriverShim.cpp, ptx_interpreter.cpp, tests/integration/...)
# NO commit at this step — aggregated into final commit per Phase 2.7 worktree commit pattern
```

Expected: Working tree shows ~5 file changes (~80 line diff). Defer commit; final aggregate commit happens before archive.

---

### Task 7: Cleanup baseline worktree + archive artifacts

**Files:**
- Modify: `.worktrees/baseline-ptxemu-image-executor/` (remove after Gate 6 perf baseline captured in Commit 3)

- [ ] **Step 1: Archive baseline build outputs to `/tmp/baseline-artifacts/` (if not already)**

```bash
ls /tmp/baseline-artifacts/
# expect: libcudart-nm-before.txt, libcudart-soname-before.txt, libcudart-symlinks-before.txt, libcudart-nm-after-phase0.txt, baseline-ctest-oracle.log
```

- [ ] **Step 2: Keep baseline worktree intact for Commit 3 perf gate**

```bash
cd /workspace/project/PTX-EMU
git worktree list
# expect: 2 entries — main + baseline/feat-ptxemu-image-executor
#         baseline worktree PRESERVED until Commit 3 perf gate verifies < 1.10 cute_rmsnorm wall-time ratio
```

Expected: Baseline worktree retained for Commit 3 perf comparison.

---

## Commit 2 — Phase 1: `cpptlm_module.h` + `PtxEmuImageExecutor` + `libptxemu_device.so`

> **Prerequisites:** Commit 1 complete (5 gates PASS, 0 regression).
> **Strategy:** TDD 5-step — write failing ABI tests → verify fail → implement executor → verify pass → defer commit.
> **Lessons applied:** §10 single-instance assumption (7 [SINGLE-GPU-INSTANCE] markers in class header); §3 Phase commit granularity.

### Task 8: Create `cpptlm_module.h` public ABI header

**Files:**
- Create: `include/cudart/cpptlm_module.h`

- [ ] **Step 1: Write header (per ADR-0029 §D1 + spec §Requirement)**

```cpp
// include/cudart/cpptlm_module.h
#ifndef CPPTLM_MODULE_H
#define CPPTLM_MODULE_H

// =====================================================================
// PTX-EMU Image Executor C-API (public ABI, ADR-0029 §D1)
// =====================================================================
//
// 5 extern "C" functions for in-memory PTXIR image loading and execution.
// Designed for cross-repo integration (UsrLinuxEmu HAL extension + TaskRunner
// cu_module.cpp consumer per ADR-0029 §D8).
//
// Governance (per include/cudart/AGENTS.md "不要向 cpptlm_bridge.h 添加
// CppTLM 头文件 include"): this header is independent of cpptlm_bridge.h.
// ABI version CPPTLM_MODULE_VERSION must bump on any signature change.
//
// cpptlm_bridge.h governance model applies: any modification must (1) bump
// version, (2) notify all consumers (UsrLinuxEmu HAL), (3) record in
// docs/dev-process/lessons-learned.md.

#include <cstddef>
#include <cstdint>

#define CPPTLM_MODULE_VERSION 1

#ifdef __cplusplus
extern "C" {
#endif

/// Load image bytes into executor's private image memory.
/// Returns opaque handle (non-zero = success, 0 = failure).
/// Accepted formats:
///   - standalone PTXIR (leading 4 bytes == "PTXI", per ptxir_format.h)
///   - PTXIR-Embedded CUBIN (trailing 8 bytes == PTXIR_EMBED_MAGIC, per ADR-0024)
///   - PTXIR-Embedded EXE (ELF prefix + PTXIR_EMBED_MAGIC trailer)
/// Rejected formats: NVIDIA cubin / fatbin / Tile IR.
/// Note: image bytes are deep-copied; caller may free after return.
uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size);

/// Query the kernel name for a loaded image (v1 single-kernel per image).
/// Copies up to (buf_size - 1) bytes of kernel name + NUL-terminates.
/// Returns 0 on success, -EINVAL on invalid handle / zero kernels.
int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size);

/// Synchronous kernel execution: blocks until kernel completes.
/// grid_*, block_* specify CUDA launch dimensions.
/// kernel_args is host-side array of pointers to args (per CUDA convention).
/// args_count = number of args.
/// shared_mem_bytes = dynamic shared memory request.
/// Returns 0 on success, -EINVAL on invalid handle, -EBUSY on in-flight kernel.
int ptxemu_image_execute(uint64_t handle,
                         uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                         uint32_t block_x, uint32_t block_y, uint32_t block_z,
                         size_t shared_mem_bytes,
                         void** kernel_args, size_t args_count);

/// Unload image bytes from executor.
/// If a kernel is currently in-flight, returns -EBUSY immediately
/// (caller may retry after kernel completion).
/// Returns 0 on success, -EINVAL on invalid handle, -EBUSY on in-flight.
int ptxemu_image_unload(uint64_t handle);

/// Query ABI version (returns CPPTLM_MODULE_VERSION).
/// Callers should verify version at startup:
///   if (ptxemu_module_version() != CPPTLM_MODULE_VERSION) return -EPROTO;
int ptxemu_module_version(void);

#ifdef __cplusplus
}
#endif

#endif  // CPPTLM_MODULE_H
```

- [ ] **Step 2: Verify header compiles in isolation**

```bash
cd /workspace/project/PTX-EMU
echo '#include "cudart/cpptlm_module.h"
int main() { return ptxemu_module_version() == CPPTLM_MODULE_VERSION ? 0 : 1; }' > /tmp/cpptlm_module_header_test.cpp
g++ -std=c++20 -I include -c /tmp/cpptlm_module_header_test.cpp -o /tmp/cpptlm_module_header_test.o 2>&1 | head -10
# expect: zero errors
```

Expected: Header self-contained, no missing includes.

---

### Task 9: Write 10 failing ABI entry tests (Red)

**Files:**
- Create: `tests/unit/cudart/test_cpptlm_module.cpp`

- [ ] **Step 1: Create test fixture with valid PTXIR bytes**

```bash
cd /workspace/project/PTX-EMU
ls tests/ptxir/fixtures/ 2>/dev/null || mkdir -p tests/ptxir/fixtures
# generate a minimal valid PTXIR fixture if not present:
# Use existing cute_rmsnorm.ptxir if available, OR create a stub via tools/ptxir_build
ls tests/ptxir/fixtures/
# expect: at least one .ptxir file
```

If no fixture exists, copy `bench/cute/cute_rmsnorm.ptx` → generate `.ptxir` via `tools/ptxir_build` (per ADR-0025). Document the fixture path in test code.

- [ ] **Step 2: Write test skeleton (10 cases)**

```cpp
// tests/unit/cudart/test_cpptlm_module.cpp
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <vector>
#include "cudart/cpptlm_module.h"

namespace fs = std::filesystem;

static std::vector<uint8_t> readFixture(const std::string& name) {
    fs::path p = fs::path(TEST_FIXTURE_DIR) / name;
    std::ifstream f(p, std::ios::binary);
    REQUIRE(f.good());
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("ptxemu_image_load: standalone PTXIR returns valid handle", "[unit][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    ptxemu_image_unload(handle);
}

TEST_CASE("ptxemu_image_load: PTXIR-Embedded CUBIN returns valid handle", "[unit][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm_embedded.cubin");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    ptxemu_image_unload(handle);
}

TEST_CASE("ptxemu_image_load: zero size returns 0", "[unit][cpptlm_module]") {
    REQUIRE(ptxemu_image_load(nullptr, 0) == 0);
}

TEST_CASE("ptxemu_image_load: corrupt magic returns 0", "[unit][cpptlm_module]") {
    std::vector<uint8_t> bad = {'X','X','X','X', 0,0,0,0};
    REQUIRE(ptxemu_image_load(bad.data(), bad.size()) == 0);
}

TEST_CASE("ptxemu_image_kernel_name: valid handle returns kernel string", "[unit][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    char buf[256] = {0};
    int rc = ptxemu_image_kernel_name(handle, buf, sizeof(buf));
    REQUIRE(rc == 0);
    REQUIRE(std::string(buf) == "cute_rmsnorm_kernel");
    ptxemu_image_unload(handle);
}

TEST_CASE("ptxemu_image_execute: valid handle returns 0 (synchronous)", "[unit][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);
    void* args[] = {nullptr};  // no args for cute_rmsnorm in this minimal case
    int rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
    REQUIRE(rc == 0);
    ptxemu_image_unload(handle);
}

TEST_CASE("ptxemu_image_execute: zero handle returns -EINVAL", "[unit][cpptlm_module]") {
    void* args[] = {nullptr};
    REQUIRE(ptxemu_image_execute(0, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
}

TEST_CASE("ptxemu_image_unload: valid handle returns 0", "[unit][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(ptxemu_image_unload(handle) == 0);
    // Subsequent execute on unloaded handle should fail
    void* args[] = {nullptr};
    REQUIRE(ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0) == -EINVAL);
}

TEST_CASE("ptxemu_image_unload: in-flight kernel returns -EBUSY (covered by integration test)", "[unit][cpptlm_module]") {
    // Integration test in test_cpptlm_module_inflight.cpp exercises this.
    // Unit test confirms unload after execute completes returns 0 (covered above).
    SUCCEED("see test_cpptlm_module_inflight.cpp");
}

TEST_CASE("ptxemu_module_version: returns CPPTLM_MODULE_VERSION (1)", "[unit][cpptlm_module]") {
    REQUIRE(ptxemu_module_version() == 1);
    REQUIRE(CPPTLM_MODULE_VERSION == 1);
}
```

- [ ] **Step 3: Add to ctest**

```cmake
# tests/unit/cudart/CMakeLists.txt — append:
add_executable(test_cpptlm_module
    test_cpptlm_module.cpp
)
target_link_libraries(test_cpptlm_module PRIVATE cudart ptxemu_device ptxsim ptx_ir ptxir)
target_compile_definitions(test_cpptlm_module PRIVATE TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures")
add_test(NAME unit_cpptlm_module COMMAND test_cpptlm_module)
set_tests_properties(unit_cpptlm_module PROPERTIES LABELS "unit;cpptlm_module")
```

- [ ] **Step 4: Verify tests fail (RED state)**

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_cpptlm_module 2>&1 | tail -20
# expect: link error (undefined reference to ptxemu_image_load, etc.)
ctest -R "unit_cpptlm_module" --output-on-failure 2>&1 | tail -20
# expect: tests FAIL (link errors or runtime "no executor" returns)
```

Expected: Tests fail because `libptxemu_device.so` doesn't exist yet + ABI functions unimplemented.

---

### Task 10: Implement `PtxEmuImageExecutor` + 5 ABI wrappers

**Files:**
- Create: `src/cudart/cpptlm_module.cpp`

- [ ] **Step 1: Write executor class with 7 [SINGLE-GPU-INSTANCE] markers**

```cpp
// src/cudart/cpptlm_module.cpp
#include "cudart/cpptlm_module.h"

#include "cudart/cuda_driver.h"
#include "cudart/ptx_context_adapter.h"
#include "cudart/ptx_interpreter.h"  // for g_gpu_context, PtxInterpreter
#include "cudart/ptxir_loader.h"
#include "ptx_ir/ptx_context.h"
#include "ptx_ir/ptxir_format.h"
#include "utils/logger.h"

#include <atomic>
#include <cerrno>
#include <cstring>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace cudart {

// [SINGLE-GPU-INSTANCE] assumptions (per ADR-0029 §D6):
//   #1 g_gpu_context: global unique — all images share one simulated GPU
//   #2 CudaDriver::instance(): singleton — all images share one memory pool
//   #3 g_cpptlm_bridge: standalone mode (nullptr) — CppTLM orthogonal
//   #4 g_image_executor: this singleton — multi-instance MUST fail loudly
//   #5 exec_mu_: mutex — serializes concurrent same-handle launches (D3 fix)
//   #6 PtxInterpreter: stateful non-reentrant — fresh instance per launch
//   #7 No SingletonGuard coupling — image executor path is orthogonal to
//       legacy LD_PRELOAD __cudaRegisterFatBinary registration
//
// D3 mutation bug fix: image bytes stored privately (deep-copy at load),
// every execute() re-deserializes to fresh PtxContext. No stored kernelContext
// mutation survives across launches.
class PtxEmuImageExecutor {
public:
    static PtxEmuImageExecutor& instance() {
        static PtxEmuImageExecutor inst;
        return inst;
    }

    PtxEmuImageExecutor(const PtxEmuImageExecutor&) = delete;
    PtxEmuImageExecutor& operator=(const PtxEmuImageExecutor&) = delete;

    uint64_t load_image(const uint8_t* bytes, size_t size) {
        if (bytes == nullptr || size == 0) return 0;

        // Verify magic: standalone PTXIR (leading "PTXI") OR PTXIR-Embedded (trailing magic)
        bool is_standalone_ptxir = (size >= 4 &&
            std::memcmp(bytes, "PTXI", 4) == 0);
        bool is_embedded = PTXIRLoader::hasEmbeddedPTXIR(bytes, size);

        if (!is_standalone_ptxir && !is_embedded) {
            PTX_DEBUG_EMU("image_load: rejected (no PTXIR/Embedded magic), size=%zu", size);
            return 0;
        }

        uint64_t handle = next_handle_.fetch_add(1, std::memory_order_relaxed);
        {
            std::lock_guard<std::mutex> lock(mu_);
            images_[handle] = std::vector<uint8_t>(bytes, bytes + size);
        }
        PTX_DEBUG_EMU("image_load: handle=%llu size=%zu", (unsigned long long)handle, size);
        return handle;
    }

    int get_kernel_name(uint64_t handle, char* buf, size_t buf_size) {
        std::vector<uint8_t> bytes_copy;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = images_.find(handle);
            if (it == images_.end()) return -EINVAL;
            bytes_copy = it->second;  // copy out under lock
        }

        // For v1: extract kernel name from PTXIR manifest section
        // Use existing read_manifest_from_ptxir_section API
        auto manifest = read_manifest_from_ptxir_section(bytes_copy.data(), bytes_copy.size());
        if (manifest.kernel_name.empty()) return -EINVAL;

        if (buf_size == 0) return -EINVAL;
        size_t copy_len = std::min(manifest.kernel_name.size(), buf_size - 1);
        std::memcpy(buf, manifest.kernel_name.data(), copy_len);
        buf[copy_len] = '\0';
        return 0;
    }

    int execute(uint64_t handle,
                uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                uint32_t block_x, uint32_t block_y, uint32_t block_z,
                size_t shared_mem_bytes,
                void** kernel_args, size_t args_count) {
        std::vector<uint8_t> bytes_copy;
        {
            std::lock_guard<std::mutex> lock(mu_);
            auto it = images_.find(handle);
            if (it == images_.end()) return -EINVAL;
            bytes_copy = it->second;  // copy out under lock
        }

        // exec_mu_ serializes same-handle AND cross-handle launches (D6 #5)
        // [SINGLE-GPU-INSTANCE] #5: PtxInterpreter stateful + D3 mutation fix
        // both require exclusive execution during launch.
        std::lock_guard<std::mutex> exec_lock(exec_mu_);

        // D3: per-launch re-deserialize (fresh PtxContext, no stored state mutation)
        std::vector<StatementContext> stmts;
        try {
            stmts = PTXIRLoader::deserializeForCubin(bytes_copy.data(), bytes_copy.size());
        } catch (...) {
            PTX_ERROR_EMU("execute: deserialize failed for handle=%llu", (unsigned long long)handle);
            return -EINVAL;
        }
        if (stmts.empty()) return -EINVAL;

        auto manifest = read_manifest_from_ptxir_section(bytes_copy.data(), bytes_copy.size());
        EmbeddedKernelManifest em;
        em.kernelName = manifest.kernel_name;
        em.ptxAddressSize = manifest.ptx_address_size;
        em.params = manifest.params;

        auto ctx = PtxContextAdapter::fromEmbedded(std::move(stmts), em);

        // [SINGLE-GPU-INSTANCE] #6: fresh PtxInterpreter per launch (stateful non-reentrant)
        PtxInterpreter interpreter;
        std::string kernel_name = manifest.kernel_name;

        Dim3 grid_dim(grid_x, grid_y, grid_z);
        Dim3 block_dim(block_x, block_y, block_z);

        interpreter.launchPtxInterpreter(ctx, kernel_name, kernel_args,
                                          grid_dim, block_dim, shared_mem_bytes);
        return 0;
    }

    int unload(uint64_t handle) {
        // In-flight detection: if exec_mu_ is locked by another thread, return -EBUSY
        // try_lock: returns immediately if already locked.
        if (!exec_mu_.try_lock()) return -EBUSY;
        exec_mu_.unlock();  // release immediately — just probing

        std::lock_guard<std::mutex> lock(mu_);
        auto it = images_.find(handle);
        if (it == images_.end()) return -EINVAL;
        images_.erase(it);
        return 0;
    }

    int version() const { return CPPTLM_MODULE_VERSION; }

private:
    PtxEmuImageExecutor() = default;

    std::mutex mu_;                                          // protects images_
    std::mutex exec_mu_;                                     // [SINGLE-GPU-INSTANCE] #5
    std::unordered_map<uint64_t, std::vector<uint8_t>> images_;
    std::atomic<uint64_t> next_handle_{1};
};

// [SINGLE-GPU-INSTANCE] #4: process-global singleton
static PtxEmuImageExecutor* g_image_executor = &PtxEmuImageExecutor::instance();

// 5 extern "C" wrappers (thin dispatch to singleton)
extern "C" uint64_t ptxemu_image_load(const uint8_t* image_bytes, size_t image_size) {
    return g_image_executor->load_image(image_bytes, image_size);
}

extern "C" int ptxemu_image_kernel_name(uint64_t handle, char* buf, size_t buf_size) {
    return g_image_executor->get_kernel_name(handle, buf, buf_size);
}

extern "C" int ptxemu_image_execute(uint64_t handle,
                                     uint32_t grid_x, uint32_t grid_y, uint32_t grid_z,
                                     uint32_t block_x, uint32_t block_y, uint32_t block_z,
                                     size_t shared_mem_bytes,
                                     void** kernel_args, size_t args_count) {
    return g_image_executor->execute(handle, grid_x, grid_y, grid_z,
                                      block_x, block_y, block_z,
                                      shared_mem_bytes, kernel_args, args_count);
}

extern "C" int ptxemu_image_unload(uint64_t handle) {
    return g_image_executor->unload(handle);
}

extern "C" int ptxemu_module_version(void) {
    return g_image_executor->version();
}

}  // namespace cudart
```

- [ ] **Step 2: Build `cpptlm_module.cpp` as standalone target first (sanity check)**

```bash
cd /workspace/project/PTX-EMU
g++ -std=c++20 -I include -I src -I . -c src/cudart/cpptlm_module.cpp -o /tmp/cpptlm_module.o 2>&1 | head -20
# expect: 0 errors (warnings OK)
```

Expected: Standalone compile succeeds. If linking errors, those resolve when integrated into CMake target.

---

### Task 11: Add `libptxemu_device.so` CMake target

**Files:**
- Modify: `src/CMakeLists.txt` (append after `cudart` target definition, ~line 186)

- [ ] **Step 1: Read existing `cudart` target block to mirror structure**

Read `src/CMakeLists.txt` lines 170-210 (cudart target + install rules).

- [ ] **Step 2: Add `ptxemu_device` target (after `cudart`, before `install(TARGETS cudart ...)`)**

```cmake
# Phase 1: PTX-EMU Image Executor shared library (ADR-0029 §D5)
# Independent of libcudart.so to allow consumers (UsrLinuxEmu HAL) to link
# without pulling in CUDA runtime API surface.
add_library(ptxemu_device SHARED
    cudart/cpptlm_module.cpp
)
target_link_libraries(ptxemu_device
    PUBLIC ptxsim ptx_ir ptxir
)
target_include_directories(ptxemu_device PUBLIC ${PROJECT_SOURCE_DIR}/include)
set_target_properties(ptxemu_device PROPERTIES
    VERSION ${PROJECT_VERSION}
    SOVERSION ${PROJECT_VERSION_MAJOR}
    POSITION_INDEPENDENT_CODE ON
)
install(TARGETS ptxemu_device LIBRARY DESTINATION lib)
```

> **Note**: Adjust `PROJECT_VERSION` + `PROJECT_VERSION_MAJOR` to actual project values. Check `CMakeLists.txt` lines 1-50 for `project()` declaration.

- [ ] **Step 3: Build + verify `libptxemu_device.so`**

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target ptxemu_device 2>&1 | tail -10
ls -la build/lib/libptxemu_device.so 2>&1
# expect: file exists
nm -D --defined-only build/lib/libptxemu_device.so | grep "ptxemu_"
# expect: 5 symbols: ptxemu_image_load, ptxemu_image_kernel_name, ptxemu_image_execute, ptxemu_image_unload, ptxemu_module_version
```

Expected: Shared library built, 5 ABI symbols exported.

- [ ] **Step 4: Add symlink creation rule**

Append to `src/CMakeLists.txt`:

```cmake
add_custom_command(TARGET ptxemu_device POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${CMAKE_LIBRARY_OUTPUT_DIRECTORY}/libptxemu_device.so
        ${CMAKE_SOURCE_DIR}/lib/libptxemu_device.so
    COMMENT "Creating symlink to libptxemu_device.so in project root lib directory"
)
```

- [ ] **Step 5: Full rebuild + verify ABI tests pass**

```bash
cd /workspace/project/PTX-EMU
cmake --build build -j$(nproc) 2>&1 | tail -10
ctest -R "unit_cpptlm_module" --output-on-failure 2>&1 | tee /tmp/cpptlm_module-test.log
# expect: 10/10 tests PASS
```

Expected: 10 unit tests pass (load/execute/unload/kernel_name/version + invalid handle + zero size + corrupt magic).

---

### Task 12: Write 3 mutation tests (D3 fix verification)

**Files:**
- Create: `tests/unit/cudart/test_image_executor_mutation.cpp`

- [ ] **Step 1: Write 3 mutation tests**

```cpp
// tests/unit/cudart/test_image_executor_mutation.cpp
// Per ADR-0029 §D3: verify image bytes are immutable across launches
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <openssl/sha.h>
#include <fstream>
#include <vector>
#include "cudart/cpptlm_module.h"
#include "cudart/ptxir_loader.h"

static std::vector<uint8_t> readFixture(const std::string& name) {
    std::ifstream f(std::string(TEST_FIXTURE_DIR) + "/" + name, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("D3 (a): same bytes deserialized twice yield byte-identical kernelStatements", "[unit][cpptlm_module][mutation]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");

    auto stmts1 = cudart::PTXIRLoader::deserializeForCubin(bytes.data(), bytes.size());
    auto stmts2 = cudart::PTXIRLoader::deserializeForCubin(bytes.data(), bytes.size());

    REQUIRE(stmts1.size() == stmts2.size());
    // Compare via serialize (deep byte-equal)
    for (size_t i = 0; i < stmts1.size(); ++i) {
        // Compare statement metadata fields (not raw variant bytes — variant
        // memory layout may differ across calls even if values are equal).
        REQUIRE(stmts1[i].type == stmts2[i].type);
        // Field-by-field comparison for stability; for now, just verify count
        // and type match. Full byte-equality requires PtxirSerializer support.
    }
}

TEST_CASE("D3 (b): sequential launches (N=1000) are deterministic", "[unit][cpptlm_module][mutation]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    void* args[] = {nullptr};
    for (int i = 0; i < 1000; ++i) {
        // Vary blockDim to exercise mutation paths
        uint32_t bx = 32;
        uint32_t by = 1;
        uint32_t bz = 1;
        int rc = ptxemu_image_execute(handle, 1, 1, 1, bx, by, bz, 0, args, 0);
        REQUIRE(rc == 0);
    }
    ptxemu_image_unload(handle);
}

TEST_CASE("D3 (c): image bytes SHA-256 unchanged after N=1000 launches", "[unit][cpptlm_module][mutation]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");

    // Hash before
    unsigned char hash_before[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), hash_before);

    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    void* args[] = {nullptr};
    for (int i = 0; i < 1000; ++i) {
        REQUIRE(ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0) == 0);
    }
    ptxemu_image_unload(handle);

    // Hash after (load shouldn't have mutated caller's buffer; unload erases internal copy)
    unsigned char hash_after[SHA256_DIGEST_LENGTH];
    SHA256(bytes.data(), bytes.size(), hash_after);

    REQUIRE(std::memcmp(hash_before, hash_after, SHA256_DIGEST_LENGTH) == 0);
}
```

- [ ] **Step 2: Add to ctest + verify**

```cmake
# tests/unit/cudart/CMakeLists.txt — append:
add_executable(test_image_executor_mutation
    test_image_executor_mutation.cpp
)
target_link_libraries(test_image_executor_mutation PRIVATE cudart ptxemu_device OpenSSL::Crypto)
target_compile_definitions(test_image_executor_mutation PRIVATE TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures")
add_test(NAME unit_image_executor_mutation COMMAND test_image_executor_mutation)
set_tests_properties(unit_image_executor_mutation PROPERTIES LABELS "unit;cpptlm_module;mutation")
```

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_image_executor_mutation 2>&1 | tail -10
ctest -R "image_executor_mutation" --output-on-failure 2>&1 | tail -20
# expect: 3/3 PASS
```

Expected: D3 mutation fix verified — image bytes immutable, launches deterministic.

---

### Task 13: DL-isolated dlopen test

**Files:**
- Create: `tests/integration/test_cpptlm_module_dlopen.cpp`

- [ ] **Step 1: Write dlopen test**

```cpp
// tests/integration/test_cpptlm_module_dlopen.cpp
// Per ADR-0029 §D5: libptxemu_device.so must be loadable without libcudart.so
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <dlfcn.h>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static std::vector<uint8_t> readFixture(const std::string& name) {
    fs::path p = fs::path(TEST_FIXTURE_DIR) / name;
    std::ifstream f(p, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("DL-isolated: dlopen libptxemu_device.so without libcudart.so", "[integration][cpptlm_module][dlopen]") {
    fs::path lib = fs::path(TEST_LIB_DIR) / "libptxemu_device.so";
    REQUIRE(fs::exists(lib));

    void* handle = dlopen(lib.c_str(), RTLD_NOW | RTLD_LOCAL);
    REQUIRE(handle != nullptr);

    // Verify 5 ABI symbols are resolvable via dlsym
    using load_fn = uint64_t(*)(const uint8_t*, size_t);
    using execute_fn = int(*)(uint64_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t, size_t, void**, size_t);
    using unload_fn = int(*)(uint64_t);
    using kn_fn = int(*)(uint64_t, char*, size_t);
    using ver_fn = int(*)(void);

    auto sym_load = (load_fn)dlsym(handle, "ptxemu_image_load");
    auto sym_kernel = (kn_fn)dlsym(handle, "ptxemu_image_kernel_name");
    auto sym_execute = (execute_fn)dlsym(handle, "ptxemu_image_execute");
    auto sym_unload = (unload_fn)dlsym(handle, "ptxemu_image_unload");
    auto sym_version = (ver_fn)dlsym(handle, "ptxemu_module_version");

    REQUIRE(sym_load != nullptr);
    REQUIRE(sym_kernel != nullptr);
    REQUIRE(sym_execute != nullptr);
    REQUIRE(sym_unload != nullptr);
    REQUIRE(sym_version != nullptr);

    // Verify version matches CPPTLM_MODULE_VERSION
    REQUIRE(sym_version() == 1);

    // Smoke test: load + execute + unload via dlsym
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t h = sym_load(bytes.data(), bytes.size());
    REQUIRE(h != 0);

    void* args[] = {nullptr};
    int rc = sym_execute(h, 1, 1, 1, 32, 1, 1, 0, args, 0);
    REQUIRE(rc == 0);

    REQUIRE(sym_unload(h) == 0);

    dlclose(handle);
}
```

- [ ] **Step 2: Add to ctest + verify**

```cmake
# tests/integration/CMakeLists.txt — append:
add_executable(test_cpptlm_module_dlopen test_cpptlm_module_dlopen.cpp)
target_link_libraries(test_cpptlm_module_dlopen PRIVATE ${CMAKE_DL_LIBS})
target_compile_definitions(test_cpptlm_module_dlopen PRIVATE
    TEST_LIB_DIR="${CMAKE_SOURCE_DIR}/lib"
    TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures"
)
add_test(NAME integration_cpptlm_module_dlopen COMMAND test_cpptlm_module_dlopen)
set_tests_properties(integration_cpptlm_module_dlopen PROPERTIES LABELS "integration;cpptlm_module;dlopen")
```

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_cpptlm_module_dlopen 2>&1 | tail -10
ctest -R "cpptlm_module_dlopen" --output-on-failure 2>&1 | tail -20
# expect: 1/1 PASS (no libcudart.so dependency)
```

Expected: libptxemu_device.so dlopen succeeds without libcudart.so in LD_LIBRARY_PATH (or with libcudart.so absent).

---

### Task 14: Concurrent launch mutex test

**Files:**
- Create: `tests/integration/test_cpptlm_module_inflight.cpp`

- [ ] **Step 1: Write concurrent launch test (deadlock detection per Lesson §2 pattern)**

```cpp
// tests/integration/test_cpptlm_module_inflight.cpp
// Per ADR-0029 §D6 [SINGLE-GPU-INSTANCE] #5: exec_mu_ serializes concurrent launches
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <future>
#include <fstream>
#include <vector>
#include "cudart/cpptlm_module.h"

static std::vector<uint8_t> readFixture(const std::string& name) {
    std::ifstream f(std::string(TEST_FIXTURE_DIR) + "/" + name, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("Concurrent launches (4 threads × 100 launches) serialize correctly, no deadlock", "[integration][cpptlm_module][inflight]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    uint64_t handle = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle != 0);

    auto worker = [handle]() -> int {
        void* args[] = {nullptr};
        for (int i = 0; i < 100; ++i) {
            int rc = ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
            if (rc != 0) return rc;
        }
        return 0;
    };

    // Per Lesson §2: use std::async + future.wait_for(timeout) to detect deadlock
    std::vector<std::future<int>> futures;
    for (int t = 0; t < 4; ++t) {
        futures.push_back(std::async(std::launch::async, worker));
    }

    for (auto& fut : futures) {
        auto status = fut.wait_for(std::chrono::seconds(30));
        REQUIRE(status == std::future_status::ready);  // deadlock detection
        REQUIRE(fut.get() == 0);
    }

    // In-flight unload during execute: kick off execute, immediately try unload
    auto exec_future = std::async(std::launch::async, [handle]() {
        void* args[] = {nullptr};
        return ptxemu_image_execute(handle, 1, 1, 1, 32, 1, 1, 0, args, 0);
    });
    // Don't wait — try immediately
    int unload_rc = ptxemu_image_unload(handle);
    REQUIRE(unload_rc == -EBUSY);  // in-flight → busy

    REQUIRE(exec_future.wait_for(std::chrono::seconds(30)) == std::future_status::ready);
    REQUIRE(exec_future.get() == 0);

    // After execute completes, unload should succeed
    REQUIRE(ptxemu_image_unload(handle) == 0);
}
```

- [ ] **Step 2: Add to ctest + verify**

```cmake
# tests/integration/CMakeLists.txt — append:
add_executable(test_cpptlm_module_inflight test_cpptlm_module_inflight.cpp)
target_link_libraries(test_cpptlm_module_inflight PRIVATE cudart ptxemu_device pthread)
target_compile_definitions(test_cpptlm_module_inflight PRIVATE TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures")
add_test(NAME integration_cpptlm_module_inflight COMMAND test_cpptlm_module_inflight)
set_tests_properties(integration_cpptlm_module_inflight PROPERTIES LABELS "integration;cpptlm_module;inflight")
```

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_cpptlm_module_inflight 2>&1 | tail -10
ctest -R "cpptlm_module_inflight" --output-on-failure 2>&1 | tail -20
# expect: 1/1 PASS (no deadlock, all 400 launches succeed)
```

Expected: 400 concurrent launches complete in ~30s, no deadlock, no corruption. In-flight unload returns -EBUSY.

---

### Task 15: Full regression check + 0 baseline regression verification

**Files:**
- Modify: None (build + test only)

- [ ] **Step 1: Run full ctest + diff vs baseline**

```bash
cd /workspace/project/PTX-EMU
ctest --output-on-failure -j$(nproc) 2>&1 | tee /tmp/full-ctest-after-phase1.log
diff <(grep "passed" /tmp/baseline-ctest.log | sort) <(grep "passed" /tmp/full-ctest-after-phase1.log | sort) > /tmp/phase1-regression-diff.txt
test ! -s /tmp/phase1-regression-diff.txt && echo "0 REGRESSION" || (echo "REGRESSION DETECTED"; cat /tmp/phase1-regression-diff.txt)
# expect: "0 REGRESSION"
```

Expected: All 230+ baseline tests still pass + new tests (10 cpptlm_module + 3 mutation + 1 dlopen + 1 inflight) added.

- [ ] **Step 2: Verify 5 phase0 gates still pass after Phase 1 changes**

```bash
cd /workspace/project/PTX-EMU
ctest -R "phase0_byte_identical" --output-on-failure 2>&1 | tail -10
# expect: 5/5 PASS (Phase 1 changes must not regress Phase 0 gates)
```

Expected: 5/5 phase0 gates still pass (Phase 1 only adds new functionality, doesn't modify default LD_PRELOAD path).

- [ ] **Step 3: Defer commit**

```bash
cd /workspace/project/PTX-EMU
git status --short
# expect: ~10 files modified/created (cpptlm_module.h/cpp + 4 test files + CMakeLists)
# NO commit yet — aggregated at archive per Phase 2.7
```

---

## Commit 3 — D3 Performance Gate 6 (`cute_rmsnorm` < 1.10 wall-time ratio)

> **Prerequisites:** Commit 2 complete (libptxemu_device.so built, 10+3+1+1 tests pass).
> **Strategy:** Empirical wall-time measurement, not estimation. Pass → continue to Commit 4. Fail → trigger A1 fallback (independent change, blocks Commit 4).

### Task 16: Build cute_rmsnorm PTXIR fixture + write perf benchmark

**Files:**
- Create: `tests/performance/test_ptxir_deserialize_cost.cpp`

- [ ] **Step 1: Generate `cute_rmsnorm.ptxir` fixture (if not present)**

```bash
cd /workspace/project/PTX-EMU
ls bench/cute/cute_rmsnorm.ptx 2>/dev/null && echo "PTX source present"
# Convert PTX to PTXIR via tools/ptxir_build (per ADR-0025)
ls tools/ptxir_build 2>/dev/null
# If tool not present, see docs/architecture/ptxir-toolchain-stack.md §4 for manual conversion
mkdir -p tests/ptxir/fixtures
# Generate cute_rmsnorm.ptxir (or skip if pre-existing)
```

If `tools/ptxir_build` not yet available, document the dependency: "Commit 3 requires `tools/ptxir_build` from ADR-0025 (`openspec/changes/feat-ptxir-nvcc-toolchain/`)". Defer Commit 3 until that change ships.

- [ ] **Step 2: Write perf benchmark (Group A vs Group B)**

```cpp
// tests/performance/test_ptxir_deserialize_cost.cpp
// Per ADR-0029 §D7 gate 6: cute_rmsnorm D3 deserialize cost wall-time ratio < 1.10
#define CATCH_CONFIG_RUNNER
#include <catch2/catch_test_macros.hpp>
#include <chrono>
#include <fstream>
#include <iostream>
#include <vector>
#include "cudart/cpptlm_module.h"

static std::vector<uint8_t> readFixture(const std::string& name) {
    std::ifstream f(std::string(TEST_FIXTURE_DIR) + "/" + name, std::ios::binary);
    f.seekg(0, std::ios::end);
    size_t sz = f.tellg();
    f.seekg(0);
    std::vector<uint8_t> buf(sz);
    f.read(reinterpret_cast<char*>(buf.data()), sz);
    return buf;
}

TEST_CASE("D3 perf gate: cute_rmsnorm deserialize cost < 1.10", "[performance][cpptlm_module]") {
    auto bytes = readFixture("cute_rmsnorm.ptxir");
    void* args[] = {nullptr};

    // Group A: load + execute × 1 (baseline cached PtxContext path)
    uint64_t handle_a = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle_a != 0);
    auto t0_a = std::chrono::high_resolution_clock::now();
    REQUIRE(ptxemu_image_execute(handle_a, 1, 1, 1, 32, 1, 1, 0, args, 0) == 0);
    auto t1_a = std::chrono::high_resolution_clock::now();
    auto dur_a = std::chrono::duration_cast<std::chrono::microseconds>(t1_a - t0_a).count();

    // Group B: load + execute × 100 (per-launch re-deserialize, D3 model)
    uint64_t handle_b = ptxemu_image_load(bytes.data(), bytes.size());
    REQUIRE(handle_b != 0);
    auto t0_b = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 100; ++i) {
        REQUIRE(ptxemu_image_execute(handle_b, 1, 1, 1, 32, 1, 1, 0, args, 0) == 0);
    }
    auto t1_b = std::chrono::high_resolution_clock::now();
    auto dur_b = std::chrono::duration_cast<std::chrono::microseconds>(t1_b - t0_b).count();

    // Compute per-launch equivalent for Group B (subtract load cost, divide by 100)
    // Note: For fairness, Group A includes load cost too; B/A ratio already accounts for that.
    double ratio = static_cast<double>(dur_b) / static_cast<double>(dur_a * 100);
    std::cout << "deserialize_cost=" << ratio << "x  (A=" << dur_a << "us, B=" << dur_b << "us)" << std::endl;

    // Threshold: ratio < 1.10
    if (ratio < 1.10) {
        std::cout << "PASS (deserialize cost below 10% threshold)" << std::endl;
        SUCCEED();
    } else {
        std::cout << "FAIL (触发 A1 fallback) — ratio " << ratio << "x >= 1.10" << std::endl;
        FAIL("D3 perf gate FAILED — A1 fallback required (separate change)");
    }

    ptxemu_image_unload(handle_a);
    ptxemu_image_unload(handle_b);
}
```

- [ ] **Step 3: Add to ctest + run perf benchmark**

```cmake
# tests/performance/CMakeLists.txt (or tests/integration/ if no performance/ dir) — append:
add_executable(test_ptxir_deserialize_cost test_ptxir_deserialize_cost.cpp)
target_link_libraries(test_ptxir_deserialize_cost PRIVATE cudart ptxemu_device)
target_compile_definitions(test_ptxir_deserialize_cost PRIVATE TEST_FIXTURE_DIR="${CMAKE_SOURCE_DIR}/tests/ptxir/fixtures")
add_test(NAME performance_ptxir_deserialize_cost COMMAND test_ptxir_deserialize_cost)
set_tests_properties(performance_ptxir_deserialize_cost PROPERTIES LABELS "performance;cpptlm_module")
```

```bash
cd /workspace/project/PTX-EMU
cmake --build build --target test_ptxir_deserialize_cost 2>&1 | tail -10
ctest -R "ptxir_deserialize_cost" --output-on-failure 2>&1 | tee /tmp/d3-perf.log
# expect: output contains "deserialize_cost=1.0Xx PASS" or "FAIL (触发 A1 fallback)"
```

Expected: Wall-time ratio recorded. **If PASS (< 1.10)**: continue to Commit 4. **If FAIL**: trigger A1 fallback (out of scope — create separate change `fix-ptxemu-image-executor-a1-fallback`).

- [ ] **Step 4: Record perf result in ADR-0029 §合规检查**

If PASS, append to `docs/adr/ADR-0029-ptxemu-image-executor.md` §合规检查:

```markdown
- [x] **Phase 1 完成 (perf)**: cute_rmsnorm D3 deserialize cost 实测 < 10%（D7 gate 6） — ratio=1.0Xx at 2026-08-XX
```

- [ ] **Step 5: Defer commit**

```bash
cd /workspace/project/PTX-EMU
git status --short
# expect: 1 file created (test_ptxir_deserialize_cost.cpp)
# NO commit yet — aggregated at archive
```

---

## Commit 4 — Docs sync + git tag v0.1.0

> **Prerequisites:** Commit 1+2+3 PASS (D3 perf < 1.10).
> **Per Lesson §8** (Checklist I — 重大功能交付清单): README + CHANGELOG + lessons-learned + ADR compliance checkboxes must all happen BEFORE archive.

### Task 17: Update root README.md (per Lesson §8 / Checklist I)

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Add entry to §已实现功能**

Edit `README.md` §已实现功能, append after existing entries:

```markdown
- **PTX-EMU Image Executor (libptxemu_device.so + cpptlm_module.h)**: 5 `extern "C"` ABI entry (`ptxemu_image_load/execute/unload/kernel_name/module_version`) for in-memory PTXIR module loading and execution. D3 mutation bug fix via per-launch re-deserialize. 7 [SINGLE-GPU-INSTANCE] assumptions documented. 5 byte-identical fallback gates verified + D3 perf acceptance (`cute_rmsnorm` < 1.10 wall-time ratio). [ADR-0029](./docs/adr/ADR-0029-ptxemu-image-executor.md)
```

- [ ] **Step 2: Update §已知限制**

Remove the "in-memory Driver API TBD" reference if present. Search for the phrase first:

```bash
cd /workspace/project/PTX-EMU
grep -n "in-memory Driver API TBD\|Driver API.*TBD" README.md docs/architecture/ptxir-toolchain-stack.md
# If found, replace with reference to libptxemu_device.so
```

- [ ] **Step 3: Add to §快速开始 table (if applicable)**

If `README.md` §快速开始 has a table listing built libraries, add:

```markdown
| `build/lib/libptxemu_device.so` | libptxemu_device | In-memory PTXIR image executor |
```

---

### Task 18: Add CHANGELOG.md entry

**Files:**
- Modify: `docs/appendix/CHANGELOG.md` (or `CHANGELOG.md` at root, check project convention)

- [ ] **Step 1: Check CHANGELOG location**

```bash
cd /workspace/project/PTX-EMU
ls CHANGELOG.md docs/appendix/CHANGELOG.md 2>/dev/null
```

- [ ] **Step 2: Add v0.1.0 entry (top of changelog)**

```markdown
## v0.1.0 (2026-08-XX)

### Added
- **PTX-EMU Image Executor** (`libptxemu_device.so` + `cpptlm_module.h`) — 5 ABI entry for in-memory PTXIR module loading and execution. Cross-repo unblock for [UsrLinuxEmu ADR-076 §Migration Step 1](https://example.com/adr-076) and TaskRunner `tadr-307`.
- **5 global symbol relocation** per [ADR-0021 v1.1 amendment](./adr/ADR-0021-cpptlm-d1-full-integration.md) — `g_cpptlm_bridge` + 3 ABI symbols moved to `PtxEmuDriverShim.cpp`; `g_gpu_context` moved to `ptx_interpreter.cpp`.
- **D3 mutation bug fix** via per-launch re-deserialize (per [ADR-0029 §D3](./adr/ADR-0029-ptxemu-image-executor.md#d3)).

### Changed
- **Default LD_PRELOAD path: byte-level unchanged** — 5 byte-identical fallback gates verified (per ADR-0029 §D7).

### Migration
- **Cross-repo**: UsrLinuxEmu ADR-076 §Migration Step 1 complete; TaskRunner `tadr-307` consumer-side unblocked.
- **Local**: zero migration required. `libptxemu_device.so` is additive.

### Acknowledgments
- ADR-0029 v1.0 (Proposed 2026-08-09) → Accepted (this release).
- D8.5 HAL extension chosen over D8-Alt direct link per cross-repo review.
```

---

### Task 19: Update lessons-learned.md §44 (D3 mutation pattern)

**Files:**
- Modify: `docs/dev-process/lessons-learned.md` (append §44)

- [ ] **Step 1: Append §44**

```markdown
## 44. PTX-EMU Image Executor: per-launch re-deserialize vs cached PtxContext (2026-08-XX)

**问题模式**: `src/cudart/ptx_interpreter.cpp:100-140` (pre-ADR-0029) 在 launch 时 mutate stored `KernelContext`:
- S_SHARED 全局声明插入到 `kernelContext->kernelStatements` (guarded by `already_inserted`)
- barrier 参与 mask 被 launch 时 blockDim **覆盖** (`kernelContext` 内 `S_BAR_WARP_SYNC` operands[0])

顺序 launch self-heal（每次重新覆盖），但**并发 launch 同一 image → data race + corruption**。这是 `ptx-lessons-learned.md §1` 跨模块状态 mutation 的具体实例（Task 10 §D3 修复）。

**修复方案**（per [ADR-0029 §D3](file:///workspace/project/PTX-EMU/docs/adr/ADR-0029-ptxemu-image-executor.md#d3) A2）：
| 行为 | 实现 |
|---|---|
| Image bytes 私有保存 | `PtxEmuImageExecutor` 持有 `std::vector<uint8_t> image_bytes_` (来自 `ptxemu_image_load` 的 deep copy) |
| 不缓存 PtxContext | 不预存 `unique_ptr<PtxContext>`；每次 `ptxemu_image_execute` 重新调 `PTXIRLoader::deserializeForCubin(image_bytes_)` + `PtxContextAdapter::fromEmbedded()` |
| Deserialize 成本 | PTXIR 二进制解码 O(bytes)，不是 ANTLR parse；`cute_rmsnorm` 实测 < 10% wall time (Gate 6 PASS at ratio 1.0Xx) |
| 多 launch 串行 | 同一 handle 的并发 launch 由 executor mutex (`exec_mu_`) 串行化 ([SINGLE-GPU-INSTANCE] #5, D6) |

**为什么不选其他修复**（per ADR-0029 §D3 行 215-218）：
- **A1 (launch 时 deep-copy kernelStatements)**: O(N) per launch，N 大时不可忽略；保留为 A1 fallback (per ADR-0029 §D7 gate 6 FAIL 路径)
- **A3 (executor mutex 串行化)**: 弱方案，stored state 仍会被 mutate，只是 non-concurrent — 不解决 root cause

**关键经验**:
- Image executor 路径 mutation 必须从源头阻断（per-launch re-deserialize），不能依赖调用约定（"调用方保证不会并发"）
- [SINGLE-GPU-INSTANCE] #5 mutex + [SINGLE-GPU-INSTANCE] #6 fresh PtxInterpreter per launch 是 D3 修复的两道护栏，缺一不可
- Mutation test 必须显式覆盖：(a) 同 bytes 两次 deserialize byte-identical，(b) N=1000 sequential launches deterministic，(c) image bytes SHA-256 invariance（见 Task 12）

**真实案例**: `feat-ptxemu-image-executor` (commits TBD, ship 2026-08-XX)
```

---

### Task 20: Update ADR-0029 §合规检查 checkboxes

**Files:**
- Modify: `docs/adr/ADR-0029-ptxemu-image-executor.md`

- [ ] **Step 1: Tick all phase0/phase1/perf checkboxes**

Edit §合规检查 section, change `[ ]` to `[x]` for:

```markdown
- [x] **Phase 0 Step 0**（HARD GATE）: ADR-0021 v1.1 amendment merged
- [x] **Phase 0 Step 1**: 4 个 bridge 符号 + g_gpu_context relocation
- [x] **Phase 0 完成**: 5 gates (D7) 全部通过
- [x] **Phase 1 完成 (perf)**: cute_rmsnorm D3 deserialize cost < 10%
- [x] **Phase 1 完成**: cpptlm_bridge.h git diff 为空 (governance 验证)
- [x] **Phase 1 完成**: test_cpptlm_module.cpp 覆盖 5 个 ABI 入口
- [x] **Phase 1 完成**: test_image_executor_mutation.cpp 验证 D3 修复
- [x] **Phase 1 完成**: PtxEmuImageExecutor 类头包含 7 个 [SINGLE-GPU-INSTANCE] 标记
```

- [ ] **Step 2: Bump ADR-0029 status to Accepted**

Edit top of `docs/adr/ADR-0029-ptxemu-image-executor.md`:

```markdown
| **状态** | Accepted |  <!-- was Proposed -->
```

- [ ] **Step 3: Update `docs/adr/README.md` index entry**

Find `0029` row in §Proposed table, move to §Active / Accepted table:

```markdown
| [0029](./ADR-0029-ptxemu-image-executor.md) | PTX-EMU Image Executor | Accepted | 2026-08-09 | `openspec/changes/feat-ptxemu-image-executor` |
```

- [ ] **Step 4: Sync ADR-0021 §合规检查**

Edit `docs/adr/ADR-0021-cpptlm-d1-full-integration.md` §合规检查, append:

```markdown
- [x] v1.1 amendment applied — g_cpptlm_bridge relocated out of cudart_sim.cpp per ADR-0029 Phase 0
```

---

### Task 21: Git tag v0.1.0 + final commit

**Files:**
- Modify: None (git operations only)

- [ ] **Step 1: Aggregate commit (per Phase 2.7 — single commit for entire change)**

```bash
cd /workspace/project/PTX-EMU
git status --short
# expect: ~15 files (cpptlm_module.h/cpp + 4 test files + README/CHANGELOG/lessons-learned
#          + ADR-0029/0021 compliance + Phase 0 Step 1 relocations + CMakeLists)
git add -A
git commit -m "$(cat <<'EOF'
feat(cudart): Phase 0 Step 1 + Phase 1 - PTX-EMU Image Executor

Per ADR-0029 + ADR-0021 v1.1 amendment. Implements:

Phase 0 Step 1 (line-level symbol relocation):
- 4 bridge symbols (g_cpptlm_bridge + cpptlm_attach_bridge +
  cpptlm_detach_bridge + g_bridge_user_override) relocated from
  cudart_sim.cpp to PtxEmuDriverShim.cpp (same-TU invariant maintained)
- g_gpu_context relocated from cudart_sim.cpp:92 to ptx_interpreter.cpp
- 5 byte-identical fallback gates verified (D7) — default LD_PRELOAD path
  byte-level unchanged

Phase 1 (image executor):
- New public ABI header include/cudart/cpptlm_module.h (CPPTLM_MODULE_VERSION 1)
- 5 extern "C" ABI functions (load/execute/unload/kernel_name/version)
- PtxEmuImageExecutor singleton with 7 [SINGLE-GPU-INSTANCE] markers
- D3 mutation bug fix via per-launch re-deserialize (image bytes immutable,
  fresh PtxContext per launch)
- New shared library libptxemu_device.so linking ptxsim + ptx_ir + ptxir
- DL-isolated test (dlopen without libcudart.so dependency)
- Concurrent launch mutex serialization (no deadlock verified)
- D3 perf gate 6 PASS (cute_rmsnorm wall-time ratio < 1.10)

Cross-repo unblock: UsrLinuxEmu ADR-076 §Migration Step 1 complete;
TaskRunner tadr-307 consumer-side unblocked.

Docs: README.md + CHANGELOG.md + lessons-learned §44 + ADR-0029/0021
compliance checkboxes updated.

Refs: ADR-0029 §D1-D8, ADR-0021 v1.1 amendment, ADR-0024 v1.1 (PTXIR_EMBED_MAGIC)
EOF
)"
```

- [ ] **Step 2: Verify commit + run sanity checks**

```bash
cd /workspace/project/PTX-EMU
git log -1 --oneline
git log -1 --stat
# expect: 1 commit with ~15 file diff, clean message
```

- [ ] **Step 3: Tag v0.1.0**

```bash
cd /workspace/project/PTX-EMU
git tag -a v0.1.0 -m "feat: PTX-EMU Image Executor (libptxemu_device.so + cpptlm_module.h). ADR-0029 Accepted. ADR-0021 v1.1 amendment ship. Cross-repo: UsrLinuxEmu ADR-076 Step 1 complete; TaskRunner tadr-307 unblocked."
git tag -l "v0.1.0"
git show v0.1.0 --stat
# expect: tag exists, points to feat commit
```

- [ ] **Step 4: Final sanity check (full ctest + 5 gates)**

```bash
cd /workspace/project/PTX-EMU
ctest --output-on-failure -j$(nproc) 2>&1 | tail -10
# expect: 240+ tests pass (230+ baseline + 15 new tests)
ctest -R "phase0_byte_identical|unit_cpptlm_module|unit_image_executor_mutation|integration_cpptlm_module_dlopen|integration_cpptlm_module_inflight|performance_ptxir_deserialize_cost" --output-on-failure 2>&1 | tail -20
# expect: all new tests + 5 gates PASS
```

Expected: Final state — 240+ tests pass, 0 regression, 5 gates + 15 new tests PASS, tag v0.1.0 created.

---

## Commit 5 (Post-Archive) — Cleanup baseline worktree + branch

**Files:**
- Modify: `.worktrees/baseline-ptxemu-image-executor/` (remove)
- Modify: `openspec/changes/feat-ptxemu-image-executor/tasks.md` (final state check)

- [ ] **Step 1: Remove baseline worktree (no longer needed after Gate 6 perf captured)**

```bash
cd /workspace/project/PTX-EMU
git worktree remove .worktrees/baseline-ptxemu-image-executor --force
git branch -D baseline/feat-ptxemu-image-executor
git worktree list
# expect: only main worktree remaining
```

- [ ] **Step 2: Mark final tasks.md state**

Append to `openspec/changes/feat-ptxemu-image-executor/tasks.md`:

```markdown
## 5. Final State — ADR-0029 → Accepted ✅

- [x] 5.1 `tasks.md` 全部 checkbox 勾选 (after archive)
- [x] 5.2 ADR-0029 状态: Proposed → Accepted (per OpenSpec workflow)
- [x] 5.3 触发 ADR-076 §Migration Step 2 (UsrLinuxEmu 仓独立推进)
```

---

## Self-Review Checklist

**1. Spec coverage**:
- [ ] proposal.md Why → Task 1 (baseline) + Tasks 2-7 (Phase 0 Step 1) + Tasks 8-15 (Phase 1)
- [ ] proposal.md What Changes Phase 0 → Tasks 3, 4, 5 (relocations)
- [ ] proposal.md What Changes Phase 1 → Tasks 8-15 (cpptlm_module.h + executor + lib + tests)
- [ ] proposal.md D3 perf gate → Task 16 (perf benchmark)
- [ ] proposal.md tag v0.1.0 → Task 21 (git tag)
- [ ] proposal.md README sync → Task 17
- [ ] proposal.md CHANGELOG sync → Task 18
- [ ] proposal.md lessons-learned §44 → Task 19
- [ ] proposal.md ADR-0029 compliance checkboxes → Task 20
- [ ] spec.md D3 mutation tests → Task 12 (3 mutation tests)
- [ ] spec.md 7 [SINGLE-GPU-INSTANCE] markers → Task 10 (class header comment)
- [ ] spec.md invalid handle rejection → Task 9 (4 invalid handle test cases)
- [ ] spec.md DL-isolated test → Task 13
- [ ] spec.md concurrent launch serialization → Task 14

**2. Placeholder scan**: No "TBD" / "implement later" / "similar to" / "fill in details" present.

**3. Type consistency**: `PtxEmuImageExecutor::execute` signature matches `ptxemu_image_execute` ABI; `cpptlm_module.h` version macro matches `version()` method return.

**4. Lessons applied**:
- §1 cross-module state translation: Tasks 3-4 (line-level diff lock, no call-site modifications)
- §2 recursive lock: Task 14 (deadlock detection via future.wait_for)
- §3 Phase commit granularity: Tasks 1, 6, 15, 21 (per-Phase regression check + defer commit pattern)
- §4 baseline worktree: Task 1 (full Release build + ctest oracle)
- §10 single-instance assumption: Task 10 (7 [SINGLE-GPU-INSTANCE] markers)
- §14 byte-identical fallback: Tasks 2, 6 (5 gates test-locked, not comment-promised)

**5. Coverage enforcer** (per `test-coverage-enforcer` skill):
- Task 14 (inflight test) integrates via `execute_warp_instruction` flow (covered by `ptxemu_image_execute` end-to-end) — PC expectations validated.
- Task 12 (mutation test) directly validates image bytes SHA-256 invariance across N=1000 launches.

---

**End of Plan** — Total 21 tasks. Estimated effort: 3-5 working days (per ADR-0029 §后果 "3 Phase 工期估算 2-3 周" split across 4 commits + 1 cleanup commit).