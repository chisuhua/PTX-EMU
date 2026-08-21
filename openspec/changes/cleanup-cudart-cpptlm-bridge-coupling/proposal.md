## Why

PTX-EMU's `libcudart.so` currently contains ~130 lines of bridge coupling to CppTLM (reverse ABI registration via `cpptlm_set_driver` + `g_cpptlm_bridge` async path + `StubBridge` auto-attach + `PendingKernel` state machine + GLOBAL LD/ST bridge in `memory.cpp`). This coupling:
- Causes Gate 1 baseline leak (handled separately by `fix-phase0-gate1-dgpu-bar-leak`)
- Adds 8 test files in `tests/{unit,integration}/cpptlm/` (15 files total, of which 7 are bridge-specific and 8 are independent ADR-0020 injection-point tests) and 3 e2e cosim `.cu` files that depend on the bridge
- Binds PTX-EMU to CppTLM at link time, blocking standalone deployment scenarios (e.g., when only the `libptxemu_device.so` consumer path is desired)
- Introduces fragile state: `g_cpptlm_bridge` consumer checks at 7 sites in `cudart_sim.cpp` that branch between sync/async paths

Per Metis independent review (2026-08-21), removing this coupling is safe because:
1. CppTLM's `KernelLaunchTLM::tick()` already has a fallback (`bridge_ == nullptr` Phase 8.A path) that doesn't require `g_ptx_emu_driver`
2. `ScoreboardTLM`/`PipelineTLM`/`TensorCoreTLM` are vendored interfaces with zero `IPtxEmuDriver` references (per Metis verification)
3. The 6 "hidden consumer sites" (`generate_kernel_id`, `g_active_streams`, `g_pending_kernels_mutex`, etc.) require careful line-level surgery but are tractable
4. `libptxemu_device.so` (the public ABI CppTLM-side alternative consumer) already exists with 8 `ptxemu_image_*` exports — its existence is unaffected

This change makes `libcudart.so` a pure CUDA runtime shim with zero CppTLM coupling, but does NOT reverse the coupling direction (no `PtxEmuSubmodule`, no dlopen to `libptxemu_device.so` — that is a separate future change explicitly out of scope).

## What Changes

**Source code REMOVALS (PTX-EMU-side)**:
- **BREAKING**: Delete `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` (63 lines) — defines reverse ABI vtable
- **BREAKING**: Delete `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` (121 lines) — implements `g_cpptlm_bridge` global + `cpptlm_attach_bridge`/`detach_bridge`
- **BREAKING**: Delete `src/cudart/stub_bridge.h` (59 lines) — `StubBridge` zero-latency class
- **BREAKING**: Delete `src/cudart/cpptlm_module.cpp` (entire file ~290 lines) — `ptxemu_image_*` implementations. NOTE: these 8 symbols will move to a separate library or be re-implemented in a follow-up reverse direction change. For now, `libptxemu_device.so` is no longer built (its only source is deleted).

Wait — `cpptlm_module.cpp` is the source for `libptxemu_device.so` per `src/CMakeLists.txt:185-187`. Deleting it breaks `libptxemu_device.so` building. RECONSIDER: this change should NOT delete `cpptlm_module.cpp` until the reverse direction is implemented. Move this deletion to the future reversal change.

Revised list:
- Delete `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h`
- Delete `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`
- Delete `src/cudart/stub_bridge.h`
- Modify `src/cudart/cudart_sim.cpp`:
  - Remove includes L105-107 (`cpptlm_bridge.h`, `PtxEmuDriverShim.h`, `stub_bridge.h`)
  - Remove `g_ptx_emu_driver_shim` definition L118
  - Remove 8 shim_* vtable functions + weak `cpptlm_set_driver` L121-158
  - **Modify**: `g_active_streams` (L180) — keep BUT remove `cudaLaunchKernel` L784 insertion; `cudaStreamCreate` (L1233) and `cudaStreamDestroy` (L1258) continue to use it for non-bridge stream tracking
  - **Modify**: `g_pending_kernels` (L179) + `PendingKernel` struct (L167-176) — DELETE
  - **Modify**: `g_pending_kernels_mutex` (L181) — DELETE (only used by bridge path + `cudaStreamDestroy` L1257 which uses it for `g_active_streams` erase; remove and use simple lock-free erase for `g_active_streams` if needed)
  - **Modify**: `generate_kernel_id()` (L183-185) + `next_kernel_id` (L178) — KEEP (used by `cudaStreamCreate` L1232 non-bridge path)
  - Remove `PtxEmuDriverShim` creation + `cpptlm_set_driver` call + `StubBridge` auto-attach L282-317 in `initialize_environment()`
  - Remove `g_cpptlm_bridge` async path L706-825 in `cudaLaunchKernel`
  - Remove `cudaMemcpy` bridge guard L890-894
  - Remove `cudaDeviceSynchronize` bridge poll L1105-1160
  - Remove `cudaStreamSynchronize` bridge poll L1283-1335
  - Remove `EMU_COSIM` env var check L309-316
  - Remove `get_max_advance_cycles()` function L197-203 + `PTX_EMU_MAX_ADVANCE_CYCLES` env var L194-203
- Modify `src/ptxsim/instructions/memory.cpp`:
  - Remove `#include "cudart/cpptlm_bridge.h"` L8
  - Remove `g_cpptlm_bridge->global_access()` GLOBAL LD block L35-56
  - Remove `g_cpptlm_bridge->global_access()` GLOBAL ST block L127-148
  - Verify internal latency fallback (L58-78 LD, L150-169 ST) still functions
- Modify `PTX-EMU/CMakeLists.txt`:
  - Remove `add_subdirectory(${CPPTLM_SOURCE_DIR})` (L157-158)
  - Remove CppTLM link block (L165-167): `add_dependencies(cudart cpptlm_core)` + `target_link_libraries(cudart -Wl,--whole-archive,--exclude-libs=ALL cpptlm_core -Wl,--no-whole-archive)`
  - Remove `include_directories(${CPPTLM_SOURCE_DIR}/include)` (L168)
  - Remove `target_compile_definitions(cudart PRIVATE BUILD_LIB_CPPTLM_CUDART)` (L169)
- Modify `PTX-EMU/src/CMakeLists.txt`:
  - Remove `cudart/cpptlm_bridge/*.cpp` from GLOB (L45)

**Header file PARTIAL REMOVAL**:
- Modify `PTX-EMU/include/cudart/cpptlm_bridge.h`:
  - **DELETE**: `struct PtxEmuDriverApi` (L190-206) — reverse ABI parameter struct
  - **DELETE**: `extern "C" PTXEMU_BRIDGE_API void cpptlm_set_driver(...)` (L211) — reverse ABI entry
  - **DELETE**: `PTXEMU_BRIDGE_API` macro (L4-10) — only used by cpptlm_attach_bridge/detach_bridge (which are also deleted via PtxEmuDriverShim.cpp deletion)
  - **DELETE**: `extern CppTLMBridge* g_cpptlm_bridge` (L154) — only consumer is bridge path
  - **DELETE**: `extern "C" PTXEMU_BRIDGE_API void cpptlm_attach_bridge(...)` (L162) — only consumer is StubBridge
  - **DELETE**: `extern "C" PTXEMU_BRIDGE_API void cpptlm_detach_bridge()` (L169) — only consumer is StubBridge
  - **KEEP**: `class CppTLMBridge` base (L77-140) — vendored by CppTLM, may be referenced by future tests or as a documentation anchor. Actually since `g_cpptlm_bridge` and consumers are all deleted, `CppTLMBridge` is also unused in PTX-EMU. DELETE entire header file `cpptlm_bridge.h` from PTX-EMU.
  - **KEEP**: `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` (L223-224) — ABI guard, useful even without bridge
  - **KEEP**: G-D4 12-endpoint `static_assert`s (L236-290) — ABI guards for IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming

  Wait, after deleting `g_cpptlm_bridge` + `CppTLMBridge` + the 3 ABI entries, the entire `cpptlm_bridge.h` becomes empty (the macro + class + 4 symbols = nothing left). The only things worth keeping (`cudaStream_t` static_assert + G-D4 static_asserts) should move to separate files.

  REVISED: Delete `PTX-EMU/include/cudart/cpptlm_bridge.h` entirely. Move the 17 static_asserts (1 cudaStream_t width + 6 PipelineId + 6 TcPrecision + 4 is_same signature = 17) to a new `include/cudart/abi_guards.h` file, with required includes (`cudart/cudart_intrinsics.h` for `cudaStream_t` + `ptxsim/{scoreboard,pipeline,tensor_core}_interface.h`). Do NOT copy CppTLM's `abi_guards.h` `sizeof(PtxEmuDriverApi) == 64` lock (that type is being deleted).

**Test file REMOVALS** (per Metis + Oracle verification, 8 files):
- Delete `tests/unit/cudart/test_stream_sync_loop.cpp` (uses `extern CppTLMBridge* g_cpptlm_bridge`)
- Delete `tests/integration/cudart/test_abi_stability.cpp` (uses `PtxEmuDriverApi`)
- Delete `tests/integration/test_phase0_byte_identical_gates.cpp` **Gate 4** only (L204-209 checks `g_cpptlm_bridge == nullptr`), AND remove L26 `#include "cudart/cpptlm_bridge.h"` + update gate-list comment L5-10
- Delete 5 of 8 `tests/unit/cpptlm/*` files (bridge-specific): `test_cosim_smoke.cpp`, `test_cpptlm_attach_bridge.cpp`, `test_cpptlm_bridge.cpp`, `test_bridge_submit_error.cpp`, `test_kernel_id_uniqueness.cpp`
- **KEEP** 3 of 8 `tests/unit/cpptlm/*` files (ADR-0020 injection-point + qualifier coverage, NOT bridge): `test_injection_interfaces.cpp`, `test_smcontext_injection.cpp`, `test_is_global_space_walk.cpp`
- Delete 3 of 7 `tests/integration/cpptlm/*` files (bridge-specific): `test_async_launchkernel.cpp`, `test_ld_st_bridge.cpp`, `test_singleton_guard.cpp`
- **KEEP** 4 of 7 `tests/integration/cpptlm/*` files (no bridge dependency): `test_libptxemu_abi_baseline.cpp`, `test_mock_injection_fast_path.cpp`, `test_mock_injection_slow_path.cpp`, `test_scoreboard_allocation.cpp`
- Delete all 3 `tests/e2e/cosim/*.cu` files (bridge co-sim)
- KEEP `tests/integration/test_phase0_byte_identical_gates.cpp` Gate 1/2/3/5
- **ALSO delete test CMakeLists registrations** (Oracle CAUTION `add_catch_test` entries must be removed or CMake configure FAILS):
  - `tests/unit/CMakeLists.txt:767,774,786,839` (`unit_cpptlm_bridge`, `unit_cpptlm_attach_bridge`, `unit_cpptlm_cosim_smoke`, `unit_bridge_submit_error`) — NOTE keep 780/793 (`unit_cpptlm_injection_interfaces`, `unit_smcontext_injection`)
  - `tests/integration/CMakeLists.txt:573,578,583` (`integration_cpptlm_singleton_guard`, `integration_cpptlm_async_launchkernel`, `integration_cpptlm_ld_st_bridge`)
  - `tests/e2e/CMakeLists.txt:61,69,75` (3 cosim tests)
- **ALSO delete** `scripts/regression-cosim.sh` (Oracle CAUTION: script depends on `EMU_COSIM` env var + `e2e_cosim_vector_add` test, both deleted → script will FAIL after Change 2)

**Documentation REMOVALS**:
- Delete or update `PTX-EMU/include/cudart/AGENTS.md` to remove `cpptlm_bridge.h` sections
- Update `PTX-EMU/src/cudart/AGENTS.md` to remove `g_cpptlm_bridge`, `cpptlm_set_driver` weak symbol sections
- Update `PTX-EMU/AGENTS.md` to remove CppTLM coupling references

**EXPLICITLY OUT OF SCOPE** (deferred to future changes):
- Reversal direction (no `PtxEmuSubmodule`, no `dlopen libptxemu_device.so` from PTX-EMU side)
- Deletion of `src/cudart/cpptlm_module.cpp` (source of `ptxemu_image_*` ABI in `libptxemu_device.so`)
- Bumping `CPPTLMBRIDGE_VERSION` or `CPPTLM_MODULE_VERSION`
- Removal of `libptxemu_device.so` build target (still needed for future reversal)
- ADR-0029 amendment (Phase 0 byte-identical gate remains HARD)
- Any `extern/PTX-EMU` submodule references (not used by current build)

## Capabilities

### New Capabilities

- `cudart-sync-only-runtime`: PTX-EMU's `libcudart.so` provides a pure synchronous CUDA runtime shim with zero CppTLM coupling. All async paths (cudaLaunchKernel, cudaMemcpy, cudaDeviceSynchronize, cudaStreamSynchronize) use only PTX-EMU's internal completion model (`g_gpu_context->wait_for_completion()`).

### Modified Capabilities

- `cpptlm-d1-full`: REMOVE the `cpptlm-bridge-interface` requirement (the CppTLMBridge contract) since `libcudart.so` no longer provides the bridge. KEEP documentation references for cross-reference (CppTLM-side vendor copy remains). The `cudart-async-launchkernel` requirement MUST be modified to remove the `g_cpptlm_bridge != nullptr` branch.

## Impact

**Source code**: ~300 lines removed across 4 files (cudart_sim.cpp, memory.cpp, CMakeLists.txt, src/CMakeLists.txt) + 3 files deleted (PtxEmuDriverShim.h/cpp, stub_bridge.h) + 1 header deleted (cpptlm_bridge.h).

**Build artifact**: `libcudart.so.12.0` symbol count drops further (after Change 1's `--exclude-libs`): ~3153 (Change 1 baseline) → ~3138 (Change 2 result, removing cpptlm_* symbols + g_cpptlm_bridge + ptxemu_*_override + PtxEmuDriverShim methods). Symbol reduction = ~15 symbols.

**Test impact**: 8 test files deleted (5 unit/cpptlm + 3 integration/cpptlm + 1 unit/cudart + 1 integration/cudart + 3 e2e/cosim + Gate 4 test case = 14 files/parts, but 7 test files kept from cpptlm/ dirs). Regression script adjusted to exclude deleted cosim e2e labels + delete `regression-cosim.sh`.

**Documentation**: 3 AGENTS.md files updated. No ADR amendment required.

**ABI break**: For external consumers linking against `libcudart.so`, the following symbols disappear:
- `cpptlm_set_driver` (was weak no-op)
- `cpptlm_attach_bridge` (was `PTXEMU_BRIDGE_API`)
- `cpptlm_detach_bridge` (was `PTXEMU_BRIDGE_API`)
- `g_cpptlm_bridge` (was global pointer)
- `ptxemu_is_bridge_user_override` / `ptxemu_set_bridge_user_override` (was `T` in PtxEmuDriverShim.cpp:16-17)
- `PtxEmuDriverShim` class methods (9 in `PtxEmuDriverShim.cpp`)
- `StubBridge` class methods
- All `shim_*` vtable functions (8 `static` functions — never entered dynamic symbol table, but removed anyway)

This is **BREAKING** for any external consumer using these symbols. Mitigation: documented in commit message + AGENTS.md; consumers must migrate to `libptxemu_device.so` ABI (separate future change).

**Risk**: MEDIUM. The change touches 8 source files + 14 test files/parts + 1 script. Phase 0 byte-identical gate baseline must be regenerated (different from Change 1 baseline regeneration — ~15 symbols removed, not 131).

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性 (N/A)
- No function migration. Pure deletion.

### 状态修改
- [x] State modification: `g_active_streams` (L180), `next_kernel_id` (L178), `generate_kernel_id()` (L183-185). 
- [x] `generate_kernel_id()` + `next_kernel_id` MUST be PRESERVED (used by `cudaStreamCreate:1232` non-bridge path)
- [x] `g_active_streams` MUST be PRESERVED (used by `cudaStreamCreate:1233` insert + `cudaStreamDestroy:1258` erase)
- [x] `g_pending_kernels` + `g_pending_kernels_mutex` + `PendingKernel` struct MUST be DELETED (only used by bridge path)

### 多 Phase 推进
- [x] Multi-phase (4 phases per `ptx-lessons-learned` §3 + Metis A5):
  - **Phase 1**: Delete test files (lowest blast radius, easy to revert) — verify test suite builds
  - **Phase 2a**: Modify `cudart_sim.cpp` (cudart library, medium blast radius) — verify Gate 1 still passes
  - **Phase 2b**: Modify `memory.cpp` (ptxsim library, low blast radius) — verify full ptxsim regression
  - **Phase 3**: Modify `CMakeLists.txt` + delete `PtxEmuDriverShim.{h,cpp}` + `stub_bridge.h` + `cpptlm_bridge.h` — verify full regression
- [x] Baseline worktree plan: `git worktree add .worktrees/baseline-pre-cleanup <this-commit-hash>` before Phase 1
- [x] Failure strategy: any regression → revert that Phase, do not mix with later Phases

### 文档同步
- [x] `PTX-EMU/AGENTS.md`: remove CppTLM coupling references
- [x] `PTX-EMU/src/cudart/AGENTS.md`: remove `g_cpptlm_bridge`, `cpptlm_set_driver` sections
- [x] `PTX-EMU/include/cudart/AGENTS.md`: remove `cpptlm_bridge.h` sections
- [x] No ADR amendment needed (Gate 1 contract preserved; new baseline regeneration documented per ADR-0029 §D7 process)

## Reference

- **Oracle investigation** (2026-08-21): Section A REMOVE table (PTX-EMU side) — 19 items
- **Metis independent review** (2026-08-21): corrects Oracle's line ranges + surfaces 6 hidden consumer sites (`generate_kernel_id`, `g_active_streams`, `g_pending_kernels_mutex`, `test_stream_sync_loop.cpp`, `test_abi_stability.cpp`, Gate 4) — all addressed in this proposal
- **ADR-0029 §D5**: `REMOVE_ITEM` precedent for `cpptlm_module.cpp`
- **ADR-0029 §D7**: byte-identical gate baseline regeneration procedure
- **PTX-EMU/CMakeLists.txt:147-171**: CppTLM link block (entire section removed)
- **PTX-EMU/src/cudart/cudart_sim.cpp:660-1335**: CUDA runtime entry points (bridge code scattered)
- **PTX-EMU/src/ptxsim/instructions/memory.cpp:8, 35-56, 127-148**: GLOBAL LD/ST bridge path (fallback at L58-78, L150-169)
- **Verification command**: `./scripts/regression.sh` + `ctest --test-dir build -R integration_phase0_byte_identical_gates --output-on-failure`
- **Companion change**: `fix-phase0-gate1-dgpu-bar-leak` (must be merged BEFORE this change, since this change depends on the new baseline)