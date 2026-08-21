## Context

After `fix-phase0-gate1-dgpu-bar-leak` (Change 1) merges, `libcudart.so` will still export 16 CppTLM bridge-related symbols (verified against PRE-cleanup baseline: 5 `cpptlm_*`/override externs/globals + `g_cpptlm_bridge` global + 11 `PtxEmuDriverShim` symbols incl. duplicated ctor/dtor C1/C2/D1/D2). The bridge code path in `cudart_sim.cpp` (~130 lines) and `memory.cpp` (~30 lines) is dead code in the default sync mode and is increasingly fragile (7 conditional branches on `g_cpptlm_bridge`).

CppTLM is no longer maintained (HSK-6 deprecation announced 2026-08-18, commit `25e36f60`). The `tests/e2e/cosim/*` path requires CppTLM's `KernelLaunchTLM::tick()` to inject `g_ptx_emu_driver` via the bridge — this is no longer needed for the default sync path which is what the regression test exercises.

The project structure constraint: `libptxemu_device.so` (source: `src/cudart/cpptlm_module.cpp`) is the ABI that future CppTLM-as-consumer will use. Deleting `cpptlm_module.cpp` would break `libptxemu_device.so` building, which is still needed for forward compatibility. So `cpptlm_module.cpp` is preserved in this change; deletion deferred to the reversal-direction future change.

## Goals / Non-Goals

**Goals:**
1. Delete `PtxEmuDriverShim.{h,cpp}`, `stub_bridge.h` entirely
2. Remove all bridge code from `cudart_sim.cpp` (L118, L121-158, L167-176, L179-181 partial, L282-317, L706-825, L890-894, L1105-1160, L1283-1335) while preserving `generate_kernel_id()` + `next_kernel_id` + `g_active_streams` for non-bridge `cudaStreamCreate`/`Destroy`
3. Remove `g_cpptlm_bridge` GLOBAL LD/ST bridge from `memory.cpp` (L8, L35-56, L127-148)
4. Delete `cpptlm_bridge.h` entirely (move 17 static_asserts to `include/cudart/abi_guards.h`)
5. Delete all 14 bridge-specific test files/parts
6. Delete `scripts/regression-cosim.sh` (depends on `EMU_COSIM` + `e2e_cosim_*` tests, both deleted)
7. Remove `BUILD_LIB_CPPTLM_CUDART` macro, `EMU_COSIM` env var, `PTX_EMU_MAX_ADVANCE_CYCLES` env var
8. Update `regression.sh` to not look for deleted test labels
9. Full regression passes after each Phase

**Non-Goals:**
1. Reversing PTX-EMU ↔ CppTLM coupling direction (no `PtxEmuSubmodule`, no dlopen of `libptxemu_device.so`)
2. Deleting `src/cudart/cpptlm_module.cpp` (the `ptxemu_image_*` source)
3. Bumping any version constants (`CPPTLMBRIDGE_VERSION`, `CPPTLM_MODULE_VERSION`)
4. Removing `libptxemu_device.so` build target
5. ADR-0029 amendment (Gate 1 contract preserved; new baseline regeneration documented)
6. CppTLM-side changes
7. Replacing async bridge semantics with new mechanism (sync-only runtime is the new model)

## Decisions

### Decision 1: 4-phase rollout per `ptx-lessons-learned` §3 + Metis A5

**Rationale** (REVISED 2026-08-21): The change touches 7 source files + 19 test files + 1 header + 3 AGENTS.md files. Per `ptx-lessons-learned` §3 ("复杂迁移分 Phase commit"), each phase must end with ctest green and be independently revertible. **Metis second-pass review A5** further recommended splitting Phase 2 into 2a (cudart_sim.cpp — cudart library) and 2b (memory.cpp — ptxsim library) because they target two different libraries; independent rollback granularity is cleaner.

**Phases**:
- **Phase 1 (lowest risk)**: Delete test files only (10 .cpp files + 3 .cu files + 9 add_catch_test registrations including 2 newly-added per Oracle H1.2: `unit_stream_sync_loop` @ tests/unit/CMakeLists.txt:573-577 + `integration_abi_stability` @ tests/integration/cudart/CMakeLists.txt:15-18). Verify test suite builds. Pass: `cmake --build build` succeeds. KEEP `unit_cpptlm_module` (L872) + `integration_cpptlm_module_dlopen` (L618) + `integration_cpptlm_module_inflight` (L629) — libptxemu_device.so ABI tests.
- **Phase 2a (medium risk, cudart library)**: Modify `cudart_sim.cpp` — strip `g_ptx_emu_driver_shim` (L102-103, L118), 8 `shim_*` vtable functions + weak `cpptlm_set_driver` (L121-158), `PendingKernel` struct (L167-176), `g_pending_kernels` (L179), `g_pending_kernels_mutex` (L181), `count_kernel_args` dead code (L187-192), `get_max_advance_cycles` (L194-203), PtxEmuDriverShim creation block (L282-317), `cudaLaunchKernel` bridge block (L706-825), `cudaMemcpy` guard (L890-894), `cudaDeviceSynchronize` bridge block (L1105-1160), `cudaStreamSynchronize` bridge block (L1283-1335), `cudaStreamDestroy` mutex removal (L1257-1258). Pass: ctest unit + integration all green. Gate 1 still passes with existing baseline (ABI surface unchanged).
- **Phase 2b (low risk, ptxsim library)**: Modify `memory.cpp` — remove `#include "cudart/cpptlm_bridge.h"` (L8), remove GLOBAL LD bridge block (L35-56), remove GLOBAL ST bridge block (L127-148). Pass: ctest unit + integration + e2e all green. Fallback paths L58-78 (LD) and L150-169 (ST) verified to compile and produce identical latency results.
- **Phase 3 (highest risk)**: Modify CMakeLists.txt (remove `--whole-archive,--exclude-libs=libcpptlm_core.a cpptlm_core` link block entirely + `add_subdirectory(CppTLM)`) + delete `PtxEmuDriverShim.{h,cpp}` + `stub_bridge.h` + `cpptlm_bridge.h` + `scripts/regression-cosim.sh` + create `include/cudart/abi_guards.h`. Pass: full regression green. Regenerate baseline (NEW symbols removed: 16 total = 5 cpptlm_*/override externs/globals + 11 PtxEmuDriverShim symbols; `g_bridge_user_override` is local-anonymous-namespace per Oracle MEDIUM #3, never exported).

**Alternatives considered**:
- **Single-phase**: REJECTED. Per lessons-learned §3, multi-file changes with potential compile breakages should be phased.
- **3-phase (originally proposed)**: REJECTED after Metis A5. Phase 2 mixing two libraries (cudart + ptxsim) makes per-library regression rollback harder.
- **5-phase (per Oracle's plan)**: REJECTED as overkill. Oracle's 5 phases include the reversal direction which is explicitly OUT.

### Decision 2: Preserve `generate_kernel_id()` + `next_kernel_id` + `g_active_streams`

**Rationale**: Per Metis verification, `cudaStreamCreate` (L1232-1233) uses `generate_kernel_id()` to generate unique stream IDs and inserts into `g_active_streams`. `cudaStreamDestroy` (L1257-1258) erases from `g_active_streams` under `g_pending_kernels_mutex`. If we delete these, `cudaStreamCreate`/`Destroy` break. The CUDA stream API is not bridge-specific — programs using streams should work in sync mode.

**Implementation**:
- KEEP `generate_kernel_id()` + `next_kernel_id` (L178, L183-185) — used by `cudaStreamCreate` non-bridge path
- KEEP `g_active_streams` (L180) — used by `cudaStreamCreate` + `cudaStreamDestroy`
- REMOVE `g_pending_kernels` (L179) + `PendingKernel` struct (L167-176) + `g_pending_kernels_mutex` (L181) — used only by bridge path
- In `cudaStreamDestroy` (L1257), replace `std::lock_guard<std::mutex> lock(g_pending_kernels_mutex)` with simple `g_active_streams.erase(stream_id)` (no mutex needed if single-threaded access pattern; verify)

**Alternatives considered**:
- **Replace `cudaStreamCreate` with returning `nullptr`**: REJECTED. Breaks CUDA programs that use streams (semantically: all streams become default stream = serializes everything).
- **Replace with simple counter**: SIMILAR EFFECT to current behavior, but introduces subtle behavior change. KEEP current implementation to minimize risk.

### Decision 3: Delete `cpptlm_bridge.h` entirely (don't keep partial)

**Rationale**: After removing `g_cpptlm_bridge` + `cpptlm_attach_bridge`/`detach_bridge` + `PtxEmuDriverApi` + `cpptlm_set_driver` + `PTXEMU_BRIDGE_API` macro, only `CppTLMBridge` base class + 17 static_asserts remain. The base class has no PTX-EMU consumers (all its consumers were bridge-related). The static_asserts belong in a focused `abi_guards.h` file.

**Implementation**:
- Delete `PTX-EMU/include/cudart/cpptlm_bridge.h` (294 lines)
- Create `PTX-EMU/include/cudart/abi_guards.h` with the 17 static_asserts (1 `cudaStream_t` width + 6 `PipelineId` endpoint + 6 `TcPrecision` endpoint + 4 `is_same` signature). Required includes: `cudart/cudart_intrinsics.h` (provides `cudaStream_t` at L344) + `ptxsim/{scoreboard,pipeline,tensor_core}_interface.h`. Do NOT copy CppTLM's `sizeof(PtxEmuDriverApi) == 64` lock (type being deleted). Mirror CppTLM's already-existing `include/cudart/abi_guards.h` structure (17 asserts) minus the PtxEmuDriverApi lock.

**Alternatives considered**:
- **Keep `cpptlm_bridge.h` as a "frozen" header**: REJECTED. The file is meant to be the ABI truth source for cross-repo consumption. Once CppTLM is no longer a runtime consumer, the file has no purpose.

### Decision 4: Update `regression.sh` to remove deleted e2e cosim references

**Rationale**: After deleting `tests/e2e/cosim/*` (3 files), the regression script's `e2e_divergence$` exclusion regex doesn't cover `e2e_cosim_*`. After deletion, `ctest -L e2e -E 'e2e_divergence$'` will pass cleanly because the deleted cosim tests no longer exist.

**Implementation**: Update `scripts/regression.sh` L76 to remove the obsolete test count comment "单元测试 (88)", or adjust to reflect the new reduced count. Delete `scripts/regression-cosim.sh` entirely (it depends on `EMU_COSIM` env var + `e2e_cosim_*` tests which are deleted in Phase 1).

### Decision 5: Delete `cpptlm_module.cpp` is **NOT** in this change

**Rationale**: `src/cudart/cpptlm_module.cpp` is the implementation of `ptxemu_image_*` ABI for `libptxemu_device.so` (per ADR-0029 §D5). Deleting it breaks `libptxemu_device.so` building. Since the reversal direction (where `libptxemu_device.so` becomes the CppTLM-side consumer entry point) is OUT of scope, we cannot delete this file — `libptxemu_device.so` is still built even though no PTX-EMU runtime currently consumes it.

**Action**: Add a comment in `src/cudart/cpptlm_module.cpp` (or design.md) noting "this file is preserved for the future reversal direction change; do not delete without coordinating with that change".

## Risks / Trade-offs

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | Phase 2 compile errors from `g_pending_kernels`/`g_pending_kernels_mutex` removal breaking `cudaStreamDestroy` | Medium | Medium | Verification: `grep -n "g_pending_kernels" src/cudart/cudart_sim.cpp` before Phase 2 commit. Replace lock with atomic-free erase if needed. |
| R2 | `cudart_sim.cpp:1283-1335` removal leaves `while (true)` loop without break condition | High | High | Per Metis verification, the `while` loops inside `if (g_cpptlm_bridge)` blocks. Removing the entire `if` block also removes the `while`. Verify by reading full block before edit. |
| R3 | `tests/unit/cudart/test_stream_sync_loop.cpp` deletion breaks ctest label indexing | Low | Low | Pre-Phase-1 verification: `ctest -N -L unit | grep stream_sync` to confirm only 1 test case |
| R4 | Gate 1 baseline changes after Phase 3 (new symbols removed) → Gate 1 fails | High (expected) | None (by design) | Phase 3 includes baseline regeneration step. Verify audit: only `<` lines (removed symbols from old baseline), zero `>` lines (no new additions). Same procedure as Change 1. |
| R5 | `tests/e2e/cosim/*` deletion removes 3 e2e tests → e2e count drops from 21 to 18 | Low (expected) | None | Update `regression.sh` log message if needed. The 21-test count was always misleading (3 of them were bridge-only). |
| R6 | External consumers linking against `libcudart.so` break due to removed ABI symbols | Low | High | Document in commit message + AGENTS.md. Out of scope to migrate consumers (they should switch to `libptxemu_device.so` separately). |
| R7 | `libptxemu_device.so` builds but is never used → wasted CI time | Low | Low | Acceptable. Future reversal direction will use it. |
| R8 | `tests/integration/cpptlm/test_libptxemu_abi_baseline.cpp` (KEEP) references some removed symbols | Medium | Medium | Pre-Phase-1 verification: read entire file. If it references `g_cpptlm_bridge` or `cpptlm_set_driver`, rewrite to use `nm` against `libptxemu_device.so` only. |
| R9 | `tests/integration/test_phase0_byte_identical_gates.cpp` Gate 2-5 still pass after Phase 2 | Low | High | Gate 2 (SONAME): no change. Gate 3 (symlinks): no change. Gate 4 (`g_cpptlm_bridge == nullptr`): MUST be deleted per Phase 1. Gate 5 (`get_gpu_clock_from_context()`): no change. |
| R10 | `cpptlm_bridge.h` deletion breaks transitive includes | Low | Medium | Verification: `grep -rn "include.*cpptlm_bridge.h" src/ include/`. Only `cudart_sim.cpp:105` and `memory.cpp:8` (both being modified in Phase 2). |
| R11 | Removal of `g_pending_kernels` + `g_pending_kernels_mutex` breaks the stream mutex protection around `g_active_streams` | Medium | Low | `cudaStreamDestroy` is typically called from main thread; concurrent stream create/destroy from multiple threads is rare in CUDA programs. If concurrency is needed, replace with `std::atomic<uint64_t>` simple set + open-addressing, or document the single-threaded assumption. |

## Impact Scope

| Component | Impact Type | Specific Change | Phase |
|-----------|-------------|-----------------|-------|
| `PTX-EMU/include/cudart/cpptlm_bridge.h` | Delete (entire) | 294 lines | 3 |
| `PTX-EMU/include/cudart/abi_guards.h` | New file | 17 static_asserts moved from cpptlm_bridge.h | 3 |
| `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` | Delete | 63 lines | 3 |
| `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` | Delete | 121 lines | 3 |
| `PTX-EMU/src/cudart/stub_bridge.h` | Delete | 59 lines | 3 |
| `PTX-EMU/src/cudart/cudart_sim.cpp` | Modify | -130 lines, 4 line additions | 2 |
| `PTX-EMU/src/ptxsim/instructions/memory.cpp` | Modify | -30 lines | 2 |
| `PTX-EMU/CMakeLists.txt` | Modify | -25 lines (L141-171 region) | 3 |
| `PTX-EMU/src/CMakeLists.txt` | Modify | -1 line (L45 GLOB entry) | 3 |
| `tests/unit/cpptlm/*` | Delete (5 kept) | 5 of 8 files deleted (3 kept: `test_injection_interfaces`, `test_smcontext_injection`, `test_is_global_space_walk`) | 1 |
| `tests/integration/cpptlm/*` | Delete (4 kept) | 3 of 7 files deleted (4 kept: `test_libptxemu_abi_baseline`, `test_mock_injection_*`, `test_scoreboard_allocation`) | 1 |
| `tests/unit/cudart/test_stream_sync_loop.cpp` | Delete | 1 file | 1 |
| `tests/integration/cudart/test_abi_stability.cpp` | Delete | 1 file | 1 |
| `tests/integration/test_phase0_byte_identical_gates.cpp` Gate 4 | Delete | L204-209 | 1 |
| `tests/e2e/cosim/*` | Delete | 3 .cu files | 1 |
| `scripts/regression.sh` | Possibly modify | cosim label adjustment | 3 |
| `PTX-EMU/AGENTS.md` | Modify | Remove CppTLM references | 3 |
| `PTX-EMU/src/cudart/AGENTS.md` | Modify | Remove bridge sections | 3 |
| `PTX-EMU/include/cudart/AGENTS.md` | Modify | Remove cpptlm_bridge.h sections | 3 |
| `/tmp/baseline-artifacts/libcudart-nm-before.txt` | Regenerate | After Phase 3 | 3 |
| `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21.txt` | Update (or new file) | nm diff archive | 3 |

## Migration Plan

### Phase 1 (lowest risk): Test deletion
1. `git worktree add .worktrees/baseline-pre-cleanup <this-change-commit-hash>`
2. Delete 5 of 8 `tests/unit/cpptlm/*` (keep `test_injection_interfaces.cpp`, `test_smcontext_injection.cpp`, `test_is_global_space_walk.cpp`)
3. Delete 3 of 7 `tests/integration/cpptlm/*` (keep `test_libptxemu_abi_baseline.cpp`, `test_mock_injection_*`, `test_scoreboard_allocation.cpp`)
4. Delete `tests/unit/cudart/test_stream_sync_loop.cpp`
5. Delete `tests/integration/cudart/test_abi_stability.cpp`
6. Delete Gate 4 in `tests/integration/test_phase0_byte_identical_gates.cpp` (L204-209) + remove L26 `#include "cudart/cpptlm_bridge.h"` + update L5-10 gate-list comment
7. Delete `tests/e2e/cosim/*` (3 .cu files)
8. **Delete test CMakeLists registrations**: `tests/unit/CMakeLists.txt:767,774,786,839` + `tests/integration/CMakeLists.txt:573,578,583` + `tests/e2e/CMakeLists.txt:61,69,75`
9. Verify: `cmake --build build` succeeds (compile errors surface if any test file references removed symbols, or CMakeLists references deleted source files)
10. Verify: `ctest -L unit -L integration -L e2e` — count drops but all remaining tests pass
11. Commit Phase 1 (atomic). Tag: `phase-1-tests-deleted`

### Phase 2 (medium risk): Source code bridge removal
1. Modify `cudart_sim.cpp`:
   - Remove L105-107 (3 includes)
   - Remove L118 (`g_ptx_emu_driver_shim`)
   - Remove L121-158 (8 shim_* + weak cpptlm_set_driver)
   - Remove L167-176 (PendingKernel struct)
   - Remove L179 (g_pending_kernels)
   - Remove L181 (g_pending_kernels_mutex)
   - Remove L194-203 (PTX_EMU_MAX_ADVANCE_CYCLES)
   - Remove L282-317 (PtxEmuDriverShim creation + cpptlm_set_driver + StubBridge auto-attach)
   - Remove L309-316 (EMU_COSIM check)
   - Modify `cudaLaunchKernel` (L706-825): remove `if (g_cpptlm_bridge)` block
   - Modify `cudaMemcpy` (L890-894): remove `if (g_cpptlm_bridge && g_ptx_emu_driver_shim)` block
   - Modify `cudaDeviceSynchronize` (L1105-1160): remove `if (g_cpptlm_bridge)` block including while loop
   - Modify `cudaStreamSynchronize` (L1283-1335): remove `if (g_cpptlm_bridge)` block including while loop
   - Modify `cudaStreamDestroy` (L1240-1260): remove `g_pending_kernels_mutex` lock, keep `g_active_streams.erase(stream_id)`
2. Modify `memory.cpp`:
   - Remove L8 (`#include "cudart/cpptlm_bridge.h"`)
   - Remove L35-56 (GLOBAL LD bridge block)
   - Remove L127-148 (GLOBAL ST bridge block)
   - Verify fallback paths L58-78 (LD) and L150-169 (ST) still compile
3. Verify: `cmake --build build` succeeds
4. Verify: `ctest -L unit -L integration -L e2e` all pass
5. Verify Gate 1: `ctest -R integration_phase0_byte_identical_gates` Gate 1/2/3/5 all pass (Gate 4 already deleted in Phase 1)
6. Commit Phase 2 (atomic). Tag: `phase-2-source-bridge-removed`

### Phase 3 (highest risk): CMake + file deletion
1. Modify `PTX-EMU/CMakeLists.txt`:
   - Remove L141-171 (entire CppTLM linkage section: add_subdirectory, --exclude-libs, include_directories, target_compile_definitions)
2. Modify `PTX-EMU/src/CMakeLists.txt`:
   - Remove L45 (`cudart/cpptlm_bridge/*.cpp` from GLOB)
3. Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.h`
4. Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`
5. Delete `PTX-EMU/src/cudart/stub_bridge.h`
6. Delete `PTX-EMU/include/cudart/cpptlm_bridge.h`
7. Create `PTX-EMU/include/cudart/abi_guards.h` with the 17 static_asserts + required includes (see Decision 3)
8. Delete `scripts/regression-cosim.sh`
9. Update `scripts/regression.sh` L76 (remove "单元测试 (88)" stale comment)
10. Update 3 AGENTS.md files
11. Rebuild: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
12. Verify: full regression (`./scripts/regression.sh`)
13. Regenerate baseline + audit artifact (similar to Change 1 procedure)
14. Commit Phase 3 (atomic). Tag: `phase-3-cleanup-complete`

### Rollback Strategy
- Each Phase is independently revertible via `git revert`
- If Phase 1 fails: revert Phase 1 commit, no impact on other Phases
- If Phase 2 fails: revert Phase 2, reapply Phase 1 (tests are deleted; source code reverted)
- If Phase 3 fails: revert Phase 3, reapply Phases 1+2 (CMake + files restored; source modifications intact)

## Open Questions

None. The change scope is well-defined by Oracle + Metis investigations. The user's "反转内容不在目前计划里" instruction explicitly bounds scope.

## Follow-up (separate changes, NOT this one)

1. **Reversal direction change** (future): `cpp-tlm-consumes-ptxemu-device` — CppTLM links `libptxemu_device.so`, calls `ptxemu_image_*` directly. Oracle's Section B hypothesis B1 (dlopen + 8-symbol ABI reuse). Out of scope for this proposal.
2. **`cpptlm_module.cpp` deletion** (future): part of the reversal direction. Out of scope.
3. **`libptxemu_device.so` removal** (future): if reversal direction takes a different path (e.g., static linking of `cpptlm_module.cpp` into a different library). Out of scope.