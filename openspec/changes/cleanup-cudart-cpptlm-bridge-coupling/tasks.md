## 1. Pre-Flight Setup

- [ ] 1.1 Verify Change 1 (`fix-phase0-gate1-dgpu-bar-leak`) is merged. **MUST**: this change depends on Change 1's baseline regeneration.
- [ ] 1.2 Capture OLD baseline for audit: `cp /tmp/baseline-artifacts/libcudart-nm-before.txt /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt`. **MUST**: preserved for diff audit in Phase 3.
- [ ] 1.3 Create baseline worktree: `git worktree add .worktrees/baseline-pre-cleanup HEAD`. NOTE: provides a clean revert target per `ptx-lessons-learned` §3.
- [ ] 1.4 Verify clean working tree: `git status` shows clean. **MUST**.
- [ ] 1.5 Verify binutils version: `ld --version | head -1`. (Already verified in Change 1.)
- [ ] 1.6 [NEW per Metis F3 + ptx-lessons-learned §6] **Commit OpenSpec artifacts BEFORE implementation begins**: `git status openspec/changes/` MUST show both `cleanup-cudart-cpptlm-bridge-coupling/` and `fix-phase0-gate1-dgpu-bar-leak/` as tracked (not `??`). If untracked, run `git add openspec/changes/cleanup-cudart-cpptlm-bridge-coupling/ openspec/changes/fix-phase0-gate1-dgpu-bar-leak/ && git commit -m "docs(openspec): commit cleanup-cudart-cpptlm-bridge-coupling + fix-phase0-gate1-dgpu-bar-leak artifacts"`. **MUST** (per `ptx-lessons-learned` §6 "OpenSpec artifacts 提交遗漏"): untracked artifacts at implementation start breaks audit chain — any implementation commit referencing these change IDs would not have its plan-of-record recoverable from git history.

## 2. Phase 1 — Test File Deletion (lowest risk)

- [ ] 2.1 Delete 5 of 8 `tests/unit/cpptlm/*` (bridge-specific: `test_cosim_smoke.cpp`, `test_cpptlm_attach_bridge.cpp`, `test_cpptlm_bridge.cpp`, `test_bridge_submit_error.cpp`, `test_kernel_id_uniqueness.cpp`). **KEEP** 3 of 8 (`test_injection_interfaces.cpp`, `test_smcontext_injection.cpp`, `test_is_global_space_walk.cpp` — ADR-0020 injection-point + qualifier coverage, NOT bridge). Verify: `ls tests/unit/cpptlm/` shows 3 kept files.
- [ ] 2.2 Delete 3 of 7 `tests/integration/cpptlm/*` (bridge-specific: `test_async_launchkernel.cpp`, `test_ld_st_bridge.cpp`, `test_singleton_guard.cpp`). **KEEP** 4 of 7 (`test_libptxemu_abi_baseline.cpp`, `test_mock_injection_fast_path.cpp`, `test_mock_injection_slow_path.cpp`, `test_scoreboard_allocation.cpp` — no bridge dependency). Verify: `ls tests/integration/cpptlm/` shows 4 kept files.
- [ ] 2.3 Delete `tests/unit/cudart/test_stream_sync_loop.cpp`. Verify with `ls tests/unit/cudart/`.
- [ ] 2.4 Delete `tests/integration/cudart/test_abi_stability.cpp`. Verify.
- [ ] 2.5 Delete Gate 4 only from `tests/integration/test_phase0_byte_identical_gates.cpp` (L204-209). **ALSO** remove L26 `#include "cudart/cpptlm_bridge.h"` (will be deleted in Phase 3, causing compile error) and update L5-10 gate-list comment to remove Gate 4 reference. Keep Gate 1, 2, 3, 5.
- [ ] 2.6 Delete `tests/e2e/cosim/*` (3 .cu files): `test_cosim_vector_add.cu`, `test_cosim_infinite_loop_ceiling.cu`, `test_cosim_multi_kernel_drain.cu`.
- [ ] 2.7 **Delete test CMakeLists registrations FIRST (before build) per Oracle CRITICAL Round-2 finding**: `tests/unit/CMakeLists.txt:767,774,786,838` (4 entries: `unit_cpptlm_bridge`, `unit_cpptlm_attach_bridge`, `unit_cpptlm_cosim_smoke`, `unit_bridge_submit_error`) + `tests/unit/CMakeLists.txt:572-577` (1 entry: `unit_stream_sync_loop` block + comment lines — paired with task 2.3 deletion; include comment lines to avoid orphan comment) + `tests/integration/CMakeLists.txt:573,578,583` (3 entries: `integration_cpptlm_singleton_guard`, `integration_cpptlm_async_launchkernel`, `integration_cpptlm_ld_st_bridge`) + `tests/integration/cudart/CMakeLists.txt:13-16` (1 entry: `integration_abi_stability` block — paired with task 2.4 deletion; CORRECTED per Oracle HIGH Round-2 from previous wrong "15-18") + `tests/e2e/CMakeLists.txt:61,69,75` (3 entries: 3 cosim tests). **MUST**: without these deletions, CMake configure will FAIL with "No rule to make target" or "Cannot find source file" (per Oracle Round-2 CRITICAL: order matters — files 2.1-2.6 deleted first leave dangling CMake refs; must delete CMake refs BEFORE next build). **NOTE**: KEEP `tests/unit/CMakeLists.txt:872` (`unit_cpptlm_module` → `cudart/test_cpptlm_module.cpp`) and `tests/integration/CMakeLists.txt:618` (`integration_cpptlm_module_dlopen` → `test_cpptlm_module_dlopen.cpp`) and `tests/integration/CMakeLists.txt:629` (`integration_cpptlm_module_inflight` → `test_cpptlm_module_inflight.cpp`) — these are `libptxemu_device.so` ABI tests, NOT bridge-specific, MUST be preserved for the future reversal direction.
- [ ] 2.8 Build: `cmake --build build -j$(nproc)`. Expected: 100% build success. Any compile error indicates a test file references removed symbols.
- [ ] 2.9 Run regression: `ctest --test-dir build -L unit -L integration -L e2e`. Expected: counts drop, all remaining tests pass.
- [ ] 2.10 Verify Gate 1 still PASS: `ctest --test-dir build -R integration_phase0_byte_identical_gates`. Expected: Gate 1/2/3/5 pass (Gate 4 deleted, so test count is now 4 instead of 5).
- [ ] 2.11 [USER ACTION] Commit Phase 1: `git add -A && git commit -m "chore(tests): delete bridge-specific test files (Phase 1 of 4)"`.

## 3. Phase 2a — cudart_sim.cpp Bridge Removal (medium risk, cudart library)

**Rationale** (per Metis A5): Phase 2a + Phase 2b split by library boundary (cudart vs ptxsim) for independent rollback granularity.

- [ ] 3.1 Modify `src/cudart/cudart_sim.cpp`:
  - Remove L102-103 `extern bool ptxemu_is_bridge_user_override();` declaration (Oracle MEDIUM #5 — only call site L310 is deleted, leaving declaration dangling)
  - Remove L105-107 includes (cpptlm_bridge.h, PtxEmuDriverShim.h, stub_bridge.h)
  - Remove L118 `g_ptx_emu_driver_shim` definition
  - Remove L121-158 (8 shim_* vtable functions + weak cpptlm_set_driver)
  - Remove L167-176 PendingKernel struct
  - Remove L179 g_pending_kernels declaration
  - Remove L181 g_pending_kernels_mutex declaration
  - Remove L187-192 `count_kernel_args()` function (Oracle MEDIUM #4 — only call site L731 is inside deleted bridge block L706-825; becomes dead code after Phase 2a)
  - Remove L194-203 PTX_EMU_MAX_ADVANCE_CYCLES + get_max_advance_cycles
  - Remove L282-317 PtxEmuDriverShim creation + cpptlm_set_driver call + StubBridge auto-attach in initialize_environment()
  - Remove L309-316 EMU_COSIM env var check
  - Modify `cudaLaunchKernel` (L706-825): remove entire `if (g_cpptlm_bridge) { ... }` block (also deletes the last `count_kernel_args` call site)
  - Modify `cudaMemcpy` (L890-894): remove bridge guard
  - Modify `cudaDeviceSynchronize` (L1105-1160): remove entire `if (g_cpptlm_bridge) { ... }` block including while loop
  - Modify `cudaStreamSynchronize` (L1283-1335): remove entire `if (g_cpptlm_bridge) { ... }` block including while loop (after removal: `cudaStreamSynchronize` returns immediately because sync-mode `cudaLaunchKernel` already completed via `wait_for_completion()`)
  - Modify `cudaStreamDestroy` (L1240-1260): remove `std::lock_guard<std::mutex> lock(g_pending_kernels_mutex)` (no longer needed; `g_active_streams` was already insert-unlocked / erase-locked-asymmetric — both ops become unlocked after `g_pending_kernels_mutex` deletion)
- [ ] 3.2 Build: `cmake --build build -j$(nproc)`. Expected: 100% build success.
- [ ] 3.3 Run regression: `ctest --test-dir build -L unit -L integration -L e2e`. Expected: all remaining tests pass.
- [ ] 3.4 Verify Gate 1: `ctest --test-dir build -R integration_phase0_byte_identical_gates`. Expected: Gate 1/2/3/5 pass.
- [ ] 3.5 Verify baseline is unchanged from Change 1: `diff /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt /tmp/baseline-artifacts/libcudart-nm-before.txt`. Expected: empty diff (Phase 2a doesn't change libcudart.so ABI surface — same symbols, just less code).
- [ ] 3.6 [USER ACTION] Commit Phase 2a: `git add -A && git commit -m "refactor(cudart): remove cpptlm bridge code paths from libcudart.so (Phase 2a of 4)"`.

## 4. Phase 2b — memory.cpp Bridge Removal (low risk, ptxsim library)

- [ ] 4.1 Modify `src/ptxsim/instructions/memory.cpp`:
  - Remove L8 `#include "cudart/cpptlm_bridge.h"`
  - Remove L35-56 GLOBAL LD bridge block (`if (g_cpptlm_bridge && space == MemorySpace::GLOBAL) { ... }`)
  - Remove L127-148 GLOBAL ST bridge block (`if (g_cpptlm_bridge && space == MemorySpace::GLOBAL) { ... }`)
  - Verify fallback paths L58-78 (LD) and L150-169 (ST) still compile and produce identical latency results
- [ ] 4.2 Build: `cmake --build build -j$(nproc)`. Expected: 100% build success.
- [ ] 4.3 Run regression: `ctest --test-dir build -L unit -L integration -L e2e`. Expected: all remaining tests pass (note: memory.cpp is in ptxsim library, NOT in libcudart.so; baseline unchanged).
- [ ] 4.4 Verify Gate 1: `ctest --test-dir build -R integration_phase0_byte_identical_gates`. Expected: Gate 1/2/3/5 pass.
- [ ] 4.5 [USER ACTION] Commit Phase 2b: `git add -A && git commit -m "refactor(ptxsim): remove cpptlm GLOBAL LD/ST bridge from memory.cpp (Phase 2b of 4)"`.

## 5. Phase 3 — CMake + File Deletion (highest risk)

- [ ] 5.1 Modify `PTX-EMU/CMakeLists.txt`:
  - Remove L141-171 entire CppTLM linkage section (CPPTLM_SOURCE_DIR set, add_subdirectory, add_dependencies, target_link_libraries with --exclude-libs, include_directories, target_compile_definitions)
- [ ] 5.2 Modify `PTX-EMU/src/CMakeLists.txt`:
  - Remove L45 `cudart/cpptlm_bridge/*.cpp` from GLOB
- [ ] 5.3 Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.h`.
- [ ] 5.4 Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`.
- [ ] 5.5 Delete `PTX-EMU/src/cudart/stub_bridge.h`.
- [ ] 5.6 Delete `PTX-EMU/include/cudart/cpptlm_bridge.h`.
- [ ] 5.7 Create `PTX-EMU/include/cudart/abi_guards.h` with the 17 static_asserts (1 `cudaStream_t` width + 6 `PipelineId` endpoint + 6 `TcPrecision` endpoint + 4 `is_same` signature). Source: copy from old `cpptlm_bridge.h` L223-290. Required includes: `cudart/cudart_intrinsics.h` (provides `cudaStream_t` at L344) + `ptxsim/{scoreboard,pipeline,tensor_core}_interface.h`. Do NOT copy CppTLM's `sizeof(PtxEmuDriverApi) == 64` lock (type being deleted).
- [ ] 5.8 Update `PTX-EMU/AGENTS.md`: remove CppTLM coupling references.
- [ ] 5.9 Update `PTX-EMU/src/cudart/AGENTS.md`: remove bridge sections.
- [ ] 5.10 Update `PTX-EMU/include/cudart/AGENTS.md`: remove cpptlm_bridge.h sections.
- [ ] 5.11 Reconfigure + rebuild: `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`. Expected: 100% build success.
- [ ] 5.12 Verify no CppTLM symbols in libcudart.so: `nm -D build/lib/libcudart.so | grep -E "cpptlm_|g_cpptlm_bridge"`. Expected: empty.
- [ ] 5.13 Capture NEW baseline: `nm -D --defined-only build/lib/libcudart.so.12.0 | sort > /tmp/baseline-artifacts/libcudart-nm-before.txt`.
- [ ] 5.14 Generate audit artifact: `diff /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt /tmp/baseline-artifacts/libcudart-nm-before.txt > docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21-cleanup.txt`. **MUST**: diff contains ONLY `<` lines (removed symbols from old baseline). **Expected removed set (Oracle MEDIUM Round-2)**: `{cpptlm_attach_bridge, cpptlm_detach_bridge, g_cpptlm_bridge, ptxemu_is_bridge_user_override, ptxemu_set_bridge_user_override}` (5 externs/globals) + `{7 PtxEmuDriverShim methods (advance, inject_scoreboard, inject_pipeline, inject_tensor_core, is_kernel_complete, mark_complete, num_sms)}` + `{ctor, dtor}` ≈ **14 symbols**. **NOTE**: `cpptlm_set_driver` does NOT appear in this diff — already hidden by Change 1's `--exclude-libs=ALL` (its strong definition lives in cpptlm_core archive, never in the PRE-cleanup baseline captured at task 1.2). `g_bridge_user_override` is local-anonymous-namespace at `PtxEmuDriverShim.cpp:12-14`, never exported → excluded from count.
- [ ] 5.15 Delete `scripts/regression-cosim.sh` (depends on `EMU_COSIM` env var + `e2e_cosim_*` tests, both deleted).
- [ ] 5.16 Update `scripts/regression.sh` L76: remove the stale "单元测试 (88)" comment.
- [ ] 5.17 Run full regression: `./scripts/regression.sh`. Expected: all categories green.
- [ ] 5.18 Verify Gate 1 with new baseline: `ctest --test-dir build -R integration_phase0_byte_identical_gates`. Expected: Gate 1/2/3/5 pass (4 tests, Gate 4 deleted).
- [ ] 5.19 [USER ACTION] Commit Phase 3: `git add -A && git commit -m "refactor(cudart): remove cpptlm linkage + bridge files (Phase 3 of 4, libcudart.so is sync-only)"`.

## 6. Verification (Post-Phase 3)

- [ ] 6.1 Verify all ~14 expected symbols removed: `comm -23 /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt /tmp/baseline-artifacts/libcudart-nm-before.txt | wc -l`. Expected: ~14.
- [ ] 6.2 Verify no new symbols added: `comm -13 /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt /tmp/baseline-artifacts/libcudart-nm-before.txt | wc -l`. Expected: 0.
- [ ] 6.3 Verify `libptxemu_device.so` still builds and exports 8 ABI symbols: `nm -D build/lib/libptxemu_device.so | grep ptxemu_ | wc -l`. Expected: 8.
- [ ] 6.4 Verify AGENTS.md updates don't have dead links: `grep -rn "cpptlm_bridge\|g_cpptlm_bridge\|cpptlm_set_driver" src/ include/ scripts/`. Expected: empty (no remaining references in source/script code). NOTE: scanned only `src/ include/ scripts/`, not `docs/` (historical docs may legitimately reference removed mechanisms).

## 7. Commit (DO NOT COMMIT — user instruction)

NOTE: Per task management rules, do NOT run `git commit`. Stop after Phase 3 task 5.17 (which is the user action) and report success.

## 8. Rollback (if Phase fails)

- [ ] 8.1 Phase 1 fails: `git revert <phase-1-commit-hash>`. No impact on other Phases.
- [ ] 8.2 Phase 2a fails: `git revert <phase-2a-commit-hash>`. Reapply Phase 1 commit.
- [ ] 8.3 Phase 2b fails: `git revert <phase-2b-commit-hash>`. Reapply Phase 1 + Phase 2a commits.
- [ ] 8.4 Phase 3 fails: `git revert <phase-3-commit-hash>`. Reapply Phase 1 + Phase 2a + Phase 2b commits.
- [ ] 8.5 Full rollback: `git worktree remove .worktrees/baseline-pre-cleanup && git checkout <pre-change-commit-hash>`. Restores to Change 1 post-state.