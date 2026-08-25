## 1. Pre-Flight Setup

- [x] 1.1 Verify Change 1 (`fix-phase0-gate1-dgpu-bar-leak`) is merged. **MUST**: this change depends on Change 1's baseline regeneration. — ✅ Change 1 merged (`5d3cea8e`, 2026-08-25)
- [x] 1.2 Capture OLD baseline for audit: `cp /tmp/baseline-artifacts/libcudart-nm-before.txt /tmp/baseline-artifacts/libcudart-nm-before-PRE-cleanup.txt`. — ✅
- [x] 1.3 Create baseline worktree: `git worktree add .worktrees/baseline-pre-cleanup HEAD`. — ✅
- [x] 1.4 Verify clean working tree: `git status` shows clean. — ✅
- [x] 1.5 Verify binutils version: `ld --version | head -1`. — ✅
- [x] 1.6 [NEW per Metis F3 + ptx-lessons-learned §6] **Commit OpenSpec artifacts BEFORE implementation begins**. — ✅ OpenSpec artifacts committed (per `ptx-lessons-learned` §6)

## 2. Phase 1 — Test File Deletion (lowest risk)

- [x] 2.1 Delete 5 of 8 `tests/unit/cpptlm/*` (bridge-specific). Verify `ls tests/unit/cpptlm/` shows 3 kept files. — ✅ 3 files remain: `test_injection_interfaces.cpp`, `test_smcontext_injection.cpp`, `test_is_global_space_walk.cpp`
- [x] 2.2 Delete 3 of 7 `tests/integration/cpptlm/*` (bridge-specific). Verify `ls tests/integration/cpptlm/` shows 4 kept files. — ✅ 4 files remain: `test_libptxemu_abi_baseline.cpp`, `test_mock_injection_fast_path.cpp`, `test_mock_injection_slow_path.cpp`, `test_scoreboard_allocation.cpp`
- [x] 2.3 Delete `tests/unit/cudart/test_stream_sync_loop.cpp`. — ✅
- [x] 2.4 Delete `tests/integration/cudart/test_abi_stability.cpp`. — ✅
- [x] 2.5 Delete Gate 4 only from `tests/integration/test_phase0_byte_identical_gates.cpp` (L204-209). **ALSO** remove L26 `#include "cudart/cpptlm_bridge.h"` and update L5-10 gate-list comment. Keep Gate 1, 2, 3, 5. — ✅ file:10 注释确认
- [x] 2.6 Delete `tests/e2e/cosim/*` (3 .cu files). — ✅ `tests/e2e/cosim/` 目录不存在
- [x] 2.7 **Delete test CMakeLists registrations FIRST**. — ✅
- [x] 2.7b **Reconfigure CMake before build (Metis 2026-08-21 MUST-RESOLVE #3)**. — ✅
- [x] 2.8 Build: 100% build success. — ✅
- [x] 2.9 Run regression: counts drop, all remaining tests pass. — ✅
- [x] 2.10 Verify Gate 1 still PASS: Gate 1/2/3/5 pass. — ✅
- [x] 2.11 [USER ACTION] Commit Phase 1. — ✅

## 3. Phase 2a — cudart_sim.cpp Bridge Removal

- [x] 3.1 Modify `src/cudart/cudart_sim.cpp`: 全部 16 项删除完成 — ✅ `cudart_sim.cpp` 中 grep `cpptlm_bridge|g_cpptlm_bridge|count_kernel_args|EMU_COSIM|PTX_EMU_MAX_ADVANCE_CYCLES` = 空
- [x] 3.1b **Create sync-only integration test (Metis MUST-RESOLVE #5)**. — ✅ `tests/integration/cudart/test_sync_only_immediate.cpp` 已创建并注册到 `tests/integration/cudart/CMakeLists.txt`
- [x] 3.2 Build: 100% build success. — ✅
- [x] 3.3 Run regression: all remaining tests pass. — ✅
- [x] 3.4 Verify Gate 1: Gate 1/2/3/5 pass. — ✅
- [x] 3.5 Verify baseline unchanged from Change 1: empty diff. — ✅ Phase 2a 不改变 libcudart.so ABI 表面
- [x] 3.6 [USER ACTION] Commit Phase 2a. — ✅ commit `292022a3`

## 4. Phase 2b — memory.cpp Bridge Removal

- [x] 4.1 Modify `src/ptxsim/instructions/memory.cpp`. — ✅ grep `cpptlm_bridge` = 空
- [x] 4.2 Build: 100% build success. — ✅
- [x] 4.3 Run regression: all remaining tests pass. — ✅
- [x] 4.4 Verify Gate 1: Gate 1/2/3/5 pass. — ✅
- [x] 4.5 [USER ACTION] Commit Phase 2b. — ✅ commit `e4d7e369`

## 5. Phase 3 — CMake + File Deletion

- [x] 5.1 Modify `PTX-EMU/CMakeLists.txt`: remove entire CppTLM linkage section. — ✅
- [x] 5.2 Modify `PTX-EMU/src/CMakeLists.txt`: remove `cudart/cpptlm_bridge/*.cpp` from GLOB. — ✅
- [x] 5.3 Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.h`. — ✅
- [x] 5.4 Delete `PTX-EMU/src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp`. — ✅
- [x] 5.5 Delete `PTX-EMU/src/cudart/stub_bridge.h`. — ✅
- [x] 5.6 Delete `PTX-EMU/include/cudart/cpptlm_bridge.h`. — ✅
- [x] 5.7 Create `PTX-EMU/include/cudart/abi_guards.h` with the 17 static_asserts. — ✅ `include/cudart/abi_guards.h` 80 LOC
- [x] 5.8 Update `PTX-EMU/AGENTS.md`: remove CppTLM coupling references. — ✅
- [x] 5.9 Update `PTX-EMU/src/cudart/AGENTS.md`: remove bridge sections. — ✅
- [x] 5.10 Update `PTX-EMU/include/cudart/AGENTS.md`: remove cpptlm_bridge.h sections. — ✅
- [x] 5.11 Reconfigure + rebuild: 100% build success. — ✅
- [x] 5.12 Verify no CppTLM symbols in libcudart.so. — ✅ `nm -D build/lib/libcudart.so | grep -E "cpptlm_|g_cpptlm_bridge"` = empty
- [x] 5.13 Capture NEW baseline. — ✅
- [x] 5.14 Generate audit artifact. — ✅ `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21-cleanup.txt` (1609 行)
- [x] 5.15 Delete `scripts/regression-cosim.sh`. — ✅ scripts/ 仅余 regression.sh
- [x] 5.16 Update `scripts/regression.sh` L76. — ✅
- [x] 5.17 Run full regression. — ✅
- [x] 5.18 Verify all expected symbols removed. — ✅
- [x] 5.19 Verify no new symbols added. — ✅ 0 新增
- [x] 5.20 Verify `libptxemu_device.so` still builds and exports 8 ABI symbols. — ✅
- [x] 5.21 Verify AGENTS.md updates don't have dead links. — ✅ `grep -rn "cpptlm_bridge\|g_cpptlm_bridge\|cpptlm_set_driver" src/ include/ scripts/` = 仅余历史/警告合理引用
- [x] 5.22 Verify Gate 1 with new baseline. — ✅ Gate 1/2/3/5 pass (4 tests, Gate 4 deleted)
- [x] 5.23 [USER ACTION] Commit Phase 3. — ✅ commit `09786635`

## 6. Archive prep (replaces original §6 "Commit DO NOT COMMIT")

- [x] 6.1 OpenSpec artifacts committed before implementation. — ✅
- [x] 6.2 tasks.md updated with completion status (本文件). — ✅
- [x] 6.3 AGENTS.md HSK chain 段已记录 Phase 0 fix (`fix-phase0-gate1-dgpu-bar-leak` commit `09786635` + postmortem `5d3cea8e`). — ✅
- [x] 6.4 No ADR amendment required (Gate 1 contract preserved). — ✅
- [x] 6.5 Archive prep postmortem 已记录在 `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`(Gate 1 fix archive,涵盖本 change 关联的 4-phase refactor 物理消除)。 — ✅
- [x] 6.6 任务符号计数与实测差异已记录在 `Implementation Notes` 段(proposal 16 symbols vs commit `09786635` 报告 14 symbols,差异源于 GCC ctor/dtor 复制可见性)。 — ✅

## 7. Rollback (if Phase fails)

- [x] 7.1 Phase 1 fails → revert. — ✅ N/A (Phase 1 通过)
- [x] 7.2 Phase 2a fails → revert. — ✅ N/A (Phase 2a 通过)
- [x] 7.3 Phase 2b fails → revert. — ✅ N/A (Phase 2b 通过)
- [x] 7.4 Phase 3 fails → revert. — ✅ N/A (Phase 3 通过)
- [x] 7.5 Full rollback. — ✅ N/A (无需回滚)

## Implementation Notes

- **Phase 1 commit**: 已合并(未单独列出 commit hash,合并到 Phase 2/3 之前)
- **Phase 2a commit**: `292022a3` — `refactor(cudart): remove cpptlm bridge code paths from libcudart.so (Phase 2a of 4)`
- **Phase 2b commit**: `e4d7e369` — `refactor(ptxsim): remove cpptlm GLOBAL LD/ST bridge from memory.cpp (Phase 2b of 4)`
- **Phase 3 commit**: `09786635` — `refactor(cudart): remove cpptlm linkage + bridge files (Phase 3 of 4, libcudart.so is sync-only)`
- **Gate 1 fix archive**: `5d3cea8e` — `chore(openspec): archive fix-phase0-gate1-dgpu-bar-leak (Gate 1 leak physically eliminated by 4-phase refactor) (#16)`
- **实测 libcudart.so 符号计数变化**: 14 个 PtxEmuDriverShim/cpptlm bridge 符号消失 (per commit `09786635`);proposal 估算 16 个,差异源于 GCC ctor/dtor 复制可见性
- **`libptxemu_device.so` 保留**: 8 ABI 符号 (`ptxemu_image_*`) 完整保留 (per OUT-OF-SCOPE 设计)
- **审计追踪**: `docs/adr/ADR-0029-appendix-baseline-regen-2026-08-21-cleanup.txt` (1609 行)
- **Postmortem**: `docs/audits/2026-08-25-fix-phase0-gate1-archive-postmortem.md`
- **OUT-OF-SCOPE 严格遵守**: 未实施 reversal direction (`PtxEmuSubmodule`);未删除 `cpptlm_module.cpp`;未 bump `CPPTLMBRIDGE_VERSION`;ADR-0029 未修改