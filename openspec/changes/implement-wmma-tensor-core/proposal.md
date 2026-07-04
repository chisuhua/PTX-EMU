# Implement WMMA / Tensor Core — Real Semantics

## Why

`replace-silent-stub-failures` (archived `2026-07-04`) replaced
`WmmaHandler::processWmmaOperation`'s silent no-op behavior with a
`throw UnsupportedInstructionException` so that any code path
hitting a wmma.* instruction fails loudly. That fixed the silent
failure mode but did not actually implement the instruction:
downstream kernels (cutlass-style GEMM, cute_rmsnorm variants that
touch Tensor Core, attention kernels using mma.sync) still cannot
run end-to-end. We now need to provide real WMMA / Tensor Core
semantics so the simulator covers more of the modern PTX ISA and
the new explicit-failure contract can be relaxed back to real
execution.

## What Changes

- Implement real WMMA / Tensor Core instruction semantics in
  `tensor.cpp` (currently throws). Target the PTX 7.x WMMA
  instruction set first (m8n8k4 / m16n16k16 f16/f32 fragments), then
  extend to mma.sync (sm_70+) and tcgen05.mma (sm_100+) if scope
  permits.
- Rename `tensor.cpp` → `wmma.cpp` to match content (current file
  name is misleading per `replace-silent-stub-failures` design.md
  Decision 1).
- Add unit + integration + e2e tests:
  - Unit: per-fragment shape arithmetic (load/store/mma round-trip
    for m8n8k4 and m16n16k16).
  - Integration: small GEMM kernel driven through
    `execute_warp_instruction` and validated lane-by-lane.
  - E2E: cutlass-style matrix-multiply that previously fell into
    the silent-failure path.
- Remove `UnsupportedInstructionException` call from
  `tensor.cpp`/`wmma.cpp` once semantics are real. Keep
  `PTX_ERROR_EMU` reachable for genuinely-unsupported variants
  (e.g. tcgen05 before sm_100 is implemented).
- Update `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS section
  + root `AGENTS.md` 已知限制 table to reflect that WMMA is
  implemented.

## Non-Goals

- Performance parity with real hardware — only functional
  correctness is in scope.
- Real TC/SASS semantics — still PTX-level interpretation.
- WMMA sparse variants (m8n8k32 etc.) unless trivially derivable.
- Migrating cutlass kernels that rely on cooperative-groups
  async-copy beyond what cute_rmsnorm already exercises.

## Goals

1. `wmma.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32` (and
   symmetric variants) executes end-to-end with correct fragment
   math.
2. No `UnsupportedInstructionException` thrown for implemented
   variants; dst registers are populated, not left uninitialized.
3. Test coverage across all three test types (类型一/二/三 per
   `AGENTS.md` 测试分类规范).
4. All existing tests still pass (no regression to the
   non-WMMA paths).

## Risks

- **Fragment layout complexity**: WMMA m16n16k16 uses row-major
  layouts that differ from cutlass's expected ordering; mismatch
  breaks downstream cutlass kernels.
- **Type-system interaction**: F16 fragments need half-precision
  utilities (`include/ptxsim/utils/half_utils.h`) — those already
  exist and are tested (cute_rmsnorm baseline) but interacting
  with the mma path is new.
- **SIMT divergence**: a divergent warp at WMMA is UB on real
  hardware; the simulator must treat the whole warp as a unit or
  raise a deterministic error. Per `ptx-lessons-learned` §1
  (跨模块状态翻译), need to verify all consumers of `state ==
  BAR_SYNC` see the new WMMA-completed state.
- **Existing cute_rmsnorm may exercise wmma?** Verified in
  `replace-silent-stub-failures` Phase 1 — zero `wmma.` / `mma.sync`
  matches in tests/. So implementing WMMA will not silently
  change cute_rmsnorm output, only unlock new kernels.

## Design-Time Checklist (Lessons-Learned)

### 多 Phase 推进
- [ ] Phase 拆分：建议 3 个独立 commit — (1) m8n8k4 f16 + tests
      (2) m16n16k16 f16 + tests (3) rename + AGENTS sync
- [ ] 基线 worktree: 复用 `.worktrees/fix-pre-p0-baseline`
- [ ] 失败处理: 任何已有测试回归 → 立即 revert 该 Phase

### 函数迁移完整性
- `tensor.cpp::WmmaHandler::processWmmaOperation` 当前实现：抛
  异常。本 change 把"抛异常"行替换为真实实现。所有 set_state /
  commit_pc 调用需在 design.md Migration Plan 列出行级 diff。
- AGENTS.md / SPEC.md / X-Macro `ptx_op.def` 中所有
  `WmmaHandler` / `wmma` 引用需 grep 一致。

### 文档同步
- `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS: 移除
  `tensor.cpp (WmmaHandler)` 异常说明
- 根 `AGENTS.md` 已知限制表: WMMA 条目从"抛异常"恢复为"实现"
- `tests/ptx/parser/test_wmma.cpp` 顶部"Known broken"注释:
  修复后可移除（前提是补 test_wmma.ptx fixture）

## Capabilities

### New Capabilities
- `wmma-tensor-core`: Real WMMA / Tensor Core instruction semantics
  for the supported PTX variants.

### Modified Capabilities
- `stub-explicit-failure`: The WMMA-Stub-Throws-Exception MUST
  requirement should be relaxed to "implement WMMA semantics"
  (not throw) for the implemented variants; the exception remains
  for genuinely unsupported future variants (tcgen05 on sm_90).

## Impact

**修改文件**:
- `src/ptxsim/instructions/tensor.cpp` → `wmma.cpp` (rename + real impl)
- `src/ptxsim/instructions/AGENTS.md` (KNOWN STUBS sync)
- `AGENTS.md` (已知限制 sync)
- `tests/unit/ptx/test_wmma_not_implemented.cpp` → repurpose to
  test real semantics, or split into separate test files
- `tests/integration/wmma/` (new directory for指令序列集成测试)
- `tests/e2e/kernel/` (add WMMA-driven e2e tests)
- `openspec/specs/stub-explicit-failure/spec.md` (relax WMMA-MUST)

**新建测试**:
- `tests/unit/ptx/test_wmma_m8n8k4.cpp`
- `tests/unit/ptx/test_wmma_m16n16k16.cpp`
- `tests/integration/wmma/test_wmma_mma_sync.cpp`
- `tests/e2e/kernel/test_wmma_gemm.cu` (or similar)

**影响范围**:
- PTX 解析器 (无变化，ANTLR 已识别 WMMA 语法)
- X-Macro 分发 (无变化，仍 `WmmaHandler::processWmmaOperation`)
- 现有 cute_rmsnorm / cute_hello_* (无变化 — 已验证零 wmma 引用)
- Multi-PTX warning (无变化 — Fix #3 仍适用)