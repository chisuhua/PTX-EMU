# Design: Implement WMMA / Tensor Core

## Context

After `replace-silent-stub-failures` (archived `2026-07-04`),
`WmmaHandler::processWmmaOperation` in `src/ptxsim/instructions/tensor.cpp`
throws `UnsupportedInstructionException` instead of silently no-op'ing.
This made WMMA paths loud but did not give them semantics: any
modern kernel that lowers to `wmma.mma.sync.*` or `mma.sync.*` still
fails at runtime. Real WMMA support is needed for:

- cutlass-style GEMM kernels (currently fall to throw)
- cute/cute_rmsnorm extensions that may grow to use Tensor Core
- Attention kernels that increasingly rely on mma.sync
- Coverage parity with the modern PTX ISA (currently ~67% per
  `HEALTH-AUDIT-2026-06-21` §3)

## Goals / Non-Goals

**Goals:**
1. Implement `wmma.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
   and symmetric variants end-to-end (load / mma / store).
2. Optionally extend to `m16n16k16` if Phase 1 is clean.
3. Tests cover all three types: 类型一 (fragment arithmetic unit),
   类型二 (`execute_warp_instruction` integration), 类型三 (cutlass-style
   e2e kernel).
4. Rename `tensor.cpp` → `wmma.cpp` to match content.
5. Keep AGENTS.md / SPEC.md / docs in sync.

**Non-Goals:**
- Performance parity — only functional correctness.
- SASS-level semantics.
- tcgen05.mma (sm_100) — separate change if/when hardware support
  is needed.
- mma.sp / mma.aligned variants with sparse tiles.
- Cooperative-groups async-copy interop with mma.
- Replacing cute_rmsnorm's hand-written FMA path with WMMA.

## Decisions

### Decision 1: Rename `tensor.cpp` → `wmma.cpp` immediately

**Context**: `tensor.cpp` currently defines `WmmaHandler` (the only
real WMMA dispatch target). The name `tensor.cpp` is misleading and
was flagged in `replace-silent-stub-failures` design.md Decision 1
as known tech debt.

**Choice**: Rename file in Phase 1 of this change.

**Rationale**:
- File name ↔ content alignment is a long-standing project hygiene
  rule (cf. `barrier_module.cpp` rename in `integrate-barrier-module-cta-warp`).
- `src/CMakeLists.txt` lists files explicitly, so the rename is a
  single-line update.

**Alternatives considered**:
- ❌ Keep name: continues to mislead new readers.
- ❌ Rename only after full implementation: delays the hygiene fix.

### Decision 2: Phase 1 covers m8n8k4 f16 only (smallest WMMA fragment)

**Context**: PTX WMMA has many shapes (m8n8k4 / m16n16k16 / m32n8k16 /
m8n32k16 / m16n8k32 etc.) and many dtype combos.

**Choice**: Phase 1 = m8n8k4 + f16 only. Phase 2 extends.

**Rationale**:
- m8n8k4 is the smallest, easiest to verify mathematically (single
  row × single column × 4-element reduction).
- Full coverage is large surface area; cutting into phases reduces
  risk.
- Per `ptx-lessons-learned` §3 (分 Phase commit): each Phase must be
  independently revertible.

**Alternatives considered**:
- ❌ All shapes in one Phase: too large a blast radius.
- ❌ Start with m16n16k16: larger fragment, harder to unit test.

### Decision 3: Use half_utils.h for F16 arithmetic

**Context**: `include/ptxsim/utils/half_utils.h` already has tested
f16 ↔ f32 conversions (cute_rmsnorm baseline).

**Choice**: Reuse `half_utils.h` rather than reinvent.

**Rationale**:
- Already tested by `unit_half_utils` and `unit_half_utils_consistency`.
- The `replace-silent-stub-failures` Lesson #5 (qualifier type
  judgment) tells us to use whole-list traversal, not `back()`.

**Alternatives considered**:
- ❌ Inline FP16 in `tensor.cpp`/`wmma.cpp`: duplicates effort.

### Decision 4: Throw on divergent WMMA (deterministic error)

**Context**: On real hardware, WMMA requires all 32 lanes to
participate. A divergent warp that reaches WMMA with only some
lanes active is undefined.

**Choice**: If `active_mask != 0xFFFFFFFF` when WMMA is dispatched,
throw `ExecutionStateException` (a different exception class than
`UnsupportedInstructionException`, since this is a state issue).

**Rationale**:
- Deterministic error beats silent garbage.
- Per `ptx-lessons-learned` §1: state translation matters;
  throwing is consistent with `barrier.cpp` divergent behavior.

**Alternatives considered**:
- ❌ Run WMMA on the active lanes only (silently): UB on real hw,
  hard to debug.
- ❌ Auto-fill inactive lanes: hides divergent logic errors.

### Decision 5: Tests must validate every fragment element

**Context**: WMMA correctness = fragment math. A subtle off-by-one
in fragment layout produces wrong outputs that look plausible.

**Choice**: Unit test asserts ALL elements of the result fragment
(8×4 for m8n8k4 f32), not just one or two.

**Rationale**:
- Per `ptx-lessons-learned` §5: partial assertions hide bugs (the
  cute_rmsnorm `is_float_type()` lesson).
- A full-element test catches row/col permutation bugs that
  spot-checks miss.

**Alternatives considered**:
- ❌ Spot-check 2-3 elements: insufficient coverage for fragment
  math.

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| Fragment layout mismatch with cutlass expectations | High | Cross-reference PTX ISA §9.7.13 fragment layout diagrams; lock down with type-3 e2e test that compares against real GPU (when available) |
| Divergent WMMA in cute_rmsnorm extension | Medium | Tests must cover both uniform and divergent paths; follow `ptx-lessons-learned` §1 cross-module state translation checklist |
| Renaming `tensor.cpp` → `wmma.cpp` breaks downstream branches | Low | Phased: rename in its own commit, follow with semantics in next; revert if rename-only fails |
| F16 precision drift vs real GPU | Medium | Cross-validate against reference CUDA run for representative inputs |
| Phased approach delays "all shapes" coverage | Low | Each phase is independently useful; later shapes can land in their own changes |
| `set_state(BAR_SYNC)` style cross-module translation for "WMMA complete" | Medium | Use `ptx-lessons-learned` §1 audit: grep downstream consumers of `state == <WMMA_DONE>` before/after each phase |

## Migration Plan

**3 Phase commits**:

```
Phase 1 (this change): rename + m8n8k4 f16 + tests + docs
  Commit 1: chore: rename tensor.cpp → wmma.cpp + AGENTS sync
  Commit 2: feat(wmma): implement m8n8k4 f16 fragment
  Commit 3: docs: remove stub-explicit-failure warning for WMMA
```

Each commit independently revertible. Phase 1 only — Phase 2
(m16n16k16) is a separate change tracked by
`fix/implement-wmma-tensor-core-m16n16k16`.

**Rollback**:
- Phase 1 commit (rename only): revert restores `tensor.cpp`
  content + CMakeLists entry.
- Phase 1 commit (real impl): revert restores throw behavior; the
  rename commit can stay.
- Phase 1 commit (docs): revert restores stub-explicit-failure
  language.

## Open Questions

1. Should `m8n8k4` f32-f32-f32-f32 (no F16 downcast) be supported
   in Phase 1, or deferred? Real PTX allows this; deferred keeps
   Phase 1 smaller.
2. Will `m16n8k32` (Ampere sparse-aware) be needed soon, or is it
   future-work? Check actual cutlass usage in
   `bench/cutlass3x_*` once Phase 1 lands.
3. The cutlass-style e2e test needs a real GPU cross-validation
   target. Does CI have one? If not, e2e test compares against
   hand-computed reference matrix.

## 影响范围

| 组件 | 影响类型 | 详情 |
|------|---------|------|
| `src/ptxsim/instructions/tensor.cpp` → `wmma.cpp` | 改名 + 重写 | 抛异常 → 真实 m8n8k4 实现 |
| `src/ptxsim/instructions/AGENTS.md` | 修改 | KNOWN STUBS 移除 WMMA stub 条目 |
| 根 `AGENTS.md` | 修改 | 已知限制表 WMMA 条目更新 |
| `tests/unit/ptx/test_wmma_not_implemented.cpp` | 改写 | 重新定位为 m8n8k4 单元测试 |
| `tests/integration/wmma/test_wmma_mma_sync.cpp` | 新建 | 集成测试 |
| `tests/e2e/kernel/test_wmma_gemm.cu` | 新建 | E2E GEMM kernel |
| `tests/ptx/parser/test_wmma.cpp` | 修复 | 补 fixtures/tests/test_wmma.ptx + 注册 ctest |
| `src/CMakeLists.txt` | 修改 | 文件名改名同步 |
| `openspec/specs/stub-explicit-failure/spec.md` | 修改 | WMMA-MUST 改为"已实现" |