# 2026-06-24-phase3-cvt-precision-bugfix (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-readme-rebuild` (Fix #4) 补齐。

## Purpose

修复 `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float` 的 denormal 路径错误，恢复 PTX `cvt.f16` 精度正确性。该 bug 在 Phase 3 T2-6 重构时被 `zero behavior change` 约束保留，需要独立 change 修复。

## Implementation

- **Proposal**: `proposal.md` (line 1-50, Bug 1 + Bug 2 详述)
- **Tasks**: `tasks.md` (TDD 三阶段: 红 → 绿 → refactor)
- **Implementation commit**: `fbb7a29` — `fix(half-utils): correct f16_to_f32 denormal path (PTX cvt.f16 precision)` (verify: `git show fbb7a29 --stat`)
- **Test commit**: `f9238cd` — `test(half-utils): verify half_utils.h consistent with cvt_helpers.cpp`
- **Step 1 prereq**: `d3c77b5` (T2-6 Step 1 helper 抽离)
- **Archive commit**: `ccbbe2a` — `chore(openspec): archive completed Phase 3 changes (CVT, half-precision, T2-1, T2-6)`

## Related

- **Successor change**: 修复后仍残留同源 bug（见 `phase3-half-precision-bugfix`），二者合并修复
- **ADR**: 无独立 ADR；遵循 ADR-0015 (CVT 策略模式)
- **T2-6 strategy pattern**: 本 bug 在策略模式重构背景下被发现

## Files

- Modified: `src/ptxsim/instructions/cvt/cvt_helpers.cpp::half_to_float`
- Modified: `tests/unit/ptx/test_cvt_helpers.cpp`
- Reference: `include/ptxsim/utils/half_utils.h::f16_to_f32`（正确实现）

---

**Status**: ✅ RESOLVED (by `fbb7a29`, 2026-06-24)
**Added by**: `docs-readme-rebuild` Fix #4 (2026-07-03)
