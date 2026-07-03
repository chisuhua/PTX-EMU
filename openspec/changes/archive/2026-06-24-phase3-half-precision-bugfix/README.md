# 2026-06-24-phase3-half-precision-bugfix (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-readme-rebuild` (Fix #4) 补齐。

## Purpose

修复 `include/ptxsim/utils/half_utils.h::f16_to_f32` 的 denormal 路径错误，与 `phase3-cvt-precision-bugfix` 同源（不同文件实现同一算法）。`phase3-cvt-precision-bugfix` (commit `32ce8a0`) 只修了本地 `cvt_helpers.cpp::half_to_float`，未触碰 `half_utils.h` 源文件。

## Implementation

- **Proposal**: `proposal.md` (line 1-30，同源 bug 描述)
- **Tasks**: `tasks.md` (TDD 三阶段)
- **Implementation commit**: `fbb7a29` — `fix(half-utils): correct f16_to_f32 denormal path` （与 CVT bugfix 同一 commit，**确为同一修复同时触达两文件**）(verify: `git show fbb7a29 --stat`)
- **Test commit**: `f9238cd` — `test(half-utils): verify half_utils.h consistent with cvt_helpers.cpp`
- **Archive commit**: `ccbbe2a` — `chore(openspec): archive completed Phase 3 changes`

## Related

- **Sibling change**: `phase3-cvt-precision-bugfix`（同一 commit `fbb7a29` 修复）
- **T2-6 delegating commit**: `2f3c150` — `refactor(cvt): delegate half_to_float/float_to_half to half_utils.h`（让两个文件实现统一）
- **ADR**: 无独立 ADR

---

**Status**: ✅ RESOLVED (by `fbb7a29`, 2026-06-24)
**Added by**: `docs-readme-rebuild` Fix #4 (2026-07-03)
