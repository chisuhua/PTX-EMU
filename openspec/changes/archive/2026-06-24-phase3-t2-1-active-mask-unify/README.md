# 2026-06-24-phase3-t2-1-active-mask-unify (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-readme-rebuild` (Fix #4) 补齐。

## Purpose

消除 DUAL STATE MECHANISM — 统一三路 active_mask 来源：
- `active_mask[]`（WarpContext，9 个核心读写点）
- `warp_state.threads[i].is_active`（WarpState，~30 写入点）
- `warp_state.exec_mask`（8 写入点）

三路并存已引发 BUG-RETHANG、BUG-POSTBARRIER-TWOHALVES 等历史问题。是 T2-3 god class 拆分的前置。

## Implementation

- **Proposal**: `proposal.md` (DUAL STATE + 上下游依赖详述)
- **Tasks**: `tasks.md` (Task 1-5)
- **Implementation commits** (multiple):
  - `5e0e315` — `docs(agents): rewrite DUAL STATE MECHANISM section + deprecate wbars`
  - `8b1d23b` — `refactor(warp): delegate is_lane_active() to is_lane_schedulable()`
  - `33e1f99` — `refactor(test): use set_state() in reset_warp for forward compat`
  - `8248303` — `refactor(test): use set_state() in test_post_barrier_divergence`
- **Archive commit**: `ccbbe2a` — `chore(openspec): archive completed Phase 3 changes`

## Related

- **Downstream**: `phase3-t2-3-god-class-split` (POD 拆分依赖此)
- **Upstream barrier**: `cleanup-deprecated-barrier-apis` (2026-06-20 archive) 部分依赖此
- **ADR-0005**: Memory region registration（隐含依赖 active_mask 语义）

---

**Status**: ✅ RESOLVED (multi-commit, 主要为 `5e0e315` + `8b1d23b`)
**Added by**: `docs-readme-rebuild` Fix #4 (2026-07-03)
