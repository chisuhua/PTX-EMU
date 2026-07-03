# 2026-06-24-phase3-t2-3-god-class-split (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-readme-rebuild` (Fix #4) 补齐。

## Purpose

解决 ThreadContext (26 公共数据字段) + WarpContext (32 数据字段) god class 问题。
- 抽取 7 POD structs（`ThreadState`, `WarpState` 等）作为测试 mock 边界
- 删除死代码字段（`thread_predicates`、`warp_pc`）
- WarpContext 6 字段减为 2 字段（threads[] + exec_mask）

T2-3 是 T2-1 的下游：拆分后状态机必须先收敛到 single source of truth（POD 边界依赖统一数据源）。

## Implementation

- **Proposal**: `proposal.md` (god class 数据 + 拆分计划)
- **Tasks**: `tasks.md` (A1-A5 + B1-B2 + C1)
- **Implementation commits** (主要 commit，按时间顺序):
  - `7054593` — `refactor(contexts): extract 7 POD structs from ThreadContext/WarpContext` (T2-3 A1)
  - `7952120` — `refactor(thread): add 4-POD members at end of class` (T2-3 A3a additive)
  - `67ad828` — `refactor(thread): init() populates 4 POD members` (T2-3 A3b)
  - `8b9b025` — `refactor(warp): add 3-POD facade members at end of class` (T2-3 A4a)
  - `2a3b48a` — `refactor(warp): add_thread() mirrors LaneMaskPod fields` (T2-3 A4b)
  - `5617665` — `refactor(warp-state): remove thread_predicates + warp_pc dead fields` (T2-3 A2)
  - `421eec9` — `refactor(test): migrate BarrierModule test from get_wbar() shim` (T2-3 A5 PoC)
  - `af6f52f` — `docs(t2-3): document A5 blocker discovery - barrier.cpp uses legacy state`
  - `a0b8281` — `docs(t2-3): update A5 status with PoC + discovery findings`
  - `33e1f99` — `refactor(test): use set_state() in reset_warp` (T2-3 B1)
  - `8248303` — `refactor(test): use set_state() in test_post_barrier_divergence` (T2-3 B2)
  - `2d24403` — `fix(factory): register InstructionFactory::cleanup() via atexit` (T2-3 C1)
- **Audit record**: `7c5bede` — `docs(audit): mark god class (M2/M3) fixed by T2-3`
- **Archive commit**: `02274a7` — `chore(openspec): archive completed T2-3 god class split`

## Related

- **Upstream**: `phase3-t2-1-active-mask-unify` (DUAL STATE 收敛)
- **Discoveries from this change**:
  - **`af6f52f` A5 blocker**: barrier.cpp 仍使用 legacy `warp_state.wbars[]` + `current_wbar_id`，触发后续 `cleanup-deprecated-barrier-apis` (archive 2026-06-20)
- **后续 cleanup**: `cleanup-deprecated-barrier-apis` (已 archive 2026-06-20) + `migrate-bar-warp-sync-to-barrier-module` (已 archive 2026-07-03)
- **ADR-0005**: Memory region registration (POD 边界依赖)
- **ADR-0012**: Per-thread PC (POD split 的核心驱动)

---

**Status**: ✅ RESOLVED (multi-commit, 主要为 `7054593` + `5617665`)
**Added by**: `docs-readme-rebuild` Fix #4 (2026-07-03)
