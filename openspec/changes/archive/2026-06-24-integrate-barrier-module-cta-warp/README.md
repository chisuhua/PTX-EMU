# 2026-06-24-integrate-barrier-module-cta-warp (Archived)

> **⚠️ CHANGE SUPERSEDED (2026-06-19)** — 实际工作通过另一个 change 完成，本归档仅保留历史决策记录。
> 本 README 由 `docs-readme-rebuild` (Fix #4) 补齐。

## Status

**Superseded by**: [`2026-06-20-cleanup-deprecated-barrier-apis`](../2026-06-20-cleanup-deprecated-barrier-apis/) (archive 已生成)

> 此 change 的实际工作通过分支 `fix/barrier-architecture-migration`（合并 commit `12390b7`）在 main 上完成。Phase 5 已 revert（commit `f033312`），后续拆分为两个独立 change：
> - `cleanup-deprecated-barrier-apis` — 删除 `Wbar` / `bsync_manager_` / `synchronize_barrier` 死代码 (archived 2026-06-20)
> - `migrate-bar-warp-sync-to-barrier-module` — BarWarpSyncHandler 迁移 (archived 2026-07-03)

## Original Purpose

将 `BarHandler` (CTA 路径) 与 `BarWarpSyncHandler` (warp 路径) 统一通过 `BarrierModule` API：
- 删除 `Wbar` struct（已 `[[deprecated]]`）
- 删除 `BsyncManager` 类
- 替换 `synchronize_barrier` 函数

## Implementation (Historical References)

- **Main merge**: `12390b7` — `Merge: fix/barrier-architecture-migration branch into main`
- **Major commits**:
  - `d4c4ceb` — namespace shadowing 修复 (TSan 前置)
  - `13b6b36` — `barrier_module.h`: release_cta_barrier 签名增加 cta_ctx + post_barrier_pc
  - `b04cdb2` — `release_cta_barrier` 实现 (CRITICAL FIX BUG-HANDLER-PC-ADVANCE)
  - `f033312` — Phase 5 revert（BarWarpSyncHandler bug 后退回）
- **Phase 5 failures**: see `docs/dev-process/lessons-learned.md` (16 个失败模式)
- **Archive commit**: `e0735ff` — `chore(openspec): archive integrate-barrier-module-cta-warp + update tasks`

## Successor Changes (Post-Split)

| Successor | Status | Reference |
|-----------|--------|-----------|
| `cleanup-deprecated-barrier-apis` (2026-06-20) | ✅ Archived | [archive](../2026-06-20-cleanup-deprecated-barrier-apis/) |
| `migrate-bar-warp-sync-to-barrier-module` (2026-07-03) | ✅ Archived | [archive](../2026-07-03-migrate-bar-warp-sync-to-barrier-module/) |
| `barrier-module-lifecycle-tests` (2026-07-03) | ✅ Archived | [archive](../2026-07-03-barrier-module-lifecycle-tests/) |
| `dead-code-cleanup` (2026-07-03) | ✅ Archived | [archive](../2026-07-03-dead-code-cleanup/) |

## Lessons Learned (⚠️ Critical)

本 change 失败模式沉淀在 [`docs/dev-process/lessons-learned.md`](../../../../docs/dev-process/lessons-learned.md)：
- **Lesson 1**: 跨模块间接状态翻译（`set_state(BAR_SYNC)` 看似冗余实为另一模块 API 契约）
- **Lesson 2**: 递归锁死锁（arrive() 持锁调 is_complete() 重复加锁）
- **Lesson 3**: 复杂迁移必须分 Phase commit（Phase 5 失败原因之一）
- **Lesson 4**: 基线 worktree 保险

后续 change 严格遵守这 4 条教训 + Checklist A/B/D/E/F/G。

---

**Status**: ✅ SUPERSEDED (by `2026-06-20-cleanup-deprecated-barrier-apis` + 2026-07 changes)
**Added by**: `docs-readme-rebuild` Fix #4 (2026-07-03)
