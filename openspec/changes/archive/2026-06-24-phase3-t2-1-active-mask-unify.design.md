**Retroactive synthesis from git log — not an original design document**
> 合成日期: 2026-07-06
> 来源: `proposal.md` (DUAL STATE MECHANISM 详述) + `tasks.md` (Task 1-5 TDD) + git log (commits `5e0e315`, `8b1d23b`, `33e1f99`, `8248303`) + 关联 change `integrate-barrier-module-cta-warp` (prereq) + ADR-0008 (barrier semantics)

# Phase 3 T2-1: active_mask 三源统一 — Retroactive Design

## Context

PTX-EMU warp 活跃状态有**三路并存**的源头：
1. `active_mask[]` (WarpContext 字段，9 个核心读写点)
2. `warp_state.threads[i].is_active` (WarpState 字段，~30 写入点)
3. `warp_state.exec_mask` (8 写入点)

三路并存已引发 BUG-RETHANG、BUG-POSTBARRIER-TWOHALVES 等历史问题。`src/ptxsim/core/AGENTS.md:49-60` 显式标注 "DUAL STATE MECHANISM" 警告。测试稀薄（J1-J10 `test_active_mask_consistency` 已写但未启用），god class 无法 mock 是根因。本 change 收敛到 **single source of truth**：权威源定为 `warp_state.threads[i].is_active`，`active_mask[]` 与 `active_count` 改为派生。WarpState 6 字段中的 `wbars` + `current_wbar_id` 标 `[[deprecated]]`（不删，等 `integrate-barrier-module-cta-warp` 合并后由 T2-3 物理删除）。

## Goals / Non-Goals

**Goals:**
- `update_active_mask()` 改为统一从 `warp_state.threads[i]` 4 字段读取（`is_active` + `!is_exited` + `!is_blocked` + `status==Active`）并双向同步
- `is_lane_active()` 委托 `is_lane_schedulable()`（消除 ISSUE-005 重复实现）
- `sync_to_warp_state(RUN)` 补 `is_active = true`（屏障释放后正确标记活跃）
- 保留 `set_active_mask` 双模式接口（`set_active_mask(0u)` 给 RetHandler 覆写语义，其他保留）
- `WarpState.wbars[]` + `current_wbar_id` 标 `[[deprecated]]`
- 重写 `src/ptxsim/core/AGENTS.md:49-60` 从 "DUAL STATE MECHANISM" 改为 "SINGLE SOURCE OF TRUTH"
- 启用 J1-J10 active_mask 一致性测试

**Non-Goals:**
- 删除 `active_mask[]` 字段（RetHandler 依赖，留给 T2-3 删）
- `ThreadContext::state` (EXE_STATE) 与 `warp_state.threads[i].status` 双源合并（留给 T2-3）
- god class POD 拆分（独立 change `phase3-t2-3-god-class-split`）
- 物理删除 `WarpState.wbars[]` + `current_wbar_id`（等 `integrate-barrier-module-cta-warp` 合并）

## Decisions

### Decision 1: 收敛目标 — `warp_state.threads[i].is_active` 为权威源

**问题**: 三路状态并存导致历史 bug。需选择**唯一权威源**。

**方案分析**:
- **方案 A**: 选 `warp_state.threads[i].is_active`（per-thread 状态）作为权威源。`active_mask[]` 改为派生。
- **方案 B**: 选 `active_mask[]`（warp 级位图）作为权威源。需重构大量写入点（~30 处 `is_active` 修改需改为位图操作）。
- **方案 C**: 选 `warp_state.exec_mask` 作为权威源。语义错位 — exec_mask 是 PTX `activemask` 指令专用源，**不应**与 `is_active` 共享。

**选择**: **方案 A**。`warp_state.threads[i].is_active` 是 per-thread 状态的标准字段，与 `warp_state.threads[i].is_blocked` / `is_exited` / `status` 同源，修改 30 处 `is_active =` 写入点工作量最小。

**证据**: tasks.md Task 1 "改为统一从 `warp_state.threads[i]` 读取" + ADR-0008 §barrier 语义 "Caller 层 OR，不可改 `set_active_mask` 全局语义"。

### Decision 2: `set_active_mask` 双模式接口

**问题**: `call.cpp:29` 的 RetHandler 依赖 `set_active_mask(0u)` 的**覆写语义**清零（让线程退出）。但统一权威源路径下，`set_active_mask(0u)` 应该 OR 还是覆写？

**方案分析**:
- **方案 A**: 保留双模式接口 — `set_active_mask(0u)` 给 RetHandler 走覆写分支，其他调用点保留现有 OR 行为。caller 层显式选择语义。
- **方案 B**: 统一为 OR。RetHandler 改用单独 API。
- **方案 C**: 统一为覆写。BarrierModule 等 OR 调用点改 caller 层 OR (`set_active_mask(get_active_mask() | arrived_mask)`)。

**选择**: **方案 A**。最小修改 + 保留 caller 灵活性 + 显式语义。

**证据**: tasks.md Task 5 Step 1 "建议：保留（OR-merge 仍由 BarrierModule 内部承担，但 caller 显式 set 可让同周期生效，避免下一周期延迟）" + ADR-0008 "Caller 层 OR，不可改 set_active_mask 全局语义 — ret handler 依赖覆写语义清零"。

### Decision 3: `wbars` + `current_wbar_id` 仅 deprecate 不删除

**问题**: 物理删除这两个字段会触发 BUG-POSTBARRIER-TWOHALVES，因 `integrate-barrier-module-cta-warp` 还未合并（barrier handler 仍直接读 `warp_state.wbars[]`）。

**方案分析**:
- **方案 A**: 标 `[[deprecated]]` 警告，**不删**。等 `integrate-barrier-module-cta-warp` 合并后再删（由 T2-3 物理删除）。
- **方案 B**: 直接物理删除。触发 BUG-POSTBARRIER-TWOHALVES 回归。
- **方案 C**: 等待 `integrate-barrier-module-cta-warp` 合并再启动本 change。

**选择**: **方案 A**。保持本 change 独立可回退，不依赖其他 change 的合并顺序。

**证据**: tasks.md Task 5 Step 3 "标 `[[deprecated]]`，**不删**"。

## Implementation Commits

> **注**: 以下 commits 在 change 归档时已合并到 main，本节为追溯原始实施链。

| Commit | Sub-task | 摘要 |
|--------|----------|------|
| `5e0e315` | Task 5 (主 commit) | `docs(agents): rewrite DUAL STATE MECHANISM section + deprecate wbars` — AGENTS.md 49-60 重写 + `WarpState.wbars[]` + `current_wbar_id` 标 `[[deprecated]]` |
| `8b1d23b` | Task 2 | `refactor(warp): delegate is_lane_active() to is_lane_schedulable()` — 消除 ISSUE-005 重复 |
| `33e1f99` | T2-3 B1 (协同) | `refactor(test): use set_state() in reset_warp for forward compat` — T2-3 前向兼容 |
| `8248303` | T2-3 B2 (协同) | `refactor(test): use set_state() in test_post_barrier_divergence` — T2-3 前向兼容 |
| `ccbbe2a` | Archive | `chore(openspec): archive completed Phase 3 changes` |
| `3006e11` | Orphan README (后续) | `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)` |

> **缺失 commit (推测)**: Task 1 (`update_active_mask()` 统一数据源) + Task 3 (`sync_to_warp_state(RUN)` 补 `is_active=true`) + Task 4 (启用 J1-J10 测试) 可能在 commit `5e0e315` 合并提交，或分散在其他 commit。**以 git log 与原 commit body 为准**。

## Risks / Trade-offs

| 风险 | 缓解（per proposal.md + tasks.md）|
|------|------|
| 物理删除 `wbars` + `current_wbar_id` 触发 BUG-POSTBARRIER-TWOHALVES | 标 `[[deprecated]]` 不删，等 `integrate-barrier-module-cta-warp` 合并后由 T2-3 删 |
| `set_active_mask` 全局语义被改坏 RetHandler | 保留双模式接口（`0u` 走覆写分支）|
| 4 字段同步 (is_active + !is_exited + !is_blocked + status==Active) 漏字段 | 跑 `test_post_barrier_two_halves` + `test_post_barrier_divergence` 回归 |
| 启用 J1-J10 测试发现新 bug | 单 commit 启用，失败立即 revert |

## Cross-References

- 原 artifacts: `openspec/changes/archive/2026-06-24-phase3-t2-1-active-mask-unify/{proposal.md,tasks.md,README.md}`
- 关联 change: `integrate-barrier-module-cta-warp` (prereq，2026-06-18 archived)
- 关联 change: `openspec/changes/archive/2026-06-24-phase3-t2-3-god-class-split/` (下游，物理删除 wbars)
- ADR-0008: barrier semantics (`Caller 层 OR，不可改 set_active_mask 全局语义`)
- AGENTS.md §DUAL STATE MECHANISM: `src/ptxsim/core/AGENTS.md:49-60` (T2-1 改写)
- Lessons-Learned: §1 (跨模块状态翻译 — `is_active` 与 `is_blocked` / `status` 的多源同步是隐性 invariant，必须行级 diff 验证)

## Notes

> 本文件为 retroactive synthesis，最佳努力重建。如发现与原 commit body 不一致，**以原 commit body 为准**。
> 任何修改归档目录内文件的尝试被禁止（per Checklist G + Decision 1）。
>
> **实施时长**: 3-5 天 (per proposal.md 顶部) | **实际**: 主要 4 commits (`5e0e315` 主 + 3 协同)，其余 Task 1/3/4 commit 链不完整（需 git log 验证）
>
> **遗留**: J1-J10 测试可能部分被 `integrate-barrier-module-cta-warp` 后续 commits 替换；`update_active_mask()` 统一 + `sync_to_warp_state(RUN)` 补 `is_active` 的 commit 需 git log 补全。
