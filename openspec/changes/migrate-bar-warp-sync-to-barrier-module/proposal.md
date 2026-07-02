## Why

PTX-EMU 在 commit `12390b7` 合并 `fix/barrier-architecture-migration` 后，`BarHandler`（CTA 路径）已切到 `BarrierModule` API。但 **`BarWarpSyncHandler::processOperation`（warp 路径，commit `36dbb9a` 实施）于 commit `f033312` revert**，原因是分歧场景下 `force_reconvergence` 与 barrier 计数交互存在 bug：

> **失败模式（设计文档 postmortem 记录）**：
> 分歧 warp 的两半 lanes（0-15 vs 16-31）在 post-barrier PC 处卡住，无法到达 MAIN_LOOP_PC。`force_reconvergence` 路径与正常 barrier 释放路径的交互存在问题。

本 change **完成 Phase 5 推迟的迁移工作**，把 `BarWarpSyncHandler` 从直接操作 `warp_state.wbars[0]` + `sm_ctx->bsync_manager_` 改为统一通过 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API。

**前置 change（已归档）**：`openspec/changes/archive/2026-06-19-integrate-barrier-module-cta-warp/` Phase 5 DEFERRED。

**关联 change（提议中）**：`cleanup-deprecated-barrier-apis` 删除 `Wbar` / `BsyncManager` / `synchronize_barrier`。本 change 假设该 cleanup 已完成（或同时进行），`warp_state.wbars[0]` 字段已迁移为 `warp_state.barrier`（`WarpBarrier` 实例）。

**前置依赖**：本 change 的实施顺序应在 `cleanup-deprecated-barrier-apis` 之后，或**同时进行**（确保字段名迁移已完成）。

## What Changes

- **修改 `BarWarpSyncHandler::processOperation`**：所有 `wbar.arrive(lane_id)` 改为 `barrier_module.arrive_at_warp_barrier(0, lane_id)`；所有 `wbar.init(...)` 改为 `barrier_module.init_warp_barrier(0, ...)`；所有 `wbar.is_complete()` 改为 `barrier_module.is_warp_barrier_complete(0)`；所有 `wbar.reset()` 改为 `barrier_module` 内部 reset 或 `warp_state.barrier.reset()`
- **修复 force_reconvergence 路径**：`force_reconvergence_at_barrier` 重新进入时 `wbar` 已初始化的场景（BUG-RECONVERGENCE-SIMPLEGEMM），通过 `WarpBarrier::init` 的 `is_initialized_` 分支处理（已设计在 `tasks.md:2.2c` 但未实施）
- **删除 `BsyncManager` 调用**：移除 `sm_ctx->bsync_manager_.bsync/release` 调用（由 `cleanup-deprecated-barrier-apis` 负责删除类本身）
- **新增 unit 测试**：覆盖 `WarpBarrier::init` 已初始化时保留 `arrived_mask` 的场景（task 2.2c.1）
- **新增 e2e 测试**：覆盖分歧 warp 两半分别到达 barrier 后正常完成场景（BUG-POSTBARRIER-TWOHALVES + BUG-RECONVERGENCE-SIMPLEGEMM）

## Capabilities

### New Capabilities
<!-- 无新能力 -->

### Modified Capabilities
- `warp-barrier-unification`: 修改 Warp 级屏障从 `warp_state.wbars[0]` 字段直接访问迁移到 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API 调度

## Impact

| 类别 | 影响 |
|------|------|
| `src/ptxsim/instructions/barrier.cpp` | **修改**：`BarWarpSyncHandler::processOperation` 调用 `BarrierModule` API；移除 `sm_ctx->bsync_manager_.bsync/release` 调用 |
| `src/ptxsim/barrier/warp_barrier.cpp` | **修改**：`WarpBarrier::init` 增加 `is_initialized_` 分支处理（task 2.2c） |
| `tests/unit/barrier/` | **新增**：`WarpBarrier::init preserves arrived_mask` 测试（task 2.2c.1） |
| `tests/integration/divergence/` | **新增**：分歧 warp 两半 barrier 完整生命周期测试 |
| `docs/adr/0008-barrier-semantics.md` | **追加**：`force_reconvergence + BarrierModule` 交互设计决策 |

## References

- 前置 change（已归档）：`openspec/changes/archive/2026-06-19-integrate-barrier-module-cta-warp/` Phase 5 DEFERRED
- 前置 change（提议中）：`cleanup-deprecated-barrier-apis`（删除 `Wbar` / `BsyncManager`，提供 `warp_state.barrier` 字段）
- 已失败尝试（参考）：commit `36dbb9a`（实施）+ commit `f033312`（revert）
- Skill：`ptx-barrier-mechanism`（屏障机制全解，特别是 BUG-POSTBARRIER-TWOHALVES + BUG-RECONVERGENCE-SIMPLEGEMM 部分）
- Skill：`ptx-instruction-pipeline`（指令执行流水线，特别是 `force_reconvergence_at_barrier` 部分）
- Skill：`regression-bisect`（重构后回归定位）
- Skill：`state-modification-audit`（状态修改交叉引用，特别是 `wbar.arrived_mask` 写入路径）
- ADR-0008（barrier 语义增强，含 `force_reconvergence` 决策）
- 调研：`docs/research/barrier-semantics/06-barrier-failure-modes.md`（如存在）

## ⚠️ 风险与历史教训

来自 `docs/dev-process/lessons-learned.md` §1（跨模块间接状态翻译）和 §4（分 Phase commit 纪律）：

1. **避免重新走失败路径**：commit `36dbb9a` 失败原因未完整记录在 lessons-learned.md，需**先做 root cause 分析**再实施
2. **分 Phase commit**：每个 Phase 独立可回退；任何已有测试回归 → 立即 revert 该 Phase
3. **基线 worktree**：实施前 1 分钟建立 baseline.txt，节省"这个失败是基线的还是我的"争论时间
4. **WarpBarrier::init 已初始化分支**（task 2.2c）是关键 — 处理 force_reconvergence 重新进入时 `wbar` 已初始化的场景