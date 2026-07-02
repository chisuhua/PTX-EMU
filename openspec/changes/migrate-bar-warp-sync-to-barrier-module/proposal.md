## Why

PTX-EMU 在 commit `12390b7` 合并 `fix/barrier-architecture-migration` 后，`BarHandler`（CTA 路径）已切到 `BarrierModule` API。但 **`BarWarpSyncHandler::processOperation`（warp 路径，commit `36dbb9a` 实施）于 commit `f033312` revert**，原因是分歧场景下 `force_reconvergence` 与 barrier 计数交互存在 bug：

> **失败模式（设计文档 postmortem 记录）**：
> 分歧 warp 的两半 lanes（0-15 vs 16-31）在 post-barrier PC 处卡住，无法到达 MAIN_LOOP_PC。`force_reconvergence` 路径与正常 barrier 释放路径的交互存在问题。

本 change **完成 Phase 5 推迟的迁移工作**，把 `BarWarpSyncHandler` 从直接操作 `warp_state.wbars[0]` + `sm_ctx->bsync_manager_` 改为统一通过 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API。

**前置 change（已归档）**：`openspec/changes/archive/2026-06-19-integrate-barrier-module-cta-warp/` Phase 5 DEFERRED。

**前置 change（已归档，2026-06-20）**：`openspec/changes/archive/2026-06-20-cleanup-deprecated-barrier-apis/` 删除了 `BsyncManager` / `synchronize_barrier` / `bsync_state.{h,cpp}`（commits `8a5573d`/`7914764`/`6ec8efd`，归档 commit `ded4f96`）。**注意**：cleanup 未删除 `Wbar` struct 和 `warp_state.wbars[]` —— 该工作由 **本 change 的 Phase 7** 完成。

**前置依赖**：`cleanup-deprecated-barrier-apis` 已归档，`BsyncManager` 在生产代码中零匹配。可直接启动本 change。`barrier.cpp` 中残留的 `sm_ctx->bsync_manager_.bsync/release` 调用可直接删除。

## What Changes

- **修改 `BarWarpSyncHandler::processOperation`**：所有 `wbar.arrive(lane_id)` 改为 `barrier_module.arrive_at_warp_barrier(0, lane_id)`；所有 `wbar.init(...)` 改为 `barrier_module.init_warp_barrier(0, ...)`；所有 `wbar.is_complete()` 改为 `barrier_module.is_warp_barrier_complete(0)`；release 路径改为调用 `barrier_module.release_warp_barrier(0, warp_ctx)`（已包含 BUG-POSTBARRIER-TWOHALVES OR 逻辑）
- **验证 force_reconvergence 不变性**：`WarpBarrier::init` 的 `is_initialized_` 分支（BUG-RECONVERGENCE-SIMPLEGEMM 修复）已于 main 提前落地（`warp_barrier.cpp:18-31`），本 change 验证其正确性
- **删除 `BsyncManager` 调用**：移除 `barrier.cpp` 中 `sm_ctx->bsync_manager_.bsync/release` 调用（`BsyncManager` 类本身已于 `cleanup-deprecated-barrier-apis` 中删除）
- **删除 `Wbar` struct 及残留引用**（Phase 7，P0-A5）：删除 `include/ptxsim/wbar.h` 全部内容；删除 `warp_state.wbars[]` + `current_wbar_id` 字段；删除 `get_wbar()` compat shim
- **新增 integration 测试**：覆盖分歧 warp 两半通过 BarrierModule API 分别到达 barrier 后正常完成场景（commit `36dbb9a` 失败案例的复现 + 修复验证）

## Capabilities

### New Capabilities
<!-- 无新能力 -->

### Modified Capabilities
- `warp-barrier-unification`: 修改 Warp 级屏障从 `warp_state.wbars[0]` 字段直接访问迁移到 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API 调度

## Impact

| 类别 | 影响 |
|------|------|
| `src/ptxsim/instructions/barrier.cpp` | **修改**：`BarWarpSyncHandler::processOperation` 调用 `BarrierModule` API；移除 `sm_ctx->bsync_manager_.bsync/release` 调用 |
| `src/ptxsim/barrier/warp_barrier.cpp` | **不变**：`WarpBarrier::init` 的 `is_initialized_` 分支已提前落地（commit `b04cdb2`），本 change 仅验证 |
| `src/ptxsim/barrier/barrier_module.cpp` | **不变**：`release_warp_barrier` OR 逻辑已提前落地 |
| `include/ptxsim/wbar.h` | **删除**（Phase 7）：整个文件（121 行），`Wbar` struct 无生产引用 |
| `include/ptxsim/warp_state.h` | **修改**（Phase 7）：删除 `wbars[]` + `current_wbar_id` 字段 + reset 中的对应逻辑 |
| `include/ptxsim/warp_context.h` | **修改**（Phase 7）：删除 `get_wbar()` compat shim 声明 |
| `src/ptxsim/core/warp_context.cpp` | **修改**（Phase 7）：删除 `get_wbar()` 实现（L540-556） |
| `tests/integration/divergence/` | **新增**：分歧 warp 两半 barrier 完整生命周期测试 |
| `docs/adr/0008-barrier-semantics.md` | **追加**：`force_reconvergence + BarrierModule` 交互设计决策 + Wbar 删除记录 |

## References

- 前置 change（已归档）：`openspec/changes/archive/2026-06-19-integrate-barrier-module-cta-warp/` Phase 5 DEFERRED
- 前置 change（已归档，2026-06-20）：`openspec/changes/archive/2026-06-20-cleanup-deprecated-barrier-apis/`（删除 `BsyncManager` / `synchronize_barrier`，commits `8a5573d`/`7914764`/`6ec8efd`）
- 技术债务审计：`docs/audits/debt-audit-2026-07-02.md` §P0-A5（Wbar 未删除 gap）
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