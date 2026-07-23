> **⚠️ CHANGE SUPERSEDED (2026-06-19)**:
> 此 change 的实际工作已通过另一分支 `fix/barrier-architecture-migration`（合并 commit `12390b7`）在 main 上完成。
> Phase 5 已 revert（commit `f033312`），仍为 open work。
> **剩余工作已拆分至两个独立 change**（提议中）：
> - `cleanup-deprecated-barrier-apis` — 删除 `Wbar` / `bsync_manager_` / `synchronize_barrier` 死代码
> - `migrate-bar-warp-sync-to-barrier-module` — BarWarpSyncHandler 迁移（独立调查，避免再次踩 Phase 5 失败模式）
>
> 此 change 仅保留为**历史档案**，即将归档。

## 1. 准备与审计

- [x] 1.1 全项目 grep 旧 API 使用点（替代完成：`fix/barrier-architecture-migration` 分支审计记录在 main）
- [x] 1.2 阅读 `src/ptxsim/core/bsync_state.cpp` 全部方法（替代完成：design.md Decision 2 已记录）
- [x] 1.3 创建 worktree 隔离环境（替代完成：`fix/barrier-architecture-migration` 分支使用 worktree）
- [x] 1.4 在 worktree 中建立基线（替代完成：commit `00f698f` baseline）

## 2. 扩展 BarrierModule（CTA 路径能力补齐）

- [x] 2.0 前置验证（TSan）（替代完成：main commit `d4c4ceb` namespace shadowing 修复）
- [x] 2.1 修改 `barrier_module.h`：release_cta_barrier 签名增加 cta_ctx + post_barrier_pc（**main commit `13b6b36`**）
- [x] 2.2 实现 release_cta_barrier 真正释放线程（**main commit `13b6b36`**）
- [x] 2.2b BarrierModule::release_warp_barrier OR active_mask（**main commit `b04cdb2`** 含 `barrier_module.cpp:111-113`）
- [x] 2.2b.1 release_warp_barrier ORs 单元测试（**main 已加**）
- [~] 2.2c WarpBarrier::init 保留 arrived_mask（**保留至 `migrate-bar-warp-sync-to-barrier-module` change**；WarpBarrier 当前语义需先独立设计）
- [~] 2.2c.1 WarpBarrier::init preserves 测试（同上）
- [x] 2.3 CTAContext 持有 BarrierModule（**main `cta_context.cpp:31` + `cta_context.h:85-86`**）
- [x] 2.4 build 编译验证（**main commit `12390b7` 合并后编译通过**）

## 3. 新增 CTABarrier 完整流程单元测试

- [x] 3.1 CTABarrier full arrive-release flow（**main commit `acb2311`**）
- [x] 3.2 CTABarrier mutex concurrent arrives（**main 已加，含 race detector 路径**）
- [x] 3.3 BarrierModule::release_warp_barrier OR active_mask（**main 已加**）
- [x] 3.4 unit_barrier_module 全部 PASS（**main 已验证**）

## 4. 切换 BarHandler 到 BarrierModule（CTA 路径）

- [x] 4.1 BarHandler::executeBarrier 改用 cta_ctx->get_barrier_module()（**main commit `b04cdb2`**, `barrier.cpp:372-377`）
- [x] 4.2 关键修复：release 路径 advance_thread_pc（**main commit `b04cdb2`**, `barrier_module.cpp:182-193`）
- [x] 4.3 integration_cta_barrier_memory_visibility PASS（**main 已验证**）
- [x] 4.4 删除 advance_thread_pc work-around（**main commit `ad7a46f`**）
- [x] 4.5 全量回归通过（**main commit `12390b7` merge 前已验证**）

## 5. 切换 BarWarpSyncHandler 到 BarrierModule（Warp 路径）

> **⚠️ DEFERRED (2026-06-18)**: 实施 commit `36dbb9a` 引入 6 个分歧/集成测试回归（基线 `00f698f` 通过），已在 commit `f033312` 中 revert。详细原因见 [`docs/adr/ADR-0008-barrier-semantics.md` 2026-06-18 Postmortem](../../docs/adr/ADR-0008-barrier-semantics.md#2026-06-18-postmortemplase-5-推迟决策) 和 [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md)。
>
> **失败模式**: 分歧 warp 的两半 lanes（0-15 vs 16-31）在 post-barrier PC 处卡住，无法到达 MAIN_LOOP_PC。`force_reconvergence` 路径与正常 barrier 释放路径的交互存在问题。
>
> **未来实施**: 单独建 change `migrate-bar-warp-sync-to-barrier-module`，优先解决分歧场景的 `force_reconvergence` + barrier 计数交互。

- [~] 5.1 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 正常路径：替换 `wbar.arrive(lane_id)` 为 `barrier_module.arrive_at_warp_barrier(wbar_id, lane_id)`；替换 `wbar.init()` 为 `barrier_module.init_warp_barrier()` — **DEFERRED, reverted in `f033312`**
- [~] 5.2 修改 force_reconvergence 路径：保留 `if (!wbar.is_initialized) ... else 保留 arrived_mask` 的 BUG-RECONVERGENCE-SIMPLEGEMM 修复逻辑，但改为调用 `BarrierModule::init_warp_barrier`（在 `BarrierModule` 内部实现"已初始化则保留 arrived_mask"逻辑）— **DEFERRED**
- [~] 5.3 修改 release 路径：替换 `wbar.is_complete()` + `wbar.arrived_mask` + `wbar.participation_mask` 为 `barrier_module.is_warp_barrier_complete()` + `barrier_module.get_warp_barrier(0)->get_arrived_mask()` — **DEFERRED**
- [~] 5.4 删除 `src/ptxsim/instructions/barrier.cpp` 中所有 `sm_ctx->bsync_manager_.bsync/release` 调用（如果行为已通过 `BarrierModule` 覆盖）；NOTE：先确认 1.2 审计结果再删 — **DEFERRED**
- [~] 5.5 验证：`ctest -R "integration_warp_barrier|integration_barrier_post_barrier|integration_divergence_sync" -V` 全部 PASS；MUST NOT 出现 BUG-POSTBARRIER-TWOHALVES 或 BUG-RECONVERGENCE-SIMPLEGEMM 回归 — **DEFERRED**
- [x] 5.6 **回退**: commit `f033312` revert Phase 5 实施，恢复旧 `Wbar` 路径；`Wbar` 标记 `[[deprecated]]`（Phase 6 占位）

## 6. 旧代码清理

> **DEFERRED → 新 change `cleanup-deprecated-barrier-apis`**：Wbar 标记 `[[deprecated]]` 但未删除；`bsync_manager_` + `synchronize_barrier` 仍存在（main commit `12390b7` 合并后保留）。此清理工作已拆分至独立 change。

- [~] 6.1 删除 `include/ptxsim/wbar.h` + warp_state.h 字段（**→ `cleanup-deprecated-barrier-apis`**）
- [~] 6.2 删除 SMContext::synchronize_barrier + 全局 barrier map（**→ `cleanup-deprecated-barrier-apis`**）
- [~] 6.3 CMakeLists.txt 清理（**→ `cleanup-deprecated-barrier-apis`**）
- [x] 6.4 删除遗留备份（**main commit `12390b7` 合并前已删除**，`find` 验证零匹配）
- [~] 6.5 全项目零匹配验证（**→ `cleanup-deprecated-barrier-apis`** 末尾验证）

## 7. 文档同步

- [x] 7.1 重写 04-ptx-emu-current-implementation.md（**main commit `5439169`**）
- [x] 7.2 barrier_module_design.md 标"已落地 v1"（**main commit `f1ac891`**）
- [x] 7.3 ADR-0008 追加 BarrierModule 集成决策（**main commit `6b7e48b`**）
- [x] 7.4 src/ptxsim/AGENTS.md 增加 barrier/ 子目录（**main commit `42f0fde`**）
- [x] 7.5 src/ptxsim/instructions/AGENTS.md 更新（**main commit `9b1bc72` + current branch state**）
- [x] 7.6 docs grep 验证（**main 已通过**）

## 8. 验证与发布

- [x] 8.1 sanity.sh --quick 通过（**main commit `12390b7` 合并前已验证**）
- [x] 8.2 sanity.sh 完整回归通过（**main 已验证**）
- [x] 8.3 test_all_ptx.sh 通过（**main 已验证**）
- [x] 8.4 worktree 中创建最终 commit（**main 多 commits 完成**）
- [x] 8.5 合并到主分支（**main commit `12390b7` Merge fix/barrier-architecture-migration**）
- [x] 8.6 无需 PR（直接 merge 到 main）

---

## ✅ 此 change 状态总结（2026-06-19）

| Phase | 任务数 | 完成 | DEFERRED | 备注 |
|-------|--------|------|----------|------|
| 1 | 4 | 4 | 0 | 替代完成 |
| 2 | 7 | 5 | 2 | 2.2c/2.2c.1 拆至 `migrate-bar-warp-sync-to-barrier-module` |
| 3 | 4 | 4 | 0 | 替代完成 |
| 4 | 5 | 5 | 0 | 替代完成 |
| 5 | 6 | 1 | 5 | 整体 DEFERRED 至 `migrate-bar-warp-sync-to-barrier-module` |
| 6 | 5 | 1 | 4 | DEFERRED 至 `cleanup-deprecated-barrier-apis` |
| 7 | 6 | 6 | 0 | 替代完成 |
| 8 | 6 | 6 | 0 | 替代完成 |
| **总计** | **43** | **32** | **11** | **main 已完成 74%；剩余 26% 已拆分至两个独立 change** |

**后续追踪**：
- `cleanup-deprecated-barrier-apis` — 提议中（低风险，6 tasks）
- `migrate-bar-warp-sync-to-barrier-module` — 提议中（高风险，10 tasks，需独立调查）
