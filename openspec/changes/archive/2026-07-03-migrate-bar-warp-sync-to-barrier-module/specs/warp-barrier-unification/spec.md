## MODIFIED Requirements

### Requirement: BarWarpSyncHandler MUST use BarrierModule API exclusively
The `BarWarpSyncHandler::processOperation` MUST manage all barrier state through `BarrierModule` API exclusively. Direct access to `warp_state.wbars[]` fields from handler code MUST be eliminated.

#### Scenario: Normal barrier path uses BarrierModule
- **WHEN** `BarWarpSyncHandler::processOperation` 路径 B（无分歧）执行
- **THEN** 调用 `barrier_module.init_warp_barrier(0, mask, reconv_pc, barrier_pc)` 初始化屏障
- **AND** 调用 `barrier_module.arrive_at_warp_barrier(0, lane_id)` 标记 lane 到达
- **AND** 调用 `barrier_module.is_warp_barrier_complete(0)` 检查完成
- **AND** 调用 `barrier_module.release_warp_barrier(0, warp_ctx)` 释放线程
- **AND** 不再直接读写 `warp_state.wbars[]` 字段

#### Scenario: Force-reconvergence barrier path uses BarrierModule
- **WHEN** `BarWarpSyncHandler::processOperation` 路径 A（分歧场景）执行 `force_reconvergence_at_barrier`
- **THEN** 调用 `barrier_module.init_warp_barrier(0, mask, reconv_pc, barrier_pc)` 重新初始化
- **AND** 若屏障已初始化（force_reconvergence 重新进入），`WarpBarrier::init` 保留 `arrived_mask_` 不重置
- **AND** 调用 `barrier_module.arrive_at_warp_barrier(0, lane_id)` 标记到达
- **AND** 完成时调用 `barrier_module.release_warp_barrier(0, warp_ctx)` 释放所有到达 lane

### Requirement: WarpBarrier::init MUST preserve arrived_mask on re-initialization
The `WarpBarrier::init` method MUST preserve `arrived_mask_` when called on an already-initialized barrier, while still updating `participation_mask_`, `reconvergence_pc_`, `barrier_pc_`, `expected_count_`, and resetting `state_` to `Waiting`.

#### Scenario: First-time init resets all state
- **WHEN** `WarpBarrier::init(participation_mask, reconvergence_pc, barrier_pc)` 在 `is_initialized_ == false` 时调用
- **THEN** `arrived_mask_ = 0`
- **AND** `arrived_count_ = 0`
- **AND** `participation_mask_ = participation_mask`
- **AND** `reconvergence_pc_ = reconvergence_pc`
- **AND** `barrier_pc_ = barrier_pc`
- **AND** `expected_count_ = popcount(participation_mask)`
- **AND** `state_ = Waiting`
- **AND** `is_initialized_ = true`

#### Scenario: Re-init preserves arrived_mask
- **WHEN** `WarpBarrier::init` 在 `is_initialized_ == true` 时调用（force_reconvergence 重新进入场景）
- **THEN** `arrived_mask_` 不变（保留之前到达的 lane）
- **AND** `arrived_count_` 不变
- **AND** `participation_mask_ = participation_mask`（更新）
- **AND** `reconvergence_pc_ = reconvergence_pc`（更新）
- **AND** `barrier_pc_ = barrier_pc`（更新）
- **AND** `expected_count_ = popcount(participation_mask)`（更新）
- **AND** `state_ = Waiting`（重置）

#### Scenario: Re-init MUST NOT reset arrived_mask for BUG-RECONVERGENCE-SIMPLEGEMM
- **WHEN** 屏障已初始化，lane 0..15 已到达（arrived_mask=0x0000FFFF）
- **AND** 再次调用 `init`（force_reconvergence 重新进入）
- **THEN** `arrived_mask_` 仍为 0x0000FFFF（含 lane 0..15 到达记录）
- **AND** 新到达 lane 16..31 通过 `arrive()` 累积到 `arrived_mask_` 直至全部到达
- **AND** 屏障完成时不丢失任何 lane 到达记录

### Requirement: BUG-POSTBARRIER-TWOHALVES OR active_mask invariant MUST hold
The `BarrierModule::release_warp_barrier` MUST call `set_active_mask(get_active_mask() | arrived_mask)` (OR-merge semantics) rather than overwriting, to preserve lanes released by prior barrier calls in divergent two-halves scenarios.

#### Scenario: Two divergent halves reach barrier at different times
- **WHEN** divergent warp lanes 0..15 到达 barrier 并被释放
- **THEN** `active_mask` 包含 lanes 0..15
- **AND** 后续 lanes 16..31 到达 barrier
- **AND** `BarrierModule::release_warp_barrier` 调用 `set_active_mask(get_active_mask() | arrived_mask)`
- **AND** `active_mask` 包含 lanes 0..31（OR 合并，不覆写）

## ADDED Requirements

### Requirement: BarrierModule API MUST be the sole entry point for warp barrier
All warp-level barrier operations MUST go through `BarrierModule` API. Handler code MUST NOT directly read or write `warp_state.wbars[]` fields.

#### Scenario: No direct warp_state.wbars access from handlers
- **WHEN** 开发者修改 barrier handler 代码
- **THEN** 不得直接读写 `warp_state.wbars[]` 字段
- **AND** 必须通过 `barrier_module.init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API
- **AND** 编译期检查：通过 `grep -rn "warp_state.wbars\|warp_state.current_wbar_id" src/ptxsim/instructions/` 输出为空

## REMOVED Requirements

### Requirement: BarWarpSyncHandler MUST NOT call BsyncManager
The `BarWarpSyncHandler::processOperation` MUST NOT call `sm_ctx->bsync_manager_.bsync(...)` or `sm_ctx->bsync_manager_.release(...)` after this change. The `BsyncManager` class MUST be deleted by the prerequisite change `cleanup-deprecated-barrier-apis`.

#### Scenario: No BsyncManager calls in BarWarpSyncHandler
- **WHEN** 实施本 change 完成
- **THEN** `BarWarpSyncHandler::processOperation` 不调用 `sm_ctx->bsync_manager_.bsync(...)`
- **AND** 不调用 `sm_ctx->bsync_manager_.release(...)`
- **AND** `BsyncManager` 类已被 `cleanup-deprecated-barrier-apis` 删除

## Notes

- 本 change **不实现 `bar.warp.sync` membermask 的 UB 检测**（已记录在 ADR-0008 未来工作）
- 本 change 假设 `cleanup-deprecated-barrier-apis` 已完成或同步进行
- commit `36dbb9a` 失败 postmortem 详见 `docs/dev-process/lessons-learned.md` 与 `docs/adr/ADR-0008-barrier-semantics.md` §2026-06-18