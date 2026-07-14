## ADDED Requirements

### Requirement: Cleanup MUST be behavior-preserving
The system MUST maintain all existing PTX barrier behaviors unchanged. The cleanup MUST remove `BsyncManager` and `SMContext::synchronize_barrier` without altering observable runtime semantics. The `BarrierModule` API MUST continue to be owned by `CTAContext::barrier_module_` as established by commit `12390b7`.

#### Scenario: Cleanup-only change
- **WHEN** 实施本 change 时
- **THEN** 所有现有 PTX barrier 行为保持不变
- **AND** `BsyncManager` / `SMContext::synchronize_barrier` / SM 级全局 barrier 状态(`barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` / 周期检查)完全删除
- **AND** `BarrierModule` API 保持不变(仍由 `CTAContext::barrier_module_` 持有)
- **AND** `BarWarpSyncHandler::processOperation` 主逻辑保持不变(仅删除 3 处 `bsync_manager_.bsync/release` 调用,保留 `warp_state.wbars[0]` 操作)

### Requirement: warp_context.cpp BAR_SYNC fallback MUST be migrated to BarrierModule
The BAR_SYNC fallback in `src/ptxsim/core/warp_context.cpp:283-296` MUST be migrated to call `cta_context_->get_barrier_module()->arrive_at_cta_barrier(...)` instead of `sm_context_->synchronize_barrier(...)`. This preserves the cross-module state translation chain (`ThreadState::Blocked` → `BAR_SYNC` → `is_blocked`).

#### Scenario: BAR_SYNC fallback replaced
- **WHEN** `warp_context.cpp:283-296` 的 BAR_SYNC fallback 被调用
- **THEN** 调用 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(thread->bar_id, thread)` 而不是 `sm_context_->synchronize_barrier(thread->bar_id, thread)`
- **AND** `cta_context_` 通过 `warp_ctx->get_cta_context()` 获取(commit b04cdb2 引入)

## REMOVED Requirements

### Requirement: BsyncManager class MUST be removed; handlers MUST operate on Wbar directly
The system MUST NOT expose the `ptxsim::BsyncManager` class after this change. The `BarWarpSyncHandler` MUST operate directly on `warp_state.wbars[0]` (Wbar struct) without any intermediate manager layer.

#### Scenario: BsyncManager class deleted
- **WHEN** 开发者引用 `ptxsim::BsyncManager`
- **THEN** 编译错误(class 不存在)
- **AND** 替代方案:`BarWarpSyncHandler` 已不再调用 `bsync_manager_.bsync/release`(所有调用点删除);直接操作 `warp_state.wbars[0]`

### Requirement: SMContext::synchronize_barrier MUST be removed; SM-level barrier state MUST be eliminated
The system MUST NOT expose `SMContext::synchronize_barrier` or any SM-level global barrier state (`barrier_waiting_threads`, `barrier_thread_counts`, `barrier_mutex_`) after this change. CTA-level barriers MUST go through `CTAContext::get_barrier_module()` exclusively. The `warp_context.cpp` BAR_SYNC fallback MUST route through `BarrierModule::arrive_at_cta_barrier`.

#### Scenario: SMContext barrier API deleted
- **WHEN** 开发者调用 `sm_context->synchronize_barrier(...)`
- **THEN** 编译错误(方法不存在)
- **AND** 替代方案:CTA 级 barrier 由 `cta_context->get_barrier_module()` 提供;CTA 同步通过 `BarrierModule::arrive_at_cta_barrier / release_cta_barrier`
- **AND** SM 级全局状态 `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` 字段全部删除
- **AND** `sm_context.cpp:204-242` 周期 barrier 检查代码块删除

## MODIFIED Requirements

### Requirement: Wbar struct MUST remain until Phase 5
The `ptxsim::Wbar` struct (`include/ptxsim/wbar.h`) MUST remain `[[deprecated]]` and continue to be used by `BarWarpSyncHandler`. Its complete removal is deferred to the independent change `migrate-bar-warp-sync-to-barrier-module` (Phase 5). The `warp_state.wbars[]` field MUST remain unchanged.

#### Scenario: Wbar struct preserved
- **WHEN** 开发者引用 `ptxsim::Wbar` 或 `warp_state.wbars[]`
- **THEN** 编译通过(Wbar struct + `warp_state.wbars[]` 字段均存在)
- **AND** `Wbar` struct 头部仍标注 `[[deprecated]]`
- **AND** Phase 5 change(`migrate-bar-warp-sync-to-barrier-module`)负责完整迁移

## Notes

- 本 change 是 Phase 6 partial cleanup(前置于被归档的 `integrate-barrier-module-cta-warp`)
- **范围澄清**:仅删除 `BsyncManager` + `SMContext::synchronize_barrier` + SM 级全局 barrier 状态;不涉及 `Wbar` struct 删除(由独立 Phase 5 change 处理)
- 不涉及 Phase 5 工作(`BarWarpSyncHandler` 完整迁移到 `BarrierModule` API),由独立 change `migrate-bar-warp-sync-to-barrier-module` 处理
- **已知 BUG 测试保留**:`tests/integration/divergence/test_post_barrier_divergence.cpp` 记录 `synchronize_barrier` 的 BUG,作为回归保护保留(Wbar struct 保留后该测试无需修改)
