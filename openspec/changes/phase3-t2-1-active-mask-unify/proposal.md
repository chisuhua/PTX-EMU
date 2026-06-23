# Phase 3 T2-1: active_mask 三源统一

## Why

DUAL STATE MECHANISM：`active_mask[]`（WarpContext, 9 个核心读写点）/ `warp_state.threads[i].is_active`（WarpState, ~30 写入点）/ `warp_state.exec_mask`（8 写入点）三路状态并存，已引发 BUG-RETHANG、BUG-POSTBARRIER-TWOHALVES 等历史问题。`src/ptxsim/core/AGENTS.md:49-60` 明确警告。测试稀薄（`test_active_mask_consistency` J1-J10 已存在但未启用），god class 无法 mock 是根因。

消除 DUAL STATE 是 T2-3 god class 拆分的前置（拆分后状态机必须先收敛），也是 `integrate-barrier-module-cta-warp` 完整收尾的协同任务。

## What Changes

- **`update_active_mask()` 统一数据源**：改为统一从 `warp_state.threads[i]` 读取（`is_active` + `!is_exited` + `!is_blocked` + `status==Active`），并双向同步到 `warp_state.threads[i].is_active`
- **`is_lane_active()` 委托 `is_lane_schedulable()`**：消除 ISSUE-005
- **`sync_to_warp_state(RUN)` 补 `is_active = true`**：屏障释放后正确标记活跃
- **保留 `set_active_mask` 双模式接口**：`set_active_mask(0u)` 给 RetHandler 用覆写语义（`call.cpp:29`），其他调用点保持现有行为
- **`WarpState.wbars[]` + `current_wbar_id` 标 `[[deprecated]]`**：**不删**（等 `integrate-barrier-module-cta-warp` 合并）
- **重写 `src/ptxsim/core/AGENTS.md:49-60` 段**：从 "DUAL STATE MECHANISM" 改为 "SINGLE SOURCE OF TRUTH"

## Out of Scope

- 删除 `active_mask[]` 字段（RetHandler 依赖，留在 T2-3 删）
- `ThreadContext::state` (EXE_STATE) 与 `warp_state.threads[i].status` 双源合并（留给 T2-3）
- god class POD 拆分（独立 change phase3-t2-3-god-class-split）
- 物理删除 `WarpState.wbars[]` + `current_wbar_id`（等 integrate-barrier-module-cta-warp 合并）

## Pre-conditions

- ✅ T1-1..T1-5（Phase 2 完成）
- ✅ T2-2, T2-4 Step 1, T2-5, T2-7
- **必须**：`integrate-barrier-module-cta-warp` 已合并（`barrier_module.cpp:107-112` 的 OR-merge 逻辑就位）
- 测试基础设施 `include/ptxsim/testing/warp_test_utils.h` 已存在

## Capabilities

### Modified Capabilities
- `warp-state-management`: 状态唯一权威化为 `warp_state.threads[i].is_active`；外部 API 行为兼容

## Reference

- **Master plan**：`docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` §Task 2
- **详细 tasks**：`openspec/changes/phase3-t2-1-active-mask-unify/tasks.md`
- **现有 14 步 plan**：`docs/superpowers/plans/2026-05-07-active-mask-consistency.md`（基本照搬，分 5 个 Task）
- **Explore 盘点报告**（会话上下文 bg_7169b052 16m43s）：8 sections A-H 含完整 file:line 引用
- **AGENTS.md DUAL STATE 不变量**：`src/ptxsim/core/AGENTS.md:49-60`
- **关键约束测试**：`tests/integration/barrier/test_post_barrier_two_halves.cpp`（BUG-POSTBARRIER-TWOHALVES regression test）
- **现有 J1-J10 测试**：`tests/unit/exec/test_active_mask_consistency.cpp`（234 行，**未启用**）
