# Phase 3 T2-3: ThreadContext / WarpContext god class POD 拆分

## Why

ThreadContext 26 public 数据字段 + 2 private（Errata §1.1 校正后"81"含方法/常量）/ WarpContext 32 数据字段 → 测试无法 mock（god class）→ `test_active_mask_consistency` 等核心测试稀薄（审计 §A1 关联 1）。WarpState 6 字段中 3 个是死代码（`thread_predicates` 0 引用、`wbars`+`current_wbar_id` 已 deprecated、`warp_pc` 0 引用）。

T2-3 是 T2-1 的下游：拆分后状态机必须先收敛到 single source of truth（POD 边界依赖统一数据源）。审计 P2-2 估计 2 周，本计划评估 7-10 天（T1-1 + T2-1 已铺路）。

## What Changes

- **拆 4 个 POD for ThreadContext**：
  - `ExecStatePod`：`state`, `bar_id`, `warp_id_`, `lane_id_`, `BlockIdx/ThreadIdx/GridDim/BlockDim`
  - `RegisterPredicatePod`：`register_bank_manager_`, `cc_reg`, `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`
  - `MemoryPod`：`shared_mem_space`, `local_mem_space`, `warp_context_`, `cta_context_`
  - `ProgramRefPod`：`statements`, `name2Sym`, `name2Share`, `label2pc`, `call_stack`, `dst_operand_reg_name_`
- **拆 3-4 个 POD for WarpContext**：
  - `LaneMaskPod`：`active_mask`, `warp_thread_ids`, `active_count`, `divergence_detected`, `is_scheduled_`
  - `IdentityPod`：`warp_id`, `physical_warp_id`, `physical_block_id`, `pc`（pc 标 deprecated 并删除）
  - `BarrierControlPod`：合并到 LaneMaskPod（WarpState 拆分后 wbars 不在 WarpContext 范围）
  - `BackendLinksPod`：`register_bank_manager_`, `sm_context_`, `cta_context_`, `simt_stack`, `threads`
- **WarpState 精简**：6 字段 → 2 字段（`threads[]` + `exec_mask`）
- **删除 deprecated**：`wbars`, `current_wbar_id`, `thread_predicates`, `warp_pc`
- **Facade 模式**：`ThreadContext` + `WarpContext` 改为持有 POD 的 facade（保持外部 API 兼容）
- **重写测试基础设施**：`include/ptxsim/testing/warp_test_utils.h`（207 行）+ `scheduler_utils.h`（72 行）+ `assertion_utils.h` 适配新 API
- **更新 69 个测试文件**（`grep -rln "WarpContext\|ThreadContext" tests/`）

## Out of Scope

- active_mask 三源统一（独立 change phase3-t2-1-active-mask-unify）
- CVT 策略模式重构（独立 change phase3-t2-6-cvt-strategy-pattern）
- WMMA / Tensor Core 相关字段
- `ThreadContext::state` (EXE_STATE) 与 `warp_state.threads[i].status` 实际合并（**T2-3 物理删除 EXE_STATE，统一到 status**，但有 breaking change 风险，需回归测试覆盖）

## Pre-conditions

- ✅ T1-1（Symtable → unique_ptr，提供 unique_ptr 基础）
- ✅ T2-1（active_mask 统一，**必须先完成**）
- ✅ `integrate-barrier-module-cta-warp` 已合并（**必须** — 否则删 wbars 触发 BUG-POSTBARRIER-TWOHALVES）

## Capabilities

### Modified Capabilities
- `thread-context-management`: ThreadContext 改为 4-POD facade
- `warp-context-management`: WarpContext 改为 3-4-POD facade
- `warp-state-management`: WarpState 精简到 2 字段

### Breaking Changes
- 内部 API：`ThreadContext::state` 字段移除（统一到 `warp_state.threads[i].status`）
- 内部 API：`WarpState.wbars` / `current_wbar_id` / `thread_predicates` / `warp_pc` 字段移除
- 外部 API：**保持兼容**（facade 模式 + 转发方法）

## Reference

- **Master plan**：`docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` §Task 3
- **详细 tasks**：`openspec/changes/phase3-t2-3-god-class-split/tasks.md`
- **Explore 盘点报告**（会话上下文 bg_7169b052 16m43s）：8 sections A-H 含完整 file:line 引用
- **审计原文**：`docs/audits/HEALTH-AUDIT-2026-06-21.md:32,62,103,104,106,519,588,629`
- **AGENTS.md DUAL STATE**：`src/ptxsim/core/AGENTS.md:49-60`（T2-1 重写后）

---

## Phase A 工期估算（核心拆分）

| 步骤 | 内容 | 天数 |
|------|------|------|
| A1 | 创建 7-8 个 POD 头文件 + 占位实现 | 1.5 |
| A2 | WarpState 精简（6→2 字段） | 0.5 |
| A3 | ThreadContext 改 facade | 1.5 |
| A4 | WarpContext 改 facade | 1.5 |
| A5 | 物理删除 EXE_STATE / wbars / thread_predicates / warp_pc | 1.0 |
| **A 合计** | | **6.0** |

## Phase B 工期估算（测试更新）

| 步骤 | 内容 | 天数 |
|------|------|------|
| B1 | 重写 warp_test_utils.h + scheduler_utils.h + assertion_utils.h | 1.0 |
| B2 | 更新 69 个测试文件（grep 列表 + 自动化） | 1.5 |
| **B 合计** | | **2.5** |

## Phase C 工期估算（验证）

| 步骤 | 内容 | 天数 |
|------|------|------|
| C1 | 全量 ctest + sanity.sh 验证 | 0.5 |
| C2 | ASan 跑 e2e_barrier_warp_sync 无泄漏 | 0.3 |
| C3 | 性能对比（callgrind 前后 step_warp 单步开销） | 0.2 |
| **C 合计** | | **1.0** |

**总工期**: 6 + 2.5 + 1 = **9.5 天**（vs 审计 2 周，-25%）
