# Phase 3 — Critical Debt Implementation Plan
# (DUAL STATE + GOD CLASS + CVT strategy pattern)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Plan Version**: v1
**Review Verdict**: PENDING (will be reviewed by Oracle/Momus after first iteration)
**Created**: 2026-06-23 (Oracle explore reports: bg_7169b052 16m43s + bg_ec7aaee3 8m50s)
**OpenSpec Changes**:
- `openspec/changes/phase3-t2-6-cvt-strategy-pattern/`（CVT 策略模式重构）
- `openspec/changes/phase3-t2-1-active-mask-unify/`（active_mask 三源统一）
- `openspec/changes/phase3-t2-3-god-class-split/`（god class POD 拆分）

**Goal:** 消除 DUAL STATE MECHANISM（active_mask 三源统一）+ ThreadContext/WarpContext god class 拆分（4+3 POD）+ CvtHandler 1288 行 1063 行 switch 策略模式重构（5+ 策略）。

**Architecture:** 严格 TDD，每个 Task 多个原子步骤（写测试→验失败→实现→验通过→提交）。Phase 3 是性能与可维护性关键路径，所有修改需 CI 拦截回归（Phase 1/2 已就位）。

**Tech Stack:** C++20, Catch2 amalgamation, ASan (per-commit verification), Catch2 + ctest

---

## 前置依赖（Phase 1 + 2 + 早期 T2 已完成）

| 任务 | 描述 | Commit | 状态 |
|------|------|--------|:---:|
| T0-1 | compile_commands.json 启用 | 5faf50b | ✅ |
| T0-2 | CI workflow 启用 | 7cb4378 | ✅ |
| T0-3 | baseline 存档 | 6274b1e | ✅ |
| T1-1 | Symtable → unique_ptr | c3d8fd3 | ✅ |
| T1-2 | cudaStream/Event handle 验证 | eeab07d | ✅ |
| T1-3 | BarWarpSyncHandler → BarrierModule | bcadf32, eb2195e | ✅（openspec integrate-barrier-module-cta-warp 仍活跃收尾）|
| T1-4 | membar/fence handler 真实现 | 95ffc23 | ✅ |
| T1-5 | 根 README 重写 | 04a62c4 | ✅ |
| T2-2 | ampere_a100.json 默认 | 41030d8 | ✅ |
| T2-4 Step 1 | usesAsyncStore/usesRedAsync 清理 | 2e339ea | ✅（Steps 2-7 待 multi-day 推进，本次不启动）|
| T2-5 | atom.global.exch.u32 tests | f4570c2 | ✅ |
| T2-7 | drop BarWarpSyncHandler friend | 7e667cd | ✅ |

---

## 关键约束（从 Phase 2 计划 + 两个 explore 报告提取）

> **DO NOT 改 `set_active_mask(0u)` 为 OR-merge**：RetHandler 依赖覆写语义（`call.cpp:29`），用 `set_active_mask(0u)` 清空所有 lane。T2-1 必须保留 `set_active_mask` 的"双模式"接口（覆写 vs OR-merge）— `warp_state.threads[i].is_active` 是覆写的 canonical 源，OR-merge 由 `BarrierModule::release_warp_barrier` 内部承担（`barrier_module.cpp:107-112` 已实现）。

> **DO NOT 删除 `WarpState.wbars[]` + `current_wbar_id` 直到 integrate-barrier-module-cta-warp 合并**：`barrier.cpp:161,215` + `warp_context.cpp:287` 仍使用，3 处必须先迁移。T2-1 仅标记 `[[deprecated]]`，T2-3 才物理删除。

> **DO NOT 拆 CvtHandler 类**：X-Macro 在 `instruction_factory.cpp:14-17` 实例化 `new CvtHandler()`，多 Handler 会改变注册机制。T2-6 必须用 **Composition over Inheritance** — 策略在 CvtHandler 内部。

> **DO NOT 改 `CvtHandler::processOperation` 签名**：所有指令 handler 通过 X-Macro 统一注册，签名变会引发链接错误。

> **DO NOT 在 T2-1 动 `ThreadContext::state`（EXE_STATE）**：与 `warp_state.threads[i].status` 双源问题留给 T2-3 合并。

> **DO NOT 在 T2-1 删 `active_mask[]`**：RetHandler 依赖覆写语义（`call.cpp:29`），T2-1 仅在 `is_lane_active()` 委托 `is_lane_schedulable()`（plan 2026-05-07 Task 2 已规划），不删字段。T2-3 才删。

> **MUST 复用 `half_utils.h`**：删除 `arithmetic_conversion.cpp` 内 `half_to_float`/`float_to_half`（line 25-58, 72-139），改用 `f16_to_f32`/`f32_to_f16`（`include/ptxsim/utils/half_utils.h`）。先做 1-2 个边界 case 对比测试，验证精度一致。

> **MUST 先修 P1-4.1 bug**：`tests/integration/ptx/test_cvt.cpp:142-145` 注释"KOWN ISSUE: P1-4.1 - CvtHandler does not write r2 in f32→s32 and f64→s64 paths"。T2-6 Step 5（tests）必须先修 P1-4.1，否则 f32→int 路径全部 SKIP，无法验证。

> **MUST 走 AGENTS.md §TDD 流程**：每步先写测试，确认失败，再实现，再确认通过，最后 commit。`./scripts/sanity.sh --quick` 必跑。

> **MUST 提交到 main**（per user 2026-06-23 选择）：跟 T1-1..T1-5 一致，不开 worktree。每个 Task 一个独立 PR（commit message 引用 `T2-6`/`T2-1`/`T2-3`）。

---

## Task 1: T2-6 — CVT 策略模式重构

**Files（最终目标结构）：**
- New: `include/ptxsim/instructions/cvt/cvt_helpers.h` + `.cpp`（4 个 helper 抽离）
- New: `include/ptxsim/instructions/cvt/cvt_strategy.h`（`ConversionStrategy` + `CvtContext` + `select_strategy()`）
- New: `include/ptxsim/instructions/cvt/{cvt_int_to_int,cvt_int_to_float,cvt_float_to_int,cvt_float_to_float,cvt_rounding,cvt_saturation}.h`（5+ 策略声明）
- New: `src/ptxsim/instructions/cvt/*.cpp`（策略实现）
- Modify: `src/ptxsim/instructions/arithmetic_conversion.cpp`（1288 → ~250 行，主 switch 收缩）
- Modify: `src/CMakeLists.txt:105`（注册新文件）
- Modify: `tests/integration/CMakeLists.txt:247`（注册新测试）
- New tests: `tests/integration/ptx/test_cvt_*.cpp`（94 个新 TEST_CASE）
- New: `tests/unit/ptx/test_cvt_helpers.cpp`（helpers 单元测试）

**核心问题：**
- `CvtHandler::processOperation` 单方法 1145 行（line 141-1286）
- 嵌套 4 层深 switch：`dst_bytes` (1/2/4/8) → `is_float` → `is_signed`/`signed` ↔ `src_bytes` (1/2/4/8)
- 4 个 case 间 int→int 4×4×4 嵌套逻辑几乎相同模板
- `half_to_float`/`float_to_half` 与 `half_utils.h` 重复 ~70 行

**RISK**：🟢 低（纯重构，零外部破坏面 — `CvtHandler` 无外部直接调用，全部经 `InstructionFactory::get_handler(S_CVT)`）

**详细 tasks**：见 `openspec/changes/phase3-t2-6-cvt-strategy-pattern/tasks.md`

---

## Task 2: T2-1 — active_mask 三源统一

**Files:**
- Modify: `src/ptxsim/core/warp_context.cpp:210-222`（`update_active_mask()` 改为统一从 `warp_state.threads[i]` 读取）
- Modify: `include/ptxsim/warp_context.h:124-126`（`is_lane_active()` 委托 `is_lane_schedulable()`，消除 ISSUE-005）
- Modify: `src/ptxsim/core/thread_context.cpp:735-742`（`sync_to_warp_state(RUN)` case 补 `is_active=true`）
- Modify: `src/ptxsim/core/AGENTS.md:49-60`（重写"DUAL STATE MECHANISM"段 → "SINGLE SOURCE OF TRUTH"）
- Modify: `include/ptxsim/warp_state.h`（`wbars[]` + `current_wbar_id` 标 `[[deprecated]]`，**不删**）
- Verify: `tests/unit/exec/test_active_mask_consistency.cpp`（J1-J10 启用 + 全过）
- Verify: `tests/integration/divergence/test_post_barrier_divergence.cpp`（不回归）
- Verify: `tests/unit/barrier/test_post_barrier_two_halves.cpp`（不回归）

**核心问题：**
- `active_mask[]`（WarpContext, 9 个核心读写点）/ `warp_state.threads[i].is_active`（WarpState, ~30 写入点）/ `warp_state.exec_mask`（8 写入点）三路状态
- `update_active_mask()` 4 个判断条件（`is_active` + `!is_exited` + `!is_blocked` + `status==Active`）有重叠
- `set_active_mask(int,bool)` + `set_active_mask(uint32_t)` + `reset()` 是 3 个违规直接写 `active_mask[]` 的点（虽然内部分别同步了 `is_active`）

**RISK**：🔴 高（涉及 BUG-POSTBARRIER-TWOHALVES / BUG-RETHANG 已知问题）

**前置依赖**：✅ `integrate-barrier-module-cta-warp` 必须合并（OR-merge 在 `barrier_module.cpp` 内置后才能安全统一）

**详细 tasks**：见 `openspec/changes/phase3-t2-1-active-mask-unify/tasks.md`（基于 `docs/superpowers/plans/2026-05-07-active-mask-consistency.md`）

---

## Task 3: T2-3 — god class POD 拆分

**Files（最终目标结构）：**
- New: `include/ptxsim/contexts/exec_state.h`（POD: state, bar_id, warp_id_, lane_id_, BlockIdx/ThreadIdx/GridDim/BlockDim）
- New: `include/ptxsim/contexts/register_predicate.h`（POD: register_bank_manager_, cc_reg, operand_collected, operand_is_immediate_, vecOp_phy_addrs）
- New: `include/ptxsim/contexts/memory_ref.h`（POD: shared_mem_space, local_mem_space, warp_context_, cta_context_）
- New: `include/ptxsim/contexts/program_ref.h`（POD: statements, name2Sym, name2Share, label2pc, call_stack, dst_operand_reg_name_）
- New: `include/ptxsim/contexts/lane_mask.h`（POD: active_mask, warp_thread_ids, active_count, divergence_detected, is_scheduled_）
- New: `include/ptxsim/contexts/warp_identity.h`（POD: warp_id, physical_warp_id, physical_block_id, pc）
- New: `include/ptxsim/contexts/backend_links.h`（POD: register_bank_manager_, sm_context_, cta_context_, simt_stack, threads）
- Modify: `include/ptxsim/thread_context.h`（300 → ~150 行，4 POD facade）
- Modify: `include/ptxsim/warp_context.h`（274 → ~150 行，3-4 POD facade）
- Modify: `include/ptxsim/warp_state.h`（71 → ~30 行，精简到 2 字段 `threads[]` + `exec_mask`）
- Modify: `src/ptxsim/core/thread_context.cpp`（848 行，按 POD 重排方法）
- Modify: `src/ptxsim/core/warp_context.cpp`（487 → ~250 行）
- Modify: `include/ptxsim/testing/warp_test_utils.h`（207 行，重写以匹配新 API）
- Update: 69 个测试文件（`grep -rln "WarpContext\|ThreadContext" tests/`）

**核心问题：**
- ThreadContext 26 public 数据字段 + 2 private = 28（Errata §1.1 校正后"81"含方法/常量）
- WarpContext 32 数据字段
- WarpState 6 字段（3 个死代码：thread_predicates 0 引用、wbars+current_wbar_id deprecated、warp_pc 0 引用）
- 4 重 writes 直接覆盖 `active_mask[]` 绕过 `update_active_mask()`
- 测试基础设施（`include/ptxsim/testing/warp_test_utils.h`）紧耦合

**RISK**：🔴🔴 HIGHEST（影响面最大：所有指令执行路径，调度/屏障/寄存器读写/PC 同步全链路）

**前置依赖**：✅ T1-1（unique_ptr 迁移）/ ✅ T2-1（active_mask 统一）/ ✅ `integrate-barrier-module-cta-warp` 合并（删 wbars[]）

**详细 tasks**：见 `openspec/changes/phase3-t2-3-god-class-split/tasks.md`

---

## Phase 3 完成门禁

```yaml
phase_3_complete:
  cvt_refactor:
    - [ ] Task 1 (T2-6):
        - [ ] cvt_helpers.h/cpp 抽离 + 复用 half_utils.h
        - [ ] CvtContext + select_strategy() 骨架
        - [ ] 5 个具体策略（IntToInt, IntToFloat, FloatToFloat, FloatToInt, RoundingMode）
        - [ ] CvtHandler::processOperation 收缩到 ~50 行
        - [ ] 94 个新 integration tests 全过
        - [ ] P1-4.1 bug 修复（f32→int 写 r2）
        - [ ] ctest -L "ptx;cvt" 全过 + sanity.sh --quick 全过

  active_mask_unify:
    - [ ] Task 2 (T2-1):
        - [ ] update_active_mask() 改为统一从 warp_state 读取
        - [ ] is_lane_active() 委托 is_lane_schedulable()
        - [ ] sync_to_warp_state(RUN) 补 is_active=true
        - [ ] WarpState.wbars[] 标记 [[deprecated]]（不删）
        - [ ] J1-J10 测试全过
        - [ ] test_post_barrier_divergence + test_post_barrier_two_halves 不回归
        - [ ] src/ptxsim/core/AGENTS.md §DUAL STATE 改写

  god_class_split:
    - [ ] Task 3 (T2-3):
        - [ ] ThreadContext 拆 4 POD (ExecState/RegisterPredicate/Memory/ProgramRef)
        - [ ] WarpContext 拆 3-4 POD (LaneMask/Identity/BarrierControl/BackendLinks)
        - [ ] WarpState 精简到 2 字段（threads[] + exec_mask）
        - [ ] 删除 deprecated wbars + current_wbar_id + thread_predicates + warp_pc
        - [ ] 69 个测试文件更新通过
        - [ ] ctest 全过 + ASan 跑 e2e_barrier_warp_sync 无泄漏
```

**Phase 3 完成后可启动 Phase 4 T2-4**（PTX 8.7+ ANTLR 占位清理，multi-day）。

---

## 风险与缓解（汇总）

| # | 风险 | 概率 | 影响 | 缓解 |
|---|------|:---:|:---:|------|
| 1.1 | `set_active_mask` OR-merge 破坏 RetHandler 覆写语义 | 🟡 中 | 🔴 高 | T2-1 锁定 set_active_mask(0u) 语义（专门写测试覆盖 `call.cpp:29`）|
| 1.2 | wbars[] 删除时机不当触发 BUG-POSTBARRIER-TWOHALVES | 🟡 中 | 🔴 高 | 等 integrate-barrier-module-cta-warp 合并；T2-1 仅 `[[deprecated]]` |
| 1.3 | 拆分后 SIMT stack 行为不匹配 | 🟢 低 | 🔴 高 | 保留 ctest barrier/simt 完整覆盖；T2-3 步骤 5 必跑 |
| 2.1 | strategy 模式误用继承而非组合 | 🟢 低 | 🟡 中 | 强制 Composition 模式（X-Macro 约束 `instruction_factory.cpp:14-17`）|
| 2.2 | 94 个新测试因 P1-4.1 全部 SKIP | 🔴 高 | 🟡 中 | T2-6 Step 5 先修 P1-4.1 bug，再补测试 |
| 2.3 | half_utils.h 复用精度差异 | 🟡 中 | 🟡 中 | 对比测试 + 1-2 个边界 case 验证（NaN/Inf/denormal）|
| 3.1 | 69 个测试文件更新遗漏 | 🟡 中 | 🟡 中 | `grep -rln` 列举全部 + 自动化 cmake 编译验证 |
| 3.2 | ASan 暴露 god class 拆分的隐藏泄漏 | 🟡 中 | 🟡 中 | Phase 2 ASan baseline 已存档可对比 |

---

## 自检清单（每个 Task 启动前核对）

- [ ] Phase 1+2 完成门禁全部勾选
- [ ] CI workflow 在最近一次 commit 触发成功
- [ ] baseline-2026-06-21.log 存档可用
- [ ] integrate-barrier-module-cta-warp 已合并（Task 2 启动前）
- [ ] P1-4.1 bug 修复（Task 1 Step 5 启动前）
- [ ] `include/ptxsim/utils/half_utils.h` 边界 case 对比测试通过（Task 1 Step 1 启动前）

---

## 与 Phase 2/4 的衔接

- **承接 Phase 2**：T1-1..T1-5 + T2-2, T2-4 Step 1, T2-5, T2-7 完成
- **承接 active openspec change**：integrate-barrier-module-cta-warp 合并后启动 T2-1
- **交付给 Phase 4**：T2-4 Steps 2-7（PTX 8.7+ 占位清理）— multi-day，独立推进
- **不再回退**：T2-1 的 `update_active_mask()` 统一数据源是不可逆的；T2-3 物理删除 `wbars[]`/`current_wbar_id`/`thread_predicates`/`warp_pc` 是不可逆的

---

## 工时估算

| Task | 来源 | 估算 | 调整因子 |
|------|------|------|---------|
| T2-6 | 审计 P2-1 = 3 天 | **3.5 天** | +0.5（94 个新测试 + P1-4.1 协同）|
| T2-1 | 含在 P2-2 内 | **3-5 天** | active_mask 一致性 plan 14 步已列 |
| T2-3 | 审计 P2-2 = 2 周 | **7-10 天** | T1-1 + T2-1 已铺路；T2-3 拆 7-8 POD |
| **合计** | | **13.5-18.5 天** | |

**推荐执行顺序**（与 explore 报告 G/H 一致）：
1. T2-6 先做（3.5 天，最低风险，可立即启动）
2. T2-1 后做（3-5 天，需等 integrate-barrier-module-cta-warp 合并）
3. T2-3 最后（7-10 天，依赖 T2-1）

---

## 与 TDD 三阶段的对应

按 AGENTS.md §TDD 流程，每个 Task 内部遵循：
1. **测试先行**：写失败的测试（`./scripts/sanity.sh --quick` 确认失败）
2. **实现代码**：写通过测试的实现
3. **验证**：跑完整 sanity 检查无回归

每步用 checkbox `- [ ]` 跟踪（具体步骤见各 openspec change 的 `tasks.md`）。

---

**最后更新**: 2026-06-23
**维护者**: Sisyphus orchestrator
**版本**: v1
