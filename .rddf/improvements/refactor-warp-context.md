# refactor-warp-context

**优先级**: P2 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-18
**阶段**: default | **分类**: arch-design
**类型**: refactor

## 架构依据

- `src/ptxsim/core/warp_context.cpp` 实测 **558 行**（roadmap 原始 537 行已过期，post-phase3-debt-roadmap.md:65）；最近 30 commits 触碰 6 次（高 churn）
- 职责混合：execute_warp_instruction 分发、active mask 管理、BarrierModule 同步、SIMT 编排（simt_stack **数据结构**已抽离至 `src/ptxsim/core/simt_stack.cpp`，残留为编排逻辑，warp_context.cpp:64-143）
- **§1 跨模块状态翻译实证站点**：warp_context.cpp:337 / :345 / :370 / :375 共 4 处 `thread->sync_to_warp_state()` 调用——lessons-learned §1（.opencode/skills/ptx-lessons-learned/SKILL.md:48-77）"下一模块的 sync_to_warp_state() 才翻译为 is_blocked"的直接实例，迁移时漏任一处即调度器死循环（失败模式速查表）
- **API 消费方实证**：sm_context.cpp:379 调 `update_active_mask()`、:468/:590 调 `check_reconvergence()`、:461/:583 调 `get_simt_stack()` —— WarpContext public API 签名是本 change 与 C-2 的共享边界

## 范围

- **In Scope**:
  - 提取指令分发为策略表
  - 提取 active mask 操作为 helper（含 set_active_mask overwrite 语义锁定）
  - 提取 SIMT **编排逻辑**（divergence/reconvergence orchestration，warp_context.cpp:64-143），不重新抽离数据结构
  - 拆分 warp_context.cpp 为 ≤ 3 个组件
- **Out Scope**:
  - **WarpContext public API 签名变更**（`update_active_mask`/`check_reconvergence`/`get_simt_stack`/`get_lanes_by_pc`）——消费方 sm_context.cpp:379/:461/:468/:583/:590，C-2 并行依赖
  - sm_context.cpp:455-623 的 reconvergence **编排调用方循环**去重（归 C-2）
  - 不修改 ThreadContext 行为；不重写 execute_warp_instruction 主循环；不修改 BarrierModule 内部；不重新提取 simt_stack 数据结构

## 关键场景

- GIVEN 任何函数跨文件迁移， WHEN 行级 diff, THEN 4 处 `sync_to_warp_state()`（:337/:345/:370/:375）逐行随迁且调用顺序不变——缺失任一处 THEN 调度器死循环（§1）
- GIVEN active mask helper 提取， WHEN set_active_mask, THEN ret handler 仍依赖 overwrite 语义（失败模式速查表"分歧场景一半 lanes 卡住 → 检查 set_active_mask 是否 OR 而非 overwrite" + AGENTS.md ANTI-PATTERNS）
- GIVEN 拆分后， WHEN sm_context.cpp 编译， THEN :379/:461/:468/:583/:590 调用点零修改（API 签名冻结，编译期验证）
- GIVEN SIMT 编排提取， WHEN 分歧/汇聚， THEN SIMT stack 行为不变（ADR-0006 + ADR-0014）

## 技术约束

- MUST 遵循 lessons-learned §1 行级 diff（SKILL.md:48-77）：4 处 `sync_to_warp_state()` 站点（warp_context.cpp:337/:345/:370/:375）列入迁移清单逐项核对
- MUST 遵循 Checklist B（SKILL.md:474-483）：baseline worktree + 分 Phase commit（建议 Phase1=active mask helper → Phase2=SIMT 编排提取 → Phase3=分发策略表），单 Phase 回归即 revert
- MUST 保持 set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS；非 §2，§2 为递归锁）
- MUST NOT 改变 WarpContext public API 签名（消费方：sm_context.cpp:379/:461/:468/:583/:590）
- MUST NOT 修改 ret handler 行为；MUST NOT 改变 execute_warp_instruction 主入口
- SHOULD 复用 BarrierModule API 与既有 simt_stack 数据结构

## 验收标准

- warp_context.cpp < 300 行（558 → <300）；新组件 ≤ 3 个
- `grep -c "sync_to_warp_state" src/ptxsim/core/warp_context*.cpp`（含新组件）合计 ≥ 4，且 4 个原始语义站点均有对应行
- sm_context.cpp 零 diff 编译通过（API 冻结的直接证据）
- barrier/active_mask/ret handler 测试全绿；test-coverage-enforcer 验证集成测试驱动 execute_warp_instruction 路径
- ptx-lessons-learned Checklist B 全部勾选（worktree 建立记录 + Phase commit 序列）

## Round-3 vs Round-1/2 deltas

- 基线维持 558 行；工时 4h → **6h**（§1 迁移清单 + Phase 化开销）
- 新增 MUST：§1 sync_to_warp_state 4 站点（:337/:345/:370/:375）行级保留 + Checklist B
- 新增 MUST NOT：WarpContext public API 签名冻结（消费方 sm_context.cpp:379/:468/:590 等，与 C-2 共享边界）
- 新增 Out Scope：reconvergence 调用方循环去重明确划归 C-2，消除范围重叠

## Oracle 评审链

- Round-1 (Oracle): APPROVE-WITH-CHANGES（§2 引用错误）
- Round-2 (Oracle): NEEDS-MORE-CHANGES（缺 §1/Checklist B + WarpContext API 边界声明）
- Round-3 (Oracle): APPROVE（在 C-2 之前落地，冻结 API 后 C-2 的去重有稳定基座）

## 依赖关系

- **必须在 C-2 之前执行**：本 change 冻结 WarpContext public API 后，C-2 才能在稳定接口上做 SM 侧去重（参见 god-class-refactor-sm-context §依赖关系）