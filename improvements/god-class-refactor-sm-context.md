# god-class-refactor-sm-context

**优先级**: P2 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2
**阶段**: default | **分类**: arch-design
**类型**: refactor

## 架构依据

- `src/ptxsim/core/sm_context.cpp` 实测 **965 行**（roadmap 原始 703 行已过期，post-phase3-debt-roadmap.md:53；增长主因 ADR-0020 cpptlm 注入，commits `367fd6a5`/`a53508c2`/`5831623c`）
- 职责混合：CTA 调度、warp 生命周期、BarrierModule 集成、ADR-0020 三段式注入（`step_b_set_blocked_cycles` + 3 setter + 3-step 编排）
- **§1 跨模块状态翻译实证站点**：sm_context.cpp:379 `w->update_active_mask()`，注释 sm_context.cpp:374 明示 "only updated by update_active_mask(). Without this fix, active_count…" —— 典型 lessons-learned §1（.opencode/skills/ptx-lessons-learned/SKILL.md:48-77）"看似冗余但不可漏"行
- **C-2/C-18 边界实证**：sm_context.cpp:455-490 与 :580-623 存在两段**近乎逐行重复的 SIMT reconvergence 编排循环**（约 130 行：stack_depth 快照 → `check_reconvergence()` pop 循环 → `get_lanes_by_pc()` 汇聚 trace 输出），调用 WarpContext public API（`get_simt_stack` :461/:583、`check_reconvergence` :468/:590）

## 范围

- **In Scope**:
  - 拆分 sm_context.cpp 为 ≤ 4 个职责单一组件（CTA 调度 / warp 生命周期 / SM 级 barrier 封装 / ADR-0020 注入编排）
  - **提取并去重 sm_context.cpp:455-490 与 :580-623 两段 reconvergence 编排循环**为共享 helper（约 -120 行）——此编排层归 C-2，WarpContext 数据结构/API 归 C-18
  - ADR-0020 注入点代码归属决策（`step_b_set_blocked_cycles` + 3 setter + 3-step 编排）
- **Out Scope**:
  - **WarpContext public API 签名**（`update_active_mask` / `check_reconvergence` / `get_simt_stack` / `get_lanes_by_pc`）——由 C-18 冻结，sm_context.cpp:379/:461/:468/:583/:590 为消费方
  - 不修改 BarrierModule 内部实现；不改变 `exe_once()` 主循环签名；不动 CTAContext 接口

## 关键场景

- GIVEN reconvergence 循环提取为共享 helper, WHEN 任一 warp 指令执行后检查汇聚， THEN 两处调用点（:455-490、:580-623）trace 输出与栈行为逐字节一致
- GIVEN 拆分涉及 sm_context.cpp:379 区域， WHEN 行级 diff 比对， THEN `update_active_mask()` 调用及其前置注释（:374）**逐行随迁**（§1 防漏）
- GIVEN `step_b_set_blocked_cycles` 移动， WHEN 拆分， THEN 4 分支单元测试锁定随迁（tests/unit/sm/test_step_b_set_blocked_cycles.cpp, lessons-learned §14, SKILL.md:409-455）
- GIVEN C-18 先/后落地， WHEN 本 change apply, THEN sm_context 对 WarpContext 的调用点零签名变更（编译期保证）

## 技术约束

- MUST 遵循 lessons-learned §1 行级 diff（SKILL.md:48-77）：迁移函数逐行比对，重点站点 sm_context.cpp:379（active_count sync fix）
- MUST 遵循 Checklist B（SKILL.md:474-483）：开工前建立 baseline worktree；拆分按 Phase 独立 commit（建议 Phase1=去重 reconvergence 循环 → Phase2=ADR-0020 编排提取 → Phase3=CTA 调度/barrier 拆分），单 Phase 回归即 revert 该 Phase
- MUST 保持 `step_b` no-op byte-identical fallback 契约的 4 分支测试锁定（§14, SKILL.md:409-455）
- MUST NOT 改变 exe_once() 主循环签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名（C-18 边界）
- SHOULD 复用 BarrierModule API（不引入新 Wbar struct）

## 验收标准

- sm_context.cpp < 250 行（965 → <250，含 reconvergence 去重 -120 行）
- 新组件 ≤ 4 个；两段重复循环收敛为 1 个共享 helper（`grep -c "check_reconvergence" src/ptxsim/core/sm_context.cpp` 调用点 ≤ 2 且均经 helper）
- step_b 4 分支测试 + barrier 测试 + sm_context 单测全绿且无需修改
- 集成测试（execute_warp_instruction 路径）零回归；每个 Phase commit 独立可 revert

## Round-3 vs Round-1/2 deltas

- 基线维持 965 行；工时 8h → **10-12h**（增加去重 + Phase 化开销）
- 新增 MUST：§1 行级 diff（锚定 sm_context.cpp:379）+ Checklist B（worktree + 3 Phase commit 计划）
- 新增 C-2/C-18 边界声明：reconvergence 编排循环（:455-623）归 C-2；WarpContext API 归 C-18 冻结
- 新增 In Scope：两段重复循环去重（Round-1/2 未覆盖的实证发现）

## Oracle 评审链

- Round-1 (Oracle): APPROVE-WITH-CHANGES（行数基线过期 + 缺 §14）
- Round-2 (Oracle): NEEDS-MORE-CHANGES（缺 §1/Checklist B + 与 C-18 scope 重叠）
- Round-3 (Oracle): CONDITIONAL APPROVE（在 C-18 之后执行，3 Phase commit 计划为硬性门禁）

## 依赖关系

- **必须在 C-18 之后执行**：C-18 冻结 WarpContext public API 后，C-2 才能在稳定接口上做 SM 侧拆分（参见 refactor-warp-context §技术约束 MUST NOT）