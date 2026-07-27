# god-class-refactor-sm-context — Design

## Context

**当前状态**：`src/ptxsim/core/sm_context.cpp` 实测 965 行（roadmap 703 行已过期）。增长主因 ADR-0020 cpptlm 注入（commits `367fd6a5`/`a53508c2`/`5831623c`，~260 行）。

**职责混合**：
1. CTA 调度（创建/销毁/状态机）
2. warp 生命周期（注册/注销）
3. BarrierModule 集成（SM 级 barrier 同步）
4. ADR-0020 注入编排（step_b_set_blocked_cycles + 3 setter + 3-step）

**Oracle Round-3 实证发现**：
- §1 站点：`sm_context.cpp:379` `w->update_active_mask()`（注释 :374 明示 "active_count sync fix"）
- 重复循环：`sm_context.cpp:455-490` 与 `:580-623` 两段近乎逐行的 reconvergence 编排循环

**API 消费方**：sm_context.cpp:379/:461/:468/:583/:590 5 站点依赖 `WarpContext` public API（已由 `refactor-warp-context` 冻结）。

## Goals / Non-Goals

**Goals:**
- 拆分 sm_context.cpp 为 ≤ 4 组件
- 提取共享 reconvergence helper（去重 130 行）
- ADR-0020 注入代码独立模块化
- 保持 step_b no-op byte-identical fallback 契约
- 主文件 < 250 行

**Non-Goals:**
- 改变 exe_once() 主循环签名
- 修改 BarrierModule 内部实现
- 修改 WarpContext public API（C-18 边界）
- 性能优化
- 重构 CTAContext 接口

## Decisions

### 决策 1: 拆分顺序（Dedup → Inject → Split）

**选择**：先去重 reconvergence 循环（最小改动，验证 C-18 API 冻结），再 ADR-0020 注入，最后 CTA/SM barrier 拆分

**理由**：
- Phase 1（去重）验证 C-18 冻结的 API 兼容
- Phase 2（注入提取）独立性强，与 step_b 4 分支测试同模块
- Phase 3（拆分）依赖前两 Phase 稳定

**替代方案**：
- A. 一次性大拆分 → 风险高
- B. 倒序 → 一样可达目标
- C. **采用**：3 Phase 顺序（Checklist B）

### 决策 2: reconvergence helper 接口

**选择**：独立 `sm_context_reconvergence.{h,cpp}`，接受 WarpContext 引用 + trace 输出流

**理由**：
- 与 WarpContext API 冻结解耦（接受引用，不修改 WarpContext）
- 现有两段循环结构近似，helper 易设计
- 编译期可立即验证（sm_context.cpp 5 站点零 diff）

**替代方案**：
- A. 模板化 helper → 编译复杂度上升
- B. 复制保留两段 → 失去去重意义
- C. **采用**：独立模块

### 决策 3: ADR-0020 注入代码归属

**选择**：`step_b_set_blocked_cycles` + 3 setter + 3-step 编排独立为 `sm_context_cpptlm_inject.{h,cpp}`

**理由**：
- ADR-0020 注入是高 churn 部分，独立模块便于 review
- step_b 4 分支测试同模块管理
- 编译期 step_b 契约验证

**替代方案**：
- A. 保留在 sm_context.cpp → 未拆分
- B. 拆为多个文件 → 过度工程
- C. **采用**：单模块

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| sm_context.cpp:379 漏迁移 | active_count 同步失败 | MUST §1 行级 diff + 注释保留 |
| 共享 helper 行为偏移 | 汇聚 trace 输出不一致 | MUST 现有 5 站点测试 + 集成测试 |
| step_b fallback 契约丢失 | 字节级偏移 | MUST 4 分支测试随迁（§14）|
| WarpContext API 签名变更 | sm_context.cpp 编译失败 | MUST NOT 改（受 C-18 冻结约束）|
| 3 Phase 相互影响 | 单 Phase 回归难定位 | MUST 独立 commit + revert |

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/ptxsim/core/sm_context.cpp` | 修改 | 主文件，965 → < 250 行 |
| `src/ptxsim/core/sm_context.h` | 修改 | 前向声明 |
| `src/ptxsim/core/sm_context_reconvergence.{h,cpp}` | 新增 | 共享 reconvergence helper |
| `src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}` | 新增 | ADR-0020 注入编排 |
| `src/ptxsim/core/sm_context_cta.{h,cpp}` | 新增 | CTA 调度 |
| `src/ptxsim/core/sm_context_barrier.{h,cpp}` | 新增 | SM barrier 封装 |
| `src/ptxsim/core/CMakeLists.txt` | 修改 | 添加 8 个新源 |

**不变范围**：
- `warp_context.cpp`（C-18 冻结）
- `BarrierModule` 内部实现
- `exe_once()` 签名
- `CTAContext` 接口
- `ThreadContext` 行为

## Migration Plan

### 部署步骤（Checklist B 3 Phase commit）

**Phase 1 (3h)**: 去重 reconvergence 循环
- 提取 sm_context.cpp:455-490 + :580-623 至 `sm_context_reconvergence.{h,cpp}`
- 行级保留 sm_context.cpp:379 update_active_mask() 调用
- 验证：grep -c 'check_reconvergence' sm_context.cpp ≤ 2
- 验证：ctest 全绿
- git commit: "refactor(sm): dedup reconvergence orchestration loops to shared helper"

**Phase 2 (3h)**: ADR-0020 注入代码提取
- 提取 step_b_set_blocked_cycles + 3 setter + 3-step 编排
- 新建 `sm_context_cpptlm_inject.{h,cpp}`
- 验证：test_step_b_set_blocked_cycles 4 分支测试全绿
- git commit: "refactor(sm): extract ADR-0020 cpptlm injection code"

**Phase 3 (4h)**: CTA 调度 + SM barrier 拆分
- 提取 CTA 调度逻辑 → `sm_context_cta.{h,cpp}`
- 提取 SM barrier 同步 → `sm_context_barrier.{h,cpp}`
- 验证：sm_context.cpp < 250 行
- 验证：exe_once() 签名不变
- git commit: "refactor(sm): split CTA scheduling + SM barrier into separate modules"

**Phase 4 (1h)**: 最终验证
- wc -l sm_context.cpp < 250
- 4 处 WarpContext API 调用点零 diff
- ctest --output-on-failure 全绿

### 回滚策略

- 每个 Phase 独立可 revert
- 任何 Phase 失败立即 `git revert HEAD`

## Open Questions

1. **是否更新 src/ptxsim/core/AGENTS.md？**
   - 推荐：YES
   - 决定：作为 Phase 4 验收可选步骤

2. **WarpContext API 冻结后，C-2 跨文件改动的 merge 策略？**
   - 推荐：feature branch + C-18 落地后合入 main
   - 决定：作为 ship 阶段决策

## 关联文档

- `improvements/god-class-refactor-sm-context.md`：完整 5 段提案
- `docs/adr/ADR-0020-cpptlm-injection-points.md`
- `openspec/changes/refactor-warp-context/`：C-18 依赖
- `.opencode/skills/ptx-lessons-learned/SKILL.md`：§1, §14, Checklist B
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2`：原债务条目