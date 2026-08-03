## Context

承接 `2026-07-27-god-class-refactor-sm-context` archive 中的 Phase 0/1/2（已 apply，sm_context.cpp 从 965 → 862 行），重启 Phase 3+。当前 sm_context.cpp 实测 862 行，仍混合：CTA 调度、warp 生命周期、SM 级 barrier 封装三大职责。

C-18（refactor-warp-context）已落地（archive `2026-07-29-refactor-warp-context`），WarpContext public API 已冻结：`update_active_mask` / `check_reconvergence` / `get_simt_stack` / `get_lanes_by_pc` 签名稳定。

## Goals / Non-Goals

**Goals:**
- sm_context.cpp 缩至 `<250 行`（净减 ~610 行）
- 提取三个职责单一组件：`sm_cta_dispatch` / `sm_barrier_wrapper` / `sm_warp_lifecycle`
- 保持 WarpContext public API 调用点零签名变更
- 保持 step_b no-op byte-identical fallback 4 分支测试锁定（lessons-learned §14）
- 行级随迁 sm_context.cpp:379 `w->update_active_mask()` 及其注释 :374（lessons-learned §1）

**Non-Goals:**
- 改 WarpContext public API 签名（C-18 边界）
- 改 `exe_once()` 主循环签名
- 改 BarrierModule 内部实现
- 改 CTAContext 接口
- 引入新 Wbar struct（已全部迁移至 BarrierModule）

## Decisions

1. **拆分粒度（≤4 个组件）**：CTA 调度 / warp 生命周期 / SM barrier 封装 / ADR-0020 注入编排（已 Phase 2 apply，保留为 `sm_context_cpptlm_inject.{h,cpp}`）
   - Rationale: roadmap §1.2 C-2 设定 ≤4 组件上限；当前 sm_context.cpp 4 大职责正好对应
   - Alternatives considered: 2 个组件（CTA + 其他）→ 仍超 500 行；5+ 组件 → 超出 ≤4 上限

2. **行级 diff 迁移（lessons-learned §1）**：迁移 `update_active_mask()` 调用时同时迁移前置注释 `:374`（"only updated by update_active_mask(). Without this fix, active_count…"）
   - Rationale: §1 实证站点 sm_context.cpp:379 + 注释 :374 是"看似冗余但不可漏"的跨模块状态翻译证据
   - Alternatives considered: 仅迁移调用不迁移注释 → 违反 lessons-learned §1

3. **Phase 化独立 commit**：Phase 3 = CTA 调度；Phase 4 = warp 生命周期；Phase 5 = SM barrier 封装；每 Phase 独立可 revert
   - Rationale: lessons-learned Checklist B 强制 worktree + 分 Phase commit，单 Phase 回归即 revert 该 Phase
   - Alternatives considered: 单一大 commit → 不可 revert，违反 §3

4. **复用 BarrierModule API（不引入新 Wbar struct）**：SM barrier 封装组件直接调用 BarrierModule public API
   - Rationale: lessons-learned §14 + ADR-0015 已迁移 Wbar struct 使用至 BarrierModule
   - Alternatives considered: 引入新 Wbar struct → 违反 §14 迁移完成约束

## Risks / Trade-offs

- [Risk] Phase 3/4/5 任一 Phase 引入 active_count sync bug（lessons-learned §1）→ Mitigation: 每个 Phase commit 后跑 `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 分支测试 + `tests/integration/barrier/` 全套 + 行级 diff 审计 sm_context.cpp:379
- [Risk] Phase 4 warp lifecycle 拆分遗漏 exe_once() 路径调用 → Mitigation: grep `set_active_mask` 调用点全列，逐行比对迁移前后
- [Risk] BarrierModule 调用点重构引入死锁（lessons-learned §2 递归锁）→ Mitigation: 不在持锁方法内调用同锁 public 方法，所有调用点写前审计

## Migration Plan

1. **前置条件验证**：C-18 archive 已存在 + 当前 baseline worktree（lessons-learned Checklist B）
2. **Phase 3**: 提取 CTA 调度 → `sm_cta_dispatch.{h,cpp}`；ctest 全绿后 commit
3. **Phase 4**: 提取 warp 生命周期 → `sm_warp_lifecycle.{h,cpp}`；ctest 全绿后 commit
4. **Phase 5**: 提取 SM barrier 封装 → `sm_barrier_wrapper.{h,cpp}`；ctest 全绿后 commit
5. **最终验证**: sm_context.cpp < 250 行 + 集成测试零回归 + 更新 AGENTS.md
6. **Rollback**: 每 Phase commit 独立 revert（git revert HEAD~N..HEAD）

## Open Questions

- 无（设计已收敛于 Round-3 Oracle CONDITIONAL APPROVE 条件）