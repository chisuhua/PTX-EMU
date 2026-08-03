# god-class-refactor-sm-context-phase3

## Why

`src/ptxsim/core/sm_context.cpp` 实测 862 行（原 roadmap 703 / 提案基线 965 → 当前 862）。Phase 1（reconvergence 循环去重）+ Phase 2（ADR-0020 cpptlm 注入提取）已 apply，但 CTA 调度 + SM 级 barrier 封装 + warp 生命周期仍未拆分，目标 `<250 行` 远未达成。C-2 是唯一恶化中的 debt（docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-2），剩余 Phase 3+ 必须重启。

原 change `2026-07-27-god-class-refactor-sm-context` archive 后 tasks.md 全勾选但代码未 apply，本 change 为续做，使用新名以避免与 archive 同名混淆。

## What Changes

- **拆分 CTA 调度逻辑** 至 `src/ptxsim/core/sm_cta_dispatch.{h,cpp}`（CTA 接收/释放/调度主循环）
- **拆分 SM 级 barrier 同步封装** 至 `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}`（封装 BarrierModule 跨 warp 协调）
- **拆分 warp 生命周期管理** 至 `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}`（warp 创建/退役/ActiveMask 维护）
- **行级随迁** `sm_context.cpp:379` `w->update_active_mask()` 调用及其前置注释 `:374`（lessons-learned §1 防漏）
- **保持** WarpContext public API 调用点零签名变更（C-2/C-18 边界）
- **保持** step_b no-op byte-identical fallback 4 分支测试锁定（lessons-learned §14）

依赖 C-18（refactor-warp-context）已落地（WarpContext API 已冻结，archive `2026-07-29-refactor-warp-context`）。

## Capabilities

### New Capabilities
- `sm-context-component-split`: SMContext 内部职责拆分为 CTA 调度 / warp 生命周期 / SM barrier 封装三个独立组件

### Modified Capabilities
- (无 spec-level 变更；纯实现级重构)

## Impact

- `src/ptxsim/core/sm_context.cpp` — 主入口从 862 行缩至 <250 行（净减 ~610 行）
- `src/ptxsim/core/sm_cta_dispatch.{h,cpp}` — 新增（CTA 调度）
- `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}` — 新增（SM barrier 封装）
- `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}` — 新增（warp 生命周期）
- `src/ptxsim/core/CMakeLists.txt` — 注册 3 个新源文件
- `tests/unit/sm/` — 新增组件单测
- 集成测试（`execute_warp_instruction` 路径）零回归（必须）

依赖 ADR-0006（SIMT v2.0）、ADR-0015、ADR-0018（C-18 边界）。
依赖 Skill: ptx-instruction-pipeline, ptx-lessons-learned, state-modification-audit。