# god-class-refactor-sm-context-phase3 - Tasks

## 1. Phase 0: 准备工作

- [ ] 1.1 MUST 验证 C-18 已落地：`ls openspec/changes/archive/*-refactor-warp-context` 存在
- [ ] 1.2 MUST 建立 baseline worktree（lessons-learned Checklist B）
- [ ] 1.3 MUST 记录基线：`wc -l src/ptxsim/core/sm_context.cpp` = 862
- [ ] 1.4 MUST 记录基线：`cd build && ctest --output-on-failure` 全绿
- [ ] 1.5 MUST 验证：`src/ptxsim/core/sm_context.cpp:379` `w->update_active_mask()` 存在
- [ ] 1.6 MUST 验证：`src/ptxsim/core/sm_context.cpp:374` 注释（active_count sync fix）完整

## 2. Phase 3: CTA 调度拆分（3h）

- [ ] 2.1 MUST 提取 CTA 接收/释放/调度主循环至 `src/ptxsim/core/sm_cta_dispatch.{h,cpp}`
- [ ] 2.2 MUST 行级随迁 `sm_context.cpp:379` `update_active_mask()` 调用（如该调用在 CTA 调度上下文）
- [ ] 2.3 MUST 保留 `sm_context.cpp:374` 注释
- [ ] 2.4 MUST NOT 改变 WarpContext public API 调用点
- [ ] 2.5 MUST NOT 改变 exe_once() 主循环签名
- [ ] 2.6 MUST 更新 `src/ptxsim/core/CMakeLists.txt` 注册新源文件
- [ ] 2.7 MUST 验证：`grep -c 'cta_dispatch' src/ptxsim/core/sm_context.cpp` 集中到 helper 调用点
- [ ] 2.8 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` 减少 ≥ 150 行
- [ ] 2.9 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 2.10 git commit -m "refactor(sm): extract CTA dispatch to sm_cta_dispatch"

## 3. Phase 4: warp 生命周期拆分（3h）

- [ ] 3.1 MUST 提取 warp 创建/退役/ActiveMask 维护至 `src/ptxsim/core/sm_warp_lifecycle.{h,cpp}`
- [ ] 3.2 MUST 集中所有 `set_active_mask` 调用点（grep 审计全列，逐行比对迁移前后）
- [ ] 3.3 MUST 行级随迁 sm_context.cpp:379 `update_active_mask()` 调用（如属于 warp lifecycle）
- [ ] 3.4 MUST NOT 改变 BarrierModule 内部实现
- [ ] 3.5 MUST 更新 `src/ptxsim/core/CMakeLists.txt`
- [ ] 3.6 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` 减少 ≥ 150 行
- [ ] 3.7 MUST 验证：warp 生命周期测试 + barrier 测试全绿
- [ ] 3.8 git commit -m "refactor(sm): extract warp lifecycle to sm_warp_lifecycle"

## 4. Phase 5: SM barrier 封装拆分（2h）

- [ ] 4.1 MUST 提取 SM 级 barrier 跨 warp 协调封装至 `src/ptxsim/core/sm_barrier_wrapper.{h,cpp}`
- [ ] 4.2 MUST 调用 BarrierModule public API（不引入新 Wbar struct，lessons-learned §14）
- [ ] 4.3 MUST NOT 改 BarrierModule 内部实现
- [ ] 4.4 MUST 更新 `src/ptxsim/core/CMakeLists.txt`
- [ ] 4.5 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` 减少 ≥ 100 行
- [ ] 4.6 MUST 验证：barrier 集成测试 + step_b 4 分支测试全绿
- [ ] 4.7 git commit -m "refactor(sm): extract SM barrier wrapper to sm_barrier_wrapper"

## 5. Phase 6: 最终验证（1h）

- [ ] 5.1 MUST 验证：`wc -l src/ptxsim/core/sm_context.cpp` < 250
- [ ] 5.2 MUST 验证：`sm_context.cpp:379` `update_active_mask()` 仍在（lessons-learned §1 关键项）
- [ ] 5.3 MUST 验证：`sm_context.cpp:374` 注释完整
- [ ] 5.4 MUST 验证：sm_context.cpp WarpContext public API 调用点零 diff
- [ ] 5.5 MUST 验证：`cd build && ctest --output-on-failure` 全绿
- [ ] 5.6 SHOULD 更新 `src/ptxsim/core/AGENTS.md` 文档化新子模块

## 6. 应用阶段

- [ ] 6.1 MUST 运行 `openspec validate god-class-refactor-sm-context-phase3 --strict`
- [ ] 6.2 MUST 通过所有验证后 archive

## 验收

- sm_context.cpp < 250 行（862 → <250）
- 新组件 ≤ 3 个 + 已有 cpptlm_inject 共 4 个
- step_b 4 分支测试 + barrier 测试 + sm 单测 + 集成测试 全绿
- 集成测试（execute_warp_instruction 路径）零回归
- 每个 Phase commit 独立可 revert
- ptx-lessons-learned §1, §14, Checklist B 全部勾选

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（lessons-learned SKILL.md:48-77）：sm_context.cpp:379 + :374 注释列入迁移清单
- MUST Checklist B（lessons-learned SKILL.md:474-483）：worktree + 3 Phase commit
- MUST §14 step_b no-op fallback 4 分支测试锁定（lessons-learned SKILL.md:409-455）
- MUST §2 递归锁审计（lessons-learned SKILL.md）：不在持锁方法内调用同锁 public 方法
- MUST NOT 改 exe_once() 签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名
- SHOULD 复用 BarrierModule API