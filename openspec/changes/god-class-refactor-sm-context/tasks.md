# god-class-refactor-sm-context — Tasks

## 1. Phase 0: 准备工作

- [x] 1.1 MUST 验证 C-18（refactor-warp-context）已落地（WarpContext API 冻结）
- [x] 1.2 MUST 建立 baseline worktree（Checklist B）
- [x] 1.3 MUST 记录基线：wc -l sm_context.cpp = 965
- [x] 1.4 MUST 记录基线：ctest 全绿
- [x] 1.5 MUST 验证：sm_context.cpp:379 update_active_mask() 存在
- [x] 1.6 MUST 验证：sm_context.cpp:455-490 与 :580-623 重复循环存在

## 2. Phase 1: 去重 reconvergence 编排循环（3h）

- [x] 2.1 MUST 提取 sm_context.cpp:455-490 与 :580-623 重复循环为共享 helper
- [x] 2.2 MUST 新建 src/ptxsim/core/sm_context_reconvergence.{h,cpp}
- [x] 2.3 MUST 行级随迁 sm_context.cpp:379 update_active_mask() 调用（§1）
- [x] 2.4 MUST 保留 sm_context.cpp:374 注释（active_count sync fix）
- [x] 2.5 MUST NOT 改变 WarpContext public API 调用点（:461/:468/:583/:590 等）
- [x] 2.6 MUST 更新 src/ptxsim/core/CMakeLists.txt
- [x] 2.7 MUST 验证：grep -c 'check_reconvergence' sm_context.cpp ≤ 2
- [x] 2.8 MUST 验证：sm_context.cpp 减少 ≥ 65 行
- [x] 2.9 MUST 验证：ctest 全绿（barrier/divergence 测试）
- [x] 2.10 git commit -m "refactor(sm): dedup reconvergence orchestration loops to shared helper"

## 3. Phase 2: ADR-0020 注入编排提取（3h）

- [ ] 3.1 MUST 提取 step_b_set_blocked_cycles + 3 setter + 3-step 编排至独立模块
- [ ] 3.2 MUST 新建 src/ptxsim/core/sm_context_cpptlm_inject.{h,cpp}
- [ ] 3.3 MUST 保持 step_b no-op byte-identical fallback 契约（lessons-learned §14）
- [ ] 3.4 MUST 保持 4 分支测试锁定（test_step_b_set_blocked_cycles.cpp）
- [ ] 3.5 MUST NOT 改变 WarpContext public API 调用
- [ ] 3.6 MUST 更新 src/ptxsim/core/CMakeLists.txt
- [ ] 3.7 MUST 验证：step_b 4 分支测试全绿
- [ ] 3.8 MUST 验证：ctest 全绿
- [ ] 3.9 git commit -m "refactor(sm): extract ADR-0020 cpptlm injection code to sm_context_cpptlm_inject"

## 4. Phase 3: CTA 调度 + SM barrier 拆分（4h）

- [ ] 4.1 MUST 拆分 CTA 调度逻辑至独立模块
- [ ] 4.2 MUST 拆分 SM 级 barrier 同步封装至独立模块
- [ ] 4.3 MUST 拆分 warp 生命周期管理至独立模块
- [ ] 4.4 MUST NOT 改变 exe_once() 主循环签名
- [ ] 4.5 MUST NOT 改变 BarrierModule 内部实现
- [ ] 4.6 MUST 行级随迁 sm_context.cpp:379 update_active_mask() 调用
- [ ] 4.7 MUST 验证：sm_context.cpp < 250 行
- [ ] 4.8 MUST 验证：sm_context.cpp 零 diff WarpContext public API 调用点
- [ ] 4.9 MUST 验证：ctest 全绿
- [ ] 4.10 git commit -m "refactor(sm): split CTA scheduling + SM barrier into separate modules"

## 5. Phase 4: 最终验证（1h）

- [ ] 5.1 MUST 验证：wc -l sm_context.cpp < 250
- [ ] 5.2 MUST 验证：sm_context.cpp:379 update_active_mask() 仍在（§1 关键项）
- [ ] 5.3 MUST 验证：sm_context.cpp:374 注释完整
- [ ] 5.4 MUST 验证：sm_context.cpp WarpContext API 调用点零 diff
- [ ] 5.5 MUST 验证：ctest --output-on-failure 全绿
- [ ] 5.6 SHOULD 更新 src/ptxsim/core/AGENTS.md 文档化新子模块

## 6. 应用阶段

- [ ] 6.1 MUST 运行 openspec validate god-class-refactor-sm-context --strict
- [ ] 6.2 MUST 通过所有验证后 archive

## 验收

- sm_context.cpp < 250 行（965 → < 250）
- 新组件 ≤ 4 个
- check_reconvergence 调用点 ≤ 2（经 helper）
- step_b 4 分支测试 + barrier 测试 + sm_context 单测全绿
- 集成测试（execute_warp_instruction 路径）零回归
- 每个 Phase commit 独立可 revert
- ptx-lessons-learned §1, §14, Checklist B 全部勾选

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（SKILL.md:48-77）：sm_context.cpp:379 列入迁移清单
- MUST Checklist B（SKILL.md:474-483）：worktree + 3 Phase commit
- MUST §14 step_b no-op fallback 4 分支测试锁定（SKILL.md:409-455）
- MUST NOT 改 exe_once() 签名、SM/CTA/Warp 三层调用链、WarpContext public API 签名
- SHOULD 复用 BarrierModule API