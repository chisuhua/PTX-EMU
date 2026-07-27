# refactor-warp-context — Tasks

## 1. Phase 0: 准备工作

- [x] 1.1 建立 baseline worktree（Checklist B：避免污染 main 分支）
- [x] 1.2 运行 `wc -l src/ptxsim/core/warp_context.cpp` 记录基线（应为 558）
- [x] 1.3 运行 ctest 记录基线全绿
- [x] 1.4 MUST 验证：grep -c 'sync_to_warp_state' src/ptxsim/core/warp_context.cpp = 4（§1 4 站点）
- [x] 1.5 MUST 验证：sm_context.cpp:379/:461/:468/:583/:590 5 站点 API 调用存在（API 冻结基线）

## 2. Phase 1: 提取 active mask helper（1.5h）

- [x] 2.1 MUST 新建 src/ptxsim/core/warp_context_active_mask.{h,cpp}
- [x] 2.2 MUST 移动 set_active_mask + 相关 active mask 操作至新模块
- [x] 2.3 MUST 保持 set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS）
- [x] 2.4 MUST 行级保留 4 处 sync_to_warp_state() 调用（§1 SKILL.md:48-77）
- [x] 2.5 MUST 在 warp_context.h 包含新模块
- [x] 2.6 MUST 更新 src/ptxsim/core/CMakeLists.txt 添加新源
- [x] 2.7 MUST 验证：sm_context.cpp 零 diff
- [x] 2.8 MUST 验证：cmake --build build 通过
- [x] 2.9 MUST 验证：ctest 全绿（特别是 active mask/ret handler 测试）
- [x] 2.10 git commit -m "refactor(warp): extract active mask helper to warp_context_active_mask"

## 3. Phase 2: 提取 SIMT 编排（2h）

- [x] 3.1 MUST 新建 src/ptxsim/core/warp_context_simt.{h,cpp}
- [x] 3.2 MUST 移动 push/pop/check_reconvergence 编排逻辑（warp_context.cpp:64-143）至新模块
- [x] 3.3 MUST NOT 重新抽离 simt_stack.cpp 数据结构（已存在）
- [x] 3.4 MUST 保持 simt_stack.cpp/h 零 diff
- [x] 3.5 MUST 行级保留 4 处 sync_to_warp_state() 调用
- [x] 3.6 MUST 在 warp_context.h 包含新模块
- [x] 3.7 MUST 更新 src/ptxsim/core/CMakeLists.txt 添加新源
- [x] 3.8 MUST 验证：sm_context.cpp 零 diff
- [x] 3.9 MUST 验证：cmake --build build 通过
- [x] 3.10 MUST 验证：ctest 全绿（特别是 barrier/divergence 测试）
- [x] 3.11 git commit -m "refactor(warp): extract SIMT orchestration to warp_context_simt"

## 4. Phase 3: 提取指令分发（1.5h）

- [ ] 4.1 MUST 新建 src/ptxsim/core/warp_context_dispatch.{h,cpp}
- [ ] 4.2 MUST 移动 execute_warp_instruction 分发逻辑至新模块
- [ ] 4.3 MUST 使用策略表/函数指针替换 switch/if-else
- [ ] 4.4 MUST 行级保留 4 处 sync_to_warp_state() 调用
- [ ] 4.5 MUST 在 warp_context.h 包含新模块
- [ ] 4.6 MUST 更新 src/ptxsim/core/CMakeLists.txt 添加新源
- [ ] 4.7 MUST 验证：sm_context.cpp 零 diff
- [ ] 4.8 MUST 验证：cmake --build build 通过
- [ ] 4.9 MUST 验证：ctest 全绿
- [ ] 4.10 git commit -m "refactor(warp): extract instruction dispatch to warp_context_dispatch"

## 5. Phase 4: 最终验证（1h）

- [ ] 5.1 MUST 验证：wc -l src/ptxsim/core/warp_context.cpp < 300
- [ ] 5.2 MUST 验证：grep -c 'sync_to_warp_state' src/ptxsim/core/warp_context*.cpp 合计 ≥ 4
- [ ] 5.3 MUST 验证：sm_context.cpp 零 diff
- [ ] 5.4 MUST 验证：ctest --output-on-failure 全绿
- [ ] 5.5 SHOULD 更新 src/ptxsim/core/AGENTS.md 文档化 3 个新子模块

## 6. 应用阶段

- [ ] 6.1 MUST 运行 openspec validate refactor-warp-context --strict
- [ ] 6.2 MUST 通过所有验证后 archive

## 验收

- warp_context.cpp < 300 行（从 558 → < 300）
- 新组件 ≤ 3 个（active_mask / simt / dispatch）
- grep -c 'sync_to_warp_state' 合计 ≥ 4（4 站点全部保留）
- sm_context.cpp 零 diff 编译通过（API 冻结证据）
- barrier/active_mask/ret handler 测试全绿
- test-coverage-enforcer 验证 execute_warp_instruction 路径
- ptx-lessons-learned Checklist B 全部勾选
- 所有 3 个 Phase commit 独立可 revert

## 关键约束（MUST/MUST NOT）

- MUST §1 行级 diff（SKILL.md:48-77）：4 站点 :337/:345/:370/:375 列入迁移清单
- MUST Checklist B（SKILL.md:474-483）：worktree + 3 Phase commit
- MUST set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS；非 §2）
- MUST NOT 改 WarpContext public API 签名（消费方 sm_context.cpp:379/:461/:468/:583/:590）
- MUST NOT 改 ret handler / execute_warp_instruction 主入口