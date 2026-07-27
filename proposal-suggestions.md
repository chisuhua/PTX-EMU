[
  {
    "name": "fix-commented-ptx-tests",
    "priority": "P0",
    "source": "Debt Audit P0-C3 — tests/unit/CMakeLists.txt:432-472",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "core-test",
    "description": "## 架构依据\n- Debt Audit 2026-07-02 §P0-C3: 7 个 PTX 单元测试被注释掉\n- 包括: unit_ptx_integer, unit_ptx_float, unit_ptx_extended, unit_ptx_bitwise, unit_ptx_cvt, unit_ptx_ld_st, unit_ptx_cvta\n\n## 范围\n- **In Scope**:\n  - 恢复 7 个被注释的 PTX 单元测试\n  - 更新测试代码以匹配当前 API\n  - 确认全部通过后激活\n- **Out Scope**:\n  - 不新增额外的测试用例\n  - 不修改被测试的实现代码\n\n## 关键场景\n- GIVEN 恢复测试, WHEN 构建, THEN 测试编译通过并执行\n- GIVEN 测试执行, WHEN 全部通过, THEN 回归保障恢复\n\n## 技术约束\n- MUST 保持现有测试框架兼容\n- MUST 全部测试通过才能标完成\n\n## 验收标准\n- 7 个测试全部恢复并激活\n- ctest 中对应标签正常执行\n- 现有测试无回归",
    "effort": "2-3天"
  },
  {
    "name": "add-cudart-unit-tests",
    "priority": "P0",
    "source": "Debt Audit P0-C2 — cudart_sim.cpp 零单元测试",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "core-test",
    "description": "## 架构依据\n- Debt Audit 2026-07-02 §P0-C2: cudart_sim.cpp 933 行核心入口零直接单元测试\n- 覆盖 cudaLaunchKernel、__cudaRegisterFatBinary、cudaStreamSynchronize 等\n\n## 范围\n- **In Scope**:\n  - 为 cudart_sim.cpp 核心函数编写单元测试\n  - 至少覆盖 cudaLaunchKernel 和 cudaStreamSynchronize\n- **Out Scope**:\n  - 不重构 cudart_sim.cpp\n  - 不修改 production 代码\n\n## 关键场景\n- GIVEN cudaLaunchKernel 调用, WHEN 验证, THEN 内核正确注册\n- GIVEN cudaStreamSynchronize, WHEN 验证, THEN 返回正确状态\n\n## 技术约束\n- MUST 使用 Catch2 测试框架\n- MUST NOT 依赖真实 CUDA 设备\n\n## 验收标准\n- 5+ 单元测试覆盖核心函数\n- 测试编译通过且绿色运行\n- 覆盖正常路径和错误路径",
    "effort": "3-4天"
  },
  {
    "name": "cleanup-wbar-api",
    "priority": "P1",
    "source": "Debt Audit P1-A1/A2 — BarWarpSyncHandler 仍用 Wbar API",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据\n- Debt Audit §P1-A1: BarWarpSyncHandler 仍用 Wbar API (barrier.cpp:161,215 ~30 处)\n- Debt Audit §P1-A2: `current_wbar_id` 11+ 处生产读写\n\n## 范围\n- **In Scope**:\n  - 将 barrier.cpp 中 Wbar API 调用替换为 BarrierModule API\n  - 移除 `current_wbar_id` 的残余使用\n  - 删除遗留的 `wbar.h` include\n\n## 关键场景\n- GIVEN bar.sync 执行, WHEN 通过 BarrierModule, THEN 行为完全一致\n- GIVEN 全量测试, WHEN 运行, THEN 无回归\n\n## 技术约束\n- MUST 保持行为不变\n- MUST 全部 barrier 测试通过\n\n## 验收标准\n- barrier.cpp 零 Wbar API 调用\n- `current_wbar_id` 零生产引用\n- 全部 barrier 测试通过",
    "effort": "2-3天"
  },
  {
    "name": "fix-wmma-tensor-stubs",
    "priority": "P1",
    "source": "Debt Audit P1-A3/A4 — WMMA/Tensor Core stub 静默无操作",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "core-impl",
    "description": "## 架构依据\n- Debt Audit §P1-A3: wmma.cpp:6-13 遇 WMMA 指令得到未初始化值\n- Debt Audit §P1-A4: tensor.cpp:8-15 Tensor Core stub 静默无操作\n\n## 范围\n- **In Scope**:\n  - WMMA: 将静默无操作改为 `throw UnsupportedInstructionException`\n  - Tensor Core: 同上\n\n## 关键场景\n- GIVEN 执行 WMMA 指令, WHEN 模拟器遇到, THEN 抛出明确异常而非返回未初始化值\n\n## 技术约束\n- MUST 与 pre-Blackwell tcgen05 策略一致 (throw UnsupportedInstructionException)\n- MUST NOT 更改正常执行路径\n\n## 验收标准\n- WMMA/tensor stub 抛出异常\n- 异常信息包含指令名\n- 不影响其他指令执行",
    "effort": "1天"
  },
  {
    "name": "add-nested-divergence-tests",
    "priority": "P1",
    "source": "Debt Audit P1-A10 — 嵌套分歧测试缺失",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "core-test",
    "description": "## 架构依据\n- Debt Audit §P1-A10: test_nested_divergence.cpp:106 嵌套分歧测试缺失\n\n## 范围\n- **In Scope**:\n  - 为嵌套分歧场景编写集成测试\n  - 覆盖两级分歧 + 汇聚\n- **Out Scope**:\n  - 不修改调度器实现\n\n## 关键场景\n- GIVEN 两级分歧, WHEN 执行, THEN warp 正确汇聚\n- GIVEN 分歧中分歧, WHEN SIMT stack 管理, THEN depth 正确\n\n## 验收标准\n- 3+ 嵌套分歧测试添加\n- 测试全部通过",
    "effort": "1-2天"
  },
  {
    "name": "implement-call-function",
    "priority": "P1",
    "source": "Debt Audit P1-A6 — call 用户函数未实现",
    "status": "已评估，不需要",
    "phase": "Phase-10",
    "category": "core-impl",
    "description": "## 架构依据\n- Debt Audit §P1-A6: call.cpp ~20% 完整度, 静默跳过\n\n## 范围\n- **In Scope**:\n  - 实现 PTX call/ret 用户函数调用\n  - 支持参数传递和返回\n- **Out Scope**:\n  - 不涉及递归调用\n\n## 技术约束\n- MUST 保持与现有执行管道兼容\n\n## 验收标准\n- call/ret 基本场景可执行\n- 集成测试覆盖\n- 无回归",
    "effort": "5-8天"
  },
  {
    "name": "create-cfg-design-doc",
    "priority": "P2",
    "source": "docs/architecture/README.md — CFG-DESIGN.md ⏳ 待创建",
    "status": "已评估，不需要",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据\n- docs/architecture/README.md: CFG-DESIGN.md 标记为 ⏳ 待创建\n\n## 范围\n- **In Scope**:\n  - 编写 CFG 分析详细设计文档\n  - 包括 post-dominator 算法说明\n- **Out Scope**:\n  - 不修改代码\n\n## 验收标准\n- CFG-DESIGN.md 创建完成\n- 文档格式与现有架构文档一致",
    "effort": "1-2天"
  },
  {
    "name": "fix-code-self-contradictions",
    "priority": "P1",
    "source": "Debt Audit P0-D1/D2 — 文档自相矛盾",
    "status": "已完成",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据\n- Debt Audit §P0-D1: ANTLR 版本号矛盾 (4.13.1 vs 4.11.1)\n- Debt Audit §P0-D2: force_set_pc() vs set_pc() 互相反指\n- Debt Audit §P0-D4: PROJECT-COMPLETION-SUMMARY.md 虚假声明\n\n## 范围\n- **In Scope**:\n  - 同步 AGENTS.md 与 copilot-instructions.md 的 ANTLR 版本号\n  - 统一 force_set_pc/set_pc 推荐策略\n  - 删除或更新 PROJECT-COMPLETION-SUMMARY.md\n\n## 验收标准\n- 版本号一致\n- PC 设置策略文档无矛盾\n- 虚假声明文档修复",
    "effort": "1-2天"
  },
  {
    "name": "split-cpptlm-core-minimal",
    "priority": "P2",
    "source": "ADR-0022 §未来 — cpptlm_core_minimal 拆分",
    "status": "暂缓",
    "phase": "Phase-10",
    "category": "infra-setup",
    "description": "## 架构依据\n- ADR-0022 §未来: 拆分 cpptlm_core_minimal 以减小体积\n\n## 范围\n- **In Scope**:\n  - 从 cpptlm_core 拆分出 minimal 子集 (bridge + IPtxEmuDriver + MemoryBridge)\n- **Out Scope**:\n  - 不更改现有链接行为\n\n## 验收标准\n- 拆分后构建通过\n- CppTLM 协同仿真测试通过",
    "effort": "2-3天"
  },
  {
    "name": "god-class-refactor-thread-context",
    "priority": "P1",
    "source": "post-phase3-debt-roadmap §1.2 C-1",
    "status": "已归档（superseded）",
    "phase": "Phase-9.5",
    "category": "arch-design",
    "description": "## 归档说明 (Oracle 评审 2026-07-26)\n- Phase 1+2 已完成：SimtPcManager + RegisterAccessLayer 提取（commits 3082e808 + 3fa5e5f7）\n- 实际行数 727（已从 885 下降），22 includes\n- 剩余 Phase 3 由 ADR-0019 接管（MemoryAccessor + InstructionPipeline accessor）\n- 原 improvement 文件已删除（与已归档 change 重名，违反 Checklist G）\n- 新工作请走 ADR-0019 Phase 3 + Ref: archive/2026-07-06-god-class-refactor-thread-context",
    "effort": "n/a"
  },
  {
    "name": "god-class-refactor-sm-context",
    "priority": "P2",
    "source": "post-phase3-debt-roadmap §1.2 C-2",
    "status": "已创建（change 已建立）",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据 (Round-3 redesign, Oracle 2026-07-26)\n- src/ptxsim/core/sm_context.cpp 实测 965 行（roadmap 703 行已过期，post-phase3-debt-roadmap.md:53）\n- §1 实证站点：sm_context.cpp:379 update_active_mask()（注释:374 明示 active_count sync fix）\n- C-2/C-18 边界：sm_context.cpp:455-490 + :580-623 两段近似重复的 SIMT reconvergence 编排循环（~130 行）\n- ADR-0020 cpptlm 注入增长 ~260 行\n\n## 范围\n- **In Scope**: 拆分为 ≤4 组件；提取并去重 :455-623 reconvergence 循环为共享 helper；ADR-0020 注入点归属决策\n- **Out Scope**: WarpContext public API 签名（由 C-18 冻结）；不动 BarrierModule；不改 exe_once() 签名\n\n## 关键场景\n- GIVEN 共享 helper 提取, WHEN 调用, THEN 两处 reconvergence trace 输出字节级一致\n- GIVEN 行级 diff, THEN sm_context.cpp:379 update_active_mask() 随迁（§1）\n- GIVEN step_b 移动, THEN 4 分支测试随迁（§14）\n\n## 技术约束\n- MUST §1 行级 diff（SKILL.md:48-77，sm_context.cpp:379）\n- MUST Checklist B（SKILL.md:474-483）baseline worktree + 3 Phase commit\n- MUST §14 step_b no-op fallback 4 分支测试锁定（SKILL.md:409-455）\n- MUST NOT 改 exe_once() 签名、WarpContext public API 签名\n\n## 验收标准\n- sm_context.cpp < 250 行\n- 新组件 ≤ 4；check_reconvergence 调用点 ≤ 2（经 helper）\n- step_b 4 分支 + barrier + execute_warp_instruction 路径测试全绿\n\n## Round-3 依赖\n- 必须在 C-18 之后执行（WarpContext API 冻结）\n\n## Oracle 评审链\n- Round-1: APPROVE-WITH-CHANGES（行数基线过期）\n- Round-2: NEEDS-MORE-CHANGES（缺 §1/Checklist B + scope 重叠）\n- Round-3: CONDITIONAL APPROVE\n\n## 批准状态\n- 2026-07-26: 经 Oracle Round-3 重写并批准（详见 proposal-approved.md）\n\n## Change 创建\n- openspec/changes/god-class-refactor-sm-context/\n- 4 artifacts (proposal + design + tasks + specs)\n- openspec validate --strict 通过",
    "effort": "10-12h"
  },
  {
    "name": "split-ptx-visitor-god-class",
    "priority": "P2",
    "source": "post-phase3-debt-roadmap §1.2 C-17",
    "status": "已创建（change 已建立）",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据 (Round-3 redesign, Oracle 2026-07-26)\n- src/ptx_parser/ptx_visitor.cpp 实测 1067 行\n- **重要**: 按 statement 类型 10 文件拆分已完成（ptx_visitor_{generic,atom,call,branch,barrier,simple,special,warp,memory,abi}.cpp，AGENTS.md:48 文档化）——不重复提案\n- 剩余 1067 行真实构成：qualifier/operand helpers (:138-317) + tcgen05 visitor (:841-902) + X-Macro dispatch (:927-931) + operand visitors (:937-1067)\n- **实证 bug**: ptx_visitor.cpp:917 与 :922 重复 #include \"ptx_visitor_warp.cpp\"\n\n## 范围\n- **In Scope**:\n  - (a) 提取 visitTcgen05Inst (:841-902) → ptx_visitor_tcgen05.cpp\n  - (b) 提取 operand/qualifier helpers (:138-317 + :937-1067) → ptx_visitor_operands.cpp\n  - (c) 提取 X-Macro dispatch (:927-931) + include 聚合区 (:905-925) → ptx_visitor_dispatch.cpp\n  - (d) 删除 :922 重复 #include \"ptx_visitor_warp.cpp\"\n- **Out Scope**: 不重拆 statement 类型 visit*（已完成）；不改 .g4/IR/CFGBuilder；不拆 visitFunctionDecl\n\n## 关键场景\n- GIVEN 重复 include 删除, WHEN 编译, THEN 零 ODR 告警\n- GIVEN tcgen05 提取, WHEN 解析 PTX, THEN IR 字节级一致（含 cta_group IMMEDIATE 提取，:866-877 §13）\n- GIVEN operand helpers 提取, WHEN 19 个 extractQualifiersFromContext 调用, THEN qualifier 完全一致\n- GIVEN 拆分后, WHEN test_all_ptx.sh, THEN 全绿\n\n## 技术约束\n- MUST Checklist H 实证验证（SKILL.md:539-555）\n- MUST IR 字节级一致；MUST NOT 改 .g4 / extractQualifiersFromContext 签名\n- MUST visitTcgen05Inst 内 C3 fix parse-tree walk 注释 (:863-877) 行级随迁（§1）\n\n## 验收标准\n- ptx_visitor.cpp ≤ 700 行\n- grep -c '#include \"ptx_visitor_warp.cpp\"' src/ptx_parser/ptx_visitor.cpp == 1\n- test_all_ptx.sh 全绿（47/47 目标）\n- parser 单测 + CFG post-dominator 集成测试零回归（ADR-0007）\n\n## Round-3 独立\n- 可独立优先执行（无 C-2/C-18 依赖）\n\n## Oracle 评审链\n- Round-1: REJECT 范畴识别错误\n- Round-2: NEEDS-MORE-CHANGES（核心 scope 已实施）\n- Round-3: APPROVE — 优先执行\n\n## 批准状态\n- 2026-07-26: 经 Oracle Round-3 重写并批准（详见 proposal-approved.md）\n\n## Change 创建\n- openspec/changes/split-ptx-visitor-god-class/\n- 4 artifacts (proposal + design + tasks + 4 specs)\n- openspec validate --strict 通过",
    "effort": "2-3h"
  },
  {
    "name": "refactor-warp-context",
    "priority": "P2",
    "source": "post-phase3-debt-roadmap §1.2 C-18",
    "status": "已创建（change 已建立）",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据 (Round-3 redesign, Oracle 2026-07-26)\n- src/ptxsim/core/warp_context.cpp 实测 558 行\n- 10 个 include；最近 30 commits 触碰 6 次（高 churn）\n- simt_stack 数据结构已抽离至 simt_stack.cpp；残留 :64-143 为编排逻辑\n- **§1 实证**: 4 处 sync_to_warp_state() (:337/:345/:370/:375)\n- **API 消费方**: sm_context.cpp:379/:461/:468/:583/:590\n\n## 范围\n- **In Scope**: 指令分发策略表；active mask helper；SIMT 编排提取（:64-143）；拆分为 ≤3 组件\n- **Out Scope**:\n  - WarpContext public API 签名变更（消费方 sm_context.cpp:379/461/468/583/590）\n  - sm_context.cpp:455-623 reconvergence 编排循环去重（归 C-2）\n  - 不重抽 simt_stack 数据结构；不改 ThreadContext/BarrierModule\n\n## 关键场景\n- GIVEN 任何迁移, WHEN 行级 diff, THEN 4 处 sync_to_warp_state() 逐行随迁（§1）\n- GIVEN set_active_mask, THEN ret handler overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS）\n- GIVEN 拆分后, WHEN sm_context.cpp 编译, THEN 调用点零修改（API 冻结证据）\n\n## 技术约束\n- MUST §1 行级 diff（SKILL.md:48-77，4 站点 :337/:345/:370/:375 列入迁移清单）\n- MUST Checklist B（SKILL.md:474-483）worktree + 3 Phase commit\n- MUST set_active_mask overwrite 语义（失败模式速查表 + AGENTS.md ANTI-PATTERNS；非 §2）\n- MUST NOT 改 WarpContext public API 签名；MUST NOT 改 ret handler / execute_warp_instruction 主入口\n\n## 验收标准\n- warp_context.cpp < 300 行；新组件 ≤ 3\n- grep -c 'sync_to_warp_state' src/ptxsim/core/warp_context*.cpp 合计 ≥ 4\n- sm_context.cpp 零 diff 编译通过\n- barrier/active_mask/ret handler 测试全绿\n- test-coverage-enforcer 验证 execute_warp_instruction 路径\n\n## Round-3 依赖\n- 必须在 C-2 之前执行（冻结 WarpContext API）\n\n## Oracle 评审链\n- Round-1: APPROVE-WITH-CHANGES（§2 引用错误）\n- Round-2: NEEDS-MORE-CHANGES（缺 §1/Checklist B + API 边界）\n- Round-3: APPROVE — 在 C-2 之前落地\n\n## 批准状态\n- 2026-07-26: 经 Oracle Round-3 重写并批准（详见 proposal-approved.md）\n\n## Change 创建\n- openspec/changes/refactor-warp-context/\n- 4 artifacts (proposal + design + tasks + specs)\n- openspec validate --strict 通过",
    "effort": "6h"
  }
]
