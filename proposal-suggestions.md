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
    "status": "待创建",
    "phase": "Phase-10",
    "category": "core-impl",
    "description": "## 架构依据\n- Debt Audit §P1-A3: wmma.cpp:6-13 遇 WMMA 指令得到未初始化值\n- Debt Audit §P1-A4: tensor.cpp:8-15 Tensor Core stub 静默无操作\n\n## 范围\n- **In Scope**:\n  - WMMA: 将静默无操作改为 `throw UnsupportedInstructionException`\n  - Tensor Core: 同上\n\n## 关键场景\n- GIVEN 执行 WMMA 指令, WHEN 模拟器遇到, THEN 抛出明确异常而非返回未初始化值\n\n## 技术约束\n- MUST 与 pre-Blackwell tcgen05 策略一致 (throw UnsupportedInstructionException)\n- MUST NOT 更改正常执行路径\n\n## 验收标准\n- WMMA/tensor stub 抛出异常\n- 异常信息包含指令名\n- 不影响其他指令执行",
    "effort": "1天"
  },
  {
    "name": "add-nested-divergence-tests",
    "priority": "P1",
    "source": "Debt Audit P1-A10 — 嵌套分歧测试缺失",
    "status": "待创建",
    "phase": "Phase-10",
    "category": "core-test",
    "description": "## 架构依据\n- Debt Audit §P1-A10: test_nested_divergence.cpp:106 嵌套分歧测试缺失\n\n## 范围\n- **In Scope**:\n  - 为嵌套分歧场景编写集成测试\n  - 覆盖两级分歧 + 汇聚\n- **Out Scope**:\n  - 不修改调度器实现\n\n## 关键场景\n- GIVEN 两级分歧, WHEN 执行, THEN warp 正确汇聚\n- GIVEN 分歧中分歧, WHEN SIMT stack 管理, THEN depth 正确\n\n## 验收标准\n- 3+ 嵌套分歧测试添加\n- 测试全部通过",
    "effort": "1-2天"
  },
  {
    "name": "implement-call-function",
    "priority": "P1",
    "source": "Debt Audit P1-A6 — call 用户函数未实现",
    "status": "待创建",
    "phase": "Phase-10",
    "category": "core-impl",
    "description": "## 架构依据\n- Debt Audit §P1-A6: call.cpp ~20% 完整度, 静默跳过\n\n## 范围\n- **In Scope**:\n  - 实现 PTX call/ret 用户函数调用\n  - 支持参数传递和返回\n- **Out Scope**:\n  - 不涉及递归调用\n\n## 技术约束\n- MUST 保持与现有执行管道兼容\n\n## 验收标准\n- call/ret 基本场景可执行\n- 集成测试覆盖\n- 无回归",
    "effort": "5-8天"
  },
  {
    "name": "create-cfg-design-doc",
    "priority": "P2",
    "source": "docs/architecture/README.md — CFG-DESIGN.md ⏳ 待创建",
    "status": "待创建",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据\n- docs/architecture/README.md: CFG-DESIGN.md 标记为 ⏳ 待创建\n\n## 范围\n- **In Scope**:\n  - 编写 CFG 分析详细设计文档\n  - 包括 post-dominator 算法说明\n- **Out Scope**:\n  - 不修改代码\n\n## 验收标准\n- CFG-DESIGN.md 创建完成\n- 文档格式与现有架构文档一致",
    "effort": "1-2天"
  },
  {
    "name": "fix-code-self-contradictions",
    "priority": "P1",
    "source": "Debt Audit P0-D1/D2 — 文档自相矛盾",
    "status": "待创建",
    "phase": "Phase-10",
    "category": "arch-design",
    "description": "## 架构依据\n- Debt Audit §P0-D1: ANTLR 版本号矛盾 (4.13.1 vs 4.11.1)\n- Debt Audit §P0-D2: force_set_pc() vs set_pc() 互相反指\n- Debt Audit §P0-D4: PROJECT-COMPLETION-SUMMARY.md 虚假声明\n\n## 范围\n- **In Scope**:\n  - 同步 AGENTS.md 与 copilot-instructions.md 的 ANTLR 版本号\n  - 统一 force_set_pc/set_pc 推荐策略\n  - 删除或更新 PROJECT-COMPLETION-SUMMARY.md\n\n## 验收标准\n- 版本号一致\n- PC 设置策略文档无矛盾\n- 虚假声明文档修复",
    "effort": "1-2天"
  },
  {
    "name": "split-cpptlm-core-minimal",
    "priority": "P2",
    "source": "ADR-0022 §未来 — cpptlm_core_minimal 拆分",
    "status": "待创建",
    "phase": "Phase-10",
    "category": "infra-setup",
    "description": "## 架构依据\n- ADR-0022 §未来: 拆分 cpptlm_core_minimal 以减小体积\n\n## 范围\n- **In Scope**:\n  - 从 cpptlm_core 拆分出 minimal 子集 (bridge + IPtxEmuDriver + MemoryBridge)\n- **Out Scope**:\n  - 不更改现有链接行为\n\n## 验收标准\n- 拆分后构建通过\n- CppTLM 协同仿真测试通过",
    "effort": "2-3天"
  }
]
