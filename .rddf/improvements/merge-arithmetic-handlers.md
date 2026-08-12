# merge-arithmetic-handlers

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-3
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `src/ptxsim/instructions/arithmetic.cpp` 478 行 + `arithmetic_ext.cpp` 764 行 + `arithmetic_muldiv.cpp` 490 行 = **1732 行**
- 3 个文件职责重叠：基础算术 / 扩展算术 / 乘除运算，实际共享大量 operand 处理逻辑
- X-Macro 分派模式下，每个 handler 独立注册，合并后可统一 operand 提取和类型分派

## 范围

- **In Scope**:
  - 合并 3 个 arithmetic 文件为统一 handler
  - 提取共享 operand 处理为 helper
  - 统一类型分派逻辑
- **Out Scope**:
  - 不改变任何算术指令的计算结果
  - 不修改 ptx_op.def
  - 不动测试文件

## 关键场景

- GIVEN 合并后, WHEN 执行任意算术指令 (add/sub/mul/div/mad 等), THEN 结果与合并前一致
- GIVEN 共享 operand 提取, WHEN 处理不同位宽 (.u32/.s32/.f32 等), THEN 类型转换正确

## 技术约束

- MUST 保持所有算术指令的计算结果不变
- MUST 保持 ptx_op.def X-Macro 注册不变
- SHOULD 按运算类型分组而非按文件大小拆分

## 验收标准

- 合并后总代码行数减少 ≥ 15%（去除重复 operand 处理）
- 所有算术指令测试通过（unit + integration）
- ctest 全绿
