# complete-x-macro-dispatch

**优先级**: P3 | **来源**: docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-15
**阶段**: default | **分类**: core-impl
**类型**: refactor

## 架构依据

- `src/ptxsim/instruction_factory.cpp` 中 X-Macro (`#include "ptx_ir/ptx_op.def"`) 仅被调用 **1 次**（line 18）
- `src/ptxsim/instruction_handlers.cpp` 同样仅 1 次（line 190）
- ptx_op.def 有 **106 个条目**，但 X-Macro 展开仅用于注册/分派，未充分利用其代码生成能力
- 其他文件（ptx_parser.cpp, ptx_visitor.cpp, statement_context.cpp）也各自独立 include ptx_op.def，存在重复展开

## 范围

- **In Scope**:
  - 审查 X-Macro 的所有使用点，确认是否每个 use case 都有独立展开
  - 评估是否可以通过模板/代码生成减少重复展开
  - 统一 X-Macro 的 X 定义模式
- **Out Scope**:
  - 不修改 ptx_op.def 条目本身
  - 不改变指令注册或分派逻辑
  - 不影响 parser 端的 X-Macro 使用

## 关键场景

- GIVEN X-Macro 统一后, WHEN 新增 ptx_op.def 条目, THEN 所有使用点自动生效
- GIVEN 编译后, WHEN 检查符号表, THEN 无重复定义的 handler 注册

## 技术约束

- MUST 保持 ptx_op.def 的 106 个条目全部正确注册
- MUST 保持 InstructionFactory::get_handler() 分派逻辑不变
- SHOULD 减少 X-Macro 重复展开次数

## 验收标准

- 所有 106 个 ptx_op.def 条目的 handler 正确注册
- ctest 全绿
- 编译时间不增加
