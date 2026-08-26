# complete-x-macro-dispatch

> **状态**: CLOSED — premise-stale(2026-08-26 验证)
> **关闭理由**: 提案前提 "X-Macro 仅用于注册/分派,未充分利用其代码生成能力" 与当前代码事实不符。X-Macro 已在 5 个使用点承担全部代码生成职责,每个使用点有独立 X 定义(这正是 X-Macro 模式的标准用法,非 "重复展开" 问题):
> - `src/ptxsim/instruction_factory.cpp:18` — 生成全部 ptx_op.def 条目的 handler 注册(`handler_map[enum_val] = new opstr##Handler()`)
> - `src/ptxsim/instruction_handlers.cpp:191` — 生成全部 handler 实现(`IMPLEMENT_##struct_kind##_HANDLER(op_str)`)
> - `src/ptx_ir/statement_context.cpp:15` — 生成 `S2s()` 字符串转换 switch 全部分支
> - `src/ptx_parser/ptx_parser.cpp:1048` — 生成 statement builder 分派(`STATEMENT_##struct_kind`)
> - `src/ptx_parser/ptx_visitor.cpp:593` — 生成 visitor 实现(`VISITOR_IMPL_##instr_kind`)
>
> 新增 ptx_op.def 条目已自动生效于全部使用点,提案目标客观上已达成,无需改动。

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
