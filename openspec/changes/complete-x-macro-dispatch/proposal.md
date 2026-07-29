# complete-x-macro-dispatch - Proposal

## Why

`include/ptx_ir/ptx_op.def` 定义了 **106 个指令条目**，通过 X-Macro 机制在
多个文件中展开。当前存在以下问题：

1. **重复展开**: `ptx_op.def` 被 `#include` 展开 **10 次**（跨 8 个文件），每次
   展开定义不同的 `X` 宏但可能存在优化空间

2. **未充分利用代码生成能力**: X-Macro 展开主要用于枚举生成、字符串表、handler
   注册/声明，但部分使用点（如 `instruction_factory.cpp`、`instruction_handlers.cpp`）
   各自独立展开，未评估是否可合并

3. **X 定义模式不统一**: 不同文件使用不同的 X 参数命名和展开模式

X-Macro 展开点清单：

| # | 文件 | 用途 | 展开次数 |
|---|------|------|---------|
| 1 | `include/ptx_ir/ptx_types.h:21` | 枚举生成（`StatementType`） | 1 |
| 2 | `include/ptx_parser/ptx_parser.h:162` | Listener 声明 | 1 |
| 3 | `include/ptx_parser/ptx_visiter.h:104` | Visitor 声明 | 1 |
| 4 | `include/ptxsim/instruction_handlers.h:139` | Handler 类声明 | 1 |
| 5 | `src/ptx_ir/statement_context.cpp:14` | 字符串转换（`S2s`） | 1 |
| 6 | `src/ptx_parser/ptx_parser.cpp:1048` | Listener 实现 | 1 |
| 7 | `src/ptx_parser/ptx_visitor.cpp:593` | Visitor 分派 | 1 |
| 8 | `src/ptx_parser/ptx_visitor_dispatch.cpp:47` | Visitor 类别分派 | 1 |
| 9 | `src/ptxsim/instruction_factory.cpp:18` | Handler 注册 | 1 |
| 10 | `src/ptxsim/instruction_handlers.cpp:190` | Handler 实现 | 1 |

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-15`

## What Changes

- **审查** 10 个 X-Macro 使用点，确认每个是否需要独立展开
- **统一** X 宏的参数命名和展开模式
- **评估** 是否可通过模板/代码生成减少重复展开
- **不修改** `ptx_op.def` 的 106 个条目本身
- **不改变** 指令注册或分派逻辑

## Capabilities

### New Capabilities

（无新增能力--X-Macro 使用优化）

### Modified Capabilities

- `x-macro-dispatch`: 统一 10 个展开点的 X 宏定义模式，减少潜在的不一致风险
- `instruction-factory`: `instruction_factory.cpp` 和 `instruction_handlers.cpp` 的
  X-Macro 展开模式统一

## Impact

**受影响代码**（审查 + 可能修改）：
- `include/ptx_ir/ptx_types.h`（枚举生成）
- `include/ptx_parser/ptx_parser.h`（Listener 声明）
- `include/ptx_parser/ptx_visiter.h`（Visitor 声明）
- `include/ptxsim/instruction_handlers.h`（Handler 类声明）
- `src/ptx_ir/statement_context.cpp`（字符串转换）
- `src/ptx_parser/ptx_parser.cpp`（Listener 实现）
- `src/ptx_parser/ptx_visitor.cpp`（Visitor 分派）
- `src/ptx_parser/ptx_visitor_dispatch.cpp`（Visitor 类别分派）
- `src/ptxsim/instruction_factory.cpp`（Handler 注册）
- `src/ptxsim/instruction_handlers.cpp`（Handler 实现）

**不受影响**：
- `include/ptx_ir/ptx_op.def`（106 个条目不变）
- `InstructionFactory::get_handler()` 分派逻辑
- 指令执行行为

**依赖**：
- 无前置 change 依赖，可独立执行
- 建议在 `dedupe-ptx-op-def-format` 之后执行（相关区域）

**工时**: 2-3h（审查 10 个展开点 + 统一模式 + 全量验证）

## Design-Time Checklist

- [ ] 确认 10 个 X-Macro 展开点的完整清单和各自用途
- [ ] 确认每个展开点的 X 宏参数命名是否一致
- [ ] 确认 `instruction_factory.cpp` 和 `instruction_handlers.cpp` 是否可合并展开
- [ ] 确认 TCGEN05_INSTR 的特殊跳过逻辑（11 个条目共享 1 个 handler）
- [ ] 确认 parser 端的 X-Macro 使用不受影响
- [ ] 评估编译时间影响（improvement 要求"编译时间不增加"）
