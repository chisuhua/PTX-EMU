# PTX-EMU 工作流程梳理（供 AI 编程参考）

## 1) 语法 → 解析器生成（ANTLR4）
- 语法源文件位于 [src/grammar/ptxLexer.g4](src/grammar/ptxLexer.g4) 与 [src/grammar/ptxParser.g4](src/grammar/ptxParser.g4)。
- CMake 在 [src/CMakeLists.txt](src/CMakeLists.txt) 中定义 `GenerateParser`，通过 ANTLR4 生成 C++ 解析器源码到 build/antlr4_generated_src，并编进 `cudart`。
- 生成的 visitor/listener 类（如 `ptxParserBaseVisitor`/`ptxParserVisitor`）在 build/antlr4_generated_src 下，但当前代码路径主要使用 `PtxListener`（listener 方式）。

## 2) PTX 解析与 `PtxContext` 构建路径
- `PtxListener` 定义于 [include/ptx_parser/ptx_parser.h](include/ptx_parser/ptx_parser.h)，实现于 [src/ptx_parser/ptx_parser.cpp](src/ptx_parser/ptx_parser.cpp)。
- 解析流程：ANTLR 产出的 `ptxParser` 构建 AST → `PtxListener` 在 `enter*/exit*` 回调中填充 `PtxContext`（见 [include/ptx_ir/ptx_context.h](include/ptx_ir/ptx_context.h)）。
- `KernelContext`、`StatementContext`、`OperandContext` 的结构分别位于 [include/ptx_ir/kernel_context.h](include/ptx_ir/kernel_context.h)、[include/ptx_ir/statement_context.h](include/ptx_ir/statement_context.h)、[include/ptx_ir/operand_context.h](include/ptx_ir/operand_context.h)。
- 之后由 `PtxInterpreter`（[src/cudart/ptx_interpreter.cpp](src/cudart/ptx_interpreter.cpp)）构建符号表并提交 `KernelLaunchRequest` 给 `GPUContext` 执行。

## 3) `ptx_op.def` 作为指令“中心配置”
- `ptx_op.def`（[include/ptx_ir/ptx_op.def](include/ptx_ir/ptx_op.def)）是指令集的单一来源（X-Macro）：
  - 生成 `StatementType` 枚举（[include/ptx_ir/ptx_types.h](include/ptx_ir/ptx_types.h)）。
  - 生成指令处理器声明（[include/ptxsim/instruction_handlers.h](include/ptxsim/instruction_handlers.h)）。
  - 驱动解析器对各类指令的 enter/exit 回调声明（[include/ptx_parser/ptx_parser.h](include/ptx_parser/ptx_parser.h)）。
- 新增/修改指令时，优先在该文件添加/调整，再同步实现处理逻辑（位于 src/ptxsim/instructions/）。

## 4) `ptx_qualifier.def` 作为限定符“中心配置”
- `ptx_qualifier.def`（[include/ptx_ir/ptx_qualifier.def](include/ptx_ir/ptx_qualifier.def)）定义所有限定符：类型、内存空间、舍入方式、cache/scope/async 等。
- 该文件用于生成 `Qualifier` 枚举与字符串/字节映射（[include/ptx_ir/ptx_types.h](include/ptx_ir/ptx_types.h)）。
- 新增限定符时，先更新该文件，确保解析、指令处理与类型判断一致。

## 5) 常见改动的最短路径
- **新增 PTX 指令**：更新 [include/ptx_ir/ptx_op.def](include/ptx_ir/ptx_op.def) → 实现对应 handler（src/ptxsim/instructions/）→ 若语法未覆盖，扩展 [src/grammar/ptxParser.g4](src/grammar/ptxParser.g4) 并触发生成。
- **新增限定符/类型**：更新 [include/ptx_ir/ptx_qualifier.def](include/ptx_ir/ptx_qualifier.def) → 确认解析与执行路径使用 `Qualifier`。
- **调整解析语法**：改动 g4 后通过 CMake 触发 `GenerateParser`，生成源码位于 build/antlr4_generated_src。
