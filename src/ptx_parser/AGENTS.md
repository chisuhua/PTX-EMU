# src/ptx_parser/ AGENTS.md
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents ptx_parser-specific content.

## OVERVIEW

ANTLR4 AST 处理层 — 将 grammar 输出的 CST 转换为 IR（StatementContext + OperandContext），供 ptxsim 执行引擎消费。

## STRUCTURE

```
src/ptx_parser/
├── ptx_visitor.cpp          # PtxVisitor: ANTLR BaseVisitor 子类, 递归 AST 遍历
│                             #   visitFunctionDecl → visitInstruction → visitOperand
├── ptx_visitor_generic.cpp  # GENERIC 类指令 (add/sub/mul/…)
├── ptx_visitor_atom.cpp     # ATOM 类指令 (atom/add/and/cas/…)
├── ptx_visitor_call.cpp     # CALL 类指令 (call/call.uni/ret)
├── ptx_visitor_branch.cpp   # BRANCH 类指令 (bra/brx/…)
├── ptx_visitor_barrier.cpp  # BARRIER 类指令 (bar.sync/bar.arrive/…)
├── ptx_visitor_memory.cpp   # 访存类指令 (ld/st/…)
├── ptx_visitor_simple.cpp   # SIMPLE 类指令 (mov/cvt/…)
├── ptx_visitor_special.cpp  # 特殊指令 (ex2/rcp/rsqrt/…)
├── ptx_visitor_warp.cpp     # Warp 级指令 (shfl/vote/…)
├── ptx_visitor_abi.cpp      # ABI 指令 (.param/.callprototype/…)
├── ptx_parser.cpp           # PtxListener: ANTLR BaseListener 树遍历器
│                             #   exitExternFuncStatement (line 956)
└── cfg_builder.cpp          # CFG 构建 + 后支配分析
    └── cfg_builder.h        #   build(), computePostDominators(), findImmediatePostDominator()
```

### PtxVisitor sub-file layout (split-ptx-visitor-god-class, 2026-07)

The 1067-line `ptx_visitor.cpp` has been split into focused sub-files,
each #include'd by the parent (NOT separate CMake targets). Pattern matches
the pre-existing `ptx_visitor_warp.cpp` / `ptx_visitor_memory.cpp` family.

| Sub-file | Responsibility |
|----------|----------------|
| `ptx_visitor.cpp` | Parent file: visitPtxFile, visitFunctionDecl, visitInstruction dispatch, top-level visitors |
| `ptx_visitor_operands.cpp` | Helpers (tokenToQualifier, extractQualifiersFromContext, createOperandFromContext) + operand visitors (visitOperand, visitSpecialRegister, visitRegister, visitImmediate, visitAddress). Includes local copy of parseRegisterFromText in anonymous namespace |
| `ptx_visitor_dispatch.cpp` | Include aggregation area (11 sub-file includes) + outer X-Macro (VISITOR_##struct_kind dispatch). Owns VISITOR_TCGEN05_INSTR override |
| `ptx_visitor_tcgen05.cpp` | visitTcgen05Inst (Blackwell tensor core handler, ADR-0016) |
| `ptx_visitor_generic.cpp` ... `ptx_visitor_abi.cpp` | Per-category instruction handlers (pre-existing) |

## WHERE TO LOOK

| Task | Location |
|------|----------|
| 指令类别分派 | `include/ptx_parser/ptx_visitor_categories.h` (X-Macro 类别宏) |
| PtxVisitor 声明 | `include/ptx_parser/ptx_visiter.h` (visitXxxInst 声明, X-Macro 生成) |
| PtxListener 声明 | `include/ptx_parser/ptx_parser.h` (enterXxx/exitXxx 声明, X-Macro 生成) |
| AST 入口 & 指令分发 | `ptx_visitor.cpp` (visitFunctionDecl, visitInstruction, 操作数解析) |
| 树遍历器入口 | `ptx_parser.cpp` (PtxListener, 解析 PTX 文本到 IR) |
| call/ret 分支 | `ptx_visitor_call.cpp` |
| barrier 分支 | `ptx_visitor_barrier.cpp` |
| 指令定义 | `include/ptx_ir/ptx_op.def` (X-Macro: opname, opstr, opcount, struct_kind) |
| 语法文件 | `src/grammar/ptxInstructions.g4` (597 行, ~200 指令规则) |
| 生成代码 | `build/antlr4_generated_src/ptxparser/` (NEVER 手动编辑) |

## CONVENTIONS

- **PtxVisitor**: 继承 `ptxParserBaseVisitor`, `visitXxxInst()` 方法通过 X-Macro 自动声明
- **PtxListener**: 继承 `ptxParserBaseListener`, `enterXxx/exitXxxStatement()` 通过 X-Macro 自动声明
- **类别分派**: `struct_kind` 字段决定 visitor 方法落入哪个 `.cpp` 文件 (generic/atom/call/branch/barrier/etc.)
- **操作数解析**: `ptx_visitor.cpp` 内含 `visitOperand`, `visitRegister`, `visitImmediate` 等通用解析
- **CFG**: 静态方法, `build()` → `identifyBasicBlocks()` → `buildEdges()` → `computePostDominators()`

## ANTI-PATTERNS

- ❌ 直接修改 `build/antlr4_generated_src/`（被覆盖）
- ❌ 改 `.g4` 不加载 `ptx-grammar-modification` skill（缺失强制 checklist）
- ❌ 改 `.g4` 后跳过 `test_all_ptx.sh`（PTX 语法回归）
- ❌ 改 `.g4` 后不运行 `cmake --build build --target GenerateParser`（解析器不更新）
- ❌ 在 visitor 中混合 listener 和 visitor 两种遍历模式（维护成本翻倍）

## COMMANDS

```bash
# 改 .g4 后重新生成解析器
cmake --build build --target GenerateParser

# PTX 语法验证（所有 tests/ptx/*.ptx）
./tests/ptx/test_all_ptx.sh

# 新增 visitor 类别: 1) 改 ptx_op.def 加 struct_kind
# 2) 改 ptx_visitor_categories.h 加 VISITOR_xxx_INSTR 宏
# 3) 新建 ptx_visitor_xxx.cpp 实现 visit 方法
```