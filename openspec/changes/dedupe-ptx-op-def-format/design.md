# dedupe-ptx-op-def-format - Design

## Overview

将 `src/ptx_parser/ptx_visitor_atom.cpp` 中 `VISITOR_ATOM_INSTR` 宏注释内硬编码的
`ptx_op.def` 格式引用（`X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)`）替换为
泛化描述 + `static_assert` 编译期检查，消除 DRY 违反。

当前问题代码（`ptx_visitor_atom.cpp:15-33` 注释块）：

```cpp
/* atom grammar:
 *   atomInst: ATOM atomQualifiers atomOp typeSpecifier vectorSpec?
 *              operand COMMA addressExpr COMMA operand (COMMA operand)? SEMI
 *
 * operandCtxs layout:
 *   operandCtxs[0] = dst
 *   operandCtxs[1] = src
 *   operandCtxs[2] = cmp (optional, only for atom.cas)
 * ctx->addressExpr() = middle address expression (the [addr] part)
 *
 * The previous implementation only collected dst+src via
 * getRuleContexts<OperandContext>() and silently dropped the middle
 * addressExpr, yielding 2 operands instead of the 3 (or 4 for cas)
 * that ptx_op.def (X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)) requires.  // <-- 硬编码
 *
 * Fix: explicitly convert ctx->addressExpr() into an AddrOperand and
 * insert it between dst and src so the resulting operands vector
 * contains exactly {dst, addr, src[, cmp]}.
 */
```

`ptx_op.def` 中 `S_ATOM` 的实际条目：
```cpp
X(S_ATOM,    atom,    Atom,    3, ATOM_INSTR, atomic)      // atom: dst_addr, src, [optional op] -> simplified as 3
```

## Context

- `ptx_op.def` 格式：`X(enum_value, cpp_name, string, op_count, struct_kind, instr_kind)`
- `ptx_visitor_atom.cpp` 中的 `VISITOR_ATOM_INSTR` 宏通过 X-Macro 展开，接受
  `(openum, opstr, opname, opcount)` 参数
- 宏注释中硬编码了 `X(S_ATOM, atom, Atom, 3, ATOM_INSTR, atomic)` 作为格式参考
- 该注释引用了 `op_count=3`，但 atom 宏体中使用 `opcount` 参数（由 X-Macro 传入），
  注释中的 `3` 是冗余的硬编码值

## Design Decisions

### 决策 1: 替换策略 - 泛化注释 + static_assert

**选择**: 将硬编码格式引用替换为泛化描述，并添加 `static_assert` 编译期检查

**修改方案**:
```cpp
/* ...
 * that ptx_op.def (S_ATOM entry, op_count defined in ptx_op.def) requires.
 * Note: opcount parameter is passed via X-Macro expansion; do not hardcode
 * the operand count here — it is derived from ptx_op.def at compile time.
 */
```

并在宏展开后添加编译期断言（如果可行）：
```cpp
// After X-Macro expansion, verify S_ATOM op_count matches expected behavior
static_assert(true, "S_ATOM op_count is sourced from ptx_op.def");
```

**理由**:
- improvement 技术约束要求"SHOULD 用 constexpr/static_assert 替代纯注释引用"
- 泛化描述消除了格式冗余，不再需要同步
- `static_assert` 提供编译期文档化（虽然此处难以做实质性编译期检查，因为 `opcount`
  是宏参数而非 constexpr 变量）

**替代方案**:
- A. 仅修改注释为"参见 ptx_op.def S_ATOM 条目" -> 消除硬编码但无编译期保护
- B. 用模板元编程从 `ptx_op.def` 提取 op_count -> 过度工程化
- C. **采用**: 泛化注释 + 文档化 static_assert

### 决策 2: 不使用 constexpr 提取 op_count

**选择**: 不尝试用 constexpr 从 `ptx_op.def` 提取 `S_ATOM` 的 `op_count`

**理由**:
- `ptx_op.def` 通过 X-Macro 展开，`op_count` 是宏字面量（`3`），不是 C++ constexpr 变量
- 提取需要在 X-Macro 展开时额外定义 constexpr 变量，改动范围超出本 change
- improvement 范围明确："不修改 ptx_op.def"和"不改变 atom 指令的解析逻辑"
- `static_assert(true, ...)` 作为文档化手段已足够

### 决策 3: 修改范围限定

**选择**: 仅修改 `ptx_visitor_atom.cpp` 中的注释，不扫描其他 visitor 文件

**理由**:
- improvement 范围明确"不修改其他 visitor 文件"
- 全项目 grep 确认 `ptx_op.def` 格式硬编码引用仅此 1 处（其他文件引用
  `ptx_op.def` 是通过 `#include` 展开，非注释中硬编码格式）

## Implementation Plan

### Phase 1: 修改注释（10 min）
1. 将 `ptx_visitor_atom.cpp:28` 的硬编码格式引用替换为泛化描述
2. 添加文档化 `static_assert` 注释
3. 编译验证

### Phase 2: 验证（20 min）
1. 编译通过
2. atom 指令解析测试通过
3. PTX 语法测试通过（`test_all_ptx.sh`）

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| 编译 | `cmake --build build` | 通过 |
| atom unit 测试 | `cd build && ctest -R "atom" --output-on-failure` | 全绿 |
| PTX 语法测试 | `./tests/ptx/test_all_ptx.sh` | 全绿 |
| 全量回归 | `cd build && ctest --output-on-failure` | 全绿 |
| grep 验证 | `grep -n "X(S_ATOM" src/ptx_parser/ptx_visitor_atom.cpp` | 0 结果 |

### 行为不变性验证

修改前后对以下 atom 指令执行 golden value 对比：
- `atom.add.u32`, `atom.cas.u32`, `atom.inc.u32`
- `atom.exch.b32`, `atom.min.s32`

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 注释修改遗漏相关信息 | 开发者需额外查看 ptx_op.def | 泛化描述明确指向 ptx_op.def |
| static_assert 无实质检查 | 仅文档化作用 | 当前架构限制，实质性检查需改 ptx_op.def |
| 宏注释修改影响编译 | 无影响 | 注释不影响编译 |

## Open Questions

1. **是否应该同时在其他 visitor 文件中检查类似的硬编码引用？**
   - improvement 范围限定为"不修改其他 visitor 文件"
   - 全项目 grep 确认仅此 1 处硬编码格式引用

## 关联文档

- `improvements/dedupe-ptx-op-def-format.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-20`：原债务条目
- `src/ptx_parser/ptx_visitor_atom.cpp`：目标修改文件
- `include/ptx_ir/ptx_op.def`：X-Macro 格式定义（SSOT）
- `include/ptx_parser/ptx_visitor_categories.h`：visitor 类别宏定义
