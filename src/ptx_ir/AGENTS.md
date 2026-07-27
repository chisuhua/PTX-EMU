# src/ptx_ir — IR Types & PTXIR Serialization

## OVERVIEW
PTX-EMU 中间表示层：PTX 指令 C++ 类型（`StatementContext`/`OperandContext`）、106 条 X-Macro 指令枚举（`ptx_op.def`）、Qualifier 定义（`ptx_qualifier.def`）、PTXIR 二进制序列化。

## STRUCTURE
```
include/ptx_ir/
├── ptx_op.def               # X-Macro: 106 条目
├── ptx_qualifier.def        # Qualifier 枚举
├── statement_context.h       # InstrVariant (std::variant<28>)
├── operand_context.h         # 6 种操作数类型
├── ptxir_format.h            # 二进制格式常量
├── ptx_types.h / ptx_context.h / kernel_context.h
└── statement_factory.h
src/ptx_ir/
├── ptxir_writer.cpp / ptxir_reader.cpp
├── statement_context.cpp / operand_context.cpp / ptx_types.cpp
└── instruction_latency_table.cpp
```

## WHERE TO LOOK

| 操作 | 文件 |
|------|------|
| 注册新指令 | `ptx_op.def` (X-Macro) |
| 加 struct + variant | `statement_context.h` |
| Qualifier 枚举 | `ptx_qualifier.def` |
| 操作数类型 | `operand_context.h` |
| PTXIR 格式 | `ptxir_format.h` |
| PTXIR 写入/读取 | `ptxir_writer.cpp` / `ptxir_reader.cpp` |

## CONVENTIONS

- **X-Macro**: `#define X(name) #include "ptx_op.def" #undef X` — 枚举/handler 映射/字符串表
- **ptx_op.def**: `X(enum, cpp_name, string, op_count, struct_kind, category)` — struct_kind 决定 InstrVariant 类型
- **InstrVariant**: `std::variant<28>` — 每个 StatementContext 持有；`std::visit` 分发
- **Qualifier 判断**: 用 `isFloat()` / `isInt()` / `isBit()`，不直接比较 `.u32` 字符串
- **PTXIR**: little-endian；header 24B + TOC 6B/entry + string table

## ANTI-PATTERNS

- ❌ `ptx_op.def` 加条目但不加 struct + InstrVariant 条目 → 编译期 variant 错误
- ❌ 改 PTXIR writer 不更新 reader → 二进制不兼容
- ❌ `std::get<T>(data)` 前不检查 `stmt.type` → `std::bad_variant_access`
- ❌ 字符串比较 Qualifier → 用 `Qualifier::isBit()` 等方法
- ❌ reader 硬编码指令尺寸 → 从 `ptxir_format.h` 常量读取

## COMMANDS

```bash
grep -c '^X(' include/ptx_ir/ptx_op.def          # 统计指令数
grep 'GENERIC_INSTR' include/ptx_ir/ptx_op.def   # 列通用指令
ctest -R unit_ptxir                               # PTXIR 格式测试
```