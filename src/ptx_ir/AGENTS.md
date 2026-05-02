# PTX IR + PTXIR Serialization

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
PTX intermediate representation types (operand/statement contexts), X-Macro instruction definitions, and PTXIR binary serialization.

## STRUCTURE
```
src/ptx_ir/              # ptxir_writer.cpp, ptxir_reader.cpp
include/ptx_ir/          # statement_context.h, operand_context.h, ptx_op.def, ptx_qualifier.def
src/ptxir/               # ptxir_serialization.cpp (public API)
include/ptxir/           # ptxir_serialization.h (public API header)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add new PTX instruction | `include/ptx_ir/ptx_op.def` | X-Macro pattern: `X(S_NAME, op_name, ClassName, operands, kind, category)` |
| Modify statement types | `include/ptx_ir/statement_context.h` | InstrVariant, StatementContext, all instr structs |
| Operand types | `include/ptx_ir/operand_context.h` | RegOperand, ImmOperand, VariableOperand, etc. |
| PTXIR serialize | `include/ptxir/ptxir_serialization.h` | `serialize_statements()`, `serialize_to_string()` |
| PTXIR deserialize | `include/ptxir/ptxir_serialization.h` | `deserialize_statements()`, `deserialize_from_string()` |
| Binary format spec | `include/ptx_ir/ptxir_format.h` | Header, instruction encoding sizes |

## CONVENTIONS (this dir)
- **X-Macro pattern**: `#define X(name, ...)` + `#include "ptx_op.def"` + `#undef X` — used for enum generation, string tables, and handler dispatch
- **InstrVariant**: `std::variant<28 types>` — every statement holds one variant; visitor dispatches by actual type
- **PTXIR format**: Custom binary (`.ptxir`). Header (32B) → String Table → Kernel Section. Each statement: `type(u16) + type-specific data`
- When adding a new instruction type to `ptx_op.def`, you MUST also add it to `InstrVariant` in `statement_context.h`

## ANTI-PATTERNS
- DO NOT add a statement type to `ptx_op.def` without adding its struct + variant entry in `InstrVariant`
- DO NOT modify PTXIR writer format without updating the reader in `ptxir_reader.cpp`
- DO NOT use `std::get<InstrType>(stmt.data)` without checking `stmt.type` first (use `stmt.type == S_XXX` guard)

## KEY FILES
| File | Purpose |
|------|---------|
| `ptx_op.def` | 202 PTX instruction X-Macro definitions |
| `ptx_qualifier.def` | Qualifier X-Macro definitions |
| `statement_context.h` | StatementContext, InstrVariant, all instruction structs |
| `operand_context.h` | OperandContext variant and operand types |
| `ptxir_writer.cpp` | PTXIR binary serialization writer |
| `ptxir_reader.cpp` | PTXIR binary deserialization reader |
| `ptxir_format.h` | Binary format constants and struct sizes |

## COMMANDS
```bash
cmake --build build --target ptx_ir       # IR library
cmake --build build --target ptxir        # Serialization library (ptxir_writer + ptxir_reader)
./build/bin/tests/test_ptxir_serialization  # Run PTXIR roundtrip tests
```
