# Phase 0 — scanner + baseline freeze

**Branch:** `feat/phase-1-5-namespace-migration`
**Date:** 2026-08-26
**OpenSpec change:** `openspec/changes/phase-1-5-namespace-migration/`

## Tasks completed

### 0.1 Scanner created
- File: `scripts/check_ptxemu_ir_names.py` (committed in c71f2b1c)
- Per design.md D4:
  - `--roots` accepts directories or single files
  - `--exclude` repeatable, matches full path suffix
  - `--list-files` deterministic enumeration with src/include/tests breakdown
  - Lexer strips `//` and `/* */` comments, char/string/raw-string literals
  - Identifier-following `::` consumed in lex so `ptxemu::ir::Token` stays adjacent
  - Walk-back qualified-name detection: `Qualifier` preceded by `[ir, ptxemu]` passes
  - Canonical `include/ptxemu/ir/` excluded by path
  - Token set: `StatementType`, `OperandType`, `InstructionState`, `Qualifier`,
    `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`,
    `Tcgen05Dtype`, `StatementContext`

### 0.1.b Fixture validation
| Fixture                                          | Expected | Result |
|--------------------------------------------------|----------|--------|
| bare `Qualifier q = Qualifier::Q_F32;`           | FAIL     | FAIL   |
| `ptxemu::ir::Qualifier q = ptxemu::ir::...`      | PASS     | PASS   |
| `// Qualifier`, `/* StatementContext */`, `"..."` | PASS     | PASS   |
| `R"(Qualifier rawstring)"`                       | PASS     | PASS   |
| `include/ptxemu/ir/statement.h` (canonical def)  | PASS     | PASS   |
| `using ptxemu::ir::Qualifier;`                   | PASS     | PASS   |

### 0.2 Caller file list frozen
```
python3 scripts/check_ptxemu_ir_names.py --roots src include tests \
  --exclude include/ptx_ir/ptx_types.h \
  --exclude include/ptx_ir/operand_context.h \
  --exclude include/ptx_ir/statement_context.h \
  --list-files
```
Result:
- src: 116
- include: 108
- tests: 209
- **total: 433**

NOTE: This differs from the original artifact estimate of 218 files. The
artifact estimate likely counted only direct include consumers; the scanner
counts every `.h`/`.hpp`/`.cpp`/`.cc` file in caller roots regardless of
direct/indirect dependency. The migration scope is now defined by this
deterministic list, per tasks.md Phase 0 requirement.

### 0.3 Baseline ctest verified
```
cmake --build build -j$(nproc)   # exit 0, 100% build
cd build && ctest                 # 100% tests passed, 0 tests failed out of 252
                                 # Total Test time (real) = 83.72 sec
```

## Scope changes vs proposal.md

Proposal claimed 218 files (src 58, include 40, tests 120). The deterministic
scanner reports **433** files (src 116, include 108, tests 209). The new
figure replaces the proposal estimate as the authoritative scope. Each phase
must read its file subset from this list.

## Next phase readiness

Phase 1.5c+d (shim + cpp namespace wrap + InstructionState alias +
serialization migration) can begin. The implementation groups will:

1. `include/ptx_ir/{ptx_types,operand_context,statement_context}.h` →
   forwarding shim with canonical include + `namespace ptx_ir = ::ptxemu::ir`
   + explicit fully-qualified type `using` (type-name-only policy)
2. `src/ptx_ir/ptx_types.cpp` + canonical IR method implementations →
   wrapped in `namespace ptxemu::ir`
3. `include/ptxsim/execution_types.h` global `InstructionState` →
   `using ::ptxemu::ir::InstructionState;` (must `#include` canonical first)
4. `src/ptxir/ptxir_serialization.cpp` + `include/ptxir/ptxir_serialization.h`:
   replace `struct StatementContext` elaborated specifiers with
   `ptxemu::ir::StatementContext`, add canonical include or namespace
   forward declaration
5. Build + ctest 252/252 verification, no regressions