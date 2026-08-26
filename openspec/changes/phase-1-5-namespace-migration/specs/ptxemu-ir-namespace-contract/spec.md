# ptxemu-ir-namespace-contract Specification

## Purpose
TBD - created by archiving change phase-1-5-namespace-migration. Update Purpose after archive.

## ADDED Requirements

### Requirement: `ptxemu::ir` MUST be the canonical namespace for all PTX-EMU IR public types

All PTX-EMU IR public type definitions (Qualifier, StatementType, OperandType, OperandKind, InstructionState, 20+ instruction structs, 6 operand variant types, StatementContext, InstrVariant) MUST be defined exclusively in `include/ptxemu/ir/` headers and wrapped in `namespace ptxemu { namespace ir { ... } }`. The `include/ptx_ir/` directory MUST contain only forwarding shims (per `task 9.4` 1 release cycle), not duplicate type definitions.

#### Scenario: IR type location verification
- **WHEN** `git grep -E "^(enum|struct|class)\s+(Qualifier|StatementType|StatementContext|OperandContext|Tcgen05Instr)\b" include/`
- **THEN** all matches are within `include/ptxemu/ir/` headers, not within `include/ptx_ir/` headers (which contain only `using` declarations + `#include <ptxemu/ir/...>`)

#### Scenario: Namespace boundary check
- **WHEN** reading any header in `include/ptxemu/ir/`
- **THEN** the type definitions are inside `namespace ptxemu { namespace ir { ... } }` block, not in global namespace

### Requirement: Old `include/ptx_ir/*.h` path MUST function as forwarding shim

`include/ptx_ir/{ptx_types,operand_context,statement_context}.h` MUST function as backward-compatible forwarding shims. Each shim MUST:
1. `#include <ptxemu/ir/<corresponding>.h>` to pull in canonical definitions
2. Provide explicit `using ::ptxemu::ir::TypeName;` declarations for each type in the canonical header
3. NOT redefine any type (no duplicate `enum class` / `struct` / `class` declarations)
4. Provide optional `namespace ptx_ir = ::ptxemu::ir;` alias for explicit opt-in

#### Scenario: Old path preserves unqualified type access
- **WHEN** legacy caller writes `Qualifier q = Qualifier::Q_F32;` after `#include "ptx_ir/ptx_types.h"`
- **THEN** the type `Qualifier` resolves to `ptxemu::ir::Qualifier` via the shim's `using` declaration, compilation succeeds

#### Scenario: Old path preserves struct constructors
- **WHEN** legacy caller writes `StatementContext{type, data}` after `#include "ptx_ir/statement_context.h"`
- **THEN** constructor ADL lookup finds `ptxemu::ir::StatementContext::StatementContext`, compilation succeeds

#### Scenario: Old path with explicit namespace alias
- **WHEN** legacy caller writes `ptx_ir::Qualifier q = ptxemu::ir::Qualifier::Q_F32;` after `#include "ptx_ir/ptx_types.h"`
- **THEN** both `ptx_ir::` and `ptxemu::ir::` namespace prefixes resolve to the same canonical type (namespace alias equality)

### Requirement: Canonical header must include complete function declarations

`include/ptxemu/ir/ptx_types.h` MUST declare all public free functions in the canonical namespace (`Q2s(Qualifier)`, `S2s(StatementType)`, `Q2bytes(Qualifier)`, `extractREG(std::string, int&, std::string&)`). Implementations live in `src/ptx_ir/ptx_types.cpp` wrapped in `namespace ptxemu::ir { ... }` block with `ptxemu::ir::` function-name prefix to avoid ODR conflicts with any legacy forward declarations.

#### Scenario: Free function linkage check
- **WHEN** linking any executable that includes `ptxemu/ir/ptx_types.h` (directly or via shim) and uses `Q2s(q)` / `S2s(s)` / `Q2bytes(q)` / `extractREG(s, idx, name)`
- **THEN** the linker finds definitions in `libptx_ir.so` / `libptxemu_core.so` etc, no undefined reference errors

#### Scenario: No ODR conflict
- **WHEN** both `include/ptxemu/ir/ptx_types.h` (declaration) and `src/ptx_ir/ptx_types.cpp` (definition) are present in a single translation unit
- **THEN** the compiler does not report "conflicts with a previous declaration" for `Q2s` / `S2s` / `Q2bytes` / `extractREG` (verified during Phase 1.5c+d trial)

### Requirement: All internal callers MUST use `ptxemu::ir::*` qualified names

All 178 src/include caller sites in `src/ptx_parser/`, `src/ptxsim/`, `src/cudart/`, `include/{ptxsim,ptxemu,cudart,ptx_parser,register,utils}/`, and `tests/{unit,integration,e2e}/` MUST use the qualified `ptxemu::ir::TypeName` form for any IR type. Bare unqualified IR type names (`Qualifier` without prefix, `StatementContext` without prefix, etc.) MUST NOT appear outside:
- The `include/ptx_ir/*.h` forwarding shim headers (where `using` declarations are mandatory)
- Comments and string literals (where the type name appears as documentation, not as a type reference)

#### Scenario: No bare IR types in src/
- **WHEN** `git grep -E "\b(Qualifier|StatementContext|OperandContext|InstrVariant|Tcgen05Instr|Tcgen05OpKind|Tcgen05Dtype)\b" src/`
- **THEN** every match is preceded by `ptxemu::ir::` (or appears in a comment / string literal)

#### Scenario: No bare IR types in include/ subdirectories
- **WHEN** `git grep -E "\b(Qualifier|StatementContext|OperandContext|InstrVariant|Tcgen05Instr|Tcgen05OpKind|Tcgen05Dtype)\b" include/ptxsim/ include/ptxemu/ include/cudart/ include/ptx_parser/ include/register/ include/utils/`
- **THEN** every match is preceded by `ptxemu::ir::` (or appears in a comment / string literal)

#### Scenario: No bare IR types in tests/
- **WHEN** `git grep -E "\b(Qualifier|StatementContext|OperandContext|InstrVariant|Tcgen05Instr|Tcgen05OpKind|Tcgen05Dtype)\b" tests/`
- **THEN** every match is preceded by `ptxemu::ir::` (or appears in a comment / string literal)

### Requirement: GPUContext interface MUST use `ptxemu::ir::StatementContext`

`include/ptxsim/gpu_context.h` line 58, 80, 173 MUST use `std::vector<ptxemu::ir::StatementContext>` instead of `std::vector<StatementContext>`. This includes:
- The `statements` member field at line 58
- The `execute_kernel_internal` parameter at line 80
- The `execute_kernel` parameter at line 173 (or wherever this signature is defined per current source)

#### Scenario: GPUContext type signature verification
- **WHEN** reading `include/ptxsim/gpu_context.h:58,80,173`
- **THEN** the type signatures reference `ptxemu::ir::StatementContext` (qualified)

#### Scenario: GPUContext callers compile
- **WHEN** any source file includes `ptxsim/gpu_context.h` and uses the public API
- **THEN** compilation succeeds (the canonical `ptxemu::ir::StatementContext` resolves via `#include "ptxsim/gpu_context.h"` chain which transitively includes `ptxemu/ir/statement.h`)

### Requirement: `using namespace ::ptxemu::ir;` MUST NOT appear anywhere

No source file, header, test, or generated artifact MUST use `using namespace ::ptxemu::ir;` (or `using namespace ptxemu::ir;`) to introduce IR types into another namespace. Only the explicit per-type `using ::ptxemu::ir::TypeName;` declarations inside the `include/ptx_ir/*.h` forwarding shim headers are permitted.

#### Scenario: No global using-directive
- **WHEN** `git grep -E "using\s+namespace\s+::?ptxemu::ir" src/ include/ptxsim/ include/ptxemu/ include/cudart/ include/ptx_parser/ include/register/ include/utils/ tests/`
- **THEN** 0 matches outside `include/ptx_ir/*.h` shim files

### Requirement: ANTLR4-generated headers MUST NOT be modified

Files under `build/antlr4_generated_src/` (generated by ANTLR4 runtime during CMake configure) MUST NOT be modified by this change. If PTX parser compilation fails due to IR type references in generated headers, the fix MUST be in the runtime include path or in the upstream `src/grammar/*.g4` files, not by hand-editing generated sources.

#### Scenario: No edits to generated headers
- **WHEN** `git diff --stat build/antlr4_generated_src/`
- **THEN** 0 changes (generated sources not tracked, but no new generated files should be created either)

### Requirement: `cpp 不暴露` invariant MUST be preserved

After Phase 1.5 completion, the `cpp 不暴露` constraint (HSK-8 spec §CppTLM 端接受条件 #3) MUST continue to hold. CppTLM consumer MUST NOT be able to include any `ptx_ir/` header, and MUST use `ptxemu/ir/` exclusively. The `ptxemu_core` CMake target's `target_include_directories` MUST be configured to expose `include/ptxemu/` as PUBLIC and `include/ptx_ir/` as PRIVATE (per `ptxemu-core-library/spec.md:17-18`).

#### Scenario: CppTLM include boundary check
- **WHEN** `git grep "#include\s+[<\"]ptx_ir/" CppTLM/ CppTLM/external/PTX-EMU/include/ 2>/dev/null`
- **THEN** 0 matches (CppTLM cannot reference old ptx_ir path)
