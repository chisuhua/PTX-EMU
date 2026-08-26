# statement-ir-public Specification (Delta)

## ADDED Requirements

### Requirement: `src/ptx_ir/*.cpp` function implementations MUST reside in `ptxemu::ir` namespace

All public free function implementations declared in `include/ptxemu/ir/ptx_types.h` (currently `Q2s`, `S2s`, `Q2bytes`, `extractREG`) MUST be defined inside a `namespace ptxemu::ir { ... }` block in the corresponding `src/ptx_ir/ptx_types.cpp` translation unit, with function names qualified as `ptxemu::ir::Q2s` / `ptxemu::ir::S2s` / etc. This is required to avoid ODR conflicts when `include/ptx_ir/ptx_types.h` forwarding shim's `using ::ptxemu::ir::Q2s;` declaration meets the canonical `ptxemu::ir::Q2s` declaration from `include/ptxemu/ir/ptx_types.h`.

#### Scenario: ODR conflict resolution
- **WHEN** both `include/ptxemu/ir/ptx_types.h` (declaration) and `src/ptx_ir/ptx_types.cpp` (definition wrapped in `namespace ptxemu::ir`) are present in a single translation unit
- **THEN** the compiler does not report "conflicts with a previous declaration" for `Q2s` / `S2s` / `Q2bytes` / `extractREG`

#### Scenario: Out-of-line class method definitions in `ptxemu::ir` namespace
- **WHEN** `src/ptx_ir/statement_context.cpp` defines out-of-line methods (e.g., `StatementContext::toString()` at line ~52) of classes originally declared in `include/ptxemu/ir/statement.h` (which is now in `ptxemu::ir` namespace via shim)
- **THEN** the definitions are either wrapped in `namespace ptxemu::ir { ... }` block or qualified as `ptxemu::ir::StatementContext::toString()`, such that the ODR-mandated single-definition-per-program rule is satisfied

### Requirement: `using` declarations in `include/ptx_ir/*.h` shim headers MUST use fully-qualified `::ptxemu::ir::TypeName` form

The forwarding shim headers under `include/ptx_ir/{ptx_types,operand_context,statement_context}.h` MUST use the fully-qualified `using ::ptxemu::ir::TypeName;` form (with leading `::` to anchor at the global namespace) for each type alias, rather than unqualified `using ptxemu::ir::TypeName;` or `using namespace ptxemu::ir;`. This guarantees the shim works correctly regardless of which namespace the shim itself is included from.

#### Scenario: Fully-qualified using declarations
- **WHEN** reading `include/ptx_ir/ptx_types.h`
- **THEN** all `using` declarations start with `::ptxemu::ir::` (or `using ::ptxemu::ir::TypeName;`)

### Requirement: `namespace ptx_ir` alias for explicit opt-in

`include/ptx_ir/{ptx_types,operand_context,statement_context}.h` MUST also provide `namespace ptx_ir = ::ptxemu::ir;` namespace alias, allowing legacy code to write `ptx_ir::Qualifier` explicitly (matching the original global-namespace access path) and resolve to the canonical `ptxemu::ir::Qualifier` type.

#### Scenario: Explicit ptx_ir:: namespace access
- **WHEN** caller writes `ptx_ir::Qualifier q = ptxemu::ir::Qualifier::Q_F32;` after `#include "ptx_ir/ptx_types.h"`
- **THEN** the type resolves correctly (both `ptx_ir::` and `ptxemu::ir::` resolve to the same canonical type via namespace alias)
