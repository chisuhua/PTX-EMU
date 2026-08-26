# ci-drift-check Specification (Delta)

## ADDED Requirements

### Requirement: drift_check MUST enforce no bare IR type names outside `include/ptx_ir/` shim (Invariant 8)

The `.github/workflows/drift_check.yml` workflow MUST include an 8th invariant that scans all source files outside the three `include/ptx_ir/{ptx_types,operand_context,statement_context}.h` shim headers (including non-shim `include/ptx_ir/` and all `include/ptxir/`) for the unqualified IR type names `Qualifier`, `StatementContext`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, `Tcgen05Dtype` and fails the workflow if any are found. The implementation MUST exclude `ptxemu::ir::`-qualified occurrences using a negative lookbehind or equivalent token-aware filter, not a bare `\bType\b` grep. This guards against regressions where new code uses bare IR type names instead of the canonical `ptxemu::ir::TypeName` qualified form.

#### Scenario: Invariant 8 grep configuration
- **WHEN** reading `.github/workflows/drift_check.yml` Invariant 8 step
- **THEN** the step uses a token-aware Python scanner (or PCRE negative lookbehind equivalent) across `src/`, `include/ptxsim/`, `include/ptxemu/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, `include/utils/`, `include/ptx_ir/`, `include/ptxir/`, `tests/`
- **AND** the step excludes only the three forwarding shim headers `include/ptx_ir/ptx_types.h`, `include/ptx_ir/operand_context.h`, and `include/ptx_ir/statement_context.h` (where `using` declarations are required)
- **AND** the step excludes any matches inside `//` comments and string literals (which are not type references)

#### Scenario: Invariant 8 fails on bare type
- **WHEN** a new file `src/ptxsim/instructions/foo.cpp` contains the line `Qualifier q = Qualifier::Q_F32;` (no `ptxemu::ir::` prefix)
- **AND** drift_check workflow runs on a PR that introduces this file
- **THEN** Invariant 8 grep matches the unqualified `Qualifier` token
- **AND** the workflow exits with non-zero status
- **AND** the PR cannot be merged (per HSK-8 §5 hard-fail on workflow)

#### Scenario: Invariant 8 passes on qualified type
- **WHEN** a file `src/ptxsim/instructions/foo.cpp` contains the line `ptxemu::ir::Qualifier q = ptxemu::ir::Qualifier::Q_F32;`
- **AND** drift_check workflow runs
- **THEN** the token-aware scanner does not match the `Qualifier` token because it is preceded by the `ptxemu::ir::` qualification
- **AND** the workflow exits with zero status for this step

#### Scenario: Invariant 8 exempts shim headers
- **WHEN** `include/ptx_ir/ptx_types.h` contains `using ::ptxemu::ir::Qualifier;` (mandatory shim declaration)
- **AND** drift_check workflow runs
- **THEN** the scanner excludes `include/ptx_ir/ptx_types.h` (the shim header is the canonical location for these declarations), while still scanning non-shim `include/ptx_ir/` headers and all `include/ptxir/` headers
- **AND** the workflow exits with zero status for this step
