# ci-drift-check Specification (Delta)

## ADDED Requirements

### Requirement: drift_check MUST enforce no bare IR type names outside `include/ptx_ir/` shim (Invariant 8)

The `.github/workflows/drift_check.yml` workflow MUST include an 8th invariant that scans all source files outside `include/ptx_ir/*.h` shim headers for the unqualified IR type names `Qualifier`, `StatementContext`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, `Tcgen05Dtype` and fails the workflow if any are found. This guards against regressions where new code uses bare IR type names instead of the canonical `ptxemu::ir::TypeName` qualified form.

#### Scenario: Invariant 8 grep configuration
- **WHEN** reading `.github/workflows/drift_check.yml` Invariant 8 step
- **THEN** the step greps for each of the 7 IR type names with `\b` word boundaries across `src/`, `include/ptxsim/`, `include/ptxemu/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, `include/utils/`, `tests/`
- **AND** the step excludes any matches that appear inside `include/ptx_ir/*.h` shim headers (where `using` declarations are required)
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
- **THEN** Invariant 8 grep does not match the bare `Qualifier` (the `ptxemu::ir::` prefix anchors the regex)
- **AND** the workflow exits with zero status for this step

#### Scenario: Invariant 8 exempts shim headers
- **WHEN** `include/ptx_ir/ptx_types.h` contains `using ::ptxemu::ir::Qualifier;` (mandatory shim declaration)
- **AND** drift_check workflow runs
- **THEN** the grep on `include/ptx_ir/ptx_types.h` is excluded from the check (the shim header is the canonical location for these declarations)
- **AND** the workflow exits with zero status for this step
