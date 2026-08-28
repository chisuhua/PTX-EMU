# ci-drift-check Specification (Delta)

## ADDED Requirements

### Requirement: drift_check MUST enforce no bare IR type names outside `include/ptx_ir/` shim (Invariant 8)

The `.github/workflows/drift_check.yml` workflow MUST include an 8th invariant that scans caller source files for unqualified IR type names. It MUST scan `src/`, non-canonical `include/ptxsim/`, `include/ptxemu/`, `include/cudart/`, `include/ptx_parser/`, `include/register/`, `include/utils/`, non-shim `include/ptx_ir/`, `include/ptxir/`, and `tests/`. It MUST exclude the three forwarding shims and canonical definition headers under `include/ptxemu/ir/`. The token set MUST include at least `StatementType`, `OperandType`, `InstructionState`, `Qualifier`, `OperandContext`, `InstrVariant`, `Tcgen05Instr`, `Tcgen05OpKind`, and `Tcgen05Dtype`. The implementation MUST exclude `ptxemu::ir::`-qualified occurrences using a token-aware filter, not a bare `\\bType\\b` grep. This guards against regressions where new caller code uses bare IR names instead of the canonical qualified form.

#### Scenario: Invariant 8 grep configuration
- **WHEN** reading `.github/workflows/drift_check.yml` Invariant 8 step
- **THEN** the step uses a token-aware Python scanner across the caller roots
- **AND** the step excludes the three forwarding shim headers and all canonical definition headers under `include/ptxemu/ir/`
- **AND** the scanner handles `//`/`/* */` comments, char literals, ordinary strings, and C++ raw strings without reporting tokens inside them
- **AND** the scanner ignores qualified tokens and bare names lexically inside `namespace ptxemu::ir` canonical definition blocks

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

#### Scenario: Invariant 8 exempts shim and canonical definition headers
- **WHEN** `include/ptx_ir/ptx_types.h` contains `using ::ptxemu::ir::Qualifier;` and `include/ptxemu/ir/statement.h` contains `Qualifier dataType;`
- **AND** drift_check workflow runs
- **THEN** the scanner excludes both the forwarding shim and canonical definition header, while still scanning non-shim `include/ptx_ir/` headers and all `include/ptxir/` headers
- **AND** the workflow exits with zero status for this step
