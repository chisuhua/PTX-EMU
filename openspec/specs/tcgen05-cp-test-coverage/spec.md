# tcgen05-cp-test-coverage Specification

## Purpose
TBD - created by archiving change tcgen05-cp-test-coverage-and-exception-cleanup. Update Purpose after archive.
## Requirements
### Requirement: tcgen05.cp handler SHALL have unit test coverage
The system SHALL provide unit tests for `tcgen05.cp` helper functions and error paths.

#### Scenario: Immediate smem offset is extracted correctly
- **WHEN** a `Tcgen05Instr` has a shared-memory source operand with an immediate offset of `0x10`
- **THEN** `extract_smem_offset_placeholder` SHALL return `16`

#### Scenario: Non-shared source address returns zero offset
- **WHEN** a `Tcgen05Instr` has a source operand that is not a shared-memory address
- **THEN** `extract_smem_offset_placeholder` SHALL return `0`

#### Scenario: Register offset returns zero as placeholder
- **WHEN** a `Tcgen05Instr` has a shared-memory source operand with a register offset
- **THEN** `extract_smem_offset_placeholder` SHALL return `0` and the test SHALL document the deferred Phase 3 behavior

#### Scenario: cta_group::2 throws UnsupportedInstructionException with ADR-0018 reference
- **WHEN** `processTcgen05Cp` is invoked with `instr.cta_group == 2`
- **THEN** it SHALL throw `UnsupportedInstructionException` and the message SHALL contain `ADR-0018`

#### Scenario: Missing shared memory throws UnsupportedInstructionException
- **WHEN** `processTcgen05Cp` is invoked with `cta->sharedMemSpace == nullptr`
- **THEN** it SHALL throw `UnsupportedInstructionException` (not `std::runtime_error`)

#### Scenario: Out-of-bounds smem access throws runtime error
- **WHEN** `processTcgen05Cp` is invoked with an smem offset such that `offset + Tmem::kSlotSize > cta->sharedMemBytes`
- **THEN** it SHALL throw an exception indicating out-of-bounds shared memory access

### Requirement: tcgen05.cp handler SHALL have integration test coverage
The system SHALL provide an integration test that executes `tcgen05.cp` through the instruction pipeline.

#### Scenario: 128-byte copy from shared memory to TMEM
- **WHEN** a warp executes a `tcgen05.cp` instruction with valid shared-memory source and TMEM destination
- **THEN** 128 bytes from the specified shared-memory offset SHALL be written to TMEM slot 0

#### Scenario: Integration test catches out-of-bounds smem access
- **WHEN** a warp executes `tcgen05.cp` with an smem offset that exceeds the declared shared memory size
- **THEN** the integration test SHALL observe an exception and the warp SHALL not corrupt memory

### Requirement: tcgen05.cp exception types SHALL be consistent
The system SHALL use `UnsupportedInstructionException` for all unsupported-environment scenarios in `tcgen05.cp`.

#### Scenario: Missing shared memory exception type matches other unsupported scenarios
- **WHEN** `processTcgen05Cp` is invoked with `cta->sharedMemSpace == nullptr`
- **THEN** the exception type SHALL be `UnsupportedInstructionException`, matching the missing WarpContext and CTAContext cases

### Requirement: tcgen05.cp placeholder behavior SHALL be explicitly tracked
The system SHALL annotate known placeholders in `tcgen05_cp.cpp` with TODO comments referencing the follow-up phase.

#### Scenario: Hardcoded destination slot is tracked
- **WHEN** reading `src/ptxsim/instructions/tcgen05_cp.cpp`
- **THEN** the `kDestSlot = 0` placeholder SHALL be accompanied by a `TODO(Phase 3 of implement-tcgen05-handlers-extended)` comment

#### Scenario: Shape qualifier placeholder is tracked
- **WHEN** reading `src/ptxsim/instructions/tcgen05_cp.cpp`
- **THEN** the 128-byte fixed transfer and shape qualifier deferral SHALL be accompanied by a `TODO(Phase 3)` comment

#### Scenario: Register offset placeholder is tracked
- **WHEN** reading `src/ptxsim/instructions/tcgen05_cp.cpp`
- **THEN** the register offset fallback in `extract_smem_offset_placeholder` SHALL be accompanied by a `TODO(Phase 3)` comment

### Requirement: Documentation SHALL reflect tcgen05.cp test coverage status
The system SHALL update `src/ptxsim/instructions/AGENTS.md` and the root `AGENTS.md` to reflect that `tcgen05.cp` has unit, integration, and (if feasible) E2E test coverage.

#### Scenario: AGENTS.md lists tcgen05.cp coverage
- **WHEN** reading `src/ptxsim/instructions/AGENTS.md`
- **THEN** the `tcgen05.cp` entry SHALL indicate that unit and integration tests exist, and note the E2E status

