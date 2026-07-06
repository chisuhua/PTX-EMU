## ADDED Requirements

### Requirement: atomic.cas Handler (Phase 1)

The system SHALL implement the `atomic.compare_and_swap` (a.k.a. `atomic.cas`) PTX instruction handler to perform compare-and-swap on global memory addresses with the following semantics:

1. The instruction reads the value at the target memory address (atomic load).
2. The handler compares the loaded value against the user-provided compare value (`cmp`).
3. If comparison succeeds (`loaded == cmp`), the handler writes the user-provided new value (`val`) to the same address (atomic conditional store).
4. The handler writes the originally loaded value (atomic load result) to the destination register `dst` regardless of compare outcome.

The system SHALL also implement the `atomic.exch` (atomic exchange) handler in the same atomic opcode extension, treating `atomic.exch` as a CAS variant where the compare always succeeds (i.e., unconditionally writes `val` and returns the old value).

#### Scenario: Single warp, all lanes cmp matches
- **WHEN** a single warp (32 lanes) executes `atom.global.cas.b32 %r0, [%addr], %cmp, %val;` where all 32 lanes target the same address and each lane's `cmp` equals the current memory value
- **THEN** the dst register `%r0` for each lane SHALL be set to the original loaded value AND the memory location SHALL be set to `val` (winner-takes-all semantics: the first lane to access updates the memory, others see the updated value but still write it back idempotently; on a single-warp serialized scheduler the result is deterministic).

#### Scenario: Single warp, no lane cmp matches
- **WHEN** a single warp (32 lanes) executes `atom.global.cas.b32 %r0, [%addr], %cmp, %val;` where no lane's `cmp` equals the current memory value
- **THEN** the dst register `%r0` for each lane SHALL be set to the original loaded value AND the memory location SHALL remain unchanged.

#### Scenario: Warp with mixed cmp outcomes
- **WHEN** a warp executes `atomic.cas` where the first 16 lanes' `cmp` matches the memory value and the second 16 lanes' `cmp` does not match
- **THEN** all 32 lanes' dst registers SHALL hold the original loaded value AND the memory SHALL be updated to `val` (since at least one lane's CAS succeeded).

#### Scenario: atomic.exch behavior
- **WHEN** a single lane executes `atom.global.exch.b32 %r0, [%addr], %val;`
- **THEN** the dst register `%r0` SHALL be set to the original loaded value AND the memory SHALL be unconditionally set to `val` (exchange semantics).

#### Scenario: Data size coverage
- **WHEN** the system processes `atomic.cas` with `.b8`, `.b16`, `.b32`, or `.b64` data size qualifiers
- **THEN** the handler SHALL correctly handle each data size (1/2/4/8 bytes respectively) and SHALL NOT silently truncate or extend values across data sizes.

#### Scenario: PTX parser passes Q_CAS_ATOM qualifier
- **WHEN** the ANTLR parser processes a `atomic.cas` PTX instruction text
- **THEN** the resulting qualifier list SHALL contain `Qualifier::Q_CAS_ATOM` (verified by `tests/ptx/atom_cas_basic.ptx` + `test_all_ptx.sh`) AND the handler SHALL be dispatched correctly (no silent no-op fallback to `Q_UNKNOWN` bail-out path).

#### Scenario: 4-operand collection through visitor
- **WHEN** the visitor (`ptx_visitor_atom.cpp:75-77`) collects operands for `atomic.cas`
- **THEN** the resulting `operands` vector SHALL contain exactly 4 elements: `[dst, addr, cmp, val]` (NOT 3 elements with cmp dropped, NOT 5 elements with extra noise) AND the handler SHALL receive these 4 operands via `processAtomicCAS(dst, addr, cmp, val, data_size, space)` signature.

### Requirement: Phase 1 Scope Boundaries (MR-6)

The Phase 1 implementation SHALL be limited to the following scope and SHALL NOT include any of the excluded items:

#### Scenario: Scope boundary enforcement
- **WHEN** a developer reviews the Phase 1 diff
- **THEN** the diff SHALL contain:
  - ✅ New `processAtomicCAS` function in `src/ptxsim/instructions/atomic.{h,cpp}`
  - ✅ New `case Qualifier::Q_CAS_ATOM` in `atomic.cpp` line ~36 qualifier detection
  - ✅ Removed "CAS is out-of-scope" comment block (lines 55-58)
  - ✅ New unit test `tests/unit/atomic/test_cas_handler_basic.cpp`
  - ✅ New integration test `tests/integration/atomic/test_atom_global_cas.cpp`
  - ✅ New PTX sample `tests/ptx/atom_cas_basic.ptx`
- **AND** SHALL NOT contain:
  - ❌ Changes to `ptx_qualifier.def` (Q_CAS_ATOM already exists, no need to modify)
  - ❌ Changes to `ptx_op.def` (opcount=3 already supports 4-operand via visitor loop)
  - ❌ Changes to any `.g4` ANTLR grammar file
  - ❌ Changes to `barrier_module.cpp` or any mutex/lock code (Phase 2)
  - ❌ Implementation of `.relaxed` / `.acq_rel` / `.scope` memory ordering qualifiers (parse-only, no semantic enforcement)
  - ❌ Implementation of `atom.shared.cas` / `atom.cta.cas` (Phase 1 limited to `.global` only)

### Requirement: Handler Quality Gates

The Phase 1 implementation SHALL meet the following quality gates defined in `docs/dev-process/lessons-learned.md`:

#### Scenario: No recursive locks
- **WHEN** `grep -rn "lock_guard\|unique_lock" src/ptxsim/instructions/atomic.cpp` runs
- **THEN** the output SHALL be empty (Phase 1 does not introduce mutexes; locks are a Phase 2 concern per the design).

#### Scenario: No qualifiers.back() usage
- **WHEN** `grep -n "qualifiers.back()" src/ptxsim/instructions/atomic.cpp` runs
- **THEN** the output SHALL remain at the existing line (the qualifier detection loop already iterates properly per `lessons-learned.md` §5; Phase 1 SHALL NOT introduce new `.back()` calls).

#### Scenario: Single atomic commit per Phase
- **WHEN** the Phase 1 changes are committed
- **THEN** the commit SHALL be a single atomic commit titled `refactor(atomic): implement CAS handler (Fix #1)` referencing this change and SHALL be independently revertable (`git revert HEAD` after commit should leave the codebase in a state equivalent to pre-change).

### Requirement: Test Verification

The Phase 1 implementation SHALL add the following tests and verify their behavior:

#### Scenario: Unit test passes single-lane CAS success
- **WHEN** `tests/unit/atomic/test_cas_handler_basic.cpp` runs with input `old=10, cmp=10, val=20` on a single thread
- **THEN** the test SHALL verify `dst == 10` (old value) AND `memory == 20` (new value, swap occurred).

#### Scenario: Unit test passes single-lane CAS failure
- **WHEN** the unit test runs with input `old=10, cmp=5, val=20`
- **THEN** the test SHALL verify `dst == 10` AND `memory == 10` (no swap occurred).

#### Scenario: Integration test verifies PTX-level behavior
- **WHEN** the integration test compiles and runs `atom.global.cas.b32 %r0, [%r1], %r2, %r3;` PTX code through the full warp execution pipeline
- **THEN** the test SHALL verify the same outcome semantics as the unit tests AND SHALL verify the PTX parser correctly produces a `StatementContext` with 4 operands.

#### Scenario: PTX syntax test passes
- **WHEN** `./tests/ptx/test_all_ptx.sh` runs (including the new `tests/ptx/atom_cas_basic.ptx`)
- **THEN** the script SHALL exit with code 0 and the parser SHALL successfully parse `atom.cas` without errors.

#### Scenario: No regression on existing atomic tests
- **WHEN** the Phase 1 implementation is applied
- **THEN** `ctest -L "integration;atomic" --output-on-failure` SHALL continue to pass `integration_ptx_atom_global_add` and `integration_ptx_atom_global_exch` (the two existing atom tests).
