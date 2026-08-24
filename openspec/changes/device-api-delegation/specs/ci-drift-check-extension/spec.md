# ci-drift-check-extension spec

## ADDED Requirements

### Requirement: drift_check workflow verifies no empty-body IPtxEmuDevice method stubs in `device_api_impl.cc`

The drift_check workflow (`.github/workflows/drift_check.yml`) MUST extend its `paths` trigger filter to include `src/ptxemu/**` (alongside existing `include/ptxemu/**`), and add Invariant 6: after Phase 2.2/2.3 implementation, no IPtxEmuDevice override method in `src/ptxemu/device_api_impl.cc` may contain an **empty body** that unconditionally returns a constant default value (`return false`, `return -1`, `return ThreadState::kIdle`, default-constructed `WarpStatus s{}; return s;`, or empty void no-op). Legitimate error-path guards (`if (!sm) return false;` followed by real delegation) MUST NOT trigger the invariant — only stub patterns (body containing a single constant return with no logic) MUST fail.

> **Invariant 6 (NEW)**: This is added as the 6th invariant in drift_check workflow, alongside the existing 5 invariants (PTXEMU_API_VERSION==1, IPtxEmuDevice ≥ 12 pure virtuals, C++17 compat, 4 symbols present, ptxemu_core STATIC target name).

#### Scenario: Phase 2.2/2.3 commit triggers drift_check on src/ptxemu changes

- **WHEN** a commit modifying `src/ptxemu/device_api_impl.cc` is pushed to any branch
- **AND** the file contains no empty-body stubs (per regex pattern below)
- **THEN** drift_check Invariant 6 PASSES
- **AND** the overall drift_check workflow exits 0

#### Scenario: Regression commit reintroducing empty-body stubs fails Invariant 6

- **WHEN** a future commit reintroduces empty-body stubs in `src/ptxemu/device_api_impl.cc`
- **THEN** drift_check Invariant 6 FAILS
- **AND** the CI pipeline blocks merge to main
- **AND** the regression is detected before reaching production (analogous to BUG-RETHANG prevention)

#### Scenario: Legitimate error-path returns do NOT trigger failure

- **WHEN** a delegation method contains error guards like `if (!sm) return false;` followed by real delegation logic
- **THEN** drift_check Invariant 6 PASSES (the `return false` is part of valid control flow, not a stub)
- **AND** only single-statement constant returns trigger failure

#### Scenario: Implementation pattern enforcement via drift_check

- **WHEN** contributors add new methods to `IPtxEmuDevice` (would require HSK-9)
- **AND** add corresponding empty-body stubs to `device_api_impl.cc`
- **THEN** drift_check Invariant 6 immediately flags the new stubs
- **AND** the contributor MUST implement the delegation before merging (no silent no-op stubs allowed)

#### Scenario: Invariant 6 regex (suggested)

- **MATCH**: `^\s*return\s+(false|nullptr|-1|ThreadState::kIdle|true);?\s*$` (single constant return, with optional semicolon)
- **EXCLUDE**: methods with >1 statement (delegation logic + error guards)
- **EXCLUDE**: `attach_timing` (void return type — stub pattern is no statements at all, but tracked by separate "empty void body" pattern)
- **IMPLEMENTATION**: bash + grep -E "^\s+return (false|nullptr|-1|ThreadState::kIdle);" -- context 5 (must be only return in method body); OR Python AST parse