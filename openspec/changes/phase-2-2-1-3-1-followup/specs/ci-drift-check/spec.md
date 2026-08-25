# ci-drift-check spec (delta for phase-2-2-1-3-1-followup)

## MODIFIED Requirements

### Requirement: drift_check workflow verifies no empty-body IPtxEmuDevice method stubs in `device_api_impl.cc`

The drift_check workflow (`.github/workflows/drift_check.yml`) MUST extend its `paths` trigger filter to include `src/ptxemu/**` (alongside existing `include/ptxemu/**`), and add Invariant 6: after Phase 2.2/2.3 + Phase 2.2.1/2.3.1 implementation, no IPtxEmuDevice override method in `src/ptxemu/device_api_impl.cc` may contain an **empty body** that unconditionally returns a constant default value (`return false`, `return -1`, `return ThreadState::kIdle`, default-constructed `WarpStatus s{}; return s;`, or empty void no-op). Legitimate error-path guards (`if (!sm) return false;` followed by real delegation) MUST NOT trigger the invariant — only stub patterns (body containing a single constant return with no logic) MUST fail.

> **PHASE 2.2.1/2.3.1 CHANGE** (delta): The deferred-stubs exemption list MUST be reduced from 3 methods to 0 methods. After `phase-2-2-1-3-1-followup` lands, `warp_exe_once` / `get_thread_state` / `get_warp_status` are no longer stubs.

> **Invariant 6 (MODIFIED)**: The exemption list is now empty. All 12 `IPtxEmuDevice` override methods MUST have real delegation logic.

#### Scenario: Phase 2.2/2.3 + Phase 2.2.1/2.3.1 commits trigger drift_check on src/ptxemu changes

- **WHEN** a commit modifying `src/ptxemu/device_api_impl.cc` is pushed to any branch
- **AND** the file contains no empty-body stubs (per regex pattern below)
- **AND** none of the previously-exempted methods (`warp_exe_once` / `get_thread_state` / `get_warp_status`) are stubs anymore
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

#### Scenario: Deferred stub methods exemption list is EMPTY after this change

- **WHEN** this change lands
- **THEN** the deferred-stubs exemption list in Invariant 6 is EMPTY (0 methods)
- **AND** the previously-documented deferred stubs are now real implementations:
  - `warp_exe_once` → delegates to `WarpContext::execute_warp_instruction()`
  - `get_thread_state` → delegates to `ThreadContext::get_state()` + `map_state`
  - `get_warp_status` → populates the existing 5-field `WarpStatus` struct via `WarpState::threads[]`
- **AND** drift_check Invariant 6 implementation MUST remove the `warp_exe_once` / `get_thread_state` / `get_warp_status` entries from its whitelist
- **AND** this removal MUST be in the same commit as the Phase 2.2.1/2.3.1 implementation

> **Rationale** (per design.md Decision 3): The exemption list existed in `device-api-delegation` archive to allow 3 methods to remain stub bodies. After Phase 2.2.1/2.3.1 follow-up lands, all 12 methods MUST have real implementations. The exemption list removal is the structural invariant that prevents future contributors from silently re-stubbing these methods.