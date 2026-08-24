# ci-drift-check-extension spec

## ADDED Requirements

### Requirement: drift_check workflow verifies zero `return false` stubs in `device_api_impl.cc` after Phase 2.2/2.3

The drift_check workflow (`.github/workflows/drift_check.yml`) MUST verify that after Phase 2.2/2.3 implementation, `src/ptxemu/device_api_impl.cc` contains zero `return false` statements, except within the legitimate `attach_timing` method (which has void return type and thus no `return false` is possible — the exception is for future-proofing against accidental reintroduction of stubs in `attach_timing`'s body).

> **Invariant 6 (NEW)**: This is added as the 6th invariant in drift_check workflow, alongside the existing 5 invariants (PTXEMU_API_VERSION==1, IPtxEmuDevice ≥ 12 pure virtuals, C++17 compat, 4 symbols present, ptxemu_core STATIC target name).

#### Scenario: Phase 2.2/2.3 commit triggers drift_check with zero stubs

- **WHEN** a commit modifying `src/ptxemu/device_api_impl.cc` is pushed to any branch
- **AND** the file contains zero `return false` lines in the implementation
- **THEN** drift_check Invariant 6 PASSES
- **AND** the overall drift_check workflow exits 0

#### Scenario: Regression commit reintroducing stubs triggers drift_check failure

- **WHEN** a future commit reintroduces `return false` stubs in `src/ptxemu/device_api_impl.cc`
- **THEN** drift_check Invariant 6 FAILS
- **AND** the CI pipeline blocks merge to main
- **AND** the regression is detected before reaching production (analogous to BUG-RETHANG prevention)

#### Scenario: Implementation pattern is enforced by CI

- **WHEN** contributors add new methods to `IPtxEmuDevice` (would require HSK-9)
- **AND** add corresponding stubs to `device_api_impl.cc`
- **THEN** drift_check Invariant 6 immediately flags the new `return false` stubs
- **AND** the contributor MUST implement the delegation before merging (no silent no-op stubs allowed)