# nested-divergence-coverage Specification

## Purpose
TBD - created by archiving change add-nested-divergence-tests. Update Purpose after archive.
## Requirements
### Requirement: Two-level nested `@%p bra` divergence is correctly dispatched

The system SHALL correctly dispatch and reconverge a 32-lane warp through
two nested predicated `bra` instructions, exercising the SIMT stack
push/pop that one-level divergence tests cannot reach.

For a program of the form:

```ptx
@p1 bra L_then
mov.u32 r1, 1000
bra L_END
L_then:
mov.u32 r1, 100
@p2 bra L_INNER
mov.u32 r2, 300
bra L_END
L_INNER:
mov.u32 r2, 200
L_END:
add.u32 r3, r1, r2
ret
```

with `p1=true` for lanes 0..15 and `p2=true` for lanes 0..7 (within the
outer-then group), all 32 lanes SHALL reach `L_END` and the per-lane
register values SHALL match the path table below.

#### Scenario: Lanes 0..7 take both inner- and outer-then arms
- **WHEN** a 32-lane warp executes the program above with `p1=true` for lanes 0..15 and `p2=true` for lanes 0..7
- **THEN** lanes 0..7 SHALL end with `r1 == 100`, `r2 == 200`, `r3 == 300` (both branches taken)

#### Scenario: Lanes 8..15 take outer-then but inner-else arm
- **WHEN** the same program and predicate setup
- **THEN** lanes 8..15 SHALL end with `r1 == 100`, `r2 == 300`, `r3 == 400` (outer-then taken, inner-fall-through)

#### Scenario: Lanes 16..31 skip the outer-then block entirely
- **WHEN** the same program and predicate setup
- **THEN** lanes 16..31 SHALL end with `r1 == 1000` (outer fall-through path), `r2 == 0` (untouched register), `r3 == 1000` (1000 + 0 from the convergence-time add)

#### Scenario: All 32 lanes reach unified convergence
- **WHEN** the warp drives the program to completion via `step_warp`
- **THEN** the warp SHALL reach `ret` with all 32 lanes active at the unified convergence PC (no lane stuck mid-flight)
- **AND** the per-lane register values SHALL match the path table scenarios above (1 failed register = test failure)

### Requirement: SIMT stack discipline under two-level divergence

The SIMT stack SHALL grow to depth 2 during the inner branch
evaluation and SHALL pop back to depth 0 after both branches
reconverge at `L_END`.

#### Scenario: Peak depth during inner divergence
- **WHEN** `step_warp` reaches the inner `@p2 bra` PC
- **THEN** the SIMT stack depth SHALL be exactly 2 (one entry for the outer divergence at `@p1 bra` + one entry for the inner divergence at `@p2 bra`)

#### Scenario: Depth unwinds to 0 at unified convergence
- **WHEN** `step_warp` reaches the `ret` PC after `L_END`
- **THEN** the SIMT stack depth SHALL be 0 (both pushed entries have been popped by their respective reconverging `bra` instructions)

### Requirement: No regression on existing divergence coverage

The new TEST_CASE SHALL coexist with the existing
`test_nested_predication` block without affecting it.

#### Scenario: Existing setp+selp scenario still passes
- **WHEN** `ctest -R integration_nested_divergence_predication` runs
- **THEN** the existing `test_nested_predication` SHALL continue to PASS unchanged

#### Scenario: New test passes alongside the existing one
- **WHEN** `ctest -R integration_nested_divergence` runs (both labels)
- **THEN** both `test_nested_predication` and the new two-level-`bra` test SHALL PASS

