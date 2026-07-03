# Spec: barrier-module-unit-tests

> **Source change**: `openspec/changes/barrier-module-lifecycle-tests/`
> **Capability**: Direct unit-level test coverage for `BarrierModule::release_warp_barrier` + `WarpBarrier` lifecycle + `participation_mask` boundary conditions

## ADDED Requirements

### Requirement: BarrierModule::release_warp_barrier MUST update thread state fields atomically

The `BarrierModule::release_warp_barrier(warp_barrier_id, warp_ctx)` method MUST, for every released thread, atomically update the following state fields in `warp_state.threads[i]`:
- `is_blocked = false`
- `status = ptxsim::ThreadStatus::Active`
- `is_active = true`

It MUST also OR-merge the new `arrived_mask` with the existing `active_mask` (not overwrite).

#### Scenario: release_warp_barrier OR-merges active_mask
- **WHEN** `warp_ctx->get_active_mask()` is `0xFFFF0000` and `release_warp_barrier` is called with arrived_mask=`0x0000FFFF`
- **THEN** the new `active_mask` MUST be `0xFFFFFFFF` (OR-merge semantics)

#### Scenario: release_warp_barrier resets is_blocked for released threads
- **WHEN** `warp_state.threads[i].is_blocked = true` before release and `release_warp_barrier` releases thread `i`
- **THEN** `warp_state.threads[i].is_blocked` MUST be `false` after release

#### Scenario: release_warp_barrier sets status to Active for released threads
- **WHEN** `warp_state.threads[i].status = Blocked` before release and `release_warp_barrier` releases thread `i`
- **THEN** `warp_state.threads[i].status` MUST be `Active` after release

#### Scenario: release_warp_barrier sets is_active to true for released threads
- **WHEN** `warp_state.threads[i].is_active = false` before release and `release_warp_barrier` releases thread `i`
- **THEN** `warp_state.threads[i].is_active` MUST be `true` after release (required: `get_lanes_by_pc()` filters on `is_active`)

### Requirement: WarpBarrier::init MUST support lifecycle (init → complete → reset → re-init → complete)

The `WarpBarrier` object MUST support a complete lifecycle: `init(participation_mask, reconvergence_pc, barrier_pc)` → `arrive(lane_id)` (multiple) → `is_complete()` returns true → `reset()` → `re-init` → `arrive` → `is_complete()` returns true again. The lifecycle MUST NOT leak state across cycles.

#### Scenario: Full lifecycle resets state across cycles
- **WHEN** `WarpBarrier` completes a barrier cycle (init → arrive(0..31) → is_complete) and then `reset()` is called
- **THEN** the next `init` call MUST reset `arrived_mask_ = 0` and `arrived_count_ = 0`
- **AND** the subsequent `arrive(0..31)` MUST produce `is_complete() == true` in the new cycle

#### Scenario: re_init preserves arrived_mask (BUG-RECONVERGENCE-SIMPLEGEMM)
- **WHEN** `WarpBarrier` is initialized and lanes 0..15 have arrived (`arrived_mask = 0x0000FFFF`)
- **AND** `init` is called again on the already-initialized barrier (force_reconvergence re-entry)
- **THEN** `arrived_mask_` MUST remain `0x0000FFFF` (preserved, not reset)
- **AND** `arrived_count_` MUST remain `16`
- **AND** subsequent `arrive(16)` MUST set bit 16 in `arrived_mask` and increment count to 17

### Requirement: participation_mask boundary conditions MUST be respected

The `WarpBarrier::is_complete()` method MUST respect `participation_mask` exactly: a barrier is complete only when **all participants** in `participation_mask` have arrived, not when the full 32-lane set has arrived.

#### Scenario: Full-mask 32 with 31 arrives is incomplete
- **WHEN** `WarpBarrier` is `init(participation_mask=0xFFFFFFFF, ...)` and lanes 0..30 have arrived
- **THEN** `is_complete()` MUST return `false`

#### Scenario: Partial-mask 16 with 16 arrives completes at 16
- **WHEN** `WarpBarrier` is `init(participation_mask=0x0000FFFF, ...)` and lanes 0..15 have arrived
- **THEN** `is_complete()` MUST return `true`
- **AND** `arrived_count_` MUST equal `16` (not 32)

### Requirement: Unit tests MUST register with `[unit;barrier]` label and `unit_*` ctest prefix

All test files added by this change MUST register with Catch2 via `add_catch_test(<name> <file>)` where:
- `<name>` starts with `unit_` prefix (per commit `ab55e06` naming convention)
- `set_tests_properties(<name> PROPERTIES LABELS "unit;barrier")` is set

#### Scenario: Three new test targets registered
- **WHEN** this change is applied
- **THEN** the following ctest targets MUST exist:
  - `unit_barrier_module_release`
  - `unit_warp_barrier_lifecycle`
  - `unit_participation_mask_boundaries`
- **AND** all three MUST have `[unit;barrier]` label
- **AND** `ctest -R "barrier"` MUST continue to pass (with +8 new test cases alongside the existing 23)
