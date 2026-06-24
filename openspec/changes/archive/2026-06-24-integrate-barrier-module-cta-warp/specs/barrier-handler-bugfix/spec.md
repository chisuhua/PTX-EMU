# Capability: barrier-handler-bugfix

修复 `BarHandler::executeBarrier` 已知 bug：释放线程时未推进 `warp_state.threads[].pc`，导致线程被标记为可调度但实际停在原 PC。删除 `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184` 的测试 work-around。

## ADDED Requirements

### Requirement: BarHandler MUST advance PC via advance_thread_pc on release

When `BarHandler::executeBarrier` completes a barrier (all threads arrived), it MUST call `advance_thread_pc(lane, post_barrier_pc)` for each released thread, so that the per-thread PC (`warp_state.threads[lane].pc`) is updated to the post-barrier instruction.

#### Scenario: Released thread has updated per-thread PC
- **WHEN** `BarHandler::executeBarrier` returns `sync_complete == true`
- **THEN** for every thread in `arrived_threads_`, `warp_state.threads[lane].pc` MUST equal `post_barrier_pc`
- **AND** `warp_state.threads[lane].next_pc` MUST equal `post_barrier_pc`
- **AND** `warp_state.threads[lane].is_blocked` MUST be `false`
- **AND** `ThreadContext::state` MUST equal `RUN`

#### Scenario: release path uses advance_thread_pc not set_next_pc
- **WHEN** implementing the fix
- **THEN** the release code MUST use `cta_ctx->get_warp(idx)->advance_thread_pc(lane, post_barrier_pc)` 
- **AND** MUST NOT rely solely on `ThreadContext::set_next_pc()` (which only updates `next_pc`, not `pc`)

### Requirement: Integration test MUST NOT manually patch per-thread PC

`tests/integration/barrier/test_cta_barrier_memory_visibility.cpp` MUST NOT contain test driver code that manually calls `advance_thread_pc` to compensate for handler bugs.

#### Scenario: Work-around code removed
- **WHEN** the handler bug fix is complete
- **THEN** `test_cta_barrier_memory_visibility.cpp` MUST NOT contain `advance_thread_pc` calls in test driver logic (lines 138-184 specifically)
- **AND** the test MUST still pass after the work-around is removed (verified by `ctest -R integration_cta_barrier_memory_visibility`)

#### Scenario: Test verifies correct PC after barrier
- **WHEN** the test runs after the fix
- **THEN** it MUST assert `warp_state.threads[lane].pc == post_barrier_pc` for all released lanes (this assertion was previously impossible due to the bug)
- **AND** MUST pass without any manual PC patching in test driver

### Requirement: Same-day regression coverage for bug

A new unit test MUST be added that directly verifies `BarHandler::executeBarrier` updates per-thread PC, independent of the integration test.

#### Scenario: Unit test catches PC update bug
- **WHEN** a unit test directly invokes `BarHandler::executeBarrier` with 2 arrived threads
- **THEN** it MUST assert `warp_state.threads[lane].pc == expected_post_pc` for both lanes
- **AND** if the handler regresses to the old bug (only `set_next_pc`), the test MUST fail

### Requirement: BUG-RECONVERGENCE-SIMPLEGEMM fix MUST be preserved

The fix from commit `5820f7e` (preserve `arrived_mask` across force_reconvergence re-init) MUST continue to work after the `BarrierModule` integration. This means `BarrierModule::init_warp_barrier` MUST NOT reset `arrived_mask` if the barrier is already initialized.

#### Scenario: Re-init preserves arrived_mask
- **WHEN** `init_warp_barrier` is called for an already-initialized wbar (e.g., from a divergent half that already passed through)
- **THEN** it MUST update `participation_mask`, `reconvergence_pc`, `expected_count`, `is_initialized`
- **AND** MUST NOT reset `arrived_mask` or `arrived_count`
- **AND** subsequent `arrive_at_warp_barrier` calls MUST accumulate onto the preserved `arrived_mask`
