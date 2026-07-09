## ADDED Requirements

### Requirement: TmemAllocator read-only methods thread-safety
SHALL hold `mu_` while accessing `allocations_` or `allocation_map_`.
All public read-only methods of `TmemAllocator` (`is_allocated_start`,
`is_allocated`, `active_allocation_count`, `total_allocated_slots`)
must do this. This eliminates the data race (undefined behavior)
present in Phase 1 where these methods accessed shared state without
synchronization.

#### Scenario: concurrent read + write no longer races
- **WHEN** thread A calls `is_allocated_start(s)` while thread B holds `mu_` and calls `allocate(1)` or `deallocate(s)`
- **THEN** thread A blocks until thread B releases `mu_`
- **AND** no undefined behavior (data race) occurs
- **AND** the result reflects a consistent snapshot of the allocator state

#### Scenario: kSlotCount consistency enforced at compile time
- **WHEN** the project is built
- **THEN** `static_assert(TmemAllocator::kSlotCount == Tmem::kSlotCount)` succeeds
- **AND** any future change that mismatches the two constants fails to compile

### Requirement: 3 handler-level integration tests
SHALL provide 3 integration tests that drive the
`processTcgen05Alloc`/`Dealloc`/`Relinquish` handlers through the
`Tcgen05Handler::processTcgen05Operation` dispatch path, using
`ptxsim::testing::step_warp` + `execute_warp_instruction`.

#### Scenario: integration tests PASS
- **WHEN** `cd build && ctest -R "integration_tcgen05_(alloc|dealloc|relinquish)_dispatch"` is run
- **THEN** all 3 tests PASS
- **AND** `cta->tmem_allocator()` state changes are observable post-dispatch

### Requirement: Multi-threaded deadlock detection functional
SHALL use `std::async` + `future.wait_for(30s)` to actively detect
deadlock. The unit test `multi_threaded_concurrent_alloc_dealloc_no_deadlock`
currently uses `th.join()` which blocks indefinitely on deadlock,
making the `REQUIRE(elapsed < 30)` assertion unreachable.

#### Scenario: deadlock triggers REQUIRE failure
- **WHEN** a deadlock is injected (e.g., recursive `mu_.lock()` in a public method)
- **THEN** `future.wait_for(30s)` returns `future_status::timeout`
- **AND** `REQUIRE(false, "deadlock suspected")` fails the test

### Requirement: Documentation synchronized per Oracle Q7-A
SHALL update the root `AGENTS.md` known-limitations table and
`src/ptxsim/instructions/AGENTS.md` TCGEN05 DISPATCH section to
reflect Phase 1 completion. The deferred count MUST be 3
(CP/MMA_WS/FENCE) instead of 6, and ALLOC/DEALLOC/RELINQUISH MUST
be listed as implemented.

#### Scenario: docs reflect current implementation state
- **WHEN** `git grep "11/11\|3 deferred\|6 deferred"` on `AGENTS.md` and `src/ptxsim/instructions/AGENTS.md`
- **THEN** the deferred count is 3 (CP/MMA_WS/FENCE), not 6
- **AND** ALLOC/DEALLOC/RELINQUISH are listed as implemented