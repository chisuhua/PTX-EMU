## ADDED Requirements

> **NOTE**: This capability targets Phase 2 and Phase 3 of the `implement-atomic-cas-and-true-atomicity` change. It is documented here as part of the full change scope but the actual implementation is deferred to a follow-up change after Phase 1 is archived.

### Requirement: Per-Warp Serialize (Phase 2)

The system SHALL introduce per-warp serialization for atomic operations (including both existing 9 non-CAS atom ops and the new `atom.cas` from Phase 1) to ensure deterministic warp-internal execution.

#### Scenario: warp-internal serialization
- **WHEN** a single warp (32 lanes) executes any atomic operation on the same memory address
- **THEN** the 32 lane operations SHALL be serialized within the warp (deterministic order chosen by the warp scheduler) AND the final memory state SHALL be deterministic across repeated runs with the same input.

### Requirement: Cross-Warp Mutex (Phase 2)

The system SHALL introduce a cross-warp mutex protecting all atomic memory operations to ensure correctness under multi-warp contention.

#### Scenario: cross-warp mutual exclusion
- **WHEN** two or more warps concurrently execute atomic operations (e.g., `atom.add` and `atom.cas`) on the same memory address
- **THEN** the operations SHALL be serialized globally across warps AND the final memory state SHALL be deterministic (the order may vary across runs but the final value SHALL be predictable from the input).

#### Scenario: no deadlock with barrier mutex
- **WHEN** atomic operations and barrier operations are interleaved (e.g., `atom.cas` followed by `bar.sync`)
- **THEN** the system SHALL NOT deadlock between `AtomicMutex` and `CTAContext::barrier_module_->mutex_` AND the system SHALL document and enforce a global lock order (e.g., always barrier_mutex < atomic_mutex).

#### Scenario: recursive lock prevention
- **WHEN** reviewing the Phase 2 design
- **THEN** the design document SHALL include an explicit "hold lock and call other locked methods" analysis covering all atom handler code paths AND the design SHALL follow the `lessons-learned.md §2` pattern of providing internal `_unsafe` helpers for nested operations (no public method re-locks the same mutex).

### Requirement: Multi-Warp Oracle Test (Phase 3)

The system SHALL add an end-to-end oracle test validating multi-warp concurrent CAS correctness under the mutex.

#### Scenario: 2-warp CAS oracle
- **WHEN** `tests/integration/atomic/test_atom_global_cas_multiwarp.cpp` runs with 2 warps (64 lanes total) concurrently executing `atom.global.cas` on the same address
- **THEN** the test SHALL verify the final memory value is one of the valid 64 possible outcomes (based on which lane's CAS wins) AND the test SHALL be deterministic across runs (the SAME outcome every time on a fixed scheduler).

#### Scenario: stress test
- **WHEN** an e2e test runs with 100+ random multi-warp CAS patterns
- **THEN** the system SHALL produce deterministic outcomes AND SHALL NOT deadlock AND SHALL NOT produce non-deterministic garbage (no race-condition-induced corruption).
