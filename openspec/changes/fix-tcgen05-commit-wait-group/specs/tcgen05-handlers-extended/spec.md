## MODIFIED Requirements

### Requirement: 6 extended tcgen05 handlers SHALL be implemented

The system SHALL provide 4 new source files (`tcgen05_alloc.cpp`,
`tcgen05_cp.cpp`, `tcgen05_fence.cpp`, `tcgen05_mma_ws.cpp`) and add
6 handler functions to `tcgen05.cpp` for: ALLOC, DEALLOC,
RELINQUISH_ALLOC_PERMIT, CP, FENCE, MMA_WS.

Additionally, the `processTcgen05Commit` and `processTcgen05Wait`
handler functions SHALL read `instr.cta_group` (populated by
`visitTcgen05Inst` from the `.cta_group::N` qualifier IMMEDIATE
value, per `specs/tcgen05-multi-group-commit-wait/spec.md`) instead
of hardcoding `group_id=1`. The hardcoded `lane_id=0` in
`processTcgen05Wait` SHALL remain pending a separate future change.

#### Scenario: 6 handlers process correctly

- **WHEN** each handler is invoked with a Tcgen05Instr
- **THEN** the appropriate subsystem (TmemAllocator/CTAContext/Warp) is updated per PTX ISA §9.7.16
- **AND** no regression in the 5 core handlers (mma/ld/st/commit/wait)
- **AND** `processTcgen05Commit` invokes `cta->tc_queue().commit(instr.cta_group)` (per `tcgen05.cpp:512` post-fix)
- **AND** `processTcgen05Wait` invokes `cta->tc_queue().wait(warp, 0, instr.cta_group)` (per `tcgen05.cpp:550` post-fix)

#### Scenario: per-CTA resource isolation for alloc/dealloc

- **WHEN** `tcgen05.alloc.cta_group::1.shared::cta.b32 [smem_addr], num_cols` is dispatched
- **THEN** the handler allocates `num_cols` TMEM slots via `TmemAllocator` (new abstraction layer, per Oracle Q1-A)
- **AND** other CTAs in same kernel are not affected

#### Scenario: cta_group::2 throws clear exception (per Oracle Q2-A)

- **WHEN** `tcgen05.*.cta_group::2.*` is dispatched
- **THEN** the handler throws `UnsupportedInstructionException` with message containing "cluster abstraction not yet implemented (ADR-0018)"
- **AND** no silent fallback to cta_group::1 behavior
- **NOTE**: `cta_group::2` parsing path is enabled by `Tcgen05Instr::cta_group` field being populated; the throw occurs at handler dispatch, not at parse time.

#### Scenario: weight-stationary mma.ws handler (per Oracle Q3-A scope)

- **WHEN** `tcgen05.mma.ws.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc` is parsed by the grammar
- **THEN** the parser produces `Tcgen05Instr{op_kind=MMA, qualifiers={Q_TCGEN_WS, Q_F16, Q_TCGEN_CTA_GROUP}, cta_group=1}` (grammar treats `.ws` as a qualifier on the MMA sub-op, not as a separate MMA_WS sub-op — see Oracle 2026-07-08 review)
- **AND** `processTcgen05Mma` scans `instr.qualifiers` for `Q_TCGEN_WS` and routes to the ws path
- **AND** the ws path calls the shared `tcgen05_fragment_mma_f16` helper (same fragment arithmetic as regular mma; ws-specific weight-stationary layout transform is deferred per Oracle A-path scope discipline)
- **AND** the result matches the same golden value as regular mma (PTX ISA §9.7.16, marked `UNVERIFIED-AGAINST-HARDWARE`)
- **AND** non-f16 kind types on the ws path throw `UnsupportedInstructionException` referencing Oracle Q3-A scope discipline
- **AND** `case Tcgen05OpKind::MMA_WS` in `Tcgen05Handler::processTcgen05Operation` is retained (for direct `Tcgen05Instr` construction in tests) but routes to `processTcgen05Mma` identically to `case MMA`

#### Scenario: cp handler reuses smem address resolution (per Oracle Q4-B)

- **WHEN** `tcgen05.cp.cta_group::1.shared::cta [tmem_dst], [smem_src], shape` is dispatched
- **THEN** the handler reads from shared memory via `SharedMemoryManager` (existing path, no new `SmemDescriptor` abstraction)
- **AND** writes to TMEM via `TmemAllocator`
- **AND** 128 bytes transferred byte-by-byte correctly

#### Scenario: fence handler is no-op marker (per Oracle Q6-B)

- **WHEN** `tcgen05.fence.before_thread_sync` or `::after_thread_sync` is dispatched
- **THEN** the handler calls `warp->record_fence_position(before/after)` (extension point)
- **AND** does NOT trigger membar / memory barrier
- **AND** does NOT block on warp arrival

#### Scenario: commit handler reads instr.cta_group (NEW per `fix-tcgen05-commit-wait-group`)

- **WHEN** `processTcgen05Commit` is called with a `Tcgen05Instr{cta_group=N}` where N ∈ {1, 2, ...}
- **THEN** `cta->tc_queue().commit(N)` is invoked (per `tcgen05.cpp:512` post-fix)
- **AND** the `(void)instr;` cast has been removed (per D4 of `design.md`)
- **AND** the TcQueue counter for group N advances (verified via `cta->tc_queue().commit_count(N)` — pending FU-3 follow-up to expose API)

#### Scenario: wait handler reads instr.cta_group (NEW per `fix-tcgen05-commit-wait-group`)

- **WHEN** `processTcgen05Wait` is called with a `Tcgen05Instr{cta_group=N}` where N ∈ {1, 2, ...}
- **THEN** `cta->tc_queue().wait(warp, /*lane_id=*/0, N)` is invoked (per `tcgen05.cpp:550` post-fix)
- **AND** the warp blocks on group N only
- **AND** the `(void)instr;` cast has been removed (per D4 of `design.md`)
- **NOTE**: `lane_id=0` hardcoding remains pending FU-3.5 follow-up; not part of this change.

### Requirement: TmemAllocator abstraction layer SHALL be added (per Oracle Q1-A)

The system SHALL provide a new `TmemAllocator` class
(`include/ptxsim/memory/tmem_allocator.h`) as an abstraction layer
over the existing fixed `Tmem` (256 slots × 128 bytes).

#### Scenario: TmemAllocator API

- **WHEN** the system is initialized
- **THEN** `CTAContext::tmem_allocator()` returns a `TmemAllocator&`
- **AND** public methods include: `allocate(num_cols) -> slot_id`, `deallocate(slot_id)`, `query(slot_id) -> bytes`
- **AND** internal state uses `std::bitset<256>` to track allocation

#### Scenario: Recursive lock audit (per ptx-lessons-learned §2, Oracle high-risk)

- **WHEN** `TmemAllocator` is implemented
- **THEN** all public methods are audited: no public method that holds `mu_` calls another public method that also holds `mu_`
- **AND** a multi-threaded concurrent `alloc`/`dealloc` unit test passes (falsification: deadlock detection)

### Requirement: Tests cover 6 extended handlers (per Oracle Q5-C mixed strategy)

The system SHALL provide 1 unit test + 1 integration test + 2 E2E kernels
covering the 6 extended handlers with **mixed oracle strategy**:
- **Unit**: hand-computed golden values, marked `UNVERIFIED-AGAINST-HARDWARE`
- **Integration**: `step_warp` + `execute_warp_instruction` driven
- **E2E**: real nvcc-generated PTX when available, fixtures otherwise

#### Scenario: tests PASS

- **WHEN** `cd build && ctest -L "unit;tcgen05|integration;tcgen05|e2e;tcgen05" -V` is run
- **THEN** 1 unit test + 1 integration test + 2 E2E kernels PASS
- **AND** no regression in core handler tests
- **AND** all new golden values include `// UNVERIFIED-AGAINST-HARDWARE` comment
- **AND** NEW per `fix-tcgen05-commit-wait-group`: `tests/integration/tcgen05/test_tcgen05_commit_wait_group.cpp` covers `mma→commit(cta_group=2)→wait(cta_group=2)→mma` sequence; PASS

#### Scenario: cta_group::2 parse test exists (NEW per `fix-tcgen05-commit-wait-group`)

- **WHEN** `tests/integration/ptx/test_tcgen05_mma_parse.cpp` is extended with a new TC
- **THEN** the TC parses `tcgen05.mma.kind::f16.cta_group::2 [addr], a, b, i;`
- **AND** asserts `instr.cta_group == 2u`
- **AND** asserts `instr.qualifiers` contains `Q_TCGEN_CTA_GROUP`

### Requirement: Documentation SHALL be updated (per Oracle Q7-A)

The system SHALL update documentation in the same change (each Phase end):
- Root `AGENTS.md` known limitations table: tcgen05 → 11/11 handler implemented
- `src/ptxsim/instructions/AGENTS.md`: `tcgen05.cpp` includes 11 handler
- `docs/ptx/README.md` status table updated
- `docs/adr/0016-blackwell-only-tcgen05.md` appends update record

#### Scenario: documentation reflects 11/11 tcgen05 handlers

- **WHEN** the change is complete (all 6 Phases done)
- **THEN** `git grep "11/11"` on `AGENTS.md` and `docs/ptx/README.md` returns the updated status
- **AND** `docs/adr/0016-blackwell-only-tcgen05.md` includes a section noting the change archive commit
- **AND** no remaining references to "deferred" or "UnsupportedInstructionException" for these 6 handlers
- **AND** NEW per `fix-tcgen05-commit-wait-group`: `docs/adr/0016-blackwell-only-tcgen05.md` includes a "2026-07-12 Postmortem: C3 fix" section
