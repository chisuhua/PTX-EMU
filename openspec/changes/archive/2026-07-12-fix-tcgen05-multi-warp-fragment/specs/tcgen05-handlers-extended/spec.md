## MODIFIED Requirements

### Requirement: 6 extended tcgen05 handlers SHALL be implemented

The system SHALL provide 4 new source files (`tcgen05_alloc.cpp`,
`tcgen05_cp.cpp`, `tcgen05_fence.cpp`, `tcgen05_mma_ws.cpp`) and add
6 handler functions to `tcgen05.cpp` for: ALLOC, DEALLOC,
RELINQUISH_ALLOC_PERMIT, CP, FENCE, MMA_WS.

Additionally, the `tcgen05_fragment_mma_f16` helper SHALL accept an
`int warp_id` parameter (between `Tmem& tmem` and `bool accumulate = false`)
to enable multi-warp C slot isolation per Oracle C4 fix. The C slot
computation SHALL be `c_slot = warp_id * 32 + 64 + lane_id`. The
`processTcgen05Mma` handler SHALL invoke the helper with
`warp->get_warp_id()` as the warp_id argument (per
[`specs/tcgen05-multi-warp-fragment/spec.md`](tcgen05-multi-warp-fragment/spec.md)).

#### Scenario: 6 handlers process correctly

- **WHEN** each handler is invoked with a Tcgen05Instr
- **THEN** the appropriate subsystem (TmemAllocator/CTAContext/Warp) is updated per PTX ISA §9.7.16
- **AND** no regression in the 5 core handlers (mma/ld/st/commit/wait)
- **AND** `processTcgen05Mma` invokes `tcgen05_fragment_mma_f16(tmem, warp->get_warp_id(), /*accumulate=*/false)` per `tcgen05.cpp:383` post-C4-fix
- **AND** the helper computes `c_slot = warp_id * 32 + 64 + lane_id` per `tcgen05_helpers.cpp:23` post-C4-fix

#### Scenario: per-CTA resource isolation for alloc/dealloc

- **WHEN** `tcgen05.alloc.cta_group::1.shared::cta.b32 [smem_addr], num_cols` is dispatched
- **THEN** the handler allocates `num_cols` TMEM slots via `TmemAllocator` (new abstraction layer, per Oracle Q1-A)
- **AND** other CTAs in same kernel are not affected

#### Scenario: cta_group::2 throws clear exception (per Oracle Q2-A)

- **WHEN** `tcgen05.*.cta_group::2.*` is dispatched
- **THEN** the handler throws `UnsupportedInstructionException` with message containing "cluster abstraction not yet implemented (ADR-0018)"
- **AND** no silent fallback to cta_group::1 behavior

#### Scenario: weight-stationary mma.ws handler (per Oracle Q3-A scope)

- **WHEN** `tcgen05.mma.ws.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc` is parsed by the grammar
- **THEN** the parser produces `Tcgen05Instr{op_kind=MMA, qualifiers={Q_TCGEN_WS, Q_F16, Q_TCGEN_CTA_GROUP}}` (grammar treats `.ws` as a qualifier on the MMA sub-op, not as a separate MMA_WS sub-op — see Oracle 2026-07-08 review)
- **AND** `processTcgen05Mma` scans `instr.qualifiers` for `Q_TCGEN_WS` and routes to the ws path
- **AND** the ws path calls the shared `tcgen05_fragment_mma_f16` helper with multi-warp slot offset (per Oracle C4 fix, `tcgen05_helpers.cpp:23` post-fix)
- **AND** the ws path passes `warp->get_warp_id()` to the helper (per `tcgen05.cpp:383` post-fix)
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

#### Scenario: multi-warp mma fragment isolation (NEW — Oracle C4 scenario)

- **WHEN** `processTcgen05Mma` is invoked from a multi-warp CTA (e.g., 2 warps from `SMContext(2, 128, ...)`)
- **THEN** warp 0's mma writes C slots `[64..95]`
- **AND** warp 1's mma writes C slots `[96..127]`
- **AND** reading C slot 64 returns warp 0's C value (not warp 1's)
- **AND** reading C slot 96 returns warp 1's C value (not warp 0's)
- **NOTE**: A/B slots `[0..63]` are shared input (per design.md D2); C slot partitioning enables FlashAttention FA3 producer-consumer multi-warp mma

#### Scenario: invalid warp_id propagation (NEW — Oracle Risk R6)

- **WHEN** `tcgen05_fragment_mma_f16(tmem, /*warp_id=*/-1, /*accumulate=*/false)` is invoked
- **THEN** the helper throws `std::invalid_argument` BEFORE any TMEM read/write
- **AND** no partial state mutation occurs (Tmem unchanged from caller perspective)
- **AND** the exception message contains "warp_id must be >= 0 (got -1)"
