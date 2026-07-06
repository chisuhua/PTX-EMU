## ADDED Requirements

### Requirement: 6 extended tcgen05 handlers SHALL be implemented
The system SHALL provide 4 new source files (`tcgen05_alloc.cpp`,
`tcgen05_cp.cpp`, `tcgen05_fence.cpp`, `tcgen05_mma_ws.cpp`) and add
6 handler functions to `tcgen05.cpp` for: ALLOC, DEALLOC,
RELINQUISH_ALLOC_PERMIT, CP, FENCE, MMA_WS.

#### Scenario: 6 handlers process correctly
- **WHEN** each handler is invoked with a Tcgen05Instr
- **THEN** the appropriate subsystem (Tmem/CTAContext/WarpScheduler) is updated per PTX ISA §9.7.16
- **AND** no regression in the 5 core handlers (mma/ld/st/commit/wait)

#### Scenario: per-CTA resource isolation for alloc/dealloc
- **WHEN** `tcgen05.alloc.cta_group::1.shared::cta.b32 [smem_addr], num_cols` is dispatched
- **THEN** the handler allocates `num_cols` TMEM slots in `cta->tmem()`
- **AND** other CTAs in same kernel are not affected

#### Scenario: weight-stationary mma.ws handler
- **WHEN** `tcgen05.mma.ws.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc` is dispatched
- **THEN** the handler executes weight-stationary variant (vs standard mma)
- **AND** the result matches a golden value (PTX ISA §9.7.16)

### Requirement: Tests cover 6 extended handlers
The system SHALL provide 1 unit test + 1 integration test + 2 E2E kernels
covering the 6 extended handlers with golden-value verification.

#### Scenario: tests PASS
- **WHEN** `cd build && ctest -L "unit;tcgen05|integration;tcgen05|e2e;tcgen05" -V` is run
- **THEN** 2 new unit tests + 1 integration test + 2 E2E kernels PASS
- **AND** no regression in core handler tests
