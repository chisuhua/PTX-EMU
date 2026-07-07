## ADDED Requirements

### Requirement: 5 core tcgen05 handlers SHALL be implemented
The system SHALL provide `src/ptxsim/instructions/tcgen05.cpp` with 5
handler functions for: mma, ld, st, commit, wait. Each handler MUST
have a per-element `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16`
annotation.

#### Scenario: tcgen05.mma handler executes 32 lane × 8x4 fragment
- **WHEN** a Tcgen05Instr with `op_kind = MMA`, `cta_group = 1`, `dtype = F16` is dispatched
- **THEN** the handler reads from TMEM slots 0..63 and writes result to TMEM slots 64..95
- **AND** the output matches a golden value (per `tests/ptx/reference/tcgen05_mma_golden.h` from `wmma.cpp:374-420` inline mma + PTX ISA §9.7.16 manual calculation)

#### Scenario: tcgen05.ld handler copies 128 bytes from TMA desc to TMEM slot 0
- **WHEN** a Tcgen05Instr with `op_kind = LD`, `num_regs = 4` is dispatched
- **THEN** the handler reads 128 bytes from TmaDescriptor[0].global_address
- **AND** writes them byte-by-byte to TMEM slot 0

#### Scenario: tcgen05.st handler copies 128 bytes from TMEM slot 0 to TMA desc
- **WHEN** a Tcgen05Instr with `op_kind = ST` is dispatched
- **THEN** the handler reads 128 bytes from TMEM slot 0
- **AND** writes them byte-by-byte to TmaDescriptor[0].global_address

#### Scenario: tcgen05.commit handler commits and cluster arrive
- **WHEN** a Tcgen05Instr with `op_kind = COMMIT` is dispatched
- **THEN** the handler calls `cta->tc_queue().commit(1)`
- **AND** if `cta->has_cluster_context()`, calls `cta->cluster_context().cta_cluster_arrive(cta->blockIdx.x)`

#### Scenario: tcgen05.wait handler waits and cluster wait
- **WHEN** a Tcgen05Instr with `op_kind = WAIT`, `lane_id = 0`, `group_id = 1` is dispatched
- **THEN** the handler calls `cta->tc_queue().wait(warp, 0, 1)`
- **AND** if `cta->has_cluster_context()`, calls `cta->cluster_context().cta_cluster_wait(cta->blockIdx.x)`

### Requirement: 5 unit + 5 integration + 1 E2E tests (11 total) SHALL cover core handlers
The system SHALL provide 5 unit tests (`tests/unit/ptx/`), 5
integration tests (`tests/integration/tcgen05/`), and 1 E2E kernel
(`tests/e2e/kernel/test_tcgen05_mma_gemm.cu`) covering the 5 core
handlers with golden-value verification.

#### Scenario: 5 unit tests PASS
- **WHEN** `cd build && ctest -L "unit;tcgen05" -V` is run
- **THEN** 5 new test targets PASS (qualifier, opkind, dtype, stmt_factory, instr_struct)

#### Scenario: 5 integration tests PASS
- **WHEN** `cd build && ctest -L "integration;tcgen05" -V` is run
- **THEN** 5 new test targets PASS (mma/ld/st/commit/wait parse → IR → handler)

#### Scenario: 1 E2E test PASS
- **WHEN** `cd build && ctest -L "e2e;tcgen05" -V` is run
- **THEN** the E2E kernel test PASSES (cuobjdump-extracted tcgen05.mma GEMM)
