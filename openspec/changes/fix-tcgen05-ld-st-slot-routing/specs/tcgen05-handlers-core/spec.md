## MODIFIED Requirements

### Requirement: 5 core tcgen05 handlers SHALL be implemented
The system SHALL provide `src/ptxsim/instructions/tcgen05.cpp` with 5
handler functions for: mma, ld, st, commit, wait. Each handler MUST
have a per-element `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.16`
annotation. The ld and st handlers MUST source the destination/source
TMEM slot from the `Tcgen05Instr.tmem_slot` field (per Oracle C2 fix)
rather than hard-coding slot 0.

#### Scenario: tcgen05.mma handler executes 32 lane × 8x4 fragment
- **WHEN** a Tcgen05Instr with `op_kind = MMA`, `cta_group = 1`, `dtype = F16` is dispatched
- **THEN** the handler reads from TMEM slots 0..63 and writes result to TMEM slots 64..95
- **AND** the output matches a golden value (per `tests/ptx/reference/tcgen05_mma_golden.h` from `wmma.cpp:374-420` inline mma + PTX ISA §9.7.16 manual calculation)

#### Scenario: tcgen05.ld handler copies 128 bytes from TMA desc to TMEM slot from instruction
- **WHEN** a Tcgen05Instr with `op_kind = LD`, `num_regs = 4`, and `tmem_slot = N` (where `0 ≤ N < kSlotCount = 256`) is dispatched
- **THEN** the handler reads 128 bytes from TmaDescriptor[0].global_address
- **AND** writes them byte-by-byte to TMEM slot `N` (NOT hardcoded slot 0, per Oracle C2 fix in `fix-tcgen05-ld-st-slot-routing`)
- **AND** if `N >= kSlotCount`, the handler throws `std::out_of_range` with message containing "tmem_slot"

#### Scenario: tcgen05.ld to default slot 0 preserves backward compatibility
- **WHEN** a Tcgen05Instr with `op_kind = LD` and `tmem_slot = 0` (default field value) is dispatched
- **THEN** the handler writes to TMEM slot 0 (identical behavior to pre-C2-fix hardcoded implementation)
- **AND** no regression in existing ld tests (`test_tcgen05_ld.cpp` etc.)

#### Scenario: tcgen05.st handler copies 128 bytes from TMEM slot from instruction to TMA desc
- **WHEN** a Tcgen05Instr with `op_kind = ST` and `tmem_slot = N` (where `0 ≤ N < kSlotCount`) is dispatched
- **THEN** the handler reads 128 bytes from TMEM slot `N` (NOT hardcoded slot 0, per Oracle C2 fix)
- **AND** writes them byte-by-byte to TmaDescriptor[0].global_address
- **AND** if `N >= kSlotCount`, the handler throws `std::out_of_range`

#### Scenario: tcgen05.st from default slot 0 preserves backward compatibility
- **WHEN** a Tcgen05Instr with `op_kind = ST` and `tmem_slot = 0` (default) is dispatched
- **THEN** the handler reads from TMEM slot 0 (identical to pre-C2-fix behavior)
- **AND** no regression in existing st tests

#### Scenario: tcgen05.commit handler commits and cluster arrive
- **WHEN** a Tcgen05Instr with `op_kind = COMMIT` is dispatched
- **THEN** the handler calls `cta->tc_queue().commit(instr.cta_group)` (per FU-1/C3 fix; pre-FU-1 was hardcoded `commit(1)`)
- **AND** if `cta->has_cluster_context()`, calls `cta->cluster_context().cta_cluster_arrive(cta->blockIdx.x)`

#### Scenario: tcgen05.wait handler waits and cluster wait
- **WHEN** a Tcgen05Instr with `op_kind = WAIT`, `lane_id = 0` (or parsed from operand), `group_id = instr.cta_group` is dispatched
- **THEN** the handler calls `cta->tc_queue().wait(warp, lane_id, instr.cta_group)` (per FU-1/C3 fix)
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

## ADDED Requirements

### Requirement: tcgen05.ld and tcgen05.st data flow SHALL route through instruction-specified slot (Oracle C2 fix)

The system SHALL provide a new integration test
`tests/integration/tcgen05/test_tcgen05_ld_st_slot_routing.cpp`
that validates the `ld → st` data flow through a non-zero TMEM slot
specified by `Tcgen05Instr.tmem_slot`. This test demonstrates that
FlashAttention's QK^T→softmax→PV data flow (which depends on ld/st
moving data into/from the same slots that mma consumes) is now
architecturally possible.

#### Scenario: ld to slot 32 + st from slot 32 round-trip preserves 128-byte pattern
- **WHEN** `processTcgen05Ld(slot=32)` is called with a known 128-byte golden pattern at the TMA source
- **AND** then `processTcgen05St(slot=32)` is called
- **THEN** the destination memory written by `st` equals the 128-byte golden pattern byte-by-byte
- **AND** the test SHALL run `ctest -R "tcgen05_ld_st_slot_routing" -V` and PASS

#### Scenario: ld to slot 0 still works (backward compat regression guard)
- **WHEN** `processTcgen05Ld(tmem_slot=0)` is called (default field value, pre-C2-fix equivalent behavior)
- **THEN** the handler writes to TMEM slot 0 identically to pre-C2-fix implementation
- **AND** existing ld parse tests (`test_tcgen05_ld_parse.cpp`) continue to PASS

#### Scenario: invalid tmem_slot throws out_of_range (not silent fallback)
- **WHEN** `processTcgen05Ld(tmem_slot=999)` is called (exceeds `kSlotCount=256`)
- **THEN** the handler throws `std::out_of_range` with message containing "tmem_slot"
- **AND** no silent fallback to slot 0 (which would mask the error and corrupt simulation state)
