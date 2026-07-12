# tcgen05-flashattention-coverage Specification

## Purpose
TBD - created by archiving change tcgen05-flashattention-coverage. Update Purpose after archive.

## Requirements

### Requirement: K=128 mma accumulator SHALL pass within numerical tolerance (FA-B1)

The system SHALL provide an integration test
(`tests/integration/tcgen05/test_tcgen05_mma_k_loop_128.cpp`) that
verifies `tcgen05.mma` with `accumulate=true` performs K=128
iterations of `C += A*B` correctly, with the final C slot value
equal to `128 × GOLDEN_MMA_F16_F16_F32` within relative error < 1e-3.

#### Scenario: K=128 sequential accumulation produces correct sum
- **WHEN** `processTcgen05Mma(instr_with_accumulate)` is invoked 128 times with identical A, B inputs (no intermediate cp/ld/st)
- **THEN** the C slot SHALL contain `128 × GOLDEN_MMA_F16_F16_F32` per-lane fragments
- **AND** relative error vs expected SHALL be < 1e-3 (per Oracle Section E: K=128 累加后随机游走 ≈ 6.74e-7, 1e-3 是 1500× 安全裕度)
- **AND** the test SHALL be tagged `[integration][tcgen05][mma][flashattention][k-loop]`

#### Scenario: K=128 with random A, B inputs validates per-iteration independence
- **WHEN** the test uses a loop counter `i` to perturb A, B per iteration (golden input varies)
- **THEN** the test SHALL validate `C[i] = sum_{k=0..i} A[k] * B[k]` after i+1 iterations
- **AND** SHALL abort early if any per-iteration drift exceeds 1e-3 relative error (per Oracle D2: K=128 上限)

### Requirement: mma → commit → wait → mma sequence SHALL preserve C slot across barrier sync (FA-B2)

The system SHALL provide an integration test
(`tests/integration/tcgen05/test_tcgen05_mma_commit_wait_sequence.cpp`)
that verifies the `tcgen05.commit` / `tcgen05.wait` barrier
correctly synchronizes the C slot state across consecutive mma
operations with `accumulate=true`.

#### Scenario: mma → commit → wait → mma accumulates 2× golden
- **WHEN** `processTcgen05Mma(accumulate=true)` is invoked, followed by `processTcgen05Commit(group_id=1)`, `processTcgen05Wait(group_id=1)`, then `processTcgen05Mma(accumulate=true)` again with identical inputs
- **THEN** the final C slot SHALL equal `2 × GOLDEN_MMA_F16_F16_F32`
- **AND** `cta->tc_queue().pending_count()` SHALL be 0 after wait (commit/wait plumbing verified)

#### Scenario: mma → commit → wait → ld → st → mma preserves data across ld/st stages
- **WHEN** the sequence is `mma(1st, accumulate=true) → commit(g=1) → wait(g=1) → ld(slot_X) → st(slot_X) → mma(2nd, accumulate=true)`
- **THEN** the C slot after 2nd mma SHALL reflect 2× golden values
- **AND** the `ld → st` data movement SHALL preserve the 1st mma's C output (verifying FU-3 C2 fix: ld/st slot routing)

### Requirement: cp → mma data flow SHALL produce numerically correct results (FA-B4)

The system SHALL provide an integration test
(`tests/integration/tcgen05/test_tcgen05_mma_cp_data_flow.cpp`)
that verifies `tcgen05.cp` loads A/B data into the TMEM slots that
subsequent `tcgen05.mma` reads from, producing a numerically
correct C output (not merely "at least one element changed" as in
the existing `test_tcgen05_mma_persistence.cpp:250-294`).

#### Scenario: cp writes to slot that mma reads from
- **WHEN** `tcgen05.cp` writes a known A matrix to TMEM slot X (per FU-3 C2 tmem_slot), and `tcgen05.mma` reads A from slot X (same lane_id mapping as `tcgen05_helpers.cpp:21-23`)
- **THEN** the mma output SHALL equal `A * B` golden value (not zero, not garbage, not "at least one element changed")
- **AND** the test SHALL verify C slot at `64 + lane_id` matches `GOLDEN_MMA_F16_F16_F32[idx]` for all 32 lanes

#### Scenario: cp with multiple slots validates per-slot data integrity
- **WHEN** `tcgen05.cp` writes different data to slots 0, 2, 4, ..., 62 (one per even lane_id per A layout)
- **THEN** subsequent `tcgen05.mma` SHALL produce 32 distinct lane outputs corresponding to the per-slot cp inputs
- **AND** no two lanes SHALL produce identical outputs (cp data integrity verified)

### Requirement: 2-warp mma SHALL isolate C slots per warp (FA-B5)

The system SHALL provide an integration test
(`tests/integration/tcgen05/test_tcgen05_multi_warp_isolation.cpp`)
that verifies `tcgen05.mma` invocations from different warps write
to disjoint TMEM slot ranges without collisions.

#### Scenario: warp 0 and warp 1 write disjoint C slot ranges
- **WHEN** `SMContext(2, 128, 4096, 0)` is configured with 2 warps
- **AND** warp 0 invokes `processTcgen05Mma` followed by warp 1 invoking `processTcgen05Mma` with identical A, B inputs
- **THEN** warp 0's C slot SHALL be at TMEM `c_slot = 0 * 32 + 64 + lane_id = [64..95]` (per FU-4 C4 formula)
- **AND** warp 1's C slot SHALL be at TMEM `c_slot = 1 * 32 + 64 + lane_id = [96..127]`
- **AND** the two C slot ranges SHALL be disjoint (no race condition, no overwrite)

#### Scenario: simultaneous 2-warp mma produces independent outputs
- **WHEN** both warp 0 and warp 1 invoke `processTcgen05Mma` concurrently (scheduler interleaving allowed)
- **THEN** warp 0's C slot SHALL contain `GOLDEN_MMA_F16_F16_F32` (per warp 0 perspective)
- **AND** warp 1's C slot SHALL contain `GOLDEN_MMA_F16_F16_F32` (per warp 1 perspective)
- **AND** both warps SHALL observe the same numerical output regardless of execution order (FA-typical pattern)

### Requirement: FlashAttention mini-kernel E2E test SHALL validate end-to-end data flow (FA-E2E)

The system SHALL provide an E2E test
(`tests/e2e/kernel/test_flashattention_mini.cu`) that compiles a
mini FlashAttention kernel (K=4 blocks, head_dim=64, block_size=32)
and validates the QK^T → softmax → PV data flow end-to-end via the
fake `libcudart.so` interception path.

#### Scenario: FA mini-kernel produces correct O output within tolerance
- **WHEN** the mini-kernel executes `Q @ K^T → softmax → @V` via tcgen05.mma + commit + wait + ld + st sequences
- **THEN** the output O SHALL match the reference CUDA fallback output within relative error < 1e-3
- **AND** the test SHALL be tagged `[e2e][flashattention]` with `priority-3` annotation if ptxas fallback path is used

#### Scenario: FA mini-kernel handles K-loop with commit/wait sync
- **WHEN** the kernel performs K=4 iterations of `mma(accumulate=true) → commit(g=QK) → wait(g=QK)`
- **THEN** the K-loop SHALL complete without deadlocks (per FU-1 C3 multi-group commit)
- **AND** each iteration's C accumulator SHALL be visible to the next iteration's mma

### Requirement: tmem_helpers.h SHALL provide reusable tmem test utilities

The system SHALL provide a helper header
(`include/ptxsim/testing/tmem_helpers.h`) in namespace
`ptxsim::testing` that consolidates the existing
`fill_tmem_with_golden_inputs` + `require_c_slot_matches` functions
(per `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp`)
and adds `compare_c_slot_to_reference(expected, tolerance)` for
lenient comparison vs strict equality.

#### Scenario: helper functions are accessible to all 5 new test files
- **WHEN** the 5 new test files (`test_tcgen05_mma_k_loop_128.cpp`, `test_tcgen05_mma_commit_wait_sequence.cpp`, `test_tcgen05_mma_cp_data_flow.cpp`, `test_tcgen05_multi_warp_isolation.cpp`, `test_flashattention_mini.cu`) include `ptxsim/testing/tmem_helpers.h`
- **THEN** each test SHALL be able to call `fill_tmem_with_golden_inputs(tmem)`, `require_c_slot_matches(tmem, golden)`, and `compare_c_slot_to_reference(tmem, expected, 1e-3)`
- **AND** the helper SHALL be ctest-tag-independent (usable from unit/integration/e2e contexts)

#### Scenario: compare_c_slot_to_reference supports multiple tolerance modes
- **WHEN** a test calls `compare_c_slot_to_reference(tmem, expected, Catch::Approx::custom().epsilon(1e-3).margin(1e-5))`
- **THEN** the helper SHALL iterate 32 lanes × 32 elements (1024 floats total) and apply the tolerance per element
- **AND** SHALL return detailed diagnostic info (`INFO()` with lane/idx/expected/actual) on first mismatch

### Requirement: Numerical tolerance SHALL be tightened to 1e-6 for f32 mma readback (FA-B7)

The system SHALL use `Catch::Approx::custom().epsilon(1e-6)` for
all `tcgen05.mma` C slot readback comparisons in the 5 new test
files (instead of the default `1.19e-5` epsilon per
`tests/catch_amalgamated.cpp:567`).

#### Scenario: 1e-6 epsilon detects ULP-level drift in K=128 accumulation
- **WHEN** the FA-B1 K=128 accumulator test asserts C slot values against `128 × GOLDEN`
- **THEN** the test SHALL use `epsilon(1e-6)` tolerance
- **AND** SHALL fail if any per-lane drift exceeds 1e-6 (detects ULP-level f32 accumulation error)

#### Scenario: 1.0..32.0 golden values remain within tolerance after epsilon tightening
- **WHEN** `GOLDEN_MMA_F16_F16_F32` contains values in range [1.0, 32.0] (all exactly representable in f32)
- **THEN** the tightening from 1.19e-5 to 1e-6 SHALL NOT cause any false negatives for single-mma reads
- **AND** the relative error margin SHALL be adequate for K=128 accumulation drift (~6.74e-7)

## Cross-Reference

- Oracle 2026-07-11 audit: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC` (7 BLOCKER/IMPORTANT gaps)
- Oracle 2026-07-11 API 审查: session `ses_0b026333bffePgrqVq7PDJNeR1`
- Metis pre-impl review: session `ses_0b1a0cdb1ffenbhbciQ1n0x236` (per checklist H)
- ADR-0016: [docs/adr/0016-blackwell-only-tcgen05.md](../../../docs/adr/0016-blackwell-only-tcgen05.md)
- ADR-0018: [docs/adr/0018-tcgen05-cta-group-restriction.md](../../../docs/adr/0018-tcgen05-cta-group-restriction.md)
- Ref: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
- [proposal.md](../../proposal.md), [design.md](../../design.md), [tasks.md](../../tasks.md)
- FU-1..FU-4 dependencies: [fix-tcgen05-mma-accumulator-and-f32-storage/proposal.md §Follow-Up Changes](../../fix-tcgen05-mma-accumulator-and-f32-storage/proposal.md)