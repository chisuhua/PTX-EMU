## MODIFIED Requirements

### Requirement: Tests cover 6 extended handlers + FlashAttention scenarios (per Oracle 2026-07-11 audit)

The system SHALL provide 1 unit test + 1 integration test + 2 E2E kernels
covering the 6 extended handlers with **mixed oracle strategy**:
- **Unit**: hand-computed golden values, marked `UNVERIFIED-AGAINST-HARDWARE`
- **Integration**: `step_warp` + `execute_warp_instruction` driven
- **E2E**: real nvcc-generated PTX when available, fixtures otherwise

The system SHALL additionally provide 4 integration tests + 1 E2E kernel
covering the FlashAttention QK^T → softmax → PV data flow end-to-end,
per the `tcgen05-flashattention-coverage` spec (archived separately as
`tcgen05-flashattention-coverage` change). These tests validate:
- K=128 accumulator stability (FA-B1)
- mma → commit → wait → mma barrier sync (FA-B2)
- cp → mma data flow integrity (FA-B4)
- 2-warp C slot isolation (FA-B5)
- E2E FlashAttention mini-kernel (FA-D-E2E)

#### Scenario: tests PASS for both extended handlers and FlashAttention scenarios
- **WHEN** `cd build && ctest -L "unit;tcgen05|integration;tcgen05|e2e;tcgen05" -V` is run
- **THEN** 1 unit test + 1 integration test + 2 E2E kernels PASS (extended handlers)
- **AND** 4 integration tests + 1 E2E kernel PASS (FlashAttention scenarios, archived separately)
- **AND** no regression in core handler tests
- **AND** all new golden values include `// UNVERIFIED-AGAINST-HARDWARE` comment

#### Scenario: FlashAttention coverage extends handler validation scope
- **WHEN** the FlashAttention coverage tests run after FU-1..FU-4 (commit/wait group, idesc parsing, ld/st slot routing, multi-warp fragment) are archived
- **THEN** the 5 FlashAttention tests SHALL validate that the handler-level fixes work together for the QK^T → softmax → PV data flow
- **AND** any regression in FU-1..FU-4 SHALL be caught by the corresponding FA test (B2 catches FU-1, B4 catches FU-3, B5 catches FU-4, etc.)

## ADDED Requirements

### Requirement: handler-level test coverage SHALL extend to FlashAttention QK^T → softmax → PV data flow

The system SHALL extend the `tcgen05-handlers-extended` test coverage
from "single mma behavior correctness" to "complete FlashAttention
QK^T → softmax → PV data flow correctness", per the
`tcgen05-flashattention-coverage` capability.

#### Scenario: handler correctness covers single-op AND multi-op sequences
- **WHEN** any of the 11 S_TCGEN05_* handlers (per `include/ptx_ir/ptx_op.def:127-137`) is invoked
- **THEN** the handler SHALL be tested for both single-invocation behavior (existing tests) AND multi-invocation sequences (FlashAttention tests)
- **AND** the multi-invocation tests SHALL include at least one sequence test per handler type where applicable:
  - mma: K=128 accumulator (FA-B1)
  - commit + wait: barrier sync sequence (FA-B2)
  - cp → mma: data flow (FA-B4)
  - ld + st: tmem stage routing (FA-B2 secondary)
  - multi-warp mma: slot isolation (FA-B5)

#### Scenario: E2E coverage extends from single-kernel to multi-kernel
- **WHEN** the E2E test suite includes both single-kernel GEMM tests (`tests/e2e/kernel/test_blackwell_gemm.cu`, `tests/e2e/kernel/test_tcgen05_mma_gemm.cu`) and multi-instruction FA mini-kernel tests
- **THEN** the E2E coverage SHALL validate that the simulator handles real-world kernel patterns (not just isolated instructions)
- **AND** the FA mini-kernel SHALL be marked `[e2e][flashattention]` with priority-3 fallback if ptxas doesn't support sm_100 tcgen05

## Cross-Reference

- Oracle 2026-07-11 audit: session `ses_0aefd09c3ffeSqBIAGdxiRBFWC`
- Ref: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../../archive/2026-07-10-implement-tcgen05-handlers-extended/) (original handler implementation)
- [proposal.md](../../proposal.md), [design.md](../../design.md), [tasks.md](../../tasks.md)
- Related spec: [`tcgen05-flashattention-coverage`](../tcgen05-flashattention-coverage/spec.md)