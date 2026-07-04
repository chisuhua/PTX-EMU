## ADDED Requirements

### Requirement: WMMA-m8n8k4-f16-fragment-arithmetic MUST

The `WmmaHandler::processWmmaOperation` handler MUST correctly
implement the `wmma.mma.sync.aligned.m8n8k4.row.col.f32.f16.f16.f32`
fragment arithmetic when invoked on a uniform warp (all 32 lanes
active). Each element of the 8×4 result fragment MUST equal the
dot-product of the corresponding row of A (8×4 f16) and column of
B (4×8 f16) plus the C accumulator (8×4 f32), matching NVIDIA PTX
ISA §9.7.13 fragment layout.

`include/ptxsim/utils/half_utils.h` MUST be reused for f16↔f32
conversions; the handler MUST NOT duplicate conversion code.

#### Scenario: full-fragment-correctness
- **WHEN** `WmmaHandler::processWmmaOperation` runs with
  `active_mask == 0xFFFFFFFF` on a uniform warp loaded with
  deterministic f16 A, B and zero C fragments
- **THEN** all 32 result elements of the 8×4 fragment equal the
  hand-computed reference (`A.row[i] · B.col[j]` for `i,j ∈ [0,8)×[0,4)`)
- **AND** no element is left uninitialized (the post-c5
  silent-failure contract is preserved)

#### Scenario: divergent-warp-throws-execution-state-exception
- **WHEN** `WmmaHandler::processWmmaOperation` runs with
  `active_mask != 0xFFFFFFFF`
- **THEN** `ExecutionStateException` (NOT `UnsupportedInstructionException`)
  is thrown
- **AND** dst register is NOT written

### Requirement: WMMA-file-renamed-to-wmma-cpp MUST

The handler implementation file MUST be renamed from
`src/ptxsim/instructions/tensor.cpp` to
`src/ptxsim/instructions/wmma.cpp`, with the corresponding update
in `src/CMakeLists.txt`. Class name `WmmaHandler` and X-Macro
registration are unchanged.

#### Scenario: rename-builds-and-tests-pass
- **WHEN** the rename is applied and the project is rebuilt
- **THEN** `cmake --build build` succeeds with no source-level
  errors
- **AND** `ctest -L "unit;integration;e2e"` passes with no regression
- **AND** `grep -rn "tensor.cpp" src/CMakeLists.txt` returns the
  updated filename

### Requirement: WMMA-e2e-gemm-kernel-passes MUST

A type-3 e2e test (compiled CUDA kernel → PTX extraction →
simulator execution → host-side verification) MUST demonstrate
that the implemented WMMA path produces correct output for at
least one small matrix-multiply shape.

#### Scenario: small-matmul-correctness
- **WHEN** a 16×16 GEMM kernel is run with deterministic f16
  inputs (A and B) and zero accumulator (C)
- **THEN** the e2e test verifies that `C[i][j] == sum_k A[i][k] * B[k][j]`
  for all `i,j ∈ [0,16)×[0,16)` within f32 rounding tolerance

### Requirement: stub-explicit-failure-wmma-relaxed MUST

The `openspec/specs/stub-explicit-failure/spec.md` MUST be
modified so that `WMMA-Stub-Throws-Exception MUST` is changed to
reflect the new real-implementation behavior: the handler still
throws `ExecutionStateException` for divergent warps and
`UnsupportedInstructionException` for genuinely-unsupported
variants (e.g., mma.sp sparse, tcgen05 on sm_90), but the
default wmma.m8n8k4 path executes correctly.

#### Scenario: spec-relaxation-matches-implementation
- **WHEN** reading `openspec/specs/stub-explicit-failure/spec.md`
- **THEN** the WMMA requirement no longer mandates
  `UnsupportedInstructionException` for the m8n8k4 variant
- **AND** lists divergent-warp + unsupported-variant as the
  remaining throw conditions