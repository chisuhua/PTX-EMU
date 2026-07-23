## ADDED Requirements

> **Scope**: Blackwell `tcgen05.*` feature 层 (Phase 1-3), per [ADR-0016](../../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md).
> 本 spec 覆盖 5 个 handler-implementation MUST (`tcgen05.mma`, `ld/st`, `commit/wait`, GEMM e2e, AGENTS sync, rename);
> 4 个基础设施子系统 MUST (TMA + TMEM + cluster + TcQueue) 由前置 change
> `implement-wmma-tensor-core-phase-0-infra` 交付并 publish。
>
> pre-Blackwell WMMA 永久抛 `UnsupportedInstructionException` (per ADR-0016).
> delta spec in `specs/stub-explicit-failure/spec.md`.

### Requirement: Tcgen05-Mma-Fragment-Arithmetic MUST

The system SHALL implement `tcgen05.mma.cta_group::1.kind::f16` real
fragment arithmetic in `src/ptxsim/instructions/wmma.cpp`. The handler
SHALL read A and B fragments from TMEM via TMA, perform the matrix
multiplication, and write the result to TMEM, delegating to `TcQueue`
(delivered by `implement-wmma-tensor-core-phase-0-infra`) for commit-group
management. `include/ptxsim/utils/half_utils.h` MUST be reused for
f16 ↔ f32 conversion.

> **⚠️ FRAGMENT LAYOUT UNVERIFIED AGAINST HARDWARE**
>
> PTX-EMU 是 Blackwell tcgen05 fragment layout 解释的 **primary source**。
> 无真实 Blackwell 硬件交叉验证，无第二独立参考实现 (GPGPU-Sim / MGPUSim /
> Accel-Sim 尚未实现 sm_100 tcgen05)。e2e test 的 hand-computed reference 来自同一份
> NVIDIA PTX ISA §9.7.13 规范。
>
> **风险**: 实施者与测试作者可能共享同一解读错误 — 测试通过但两者同样错误（vs 真实硬件）。
>
> **Mitigation**:
> 1. 每个输出 fragment element MUST 添加 `// UNVERIFIED-AGAINST-HARDWARE` 注释，标注
>    `lane_idx → (row, col)` 映射 + PTX ISA §9.7.13 章节行号引用。
> 2. Phase 1 commit 前必须有 ≥ **256** 个此类注释（覆盖 32 lane × 8x4 矩阵）。
> 3. 当真实 Blackwell 硬件可用时，整个 requirement 必须重新验证 against 实际硬件输出。

#### Scenario: full-fragment-correctness
- **WHEN** `WmmaHandler::processWmmaOperation` invoked with
  `tcgen05.mma.cta_group::1.kind::f16` qualifiers on a uniform warp
  loaded with deterministic A, B fragments and zero C accumulator
- **THEN** all output fragment elements equal the hand-computed reference
  (per NVIDIA PTX ISA §9.7.13 Blackwell fragment layout,
  **UNVERIFIED-AGAINST-HARDWARE**)
- **AND** each fragment element carries a `// UNVERIFIED-AGAINST-HARDWARE`
  annotation referencing the PTX ISA §9.7.13 line
- **AND** no element is left uninitialized

#### Scenario: tcgen05-mma-divergent-warp-async-wait
- **WHEN** `WmmaHandler::processWmmaOperation` invoked on a
  divergent warp (`active_mask != 0xFFFFFFFF`)
- **THEN** the handler does NOT throw at fetch (async semantics)
- **AND** `TcQueue::wait(group=N)` synchronizes correctly regardless of
  which lanes issued the original mma (per design.md Decision 5)

#### Scenario: half-utils-reuse-not-reimplemented
- **WHEN** `wmma.cpp` needs f16 ↔ f32 conversion
- **THEN** `include/ptxsim/utils/half_utils.h::f16_to_f32` is used
- **AND** no inline conversion logic is duplicated in `wmma.cpp`

### Requirement: Tcgen05-Ldst-CommitWait-Full-Flow MUST

The system SHALL implement `tcgen05.ld`, `tcgen05.st`, `tcgen05.commit`,
and `tcgen05.wait` instructions, integrating with the Phase 0
infrastructure (TMA + TMEM + TcQueue; delivered by phase-0-infra archive).
A complete mma sequence (ld → mma → commit → wait → st) SHALL execute
end-to-end on TMEM.

The `tcgen05.wait` handler **MUST** call `TcQueue::wait(group_id)` rather
than introducing a new exec state. The BAR_SYNC state translation chain
established in phase-0-infra is reused.

#### Scenario: ld-mma-commit-wait-st-roundtrip
- **WHEN** a warp executes the full sequence:
  - `tcgen05.ld` (load A/B from TMA descriptor into TMEM)
  - `tcgen05.mma` (compute fragment)
  - `tcgen05.commit`
  - `tcgen05.wait`
  - `tcgen05.st` (store result from TMEM via TMA descriptor)
- **THEN** the final memory state matches the hand-computed reference
  for all fragment elements

#### Scenario: commit-group-counter-increments
- **WHEN** `tcgen05.commit` is invoked three times with group ids 1, 2, 3
- **THEN** `TcQueue::commit_group_counter` reaches 3
- **AND** `wait(2)` returns after commit-2 completes (not requiring
  commit-3 to complete)

#### Scenario: tcgen05-wait-reuses-BAR-SYNC (no new exec state)
- **WHEN** `tcgen05.wait` blocks the calling warp
- **THEN** the wait handler delegates to `TcQueue::wait(group_id)` which
  uses `set_warp_state(BAR_SYNC)` (NOT a new `TC_WAIT` state)
- **AND** no new exec state enum value is added in this change

### Requirement: Blackwell-GEMM-E2E-Kernel-Passes MUST

SHALL produce correct output for a Blackwell `tcgen05` matrix-multiply
kernel. The e2e test MUST verify correctness via host-side comparison.

A type-3 e2e test (compiled CUDA kernel → PTX extraction → simulator
execution → host-side verification) SHALL demonstrate that the
implemented Blackwell tcgen05 path produces correct output for a
small matrix-multiply.

#### Scenario: small-matmul-correctness
- **WHEN** a 16×16 GEMM kernel is run with deterministic f16 inputs and
  zero accumulator, target sm_100
- **THEN** the e2e test verifies `C[i][j] == sum_k A[i][k] * B[k][j]`
  for all `i,j ∈ [0,16)²` within f32 rounding tolerance

#### Scenario: e2e-kernel-compiles-and-runs-blackwell-path
- **WHEN** `tests/e2e/kernel/test_blackwell_gemm.cu` is compiled for
  sm_100 with cute headers from `bench/cute/include/` and executed via
  `cudaLaunchKernel` (fake libcudart interception)
- **THEN** the PTX is extracted, parsed by ANTLR, and executed by the
  Blackwell execution infrastructure (TMA + TMEM + TcQueue ready)
- **AND** `cudaDeviceSynchronize()` returns without timeout
- **AND** both the 16×16 GEMM correctness test and the identity-matrix
  sanity test pass with zero mismatches

#### Scenario: e2e-kernel-no-regression-on-existing-tests
- **WHEN** the e2e GEMM test is registered in the CMake build
- **THEN** all existing ctest targets continue to pass (no regression
  on `ctest -L "unit|integration|e2e"`)
- **AND** `./scripts/sanity.sh --quick` returns 0 unexpected FAIL

### Requirement: File-Renamed-To-wmma-cpp MUST

`src/ptxsim/instructions/wmma.cpp` SHALL be the file name. The previous
name `tensor.cpp` SHALL be renamed as part of this change (Phase 1.1).
The class name `WmmaHandler` and X-Macro registration are unchanged.

#### Scenario: rename-builds-and-tests-pass
- **WHEN** the project is built
- **THEN** `cmake --build build` succeeds with no source-level errors
- **AND** `ctest -L "unit|integration|e2e"` passes with no regression
- **AND** `grep -rn "tensor.cpp" src/ docs/ tests/` does not return
  the old filename (only post-rename references to `wmma.cpp` are valid)

### Requirement: AGENTS-Sync-Blackwell-Implemented MUST

SHALL synchronize AGENTS.md documentation to reflect the new Blackwell
`tcgen05` implementation. The pre-Blackwell throw behavior SHALL be
documented as permanent per ADR-0016.

The KNOWN STUBS section in `src/ptxsim/instructions/AGENTS.md` and
the "已知限制" table in the root `AGENTS.md` SHALL be updated to
describe the new Blackwell tcgen05 implementation.

#### Scenario: agents-md-reflects-blackwell-impl
- **WHEN** reading `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS section
- **THEN** the WMMA entry describes Blackwell tcgen05 as implemented
- **AND** references ADR-0016 for pre-Blackwell throw policy

#### Scenario: root-agents-md-known-limitations-sync
- **WHEN** reading the root `AGENTS.md` "已知限制" table
- **THEN** the WMMA entry states "Blackwell tcgen05 已实现；
  pre-Blackwell 永久抛异常（ADR-0016）"
