## MODIFIED Requirements

> **Delta scope**: per [ADR-0016](../../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md),
> pre-Blackwell WMMA 抛 `UnsupportedInstructionException` **永久**（per
> `replace-silent-stub-failures` baseline + `implement-wmma-tensor-core-phase-0-infra`
> archive unchanged behavior）。
>
> 本 delta 由 `implement-wmma-tensor-core-tcgen05` (Phase 1-3) change 引入，
> 添加 Blackwell `tcgen05.*` 真实执行路径场景 + Phase 0-archive 期间 transitional
> scenarios（本 archive 后必须移除）。

### Requirement: WMMA-Stub-Throws-Exception MUST

The system SHALL throw `UnsupportedInstructionException` from
`WmmaHandler::processWmmaOperation` (in `src/ptxsim/instructions/wmma.cpp`,
renamed from `tensor.cpp` by `implement-wmma-tensor-core-tcgen05` Phase 1.1)
for any **pre-Blackwell** wmma.* or mma.* instruction that does not have
an implemented Blackwell `tcgen05.*` equivalent. The exception class
auto-sets `error_code` to `UNSUPPORTED_INSTRUCTION`. This behavior is
**permanent** per ADR-0016 (no future change).

For Blackwell `tcgen05.*` instructions (sm_100 / sm_120), the handler
SHALL instead execute the real fragment arithmetic per the
`wmma-tensor-core` spec, throwing `ExecutionStateException` only for
divergent warp on the sync path (the async path delegates synchronization
to `TcQueue::wait` per `implement-wmma-tensor-core-phase-0-infra` archive
Decision 7).

异常构造**MUST** 显式传 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION` 作为第二参数。
异常 message **MUST** 以 `"wmma."` 前缀。

#### Scenario: WmmaHandler-throws-when-invoked-pre-blackwell
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for any
  pre-Blackwell wmma.* instruction (`wmma.mma.sync.*`, `wgmma.async.*`,
  `mma.sync.*`)
- **THEN** `UnsupportedInstructionException` is thrown
- **AND** the exception `what()` message starts with `"wmma."`
- **AND** the exception `error_code()` returns `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`

#### Scenario: WmmaHandler-executes-real-arithmetic-blackwell (NEW Phase 1-3)
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for an
  implemented Blackwell `tcgen05.mma.cta_group::1.kind::f16` variant
  on a uniform warp
- **THEN** the handler executes the real fragment arithmetic per the
  `wmma-tensor-core` spec
- **AND** no exception is thrown

#### Scenario: WmmaHandler-divergent-warp-async-wait-blackwell (NEW Phase 1-3)
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for an
  implemented Blackwell variant on a divergent warp
- **THEN** the handler does NOT throw at fetch (async semantics)
- **AND** `TcQueue::wait(group=N)` synchronizes correctly regardless
  of which lanes issued the original mma

### Requirement: TensorCore-Stub-Throws-Exception MUST

SHALL follow the WMMA-Stub-Throws-Exception contract for all
**pre-Blackwell** Tensor Core variants: handlers MUST throw
`UnsupportedInstructionException` with `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`
instead of silently no-op'ing. This is **permanent** per ADR-0016.

#### Scenario: TensorCore-stub-reuses-exception-pre-blackwell
- **WHEN** any pre-Blackwell Tensor Core stub handler is called
- **THEN** `UnsupportedInstructionException` is thrown

#### Scenario: TensorCore-stub-executes-real-blackwell (NEW Phase 1-3)
- **WHEN** any implemented Blackwell Tensor Core stub handler is called
  (e.g. `tcgen05.mma.cta_group::1.kind::f16`)
- **THEN** real fragment arithmetic executes
- **AND** no exception is thrown

## ADDED Requirements

> **Delta scope**: 新增 Requirements 反映 ADR-0016 的 forward-looking policy
> （永久不实现 pre-Blackwell WMMA）+ Phase 1-3 transitional e2e 验证。
> 由 `implement-wmma-tensor-core-tcgen05` (Phase 1-3) change 引入。

### Requirement: Stub-Explicit-Failure-Permanent-Policy MUST

The system SHALL NOT implement `wmma.mma.sync.*`, `wgmma.async.*`, or
`mma.sync.*` (sm_70 / sm_75 / sm_80 / sm_86 / sm_90) in any future
change. Future Tensor Core work targets Blackwell `tcgen05.*` only.

#### Scenario: future-change-must-not-add-pre-blackwell-wmma
- **WHEN** a future OpenSpec change is proposed that adds pre-Blackwell
  WMMA implementation
- **THEN** the proposal MUST be rejected at review with reference to
  ADR-0016 unless it explicitly Supersedes this ADR

#### Scenario: AGENTSMD-must-reference-ADR-0016 (NEW Phase 1-3 doc sync)
- **WHEN** reading `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS section
  after `implement-wmma-tensor-core-tcgen05` archive
- **THEN** the WMMA entry MUST reference ADR-0016 as the policy basis
- **AND** the root `AGENTS.md` "已知限制" table WMMA row MUST also reference ADR-0016

#### Scenario: mixed-cubin-pre-blackwell-and-blackwell-coexist
- **WHEN** a single cubin contains BOTH sm_80 `wmma.mma.sync` instructions
  AND sm_100 `tcgen05.mma` instructions — as can occur when cute
  templates link sm_80 fallback alongside sm_100 code paths
- **THEN** `WmmaHandler::processWmmaOperation` dispatches per-qualifier:
  sm_80 path throws `UnsupportedInstructionException`, sm_100 path
  executes real fragment arithmetic
- **AND** the two paths do not interfere (throw on sm_80 does not corrupt
  TMEM state from sm_100, and vice versa)

### Requirement: Phase-1-3-Transitional-E2E-Validation MUST PASS (TRANSITIONAL)

The system SHALL validate that the full e2e execution pipeline works
correctly during the Phase 1-3 implementation window via the e2e GEMM
kernel (`tests/e2e/kernel/test_blackwell_gemm.cu`, targeting sm_100),
demonstrating that the ANTLR parser, WmmaHandler dispatch, and fake
libcudart interception work end-to-end for a Blackwell-style kernel.

> **⚠️ TRANSITIONAL** — this requirement validates the e2e pipeline
> during the Phase 1-3 implementation window. It SHALL be removed when
> `implement-wmma-tensor-core-tcgen05` is archived and published as
> main specs.

#### Scenario: e2e-gemm-ptx-parsed-and-executed-transitional
- **GIVEN** Phase 1-3 tcgen05.mma + ld/st + commit/wait handlers are
  merged and built into `libcudart.so`
- **WHEN** the 16×16 GEMM kernel is compiled by nvcc for sm_100 and
  launched via `cudaLaunchKernel`
- **THEN** the ANTLR parser successfully parses the PTX
- **AND** the emulator executes all instructions without timeout
- **AND** `cudaDeviceSynchronize()` returns cudaSuccess
- **AND** the computed output matches the host-side reference within
  f32 rounding tolerance

#### Scenario: e2e-gemm-no-stub-fallback-transitional
- **WHEN** the e2e GEMM kernel PTX is processed by WmmaHandler dispatch
- **THEN** no `UnsupportedInstructionException` is thrown for any
  instruction in the kernel (the standard CUDA ops are handled by
  existing handlers; any tcgen05 ops dispatch through WmmaHandler
  which executes real arithmetic)
- **AND** the KNOWN STUBS section in `src/ptxsim/instructions/AGENTS.md`
  no longer lists WMMA as a stub (marked as Blackwell implemented)
