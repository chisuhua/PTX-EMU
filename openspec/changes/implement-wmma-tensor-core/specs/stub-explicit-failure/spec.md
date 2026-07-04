## MODIFIED Requirements

> **Delta scope**: per [ADR-0016](../../../../docs/adr/0016-blackwell-only-tcgen05.md),
> pre-Blackwell WMMA throws `UnsupportedInstructionException` **permanently**
> (no future reversion). Blackwell `tcgen05.*` follows the real-execution path
> described in the `wmma-tensor-core` spec.

### Requirement: WMMA-Stub-Throws-Exception MUST

The system SHALL throw `UnsupportedInstructionException` from
`WmmaHandler::processWmmaOperation` (in `src/ptxsim/instructions/wmma.cpp`,
formerly `tensor.cpp`) for any **pre-Blackwell** wmma.* or mma.* instruction
that does not have an implemented Blackwell `tcgen05.*` equivalent. The
exception class auto-sets `error_code` to `UNSUPPORTED_INSTRUCTION`. This
behavior is **permanent** per ADR-0016 (no future change) and replaces the
prior silent no-op behavior that left dst registers with uninitialized
values.

For Blackwell `tcgen05.*` instructions (sm_100 / sm_120), the handler
SHALL instead execute the real fragment arithmetic per the `wmma-tensor-core`
spec, throwing `ExecutionStateException` only for divergent warp on the
sync path (the async path delegates synchronization to `TcQueue::wait`).

`WmmaHandler::processWmmaOperation` 在 `src/ptxsim/instructions/wmma.cpp`
针对 pre-Blackwell wmma.* 或 mma.* 指令（无 Blackwell tcgen05.* 等价实现）
**MUST** 调用 `PTX_ERROR_EMU` 日志宏 + `throw UnsupportedInstructionException`
异常，**禁止**静默无操作（silent failure）。该行为永久（ADR-0016）。

异常构造**MUST** 显式传 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION` 作为第二参数
（防止被默认记为 `INTERNAL_ERROR`）。

异常 message **MUST** 以 `"wmma."` 前缀（包含指令名前缀便于日志过滤）。

#### Scenario: WmmaHandler-throws-when-invoked-pre-blackwell
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for any
  pre-Blackwell wmma.* instruction (`wmma.mma.sync.*`, `wgmma.async.*`,
  `mma.sync.*`)
- **THEN** `PTX_ERROR_EMU` is logged
- **AND** `UnsupportedInstructionException` is thrown
- **AND** the exception `what()` message starts with `"wmma."`
- **AND** the exception `error_code()` returns `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`

#### Scenario: WmmaHandler-throws-not-mutable-state
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked
- **THEN** 目标寄存器**保持未初始化**（不写入任何值）
- **AND** 其他 context state（PC, predicate）**不修改**

#### Scenario: WmmaHandler-executes-real-arithmetic-blackwell
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for an
  implemented Blackwell `tcgen05.mma.cta_group::1.kind::f16` variant
  on a uniform warp
- **THEN** the handler executes the real fragment arithmetic per the
  `wmma-tensor-core` spec
- **AND** no exception is thrown

#### Scenario: WmmaHandler-divergent-warp-async-wait-blackwell
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

Blackwell `tcgen05.*` variants follow the real-execution path per the
`wmma-tensor-core` spec.

`tensor.cpp`（已改名为 `wmma.cpp`）中所有 pre-Blackwell Tensor Core 相关
handler **MUST** 遵循 WMMA-Stub-Throws-Exception 同等约束：抛
`UnsupportedInstructionException` with `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`
而非静默无操作。该行为永久（ADR-0016）。

#### Scenario: TensorCore-stub-reuses-exception-pre-blackwell
- **WHEN** 任何 pre-Blackwell Tensor Core stub handler 被调用
- **THEN** 抛出 `UnsupportedInstructionException`
- **AND** message 包含指令名（如 `"tcgen05.mma"` 用于未实现变体）

#### Scenario: TensorCore-stub-executes-real-blackwell
- **WHEN** 任何已实现的 Blackwell Tensor Core stub handler 被调用
- **THEN** 真实执行 fragment arithmetic
- **AND** 不抛异常

### Requirement: Stub-Explicit-Failure-Permanent-Policy MUST

SHALL NOT implement pre-Blackwell WMMA permanently per ADR-0016: the
pre-Blackwell WMMA throw behavior is **permanent**. Future Tensor Core
work targets Blackwell `tcgen05.*` only.

The system SHALL NOT implement `wmma.mma.sync.*`, `wgmma.async.*`, or
`mma.sync.*` (sm_70 / sm_75 / sm_80 / sm_86 / sm_90) in any future
change.

Future Tensor Core work targets Blackwell `tcgen05.*` only.

#### Scenario: future-change-must-not-add-pre-blackwell-wmma
- **WHEN** a future OpenSpec change is proposed that adds pre-Blackwell
  WMMA implementation
- **THEN** the proposal MUST be rejected at review with reference to
  ADR-0016 unless it explicitly Supersedes this ADR

#### Scenario: AGENTSMD-must-reference-ADR-0016
- **WHEN** reading `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS section
- **THEN** the WMMA entry MUST reference ADR-0016 as the policy basis
- **AND** the root `AGENTS.md` "已知限制" table WMMA row MUST also reference
  ADR-0016

#### Scenario: mixed-cubin-pre-blackwell-and-blackwell-coexist
- **WHEN** a single cubin contains BOTH sm_80 `wmma.mma.sync` instructions
  (from cute sm_80 fallback) AND sm_100 `tcgen05.mma` instructions (from
  the primary path) — as can occur when cute templates link sm_80 fallback
  alongside sm_100 code paths
- **THEN** `WmmaHandler::processWmmaOperation` dispatches per-qualifier:
  sm_80 path throws `UnsupportedInstructionException`, sm_100 path executes
  real fragment arithmetic
- **AND** the two paths do not interfere (throw on sm_80 does not corrupt
  TMEM state from sm_100, and vice versa)

#### Scenario: phase-0-blackwell-still-throws
- **WHEN** Phase 0 (infrastructure) of `implement-wmma-tensor-core` is
  merged to main, but Phase 1 (tcgen05.mma implementation) has not yet
  been merged
- **THEN** `WmmaHandler::processWmmaOperation` still throws
  `UnsupportedInstructionException` for ALL wmma.* instructions including
  Blackwell tcgen05.mma (temporary behavior)
- **AND** the exception message includes a reference to
  "implement-wmma-tensor-core Phase 1"
- **NOTE**: this scenario MUST be removed during the archive of
  `implement-wmma-tensor-core` (Phase 3), as it documents a transitional
  state that expires when Phase 1 is complete