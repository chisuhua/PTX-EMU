# stub-explicit-failure Specification

## Purpose
TBD - created by archiving change replace-silent-stub-failures. Update Purpose after archive.
## Requirements
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

### Requirement: MultiPTX-Extraction-Warns MUST

`src/utils/cubin_utils.cpp` 的 PTX section 提取循环**MUST** 维护 section
计数器，当计数器 > 1 时调用 `PTX_WARN_EMU` 输出警告，提示用户二进制
含多个 .cu 来源。警告**MUST** 包含 section 数量。

> 背景：实测 `cubin_utils.cpp` 当前 append 实现已正确（无功能 bug），
> 警告仅起"诊断/日志"作用，不中断 kernel 执行。

#### Scenario: Multi-section-triggers-warning
- **WHEN** cubin 提取循环累计 section 数 > 1
- **THEN** 调用 `PTX_WARN_EMU` 输出警告
- **AND** warning message 包含 section 数量（如 `"Multiple PTX sections found in cubin (count=N)"`）
- **AND** kernel 执行**不中断**（warning 而非 error）

#### Scenario: Single-section-no-warning
- **WHEN** cubin 仅含 1 个 PTX section
- **THEN** 无 `PTX_WARN_EMU` 输出
- **AND** 行为与现状一致

### Requirement: Dead-WmmaCpp-Removed MUST

`src/ptxsim/instructions/wmma.cpp` **MUST** 物理删除。

> 背景：实测 `WMMA_Handler`（全大写）是死代码，未被 CMake 编译，
> 保留是编译错误源（LSP 已报告 `WMMA_Handler` undeclared identifier）。

#### Scenario: wmma-cpp-deleted
- **WHEN** `rm src/ptxsim/instructions/wmma.cpp`
- **THEN** 文件不存在于仓库
- **AND** `src/CMakeLists.txt` 无需修改（本就未引用该文件）
- **AND** `grep -rn "WMMA_Handler" src/ include/` 输出为空

### Requirement: AGENTSMD-KnownStubs-Updated MUST

SHALL sync documentation: the KNOWN STUBS section in
`src/ptxsim/instructions/AGENTS.md` and the "已知限制" table in the
root `AGENTS.md` MUST describe the new explicit-failure behavior
(throw / warn) instead of "silent stub".

`src/ptxsim/instructions/AGENTS.md` 的 KNOWN STUBS 章节 + 根 `AGENTS.md`
的"已知限制"章节**MUST** 同步反映新行为：
- WMMA/Tensor Core：从"是 stub" → "抛 `UnsupportedInstructionException` 异常"
- Multi-PTX cubins：从"仅提取第一个 PTX" → "输出 PTX_WARN，保留最后一个 PTX"

#### Scenario: AGENTSMD-knstubs-reflects-throw
- **WHEN** 阅读 `src/ptxsim/instructions/AGENTS.md` KNOWN STUBS 章节
- **THEN** wmma/tensor 相关描述**MUST** 包含 "throw" 或 "UnsupportedInstructionException"
- **AND** 不再描述为"stub 无操作"

#### Scenario: Root-AGENTSMD-known-limitations-sync
- **WHEN** 阅读根 `AGENTS.md` "已知限制"章节
- **THEN** WMMA/Tensor Core 条目描述为"抛异常"而非"是 stub"
- **AND** Multi-PTX cubins 条目描述为"输出 warning"

### Requirement: No-Regression-AllTestsPass MUST

现有所有 ctest（unit + integration + e2e + PTX 语法测试）**MUST** 全部
PASS，**禁止**任何新增 FAIL。

特别需要验证：
- `cute_rmsnorm_debug`（不含 wmma/mma 指令）
- `barrier_warp_sync`
- `cute_rmsnorm_bar_sync_pattern`
- 所有 `barrier` 和 `warp` label 测试

#### Scenario: sanity-sh-quick-passes
- **WHEN** 运行 `./scripts/sanity.sh --quick`
- **THEN** exit code 0
- **AND** 无新增 FAIL

#### Scenario: no-wmma-test-regression
- **WHEN** 运行 `ctest -L "unit;integration;e2e"`
- **THEN** 与 baseline (`.worktrees/fix-pre-p0-baseline`) 对比无新增 FAIL
- **AND** 无 wmma/mma 相关测试被破坏

### Requirement: Artifacts-Git-Tracked MUST

SHALL be `git add`-ed: all OpenSpec artifacts (`proposal.md` /
`design.md` / `specs/` / `tasks.md`) MUST be tracked before the
implementation commits merge into main. Untracked artifacts in the
working tree after merge MUST NOT occur.

OpenSpec artifacts (`proposal.md` / `design.md` / `specs/` / `tasks.md`)
**MUST** 在实施 commits 合并前已 `git add` 并 tracked。**禁止** working tree
遗留 untracked artifacts（避免 lessons-learned §6 `cleanup-deprecated-barrier-apis`
模式：实施 commits 遗漏 artifacts → 12 天后债务审计误判为 active debt）。

#### Scenario: artifacts-tracked-pre-merge
- **WHEN** 准备合并 `fix/replace-silent-stub-failures` 分支到 main
- **THEN** `git ls-files openspec/changes/replace-silent-stub-failures/` 不为空
- **AND** `git status openspec/changes/replace-silent-stub-failures/` 无 untracked files
- **AND** 至少 1 个 commit message 包含 "docs(openspec):" 前缀（artifacts 提交）

#### Scenario: artifacts-match-implementation
- **WHEN** 实施完成归档时
- **THEN** artifacts 中的 requirements 与代码实现一一对应
- **AND** 无过期 task 编号（如已完成但仍标记 [ ]）

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

