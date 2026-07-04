## MODIFIED Requirements

### Requirement: WMMA-Stub-Throws-Exception MUST

The system SHALL throw `UnsupportedInstructionException` from
`WmmaHandler::processWmmaOperation` for any wmma variant not yet
implemented (e.g. mma.sp sparse, tcgen05 on sm_90). For
implemented variants (m8n8k4 f16 per `implement-wmma-tensor-core`)
the handler MUST execute the fragment arithmetic correctly
without throwing, and MUST throw `ExecutionStateException` if
the warp is divergent (`active_mask != 0xFFFFFFFF`).

`WmmaHandler::processWmmaOperation` 在 `src/ptxsim/instructions/tensor.cpp:8-15`
（或重命名后的 `wmma.cpp`）针对尚未实现的 wmma 变体
**MUST** 调用 `PTX_ERROR_EMU` 日志宏 + `throw UnsupportedInstructionException`
异常，**禁止**静默无操作（silent failure）。已实现变体按
`implement-wmma-tensor-core` 设计执行真实片段算术。

异常构造**MUST** 显式传 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION` 作为第二参数
（防止被默认记为 `INTERNAL_ERROR`）。

异常 message **MUST** 以 `"wmma."` 前缀（包含指令名前缀便于日志过滤）。

#### Scenario: implemented-variant-executes
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for an
  implemented variant (e.g. m8n8k4 f16) on a uniform warp
- **THEN** the handler writes the correct fragment result to dst
  (per `wmma-tensor-core` spec)
- **AND** no exception is thrown

#### Scenario: divergent-warp-throws-execution-state-exception
- **WHEN** `WmmaHandler::processWmmaOperation` runs with
  `active_mask != 0xFFFFFFFF`
- **THEN** `ExecutionStateException` is thrown
- **AND** dst register is NOT written

#### Scenario: unimplemented-variant-throws-unsupported
- **WHEN** `WmmaHandler::processWmmaOperation` is invoked for a
  not-yet-implemented variant (e.g. mma.sp, tcgen05.mma)
- **THEN** `UnsupportedInstructionException` is thrown with
  error_code `UNSUPPORTED_INSTRUCTION` and `wmma.*` prefix in
  `get_instruction_name()`

#### Scenario: divergent-state-preserved
- **WHEN** `WmmaHandler::processWmmaOperation` throws
- **THEN** 目标寄存器**保持未初始化**（不写入任何值）
- **AND** 其他 context state（PC, predicate）**不修改**