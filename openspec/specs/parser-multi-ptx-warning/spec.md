# parser-multi-ptx-warning Specification

## Purpose
TBD - created by archiving change parser-completeness. Update Purpose after archive.
## Requirements
### Requirement: MultiPTX-Parser-Append-All MUST

The system SHALL restore `+=` 累加语义 at `src/ptx_parser/ptx_parser.cpp:60`,
ensuring all PTX sections extracted from cubin are concatenated into the
final `ptx_code` (instead of being overwritten by the last section only).

背景：实测 `src/utils/cubin_utils.cpp` 的 PTX section 提取循环已正确
实现累加语义（commit `c5d4e73` + c5 Fix #3）。parser 层 `ptx_code =`
覆盖语义与 cubin_utils 行为不一致，导致最终仅最后 section 生效。

#### Scenario: multi-section-all-ptx-kept
- **WHEN** cubin 包含 N 个 PTX section（N >= 2）
- **THEN** `ptx_code` 包含全部 N 个 section 的 PTX 代码（累加）
- **AND** 不再仅保留最后 1 个 section

#### Scenario: single-section-unaffected
- **WHEN** cubin 包含 1 个 PTX section
- **THEN** `ptx_code` 等于该 section 内容
- **AND** 行为与现状一致

### Requirement: MultiPTX-Parser-Warn-Emits MUST

The system SHALL emit `PTX_WARN_EMU` when the PTX section counter
accumulates more than 1 section, alerting the user that the cubin
contains multiple .cu sources.

警告 message **MUST** 包含 section 数量，**MUST** 不中断 kernel 执行
（warning 而非 error）。

#### Scenario: multi-section-triggers-warning
- **WHEN** PTX section 累加器 > 1
- **THEN** 调用 `PTX_WARN_EMU` 输出警告
- **AND** warning message 包含 section 数量（如 `"Multiple PTX sections found in cubin (count=N) — all sections accumulated"`)
- **AND** kernel 执行**不中断**

#### Scenario: single-section-no-warning
- **WHEN** PTX section 累加器 == 1
- **THEN** 无 `PTX_WARN_EMU` 输出
- **AND** 行为与现状一致

#### Scenario: warning-includes-section-count
- **WHEN** 警告触发
- **THEN** warning message **MUST** 包含 section 数量
- **AND** 引用 PTX 解析器代码位置（如 `"src/ptx_parser/ptx_parser.cpp:extract_ptx_sections()"`）

### Requirement: MultiPTX-Symbol-Conflict-Documented MUST

The system SHALL document in AGENTS.md the potential risk of symbol
conflicts when multiple PTX sections are accumulated (不同 section 可能
定义同名符号）。

#### Scenario: AGENTSMD-known-limitations-multiptx
- **WHEN** 阅读根 `AGENTS.md` "已知限制"章节
- **THEN** Multi-PTX cubins 条目描述为：
  - "输出 PTX_WARN_EMU 警告"
  - "保留所有 PTX sections（累加）"
  - "潜在风险：不同 section 可能定义同名符号（warning 告知用户检查）"

#### Scenario: KNOWN-LIMITATIONS-wmma-tensor-not-stub
- **WHEN** 阅读根 `AGENTS.md` "已知限制"章节
- **THEN** WMMA/Tensor Core 条目描述为"完整实现 tcgen05.mma/ld/st/commit/wait"
- **AND** 不再描述为 "是 stub"（sync-readme-after-tcgen05 已修复，但需二次确认）

### Requirement: MultiPTX-Parser-Oracle-Test MUST

The system SHALL provide unit test coverage for multi-PTX parser behavior.

#### Scenario: test-multi-ptx-warning-exists
- **WHEN** `tests/unit/parser/test_multi_ptx.cpp` 存在
- **THEN** 至少 3 个测试场景：
  - (1) multi-section cubin → `PTX_WARN_EMU` 触发 + 累加所有 sections
  - (2) single-section cubin → 无 warning + 行为与现状一致
  - (3) 0-section cubin（异常情况）→ 不 crash + 不 warning

#### Scenario: test-multi-ptx-mock
- **WHEN** 测试构造 multi-section cubin mock
- **THEN** mock 包含至少 2 个不同的 `.ptx` section 内容
- **AND** 测试验证 `ptx_code` 包含全部 section 内容（拼接字符串）
- **AND** 测试使用 capture/cout 验证 warning message 包含 "count="

### Requirement: No-Regression-MultiPTX MUST

Multi-PTX fix **MUST** 不破坏现有 single-PTX 测试。

#### Scenario: existing-ptx-tests-pass
- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh`
- **THEN** exit code 0
- **AND** 与 baseline 对比无新增 FAIL
- **AND** 所有现有 e2e kernel（cute_rmsnorm_debug, barrier_warp_sync 等）仍 PASS

#### Scenario: baseline-comparison-required
- **WHEN** Phase 2 完成前
- **THEN** 已在 `.worktrees/parser-completeness-baseline` 中建立 ctest baseline
- **AND** Phase 2 实施后 diff baseline 验证 0 regression

