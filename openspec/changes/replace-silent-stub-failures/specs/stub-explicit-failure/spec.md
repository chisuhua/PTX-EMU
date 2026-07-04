## ADDED Requirements

### Requirement: WMMA-Stub-Throws-Exception

`WmmaHandler::processWmmaOperation` 在 `src/ptxsim/instructions/tensor.cpp:8-15`
**MUST** 调用 `PTX_ERROR_EMU` 日志宏 + `throw UnsupportedInstructionException`
异常，**禁止**静默无操作（silent failure）。

异常构造**MUST** 显式传 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION` 作为第二参数
（防止被默认记为 `INTERNAL_ERROR`）。

异常 message **MUST** 以 `"wmma."` 前缀（包含指令名前缀便于日志过滤）。

#### Scenario: WmmaHandler-throws-when-invoked
- **WHEN** `WmmaHandler::processWmmaOperation` 被调用（任意 operands）
- **THEN** 调用 `PTX_ERROR_EMU` 输出错误日志
- **AND** 抛出 `UnsupportedInstructionException`
- **AND** 异常的 `what()` message 以 `"wmma."` 开头
- **AND** 异常的 `error_code()` 返回 `PtxEmuErrorCode::UNSUPPORTED_INSTRUCTION`

#### Scenario: WmmaHandler-throws-not-mutable-state
- **WHEN** `WmmaHandler::processWmmaOperation` 被调用
- **THEN** 目标寄存器**保持未初始化**（不写入任何值）
- **AND** 其他 context state（PC, predicate）**不修改**

### Requirement: TensorCore-Stub-Throws-Exception

`tensor.cpp` 中所有 Tensor Core 相关 handler（共享 `tensor.cpp` 文件）
**MUST** 遵循 WMMA-Stub-Throws-Exception 同等约束。

> 注：当前 `tensor.cpp` 唯一 handler 是 `WmmaHandler::processWmmaOperation`，
> 文件名误导是已知技术债（应随真实实现一起改名 `tensor.cpp` → `wmma.cpp`）。

#### Scenario: TensorCore-stub-reuses-exception
- **WHEN** 任何 Tensor Core stub handler 被调用
- **THEN** 抛出 `UnsupportedInstructionException`
- **AND** message 包含指令名（如 `"tcgen05.mma"`）

### Requirement: MultiPTX-Extraction-Warns

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

### Requirement: Dead-WmmaCpp-Removed

`src/ptxsim/instructions/wmma.cpp` **MUST** 物理删除。

> 背景：实测 `WMMA_Handler`（全大写）是死代码，未被 CMake 编译，
> 保留是编译错误源（LSP 已报告 `WMMA_Handler` undeclared identifier）。

#### Scenario: wmma-cpp-deleted
- **WHEN** `rm src/ptxsim/instructions/wmma.cpp`
- **THEN** 文件不存在于仓库
- **AND** `src/CMakeLists.txt` 无需修改（本就未引用该文件）
- **AND** `grep -rn "WMMA_Handler" src/ include/` 输出为空

### Requirement: AGENTSMD-KnownStubs-Updated

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

### Requirement: No-Regression-AllTestsPass

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

### Requirement: Artifacts-Git-Tracked

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