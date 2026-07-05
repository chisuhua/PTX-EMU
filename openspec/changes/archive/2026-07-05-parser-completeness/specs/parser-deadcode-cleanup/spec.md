# parser-deadcode-cleanup Specification

## Purpose

清理 PTX-EMU parser 代码中积累的死代码与过期 TODO/FIXME 注释，建立 lessons-learned §20 4 类表象对照的**清理契约**：死代码删除（无行为变更）+ 过期注释更新（语义保留）。

来源：Metis pre-implementation review（2026-07-05）实证分类 12 条债务中的 7 条死代码/过期注释清理需求。

## ADDED Requirements

### Requirement: CalculateTypeSize-Dead-Code-Removed MUST

The system SHALL physically remove `PtxVisitor::calculateTypeSize` function
body and declaration from `src/ptx_parser/ptx_visitor.cpp:323-326` and
`include/ptx_parser/ptx_visitor.h`.

理由：实测 `calculateTypeSize()` 调用者数量为 0（grep 验证），
替代品 `getBytes(const std::vector<Qualifier>&)` 在
`src/ptxsim/utils/qualifier_utils.cpp:56` 已正确实现类型大小计算。

#### Scenario: calculateTypeSize-removed
- **WHEN** `grep -rn "calculateTypeSize" src/ include/ tests/ --include="*.cpp" --include="*.h"`
- **THEN** 输出为空（仅 0 匹配）
- **AND** `wc -l src/ptx_parser/ptx_visitor.cpp` 减少 ≥ 4 行

#### Scenario: getBytes-still-functional
- **WHEN** PTX 解析器需要计算类型大小
- **THEN** 调用 `qualifier_utils.cpp::getBytes(qualifier)` 替代
- **AND** 行为与删除前一致（qualifier 列表遍历方式由 commit `d163e7b` lessons-learned §5 修复）

### Requirement: ProcessFunctionAttributes-Dead-Code-Removed MUST

The system SHALL physically remove `PtxVisitor::processFunctionAttributes`
function body and declaration from `src/ptx_parser/ptx_visitor.cpp:303-310`
and the corresponding TODO comment at `ptx_visitor.cpp:607`.

理由：实测 `processFunctionAttributes()` 调用者数量为 0（grep 验证）。
`src/grammar/ptxParser.g4` 当前语法**不支持** `functionAttribute` 规则
（grep 验证），实施此函数需要 ANTLR 语法扩充——触及
`ptx-grammar-modification` 高危流程（强制 TDD + `test_all_ptx.sh` 全量验证），
属于独立 change 范畴。

#### Scenario: processFunctionAttributes-removed
- **WHEN** `grep -rn "processFunctionAttributes" src/ include/ tests/`
- **THEN** 输出为空（仅 0 匹配）
- **AND** `ptx_visitor.cpp` line 607 的 `// TODO: Process function attributes` 注释删除

#### Scenario: grammar-remains-unchanged
- **WHEN** 死代码删除后
- **THEN** `src/grammar/ptxParser.g4` 无变更
- **AND** `./tests/ptx/test_all_ptx.sh` 仍 100% PASS（验证无 ANTLR regression）

### Requirement: Stale-FIXME-Comments-Updated MUST

The system SHALL update 5 处过期 TODO/FIXME 注释，反映实际代码状态（按
lessons-learned §6 + §20 第 4 类表象对照）。

#### Scenario: statement-context-h24-optional-semantics
- **WHEN** 阅读 `include/ptx_ir/statement_context.h:24`
- **THEN** `size` 字段注释描述为 "optional total bytes（may be unset for forward declarations）"
- **AND** 不再包含 "FIXME total size in bytes"
- **AND** 引用 `ptx_interpreter.cpp:124-145` 作为已知 `std::optional` 处理点

#### Scenario: barrier-cpp11-stage3-todo-removed
- **WHEN** 阅读 `src/ptxsim/instructions/barrier.cpp:11`
- **THEN** 不包含 Stage 3 TODO 列表（所有子项已 ✓）
- **AND** 文件头注释保留 Stage 3 描述（历史记录）

#### Scenario: ptx-interpreter-cpp124-bugfix-commented
- **WHEN** 阅读 `src/cudart/ptx_interpreter.cpp:124`
- **THEN** BUGFIX 注释保留为历史记录（commit 追溯）但精简为 1-2 行
- **AND** 不包含 "FIXME 字段，几乎未设置" 字样（避免误导新读者）

#### Scenario: ptx-visitor-cpp435-436-cleaned
- **WHEN** 阅读 `src/ptx_parser/ptx_visitor.cpp:435-436`
- **THEN** `// P0 cleanup: silently using "TODO" as identifier was a bug` 注释精简为说明性注释
- **AND** 解释此位置处理的是 anonymous declaration naming（避免再次误用 "TODO" 标识符）

#### Scenario: thread-context-cpp171-181-dumpcontext-marked
- **WHEN** 阅读 `src/ptxsim/core/thread_context.cpp:171,181`
- **THEN** dumpContext 中的 TODO 注释更新为 "DEBUG STUB: register dump not implemented"
- **AND** 引用相关 issue 或 future work 标记（如有）

### Requirement: ThreadContext-410-Design-Decision-Documented MUST

The system SHALL replace `src/ptxsim/core/thread_context.cpp:410` 的
`// FIXME should use stmt qualifier?` 注释为 Design Decision 风格说明
（lessons-learned §6 模板），明确 `instr.qualifiers` 是 canonical source。

#### Scenario: thread-context-410-design-decision
- **WHEN** 阅读 `src/ptxsim/core/thread_context.cpp:410`
- **THEN** FIXME 注释替换为：
  ```
  // Design Decision: instr.qualifiers 是 operand qualifier 的 canonical
  // source。stmt.qualifier 在 CFGBuilder 阶段可能未设置或过时。
  // 当前实现正确，无需修改。
  ```
- **AND** `commit_operand()` 仍使用 `instr.qualifiers`（不变）
- **AND** 任何未来 reader 看到此注释理解为何不修改

### Requirement: No-Behavior-Change-Cleanup MUST

死代码删除 + 注释更新**MUST** 零行为变更。所有现有测试**MUST** 100% PASS，
**禁止**任何新增 FAIL。

#### Scenario: ctest-full-pass
- **WHEN** 运行 `ctest --output-on-failure`
- **THEN** exit code 0
- **AND** 与 baseline 对比无新增 FAIL（仅 0 数量变化——已删除的测试仍 PASS 或被移除）
- **AND** `cute_rmsnorm_debug` + `barrier_warp_sync` + `cute_rmsnorm_bar_sync_pattern` 关键 e2e 仍 PASS

#### Scenario: wmma-handler-unaffected
- **WHEN** 死代码清理后
- **THEN** `src/ptxsim/instructions/wmma.cpp` 中 `WmmaHandler::processWmmaOperation` 行为不变
- **AND** tcgen05 路径仍按 `wmma-tensor-core` spec 执行真实 arithmetic
- **AND** pre-Blackwell 路径仍抛 `UnsupportedInstructionException`

### Requirement: AGENTSMD-Parse-Section-Sync MUST

The system SHALL sync documentation reflecting dead code removal +
stale comment cleanup, per lessons-learned Checklist I.

#### Scenario: parser-AGENTSMD-no-todo-references
- **WHEN** 阅读 `src/ptx_parser/AGENTS.md`（如不存在可创建）
- **THEN** STRUCTURE 章节不引用已删除的 `calculateTypeSize` 或 `processFunctionAttributes`
- **AND** KNOWN ISSUES 章节不包含 "calculateTypeSize hardcoded 4" 等已修复条目

#### Scenario: core-AGENTSMD-dumpcontext-note
- **WHEN** 阅读 `src/ptxsim/core/AGENTS.md`
- **THEN** 如有 DUMP_CONTEXT 章节，描述 `dumpContext()` 为 debug stub（register dump not implemented）
- **AND** 引用 `thread_context.cpp:171,181` 作为代码位置