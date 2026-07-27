# ptx-visitor-operand-extraction — Spec

## Purpose

Qualifier/operand 解析 helpers (ptx_visitor.cpp:138-317 + :937-1067, 约 310 行) 是 PTX visitor 的核心工具层，被 19+ 处调用点引用。本 spec 定义提取这些 helpers 至独立 `ptx_visitor_operands.cpp` 文件的需求。

## ADDED Requirements

### Requirement: 提取范围

The system MUST extract qualifier/operand helpers and visitors to `ptx_visitor_operands.cpp`.

#### Scenario: 所有 helpers 在新文件定义

- **WHEN** 提取后运行 `grep -l 'tokenToQualifier\|extractQualifiersFromContext\|createOperandFromContext' src/ptx_parser/ptx_visitor*.cpp`
- **THEN** 仅 `ptx_visitor_operands.cpp` 命中
- **AND** `ptx_visitor.cpp` 命中 0

#### Scenario: 所有 visitor override 在新文件定义

- **WHEN** 提取后运行 `grep -l 'visitOperand\|visitSpecialRegister\|visitRegister\|visitImmediate\|visitAddress' src/ptx_parser/ptx_visitor*.cpp`
- **THEN** 仅 `ptx_visitor_operands.cpp` 包含这些 override
- **AND** `ptx_visitor.cpp` 仅 include 该子文件

### Requirement: 签名锁定

The system MUST preserve all function signatures without modification.

#### Scenario: extractQualifiersFromContext 签名不变

- **GIVEN** 19 个调用点依赖 `extractQualifiersFromContext` 签名
- **WHEN** 提取后
- **THEN** 签名 `(const auto* context) -> std::vector<Qualifier>` 与原文件一致
- **AND** 编译期所有 19 调用点通过

#### Scenario: ANTLR visitor override 签名兼容

- **GIVEN** ANTLR 生成的 base visitor 期望特定签名
- **WHEN** 提取后编译
- **THEN** 所有 visitor override 签名与 ANTLR 兼容
- **AND** 无 override 错误

### Requirement: 行为不变性

The system MUST maintain byte-level identical IR output for all PTX programs.

#### Scenario: 全 PTX 语法测试通过

- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh`
- **THEN** 47/47 测试全绿
- **AND** IR 输出与提取前字节级一致

### Requirement: 构建系统

The system MUST update `src/ptx_parser/CMakeLists.txt`.

#### Scenario: 新源文件被编译

- **WHEN** 运行 `cmake --build build`
- **THEN** `ptx_visitor_operands.cpp` 被编译
- **AND** 链接期无未定义符号

### Requirement: 验证

The system MUST pass all of:
- `./tests/ptx/test_all_ptx.sh` — PTX 语法测试全绿
- 所有 PTX 单元测试 — operand 相关测试零回归
- 集成测试 — 任何含 operand 解析的路径零回归

#### Scenario: 全量测试通过

- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh && ctest --output-on-failure`
- **THEN** 所有测试通过
- **AND** 零回归

## 关联

- `.opencode/skills/ptx-lessons-learned/SKILL.md:48-77` — §1 跨模块状态翻译
- `src/ptx_parser/AGENTS.md:48` — 类别分派文档
- `improvements/split-ptx-visitor-god-class.md` — 完整提案