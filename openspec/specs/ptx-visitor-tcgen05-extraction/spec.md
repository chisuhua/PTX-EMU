# ptx-visitor-tcgen05-extraction Specification

## Purpose
TBD - created by archiving change split-ptx-visitor-god-class. Update Purpose after archive.
## Requirements
### Requirement: 模块结构

The system MUST extract `visitTcgen05Inst` (ptx_visitor.cpp:841-902) to a new file `ptx_visitor_tcgen05.cpp`.

#### Scenario: 提取后 visitTcgen05Inst 仅在新文件中定义

- **WHEN** 运行 `grep -c 'visitTcgen05Inst' src/ptx_parser/ptx_visitor.cpp`
- **THEN** 命中 0（已移出主文件）
- **AND** `grep -c 'visitTcgen05Inst' src/ptx_parser/ptx_visitor_tcgen05.cpp` 命中 1

#### Scenario: 函数签名与 ANTLR 兼容

- **GIVEN** `PtxVisitor` 类继承自 ANTLR `PtxBaseVisitor`
- **WHEN** 提取后调用 `ptx_visitor_tcgen05.cpp` 中的 `visitTcgen05Inst`
- **THEN** 函数签名 `void PtxVisitor::visitTcgen05Inst(...)` 与原文件一致
- **AND** 编译期无 override 错误

### Requirement: C3 fix parse-tree walk 注释保留

The system MUST preserve the C3 fix parse-tree walk 注释 (ptx_visitor.cpp:863-877) 行级随迁.

#### Scenario: 关键注释完整保留

- **GIVEN** 提取后 `ptx_visitor_tcgen05.cpp` 中 `visitTcgen05Inst` 函数
- **WHEN** 检查 cta_group IMMEDIATE 提取逻辑
- **THEN** 注释 ptx_visitor.cpp:863-877 完整保留（lessons-learned §1 跨模块状态翻译）
- **AND** 行为与提取前字节级一致

### Requirement: 构建系统

The system MUST update `src/ptx_parser/CMakeLists.txt`.

#### Scenario: 新源文件在构建系统中

- **WHEN** 运行 `cmake --build build`
- **THEN** `ptx_visitor_tcgen05.cpp` 被编译
- **AND** 链接期无未定义符号

### Requirement: 验证

The system MUST pass all of:
- `./tests/ptx/test_all_ptx.sh` — PTX 语法测试全绿
- ctest — parser/visitor 相关测试通过
- CFG post-dominator 集成测试（ADR-0007）零回归

#### Scenario: 全量测试通过

- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh && ctest --output-on-failure`
- **THEN** 所有测试通过
- **AND** 零回归

