# ptx-visitor-dispatch-extraction Specification

## Purpose
TBD - created by archiving change split-ptx-visitor-god-class. Update Purpose after archive.
## Requirements
### Requirement: 提取范围

The system MUST extract the dispatch layer to `ptx_visitor_dispatch.cpp`.

#### Scenario: include 聚合区在新文件

- **WHEN** 提取后运行 `grep -l 'ptx_visitor_generic\|ptx_visitor_atom\|ptx_visitor_call\|ptx_visitor_branch\|ptx_visitor_barrier\|ptx_visitor_simple\|ptx_visitor_special\|ptx_visitor_warp\|ptx_visitor_memory\|ptx_visitor_abi' src/ptx_parser/ptx_visitor.cpp`
- **THEN** 命中 0（已移出主文件）
- **AND** `grep -c` 在 `ptx_visitor_dispatch.cpp` 命中 ≥ 10

#### Scenario: X-Macro dispatch 在新文件

- **WHEN** 提取后运行 `grep -c 'ptx_op.def\|VISITOR_' src/ptx_parser/ptx_visitor.cpp`
- **THEN** 命中 0
- **AND** 在 `ptx_visitor_dispatch.cpp` 命中 ≥ 2

### Requirement: 依赖关系

The system MUST maintain the correct ordering.

#### Scenario: dispatch 引用所有类别子文件

- **GIVEN** 现有 10 个 `ptx_visitor_<category>.cpp` 文件
- **WHEN** 提取后
- **THEN** `ptx_visitor_dispatch.cpp` 包含所有 10 个文件
- **AND** 现有 10 个文件无需修改

#### Scenario: X-Macro 在 visitor 方法定义后展开

- **GIVEN** X-Macro 依赖 VISITOR_<struct_kind> 宏定义
- **WHEN** 提取后展开
- **THEN** 所有宏在使用前已定义
- **AND** 链接期所有 weak symbol 正确分发

### Requirement: 行为不变性

The system MUST maintain byte-level identical IR output.

#### Scenario: 任何 PTX 指令分派结果不变

- **WHEN** 运行全 PTX 语法测试 + ctest
- **THEN** 所有指令分派结果与提取前一致
- **AND** 零回归

### Requirement: 构建系统

The system MUST update `src/ptx_parser/CMakeLists.txt`.

#### Scenario: 新源文件被编译

- **WHEN** 运行 `cmake --build build`
- **THEN** `ptx_visitor_dispatch.cpp` 被编译
- **AND** 链接期无未定义符号

### Requirement: 验证

The system MUST pass all of:
- `./tests/ptx/test_all_ptx.sh` — PTX 语法测试全绿
- 所有 X-Macro 分发的指令测试
- 链接期无未定义符号

#### Scenario: 全量测试通过

- **WHEN** 运行 `./tests/ptx/test_all_ptx.sh && ctest --output-on-failure`
- **THEN** 所有测试通过
- **AND** 链接成功无 warning

