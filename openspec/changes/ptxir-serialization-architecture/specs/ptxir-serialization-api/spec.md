## ADDED Requirements

### Requirement: serialize_statements 函数
`test_helpers.hpp` SHALL 提供 `serialize_statements(const std::vector<StatementContext>& stmts, const std::string& path)` 函数，将 StatementContext 序列化为 `.ptxir` 文件。

#### Scenario: 序列化成功返回 true
- **WHEN** 调用 `serialize_statements(stmts, "/path/to/file.ptxir")` 且路径可写
- **THEN** 返回 `true`，文件被创建并包含完整的语句序列

#### Scenario: 文件无法创建时返回 false
- **WHEN** 调用 `serialize_statements(stmts, "/nonexistent/path/file.ptxir")`
- **THEN** 返回 `false`（不抛出异常）

### Requirement: deserialize_statements 函数
`test_helpers.hpp` SHALL 提供 `deserialize_statements(const std::string& path)` 函数，从 `.ptxir` 文件反序列化为 `std::vector<StatementContext>`。

#### Scenario: 成功反序列化返回语句向量
- **WHEN** 调用 `deserialize_statements("tests/ptxir/kernel.ptxir")`
- **THEN** 返回包含完整 StatementContext 的向量，`size() > 0`

#### Scenario: 文件不存在或格式错误抛出异常
- **WHEN** 调用 `deserialize_statements("/nonexistent/file.ptxir")`
- **THEN** 抛出 `std::runtime_error("PTXIR file not found")`

### Requirement: generate_ptxir 工具函数
`test_helpers.hpp` SHALL 提供 `generate_ptxir(const std::string& ptx_path, const std::string& ptxir_path, const std::string& kernel_name)`，将 PTX 文本文件通过 ANTLR 解析后序列化为 `.ptxir`。

#### Scenario: PTX 文本转换为 .ptxir
- **WHEN** 调用 `generate_ptxir("tests/ptx/kernel.ptx", "tests/ptxir/kernel.ptxir", "")`
- **THEN** 内部执行 ANTLR 解析，生成 `.ptxir` 文件，并返回 `true`

#### Scenario: 解析失败时返回 false
- **WHEN** PTX 文件有语法错误
- **THEN** 抛出 `std::runtime_error` 描述解析错误，不生成文件

### Requirement: load_ptxir 函数
`test_helpers.hpp` SHALL 提供 `load_ptxir(const std::string& ptxir_path, bool apply_cfg)` 函数，从 `.ptxir` 反序列化并可选应用 CFGBuilder。

#### Scenario: 基本加载（无 CFG）
- **WHEN** 调用 `load_ptxir("tests/ptxir/kernel.ptxir", false)`
- **THEN** 返回反序列化的 StatementContext，`reconvergence_pc == -1`

#### Scenario: 加载后应用 CFG
- **WHEN** 调用 `load_ptxir("tests/ptxir/kernel.ptxir", true)`
- **THEN** 反序列化后应用 CFGBuilder，所有分支指令的 `reconvergence_pc >= 0`

### Requirement: 向后兼容性——load_ptx_file 保持不变
现有 `load_ptx_file(const std::string& path)` 函数 SHAL NOT be modified，保持返回 `std::string` PTX 文本的行为。

#### Scenario: 现有 Mode 2 测试不受影响
- **WHEN** `load_ptx_file("tests/ptx/kernel.ptx")` 被调用
- **THEN** 返回原始 PTX 文本字符串，无任何变化
