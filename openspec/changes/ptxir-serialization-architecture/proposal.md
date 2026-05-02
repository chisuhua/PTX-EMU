## Why

PTX-EMU 的测试管道目前每次都通过 ANTLR 重新解析 PTX 文本（Mode 1/2），或者完全跳过解析直接手写 StatementContext（Mode 3）。这两种方案都有问题：重复解析带来 ~200ms 级别的开销（大型 kernel 上千条指令），而手写 StatementContext 则无法复用真实编译流程的产物。需要引入第四种模式：预先序列化好的 `.ptxir` 二进制文件，快速反序列化后直接执行。

本质问题：**PTX 作为中间表示层，已经完成了语法解析和语义提取，却没有持久化**。这与 LLVM-IR 的设计哲学相悖——LLVM 在 clang 一次编译后可以将 `.bc` 文件缓存起来，后续的 LLVM pass、链接、执行都可以直接加载。

## What Changes

1. **新增 `.ptxir` 二进制格式**：一种扁平二进制格式，包含已解析的 PTX 指令序列、寄存器声明、符号表和类型信息，文件头部有 TOC（Table of Contents）索引，支持随机访问。

2. **新增 Mode 4（快速加载）**：在现有 Mode 1/2/3/3C 基础上增加 Mode 4——直接反序列化 `.ptxir` 文件到 `std::vector<StatementContext>`，绕过 ANTLR 解析，开销从 ~200ms 降至 ~5ms。

3. **新增序列化工具函数**（`test_helpers.hpp`）：
   - `serialize_statements(stmts, path)`：将 StatementContext 序列化为 `.ptxir`
   - `deserialize_statements(path)`：从 `.ptxir` 反序列化
   - `generate_ptxir(ptx_path, ptxir_path, kernel_name)`：PTX 文本 → ANTLR 解析 → 序列化（离线工具）
   - `load_ptxir(ptxir_path, apply_cfg)`：加载 `.ptxir` 并可选应用 CFGBuilder

4. **更新三模式测试框架文档**为四模式，增加 Mode 4 的使用说明和调试工作流。

5. **保留全部现有模式**（Mode 1/2/3/3C）——这是向后兼容的重构，不破坏任何现有功能。

## Capabilities

### New Capabilities

- **ptxir-binary-format**：定义 `.ptxir` 二进制格式规范（头部结构、Section TOC、指令编码、版本管理）
- **ptxir-serialization-api**：在 `test_helpers.hpp` 中提供 `serialize_statements()` / `deserialize_statements()` API
- **ptxir-generator-tool**：离线工具：将 PTX 文本通过 ANTLR 解析后序列化为 `.ptxir` 文件
- **ptxir-test-mode**：Mode 4 测试模式——从 `.ptxir` 快速加载并执行，用于单元测试和回归测试

### Modified Capabilities

（无——现有测试模式全部保留，API 仅为增量添加）

## Impact

**核心影响范围**：

- `tests/three_mode_testing/`：新增 Mode 4 测试文件，CMakeLists.txt 增加 `ptxir` 相关构建目标
- `tests/three_mode_testing/test_helpers.hpp`：新增 4 个序列化函数，现有 `load_ptx_file()` 保持不变
- `include/ptx_ir/`：`StatementContext` 及其所有 `InstrVariant` 类型需要可序列化（无手释指针，std::variant 友好）
- `docs/developer-guide/THREE-MODE-TESTING-GUIDE.md`：升级为四模式测试框架文档
- `docs/developer-guide/PTX-TO-STATEMENTS-IMPLEMENTATION.md`：（已存在但未实现）可与 `.ptxir` 合并

**新增文件**：

- `include/ptx_ir/ptxir_format.h` —— `.ptxir` 格式定义（头部、Section 枚举、版本常量）
- `src/ptx_ir/ptxir_writer.cpp` —— 序列化实现
- `src/ptx_ir/ptxir_reader.cpp` —— 反序列化实现
- `tests/three_mode_testing/test_ptxir_mode4.cpp` —— Mode 4 测试模板
- `tests/ptxir/` —— 预生成的 `.ptxir` 文件（git 追踪版本化 PTX 内容）
- `docs/skills/ptxir-serialization/` —— 技能文档

**不影响**：

- `src/cudart/` 的运行时路径（`__cudaRegisterFatBinary` 仍走 ANTLR）
- `src/ptx_parser/` 的解析逻辑（ANTLR 本身不变）
- `src/ptxsim/` 的指令执行逻辑（StatementContext 格式不变）
