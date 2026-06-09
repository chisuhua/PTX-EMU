## ADDED Requirements

### Requirement: Mode 4 测试模式
三模式测试框架 SHALL 扩展为四模式测试框架，新增 Mode 4：从 `.ptxir` 快速加载并执行。

#### Scenario: Mode 4 基本执行
- **WHEN** `test_ptxir_mode4.cpp` 加载 `tests/ptxir/kernel.ptxir`
- **THEN** 反序列化后通过 `run_statement_sequence()` 执行，验证执行结果与 Mode 2 一致

### Requirement: Mode 4 构建目标自动检测
`tests/three_mode_testing/CMakeLists.txt` SHALL 自动检测 `*_mode4.cpp` 文件并创建构建目标。

#### Scenario: 新增 mode4 文件自动构建
- **WHEN** 添加 `test_foo_mode4.cpp` 到 `tests/three_mode_testing/`
- **THEN** CMake 重新配置时自动检测并创建 `test_foo_mode4` 可执行文件

### Requirement: Mode 4 与 Mode 2 结果一致性
Mode 4（.ptxir 加载）和 Mode 2（.ptx 解析后加载）执行同一 kernel 的结果 SHALL 一致。

#### Scenario: 回归验证
- **WHEN** Mode 2 测试通过
- **THEN** Mode 4 测试必须通过（反之亦然），确保 `.ptxir` 不丢失语义信息

### Requirement: Mode 4 标签
所有 Mode 4 测试用例 SHALL 标记标签 `[mode4]`，可通过 `ctest -L "mode4"` 单独运行。

#### Scenario: 仅运行 Mode 4 测试
- **WHEN** 运行 `ctest -L "mode4" -V`
- **THEN** 仅执行所有 Mode 4 测试

### Requirement: Mode 4 调试工作流
文档 SHALL 定义 Mode 4 在调试工作流中的位置：

```
Mode 1 (发现问题) → Mode 2 (稳定复现) → Mode 3a/3b (精确定位)
→ Mode 4 (快速回归验证) → Mode 1 (端到端)
```

#### Scenario: Mode 4 用于回归测试
- **WHEN** 修复 bug 后验证
- **THEN** 运行 Mode 4 测试快速确认修复有效，无需重新走 Mode 1 的 cuobjdump 提取流程
