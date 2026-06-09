## ADDED Requirements

### Requirement: 离线生成工具
系统 SHALL 提供一个独立工具 `ptxir_generator`（或 Python 脚本 `scripts/generate_ptxir.py`），输入 PTX 文本文件，输出 `.ptxir` 二进制文件。

#### Scenario: 基本生成
- **WHEN** 运行 `ptxir_generator tests/ptx/kernel.ptx -o tests/ptxir/kernel.ptxir`
- **THEN** 生成 `.ptxir` 文件，包含完整 kernelStatements

#### Scenario: 指定 kernel 名称
- **WHEN** PTX 文件包含多个 `.entry` kernel
- **THEN** `--kernel <name>` 参数选择特定 kernel，默认取第一个 `.entry`

### Requirement: 批量生成模式
工具 SHALL 支持批量模式，输入目录，输出 `.ptxir` 到对应目录。

#### Scenario: 目录批量转换
- **WHEN** 运行 `ptxir_generator --batch tests/ptx/ -o tests/ptxir/`
- **THEN** `tests/ptx/` 下的每个 `.ptx` 文件转换为同名 `.ptxir`

### Requirement: 生成与 generate_tests.py 集成
`docs/skills/three-mode-testing/generate_tests.py` SHALL 增加 `--ptxir` 选项，自动调用 `.ptxir` 生成。

#### Scenario: 生成 Mode 4 测试时同时生成 .ptxir
- **WHEN** 运行 `generate_tests.py --benchmark kernel --mode mode4`
- **THEN** 自动调用 `generate_ptxir()` 生成 `tests/ptxir/kernel.ptxir`，并生成 `test_kernel_mode4.cpp`
