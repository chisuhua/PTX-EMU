## Why

Debt Audit 2026-07-02 §P0-C3 发现 `tests/unit/CMakeLists.txt:432-472` 有 7 个 PTX 单元测试被注释掉（注释说"移至 reference/"但从未恢复），导致 7 类 PTX 指令无回归保障。这些测试覆盖了模拟器中最基础、最常用的 PTX 算术/位运算/类型转换/访存指令。

## What Changes

- 恢复 7 个被注释的 PTX 单元测试的 CMake 注册：`unit_ptx_integer`、`unit_ptx_float`、`unit_ptx_extended`、`unit_ptx_bitwise`、`unit_ptx_cvt`、`unit_ptx_ld_st`、`unit_ptx_cvta`
- 更新测试代码以匹配当前 API（测试可能引用已重构/删除的函数）
- 确认全部 7 个测试编译通过并绿色执行
- 在 `ctest` 中恢复对应标签

## Capabilities

### New Capabilities
- `ptx-regression-restore`: 恢复被注释的 PTX 指令单元测试，为 7 类基础指令提供回归检测

### Modified Capabilities
（无修改既有 capability）

## Impact

- **测试**: 7 个 PTX 单元测试重新激活，回归保障恢复
- **CMake**: `tests/unit/CMakeLists.txt` 中对应 `add_catch_test` 行取消注释
- **实现代码**: 可能需小幅调整测试代码以适配当前 API（不修改生产代码）
- **相关 ADR**: ADR-0013 (statement-factory-test-unification)，测试框架兼容性