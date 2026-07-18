## Why

tests/unit/CMakeLists.txt 第 535-580 行有 7 个被注释的 add_catch_test 块（unit_ptx_integer/_float/_extended/_bitwise/_cvt/_ld_st/_cvta），注释头标注 "Disabled per Oracle A2 recommendation"。对应的 integration 等价覆盖已在 tests/integration/ptx/ 中通过 24 个测试文件实现。这些注释块是 CMakeLists.txt 中的死代码——增加阅读负担，且注释中引用的 .cu 源文件可能已不存在。对应 debt-audit-2026-07-02.md P0-C3。

## What Changes

- **删除 tests/unit/CMakeLists.txt 第 535-580 行**: 7 个被注释的 add_catch_test 块及其注释头
- 不修改任何 .cu/.cpp 源文件
- 不修改任何测试行为

## Capabilities

### New Capabilities
- `cmake-cleanup-commented-tests`: 从 CMakeLists.txt 中清理已被 integration 测试覆盖的死注释测试目标

### Modified Capabilities
<!-- none -->

## Impact

- 文件: tests/unit/CMakeLists.txt（-46 行）
- 代码: 零影响（无 .cu/.cpp 变更）
- 编译: 零影响（注释块本就不参与编译）
- 测试: 零回归风险（integration/ptx/ 的 24 个测试文件已覆盖 7 个域）
- 风险: 极低 — 纯删除
- 相关: debt-audit-2026-07-02.md §P0-C3