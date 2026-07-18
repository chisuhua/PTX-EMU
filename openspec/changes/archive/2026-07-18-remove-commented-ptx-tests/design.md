# 清理被注释的 PTX 单元测试目标

## Context

tests/unit/CMakeLists.txt 第 535-580 行有 7 个被注释的 add_catch_test 块，注释头标注 "Disabled per Oracle A2 recommendation"。integration 等价覆盖已在 tests/integration/ptx/ 的 24 个测试文件中完整实现。

## Goals / Non-Goals

- **Goals**: 纯删除注释块，清理 CMakeLists.txt 死代码
- **Non-Goals**: 不修改任何 .cu/.cpp 文件，不改变测试行为

## Decisions

1. **仅删除注释块**: 不修改被引用的 .cu 源文件
2. **验证 integration 覆盖**: 运行 ctest -L "integration;ptx" 确认全部通过后提交

## Risk

- 极低：注释块本就不参与编译和测试执行
