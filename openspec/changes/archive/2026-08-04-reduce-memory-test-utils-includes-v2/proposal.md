# reduce-memory-test-utils-includes-v2

## Why

`include/ptxsim/testing/memory_test_utils.h` 当前 18 个 include（与原基线一致，未变）。原 change `2026-07-29-reduce-memory-test-utils-includes` archive 后 tasks.md 全勾选但代码未 apply。本 change 为 1h quick win 重做，目标 ≤12 include。

## What Changes

- 分析 18 个 include 的必要性（值类型 vs 指针/引用 vs inline 函数体）
- 对仅 .cpp 使用的标准库 include 移到 .cpp 实现
- 对 inline 函数体不依赖的项目头文件改为前向声明
- 保持所有函数签名 + 行为不变

## Capabilities

### New Capabilities
- (无新 capability；纯测试工具头文件 include 精简)

### Modified Capabilities
- (无 spec-level 变更)

## Impact

- `include/ptxsim/testing/memory_test_utils.h` — include 数量 18 → ≤12（净减 ≥6）
- `tests/memory/test_utils.cpp` 或类似 .cpp — 增加移入的 include
- 所有使用该头文件的测试文件 — 编译时间缩短

依赖 Skill: ptx-instruction-pipeline。