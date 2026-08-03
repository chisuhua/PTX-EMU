# reduce-thread-context-includes-v2

## Why

`include/ptxsim/thread_context.h` 当前 25 个 include（含 `<iostream>` 等可前向声明类型），基线 21 个反而恶化。原 change `2026-07-29-reduce-thread-context-includes` archive 后 tasks.md 全勾选但代码未 apply。本 change 重做，目标 ≤15 include。

## What Changes

- 前向声明所有仅以指针/引用形式出现的项目类型（不展开完整定义）
- 移除实现特有的标准库 include（移至 .cpp）
- 集中 forward declaration 区于头文件顶部
- 保持 ThreadContext public API 不变
- 保持所有调用点编译通过

## Capabilities

### New Capabilities
- (无新 capability；纯实现级 include 精简)

### Modified Capabilities
- (无 spec-level 变更)

## Impact

- `include/ptxsim/thread_context.h` — include 数量 25 → ≤15（净减 ≥10）
- `src/ptxsim/core/thread_context.cpp` — 增加移入的 include
- 所有使用 `thread_context.h` 的文件 — 编译时间缩短
- 测试覆盖 `tests/unit/core/` — 全部通过（无新增 warning）

依赖 Skill: ptx-instruction-pipeline。