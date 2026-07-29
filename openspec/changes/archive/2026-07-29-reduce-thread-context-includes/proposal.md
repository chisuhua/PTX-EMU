# reduce-thread-context-includes - Proposal

## Why

`include/ptxsim/thread_context.h` 头部有 **25 个 `#include`** 指令，远超合理上限。过多 include 导致编译依赖膨胀：任何被包含头文件的变更都会触发所有依赖 `thread_context.h` 的文件重新编译。

当前 25 个 include 分类：

| 分类 | 数量 | 示例 |
|------|------|------|
| 项目头文件 | 14 | `ptx_ir/operand_context.h`, `ptxsim/contexts/exec_state.h`, `register/register_bank_manager.h` 等 |
| 标准库 | 11 | `<any>`, `<array>`, `<iostream>`, `<map>`, `<memory>`, `<stack>`, `<string>`, `<unordered_map>`, `<vector>` 等 |

核心问题：
- `ThreadContext` 是执行层次最底层的上下文类，被大量源文件包含，include 膨胀的编译时间影响放大
- 部分 include 可能仅需前向声明（当类型仅以指针/引用参数出现，非值类型成员）
- 实现特有的 include 应移到 `.cpp` 文件，减少头文件依赖传播

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-7`

## What Changes

- **分析** 25 个 include 的使用方式（值类型成员 vs 指针/引用参数 vs 实现内部使用）
- **替换** 仅需指针/引用的类型为前向声明
- **移动** 实现特有的 include 到对应的 `.cpp` 文件
- **保持** ThreadContext 的 public API 不变

## Capabilities

### New Capabilities
- `thread-context-include-optimization`: 通过前向声明和 include 移动减少编译依赖

### Modified Capabilities
（无 API 行为变更。纯编译依赖优化，不影响运行时行为。）

## Impact

**受影响文件**：
- `include/ptxsim/thread_context.h`（25 -> ≤ 15 个 include）
- `src/ptxsim/core/thread_context.cpp`（可能新增 include 从头文件移入的）

**不受影响**：
- ThreadContext 的 public API（签名不变）
- 所有使用 `thread_context.h` 的源文件（编译通过，无行为变化）
- 运行时行为

**依赖**：
- 无前置 change 依赖，可独立执行
- 纯重构，无功能变更

**工时**: 1-1.5h（include 分析 + 逐步精简 + 编译验证）

## Design-Time Checklist

- [ ] 确认每个被移除的 include 对应的类型确实可前向声明或仅在 .cpp 使用
- [ ] 确认值类型成员的 include 保留（不可前向声明）
- [ ] 确认编译通过且无新增 warning
- [ ] 确认 include 分组保持（标准库 / 项目 / 第三方）
