# reduce-memory-test-utils-includes - Proposal

## Why

`include/ptxsim/testing/memory_test_utils.h` 头部有 **18 个 `#include`** 指令。该头文件被多个测试文件包含，include 膨胀直接影响测试编译时间。

当前 18 个 include 分类：

| 分类 | 数量 | 示例 |
|------|------|------|
| 第三方 | 1 | `catch_amalgamated.hpp` |
| 项目头文件 | 8 | `ptxsim/cta_context.h`, `ptxsim/warp_context.h`, `ptxsim/sm_context.h`, `ptxsim/instruction_factory.h`, `ptx_ir/operand_context.h`, `ptx_ir/statement_context.h`, `memory/resource_manager.h`, `register/register_bank_manager.h` |
| 标准库 | 9 | `<algorithm>`, `<cstdint>`, `<cstdlib>`, `<map>`, `<memory>`, `<string>`, `<vector>` |

核心问题：
- 测试工具头文件的 include 通过传递性导致所有包含它的测试文件都依赖这 18 个头文件
- 任何被包含头文件的变更（如 `warp_context.h` 修改）触发所有使用 `memory_test_utils.h` 的测试重编译
- 部分 include 可能通过前向声明或移到 `.cpp`/inline 实现区来消除

来源：`docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-8`

## What Changes

- **分析** 18 个 include 的必要性（值类型使用 vs 指针/引用 vs 仅实现内部）
- **替换** 可前向声明的类型为前向声明
- **移动** 实现特有的 include 到对应的 `.cpp` 文件或 inline 实现区
- **保持** `memory_test_utils.h` 的所有函数签名不变

## Capabilities

### New Capabilities
- `memory-test-utils-include-optimization`: 通过前向声明和 include 移动减少测试编译依赖

### Modified Capabilities
（无函数签名或行为变更。纯编译依赖优化。）

## Impact

**受影响文件**：
- `include/ptxsim/testing/memory_test_utils.h`（18 -> ≤ 12 个 include）
- 对应的 `.cpp` 文件或 inline 实现区（可能新增从头文件移入的 include）

**不受影响**：
- `memory_test_utils.h` 的任何函数签名
- 使用该头文件的测试文件（编译通过，行为不变）
- 运行时行为

**依赖**：
- 无前置 change 依赖，可独立执行
- 纯重构，无功能变更

**工时**: 1-1.5h（include 分析 + 逐步精简 + 测试编译验证）

## Design-Time Checklist

- [ ] 确认每个被移除的 include 对应的类型可前向声明或可移到实现区
- [ ] 确认 inline 函数完整性（如移到 .cpp 则不能是 inline）
- [ ] 确认所有测试编译通过
- [ ] 确认无函数签名变更
