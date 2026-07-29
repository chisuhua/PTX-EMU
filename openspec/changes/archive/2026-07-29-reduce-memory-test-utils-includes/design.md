# reduce-memory-test-utils-includes - Design

## Overview

`include/ptxsim/testing/memory_test_utils.h` 是测试工具头文件，为内存相关测试提供辅助函数。当前包含 18 个 `#include`，被多个测试文件包含，include 膨胀导致测试编译依赖传播。

本 change 通过前向声明替代和 include 移动策略，将 include 数量从 18 减至 ≤ 12。

## Design Decisions

### 决策 1: Catch2 头文件处理

**选择**: 保留 `catch_amalgamated.hpp`

**理由**:
- 测试工具头文件中大量使用 Catch2 宏（`REQUIRE`, `CHECK`, `SECTION` 等）
- 宏必须在使用前包含，无法前向声明
- 这是测试工具头文件的固有依赖

### 决策 2: 项目头文件处理策略

**选择**: 逐个分析 8 个项目头文件的使用方式

**候选前向声明**（头文件函数签名中仅以指针/引用出现的类型）：
- `ptxsim/cta_context.h` -> `CTAContext` 如果仅以 `CTAContext*` / `CTAContext&` 参数 -> 前向声明
- `ptxsim/warp_context.h` -> `WarpContext` 同上分析
- `ptxsim/sm_context.h` -> `SMContext` 同上分析
- `ptxsim/instruction_factory.h` -> `InstructionFactory` 同上分析

**可能移到 .cpp**（仅在函数实现内部使用的完整类型）：
- 如果某些类型仅在函数体内使用（非签名参数），可移到 `.cpp`
- 如 `memory/resource_manager.h` -> `ResourceManager` 如仅内部使用

**必须保留**（值类型参数/成员或 inline 使用）：
- `ptx_ir/operand_context.h` -> 如 `OperandContext` 为值参数
- `ptx_ir/ptx_types.h` -> 枚举/类型定义
- `ptx_ir/statement_context.h` -> 如 `StatementContext` 为值参数
- `register/register_bank_manager.h` -> 如为值类型

### 决策 3: 标准库 include 处理

**选择**: 分析每个标准库 include 的使用方式，移除不必要的

**分析**：
- `<algorithm>` - `std::sort`/`std::find` 等 -> 如仅 .cpp 使用则移到 .cpp
- `<cstdint>` - `uint32_t` 等固定宽度整型 -> 保留（可能被值参数使用）
- `<cstdlib>` - `std::malloc`/`std::free` 等 -> 如仅 .cpp 使用则移到 .cpp
- `<map>` - `std::map` 值类型 -> 保留（如为值成员/参数）
- `<memory>` - `std::shared_ptr`/`std::unique_ptr` -> 保留（smart pointer 需完整定义在销毁时）
- `<string>` - `std::string` 值类型 -> 保留
- `<vector>` - `std::vector` 值类型 -> 保留

### 决策 4: inline 函数约束

**选择**: 注意 inline 函数的 include 需求

**约束**:
- 如果 `memory_test_utils.h` 中的函数是 `inline` 的，且函数体内使用某类型（即使是 `WarpContext` 指针方法调用），则**需要完整类型定义**，不能仅前向声明
- 仅**函数声明**中的指针/引用参数可前向声明
- 如果函数定义在头文件中（inline），则所有函数体内使用的类型都需要 include

**策略**:
1. 将可前向声明类型的函数定义移到 `.cpp`（去掉 inline）
2. 头文件仅保留函数声明 + 前向声明
3. `.cpp` 包含所有需要的完整类型 include

**替代策略**（如函数必须 inline）:
- 保留函数所需的 include
- 仅移除确实未使用的 include

### 决策 5: 逐步验证策略

**选择**: 每移除 2-3 个 include 后编译验证

**理由**:
- 与 thread_context.h change 相同的策略
- 前向声明错误可通过编译错误快速定位
- 测试工具头文件的 inline 特性需要特别注意

## Implementation Plan

### Phase 1: 基线记录
1. 记录当前 include 数量（18）
2. 编译基线（确保当前可编译）
3. 运行测试基线（`ctest -L unit` 全绿）

### Phase 2: 分析函数签名和 inline 状态
1. 列出 `memory_test_utils.h` 中所有函数
2. 标注每个函数是否 inline
3. 标注每个函数签名中使用的类型
4. 标注 inline 函数体内使用的类型
5. 确定哪些 include 可移除/前向声明/移到 .cpp

### Phase 3: 标准库 include 精简
1. 移除 `<algorithm>` 如仅 .cpp 使用
2. 移除 `<cstdlib>` 如仅 .cpp 使用
3. 每步编译验证
4. 目标：减少 2-4 个标准库 include

### Phase 4: 项目头文件前向声明/移动
1. 对可前向声明的类型添加前向声明
2. 将 inline 函数定义移到 .cpp（如适用）
3. 移除对应的项目头文件 include
4. 每步编译验证
5. 目标：减少 3-5 个项目 include

### Phase 5: 最终验证
1. 确认 include 数量 ≤ 12
2. 全量测试编译通过
3. `ctest -L unit` 全绿
4. 无函数签名变更

## Testing Strategy

### 验证维度

| 测试类型 | 命令 | 预期 |
|---------|------|------|
| 全量编译 | `cmake --build build` | 通过 |
| 单元测试 | `cd build && ctest -L unit --output-on-failure` | 全绿 |
| 编译警告 | `cmake --build build 2>&1 \| grep warning` | 无新增 |
| include 数量 | `grep -c '#include' include/ptxsim/testing/memory_test_utils.h` | ≤ 12 |
| 函数签名不变 | diff 函数声明 | 无变化 |

### 前向声明正确性验证

```bash
# 修改后编译，如出现 "incomplete type" 错误说明误判
cmake --build build 2>&1 | grep "incomplete type"
# 应无输出
```

### 测试文件编译验证

```bash
# 确认所有使用 memory_test_utils.h 的测试文件仍可编译
cmake --build build 2>&1 | grep "error"
# 应无输出
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| inline 函数移到 .cpp 导致 ODR 问题 | 链接错误 | 确保函数声明在头文件，定义在 .cpp，去掉 inline |
| 误判可前向声明类型 | 编译错误 | 逐步验证 |
| 隐式包含被移除 | 依赖测试文件编译失败 | 全量编译验证 |
| Catch2 宏依赖完整类型 | inline 函数中的宏调用需要完整 include | 保留必要的 include |

## Open Questions

1. **函数是否可以非 inline？**
   - 取决于是否有性能需求（测试工具函数通常无此需求）
   - 决定：如移到 .cpp 可解决 include 问题，则去 inline

2. **是否需要新建 .cpp 文件？**
   - 如当前 `memory_test_utils.h` 仅有头文件（无 .cpp），需新建 `src/ptxsim/testing/memory_test_utils.cpp`
   - 需更新 CMakeLists.txt 添加新源文件
   - 决定：分析后确定，如需新建则纳入范围

## 关联文档

- `improvements/reduce-memory-test-utils-includes.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-8`：原债务条目
- `include/ptxsim/testing/memory_test_utils.h`：目标文件
