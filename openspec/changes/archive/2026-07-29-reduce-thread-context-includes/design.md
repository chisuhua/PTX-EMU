# reduce-thread-context-includes - Design

## Overview

`include/ptxsim/thread_context.h` 是 ThreadContext 类的定义文件，位于执行层次最底层。当前包含 25 个 `#include` 指令，导致编译依赖传播严重。

本 change 通过前向声明替代和 include 移动策略，将 include 数量从 25 减至 ≤ 15，减少编译依赖膨胀。

## Design Decisions

### 决策 1: include 分类策略

**选择**: 按 C++ 使用规则对每个 include 分类：

| 使用方式 | 处理策略 | 可否前向声明 |
|---------|---------|-------------|
| 值类型成员（`T member;`） | **保留** include | NO（需完整类型定义） |
| 指针/引用成员（`T* ptr; T& ref;`） | 前向声明替代 | YES |
| 函数参数（`void f(T* p)` / `void f(T& r)`） | 前向声明替代 | YES |
| 函数返回值（`T f()`） | 前向声明替代 | YES（仅声明，定义处需 include） |
| 仅 .cpp 内部使用 | 移到 .cpp | N/A |

**理由**: C++ 前向声明规则明确——指针/引用参数仅需不完整类型声明，值类型需要完整类型定义。

### 决策 2: 标准库 include 处理

**选择**: 保留实际使用的标准库 include，移除未使用的

**分析**（基于头文件实际使用）：
- `<memory>` — `std::shared_ptr`/`std::unique_ptr` 成员 -> 可前向声明指向的类型，但保留 `<memory>`
- `<vector>` — `std::vector<T>` 值类型成员 -> 保留
- `<string>` — `std::string` 值类型成员 -> 保留
- `<map>` / `<unordered_map>` — `std::map`/`std::unordered_map` 值类型成员 -> 保留
- `<stack>` — `std::stack` 值类型成员 -> 保留
- `<any>` — `std::any` 值类型成员 -> 保留
- `<array>` — `std::array` 值类型成员 -> 保留
- `<iostream>` — 可能仅在调试输出使用 -> **移到 .cpp**（如仅 .cpp 使用 `std::cout`）
- `<cstdint>` — `uint32_t` 等固定宽度整型 -> 保留

### 决策 3: 项目头文件处理

**选择**: 逐个分析项目头文件的使用方式

**候选前向声明**（头文件中仅以指针/引用出现的类型）：
- `ptx_ir/statement_context.h` -> 如果 `StatementContext` 仅以 `const StatementContext&` 参数出现 -> 前向声明
- `ptxsim/contexts/exec_state.h` -> 如果 `ExecState` 仅以指针/引用成员出现 -> 前向声明
- `ptxsim/contexts/memory_ref.h` -> 同上分析
- `ptxsim/contexts/program_ref.h` -> 同上分析
- `ptxsim/contexts/register_predicate.h` -> 同上分析
- `ptxsim/simt_pc_manager.h` -> 如果 `SimtPcManager` 以 `std::unique_ptr<SimtPcManager>` 成员 -> 前向声明 + 保留 `<memory>`

**必须保留**（值类型成员）：
- `ptx_ir/operand_context.h` — 如 `OperandContext` 为值成员
- `ptx_ir/ptx_types.h` — 枚举/类型定义
- `ptxsim/common_types.h` — 通用类型定义
- `register/register_bank_manager.h` — 如为值成员或需要完整类型
- `utils/logger.h` — 如为值成员

**可能移到 .cpp**（仅实现内部使用）：
- 某些仅在 `.cpp` 文件的方法体中使用的类型

### 决策 4: 前向声明集中区域

**选择**: 在头文件 `#include` 块之后、类定义之前集中声明前向类型

```cpp
// --- Forward declarations ---
namespace ptx_ir {
class StatementContext;
class OperandContext;
}  // namespace ptx_ir

namespace ptxsim {
class ExecState;
// ...
}  // namespace ptxsim
```

**理由**:
- 集中管理前向声明，易于审查
- 明确标注 namespace，避免歧义
- 标准实践

### 决策 5: 逐步验证策略

**选择**: 每移除 2-3 个 include 后编译验证，而非批量修改

**理由**:
- 前向声明错误（值类型误判）会产生编译错误，逐步验证可快速定位
- 避免一次性修改大量 include 导致错误堆积难以排查

## Implementation Plan

### Phase 1: 基线记录
1. 记录当前 include 数量（25）
2. 编译基线（确保当前可编译）
3. 记录编译时间基线（可选）

### Phase 2: 标准库 include 精简
1. 确认 `<iostream>` 是否仅在 .cpp 使用 -> 移到 .cpp
2. 确认其他标准库 include 的使用方式
3. 每步编译验证
4. 目标：减少 1-3 个标准库 include

### Phase 3: 项目头文件前向声明替代
1. 分析每个项目头文件类型的使用方式（值 vs 指针/引用）
2. 对指针/引用类型添加前向声明，移除 include
3. 每移除 2-3 个编译验证
4. 目标：减少 5-8 个项目 include

### Phase 4: .cpp include 补充
1. 将从头文件移除的 include 添加到 `.cpp`（如 .cpp 实现需要）
2. 编译验证
3. 全量测试验证

### Phase 5: 最终验证
1. 确认 include 数量 ≤ 15
2. 全量编译通过
3. 无新增 warning
4. `ctest` 全绿

## Testing Strategy

### 验证维度

| 测试类型 | 命令 | 预期 |
|---------|------|------|
| 全量编译 | `cmake --build build` | 通过 |
| 全量测试 | `cd build && ctest` | 全绿 |
| 编译警告 | `cmake --build build 2>&1 \| grep warning` | 无新增 |
| include 数量 | `grep -c '#include' include/ptxsim/thread_context.h` | ≤ 15 |

### 前向声明正确性验证

```bash
# 修改后编译，如出现 "incomplete type" 错误说明误判值类型
cmake --build build 2>&1 | grep "incomplete type"
# 应无输出
```

### 依赖传播减少验证（可选）

```bash
# 触摸一个被前向声明替代的头文件，确认 thread_context.h 依赖者不重编译
touch include/ptxsim/contexts/exec_state.h
cmake --build build --target ptxsim 2>&1 | grep "thread_context"
# 如 thread_context.o 重编译但依赖者不重编译，说明前向声明生效
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 误判值类型为可前向声明 | 编译错误 "incomplete type" | 逐步验证，每步编译检查 |
| .cpp 缺少移入的 include | 编译错误 | Phase 4 补充 .cpp include |
| 前向声明 namespace 错误 | 编译错误或行为偏移 | 集中声明区域，明确 namespace |
| 隐式包含被移除 | 某些依赖 thread_context.h 的文件依赖传递性 include | 全量编译验证；如发现需补充直接 include |

## Open Questions

1. **`utils/logger.h` 是否可移到 .cpp？**
   - 取决于头文件是否有 `Logger` 类型成员或 inline 使用
   - 决定：分析后确定，如为 `std::unique_ptr<Logger>` 则可前向声明

2. **`register/register_bank_manager.h` 是否可前向声明？**
   - 取决于 `RegisterBankManager` 是值成员还是指针成员
   - 决定：分析后确定

## 关联文档

- `improvements/reduce-thread-context-includes.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-7`：原债务条目
- `include/ptxsim/thread_context.h`：目标文件
- `src/ptxsim/core/thread_context.cpp`：对应实现文件
