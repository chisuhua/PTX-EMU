# replace-assert-false-with-throw - Design

## Overview

将 `src/ptx_ir/statement_context.cpp:19` 的 `assert(false && "Unknown StatementType")`
替换为 `throw PtxEmuException(...)`，确保 Release 构建（`-DNDEBUG`）中遇到未知
`StatementType` 时正确抛出异常而非静默继续。

当前代码：

```cpp
// statement_context.cpp:9-22
std::string S2s(StatementType s) {
    switch (s) {
#define X(stype, opkind, opname, count, struct_kind, instr_kind)  \
    case stype:                                                    \
        return #opname;
#include "ptx_ir/ptx_op.def"
#undef X
    case S_UNKNOWN:
        return "unknown";
    default:
        assert(false && "Unknown StatementType");  // <-- 问题行
        return "invalid";
    }
}
```

X-Macro 展开 `ptx_op.def` 的 106 个条目后，switch 覆盖所有已知 `StatementType`。
`default` 分支仅在以下场景触发：
1. 枚举值被非法构造（内存损坏、未初始化变量）
2. `ptx_op.def` 新增条目但 `S2s()` 的 X-Macro 未同步展开（不太可能，同一文件展开）

## Context

- `StatementType` 枚举定义在 `include/ptx_ir/ptx_types.h:19-24`，通过 X-Macro 从
  `ptx_op.def` 自动生成 106 个枚举值 + `S_UNKNOWN`
- `S2s()` 是 `StatementType -> std::string` 转换函数，用于日志/调试输出
- `include/ptxsim/ptx_exceptions.h` 提供完整异常层次：
  - `PtxEmuException`（基类，`PtxEmuErrorCode::INTERNAL_ERROR`）
  - `UnsupportedInstructionException`、`InvalidMemoryAccessException` 等
- 该场景属于"内部错误"（不可达代码路径被触发），使用 `PtxEmuException` 基类即可

## Design Decisions

### 决策 1: 使用 `PtxEmuException` 而非 `UnsupportedInstructionException`

**选择**: `throw PtxEmuException("Unknown StatementType: " + std::to_string(static_cast<int>(s)))`

**理由**:
- `UnsupportedInstructionException` 语义是"PTX 指令未实现"，此处是"枚举值非法"
  --属于内部错误（`INTERNAL_ERROR`），不是指令缺失
- `PtxEmuException` 默认错误码为 `INTERNAL_ERROR`，语义匹配
- 异常消息包含 `StatementType` 的整数值，便于调试定位

**替代方案**:
- A. `throw UnsupportedInstructionException(...)` -> 语义不匹配（非指令未实现）
- B. 新增 `InvalidStatementTypeException` -> improvement 明确要求"不引入新异常类型"
- C. **采用**: `PtxEmuException` 基类 + 数值消息

### 决策 2: 异常消息包含 StatementType 数值

**选择**: `std::to_string(static_cast<int>(s))`

**理由**:
- improvement 技术约束要求"SHOULD 在异常消息中包含 StatementType 的数值"
- 数值比字符串更有用（因为 `S2s()` 本身就是做字符串转换，此处无法用字符串表示）
- 整数值可帮助快速定位是哪个枚举值出了问题

### 决策 3: include 路径

**选择**: 在 `statement_context.cpp` 中新增 `#include "ptxsim/ptx_exceptions.h"`

**理由**:
- 当前 `statement_context.cpp` 未 include `ptx_exceptions.h`
- `ptx_exceptions.h` 位于 `include/ptxsim/`，`src/ptx_ir/` 可直接引用
- include 路径 `ptxsim/ptx_exceptions.h` 与项目其他文件一致

### 决策 4: 保留 `return "invalid"` 还是删除

**选择**: 删除 `return "invalid";`，改为 `throw` 后不需要 return

**理由**:
- `throw` 后编译器知道该路径不会返回，无需 `return`
- 保留 `return` 会导致编译器警告"unreachable code"
- 删除后 `default` 分支仅含 `throw`，语义清晰

## Implementation Plan

### Phase 1: 修改 statement_context.cpp（10 min）
1. 新增 `#include "ptxsim/ptx_exceptions.h"`
2. 替换 `assert(false && "Unknown StatementType"); return "invalid";` 为
   `throw PtxEmuException("Unknown StatementType: " + std::to_string(static_cast<int>(s)));`
3. 删除 `#include <cassert>`（如该文件无其他 assert 使用）

### Phase 2: 验证（20 min）
1. Debug 构建编译通过
2. Release 构建编译通过
3. `ctest` 全绿
4. `grep -rn "assert(false" src/ include/` 返回 0 结果

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| Debug 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build` | 通过 |
| Release 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` | 通过 |
| 全量测试 | `cd build && ctest --output-on-failure` | 全绿 |
| grep 验证 | `grep -rn "assert(false" src/ include/` | 0 结果 |
| statement_context 测试 | `ctest -R "statement_context\|ptxir" --output-on-failure` | 全绿 |

### 异常触发验证（可选）

构造非法 `StatementType` 值并调用 `S2s()`，验证抛出 `PtxEmuException`：
```cpp
// 测试伪码（如有 unit test 框架支持）
try {
    S2s(static_cast<StatementType>(99999));
    FAIL("Should have thrown");
} catch (const PtxEmuException& e) {
    REQUIRE(e.get_error_code() == PtxEmuErrorCode::INTERNAL_ERROR);
    REQUIRE(e.what() contains "Unknown StatementType");
    REQUIRE(e.what() contains "99999");
}
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 异常传播导致调用栈 unwind | 调用方可能未捕获异常 | `S2s()` 主要用于日志/调试，异常传播到顶层被 catch-all 捕获 |
| Release 构建行为变化 | 原静默返回 `"invalid"`，现抛异常 | 这是预期行为改进 |
| 删除 `#include <cassert>` 后其他代码依赖 | 编译错误 | 检查 `statement_context.cpp` 中是否有其他 `assert` 使用 |

## Open Questions

1. **`statement_context.cpp` 中是否有其他 `assert` 使用？**
   - 检查结果：无其他 assert 使用，可安全删除 `#include <cassert>`

## 关联文档

- `improvements/replace-assert-false-with-throw.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-21`：原债务条目
- `include/ptxsim/ptx_exceptions.h`：异常类层次定义
- `src/ptx_ir/statement_context.cpp`：目标修改文件
