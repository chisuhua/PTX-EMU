# complete-x-macro-dispatch - Design

## Overview

审查 `ptx_op.def` X-Macro 的全部 10 个展开点，统一 X 宏定义模式，评估是否可
减少重复展开。`ptx_op.def` 定义 106 个指令条目，通过 X-Macro 在 8 个文件中
展开 10 次，分别用于枚举生成、字符串转换、Listener/Visitor 声明与实现、
Handler 声明/注册/实现。

当前每个展开点 `#define X(...)` 后 `#include "ptx_op.def"` 再 `#undef X`，
参数命名不统一（如 `op_name` vs `opstr` vs `cpp_name`），但展开逻辑正确。

## Context

### ptx_op.def 格式

```cpp
X(enum_value, cpp_name, string, op_count, struct_kind, instr_kind)
```

### 10 个展开点详情

| # | 文件 | X 参数命名 | 用途 | 可合并？ |
|---|------|-----------|------|---------|
| 1 | `ptx_types.h:20` | `enum_val, struct_name, str, opcount, _, instr_kind` | 枚举生成 | 否（独立 TU） |
| 2 | `ptx_parser.h:162` | `openum, opname, opstr, opcount, struct_kind` | Listener 声明 | 否（header） |
| 3 | `ptx_visiter.h:104` | `openum, opstr, opname, opcount, struct_kind, instr_kind` | Visitor 声明 | 否（header） |
| 4 | `instruction_handlers.h:137` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` | Handler 声明 | 否（header） |
| 5 | `statement_context.cpp:11` | `stype, opkind, opname, count, struct_kind, instr_kind` | 字符串转换 | 否（独立 TU） |
| 6 | `ptx_parser.cpp:1046` | `openum, opname, opstr, opcount, struct_kind` | Listener 实现 | 否（独立 TU） |
| 7 | `ptx_visitor.cpp:590` | `openum, opstr, opname, opcount, _, instr_kind` | Visitor 分派 | 否（独立 TU） |
| 8 | `ptx_visitor_dispatch.cpp:44` | `openum, opstr, opname, opcount, struct_kind, instr_kind` | 类别分派 | 否（独立 TU） |
| 9 | `instruction_factory.cpp:16` | `enum_val, op_name, opstr, op_count, struct_kind, instr_kind` | Handler 注册 | 否（独立 TU） |
| 10 | `instruction_handlers.cpp:188` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` | Handler 实现 | 否（独立 TU） |

### TCGEN05_INSTR 特殊处理

`ptx_op.def` 中 11 个 `S_TCGEN05_*` 条目共享单一 `Tcgen05Handler` 类。在展开点
4（`instruction_handlers.h`）和 10（`instruction_handlers.cpp`）中需要跳过
TCGEN05_INSTR 的 X-Macro 展开，避免 11 次重复定义。当前通过
`#define DECLARE_TCGEN05_INSTR_HANDLER(Name) /* no-op */` 和
`#define IMPLEMENT_TCGEN05_INSTR_HANDLER(Name) /* no-op */` 实现。

## Design Decisions

### 决策 1: 保持独立展开，不做合并

**选择**: 10 个展开点保持独立，不合并

**理由**:
- 每个展开点位于不同的翻译单元（TU）或头文件，C++ 编译模型要求每个 TU 独立展开
- X-Macro 的核心设计就是"同一 .def 文件多次展开生成不同代码"，这是预期行为
- 合并需要引入中间生成步骤（如代码生成器），超出 improvement 范围
- improvement 范围明确"不影响 parser 端的 X-Macro 使用"

**替代方案**:
- A. 用模板元编程替代 X-Macro -> 需重写全部 10 个展开点，风险过高
- B. 用代码生成器（如 CMake 脚本）从 `ptx_op.def` 生成 .cpp 文件 -> 引入构建复杂度
- C. **采用**: 保持 X-Macro 架构，统一命名和模式

### 决策 2: 统一 X 宏参数命名

**选择**: 将所有 10 个展开点的 X 宏参数统一为：
```cpp
#define X(enum_val, op_name, op_str, op_count, struct_kind, instr_kind) ...
```

**理由**:
- 当前参数命名不一致（`opstr` vs `op_str` vs `str`，`opname` vs `op_name` vs `cpp_name`）
- 统一命名提高可读性和可维护性
- 不改变展开结果（参数是位置匹配的，命名仅影响可读性）

**修改清单**:
| # | 文件 | 当前命名 | 统一后 |
|---|------|---------|--------|
| 1 | `ptx_types.h:20` | `enum_val, struct_name, str, opcount, _, instr_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 2 | `ptx_parser.h:162` | `openum, opname, opstr, opcount, struct_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 3 | `ptx_visiter.h:104` | `openum, opstr, opname, opcount, struct_kind, instr_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 5 | `statement_context.cpp:11` | `stype, opkind, opname, count, struct_kind, instr_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 6 | `ptx_parser.cpp:1046` | `openum, opname, opstr, opcount, struct_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 7 | `ptx_visitor.cpp:590` | `openum, opstr, opname, opcount, _, instr_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |
| 8 | `ptx_visitor_dispatch.cpp:44` | `openum, opstr, opname, opcount, struct_kind, instr_kind` | `enum_val, op_name, op_str, op_count, struct_kind, instr_kind` |

**注意**: #4、#9、#10 已使用统一命名，无需修改。

### 决策 3: 统一 #undef X 后的清理

**选择**: 确保所有展开点在 `#include` 后都有 `#undef X`

**理由**:
- 部分展开点可能遗漏 `#undef X`，导致后续代码中 X 宏泄漏
- 统一 `#undef X` 确保宏卫生

**验证**: 全部 10 个展开点已有 `#undef X`，无需修改。

### 决策 4: 文档化 TCGEN05_INSTR 跳过模式

**选择**: 在 `instruction_handlers.h` 和 `instruction_handlers.cpp` 中添加注释
说明 TCGEN05_INSTR 的特殊跳过逻辑

**理由**:
- 当前注释已有说明，但可进一步明确"为何跳过"和"如何维护"
- 统一两个文件中的注释描述

### 决策 5: 不减少展开次数

**选择**: 保持 10 次展开，不尝试减少

**理由**:
- 每个展开点服务于不同的编译目标（枚举/声明/实现），无法合并
- improvement 验收标准为"编译时间不增加"，保持现有展开次数不会增加编译时间
- 减少 X-Macro 展开需要架构变更，超出 improvement 范围
- improvement 技术约束仅要求"SHOULD 减少"（非 MUST）

## Implementation Plan

### Phase 1: 审计现有展开点（30 min）
1. 记录每个展开点的 X 宏参数命名
2. 确认每个展开点的 `#undef X` 清理
3. 记录 TCGEN05_INSTR 特殊处理逻辑
4. 确认无遗漏的展开点

### Phase 2: 统一参数命名（45 min）
1. 统一 `ptx_types.h` 的 X 宏参数命名
2. 统一 `ptx_parser.h` 的 X 宏参数命名
3. 统一 `ptx_visiter.h` 的 X 宏参数命名
4. 统一 `statement_context.cpp` 的 X 宏参数命名
5. 统一 `ptx_parser.cpp` 的 X 宏参数命名
6. 统一 `ptx_visitor.cpp` 的 X 宏参数命名
7. 统一 `ptx_visitor_dispatch.cpp` 的 X 宏参数命名
8. 每步编译验证

### Phase 3: 文档化 TCGEN05 跳过逻辑（15 min）
1. 完善 `instruction_handlers.h` 中的 TCGEN05_INSTR 注释
2. 完善 `instruction_handlers.cpp` 中的 TCGEN05_INSTR 注释

### Phase 4: 全量验证（30 min）
1. Debug + Release 构建通过
2. `ctest` 全绿
3. 确认 106 个 handler 正确注册
4. 编译时间对比（不增加）

## Testing Strategy

| 测试场景 | 命令 | 预期 |
|---------|------|------|
| Debug 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug && cmake --build build` | 通过 |
| Release 编译 | `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build` | 通过 |
| 全量测试 | `cd build && ctest --output-on-failure` | 全绿 |
| handler 注册 | 验证 106 个 handler 均在 `handler_map` 中注册 | 106 个 |
| PTX 语法测试 | `./tests/ptx/test_all_ptx.sh` | 全绿 |
| 编译时间 | 对比修改前后 `cmake --build build` 时间 | 不增加 |

### Handler 注册验证

```cpp
// 验证 InstructionFactory::initialize() 后 handler_map.size() == 106
// （减去 TCGEN05_INSTR 的 11 个共享 handler，实际注册 106 个 key -> Tcgen05Handler）
```

### 编译时间验证

```bash
# 修改前
time cmake --build build 2>&1 | tail -1
# 修改后
time cmake --build build 2>&1 | tail -1
# 对比时间差（应 ≤ 0）
```

## Risks / Trade-offs

| 风险 | 影响 | 缓解 |
|------|------|------|
| 参数命名修改导致编译错误 | 编译失败 | 参数是位置匹配的，改名不影响展开结果 |
| TCGEN05_INSTR 跳过逻辑被破坏 | 11 次重复定义错误 | 不修改跳过逻辑，仅统一命名 |
| 编译时间增加 | 违反验收标准 | 不增加展开次数，编译时间不变 |
| parser 端 X-Macro 受影响 | 解析行为变化 | improvement 明确"不影响 parser 端" |

## Open Questions

1. **是否应该统一 `ptx_parser.h` 的 5 参数 X 宏为 6 参数？**
   - `ptx_parser.h:162` 使用 5 参数（缺少 `instr_kind`），因为 Listener 不需要 `instr_kind`
   - 统一为 6 参数但忽略最后一个，还是保持 5 参数？
   - **决定**: 保持 5 参数（Listener 确实不需要 `instr_kind`，强加会引入 unused 警告）

2. **是否应该在 `ptx_op.def` 中添加参数名注释？**
   - `ptx_op.def` 头部已有格式注释（line 3-4）
   - 无需额外修改

## 关联文档

- `improvements/complete-x-macro-dispatch.md`：完整 5 段提案
- `docs/roadmap/post-phase3-debt-roadmap.md §1.2 C-15`：原债务条目
- `include/ptx_ir/ptx_op.def`：X-Macro 定义（106 条目，SSOT）
- `include/ptx_ir/AGENTS.md`：X-Macro 使用约定
- `src/ptxsim/AGENTS.md`：Handler 注册架构
