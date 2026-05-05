# ADR-0009: X-Macro + Weak Symbol 指令分发模式

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 0-9 (初始架构) |
| **作者** | PTX-EMU Team |

## 上下文

PTX-EMU 需要支持约 70 条 PTX 指令，每条指令需要一个 handler 函数。指令分发的核心问题是：

**如何将解析后的指令类型映射到对应的 handler 函数？**

传统的解决方案包括：
- switch-case 语句（需要手动维护映射）
- 函数指针表（需要手动注册）
- 虚函数表（需要基类和继承体系）

项目选择了 X-Macro + `__attribute__((weak))` 的组合模式。

## 决策驱动因素

1. **减少重复代码**：70+ 指令的手动映射容易出错和遗漏
2. **支持渐进实现**：未实现的指令应有 stub 而非编译错误
3. **单一真相源**：指令列表只在一个地方定义
4. **编译时生成**：无运行时开销

## 考虑的替代方案

### 方案 A: Switch-Case 语句

**描述**: 在 dispatcher 中使用 switch-case 分发到各 handler

**优点**:
- 简单直观
- 编译器优化好

**缺点**:
- 手动维护 70+ case 分支
- 新增指令需要修改 dispatcher
- 容易遗漏某些指令

### 方案 B: 函数指针注册表

**描述**: 运行时维护 opcode → handler 的 map

**优点**:
- 支持动态注册
- 可扩展性好

**缺点**:
- 需要手动注册每个 handler
- 运行时查找开销
- 注册遗漏导致运行时错误

### 方案 C: X-Macro + Weak Symbol (✅ 选中)

**描述**: 
1. 用 X-Macro 在 `ptx_op.def` 中定义所有指令
2. 用 X-Macro 自动生成 dispatcher 的 switch-case
3. 每个 handler 用 `__attribute__((weak))` 提供默认 stub

**优点**:
- 指令列表单一真相源
- 新增指令只需在 ptx_op.def 添加一行 + 实现 handler
- 未实现的指令有默认 stub（什么都不做）
- 编译时生成，零运行时开销

**缺点**:
- X-Macro 语法晦涩，调试困难
- weak symbol 可能隐藏未实现的指令（运行时静默跳过）
- IDE 支持差（无法跳转、无法自动补全）

**选择理由**: 在快速原型阶段，X-Macro 能显著减少重复代码，weak symbol 允许渐进实现。但长期来看，这个模式的可维护性较差。

## 决策内容

### 设计原则

1. **ptx_op.def 是单一真相源**：所有指令在此定义
2. **weak symbol 提供默认 stub**：未实现的指令不报错，静默跳过
3. **X-Macro 自动生成**：dispatcher、枚举、字符串转换都由宏生成

### 实现要点

```cpp
// ptx_op.def - 指令定义（单一真相源）
#ifndef PTX_OP
#define PTX_OP(op, category)
#endif

PTX_OP(mov, DATA_TRANSFER)
PTX_OP(add, ARITHMETIC)
PTX_OP(sub, ARITHMETIC)
PTX_OP(bra, CONTROL_FLOW)
PTX_OP(bar, BARRIER)
// ... 70+ 指令

// instruction_handlers.cpp - X-Macro 生成 dispatcher
StatementType string_to_statement_type(const std::string& s) {
    #define PTX_OP(op, cat) if (s == #op) return S_##op;
    #include "ptx_op.def"
    #undef PTX_OP
    return S_UNKNOWN;
}

// 指令 handler 声明（weak symbol）
#define PTX_OP(op, cat) \
    __attribute__((weak)) void handle_##op(ThreadContext* ctx, StatementContext& stmt);
#include "ptx_op.def"

// 默认 stub 实现
#define PTX_OP(op, cat) \
    __attribute__((weak)) void handle_##op(ThreadContext* ctx, StatementContext& stmt) { \
        /* stub: do nothing */ \
    }
#include "ptx_op.def"

// dispatcher
void dispatch_instruction(ThreadContext* ctx, StatementContext& stmt) {
    switch (stmt.type) {
        #define PTX_OP(op, cat) case S_##op: handle_##op(ctx, stmt); break;
        #include "ptx_op.def"
        #undef PTX_OP
    }
}

// 具体实现（覆盖 weak stub）
void handle_mov(ThreadContext* ctx, StatementContext& stmt) {
    // 实际实现
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/ptx_op.def` | 新增 | 指令定义（单一真相源） |
| `src/ptxsim/instruction_handlers.cpp` | 修改 | X-Macro 生成 dispatcher |
| `src/ptxsim/instructions/*.cpp` | 新增 | 各指令 handler 实现 |

## 后果

### 正面影响

- 新增指令只需修改两处（ptx_op.def + handler 实现）
- 未实现指令不会导致编译错误
- 编译时生成，无运行时开销

### 负面影响

- X-Macro 语法晦涩，新开发者学习成本高
- weak symbol 可能隐藏未实现的指令（运行时静默跳过，难以发现）
- IDE 无法识别宏生成的代码（无法跳转、补全）
- 调试困难（栈显示的是宏展开后的代码）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| weak stub 被误认为已实现 | 中 | 中 | 日志输出 "stub: instruction not implemented" |
| X-Macro 语法错误难以定位 | 低 | 高 | 使用 `gcc -E` 预编译查看展开结果 |
| 新增指令遗漏 handler | 中 | 低 | weak stub 确保不会崩溃，但功能缺失 |

### 未来演进方向

在 Phase 12 中计划迁移到**指令注册表模式**（ADR-0010），替代 X-Macro + weak symbol：

- 编译时注册（constexpr 或 template）
- 运行时查询
- 更好的 IDE 支持
- 遗漏 handler 在编译时报错

## 合规检查

后续相关开发应检查：

- [ ] 新增指令在 ptx_op.def 中定义
- [ ] 实现 handler 时覆盖 weak stub（不加 weak 属性）
- [ ] stub handler 输出日志提示 "not implemented"
- [ ] 定期扫描 weak handler，确认是否应实现

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [架构评审报告 - 5.2.2 指令实现的宏展开模式](../reports/architecture-review-report.md#522-指令实现的宏展开模式)
- [任务计划 - T12.3.x 指令分发重构](../reports/task-plan.md#sprint-123-指令分发重构--antlr-统一day-22-25)
