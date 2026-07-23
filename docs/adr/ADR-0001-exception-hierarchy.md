# ADR-0001: 异常层次体系替代 assert

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-03 |
| **关联任务** | T11.1.1-T11.1.4 |
| **作者** | PTX-EMU Team |

## 上下文

PTX-EMU 项目早期使用 `assert(false)` 处理不可达路径和错误情况。这种方式存在以下问题：

- **无法恢复**：assert 直接终止程序，调用方无法捕获和处理
- **缺少上下文**：错误信息有限，不包含错误类型、位置、原因等结构化信息
- **Release 模式失效**：NDEBUG 定义时 assert 会被完全移除
- **顶层无法捕获**：cudart 层没有 try-catch，异常无法转化为错误码返回给用户

## 决策驱动因素

1. **可控的错误处理**：需要调用方能选择捕获、恢复或优雅降级
2. **结构化错误信息**：错误应包含错误码、类型、描述、上下文
3. **Release 模式一致性**：错误处理不应因编译模式不同而行为不同
4. **与 CUDA Runtime 兼容**：cudart 需要返回 cudaError_t，不能直接 crash

## 考虑的替代方案

### 方案 A: 继续使用 assert + 日志

**描述**: 保留 assert，增加详细日志输出

**优点**:
- 实现简单，无需修改现有代码结构
- 开发阶段能快速暴露问题

**缺点**:
- Release 模式下 assert 失效
- 调用方无法捕获和恢复
- 无法返回结构化错误码

### 方案 B: 错误码返回 (Error Code Pattern)

**描述**: 所有函数返回 int/enum 错误码

**优点**:
- 与 C 风格 API 兼容
- 性能开销小

**缺点**:
- 错误处理代码与业务代码混杂
- 容易忽略错误返回值
- 错误传播路径长时容易丢失上下文

### 方案 C: 异常层次体系 (✅ 选中)

**描述**: 定义基类 `PtxEmuException` 和多个子类，替代所有 assert(false)

**优点**:
- 调用方可选择性捕获和处理
- 支持携带丰富的错误上下文
- 编译模式无关，行为一致
- 可通过 try-catch 在顶层统一处理

**缺点**:
- 异常有轻微性能开销（但错误路径本就不在热路径）
- 需要确保异常构造函数 noexcept

**选择理由**: PTX-EMU 是模拟器，错误路径不是性能敏感路径。异常提供了最佳的可恢复性和错误信息表达能力，且与 C++ 项目惯例一致。

## 决策内容

### 设计原则

1. **继承自 std::exception**：保证与标准库兼容
2. **noexcept 构造函数**：防止异常构造时抛出二次异常
3. **错误码枚举**：每个错误类型有唯一编码
4. **丰富上下文**：异常消息包含文件、行号、指令信息

### 实现要点

```cpp
// 基类
class PtxEmuException : public std::exception {
    ErrorCode error_code_;
    std::string message_;
    const char* file_;
    int line_;
public:
    const char* what() const noexcept override;
    ErrorCode get_error_code() const noexcept;
    const char* get_error_code_name() const noexcept;
};

// 4 个子类
class UnsupportedInstructionException : public PtxEmuException { ... };
class InvalidMemoryAccessException : public PtxEmuException { ... };
class PTXParseException : public PtxEmuException { ... };
class ExecutionStateException : public PtxEmuException { ... };
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/ptx_exceptions.h` | 新增 | 异常类定义 |
| `src/ptxsim/core/thread_context.cpp` | 修改 | 3 处 assert → throw |
| `src/cudart/cudart_sim.cpp` | 修改 | 添加顶层 try-catch |

## 后果

### 正面影响

- 所有错误路径可被捕获和处理
- 错误消息包含文件、行号、错误码，调试效率提升
- Release 模式行为一致

### 负面影响

- 异常路径有轻微性能开销（可接受，错误不在热路径）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 异常构造时抛出二次异常 | 低 | 高 | 所有异常构造函数标记 noexcept |
| 遗漏某些 assert(false) 未替换 | 中 | 中 | 代码搜索 `assert(false)` 确保全覆盖 |

## 合规检查

后续相关开发应检查：

- [ ] 新错误路径使用异常而非 assert(false)
- [ ] 异常构造函数标记 noexcept
- [ ] 顶层入口有 try-catch 捕获 PtxEmuException

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-03 | 初始版本 | PTX-EMU Team |

## 参考

- [任务计划](../reports/task-plan.md#phase-11)
- [架构评审报告](../reports/architecture-review-report.md)
