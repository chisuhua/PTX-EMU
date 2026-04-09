# PTX-EMU 调试功能指南

本文档介绍了 PTX-EMU 中的调试功能，包括如何配置日志系统、使用断点以及分析程序执行。

## 1. 日志系统

PTX-EMU 使用一个灵活的日志系统，允许用户配置不同组件的日志级别和输出目标。

### 1.1 配置文件

日志系统可以通过配置文件进行配置。配置文件使用 INI 格式，示例如下：

```ini
# 全局日志级别
global_level=debug

# 日志输出目标 (console, file, both)
target=both

# 日志文件路径（当target包含file时使用）
logfile=ptx_emu_debug.log

# 是否启用异步日志记录
async=false

# 是否启用颜色输出（控制台输出时）
colorize=true

# 组件特定的日志级别设置
component.emu=debug
component.exec=trace
component.mem=debug
component.reg=info
component.thread=debug
component.func=trace
```

### 1.2 日志级别

日志系统支持以下级别（按严重程度递增）：

1. `trace` - 最详细的日志信息，通常用于跟踪程序执行流程
2. `debug` - 调试信息，用于诊断问题
3. `info` - 一般信息，描述程序正常运行状态
4. `warning` - 警告信息，表示可能出现问题但不影响程序运行
5. `error` - 错误信息，表示发生了错误但程序还能继续运行
6. `fatal` - 致命错误，程序将终止运行

### 1.3 组件分类

日志消息按照来源组件进行分类：

- `emu` - 模拟器核心
- `exec` - 指令执行
- `mem` - 内存操作
- `reg` - 寄存器操作
- `thread` - 线程相关
- `func` - 函数调用追踪

## 2. 断点功能

PTX-EMU 支持在 PTX 指令上设置断点。断点可以在配置文件中定义，也可以在运行时动态添加。

### 2.1 配置文件中的断点

断点可以在配置文件中定义：

```ini
# 断点配置示例
breakpoint.0.address=100
breakpoint.0.condition=reg.r0 == 5
breakpoint.1.address=200
breakpoint.1.condition=mem[0x1000] > 100
```

### 2.2 断点条件表达式

断点条件支持简单的表达式，包括：

- 寄存器访问: `reg.r0`, `reg.%tid.x`
- 内存访问: `mem[0x1000]`
- 比较操作: `==`, `!=`, `<`, `<=`, `>`, `>=`
- 数学运算: `+`, `-`, `*`, `/`

## 3. 性能分析

PTX-EMU 提供性能分析功能，可以帮助用户了解程序的执行性能。

### 3.1 PerfTimer 类

`PerfTimer` 类用于测量代码段的执行时间。它会在析构时自动输出统计信息：

```cpp
{
    PerfTimer timer("kernel_execution");
    // 执行需要计时的代码
} // 在这里自动输出计时结果
```

## 4. 使用示例

### 4.1 运行带有调试配置的程序

要使用调试配置运行程序，请将配置文件传递给程序：

```bash
./your_program --debug-config debug_config.ini
```

或者，如果程序支持默认配置文件名：

```bash
./your_program
# 程序将在当前目录查找 ptx_debug.conf 文件
```

### 4.2 分析日志输出

当日志目标设置为 `both` 或 `file` 时，日志将写入指定的文件中。您可以使用文本编辑器或日志分析工具查看和分析日志：

```bash
tail -f ptx_emu_debug.log
```

## 5. 扩展调试功能

开发者可以通过以下方式扩展调试功能：

### 5.1 添加新的日志宏

在 `ptx_debug.h` 中添加新的日志宏：

```cpp
#define PTX_DEBUG_NEWCOMPONENT(fmt, ...) \
    PTX_LOG(ptxsim::log_level::debug, "newcomponent", fmt, ##__VA_ARGS__)
```

### 5.2 自定义断点条件

扩展断点条件解析器以支持新的表达式类型。

## 6. 最佳实践

1. **合理设置日志级别**：在生产环境中使用较高的日志级别（如 `info` 或 `warning`），在调试时使用较低的级别（如 `debug` 或 `trace`）。

2. **使用组件特定的日志**：为不同的功能模块使用不同的组件标签，方便过滤和分析。

3. **定期清理日志文件**：日志文件可能会变得很大，定期清理或轮换日志文件。

4. **使用性能分析工具**：利用 `PerfTimer` 类识别性能瓶颈。