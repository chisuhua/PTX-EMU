# PTX-EMU 调试配置使用指南

本文档介绍如何使用不同的调试配置来分析和调试 PTX-EMU 程序。

## 📋 配置文件列表

| 配置文件 | 用途 | 日志级别 | 输出目标 | 适用场景 |
|---------|------|---------|---------|---------|
| `release_config.ini` | 生产运行 | info | console | 正常运行程序，最佳性能 |
| `debug_config.ini` | 日常开发 | debug | both | 日常调试，平衡性能和信息 |
| `verbose_trace_config.ini` | 详细跟踪 | trace | file | 深入分析问题，理解执行流程 |
| `memory_debug_config.ini` | 内存调试 | debug/trace | both | 调试内存相关问题 |
| `instruction_debug_config.ini` | 指令调试 | info/trace | both | 调试特定指令问题 |
| `perf_config.ini` | 性能分析 | warning | console | 性能测试，最小开销 |

## 🚀 快速使用方法

### 方法 1：复制到工作目录（推荐）

```bash
# 选择需要的配置文件
cp configs/debug_config.ini ./ptx_debug.conf

# 运行程序（程序会自动查找 ptx_debug.conf）
./build/bin/dummy-args
```

### 方法 2：直接指定配置文件

```bash
# 使用详细跟踪配置
cp configs/verbose_trace_config.ini ./ptx_debug.conf
./build/bin/dummy-args

# 或使用内存调试配置
cp configs/memory_debug_config.ini ./ptx_debug.conf
./build/bin/dummy-args
```

### 方法 3：使用快捷脚本

```bash
# 使用调试配置
./debug-run.sh debug ./build/bin/dummy-args

# 使用详细跟踪配置
./debug-run.sh trace ./build/bin/dummy-args

# 使用内存调试配置
./debug-run.sh memory ./build/bin/dummy-args
```

## 🔍 调试场景和配置选择

### 场景 1：程序崩溃或行为异常

**推荐配置**: `verbose_trace_config.ini`

```bash
cp configs/verbose_trace_config.ini ./ptx_debug.conf
./build/bin/your_program

# 查看日志文件
tail -f ptx_emu_trace.log
```

**分析步骤**:
1. 启用 trace 级别日志，记录所有执行细节
2. 查看日志文件，定位崩溃前的最后几条指令
3. 检查寄存器状态和内存访问

### 场景 2：内存访问错误

**推荐配置**: `memory_debug_config.ini`

```bash
cp configs/memory_debug_config.ini ./ptx_debug.conf
./build/bin/your_program

# 查看内存相关日志
grep "mem" ptx_emu_memory_debug.log | tail -50
```

**分析步骤**:
1. 启用内存组件的 trace 级别日志
2. 跟踪所有内存读写操作
3. 检查非法地址访问或越界访问

### 场景 3：指令执行错误

**推荐配置**: `instruction_debug_config.ini`

```bash
cp configs/instruction_debug_config.ini ./ptx_debug.conf
./build/bin/your_program

# 查看指令执行日志
grep "instr" ptx_emu_instr_debug.log | tail -100
```

**分析步骤**:
1. 启用指令执行的详细跟踪
2. 检查每条指令的输入输出
3. 对比预期行为和实际行为

### 场景 4：性能问题

**推荐配置**: `perf_config.ini`

```bash
cp configs/perf_config.ini ./ptx_debug.conf
time ./build/bin/your_program
```

**分析步骤**:
1. 最小化日志开销
2. 使用 `time` 命令测量执行时间
3. 对比不同优化方案的性能

### 场景 5：日常开发调试

**推荐配置**: `debug_config.ini`

```bash
cp configs/debug_config.ini ./ptx_debug.conf
./build/bin/your_program

# 实时查看日志
tail -f ptx_emu_debug.log
```

**分析步骤**:
1. 平衡的日志级别，不会信息过载
2. 控制台和文件同时输出
3. 适合日常开发和调试

## 📊 日志分析方法

### 使用 grep 过滤日志

```bash
# 查找错误日志
grep "ERROR" ptx_emu_debug.log

# 查找特定组件日志
grep "\[mem\]" ptx_emu_debug.log

# 查找特定指令
grep "st.global" ptx_emu_debug.log

# 查找寄存器操作
grep "Register.*contains" ptx_emu_debug.log
```

### 使用日志分析工具

```bash
# 实时跟踪日志
tail -f ptx_emu_debug.log

# 分页查看大文件
less -R ptx_emu_trace.log

# 统计日志级别分布
grep -o '\[DEBUG\]\|\[INFO\]\|\[ERROR\]\|\[TRACE\]' ptx_emu_debug.log | sort | uniq -c
```

## 🛠️ 高级调试技巧

### 1. 组合使用配置文件

可以基于现有配置文件创建自定义配置：

```bash
# 基于 debug 配置，增加指令跟踪
cp configs/debug_config.ini ./custom_debug.ini
# 编辑 custom_debug.ini，修改需要的配置项
```

### 2. 临时修改配置

在调试过程中，可以直接修改 `./ptx_debug.conf`：

```bash
# 临时增加某个组件的日志级别
sed -i 's/component.mem=info/component.mem=trace/' ./ptx_debug.conf
```

### 3. 对比不同配置的日志

```bash
# 使用 debug 配置运行
cp configs/debug_config.ini ./ptx_debug.conf
./build/bin/your_program
mv ptx_emu_debug.log debug_run.log

# 使用 verbose 配置运行
cp configs/verbose_trace_config.ini ./ptx_debug.conf
./build/bin/your_program
mv ptx_emu_trace.log verbose_run.log

# 对比日志
diff debug_run.log verbose_run.log
```

## ⚠️ 注意事项

1. **日志文件大小**: trace 级别会产生大量日志（可能数百 MB），确保有足够磁盘空间
2. **性能影响**: 详细日志会显著降低执行速度（可能 10-100 倍）
3. **输出目标**: 使用 `target=file` 避免控制台输出过慢
4. **及时清理**: 定期清理旧的日志文件

## 📝 配置模板

如果需要创建自定义配置，可以参考以下模板：

```ini
[logger]
global_level=debug
target=both
logfile=ptx_emu_debug.log
async=false
colorize=true
show_timestamp=true
show_level=true
show_component=true
show_location=false
show_thread_id=true

# 组件日志级别
component.emu=debug
component.exec=debug
component.mem=info
component.reg=info
component.thread=debug
component.func=debug
component.instr=info

[debugger]
trace_instruction=true
trace_instruction_type.memory=true
trace_instruction_type.arithmetic=true
trace_instruction_type.control=false
trace_instruction_type.logic=false
trace_instruction_type.convert=false
trace_instruction_type.special=false
trace_instruction_type.other=false

trace_registers=true
trace_memory=true
trace_warp=false
trace_instruction_status=true
trace_lanes=0x1

[gpu]
gpu_config_file=mini.json
```

## 🔗 相关文档

- [调试功能完整指南](docs/debugging_guide.md)
- [GPU 配置说明](configs/mini.json)
- [日志系统架构](docs/architecture.md)
