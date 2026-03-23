# PTX-EMU 调试速查表

## 🎯 快速启动

### 使用快捷脚本（推荐）

```bash
# 查看所有配置
./scripts/debug-run.sh help

# 日常调试
./scripts/debug-run.sh debug ./build/bin/dummy-args

# 详细跟踪
./scripts/debug-run.sh trace ./build/bin/dummy-args

# 内存调试
./scripts/debug-run.sh memory ./build/bin/dummy-args

# 指令调试
./scripts/debug-run.sh instruction ./build/bin/dummy-args

# 性能测试
./scripts/debug-run.sh perf ./build/bin/RAY 512 512
```

### 手动配置

```bash
# 复制配置文件
cp configs/debug_config.ini ./ptx_debug.conf

# 运行程序
./build/bin/dummy-args

# 查看日志
tail -f ptx_emu_debug.log
```

---

## 🔍 配置选择速查

| 场景 | 命令 | 配置文件 | 日志文件 |
|------|------|---------|---------|
| **日常调试** | `./scripts/debug-run.sh debug ...` | `debug_config.ini` | `ptx_emu_debug.log` |
| **详细跟踪** | `./scripts/debug-run.sh trace ...` | `verbose_trace.ini` | `ptx_emu_trace.log` |
| **内存问题** | `./scripts/debug-run.sh memory ...` | `memory_debug.ini` | `ptx_emu_memory_debug.log` |
| **指令错误** | `./scripts/debug-run.sh instruction ...` | `instruction_debug.ini` | `ptx_emu_instr_debug.log` |
| **性能测试** | `./scripts/debug-run.sh perf ...` | `perf_config.ini` | 控制台 |
| **生产运行** | `./scripts/debug-run.sh release ...` | `release_config.ini` | 控制台 |

---

## 📋 常用日志分析命令

### 查看错误

```bash
grep "ERROR\|FATAL" ptx_emu_*.log
```

### 查看特定组件

```bash
# 内存操作
grep "\[mem\]" ptx_emu_*.log

# 指令执行
grep "\[instr\]" ptx_emu_*.log

# 执行引擎
grep "\[exec\]" ptx_emu_*.log

# 寄存器操作
grep "Register.*contains" ptx_emu_*.log
```

### 查看特定指令

```bash
# 内存指令
grep "st.global\|ld.global" ptx_emu_*.log

# 算术指令
grep "\[CLK.*\] \[TRACE\] \[instr\].*Add\|Sub\|Mul" ptx_emu_*.log
```

### 时间线分析

```bash
# 查看前 N 个时钟周期
grep "\[CLK:" ptx_emu_*.log | head -100

# 查看特定周期
grep "\[CLK:100\]" ptx_emu_*.log
```

### 崩溃分析

```bash
# 查看崩溃前的指令
tac ptx_emu_trace.log | grep -A 50 "PC\["
```

---

## 🐛 调试场景速查

### 场景 1: PTX 解析错误

**症状**: "语法错误", "ANTLR 错误", "解析失败"

**步骤**:
```bash
# 1. 使用详细跟踪配置
./scripts/debug-run.sh trace ./build/bin/dummy-args

# 2. 查看解析日志
grep "PTX version\|PTX target" ptx_emu_trace.log
grep "parser\|lexer" ptx_emu_trace.log

# 3. 定位错误位置
grep "ERROR" ptx_emu_trace.log
```

### 场景 2: 测试失败

**症状**: "ctest 失败", "单元测试不通过"

**步骤**:
```bash
# 1. 使用 debug 配置
./scripts/debug-run.sh debug ./build/bin/测试程序

# 2. 查看详细输出
cd build && ctest -R 测试名 -V --output-on-failure

# 3. 分析日志
tail -100 ptx_emu_debug.log
```

### 场景 3: 程序崩溃

**症状**: "segfault", "core dumped", "崩溃"

**步骤**:
```bash
# 1. 使用详细跟踪配置
./scripts/debug-run.sh trace ./build/bin/程序

# 2. 获取堆栈跟踪
gdb -batch -ex "run" -ex "bt" ./build/bin/程序

# 3. 分析崩溃点
grep "PC\[" ptx_emu_trace.log | tail -20
```

### 场景 4: 内存问题

**症状**: "非法访问", "越界", "内存泄漏"

**步骤**:
```bash
# 1. 使用内存调试配置
./scripts/debug-run.sh memory ./build/bin/程序

# 2. 查看内存操作
grep "\[mem\]" ptx_emu_memory_debug.log

# 3. 检查分配/释放配对
grep "Memory allocated\|Memory freed" ptx_emu_memory_debug.log
```

### 场景 5: 指令错误

**症状**: "结果不对", "执行错误"

**步骤**:
```bash
# 1. 使用指令调试配置
./scripts/debug-run.sh instruction ./build/bin/程序

# 2. 查看指令序列
grep "\[instr\]" ptx_emu_instr_debug.log

# 3. 对比寄存器状态
grep "Commit.*%" ptx_emu_instr_debug.log
```

### 场景 6: 性能问题

**症状**: "太慢", "需要优化"

**步骤**:
```bash
# 1. 使用性能配置
./scripts/debug-run.sh perf ./build/bin/程序

# 2. 测量时间
time ./build/bin/程序

# 3. 性能分析（如果可用）
perf record ./build/bin/程序
perf report
```

---

## 🛠️ 调试工具

### GDB 调试

```bash
# 启动 gdb
gdb ./build/bin/程序

# 运行并捕获崩溃
(gdb) run
(gdb) bt  # 崩溃后查看堆栈

# 设置断点
(gdb) break 文件名：行号

# 单步执行
(gdb) next
(gdb) step
```

### Valgrind 内存检查

```bash
# 内存泄漏检查
valgrind --leak-check=full ./build/bin/程序

# 详细模式
valgrind --leak-check=full --track-origins=yes ./build/bin/程序
```

### AddressSanitizer

```bash
# 需要重新编译
cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=On ..
make
./build/bin/程序  # ASan 会自动检测内存错误
```

---

## 📊 日志级别说明

| 级别 | 说明 | 使用场景 |
|------|------|---------|
| `trace` | 最详细信息 | 深入分析问题 |
| `debug` | 调试信息 | 日常开发调试 |
| `info` | 一般信息 | 了解程序流程 |
| `warning` | 警告 | 性能测试 |
| `error` | 错误 | 生产环境 |
| `fatal` | 致命错误 | 生产环境 |

---

## 🎓 最佳实践

1. **选择合适的配置**: 不要总是用 trace，日常用 debug 即可
2. **及时清理日志**: trace 日志可能很大（数百 MB）
3. **保留关键日志**: 分析问题时不要删除
4. **使用快捷脚本**: 比手动配置更方便
5. **恢复配置**: 调试完成后用 release 配置运行

---

## 🔗 相关文档

- [完整调试指南](docs/debug-config-guide.md)
- [调试技能说明](docs/ptx-debug-skill-usage.md)
- [快捷脚本](scripts/debug-run.sh)

---

## 💡 提示

**自动调试**: 遇到问题时，直接描述问题，`ptx-debug` 技能会自动选择配置并分析！

**示例**:
```
用户："PTX 解析失败，怎么办？"
→ 技能自动选择 trace 配置并分析日志
```
