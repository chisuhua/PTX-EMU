---
name: "ptx-debug"
description: "PTX-EMU 专用调试技能 - 自动化调试配置选择和场景化调试方法"
when_to_use: |
  当用户在 PTX-EMU 项目中遇到以下问题时自动触发：
  - "PTX 解析错误", "语法错误", "ANTLR 错误"
  - "测试失败", "单元测试不通过"
  - "程序崩溃", "segfault", "core dumped"
  - "内存错误", "非法访问", "越界"
  - "指令执行错误", "结果不对"
  - "性能慢", "优化"
  - "调试这个", "分析一下问题"
  
  触发关键词：
  - "ptx", "cubin", "GPU", "kernel"
  - "test_", "ctest", "单元测试"
  - "debug", "调试", "analyze", "分析"
  - "为什么失败", "怎么回事", "哪里错了"

skills_required: ["cpp-debug", "systematic-debugging"]
---

# PTX-EMU 调试技能

## 核心原则

**自动化调试配置选择**：根据问题类型自动选择最合适的调试配置，避免手动配置。

**场景化调试方法**：针对不同调试场景（解析错误、内存问题、指令错误、性能问题）提供专门的调试流程。

**证据驱动**：使用适当的调试配置收集证据，定位问题根源。

---

## 调试配置自动选择

### 配置选择矩阵

| 问题类型 | 关键词 | 自动选择配置 | 日志级别 | 输出目标 |
|---------|--------|-------------|---------|---------|
| **PTX 解析错误** | "解析错误", "语法错误", "ANTLR", "grammar" | `verbose_trace_config.ini` | trace | file |
| **测试失败** | "测试失败", "test failed", "ctest" | `debug_config.ini` | debug | both |
| **程序崩溃** | "崩溃", "segfault", "SIGSEGV", "core" | `verbose_trace_config.ini` | trace | file |
| **内存问题** | "内存", "memory", "越界", "非法访问" | `memory_debug_config.ini` | debug/trace | both |
| **指令错误** | "指令", "instruction", "结果不对", "执行错误" | `instruction_debug_config.ini` | info/trace | both |
| **性能问题** | "性能", "慢", "优化", "benchmark" | `perf_config.ini` | warning | console |
| **日常调试** | "调试", "debug", "看一下" | `debug_config.ini` | debug | both |
| **默认/未知** | 其他情况 | `debug_config.ini` | debug | both |

### 自动选择流程

```
1. 分析用户问题描述
   ↓
2. 提取关键词匹配问题类型
   ↓
3. 根据配置选择矩阵确定调试配置
   ↓
4. 自动复制配置文件到工作目录
   ↓
5. 运行程序收集调试信息
   ↓
6. 分析日志文件，定位问题
```

---

## 场景化调试方法

### 场景 1: PTX 解析错误

**触发条件**: 用户报告 "PTX 解析错误", "语法错误", "ANTLR 错误"

**自动行动**:

1. **选择配置**: `verbose_trace_config.ini`
   ```bash
   cp configs/verbose_trace_config.ini ./ptx_debug.conf
   ```

2. **运行解析器**:
   ```bash
   ./build/bin/dummy-args  # 或其他触发解析的程序
   ```

3. **收集证据**:
   ```bash
   # 查看解析器日志
   grep "PTX version\|PTX target\|Address size" ptx_emu_trace.log
   
   # 查看 ANTLR 解析过程
   grep "Visiting PTX\|parser\|lexer" ptx_emu_trace.log
   ```

4. **分析重点**:
   - PTX 版本号是否正确
   - 目标架构是否支持
   - 解析到哪条指令时失败
   - ANTLR 错误消息的具体内容

5. **定位语法文件**:
   ```bash
   # 查找相关的语法文件
   grep -r "错误中提到的指令名" src/grammar/
   ```

6. **生成测试用例**:
   - 提取导致错误的 PTX 代码
   - 添加到 `tests/ptx/` 目录
   - 运行 `./tests/ptx/test_all_ptx.sh` 验证

---

### 场景 2: 测试失败

**触发条件**: 用户报告 "测试失败", "test failed", "ctest 不通过"

**自动行动**:

1. **选择配置**: `debug_config.ini`
   ```bash
   cp configs/debug_config.ini ./ptx_debug.conf
   ```

2. **运行失败测试**:
   ```bash
   cd build && ctest -R 测试名 -V
   ```

3. **收集证据**:
   ```bash
   # 查看测试输出
   ctest -R 测试名 --output-on-failure
   
   # 查看详细日志
   tail -100 ptx_emu_debug.log
   ```

4. **分析重点**:
   - 测试期望值 vs 实际值
   - 失败前的最后几条指令
   - 寄存器状态是否正确
   - 内存访问是否越界

5. **对比参考实现**:
   ```bash
   # 查找类似的通过测试
   grep -r "类似功能" tests/
   ```

6. **生成调试脚本**:
   ```bash
   # 创建最小复现
   cat > debug_test.sh << 'EOF'
   #!/bin/bash
   cp configs/debug_config.ini ./ptx_debug.conf
   ./build/bin/测试程序
   EOF
   chmod +x debug_test.sh
   ```

---

### 场景 3: 程序崩溃

**触发条件**: 用户报告 "崩溃", "segfault", "SIGSEGV", "core dumped"

**自动行动**:

1. **选择配置**: `verbose_trace_config.ini`
   ```bash
   cp configs/verbose_trace_config.ini ./ptx_debug.conf
   ```

2. **运行程序获取堆栈**:
   ```bash
   # 使用 gdb 获取堆栈
   gdb -batch -ex "run" -ex "bt" ./build/bin/程序
   
   # 或直接运行获取 core dump
   ulimit -c unlimited
   ./build/bin/程序
   ```

3. **收集证据**:
   ```bash
   # 查看崩溃前的指令序列
   grep "PC\[" ptx_emu_trace.log | tail -50
   
   # 查看内存访问
   grep "Memory allocated\|Memory freed" ptx_emu_trace.log
   
   # 查看寄存器状态
   grep "Register.*contains" ptx_emu_trace.log
   ```

4. **分析重点**:
   - 崩溃时的 PC 值
   - 最后执行的指令
   - 访问的内存地址是否合法
   - 寄存器值是否异常

5. **定位崩溃点**:
   ```bash
   # 使用 addr2line 定位
   addr2line -e ./build/bin/程序 崩溃地址
   ```

6. **生成修复方案**:
   - 检查空指针
   - 检查数组越界
   - 检查 use-after-free
   - 检查资源未初始化

---

### 场景 4: 内存问题

**触发条件**: 用户报告 "内存错误", "越界", "非法访问"

**自动行动**:

1. **选择配置**: `memory_debug_config.ini`
   ```bash
   cp configs/memory_debug_config.ini ./ptx_debug.conf
   ```

2. **运行程序**:
   ```bash
   ./build/bin/程序
   ```

3. **收集证据**:
   ```bash
   # 查看所有内存操作
   grep "\[mem\]" ptx_emu_memory_debug.log
   
   # 查看特定地址的访问
   grep "0x具体地址" ptx_emu_memory_debug.log
   
   # 查看内存分配/释放配对
   grep "Memory allocated\|Memory freed" ptx_emu_memory_debug.log
   ```

4. **分析重点**:
   - 内存分配大小 vs 访问大小
   - 释放后是否再次访问
   - 是否越界访问
   - 地址对齐是否正确

5. **使用工具验证**:
   ```bash
   # 如果有 valgrind 支持
   valgrind --leak-check=full --track-origins=yes ./build/bin/程序
   
   # 或使用 AddressSanitizer（需要重新编译）
   # cmake -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON ..
   ```

---

### 场景 5: 指令执行错误

**触发条件**: 用户报告 "指令错误", "结果不对", "执行异常"

**自动行动**:

1. **选择配置**: `instruction_debug_config.ini`
   ```bash
   cp configs/instruction_debug_config.ini ./ptx_debug.conf
   ```

2. **运行程序**:
   ```bash
   ./build/bin/程序
   ```

3. **收集证据**:
   ```bash
   # 查看指令执行序列
   grep "\[instr\]" ptx_emu_instr_debug.log
   
   # 查看特定指令类型
   grep "st.global\|ld.global\|add" ptx_emu_instr_debug.log
   
   # 查看寄存器变化
   grep "Commit.*%" ptx_emu_instr_debug.log
   ```

4. **分析重点**:
   - 指令的输入操作数
   - 指令的输出结果
   - 前后寄存器状态对比
   - 指令语义是否正确实现

5. **对比参考**:
   ```bash
   # 查找 PTX ISA 文档
   # 或查看其他实现
   grep -r "指令名" src/ptxsim/instructions/
   ```

6. **验证修复**:
   - 创建单元测试
   - 验证各种边界情况
   - 对比硬件行为

---

### 场景 6: 性能问题

**触发条件**: 用户报告 "性能慢", "需要优化", "benchmark"

**自动行动**:

1. **选择配置**: `perf_config.ini`
   ```bash
   cp configs/perf_config.ini ./ptx_debug.conf
   ```

2. **性能测量**:
   ```bash
   # 测量执行时间
   time ./build/bin/程序
   
   # 多次运行取平均
   for i in {1..10}; do time ./build/bin/程序; done
   ```

3. **性能分析**:
   ```bash
   # 使用 perf（如果可用）
   perf record ./build/bin/程序
   perf report
   
   # 或使用 gprof
   cmake -DCMAKE_BUILD_TYPE=Profiling ..
   make
   ./build/bin/程序
   gprof ./build/bin/程序 gmon.out > profile.txt
   ```

4. **分析重点**:
   - 热点函数

---

### 场景 7: Barrier 同步卡住 (SIMT 特定)

**触发条件**:
- 测试超时或长时间无响应
- barrier 相关测试卡住 (test_syncthreads, test_warp_divergence 等)
- 日志显示 threads blocked at barrier

**自动行动**:

1. **快速诊断 (类型 1 - 不需要仿真)**:
   ```bash
   # 1. 验证 PTX 解析是否正确
   /usr/local/cuda/bin/cuobjdump -ptx ./build/bin/test_xxx 2>/dev/null | \
     tail -n+12 > /tmp/test_xxx.ptx
   ./build/bin/test-ptx /tmp/test_xxx.ptx 2>&1 | grep "bar.warp.sync"
   ```

2. **启用详细日志**:
   ```bash
   cp configs/verbose_trace_config.ini ./ptx_debug.conf
   ```

3. **运行收集日志**:
   ```bash
   timeout 120 ./build/bin/test_xxx 2>&1 | tee /tmp/test_output.log
   ```

4. **分析关键模式**:
   ```bash
   grep "Initialized wbar" /tmp/test_output.log
   grep "Barrier complete" /tmp/test_output.log
   grep "blocked at bar.warp.sync" /tmp/test_output.log | tail -10
   ```

5. **预期行为**:
   ```
   # 正常:
   bar.warp.sync: Initialized wbar[0] with mask=0xFFFFFFFF, reconvergence_pc=N
   Lane 0 arrived (arrived=1/32)
   ...
   Lane 31 arrived (arrived=32/32)
   bar.warp.sync: Barrier complete, releasing 32 threads to PC=N
   
   # 异常:
   - 停在 arrived=X/32 (X<32) → 部分线程未到达
   - Barrier complete 但仍卡住 → PC 更新或调度问题
   ```

6. **常见问题诊断表**:

| 现象 | 可能原因 | 排查步骤 |
|------|---------|---------|
| 一直 blocked | arrive() 未被所有线程调用 | 检查指令调度逻辑 |
| is_complete 总 false | arrived_mask 未递增 | 检查 lane_id 是否正确 |
| PC 未更新 | set_thread_pc() 未调用 | 检查 complete 后代码路径 |
| 唤醒后错误指令 | pc_stack 未同步 | 检查 update_pc_stack() |

---

### 场景 7.5: Divergent Execution 路径不同步 (高级)

**触发条件**:
- test_nested_sync 或类似 divergent execution 测试卡住
- 日志显示不同的 lane 到达不同的 reconvergence_pc
- arrived 计数停滞 (如 arrived=16/32)

**根因分析**:
1. **PTX 语法**: `@%p1 bra $L__BB2_2` 创建两条执行路径
2. **Visitor 翻译**: `bar.sync` → `bar.warp.sync` 时使用固定 `reconvergence_pc = size + 1`
3. **问题**: 当 divergent 后再次汇合时，不同路径的下一条指令位置不同

**调试步骤**:
```bash
# 1. 查看每个 lane 到达的 PC
grep "arrived at bar.warp.sync" /tmp/test_output.log | 
  awk -F' '"{print
### 场景 8: 数值异常

**触发条件**: 测试报告 Numerical Exception

**自动行动**:

1. **选择配置**: `debug_config.ini`
2. **运行分析**: `grep -E "Numerical|NaN|Inf" test_output.log`
3. **分析重点**: 哪个指令产生异常值，输入操作数是否合法

---

   - 内存访问模式
   - 缓存命中率
   - 分支预测

---

## 自动化调试流程

### 标准调试流程（自动触发）

```
1. 接收用户问题
   ↓
2. 分析问题类型，选择调试配置
   ↓
3. 复制配置文件：cp configs/{selected}.ini ./ptx_debug.conf
   ↓
4. 运行程序收集日志
   ↓
5. 分析日志文件，提取关键信息
   ↓
6. 定位问题根源
   ↓
7. 生成修复方案
   ↓
8. 验证修复
```

### 快捷脚本使用

**技能会自动使用或指导用户使用快捷脚本**:

```bash
# 技能自动执行
./scripts/debug-run.sh {配置名} ./build/bin/{程序}

# 示例
./scripts/debug-run.sh memory ./build/bin/dummy-args
./scripts/debug-run.sh trace ./build/bin/RAY 512 512
```

---

## 日志分析技术

### 常用 grep 命令

```bash
# 查看错误
grep "ERROR\|FATAL" ptx_emu_*.log

# 查看特定组件
grep "\[mem\]\|\[instr\]\|\[exec\]" ptx_emu_*.log

# 查看特定指令
grep "st.global\|ld.global" ptx_emu_*.log

# 查看寄存器
grep "Register.*contains\|Commit.*%" ptx_emu_*.log

# 查看内存操作
grep "Memory allocated\|Memory freed" ptx_emu_*.log

# 时间线分析
grep "\[CLK:" ptx_emu_*.log | head -100
```

### 日志分析模式

**模式 1: 崩溃分析**
```bash
# 找到崩溃前的最后状态
tac ptx_emu_trace.log | grep -A 50 "最后的消息"
```

**模式 2: 内存泄漏**
```bash
# 对比分配和释放
grep "Memory allocated" ptx_emu.log > alloc.txt
grep "Memory freed" ptx_emu.log > free.txt
diff alloc.txt free.txt
```

**模式 3: 指令追踪**
```bash
# 查看指令执行序列
grep "PC\[" ptx_emu_instr_debug.log | awk '{print $5}'
```

---

## 调试输出位置

### 日志文件

| 配置文件 | 日志文件 | 用途 |
|---------|---------|------|
| `debug_config.ini` | `ptx_emu_debug.log` | 日常调试 |
| `verbose_trace_config.ini` | `ptx_emu_trace.log` | 详细跟踪 |
| `memory_debug_config.ini` | `ptx_emu_memory_debug.log` | 内存调试 |
| `instruction_debug_config.ini` | `ptx_emu_instr_debug.log` | 指令调试 |

### 快速查看

```bash
# 实时查看
tail -f ptx_emu_*.log

# 查看最新 N 行
tail -N ptx_emu_debug.log

# 分页查看
less -R ptx_emu_trace.log

# 搜索查看
grep "关键词" ptx_emu_debug.log | less
```

---

## 与其他技能协作

### 依赖技能

- **`cpp-debug`**: 提供通用 C++ 调试方法
- **`systematic-debugging`**: 提供系统化调试流程
- **`cuda-ptx`**: 提供 CUDA/PTX 专业知识

### ⚠️ 与 `ptx-grammar-modification` 的分工

**重要**: 本项目有专门的 PTX 语法修复技能 `ptx-grammar-modification`（位于 `docs/skills/`）。

**职责划分**:

| 问题类型 | 使用技能 | 说明 |
|---------|---------|------|
| **ANTLR 解析错误** | `ptx-grammar-modification` | 语法文件修改、重新生成解析器 |
| **语法文件修改** | `ptx-grammar-modification` | 修改 `.g4` 文件 |
| **`no viable alternative`** | `ptx-grammar-modification` | 语法规则修复 |
| **`mismatched input`** | `ptx-grammar-modification` | Token 定义修复 |
| **测试失败** | `ptx-debug` | 运行时错误、逻辑错误 |
| **内存问题** | `ptx-debug` | 内存访问、越界 |
| **指令错误** | `ptx-debug` | 指令执行结果不对 |
| **程序崩溃** | `ptx-debug` | segfault、core dump |
| **性能问题** | `ptx-debug` | 性能优化、benchmark |

**自动识别逻辑**:

```
当用户提到 "PTX 解析错误" 时:
  1. ptx-debug 技能触发
  2. 检查错误类型
  3. IF 错误是 ANTLR 解析错误 (no viable alternative, mismatched input):
       → 🛑 停止调试流程
       → 建议用户使用 ptx-grammar-modification 技能
       → 提示运行 ./tests/ptx/test_all_ptx.sh
  4. ELSE 错误是运行时错误:
       → 继续调试流程
       → 选择 verbose_trace 配置
       → 收集日志分析
```

### 协作场景

1. **内存泄漏**: 使用 `cpp-debug` 的 valgrind/ASan 方法
2. **解析错误**: 
   - 语法问题 → `ptx-grammar-modification`
   - 日志分析 → `ptx-debug` 辅助
3. **系统调试**: 遵循 `systematic-debugging` 的四阶段流程
4. **PTX ISA 查询**: 使用 `cuda-ptx` 的本地文档

---

## 验证标准

**调试完成的标志**:

- [ ] 已选择合适的调试配置
- [ ] 已收集充分的调试证据
- [ ] 已定位问题的根本原因
- [ ] 已生成修复方案
- [ ] 已验证修复有效
- [ ] 已添加防止复发的测试

---

## 示例使用

### 示例 1: 解析错误

**用户**: "PTX 解析失败，报告语法错误"

**技能自动行动**:
1. 识别为"PTX 解析错误"
2. 选择 `verbose_trace_config.ini`
3. 执行：
   ```bash
   cp configs/verbose_trace_config.ini ./ptx_debug.conf
   ./build/bin/dummy-args
   grep "parser\|lexer" ptx_emu_trace.log
   ```
4. 分析错误位置
5. 修复语法文件
6. 验证修复

### 示例 2: 测试失败

**用户**: "test_memory_manager 测试失败"

**技能自动行动**:
1. 识别为"测试失败"
2. 选择 `debug_config.ini`
3. 执行：
   ```bash
   cp configs/debug_config.ini ./ptx_debug.conf
   cd build && ctest -R test_memory_manager -V
   tail -100 ptx_emu_debug.log
   ```
4. 分析失败原因
5. 生成修复

### 示例 3: 性能优化

**用户**: "RAY benchmark 运行太慢"

**技能自动行动**:
1. 识别为"性能问题"
2. 选择 `perf_config.ini`
3. 执行：
   ```bash
   cp configs/perf_config.ini ./ptx_debug.conf
   time ./build/bin/RAY 512 512
   ```
4. 分析热点
5. 提出优化建议

---

## 诊断方法分类

> **核心原则**: 80% 的问题可以通过**不需要仿真**的快速诊断定位，只在必要时运行完整仿真

### 快速诊断 (<30s) - 不需要仿真

**适用场景**:
- PTX 语法解析错误
- Label 注册问题  
- CFG 分析警告
- 指令翻译问题 (如 bar.sync → bar.warp.sync)

**验证方法**:

| 方法 | 命令 | 耗时 | 适用 |
|------|------|------|------|
| **test-ptx 解析** | `./build/bin/test-ptx tests/ptx/test.ptx` | <1s | 语法、翻译验证 |
| **手动提取 PTX** | `cuobjdump -ptx binary > test.ptx` | <5s | 检查生成的 PTX |
| **最小化 PTX** | 手动创建简化 PTX 文件 | <10s | 隔离特定指令 |
| **CFG 日志** | 添加日志到 cfg_builder.cpp | <30s | 控制流分析 |

**快速验证流程**:

```bash
# 1. 从二进制提取 PTX
/usr/local/cuda/bin/cuobjdump -ptx ./build/bin/test_xxx 2>/dev/null | tail -n+12 > /tmp/test.ptx

# 2. 用 test-ptx 验证解析
./build/bin/test-ptx /tmp/test.ptx 2>&1 | grep -E "bar|bra|label|PASS|FAIL"

# 3. 查看特定指令翻译
./build/bin/test-ptx /tmp/test.ptx 2>&1 | grep -A2 "bar.warp.sync"
```

**成功案例**:
- ✅ PTX label 语法修复：通过 test-ptx 快速验证 33/33 语法测试
- ✅ bra 指令 predicate: 通过解析输出确认 target 正确注册
- ✅ CFG 空 target 警告：通过添加日志快速定位并修复

---

### 仿真诊断 (>30s) - 需要完整运行

**适用场景**:
- Barrier 同步卡住
- 线程调度问题
- 内存访问错误
- 寄存器值错误
- numerical exception

**验证方法**:

| 方法 | 命令 | 耗时 | 适用 |
|------|------|------|------|
| **完整单元测试** | `ctest -R test_xxx -V` | 30-120s | 功能回归 |
| **日志分析** | 启用 PTX_DEBUG | 30-60s | 执行跟踪 |
| **GDB 调试** | `gdb --args ./bin/test_xxx` | 5-10min | 崩溃定位 |
| **ASan/UBSan** | Debug 构建 + sanitizer | 2-5min | 内存错误 |

**调试流程**:

```bash
# 1. 启用详细日志
export PTX_LOG_LEVEL=debug
./build/bin/test_xxx 2>&1 | grep -E "bar|sync|blocked|complete"

# 2. 定位卡住点
./build/bin/test_xxx 2>&1 | tail -50

# 3. GDB 调试 (如果崩溃)
gdb --args ./build/bin/test_xxx
(gdb) run
(gdb) bt  # 崩溃时获取堆栈
```

**决策树**:

```
遇到问题
    │
    ├─ 解析错误/语法问题？
    │   └─→ 使用 test-ptx (1s) ← 优先
    │
    ├─ Label/CFG 警告？
    │   └─→ 添加日志，重解析 (10s) ← 优先
    │
    ├─ 指令执行卡住？
    │   └─→ 启用调试日志 (30s)
    │       └─ 仍无法定位？
    │           └─→ GDB 调试 (5min)
    │
    └─ 崩溃/SegFault？
        └─→ GDB + ASan (2min)
```

---

## Barrier 调试最佳实践

### 屏障调试流程 (类型 2 - 需要仿真)

```bash
# 1. 验证 PTX 解析 (类型 1 - 快速)
./build/bin/test-ptx test.ptx | grep "bar.warp.sync"

# 2. 启用 barrier 日志
grep -r "PTX_DEBUG" src/ptxsim/instructions/barrier.cpp
# 确保相关日志启用

# 3. 运行测试并抓取日志
./build/bin/test_xxx 2>&1 | grep -E "arrived|complete|blocked|released"

# 4. 分析关键指标
# - participation_mask 是否正确？
# - arrived_mask 是否递增？
# - is_complete() 何时返回 true？
# - set_thread_pc() 是否被调用？
# - update_pc_stack() 是否同步？
```

**关键日志模式**:
```
bar.warp.sync: Initialized wbar[0] with mask=0xFFFFFFFF, reconvergence_pc=N
Lane X arrived at bar.warp.sync (mask=0xFFFFFFFF, pc=N)
bar.warp.sync: Barrier complete, releasing N threads to PC=N
Lane X blocked at bar.warp.sync (arrived=N/32)
```

**预期行为**:
1. 初始化 wbar，participation_mask = 0xFFFFFFFF
2. 每个 lane 调用 arrive()，arrived_mask 递增
3. 当 arrived_mask == participation_mask 时，barrier complete
4. 所有参与线程 PC 更新为 reconvergence_pc
5. is_blocked 清除，status 设为 Active

**常见问题诊断表**:

| 现象 | 可能原因 | 排查步骤 |
|------|---------|------|
| 一直 blocked | arrive() 未被调用 | 检查指令调度器是否调度过 Blocked 线程 |
| is_complete 总 false | arrived_mask 未递增 | 检查 lane_id 是否正确传入 arrive() |
| PC 未更新 | set_thread_pc() 未被调用 | 检查 is_complete() 返回后的代码路径 |
| 唤醒后执行错误指令 | pc_stack 未同步 | 检查 update_pc_stack() 调用点 |
| participation_mask=0 | 指令翻译错误 | 检查 visitor 中 barrier operand 提取 |

**关键代码位置**:
```
src/ptx_parser/ptx_visitor_barrier.cpp  - bar.sync → bar.warp.sync 翻译
src/ptxsim/instructions/barrier.cpp     - Wbar 实现与 arrive()/is_complete()
include/ptxsim/wbar.h                    - Wbar 数据结构定义
```

---

##  经验教训

1. **日志文件大小**: trace 级别可能产生数百 MB 日志
2. **性能影响**: 详细日志会降低 10-100 倍性能
3. **及时清理**: 定期清理旧日志文件
4. **配置恢复**: 调试后记得恢复到 release 配置

---

## 更新记录

- **2026-03-23**: 初始版本，包含 6 种调试场景的自动化方法
- 后续更新：根据新场景添加调试配置和方法

### 场景 7.5: Divergent Execution 路径不同步 (高级)

**触发条件**:
- test_nested_sync 或类似 divergent execution 测试卡住
- 日志显示不同的 lane 到达不同的 reconvergence_pc
- arrived 计数停滞 (如 arrived=16/32)

**根因分析**:
1. **PTX 语法**: `@%p1 bra $L__BB2_2` 创建两条执行路径
2. **Visitor 翻译**: `bar.sync` → `bar.warp.sync` 时使用固定 `reconvergence_pc = size + 1`
3. **问题**: 当 divergent 后再次汇合时，不同路径的下一条指令位置不同

**调试步骤**:
```bash
# 1. 查看每个 lane 到达的 PC
grep "arrived at bar.warp.sync" /tmp/test_output.log | \
  grep -oP 'pc=\K\d+' | sort -n | uniq -c

# 2. 检查是否所有 lane 都到达同一个 barrier
grep "arrived=" /tmp/test_output.log | \
  grep -oP 'arrived=\K\d+/\d+' | sort | uniq -c

# 3. 对比 label 注册位置
grep "Registering label" /tmp/test_output.log
```

**实际案例 (test_nested_sync)**:
```
Lane 13 arrived pc=26  # 在分支路径中
Lane 14 arrived pc=12  # 直接跳转路径
arrived=16/32          # 停滞
```

**问题根因**: threads 16-31 跳转到 L__BB2_2 (PC=23)，然后执行第二个 barrier (PC=25)，但实际到达的 reconvergence_pc=26 不正确。

**修复建议**:
- 检查 visitor 中的 barrier PC 计算逻辑
- 确保 divergent 路径正确汇合到同一个 label
- 使用 CFG analysis 验证 reconvergence 点
- 查看 `ptx_visitor_barrier.cpp` 中 `next_pc` 的计算

---

