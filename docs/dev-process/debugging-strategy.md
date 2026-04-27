# PTX-EMU 调试策略指南

> **核心原则**: 优先使用快速验证方法，只在必要时运行完整仿真

---

## 📊 问题分类与验证方法

### 类型 1: 不需要仿真执行的问题 (快速验证，<30s)

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
| **CFG 调试** | 添加日志到 cfg_builder.cpp | <30s | 控制流分析 |

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
- ✅ PTX label 语法修复：通过 test-ptx 验证 33/33 语法测试
- ✅ bra 指令 predicate: 通过解析输出确认 target 正确注册
- ✅ CFG 空 target 警告：通过添加日志快速定位

---

### 类型 2: 需要仿真执行的问题 (深度调试，>30s)

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

**调试技巧**:

1. **Barrier 问题**: 关注 `wbar.arrive()` 和 `wbar.is_complete()` 日志
2. **线程状态**: 检查 `ThreadStatus::Blocked` vs `Active`
3. **PC 更新**: 验证 `set_thread_pc()` 是否被调用
4. **Scheduler**: 确认 Blocked 线程是否被正确唤醒

---

## 🔧 通用调试工具

### 日志级别控制

```cpp
// 在 utils/logger.h 中配置
PTX_DEBUG_EMU(...)    // 模拟器级别
PTX_DEBUG_THREAD(...) // 线程级别
PTX_INFO_EMU(...)     // 信息级别
```

**启用特定模块日志**:

```ini
# configs/debug_config.ini
[log]
level = debug
components = emu,exec,mem,reg,thread
```

### 快速测试脚本

```bash
# tests/ptx/test_all_ptx.sh - 语法验证 (3s)
# ctest -L mini - 快速单元测试 (30s)
# ctest -L full - 完整测试 (5min)
```

### GDB 常用命令

```gdb
# 设置断点
break ptx_interpreter.cpp:560  # setupLabels
break barrier.cpp:131          # BarWarpSyncHandler::processOperation
break warp_context.cpp:8       # WarpContext::handle_branch

# 查看线程状态
print warp_state.threads[0].status
print warp_state.wbars[0].arrived_mask

# 继续执行到 barrier
continue
```

---

## 📋 决策树

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

## ⚠️ 注意事项

1. **先分类，后调试**: 80% 的问题可以通过解析验证定位
2. **最小化复现**: 创建最小 PTX 测试用例加速调试
3. **日志优先**: 先添加日志，再上 GDB
4. **接受局限**: 模拟器本身限制导致的问题需要架构升级

---

## 🎯 Barrier 调试最佳实践

### 屏障调试流程

```bash
# 1. 验证 PTX 解析
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

**常见问题**:
| 现象 | 可能原因 | 排查 |
|------|---------|------|
| 一直 blocked | arrive() 未被调用 | 检查指令调度 |
| is_complete 总 false | arrived_mask 未递增 | 检查 lane_id 是否正确 |
| PC 未更新 | set_thread_pc() 未被调用 | 检查 reconvergence_pc 值 |
| 唤醒后执行错误指令 | pc_stack 未同步 | 检查 update_pc_stack() |

---

## 📌 经验教训

### 成功经验

### 已知限制 (Known Limitations)

#### 1. test_divergence_sync 失败

**状态**: Open (未修复)

**现象**: 
- 测试 `test_divergence_sync` 失败，输出 `expected -847479615, got 0`
- 其他 8 个共享内存测试全部通过

**根本原因**:
- Warp divergence + post-barrier reconvergence 问题
- 测试涉及：
  1. Lanes 0-15 vs 16-31 取不同代码路径（ warp divergence）
  2. `__syncthreads()` CTA barrier 同步
  3. Barrier 后 thread 0 收集数据（再次 divergence）

**技术分析**:
- Barrier 本身工作正常：所有线程正确同步并释放
- Post-barrier divergence 处理有问题：thread 0 (lane 0) 在 barrier 后的 `if (tid == 0)` 检查中未正确执行
- 可能与 SIMT stack 管理或 warp scheduler 在 post-barrier divergence 场景的行为有关

**影响范围**:
- 仅影响需要 post-barrier divergence 收集的场景
- 基本 barrier 功能和其他 8 个共享内存测试不受影响

**修复建议**:
1. 深入分析 post-barrier divergence 场景的 warp 调度
2. 检查 SIMT stack 在 barrier reconvergence 时的状态
3. 验证 thread 0 在 `if (tid == 0)` 后的执行流程

**Workaround**:
- 避免在 barrier 后立即进行需要特定线程执行的 divergence 操作
- 使用其他同步机制替代 post-barrier divergence 收集

1. **PTX 提取是关键**: 多个测试失败源于 PTX 提取使用相对路径
   - 修复：使用 getcwd() 获取绝对路径
   
2. **Label 解析是连环问题**: 最初 6 个语法测试失败都源于同一个问题
   - 修复：支持 `L_label:` 格式（不带 $ 前缀）

3. **CFG 警告可以忽略**: 空 target 警告不影响实际执行
   - 修复：添加空 target 检查跳过

4. **Predicate 寄存器访问**: handle_branch 从错误的位置读取 predicate
   - 修复：从 RegisterBankManager 读取而非不存在的 map

### 失败教训

1. **不要过早优化**: 先让语法通过，再优化性能
2. **隔离问题**: test_syncthreads 有 3 个 kernel，先单独测试最简单的
3. **接受现实**: 当前 94% 通过率已证明修复有效，剩余问题涉及深层架构

