# 回归调试指南

> **版本**: v1.0  
> **创建日期**: 2026-05-04  
> **适用场景**: 重构或功能修改后出现测试回归时的调试方法论

---

## 目录

1. [案例：ThreadContext PC 重构回归](#1-案例threadcontext-pc-重构回归)
2. [教训总结](#2-教训总结)
3. [调试检查清单](#3-调试检查清单)
4. [常见陷阱](#4-常见陷阱)
5. [参考命令](#5-参考命令)

---

## 1. 案例：ThreadContext PC 重构回归

### 1.1 背景

Commit `92f7585` 将 `ThreadContext` 的 `pc`/`next_pc` 字段移除，改为通过 `warp_context_` 访问 `WarpState`。这是典型的**状态归属重构**——将原本属于 ThreadContext 的状态委托到共享的 WarpState 中。

### 1.2 表象

`test_ptx_ld_st` 中 2 个 shared memory 测试失败（`result == 0`，期望 42/1），其余 12 个单线程 (`<<<1,1>>>`) 测试通过。

### 1.3 根因

**`PipelineHandler::ExecPipe`**（instruction_base.cpp:102）在屏障处理器运行后调用：
```cpp
context->set_next_pc(context->get_pc() + 1);
```

**旧代码**中 `context->pc` 是 ThreadContext 的成员字段，屏障处理器调用 `set_thread_pc()` 修改的是 `WarpState`，不会影响 ThreadContext.pc。所以 `pc + 1 = barrier_pc + 1` → **正确**（指向 `mov 42` 指令）。

**新代码**中 `context->get_pc()` 从 `WarpState` 读取，而屏障处理器通过 `set_thread_pc(i, reconvergence_pc)` **已经修改了 WarpState 中的 PC**。所以 `get_pc() + 1 = reconvergence_pc + 1 = barrier_pc + 2` → **错误**（跳过了 `mov 42` 指令，直接执行 `st.global`，写入未初始化的 0）。

### 1.4 修复

```cpp
// instruction_base.cpp - PipelineHandler::ExecPipe
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int saved_pc = context->get_pc();  // 在处理器执行前保存PC
    // ... prepare → execute → commit ...
    context->set_next_pc(saved_pc + 1);  // 使用保存的PC
}
```

**改动量**：1 个文件，2 行。

---

## 2. 教训总结

### 2.1 核心教训：状态归属变更的隐形副作用

| 旧架构 | 新架构 | 副作用 |
|--------|--------|--------|
| `ThreadContext::pc` 是本地字段 | `get_pc()` 读 WarpState | 屏障处理器修改 WarpState 后，`get_pc()` 返回值已变化 |
| 屏障处理器只修改其他线程的 PC | 屏障处理器修改**所有**线程的 PC（包括当前线程） | `PipelineHandler` 读取的 PC 已被屏障处理器修改 |
| `next_pc = pc + 1` 使用编译器/指令自身 PC | `next_pc = get_pc() + 1` 使用**被修改后**的 PC | 多加了 1 |

**教训**：当把局部状态迁移到共享状态时，必须检查**所有**对这个状态的读操作，确认读到的值是否已经被并发逻辑修改。

### 2.2 调试马拉松中的时间黑洞

| 耗时活动 | 耗时（估算） | 是否有效 | 教训 |
|---------|------------|---------|------|
| Oracle 子代理分析 | ~45 min | ❌ 完全错误 | 大模型擅长推理，但缺乏对具体代码变更的精确理解。不要依赖其结论。 |
| 修复 `sync_from_warp_state` 别名问题 | ~30 min | ❌ 无关 | 虽然发现了真实 bug，但与本次回归无关。不要恋战无关问题。 |
| 修改 `set_pc()` 不设置 `next_pc` | ~20 min | ❌ 无关 | 未经根因确认就修改代码。先找根因，再动手。 |
| 对比新旧代码 diff 逐行追踪 | ~5 min | ✅ 最终解决 | **最高效的方法**：理解每一次访问器调用在新旧代码中的差异。 |
| 添加日志并分析执行流 | ~15 min | ✅ 提供关键线索 | 日志揭示了屏障后无执行、CFG 统计等关键信息。 |

### 2.3 方法论：对比追踪法

当面对 "重构后回归" 问题时，最有效的方法是：

1. **精确定位回归 commit**（`git bisect` 或二分测试）
2. **阅读 commit diff**，理解**每个访问器/方法的语义变化**
3. **逐个对比**新旧代码中关键路径的行为差异：
   ```
   OLD: pc = next_pc           → 读 ThreadContext.next_pc，写 ThreadContext.pc
   NEW: set_pc(get_next_pc())  → 读 WarpState.next_pc，写 WarpState.pc
   ```
4. **查找所有修改同一状态的路径**（如屏障处理器中的 `set_thread_pc`）
5. **推理并发修改场景**下的最终值

### 2.4 诊断信号识别

| 信号 | 含义 | 本文案例 |
|------|------|---------|
| 所有单线程测试通过，多线程测试失败 | 问题在同步/并发 | ✅ `<<<1,1>>>` 通过，`<<<1,32>>>` 失败 |
| 测试结果为 0（未初始化值） | 写入指令被跳过 | ✅ `mov 42` 被跳过，`st.global` 写入 0 |
| CFG 分析正确但执行结果错误 | 问题在执行阶段，非解析阶段 | ✅ CFG 检测到 1 个屏障 |
| 屏障日志显示 "PC=X -> X"（无变化） | reconvergence_pc 等于当前 PC → 死循环或跳转错误 | ✅ `Released lane: PC=5 -> 5`，但随后 `next_pc` 被覆写为 6 |

---

## 3. 调试检查清单

按优先级排序，遇到回归时逐一检查：

- [ ] **1. 定位回归 commit**：`git bisect` 或 `git checkout` + 构建测试
- [ ] **2. 查看 commit diff**：`git diff base..regression --stat`，然后逐个文件细读
- [ ] **3. 识别状态归属变更**：是否有字段从 A 移到 B？是否有新的间接访问？
- [ ] **4. 追踪关键路径**：选择一条失败测试的执行路径，逐行对比新旧代码
- [ ] **5. 查找并发修改**：哪些代码路径会修改同一个状态？
- [ ] **6. 添加最小诊断日志**：在关键读写点添加 `PTX_WARN_EMU`
- [ ] **7. 确认修复**：只改最少代码，运行全部相关测试
- [ ] **8. 回归测试**：确保修复不引入新问题

---

## 4. 常见陷阱

### 4.1 别名引用

```cpp
// 危险：thread_state 是 warp_state 内部数组元素的引用
ptxsim::ThreadState& thread_state = warp_state.threads[lane_id];
set_pc(thread_state.pc);       // 写回同一位置，可能触发意外行为
set_next_pc(thread_state.next_pc);
```

**规则**：当 `set_xxx()` 修改 `warp_state` 时，上面代码中的 `thread_state` 引用已被"污染"。

### 4.2 管道处理器覆盖 PC

`PipelineHandler::ExecPipe` 在所有操作完成后无条件执行 `set_next_pc(get_pc() + 1)`。如果指令处理器（如屏障处理器）已经显式设置了 PC/NEXT_PC，此行会覆盖。

**规则**：管道处理器的 `ExecPipe` 应使用执行前的 PC 计算默认 next_pc，而非执行后的。

### 4.3 测试生成 PTX 与运行时 PTX 不一致

测试编译时 `.ptx` 文件可能有多份（多目标架构），运行时提取的 PTX 可能与 `.ptx` 文件不同。

```bash
# 查看二进制中有多少 PTX 段
cuobjdump -lptx ./build/bin/tests/test_xxx
```

**规则**：调试时检查运行时实际提取的 PTX（`extracted.ptx` 文件），而非编译输出的 `.ptx` 文件。

### 4.4 大模型代理分析偏差

大模型（Oracle 等）擅长推理但缺乏对具体 commit 变更细节的精确理解。它可能：
- 提出看似合理但代码中不存在的假设（如引用不存在的 `ptx_interpreter.cpp`）
- 忽略关键的微小差异（如 `get_pc()` 的语义变化）

**规则**：将大模型作为辅助工具，最终判断必须基于代码阅读和实际测试。

---

## 5. 参考命令

### Git 调试

```bash
# 二分查找回归 commit
git bisect start
git bisect bad HEAD
git bisect good <known-good-commit>
# 每个步骤：cmake --build build && ctest ...

# 查看 commit diff
git diff <good>..<bad> --stat
git diff <good>..<bad> -- src/ptxsim/

# 查看旧版本某个文件
git show <good>:src/ptxsim/core/thread_context.cpp | head -120
```

### 构建与测试

```bash
# 构建特定测试目标
cmake --build build --target test_ptx_ld_st -- -j$(nproc)

# 运行单个测试（verbose）
cd build && ctest -R test_ptx_ld_st -V 2>&1 | tee /tmp/test.log

# 查看 PTX 内容
cuobjdump -xptx <ptx_section_name> <binary>

# 搜索日志中的关键模式
strings /tmp/test.log | grep "Released lane\|stmt_type=79\|CFG analysis"
```

### 添加临时诊断日志

```cpp
// 使用 WARN 级别确保在默认日志配置下可见
PTX_WARN_EMU("set_pc: lane=%d, old_pc=%d, new_pc=%d", lane_id_, old, new);
```

---

**维护**: 持续更新  
**最后更新**: 2026-05-04  
**作者**: Sisyphus（基于 ThreadContext PC 重构回归调试实战总结）
