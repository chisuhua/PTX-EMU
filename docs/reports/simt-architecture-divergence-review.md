# PTX-EMU SIMT 架构综述: 控制流发散从软件到硬件

> **文档版本**: v1.0  
> **最后更新**: 2026-04-14  
> **状态**: 持续更新中  
> **关联问题**: test_syncthreads Test 3 修复计划

---

## 目录

- [1. 概述](#1-概述)
- [2. SIMT 执行模型](#2-simt-执行模型)
- [3. 核心数据结构](#3-核心数据结构)
- [4. 控制流发散处理](#4-控制流发散处理)
- [5. 屏障同步机制](#5-屏障同步机制)
- [6. PC 管理架构](#6-pc-管理架构)
- [7. 静态 CFG 分析](#7-静态-cfg-分析)
- [8. 调度层设计](#8-调度层设计)
- [9. 已修复的 Bug 和架构改进](#9-已修复的-bug-和架构改进)
- [10. Test 3 问题分析路径](#10-test-3-问题分析路径)
- [11. 下一步计划](#11-下一步计划)

---

## 1. 概述

PTX-EMU 是一个基于 CPU 的 PTX (Parallel Thread Execution) 指令集模拟器, 用于在 NVIDIA GPU 硬件上执行 PTX 代码的离线分析和验证。其核心是 **SIMT (Single Instruction, Multiple Thread)** 执行模型, 该模型模仿 NVIDIA GPU 硬件的 warp 级并行执行机制。

**架构范围**: 832 行核心仿真代码, 覆盖以下层次:
- 静态分析: CFG 构建和后支配点计算
- 执行层: GPUContext → SMContext → CTAContext → WarpContext → ThreadContext
- 指令处理: 分支、屏障、算术、访存等 50+ 指令类型
- 同步: CTA 级和 Warp 级双重屏障

**本文档目标**: 系统阐述控制流发散 (Divergence) 从软件 (PTX 语法、CFG 分析) 到硬件 (SIMT 栈、执行掩码、屏障同步) 的完整实现路径, 并为 test_syncthreads Test 3 的持续修复提供上下文。

---

## 2. SIMT 执行模型

### 2.1 基本概念

SIMT 是 NVIDIA GPU 的执行范式:
- **Warp**: 32 个线程组成一个 warp, 共享一个指令指针 (PC)
- **发散 (Divergence)**: 当 warp 中部分线程执行分支, 部分不执行时, warp "拆分"
- **收敛 (Convergence)**: 分散的线程在执行流中重新汇合
- **屏障 (Barrier)**: 显式同步点, 所有参与的线程必须到达后才能继续

### 2.2 执行层次

```
┌─────────────────────────────────────────────────┐
│ GPUContext (GPU 级)                              │
│   - 管理多个 SM                                    │
│   - 处理 kernel 启动请求                          │
│                                                   │
│   └─> SMContext (Streaming Multiprocessor)        │
│       - Warp 调度 (Round-Robin/Greedy)             │
│       - 发散指令分发 (get_lanes_by_pc)             │
│       - CTA 级屏障同步                             │
│                                                    │
│       └─> CTAContext (Cooperative Thread Array)    │
│           - 管理 block 内的所有线程                 │
│           - 创建和管理 w                        │
│                                                    │
│           └─> WarpContext (32 线程 warp)           │
│               - handle_branch() 发散检测           │
│               - get_lanes_by_pc() 线程分组         │
│               - SIMT 栈管理                        │
│               - 屏障同步 (Wbar)                    │
│                                                    │
│               └─> ThreadContext (每线程)           │
│                   - pc, next_pc                    │
│                   - sync_from/to_warp_state()      │
│                   - _execute_once() 指令执行        │
└─────────────────────────────────────────────────┘
```

### 2.3 核心执行循环

每个 GPU 时钟周期, 执行以下流程:

```cpp
// GPUContext::exe_once()
for (每个 SM) {
    sm->exe_once();  // 调度一个 warp, 执行一条 (或多条分发的) 指令
}

// SMContext::exe_once()
WarpContext* warp = warp_scheduler->schedule_next();  // 选择下一个 warp
auto lanes_by_pc = warp->get_lanes_by_pc();           // 按 PC 分组线程

if (lanes_by_pc.size() == 1) {
    // 非发散: 所有线程在同一 PC
    execute_warp_instruction(stmt, target_pc);
} else {
    // 发散: 每个 PC 组独立执行
    for (const auto& [pc, lanes] : lanes_by_pc) {
        execute_warp_instruction(stmt, pc);
    }
}
```

---

## 3. 核心数据结构

### 3.1 ThreadState (`include/ptxsim/thread_state.h`)

每线程状态, 是 SIMT 架构的**基础数据结构**:

```cpp
struct ThreadState {
    uint32_t pc = 0;          // 当前程序计数器 (核心: 每线程独立 PC)
    uint32_t next_pc = 0;     // 下一条指令 PC
    ThreadStatus status;      // Active / Blocked / Exited / Yielded
    bool is_exited = false;   // 线程是否已永久退出
    bool is_blocked = false;  // 是否在屏障等待
    bool is_active = true;    // 是否可调度
    
    bool is_schedulable() const {
        return is_active && !is_exited && !is_blocked && 
               (status == ThreadStatus::Active);
    }
};
```

**关键设计决策**: 每线程独立 PC 允许发散线程独立推进, 这是 SIMT 正确性的核心。

### 3.2 SIMTStackEntry (`include/ptxsim/simt_stack.h`)

SIMT 栈条目, 记录分支上下文:

```cpp
struct SIMTStackEntry {
    int branch_pc;              // 分支发生的 PC
    int reconvergence_pc;       // 线程汇合的 PC
    uint32_t active_mask;       // 进入此分支的线程掩码
    uint32_t return_mask;       // 原始执行掩码
    int return_pc;              // 返回 PC
};
```

**栈操作**: `push()` 记录分支, `pop()` 在汇合时释放, `check_reconvergence()` 检测所有线程是否到达汇合点。

### 3.3 Wbar (`include/ptxsim/wbar.h`)

Warp 级屏障数据结构:

```cpp
struct Wbar {
    uint32_t participation_mask = 0;  // 参与屏障的线程掩码
    uint32_t arrived_mask = 0;        // 已到达的线程掩码
    int reconvergence_pc = -1;        // 屏障后继续的 PC
    bool is_initialized = false;
    int expected_count = 0;
    std::vector<std::pair<int, int>> pre_barrier_stores;  // 屏障前存储记录
};
```

### 3.4 WarpState (`include/ptxsim/warp_state.h`)

Warp 级状态容器:

```cpp
struct WarpState {
    std::array<ThreadState, 32> threads;  // 32 个线程的状态
    uint32_t exec_mask = 0xFFFFFFFF;      // 当前执行掩码
    std::array<Wbar, 4> wbars;            // 4 个屏障寄存器
    int current_wbar_id = -1;             // 当前活跃的屏障 ID
    uint32_t warp_pc = 0;                 // Warp 级 PC (兼容旧代码)
    std::array<int, 16> pc_stack;         // PC 栈
    int pc_stack_depth = 0;
};
```

### 3.5 ThreadContext (`include/ptxsim/thread_context.h`)

线程上下, 是执行的基本单元:

```cpp
class ThreadContext {
public:
    int pc = 0;                          // 当前 PC (本地缓存)
    int next_pc = 0;                     // 下一条指令 PC
    EXE_STATE state = RUN;               // 执行状态 (RUN/BAR_SYNC/EXIT)
    int lane_id_ = -1;                   // 在 warp 中的 lane ID
    WarpContext* warp_context_ = nullptr;
    
    void sync_from_warp_state();  // Warp → Thread
    void sync_to_warp_state();    // Thread → Warp
    void _execute_once();         // 执行当前指令
};
```

---

## 4. 控制流发散处理

### 4.1 发散检测: `WarpContext::handle_branch()`

```cpp
void WarpContext::handle_branch(
    const std::string& predicate,
    bool predicate_negated,
    int target_pc,
    int reconvergence_pc) 
{
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    // 估每个活跃线程的谓词条件
    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active) continue;
        // ... 谓词求值 ...
        if (should_branch) taken_mask |= (1u << i);
        else not_taken_mask |= (1u << i);
    }
    
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    // ...
}
```

**关键判断**: `is_divergent = (taken_mask != 0) && (not_taken_mask != 0)` — 部分线程分支, 部分不分支。

### 4.2 发散路径: SIMT 栈推送

```cpp
if (is_divergent) {
    // 1. 推送 SIMT 栈条目
    ptxsim::SIMTStackEntry entry;
    entry.branch_pc = pc;
    entry.reconvergence_pc = reconvergence_pc;
    entry.active_mask = taken_mask;
    entry.return_mask = warp_state.exec_mask;
    entry.return_pc = reconvergence_pc;
    simt_stack.push(entry);
    
    // 2. 设置发散 PC
    for (int i = 0; i < 32; i++) {
        if (taken_mask & (1u << i)) {
            warp_state.threads[i].pc = target_pc;
            warp_state.threads[i].next_pc = target_pc;
        } else if (not_taken_mask & (1u << i)) {
            warp_state.threads[i].pc = pc + 1;
            warp_state.threads[i].next_pc = pc + 1;
        }
    }
    
    // 3. 更新执行掩码 — 只有 taken 的线程继续执行
    warp_state.exec_mask = taken_mask;
}
```

### 4.3 非发散路径: 统一推进

```cpp
else {
    int next_pc = (taken_mask != 0) ? target_pc : pc + 1;
    
    // 更新 WarpState
    for (int i = 0; i < 32; i++) {
        if (warp_state.threads[i].is_active) {
            warp_state.threads[i].pc = next_pc;
            warp_state.threads[i].next_pc = next_pc;
        }
    }
    
    // 同步 ThreadContext — 统一 PC 管理接口
    advance_all_threads(next_pc);
}
```

### 4.4 收敛检测: SIMT 栈弹出

```cpp
// simt_stack.cpp
bool SIMTStack::check_reconvergence(
    const std::array<ThreadState, 32>& threads) 
{
    if (entries_.empty()) return true;
    
    SIMTStackEntry& top = entries_.back();
    if (top.is_converged(threads)) {
        entries_.pop_back();  // 弹出已收敛的栈条目
        return true;
    }
    return false;
}

bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        if (return_mask & (1u << i)) {
            if ((int)threads[i].pc != reconvergence_pc) {
                return false;  // 还有线程未到达汇合点
            }
        }
    }
    return true;
}
```

### 4.5 线程分组: `get_lanes_by_pc()`

```cpp
std::map<int, std::vector<int>> WarpContext::get_lanes_by_pc() const {
    std::map<int, std::vector<int>> pc_to_lanes;
    
    for (int lane = 0; lane < WARP_SIZE; lane++) {
        if (lane < (int)threads.size() && threads[lane] != nullptr &&
            warp_state.threads[lane].is_active && 
            !warp_state.threads[lane].is_exited) {
            int pc = warp_state.threads[lane].pc;  // 使用 warp_state 作为权威源
            pc_to_lanes[pc].push_back(lane);
        }
    }
    
    return pc_to_lanes;  // { PC12: [0,1,2,...], PC15: [16,17,...] }
}
```

**关键决策**: 使用 `warp_state.threads[lane].pc` 而非 `threads[lane]->pc`, 确保单一 PC 权威源。

### 4.6 分支指令处理器: `BraHandler::executeBranch()`

```cpp
// src/ptxsim/instructions/control.cpp
void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    WarpContext* warp_ctx = context->warp_context_;
    
    // 1. 解析目标 PC
    int target_pc = -1;
    auto it = context->label2pc.find(instr.target);
    if (it != context->label2pc.end()) {
        target_pc = it->second;
    } else {
        target_pc = context->pc + 1;  // fallback
    }
    
    // 2. 委托给 WarpContext 处理发散
    warp_ctx->handle_branch(
        instr.predicate,
        instr.predicate_negated,
        target_pc,
        instr.reconvergence_pc
    );
    
    // 3. 同步当前线程 PC
    context->pc = warp_ctx->get_thread_pc(context->lane_id_);
}
```

---

## 5. 屏障同步机制

### 5.1 双重屏障体系

| 类型 | PTX 指令 | 作用域 | 处理器 |
|------|----------|--------|--------|
| **CTA 级** | `bar.sync bar_id` | block 内所有线程 | `SMContext::synchronize_barrier()` + `BarHandler` |
| **Warp 级** | `bar.warp.sync mask` | 参与掩码指定的线程 | `BarWarpSyncHandler` |

### 5.2 屏障参数

CTA 级 `bar.sync` 在解析时根据 CTA 大小转换为 `bar.warp.sync`:

```cpp
// ptx_visitor_barrier.cpp
if (openum == S_BAR && isWarpLevelBarrier(currentKernel)) {
    // 对于单 warp CTA (线程数 ≤ 32), 转换为 warp 级屏障
    stmtCtx.type = S_BAR_WARP_SYNC;
    // 参与掩码 = (1 << 线程数) - 1
    // CFG 分析会覆盖为正确的值
}
```

### 5.3 CFG 分析中的屏障处理

```cpp
// src/cudart/ptx_interpreter.cpp
else if (stmt.type == S_BAR_WARP_SYNC) {
    auto &barrier = std::get<BarWarpSyncInstr>(
        kernelContext->kernelStatements[i].data);
    if (barrier.operands.size() >= 2) {
        // 屏障后总是执行下一条指令
        barrier.operands[0] = OperandContext{ImmOperand{...参与掩码...}};
        barrier.operands[1] = OperandContext{ImmOperand{std::to_string(i + 1)}};
    }
}
```

### 5.4 屏障完成流程

```
1. 线程到达 bar.warp.sync
   └─> Wbar::arrive(lane_id)
       └─> arrived_mask |= (1u << lane_id)

2. 检查是否全部到达
   └─> wbar.is_complete()
       └─> (arrived_mask & participation_mask) == participation_mask

3. 如果完成:
   ├─> 所有已到达线程: pc = reconvergence_pc
   ├─> 清除 is_blocked = false
   ├─> 设置 status = Active
   └─> wbar.reset()

4. 如果未完成:
   └─> is_blocked = true, status = Blocked
```

### 5.5 屏障释放 (最新修复)

```cpp
// barrier.cpp — 只更新已到达的线程
if (wbar.is_complete()) {
    for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
        if ((wbar.arrived_mask & (1u << i)) && warp_state.threads[i].is_active) {
            warp_ctx->set_thread_pc(i, reconvergence_pc);
            warp_ctx->update_pc_stack(i, reconvergence_pc);
            warp_state.threads[i].is_blocked = false;
            warp_state.threads[i].status = ptxsim::ThreadStatus::Active;
        }
    }
    wbar.reset();
    warp_state.current_wbar_id = -1;
}
```

**关键修复**: 使用 `arrived_mask` 而非 `participation_mask`, 防止更新未执行屏障的线程。

---

## 6. PC 管理架构

### 6.1 问题背景: 双 PC 不一致

**旧架构缺陷**: `ThreadContext::pc` 和 `warp_state.threads[i].pc` 是两个独立的存储, 存在时序不一致:
- 调度器通过 `get_lanes_by_pc()` 读取 `ThreadContext::pc`
- 分支/屏障处理器写入 `warp_state.threads[i].pc`
- `sync_from_warp_state()` 在调度决策**之后**调用

### 6.2 解决方案: 统一 PC 权威源

**架构决策**: `warp_state.threads[i].pc` 是唯一权威源, `ThreadContext::pc` 仅作为执行本地缓存。

### 6.3 统一 PC 更新接口

```cpp
// warp_context.h/cpp — 新 API
void advance_thread_pc(int lane_id, int new_pc) {
    if (lane_id < 0 || lane_id >= WARP_SIZE) return;
    // 同时更新 warp_state 和 ThreadContext
    warp_state.threads[lane_id].pc = new_pc;
    warp_state.threads[lane_id].next_pc = new_pc;
    if (lane_id < (int)threads.size() && threads[lane_id]) {
        threads[lane_id]->pc = new_pc;
        threads[lane_id]->next_pc = new_pc;
    }
}

void advance_all_threads(int new_pc) {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active) continue;
        warp_state.threads[i].pc = new_pc;
        warp_state.threads[i].next_pc = new_pc;
        if (i < (int)threads.size() && threads[i]) {
            threads[i]->pc = new_pc;
            threads[i]->next_pc = new_pc;
        }
    }
}
```

### 6.4 PC 同步保护

```cpp
// thread_context.cpp: sync_to_warp_state()
void ThreadContext::sync_to_warp_state() {
    // 保护屏障设置的 PC 不被回退
    if (thread_state.pc > static_cast<uint32_t>(pc)) {
        // 屏障完成已推进 PC — 保留它
    } else {
        thread_state.pc = pc;
    }
    thread_state.next_pc = next_pc;
    // ...
}
```

### 6.5 调度器指令获取修复

```cpp
// sm_context.cpp: exe_once()
if (target_pc >= 0 && target_pc < static_cast<int>(sample_thread->statements_size())) {
    StatementContext* stmt = sample_thread->get_statement_at(target_pc);
    // 使用 target_pc 获取正确的指令, 而非 get_current_statement()
}
```

### 6.6 PC 流图

```
正常执行:
  ThreadContext::_execute_once()
    ├─> next_pc = pc + 1
    ├─> handler.ExecPipe() (可能覆盖 next_pc)
    └─> pc = next_pc
    └─> sync_to_warp_state()

分支执行:
  BraHandler::executeBranch()
    └─> handle_branch()
        ├─> (发散) 设置 warp_state.threads[].pc = target_pc 或 pc+1
        └─> (非发散) advance_all_threads(next_pc)
    └─> context->pc = get_thread_pc()

屏障完成:
  BarWarpSyncHandler (最后一个到达的线程)
    └─> wbar.is_complete() → true
        └─> for each arrived lane: set_thread_pc(i, reconvergence_pc)
```

---

## 7. 静态 CFG 分析

### 7.1 CFG 构建

PTX 内核加载时, `PtxInterpreter::setupLabels()` 执行静态分析:

```cpp
// 1. 构建控制流图
ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(
    kernelContext->kernelStatements, label2pc);

// 2. 计算后支配点
ptx::cfg::PostDominatorMap postDoms = 
    ptx::cfg::CFGBuilder::computePostDominators(cfg);

// 3. 标注分支的 reconvergence PC
for (int i = 0; i < statements.size(); i++) {
    const auto &stmt = kernelContext->kernelStatements[i];
    
    if (stmt.type == S_BRA) {
        int reconvergence_pc = postDoms[i];
        branch.reconvergence_pc = reconvergence_pc >= 0 ? reconvergence_pc : i + 1;
    }
    else if (stmt.type == S_BAR_WARP_SYNC) {
        barrier.operands[1] = OperandContext{ImmOperand{std::to_string(i + 1)}};
    }
}
```

### 7.2 后支配点概念

**定义**: 节点 Y 是节点 X 的后支配点, 当且仅当从 X 到出口的所有路径都必须经过 Y。

**应用**: 对于分支指令, 后支配点即为所有执行路径的汇合点 (reconvergence point)。

### 7.3 CFG 分析输出

对于 test_nested_sync (16 线程), CFG 分析结果显示:
```
Total statements: 31
Barriers at PCs: 11, 25
Branches annotated with reconvergence PCs
```

---

## 8. 调度层设计

### 8.1 Warp 调度器

```cpp
// src/ptxsim/core/warp_scheduler.cpp
WarpContext* RoundRobinWarpScheduler::schedule_next() {
    // 简单的轮询调度
    // 在真实 GPU 中还有 Greedy/Longest-Warp 等策略
}
```

### 8.2 发散感知调度

```cpp
// SMContext::exe_once() 核心逻辑
auto lanes_by_pc = next_warp->get_lanes_by_pc();

if (lanes_by_pc.size() == 1) {
    // 非发散快速路径: 所有线程在同一 PC
    next_warp->execute_warp_instruction(*stmt, target_pc);
} else if (!lanes_by_pc.empty()) {
    // 发散路径: 每个 PC 组独立执行
    for (const auto& [pc, lanes] : lanes_by_pc) {
        StatementContext* stmt = sample_thread->get_statement_at(pc);
        next_warp->execute_warp_instruction(*stmt, pc);
    }
}
```

### 8.3 Warp 指令执行

```cpp
void WarpContext::execute_warp_instruction(StatementContext &stmt, int target_pc) {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!is_lane_active(i)) continue;
        if (warp_state.threads[i].pc != static_cast<uint32_t>(target_pc)) continue;
        
        ThreadContext *thread = threads[i].get();
        thread->sync_from_warp_state();
        
        if (thread->get_state() == BAR_SYNC) {
            // 重新进入屏障处理
            sm_context_->synchronize_barrier(thread->bar_id, thread);
            thread->sync_to_warp_state();
            continue;
        }
        
        thread->execute_thread_instruction();
        thread->sync_to_warp_state();
    }
    update_active_mask();
}
```

---

## 9. 已修复的 Bug 和架构改进

### 9.1 Bug #1: CFG Barrier Reconvergence PC

**问题**: 屏障后支配点分析错误, 使用 post-dominator 而非 `i+1`

**修复** (`ptx_interpreter.cpp`):
```diff
- barrier.reconvergence_pc = postDoms[i];  // 后支配点 (错误)
+ barrier.operands[1] = OperandContext{ImmOperand{std::to_string(i + 1)}};  // 下一条指令
```

**影响**: Test 3 屏障后跳过所有中间计算

### 9.2 Bug #2: cudaMemset 地址计算

**问题**: 设备地址空间未减去 global_pool 基地址

**修复** (`cudart_sim.cpp`):
```diff
- cudaMemcpy(dev_ptr + offset, ...);
+ cudaMemcpy(dev_ptr - global_pool_base + offset, ...);
```

### 9.3 Bug #3: sync_to_warp_state PC 保护

**问题**: 屏障完成后 ThreadContext 同步覆盖了 barrier 设置的 PC

**修复** (`thread_context.cpp:721`):
```cpp
if (thread_state.pc > static_cast<uint32_t>(pc)) {
    // 保留屏障设置的 PC
} else {
    thread_state.pc = pc;
}
```

### 9.4 Bug #4: SETP 指令未执行

**问题**: 虚函数分发到 SetpHandler::processOperation 失败

**修复** (`instruction_base.cpp`): 在 GenericPipelineHandler::executeOperation 中直接实现 SETP 逻辑

### 9.5 Bug #5: is_blocked 过滤导致屏障死锁

**问题**: `get_lanes_by_pc()` 过滤掉 `is_blocked` 的线程, 导致调度器找不到阻塞的线程, 屏障永远无法完成

**修复** (`warp_context.cpp:298`):
```diff
- !warp_state.threads[lane].is_blocked &&
```

### 9.6 Bug #6: 屏障释放使用错误掩码

**问题**: 使用 `participation_mask` 而非 `arrived_mask`, 更新所有参与线程的 PC, 而非仅已到达的线程

**修复** (`barrier.cpp`):
```diff
- if (wbar.participation_mask & (1u << i))
+ if (wbar.arrived_mask & (1u << i))
```

### 9.7 架构改进 #1: 统一 PC 权威源

**新增 API** (`warp_context.h`):
- `advance_thread_pc(lane_id, new_pc)` — 统一单线程 PC 更新
- `advance_all_threads(new_pc)` — 统一批量 PC 更新

**重构点**:
- `get_lanes_by_pc()` 读取 `warp_state.threads[lane].pc`
- `execute_warp_instruction` lane filter 使用 `warp_state.threads[i].pc`
- `sm_context.cpp` 调度器使用 `get_statement_at(target_pc)` 获取正确指令

### 9.8 修复验证状态

| 测试 | 状态 | 备注 |
|------|------|------|
| Test 1 (basic barrier) | ✅ PASS | 无回归 |
| Test 2 (multi-block) | ✅ PASS | 无回归 |
| Test 3 (nested sync) | ⚠️ 阻塞 | 第一个屏障完成, 但内核无限循环 |

---

## 10. Test 3 问题分析路径

### 10.1 测试场景

```cuda
// test_nested_sync<<<1, 16>>>
__shared__ int data_a[16], data_b[16];
data_a[tid] = tid;                  // PC 8-10
__syncthreads();                     // PC 11: 第一屏障 (reconv=12)
if (tid < 16) {                      // PC 12-18: setp + predicated branch
    data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16];
}
__syncthreads();                     // PC 25: 第二屏障 (reconv=26)
output[tid] = data_b[tid];           // PC 26-29
```

### 10.2 CFG 分析确认

```
Total statements: 31
Barriers at PCs: 11, 25
CFG 分配: PC 11 → 12, PC 25 → 26  ✅ 正确
```

### 10.3 运行时观察

```
[CLK:10]  [BARRIER_DBG] PC=11 reconvergence=12  ← 第一屏障第一次完成
[CLK:25]  [BARRIER_DBG] PC=25 reconvergence=26  ← 第二屏障完成
[CLK:10008] [BARRIER_DBG] PC=11 reconvergence=12  ← 无限循环! 回到 PC 11
```

**关键发现**: 第一屏障完成 → 线程释放到 PC 12 → 第二屏障也完成了一次 (PC 25 → 26) → 但执行又回到了 PC 11

### 10.4 假设链

| 假设 | 证据 | 状态 |
|------|------|------|
| H1: 双 PC 不一致导致指令跳转错误 | `get_lanes_by_pc` 已改用 warp_state | ✅ 已修复 |
| H2: 屏障释放后状态未重置为 Active | 已添加 `status = Active` | ✅ 已修复 |
| H3: 非发散分支的 advance_all_threads 未正确同步 | 检查 warp_state ↔ ThreadContext 同步 | 🔍 调查中 |
| H4: SIMT 栈在屏障完成后未正确清理 | barrier 完成后 simt_stack 应为空 | 🔍 待验证 |
| H5: `execute_warp_instruction` 中 BAR_SYNC 重入导致循环 | 屏障完成后线程状态被重置为 BAR_SYNC | 🔍 待验证 |

### 10.5 下一步调试方向

1. 在 `execute_warp_instruction` 中打印每个线程的 warp_state.pc 和执行状态
2. 追踪 PC 从 26 回退到 11 的完整路径
3. 验证 SIMT 栈在屏障完成后的状态

---

## 11. 下一步计划

### Phase 1: 定位 PC 回退根因

**目标**: 确定为何执行从 PC 26 (或 12) 回到 PC 11

**方法**:
- 在 `get_lanes_by_pc()` 中打印每个返回的 `{PC: [lanes]}` 映射
- 在 `execute_warp_instruction()` 中打印 `target_pc` 和执行的每个线程的 warp_state.pc
- 追踪从第二屏障完成 (PC 26) 到第一次回到 PC 11 之间的所有指令执行

### Phase 2 (备选): 创建精确模拟 16 线程 barrier 交互的测试

**目标**: 创建独立单元测试, 隔离屏障同步逻辑

### Phase 3 (长期): SIMT 架构重构

**目标**: 完全消除 ThreadContext::pc 和 warp_state::pc 的双源问题

**方案**:
- 将 `ThreadContext::pc` 改为只读视图, 通过 `WarpContext::get_thread_pc()` 访问
- 所有 PC 写入统一通过 `advance_thread_pc()` / `advance_all_threads()`
- 移除 `sync_from_warp_state()` 和 `sync_to_warp_state()` 中的 PC 同步逻辑

---

*文档结束 — 持续更新中*