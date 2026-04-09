# SIMT 架构 Stage 4 实施计划

**Created**: 2026-04-07  
**Goal**: 完成 Stage 4 调度器集成，使 test_syncthreads 通过  
**Estimated Time**: 16 小时 (4 hours x 4 tasks)

---

## 当前状态总结

### ✅ 已完成 (Stages 1-3)
- 数据结构 (ThreadState, WarpState, ExecMask, Wbar, SchedulerConfig)
- 语法扩展 (bar.warp.sync, activemask)
- Parser Visitor (visitBarWarpSyncInst, visitActivemaskInst)
- Instruction Handlers (BarWarpSyncHandler, ActivemaskHandler)
- 单元测试：32 tests, 619 assertions - **全部通过 ✅**

### ❌ 待完成 (Stage 4)
- Warp Scheduler 集成到执行流程
- __syncthreads() → bar.warp.sync 翻译层

### 当前代码修改状态
- `warp_context.cpp`: `update_active_mask()` 修复 ✅ (保留)
- `thread_context.cpp`: 移除 assert(state==RUN) ✅ (保留)
- `sm_context.cpp`: 无修改 (恢复原始状态)

---

## Stage 4 详细任务

### Task 4a: Warp Scheduler 集成 (4 小时)

**目标**: 让 `WarpContext::execute_warp_instruction()` 使用 per-thread PC

**修改文件**:
1. `src/ptxsim/core/warp_context.cpp`
2. `src/ptxsim/core/warp_scheduler.cpp`

**具体步骤**:

#### Step 1: 更新 execute_warp_instruction 使用 per-thread PC (1h)
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (is_lane_active(i) && i < threads.size() && threads[i] != nullptr) {
        ThreadContext *thread = threads[i].get();
        
        // 从 warp_state 获取该 thread 的 PC (不是 warp 级 PC)
        if (warp_state.threads[i].pc_stacks.empty()) {
            warp_state.threads[i].pc_stacks.push_back(0);
        }
        thread->set_pc(warp_state.threads[i].pc_stacks.back());
        
        // ... rest of execution
        
        // 更新 warp_state 中的 PC
        warp_state.threads[i].pc_stacks.back() = thread->get_pc();
    }
}
```

#### Step 2: 移除 is_blocked 检查 (30min)
当前 barrier 同步使用 `thread->get_state() == BAR_SYNC` 检查，
这与 warp_state.threads[i].is_blocked 重复。

简化为只使用 thread state。

#### Step 3: 更新 WarpScheduler使用priority (1.5h)
修改 `RoundRobinWarpScheduler::schedule_next()`:
1. 使用 `warp.count_schedulable_lanes()` 过滤已完成/全 blocked 的 warps
2. 从 scheduler_config.ini 读取权重 (如果需要)
3. 实现 anti-starvation (记录每个 warp 被调度次数)

#### Step 4: 验证编译 (1h)
```bash
cmake --build build --target ptxsim
# 确保 0 errors, 0 warnings
```

---

### Task 4b: __syncthreads() 翻译层 (4 小时)

**目标**: 将 CUDA __syncthreads() 翻译为 bar.warp.sync 指令

**修改文件**:
1. `src/cudart/cudart_sim.cpp` 或 `src/ptx_parser/ptx_visitor.cpp`
2. `src/grammar/ptxInstructions.g4` (可能需要)

**具体步骤**:

#### Step 1: 分析 __syncthreads() 编译流程 (1h)
__syncthreads() 编译流程:
```
CUDA: __syncthreads()
  ↓ (nvcc)
PTX: bar.sync %bar0, ALL;
  ↓ (our parser)
StatementContext { type: S_BAR_SYNC }
  ↓ (dispatch)
BarHandler::executeBarrier() ← 旧逻辑
```

我们需要添加翻译层在 parser 阶段：
```
PTX: bar.sync %bar0, ALL;
  ↓ (detect warp-level barrier)
PTX: bar.warp.sync 0xFFFFFFFF, $+1;  ← 新翻译
  ↓
StatementContext { type: S_BAR_WARP_SYNC }
  ↓
BarWarpSyncHandler ← 新逻辑 (Wbar)
```

#### Step 2: 实现 Parser 翻译 (1.5h)
在 `PtxVisitor::visitBarInst()`:
```cpp
PTXVisitor::visitBarInst(BarInstContext *ctx) {
    // 检查是否是 warp-level 同步
    // bar.sync %bar0, ALL; 且只有一个 warp → 翻译为 bar.warp.sync
    if (isWarpLevelBarrier(ctx)) {
        return visitBarWarpSyncInst(ctx);  // 委托到新 handler
    } else {
        return visitOldBarSync(ctx);  // CTA-level barrier, 用旧逻辑
    }
}
```

#### Step 3: 实现 isWarpLevelBarrier 检测 (1h)
```cpp
bool PtxVisitor::isWarpLevelBarrier(BarInstContext *ctx) {
    // 简单规则：如果 CTA 只有 32 线程 (1 warp), 则是 warp-level
    return (BlockDim.x * BlockDim.y * BlockDim.z <= 32);
}
```

#### Step 4: 保留旧 barrier 用于 CTA-level (30min)
确保 __syncthreads_array() 和 multi-block barriers 仍用旧逻辑。

---

### Task 4c: ThreadContext ↔ WarpState 同步 (4 小时)

**目标**: 确保 ThreadContext 状态与 warp_state.threads[i] 同步

**修改文件**:
1. `src/ptxsim/core/thread_context.cpp`
2. `src/ptxsim/core/warp_context.cpp`

**具体步骤**:

#### Step 1: 添加 WarpContext 引用到 ThreadContext (1h)
```cpp
// thread_context.h
class ThreadContext {
    WarpContext* warp_context_ = nullptr;  // 已有
    
    // 添加辅助方法
    void sync_from_warp_state();
    void sync_to_warp_state();
};
```

#### Step 2: 实现状态同步方法 (1.5h)
```cpp
void ThreadContext::sync_to_warp_state() {
    if (!warp_context_) return;
    
    int lane_id = get_lane_id();
    if (lane_id < 0) return;
    
    auto& thread_state = warp_context_->warp_state.threads[lane_id];
    thread_state.pc = pc;
    thread_state.next_pc = next_pc;
    thread_state.is_blocked = (state == BAR_SYNC);
    thread_state.is_exited = (state == EXIT);
}

void ThreadContext::sync_from_warp_state() {
    if (!warp_context_) return;
    
    int lane_id = get_lane_id();
    if (lane_id < 0) return;
    
    auto& thread_state = warp_context_->warp_state.threads[lane_id];
    pc = thread_state.pc;
    next_pc = thread_state.next_pc;
    // is_blocked 是派生状态，不直接同步
}
```

#### Step 3: 在关键点调用同步 (1.5h)
- `execute_thread_instruction()` 前后
- `set_state()` 中
- `_execute_once()` 中

```cpp
void ThreadContext::execute_thread_instruction() {
    sync_from_warp_state();
    _execute_once();
    sync_to_warp_state();
}
```

---

### Task 4d: Barrier 机制简化 (4 小时)

**目标**: 简化当前 SMContext 的旧 barrier 逻辑，减少死锁风险

**修改文件**:
1. `src/ptxsim/core/sm_context.cpp`
2. `src/ptxsim/core/cta_context.cpp`

**具体步骤**:

#### Step 1: 分析当前 barrier 实现 (1h)
当前 `SMContext::synchronize_barrier()`:
- 使用 `barrier_waiting_threads[barId]` set 存储等待线程
- 使用 `barrier_thread_counts[barId]` 存储期望线程数
- 当 `set.size() >= expected` 时释放

#### Step 2: 简化逻辑 - 只检查执行线程 (1.5h)
```cpp
bool SMContext::synchronize_barrier(int barId, ThreadContext* thread) {
    // 初始化 barrier 计数
    if (barrier_thread_counts.find(barId) == barrier_thread_counts.end()) {
        size_t total_threads = get_cta_context()->get_thread_count();
        barrier_thread_counts[barId] = total_threads;
        barrier_arrival_counts[barId] = 0;  // 新计数器
    }
    
    // 标记到达
    barrier_arrival_counts[barId]++;
    
    // 检查所有线程是否到达
    if (barrier_arrival_counts[barId] >= barrier_thread_counts[barId]) {
        // 释放：将所有线程状态设为 RUN
        release_barrier(barId);
        barrier_arrival_counts[barId] = 0;  // 重置计数器
        return true;
    }
    
    // 等待
    thread->set_state(BAR_SYNC);
    return false;
}
```

#### Step 3: 实现 release_barrier (1h)
```cpp
void SMContext::release_barrier(int barId) {
    CTAContext* cta = get_cta_context();
    for (int i = 0; i < cta->get_thread_count(); i++) {
        ThreadContext* thread = cta->get_thread(i);
        if (thread && thread->get_state() == BAR_SYNC) {
            thread->set_state(RUN);
            // 更新 PC 到 barrier 后的指令
            int next_pc = thread->get_pc() + 1;
            thread->set_pc(next_pc);
            thread->set_next_pc(next_pc);
            
            // 更新 warp_state
            WarpContext* warp = thread->get_warp_context();
            if (warp) {
                int lane_id = thread->get_lane_id();
                if (lane_id >= 0) {
                    warp->warp_state.threads[lane_id].pc = next_pc;
                    warp->warp_state.threads[lane_id].next_pc = next_pc;
                    warp->warp_state.threads[lane_id].is_blocked = false;
                }
            }
        }
    }
}
```

#### Step 4: 测试 barrier 修复 (30min)
```bash
. env.sh && cmake --build build --target test_syncthreads
timeout 30 ./build/bin/test_syncthreads
# 应该不再死锁
```

---

## 验证计划

### 编译验证
```bash
cmake --build build 2>&1 | grep -E "error:|warning:"
# 预期：0 errors, 0 warnings
```

### 功能验证
```bash
# Run test_syncthreads
cd build && ctest -R syncthreads -V
# 预期：所有子测试通过 (Test 1, Test 2, Test 3)
```

### 回归验证
```bash
cd build && ctest -L mini
# 预期：所有现有测试仍通过
```

---

## 风险与应对

| 风险 | 可能性 | 影响 | 应对 |
|------|--------|------|------|
| 翻译层逻辑复杂 | 中 | 延期 | 先实现简单版本：所有 bar.sync 都翻译为 bar.warp.sync |
| 状态同步遗漏 | 高 | 死锁 | 添加详细日志，在每个同步点打印 |
| 性能回退 | 低 | 需要优化 | Stage 4 完成后，Stage 5 进行性能分析 |
| 破坏现有测试 | 低 | 回归 | 运行完整测试套件 |

---

## 提交策略

### 分批提交
1. **Batch 1**: Task 4a (warp scheduler) - PC 切换
2. **Batch 2**: Task 4b (translation layer) - __syncthreads 翻译
3. **Batch 3**: Task 4c (sync) - ThreadContext ↔ WarpState
4. **Batch 4**: Task 4d (barrier simplify) - barrier 逻辑简化

### 每批提交前
- ✅ 编译通过
- ✅ LSP diagnostics clean
- ✅ 测试当前修改的功能
- ✅ 运行现有测试确保不破坏

### 放弃条件
如果单个 Task 超过 8 小时未完成：
1. 回滚修改
2. 咨询 Oracle agent
3. 考虑简化方案

---

## 下一步

1. **清理代码**: 删除临时文件，确认要保留的修改
2. **提交当前进度**: commit warp_context.cpp 和 thread_context.cpp 的修复
3. **开始 Task 4a**: Warp scheduler 集成

---

**Estimated Total**: 16 hours  
**Success Criteria**: test_syncthreads 3/3 子测试全部通过
