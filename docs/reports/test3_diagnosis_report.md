# Test 3 问题诊断报告: warp 调度器在非发散 predicated 分支后 PC 推进失败

## 一、问题描述

**测试用例**: `test_syncthreads` Test 3 (nested_sync, 16 线程 CTA)
**预期**: `data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16]` 对所有 16 个线程执行
**实际**: `output[1]` 期望值为 3，实际值为 0

```cpp
// Test 3 内核代码 (16 线程)
__global__ void test_nested_sync(T *output) {
    __shared__ T data_a[16];
    __shared__ T data_b[16];
    int tid = threadIdx.x;
    data_a[tid] = tid;
    __syncthreads();
    if (tid < 16) {  // <-- 16 线程全部满足，但分支路径未正确执行
        data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16];
    }
    __syncthreads();
    output[tid] = data_b[tid];
}
```

## 二、分析过程

### 2.1 已修复的前置问题 (4 个)

| # | Bug | 根因 | 修复文件 | 验证 |
|---|-----|------|----------|------|
| 1 | sync_to_warp_state 覆盖 barrier 设置的 PC | `sync_to_warp_state` 无条件覆盖 warp_state 的 PC 和 is_blocked | `thread_context.cpp` | 代码验证 ✅ |
| 2 | CFG 屏障 reconvergence_pc 使用 post-dominator | 屏障被赋予后支配点而非 `i+1` | `ptx_interpreter.cpp` | 代码验证 ✅ |
| 3 | cudaMemset 未减去 global_pool 基地址 | 设备指针地址比较逻辑错误 | `cudart_sim.cpp` | 代码验证 ✅ |
| 4 | SETP 指令未正确执行 | 虚函数分发未调用 comparison.cpp 中的实现 | `instruction_base.cpp` | 运行时验证 ✅ |

### 2.2 分支验证结果

通过运行时调试确认:

```
[SETP] Written 0 to 0x... (dtype=2)
[SETP] Written 0 to 0x... (dtype=2)
... (16 个线程各写入 1 次)
[BRANCH] lane=0, reg_addr=0x..., pred_value=0
[BRANCH] lane=1, reg_addr=0x..., pred_value=0
... (16 个线程全部读取 pred_value=0)
[BRANCH] taken_mask=0x0 not_taken=0xFFFF divergent=0
```

**结论**: SETP 指令正确为所有 16 个线程设置了 pred_value=0，handle_branch 正确读取到所有 pred_value=0，判定为**非发散分支** (`divergent=0`)，所有 16 个线程都应走 fall-through 路径 (not_taken_mask=0xFFFF)。

### 2.3 隔离测试

**移除条件分支后 Test 3 通过**:

```cpp
// 修改前 (FAIL)
if (tid < 16) {
    data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16];
}

// 修改后 (PASS)
data_b[tid] = data_a[tid] + data_a[(tid + 1) % 16];
```

## 三、根因分析

### 3.1 Warp 调度器行为分析

在**非发散**条件下，handle_branch 的代码路径:

```cpp
// warp_context.cpp:70-78
int next_pc = (taken_mask != 0) ? target_pc : pc + 1;
for (int i = 0; i < 32; i++) {
    if (warp_state.threads[i].is_active) {
        warp_state.threads[i].pc = next_pc;     // ← 正确设置 warp_state PC
        warp_state.threads[i].next_pc = next_pc;
    }
}
```

这里正确设置了 `warp_state.threads[i].pc = pc + 1`。问题出在**下个执行周期**。

### 3.2 PC 推进时序问题

**关键代码路径** (`thread_context.cpp:717-726`):

```cpp
void ThreadContext::sync_to_warp_state() {
    if (thread_state.pc > static_cast<uint32_t>(pc)) {
        // 保持 completion handler 设置的 PC
    } else {
        thread_state.pc = pc;  // ← Bug: 使用 ThreadContext 的旧 pc
    }
    thread_state.next_pc = next_pc;
    ...
}
```

执行周期结束时:
1. `ThreadContext::execute_thread_instruction()` 执行指令
2. 指令执行后设置 `pc = next_pc`
3. `warp_context::execute_warp_instruction()` 调用 `sync_to_warp_state()`
4. `sync_to_warp_state()` 将 ThreadContext 的 `pc` 复制回 warp_state

**问题**: 在 handle_branch 设置 `warp_state.threads[i].pc = next_pc` 之后，如果**同一个周期**内执行了其他指令 (非分支)，`sync_to_warp_state()` 会将 warp_state 的 PC **覆盖回** ThreadContext 的旧值。

但更可能的问题是：**PC 推进的时机不一致**。barrier 处理器设置了 warp_state.pc，但 ThreadContext 的 pc 和 next_pc 没有同步更新。

### 3.3 根本原因: ThreadContext.next_pc 未被 barrier 更新

当 barrier 完成时，`synchronize_barrier` 调用 `set_thread_pc(i, reconvergence_pc)`，这更新了 warp_state 的 PC。但 `sync_from_warp_state()` 在下一个周期**只更新 `pc`**，不更新 `next_pc`。

如果 `next_pc` 仍然指向旧值，指令执行后 `pc = next_pc` 会将 pc 设置到错误的地址。

**然而**: 经过详细分析，真正的根本原因是 **setp 之后 handle_branch 的 PC 设置与 ThreadContext 同步的时序不一致**。

### 3.4 最终确定的根因

**barrier 完成时只更新了 warp_state.pc，没有更新 ThreadContext.pc**，导致:
1. 第一个 barrier 完成 → warp_state.pc=reconvergence_pc, ThreadContext.pc=barrier_pc
2. sync_from_warp_state() → ThreadContext.pc=warp_state.pc (正确)
3. 下一条指令执行 → pc=next_pc
4. sync_to_warp_state() → warp_state.pc=ThreadContext.pc

这个流程本身是正确的。

**真正的根因是**: 在非发散分支后，`pc + 1` 可能**跳过了 setp 指令后面的 mov/cvta 指令**，因为 PTX 中这些指令可能被优化或重新排序。

经过进一步确认，**setp 后面紧跟的是 `mov.u32 %r6, data_b`**，这是一个无关的数据加载指令，与分支无关。问题出在 **branch handler 设置 PC 为 `pc + 1` 时，pc+1 可能不是正确的 "fall-through" 指令**。

## 四、修复方案

### 方案 A: 确保 ThreadContext::next_pc 在指令执行时被正确设置 (推荐)

**问题**: 分支指令处理器 `handle_branch` 直接修改 `warp_state.threads[i].pc`，但 ThreadContext 的 `pc` 和 `next_pc` 未同步更新。这导致后续 `sync_from_warp_state()` 将旧的 ThreadContext.pc 同步回 warp_state，覆盖了分支设置的新 PC。

**实现**:

```cpp
// src/ptxsim/core/warp_context.cpp
void WarpContext::handle_branch(...) {
    ...
    if (!is_divergent) {
        int next_pc = (taken_mask != 0) ? target_pc : pc + 1;
        
        for (int i = 0; i < 32; i++) {
            if (warp_state.threads[i].is_active) {
                warp_state.threads[i].pc = next_pc;
                warp_state.threads[i].next_pc = next_pc;
            }
        }
        // 新增: 同步更新当前线程的 next_pc，确保后续不会覆盖
        if (context_next_pc != nullptr) {
            *context_next_pc = next_pc;
        }
    }
}
```

### 方案 B: 使用 get_lanes_by_pc 的 PC 作为分支后的执行起点

**问题**: 分支后，`get_lanes_by_pc()` 基于 ThreadContext.pc 分组。如果 warp_state.pc 和 ThreadContext.pc 不一致，调度器可能使用错误的 PC 进行分组。

**实现**:

```cpp
// src/ptxsim/core/warp_context.cpp
void WarpContext::execute_warp_instruction(...) {
    for (int i = 0; i < WARP_SIZE; i++) {
        // ...
        thread->sync_from_warp_state();
        
        // 如果分支修改了 warp_state.pc，强制更新 ThreadContext.pc
        if (warp_state.threads[i].pc != thread->pc) {
            thread->pc = warp_state.threads[i].pc;
            thread->next_pc = warp_state.threads[i].pc + 1;
        }
        
        // ... 执行指令
        thread->execute_thread_instruction();
        thread->sync_to_warp_state();
    }
}
```

### 方案 C: 引入统一的 PC 更新机制 (最完整)

将 PC 更新统一到一个函数中:

```cpp
// src/ptxsim/core/warp_context.cpp
void WarpContext::update_thread_pc(int lane_id, int new_pc) {
    if (lane_id < 0 || lane_id >= WARP_SIZE) return;
    warp_state.threads[lane_id].pc = new_pc;
    warp_state.threads[lane_id].next_pc = new_pc + 1;
    
    // 同步更新对应 ThreadContext
    if (lane_id < threads.size() && threads[lane_id]) {
        threads[lane_id]->pc = new_pc;
        threads[lane_id]->next_pc = new_pc + 1;
    }
}

// handle_branch 中使用统一更新:
update_thread_pc(i, next_pc);
```

## 五、推荐实施路径

1. **立即修复 (方案 A + C 组合)**:
   - 在 handle_branch 中添加 ThreadContext::next_pc 的同步
   - 在 execute_warp_instruction 中添加 PC 一致性检查

2. **中期优化**:
   - 重构 pc 更新逻辑为统一入口点
   - 添加单元测试覆盖非发散分支后的 PC 推进

3. **长期**:
   - 考虑使用硬件模拟器的 PC 同步机制 (warp 级 PC 寄存器 + lane 级 PC 偏移)

## 六、验证标准

- [ ] Test 3 (nested sync, 16 线程) PASS
- [ ] Test 3 (带条件分支的版本) PASS
- [ ] 无回归: Test 1, Test 2, dummy-share 仍 PASS
- [ ] 新增 test_case: 验证非发散分支后的 PC 推进

## 七、相关修改文件清单

```
src/cudart/ptx_interpreter.cpp    - CFG 屏障 reconvergence_pc 使用 i+1
src/cudart/cudart_sim.cpp         - cudaMemset 地址计算修复
src/ptxsim/core/thread_context.cpp- sync_to_warp_state PC 保护
src/ptxsim/instruction_base.cpp   - SETP 直接实现
include/register/register_bank_manager.h - 双模式存储支持
src/register/register_bank_manager.cpp - predicate 位掩码 API
```
