# 强制汇聚机制设计（Implicit Reconvergence Enforcement）

**版本**: 1.0
**日期**: 2026-05-29
**状态**: 草稿 — 待实现
**负责人**: PTX-EMU Team

---

## 目录

1. [背景与问题](#1-背景与问题)
2. [当前架构分析](#2-当前架构分析)
3. [设计方案](#3-设计方案)
4. [实现细节](#4-实现细节)
5. [执行流程对比](#5-执行流程对比)
6. [测试验证计划](#6-测试验证计划)
7. [边界情况处理](#7-边界情况处理)
8. [审查要点](#8-审查要点)

---

## 1. 背景与问题

### 1.1 真实硬件行为（Hopper/Blackwell ITS）

根据 `docs/architecture/sm90_100.md`，NVIDIA Hopper/Blackwell 通过 BSSY/BSYNC 机制实现隐式汇聚：

```
BSSY B0, reconvergence_pc    // 设置屏障点，所有到达此点的线程挂起等待
BSYNC B0                      // 当前线程到达屏障，挂起等待其他线程
```

**关键语义**：
- 线程到达 `reconvergence_pc` 时，如果其他线程未到达，该线程**阻塞等待**
- 调度器切换到其他已 Ready 的线程执行
- 当所有线程都到达屏障点后，屏障释放，所有线程统一继续执行

### 1.2 当前模拟器行为缺陷

当前 PTX-EMU 的汇聚是**检测型**而非**强制型**：

| 组件 | 当前行为 | 问题 |
|------|---------|------|
| SIMT Stack | `reconvergence_pc` 只用于 `is_converged()` 检测 | 不强制等待 |
| 调度器 | `get_lanes_by_pc()` 选择最低 PC 组执行 | 不考虑汇聚状态 |
| 执行层 | `execute_warp_instruction()` 无条件执行 | 不检查是否该等待 |

**现象**：Path A 执行到 `ret(PC=27)` 后才切换到 Path B，Path A 实际执行了 PC=14 之后的所有指令，这违反了 PTX 语义。

### 1.3 测试案例分析

`tests/test_divergence_sync_convergence.cpp` Test A：

```
指令布局：
  PC=0..3:    MOV（分歧前）
  PC=4:       @%p1 bra $L__BB0_4（分歧）→ taken=PC=28, not_taken=PC=5
  PC=5..13:   MOV（Path A, lanes 0-15）
  PC=14..26:  MOV（汇聚后代码）
  PC=27:      ret
  PC=28..33:  MOV（Path B, lanes 16-31）
  PC=34:      bra.uni $L__BB0_3（→PC=14）

当前错误行为：
  Path A: 5→...→13→14→15→...→27(ret)→[切换]→Path B: 28→...→33→bra.uni→14
         ↑ PC=14 后未等待，直接执行到 ret

期望正确行为：
  Path A: 5→...→13→14【阻塞】← 等待 Path B
  Path B:                              28→...→33→bra.uni→14【汇聚检测→POP】
                                                         ↓
  统一执行:                                            14→15→...→27→ret
```

---

## 2. 当前架构分析

### 2.1 核心数据结构

**SIMTStackEntry**（`include/ptxsim/simt_stack.h`）：
```cpp
struct SIMTStackEntry {
    int branch_pc;           // 分歧点 PC
    int reconvergence_pc;    // 汇聚点 PC
    uint32_t active_mask;    // 跟踪的 lane 掩码
    uint32_t return_mask;    // 恢复用的掩码
    int return_pc;           // 恢复点 PC
};
```

**ThreadState**（`include/ptxsim/thread_state.h`）：
```cpp
struct ThreadState {
    uint32_t pc = 0;           // 当前 PC
    bool is_exited = false;    // 已退出标记
    bool is_blocked = false;   // 阻塞标记（用于 barrier 等待）
};
```

### 2.2 关键函数分析

**`handle_branch()`** (`warp_context.cpp:10-104`)：
- 分歧时 push SIMT stack entry
- 设置 `entry.reconvergence_pc` 和 `entry.active_mask`
- 当 `reconvergence_pc == target_pc` 时，`active_mask` 包含 taken + not_taken（保证外层追踪）

**`is_converged()`** (`simt_stack.cpp:7-22`)：
```cpp
bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        if (active_mask & (1u << i)) {
            if (threads[i].is_exited) continue;  // 跳过已退出
            if ((int)threads[i].pc != reconvergence_pc) return false;
        }
    }
    return true;
}
```

**注意**：`is_converged()` 只检查 PC 是否等于 `reconvergence_pc`，不强制阻塞。

**`check_reconvergence()`** (`warp_context.cpp:112-142`)：
- 调用 `simt_stack.check_reconvergence()`
- 若 `is_converged()` 为 true，pop stack
- 更新 `exec_mask`

**`execute_warp_instruction()`** (`warp_context.cpp:186-236`)：
```cpp
void WarpContext::execute_warp_instruction(StatementContext &stmt, int target_pc) {
    for (int i = 0; i < WARP_SIZE; i++) {
        // ... lane 选择逻辑 ...
        
        if (warp_state.threads[i].pc != static_cast<uint32_t>(target_pc)) {
            continue;  // 只执行 target_pc 的 lane
        }
        
        thread->execute_thread_instruction();  // 无条件执行
        thread->sync_to_warp_state();
    }
    update_active_mask();
}
```

**问题**：执行前未检查"是否应该等待"。

**`sm_context.cpp` 调度器** (`sm_context.cpp:250-305`)：
```cpp
for (const auto& [candidate_pc, candidate_lanes] : lanes_by_pc) {
    bool all_non_blocked = true;
    for (int lane : candidate_lanes) {
        if (ws.threads[lane].is_blocked) {
            all_non_blocked = false;
            break;
        }
    }
    if (all_non_blocked) {
        pc = candidate_pc;  // 选择最低 PC 的非阻塞组
        selected_lanes = &candidate_lanes;
        break;
    }
}
```

**观察**：调度器已经支持 `is_blocked` 检测，但当前没有机制在汇聚点设置 `is_blocked=true`。

### 2.3 `bra.uni` 处理

`bra.uni` 是无条件的统一跳转（在分歧路径末尾）。`handle_branch` 检测到 `is_divergent=false`（因为 predicate 为空），不 push 新 entry，所有活跃线程直接跳转到 target。

这意味着 `bra.uni` 跳转到 `reconvergence_pc` 时，会触发 `is_converged()` 检测：

```cpp
// Path B 执行 bra.uni 后，threads[i].pc 变为 14
// check_reconvergence() 调用 is_converged()
// 发现所有 active_mask 中的 lane 的 pc == 14 → true → pop
```

---

## 3. 设计方案

### 3.1 核心思路

当 warp 中的线程执行到 `reconvergence_pc` 时：
1. **检查**：是否所有 `active_mask` 中的线程都已到达 `reconvergence_pc`
2. **若不是**：将已到达 `reconvergence_pc` 的线程标记为 `is_blocked=true`，阻止执行
3. **调度器切换**：调度器检测到该 PC 组有 blocked 线程，选择其他 PC 组执行
4. **恢复**：当其他线程通过 `bra.uni` 到达 `reconvergence_pc` 后，汇聚条件满足，stack pop

### 3.2 设计原则

| 原则 | 说明 |
|------|------|
| 最小侵入性 | 不修改指令布局，不引入新指令类型 |
| 复用现有机制 | 利用 `is_blocked` 已有机制，复用调度器的 blocked 检测 |
| 语义一致性 | 与硬件 BSSY/BSYNC 语义一致 |
| 向后兼容 | 不影响现有的 `bar.warp.sync` 机制 |

### 3.3 实现位置

**修改文件**：
- `src/ptxsim/core/warp_context.cpp` — 新增汇聚阻塞逻辑
- `include/ptxsim/warp_context.h` — 新增方法声明

**不修改**：
- `simt_stack.cpp` — 汇聚检测逻辑不变
- `sm_context.cpp` — 调度器已有 blocked 检测逻辑

---

## 4. 实现细节

### 4.1 新增方法声明

在 `include/ptxsim/warp_context.h` 中添加：

```cpp
/**
 * @brief 检查当前 warp 是否应该在 reconvergence_pc 阻塞等待
 * @param target_pc 即将执行的指令 PC
 * @param[out] blocked_lanes 输出参数，返回被阻塞的 lane 列表
 * @return true 如果有 lane 被阻塞，false 正常执行
 * 
 * @details
 * 当 SIMT stack 非空且 target_pc 等于栈顶的 reconvergence_pc 时：
 * 1. 检查 active_mask 中是否所有线程都已到达 reconvergence_pc
 * 2. 若有线程未到达，将已到达的线程标记为 is_blocked=true
 * 3. 调度器会切换到其他已 Ready 的 PC 组执行
 * 
 * 语义：模拟 BSSY/BSYNC 机制 — 到达屏障点的线程挂起等待
 */
bool check_and_block_at_reconvergence_point(int target_pc, 
                                           std::vector<int>& blocked_lanes);
```

### 4.2 实现逻辑

在 `src/ptxsim/core/warp_context.cpp` 中添加：

```cpp
bool WarpContext::check_and_block_at_reconvergence_point(int target_pc,
                                                         std::vector<int>& blocked_lanes) {
    blocked_lanes.clear();
    
    // 1. SIMT stack 为空，无汇聚需求
    if (simt_stack.empty()) {
        return false;
    }
    
    const SIMTStackEntry& top = simt_stack.top();
    int reconv_pc = top.reconvergence_pc;
    
    // 2. target_pc 不是汇聚点，正常执行
    if (target_pc != reconv_pc) {
        return false;
    }
    
    // 3. target_pc == reconvergence_pc，检查是否所有 lane 都已到达
    //    遍历 active_mask 中的所有 lane
    
    // 首先统计有多少 active lane 还未到达 reconvergence_pc
    int lanes_not_at_reconv = 0;
    std::vector<int> lanes_at_reconv;  // 已到达汇聚点的 lane
    
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!(top.active_mask & (1u << i))) continue;
        
        // 跳过已退出的线程
        if (warp_state.threads[i].is_exited) {
            continue;
        }
        
        if ((int)warp_state.threads[i].pc == reconv_pc) {
            lanes_at_reconv.push_back(i);
        } else {
            lanes_not_at_reconv++;
        }
    }
    
    // 4. 所有 lane 都已到达或已退出，无需阻塞
    if (lanes_not_at_reconv == 0) {
        return false;
    }
    
    // 5. 有 lane 尚未到达，将已到达的 lane 标记为 blocked
    for (int lane : lanes_at_reconv) {
        if (!warp_state.threads[lane].is_blocked) {
            warp_state.threads[lane].is_blocked = true;
            blocked_lanes.push_back(lane);
        }
    }
    
    return !blocked_lanes.empty();
}
```

### 4.3 调用位置

在 `execute_warp_instruction()` 入口处调用：

```cpp
void WarpContext::execute_warp_instruction(StatementContext &stmt, int target_pc) {
    // === 新增：强制汇聚检查 ===
    std::vector<int> blocked_lanes;
    if (check_and_block_at_reconvergence_point(target_pc, blocked_lanes)) {
        // 有 lane 在汇聚点被阻塞，不执行指令，等待调度器切换
        // blocked 状态会通过 update_active_mask() 反映到 active_mask
        update_active_mask();
        return;
    }
    // === 原有逻辑继续 ===
    
    for (int i = 0; i < WARP_SIZE; i++) {
        // ... 原有执行逻辑 ...
    }
    update_active_mask();
}
```

### 4.4 状态转移图

```
                    ┌─────────────────────────────────────────┐
                    │          分歧发生 (PC=4)               │
                    │  handle_branch() → SIMT stack push     │
                    │  active_mask = taken | not_taken       │
                    └─────────────────┬─────────────────────┘
                                      │
                                      ▼
        ┌─────────────────┬─────────────────────────┬─────────────────┐
        │                 │                         │                 │
        ▼                 ▼                         ▼                 ▼
   Path A lanes      Path B lanes              Scheduler          Scheduler
   (PC=5)            (PC=28)                 选择 PC=5          选择 PC=28
        │                 │                         │                 │
        ▼                 ▼                         ▼                 ▼
   执行指令...        执行指令...              到达 PC=14？         执行指令...
        │                 │                         │                 │
        │                 │                         ▼
        │                 │              check_and_block_at_
        │                 │              reconvergence_point()
        │                 │                         │
        │                 │            ┌──────────┴──────────┐
        │                 │            │                     │
        │                 │            ▼                     ▼
        │                 │      有其他lane          所有lane
        │                 │      未到达PC=14          已到达PC=14
        │                 │            │                     │
        │                 │            ▼                     │
        │                 │      is_blocked=true            │
        │                 │            │                     │
        │                 │            ▼                     │
        │                 │      不执行指令                正常执行
        │                 │      update_active_mask()        │
        │                 │            │                     │
        │                 │            ▼                     │
        │                 │      调度器跳过PC=14组          │
        │                 │      选择 PC=28 执行            │
        │                 │            │                     │
        │                 │            │         ┌────────────┘
        │                 │            │         │
        │                 │            ▼         ▼
        │                 │      Path B 执行    继续
        │                 │      bra.uni→14
        │                 │            │
        │                 │            ▼
        │                 │      check_reconvergence()
        │                 │      is_converged()=true
        │                 │            │
        │                 │            ▼
        │                 │      SIMT stack POP
        │                 │      is_blocked 恢复
        │                 │            │
        │                 │            ▼
        │                 │      统一执行 PC=14+
        └─────────────── │ ◄──────────┘
                         │
                         ▼
                    汇聚完成
```

---

## 5. 执行流程对比

### 5.1 修改前（错误行为）

```
时间步  活跃线程       执行的指令   说明
─────────────────────────────────────────────────────
t1     T0-T31        @%p1 bra     分歧发生
                                    ↓
t2     T0-T15 (16)   MOV PC=5      Scheduler 选 Path A（最低 PC）
t3     T0-T15        MOV PC=6      ↓
...                                    Path A 继续执行
t10    T0-T15        MOV PC=13     ↓
t11    T0-T15        MOV PC=14     【错误】未等待，直接执行
t12    T0-T15        MOV PC=15     ↓
...                                    继续执行后续指令
t24    T0-T15        ret PC=27     Path A ret 退出
                                    ↓
t25    T16-T31 (16)  MOV PC=28     Scheduler 切换到 Path B
t26    T16-T31       MOV PC=29     ↓
...                                    Path B 执行
t31    T16-T31       bra.uni→14   跳转到汇聚点
                                    ↓
t32    T0-T31        MOV PC=14     统一执行（但 Path A 已 ret）
```

**问题**：Path A 在 PC=14 时未等待 Path B，直接执行到 ret，违背 PTX 语义。

### 5.2 修改后（正确行为）

```
时间步  活跃线程       执行的指令   说明
─────────────────────────────────────────────────────
t1     T0-T31        @%p1 bra     分歧发生
                                    ↓
t2     T0-T15 (16)   MOV PC=5      Scheduler 选 Path A（最低 PC）
t3     T0-T15        MOV PC=6      ↓
...                                    Path A 继续执行
t10    T0-T15        MOV PC=13     ↓
t11    T0-T15        MOV PC=14     【正确】检测到需要等待，阻塞
        T0-T15       is_blocked=true  标记为 blocked
                                    ↓
t12    T16-T31 (16)  MOV PC=28     Scheduler 切换到 Path B
t13    T16-T31       MOV PC=29     ↓
...                                    Path B 执行
t17    T16-T31       bra.uni→14   跳转到汇聚点
                                    ↓
t18    T0-T31        汇聚检测      is_converged()=true
                           SIMT stack POP
                           is_blocked 恢复
                                    ↓
t19    T0-T31 (32)   MOV PC=14     统一执行
t20    T0-T31        MOV PC=15     ↓
...                                    继续执行
t33    T0-T31        ret PC=27     warp 结束
```

---

## 6. 测试验证计划

### 6.1 单元测试修改

`tests/test_divergence_sync_convergence.cpp` Test A 需要修改断言：

```cpp
// === 修改后的期望行为 ===

// 分歧后状态: Path A(0-15)→PC=14(阻塞), Path B(16-31)→PC=28
CHECK(w->get_thread_pc(0)  == PATH_B_TARGET);   // Path B 还在 PC=28
CHECK(w->get_thread_pc(31) == CONV_PC);           // Path A 阻塞在 PC=14

// 调度器选择 Path B（PC=28）执行
{   // Path B 执行 PC=28..34
    int pc;
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET);      // 28
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET + 1);  // 29
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET + 2);  // 30
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET + 3);  // 31
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET + 4);  // 32
    pc = step_warp(w, v); CHECK(pc == PATH_B_TARGET + 5);  // 33
    pc = step_warp(w, v); CHECK(pc == BRA_UNI_PC);         // 34: bra.uni → 14
}

// bra.uni 到达 PC=14，触发汇聚检测 → SIMT stack POP
CHECK(w->get_simt_stack().empty());
CHECK(w->get_exec_mask() == 0xFFFFFFFFu);

// 汇聚后统一执行
{   // 统一执行 PC=14..27
    int pc;
    pc = step_warp(w, v); CHECK(pc == CONV_PC);            // 14
    pc = step_warp(w, v); CHECK(pc == CONV_PC + 1);        // 15
    ...
    pc = step_warp(w, v); CHECK(pc == 27);                 // ret
}
```

### 6.2 新增测试用例

建议添加以下边界测试：

| 测试用例 | 描述 |
|---------|------|
| `test_convergence_blocked_lanes` | 验证 Path A 在 PC=14 被正确阻塞 |
| `test_convergence_multiple_paths` | 验证三层以上嵌套分歧的汇聚 |
| `test_convergence_with_barrier` | 验证隐式汇聚 + 显式 bar.sync 同时存在 |
| `test_convergence_with_yield` | 验证长延迟指令导致的阻塞与汇聚解耦 |

### 6.3 回归测试

修改后必须通过以下现有测试：

```bash
./scripts/sanity.sh --quick
./tests/ptx/test_all_ptx.sh
```

---

## 7. 边界情况处理

### 7.1 多层分歧嵌套

当 SIMT stack depth > 1 时，只检查**最内层**（top entry）的 reconvergence_pc：

```
PC=4:  @%p1 bra $L_B  → push entry {reconv_pc=14}
PC=10: @%p2 bra $L_C  → push entry {reconv_pc=20}
```

- 内层 entry (reconv_pc=20) 检查时，外层 entry (reconv_pc=14) 保持不变
- 当内层 POP 后，外层 entry 成为新的 top，继续检查

### 7.2 `ret` 线程处理

`is_converged()` 已经有处理：

```cpp
if (threads[i].is_exited) {
    continue;  // 跳过已退出线程，不参与汇聚检查
}
```

先 ret 的线程不阻塞其他线程。

### 7.3 `bar.warp.sync` 阻塞共存

两种阻塞机制独立存在：

| 阻塞类型 | 设置 `is_blocked` 的位置 | 恢复条件 |
|---------|------------------------|---------|
| 汇聚阻塞（新增）| `check_and_block_at_reconvergence_point()` | 其他线程到达汇聚点 |
| Barrier 阻塞（已有）| `BarWarpSyncHandler` | 所有线程到达 barrier |

两者都使用 `is_blocked=true`，调度器统一处理。

### 7.4 `bra.uni` 不 push 新 entry

`handle_branch` 对 `bra.uni`（predicate 为空）检测到 `is_divergent=false`，不 push 新 entry：

```cpp
// handle_branch 中的逻辑：
bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
if (is_divergent) {
    simt_stack.push(entry);  // 只有分歧时才 push
}
```

所以 `bra.uni` 跳转到 `reconvergence_pc` 时，直接触发外层 entry 的 `is_converged()` 检测。

### 7.5 线程主动让出（Yield）

`ThreadStatus::Yielded` 用于长延迟操作。当前设计不处理 yield，因为 yield 是独立机制，不应影响汇聚检测。

---

## 8. 审查要点

### 8.1 正确性验证

| 检查项 | 验证方式 |
|--------|---------|
| Path A 在 PC=14 阻塞 | `step_warp` 返回 PC=28 而非 PC=14 |
| 调度器跳过 blocked 组 | `get_lanes_by_pc()` 返回 PC=28 组 |
| Path B 到达后汇聚 | SIMT stack depth 从 1 变为 0 |
| 汇聚后统一执行 | 后续 `step_warp` 返回 PC=14 |

### 8.2 无回归检查

| 模块 | 验证 |
|------|------|
| Barrier 机制 | `test_barrier_*` 系列测试通过 |
| 分歧嵌套 | 多层 bra 测试正常 |
| 线程退出 | ret 后的线程不参与调度 |
| Memory 指令 | ld/st 测试正常 |

### 8.3 潜在风险

| 风险 | 缓解措施 |
|------|---------|
| 死锁（所有 lane 都被阻塞）| 调度器 fallback：若所有组 blocked，选择 lowest PC |
| 性能下降 | 最小化检查成本，只在 target_pc == reconv_pc 时检查 |
| 状态不一致 | `update_active_mask()` 确保 blocked 状态同步到 active_mask |

### 8.4 文档更新

实现完成后需更新：

- `docs/ptx-emu_arch.md` — 更新执行流程图
- `docs/debugging_guide.md` — 添加汇聚相关调试方法
- `docs/architecture/sm90_100.md` — 添加"模拟器实现 vs 真实硬件"对比

---

## 参考文档

- `docs/architecture/sm90_100.md` — Hopper/Blackwell 架构
- `docs/architecture/SIMT-ARCHITECTURE-V2.md` — SIMT v2.0 架构设计
- `docs/adr/ADR-0014-independent-thread-scheduling.md` — ITS 设计 ADR
- `include/ptxsim/simt_stack.h` — SIMT Stack 接口
- `include/ptxsim/warp_context.h` — WarpContext 接口
- `src/ptxsim/core/warp_context.cpp` — WarpContext 实现