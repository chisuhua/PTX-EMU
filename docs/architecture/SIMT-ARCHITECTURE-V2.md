# SIMT 架构 v2.0 设计文档
## ——面向 Hopper/Blackwell 的 SIMT 控制流管理

**版本**: 2.1
**日期**: 2026-04-09
**最后更新**: 2026-05-25
**状态**: ✅ 已完成 (文档已对齐)  
**作者**: PTX-EMU Team

---

## 0. 实现状态与文档对齐说明

> ⚠️ **重要提示**: 本文档于 2026-04-09 编写，描述的是设计期望。后续代码实现有所调整。核心设计决策 (85%) 仍然适用，但部分实现细节已变化。**2026-05-05 已进行文档对齐更新**。

### 0.0 更新日志

| 日期 | 更新内容 |
|------|---------|
| 2026-04-14 | 初始对齐说明添加 |
| 2026-05-05 | 移除已删除的字段描述 (pc_stack, simt_stack_depth, predicate_state, last_issue_cycle) |
| 2026-05-25 | 添加 ADR-0014 引用，相关 ADR 已更新索引 |

### 0.1 关键差异速查

| 文档描述 | 实际实现 | 文件/行 |
|---------|---------|---------|
| `std::vector<SIMTStackEntry> simt_stack` | `ptxsim::SIMTStack simt_stack` (封装类) | `warp_context.h:235` |
| `int simt_stack_depth` (ThreadState) | **不存在** — 已移除 | `thread_state.h` |
| `pc_stack` / `pc_stack_depth` (WarpState) | **不存在** — 已移除，改用 `warp_state.threads[i].pc` | `warp_state.h:20` |
| `std::atomic<bool> memory_fence_complete` (Wbar) | `bool is_initialized` + `bool memory_fence_verification_enabled` | `wbar.h:16-17` |
| `ThreadState::BLOCKED_BARRIER` | `BAR_SYNC` (+ `ThreadStatus::Blocked`) | `thread_context.h` |
| `execute_branch_instruction()` | `handle_branch()` | `warp_context.cpp:8` |
| 从 `ThreadContext::pc` 读取 PC | 从 `warp_state.threads[lane].pc` 读取 (单一权威源) | `warp_context.cpp:298` |

### 0.2 未文档化的关键实现

1. **`advance_thread_pc()` / `advance_all_threads()`** — 统一 PC 更新接口 (`warp_context.cpp:80-99`)
2. **`get_statement_at(target_pc)`** — 指令按 PC 精确查找，替代 `get_current_statement()` (`sm_context.cpp:196`)
3. **`sync_from_warp_state()` / `sync_to_warp_state()`** — ThreadContext ↔ WarpState 双同步机制 (`thread_context.cpp:679-748`)
4. **`is_blocked` 过滤移除** — 防止屏障线程被调度器遗漏 (`warp_context.cpp:298`)
5. **屏障释放使用 `arrived_mask`** — 只更新已到达线程的 PC (`barrier.cpp:167`)
6. **屏障完成后重置 `status = Active`** — 防止线程状态泄漏 (`barrier.cpp:172`)

### 0.3 设计决策状态

| 决策 | 文档 (§7) | 实现状态 |
|------|---------|---------|
| Per-Thread PC | ✅ 选择 | ✅ 完全实现 |
| CFG Post-Dominator | ✅ 选择 | ✅ 完全实现 |
| Debug-Only Barrier 验证 | ✅ 选择 | ✅ 完全实现 |
| SIMT Stack 显式管理 | ✅ 选择 | ✅ 核心功能 |
| Warp-Level Divergence | ✅ 选择 | ✅ 完全支持 |

---

## 1. 执行摘要

本文档描述了 PTX-EMU 项目新一代 SIMT（Single Instruction Multiple Thread）架构的设计，目标是精确模拟 NVIDIA Hopper/Blackwell 架构的控制流管理行为。

### 1.1 设计目标

| 目标 | 当前架构 (v1.0) | 新架构 (v2.0) |
|------|----------------|--------------|
| **PC 管理粒度** | Per-Warp PC | **Per-Thread PC** |
| **收敛机制** | Hardcoded reconvergence | **CFG-based Post-Dominator** |
| **Barrier 语义** | Counting barrier | **Convergence barrier + Memory fence** |
| **Branch 处理** | Immediate resolution | **SIMT Stack with explicit reconvergence** |
| **Divergence 支持** | Limited | **Full support (Hopper/Blackwell)** |

### 1.2 关键创新

1. **显式 SIMT Stack**: 实现硬件级别的控制流栈，跟踪 branch 和 reconvergence 点
2. **CFG 分析**: 编译时计算 Post-Dominator Tree，确定精确的 reconvergence PC
3. **Barrier 验证**: 运行时验证所有前置内存操作已完成
4. **独立线程调度**: 支持 Hopper 的 Independent Thread Scheduling

---

## 2. 理论基础

### 2.1 SIMT 执行模型（NVIDIA 官方定义）

根据 NVIDIA PTX ISA 9.1 文档：

> "PTX supports a SIMT (Single Instruction, Multiple Thread) execution model.
> In SIMT, threads are grouped into warps (typically 32 threads), and all
> threads in a warp execute the same instruction at the same time."

**关键特性**：
- **Lockstep Execution**: Warp 内所有线程执行相同指令
- **Divergence Handling**: 当线程分支时，warp 串行执行每个路径
- **Convergence**: 分支后在 reconvergence point 重新合并

### 2.2 Post-Dominator Theory

**定义**: 在控制流图 (CFG) 中，节点 B post-dominates 节点 A，当且仅当从 A 到 exit 的所有路径都必须经过 B。

**在 SIMT 中的应用**：
```
CFG Example:
     [PC=N: bra target]
        /           \
   [PC=N+1]      [target: PC=M]
       |              |
   [PC=N+2]          |
        \            /
     [PC=R: reconvergence point]

Post-Dominator: R post-dominates both N+1 and M
Reconvergence PC = R
```

### 2.3 Barrier 语义（PTX ISA 规范）

根据 PTX ISA 9.1 Section 9.7.13.1:

```
bar.sync [bar_id], count;

Semantics:
1. Suspend thread execution until all specified threads arrive
2. A memory fence is automatically inserted BEFORE the barrier
3. All memory writes BEFORE bar.sync are visible AFTER barrier
4. ALL threads in CTA must execute the same barrier instruction
```

**关键约束**：
- Barrier 必须在所有发散路径的 reconvergence 点**之后**
- Barrier 隐含 memory fence 语义
- 违反要求的程序行为未定义

---

## 3. 架构设计

### 3.1 总体架构图

```
┌─────────────────────────────────────────────────────────────┐
│                    GPUContext                                │
│  (GPU-level resource management, multi-SM coordination)      │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    SMContext                                 │
│  (SIMT processor, warp schedulers, shared memory)            │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐            │
│  │ Warp #0    │  │ Warp #1    │  │ ...        │            │
│  └────────────┘  └────────────┘  └────────────┘            │
└─────────────────────────────────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│                    WarpContext                               │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ SIMT Stack (Control Flow Management)                  │   │
│  │ ┌────────────────────────────────────────────────┐   │   │
│  │ │ Stack Entry #0 (Top)                           │   │   │
│  │ │ • branch_pc: Branch instruction PC              │   │   │
│  │ │ • reconvergence_pc: Where to converge           │   │   │
│  │ │ • active_mask: Which lanes took this branch     │   │   │
│  │ │ • return_pc: PC to resume after reconvergence   │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  │ ┌────────────────────────────────────────────────┐   │   │
│  │ │ Stack Entry #1                                 │   │   │
│  │ │ ...                                            │   │   │
│  │ └────────────────────────────────────────────────┘   │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Thread State Array (Per-Thread PC)                    │   │
│  │ [Lane 0] [Lane 1] ... [Lane 31]                       │   │
│  │   PC     PC            PC                              │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ Wbar (Convergence Barrier)                            │   │
│  │ • participation_mask: Expected arrivals               │   │
│  │ • arrived_mask: Actual arrivals                       │   │
│  │ • reconvergence_pc: Where threads resume              │   │
│  └──────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### 3.2 核心数据结构

#### 3.2.1 SIMT Stack Entry

```cpp
struct SIMTStackEntry {
    // Branch information
    int branch_pc;              // PC of the branch instruction
    int reconvergence_pc;       // Post-dominator PC (computed by CFG analysis)
    
    // Execution state
    uint32_t active_mask;       // Which lanes are active after branch
    uint32_t return_mask;       // Which lanes should resume at return_pc
    
    // Reconvergence tracking
    int return_pc;              // PC where all lanes reconverge
    
    // Helper methods
    bool is_converged(const WarpState& warp) const;
    uint32_t get_divergent_mask() const;
};
```

#### 3.2.2 WarpContext with SIMT Stack

```cpp
class WarpContext {
public:
    // Per-thread state (NEW in v2.0)
    ThreadState threads[WARP_SIZE];
    
    // SIMT Stack for control flow (NEW in v2.0)
    std::vector<SIMTStackEntry> simt_stack;
    
    // Convergence tracking
    Wbar convergence_barrier;   // For explicit reconvergence
    
    // Execution state
    uint32_t execution_mask;    // Currently executing lanes
    int current_pc;             // Current warp PC (for non-divergent code)
    
    // Methods
    void push_branch(int branch_pc, int reconvergence_pc, uint32_t active_mask);
    void pop_reconvergence();
    bool check_reconvergence();
    void execute_warp_instruction(StatementContext& stmt);
    
private:
    void update_execution_mask();
    int compute_reconvergence_pc(int branch_pc, const std::string& target);
};
```

#### 3.2.3 ThreadState (Per-Thread PC)

```cpp
struct ThreadState {
    // Control flow state
    int pc;                     // Current program counter (per-thread!)
    int next_pc;                // Next PC (for branch resolution)
    bool is_active;             // Is this thread active?
    bool is_blocked;            // Is this thread blocked (barrier, memory)?
    
    // SIMT stack 相关字段已移除 - 使用 WarpContext::simt_stack 替代
    // 注意: simt_stack_depth, predicate_state, last_issue_cycle 字段不存在

    // Execution state
};
```

### 3.3 CFG 分析模块

#### 3.3.1 控制流图构建

```cpp
class CFGBuilder {
public:
    struct BasicBlock {
        int start_pc;
        int end_pc;
        std::vector<int> successors;  // Branch targets
        std::vector<int> predecessors;
        bool is_branch_target;
    };
    
    // Build CFG from kernel statements
    static CFG build(const std::vector<StatementContext>& statements);
    
    // Compute post-dominators
    static std::map<int, int> computePostDominators(const CFG& cfg);
    
private:
    static void identifyBasicBlocks(CFG& cfg);
    static void computeDominators(CFG& cfg);
    static void computePostDominatorsReverse(CFG& cfg);
};
```

#### 3.3.2 Post-Dominator 计算

```cpp
// Algorithm: Iterative Post-Dominator Computation
// Reference: "Simple and Efficient Construction of Static Single Assignment 
//             Forms with Optimal Dominator Frontiers" (Cytron et al.)

std::map<int, int> CFGBuilder::computePostDominators(const CFG& cfg) {
    std::map<int, std::set<int>> postDomSets;
    std::map<int, int> result;
    
    // Initialize: exit block post-dominates everything
    int exit_block = cfg.exit_block_id;
    for (const auto& block : cfg.blocks) {
        if (block.id == exit_block) {
            postDomSets[block.id] = {block.id};
        } else {
            postDomSets[block.id] =getAllBlockIds();  // Start with all blocks
        }
    }
    
    // Iterate until fixed point
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& block : cfg.blocks) {
            if (block.id == exit_block) continue;
            
            // Post-dom set = intersection of successors' post-dom sets + self
            std::set<int> newSet = {block.id};
            for (int succ_id : block.successors) {
                std::set<int> intersection;
                std::set_intersection(
                    newSet.begin(), newSet.end(),
                    postDomSets[succ_id].begin(), postDomSets[succ_id].end(),
                    std::inserter(intersection, intersection.begin())
                );
                newSet = intersection;
            }
            
            if (newSet != postDomSets[block.id]) {
                postDomSets[block.id] = newSet;
                changed = true;
            }
        }
    }
    
    // Extract immediate post-dominator (closest post-dominator)
    for (const auto& block : cfg.blocks) {
        result[block.start_pc] = findImmediatePostDominator(cfg, block, postDomSets);
    }
    
    return result;
}
```

### 3.4 Barrier 机制

#### 3.4.1 Wbar (Convergence Barrier)

```cpp
class Wbar {
public:
    // Barrier state
    int barrier_id;
    int reconvergence_pc;
    uint32_t participation_mask;  // Which threads must arrive
    uint32_t arrived_mask;        // Which threads have arrived

    // Memory fence state (2026-05-06: 已更新为实际实现)
    bool is_initialized = false;
    bool memory_fence_verification_enabled = false;

    // Lifecycle
    void init(int _reconvergence_pc, uint32_t _participation_mask);
    void arrive(int lane_id);
    bool is_complete() const;
    void reset();

    // Verification (debug only)
    void enable_memory_fence_verification(bool enable);
    bool is_memory_fence_verification_enabled() const;
    void verify_memory_fence() const;

private:
    // Track memory operations before barrier
    #ifdef PTX_DEBUG
    std::vector<std::pair<int, uint64_t>> pre_barrier_stores;
    #endif
};
```

#### 3.4.2 Barrier 验证

```cpp
void Wbar::enable_memory_fence_verification(bool enable) {
    memory_fence_verification_enabled = enable;
}

void Wbar::verify_memory_fence() const {
    if (!memory_fence_verification_enabled) return;

    // Verify all pre-barrier stores are visible
    for (const auto& store : pre_barrier_stores) {
        int lane_id = store.first;
        uint64_t addr = store.second;

        // Check if store is visible to all participating lanes
        for (int i = 0; i < 32; i++) {
            if (participation_mask & (1u << i)) {
                if (!is_store_visible(i, addr)) {
                    PTX_ERROR("Lane %d store to 0x%lx not visible to lane %d!",
                              lane_id, addr, i);
                }
            }
        }
    }
}
#endif
```

---

## 4. 执行流程

### 4.1 SIMT 指令执行流程

```
Instruction Execute Flow (v2.0):

1. Warp Fetch
   ↓
   simt_stack.top() → check if converged
   ↓
2. Thread Dispatch
   ↓
   for each active lane:
     - Check if lane pc == current_pc
     - If yes: execute instruction
     - If no: skip (lane is at different PC)
   ↓
3. Branch Resolution
   ↓
   if instruction is BRA:
     - Check predicate for each lane
     - Compute active_mask for taken/not-taken
     - if divergence detected:
       push_SIMT_stack(branch_pc, reconvergence_pc, active_mask)
     - Set lane pc = target_pc (taken) or next_pc (not-taken)
   ↓
4. Reconvergence Check
   ↓
   if simt_stack not empty:
     check if all lanes at reconvergence_pc
     if yes:
       pop_SIMT_stack()
       all lanes now execute in lockstep
   ↓
5. Barrier Handling
   ↓
   if instruction is BAR:
     - All active lanes call Wbar.arrive()
     - Wait for is_complete()
     - Verify memory fence
     - All lanes resume at reconvergence_pc
```

### 4.2 Branch 处理详细流程

```cpp
void WarpContext::execute_branch_instruction(StatementContext& stmt) {
    const BranchInstr& branch = std::get<BranchInstr>(stmt.data);
    
    // Step 1: Evaluate predicate for each lane
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!threads[i].is_active) continue;
        
        bool pred_value = evaluate_predicate(i, branch.predicate);
        if (branch.predicate_negated) pred_value = !pred_value;
        
        if (pred_value) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }
    
    // Step 2: Check for divergence
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    
    if (is_divergent) {
        // Step 3a: Push SIMT stack entry
        SIMTStackEntry entry;
        entry.branch_pc = stmt.pc;
        entry.reconvergence_pc = branch.reconvergence_pc;
        entry.active_mask = taken_mask;  // Lanes that took the branch
        entry.return_mask = execution_mask;  // All lanes before branch
        entry.return_pc = branch.reconvergence_pc;
        
        simt_stack.push_back(entry);
        
        PTX_DEBUG_EMU("[SIMT:STACK] PUSH branch_pc=%d reconvergence_pc=%d taken_mask=0x%x",
                      entry.branch_pc, entry.reconvergence_pc, taken_mask);
        
        // Step 4a: Set PC for divergent lanes
        for (int i = 0; i < WARP_SIZE; i++) {
            if (taken_mask & (1u << i)) {
                threads[i].pc = get_label_pc(branch.target);
            } else if (not_taken_mask & (1u << i)) {
                threads[i].next_pc = stmt.pc + 1;
            }
        }
        
    } else {
        // Step 3b: Non-divergent - all lanes take same path
        int target_pc = (taken_mask != 0) ? get_label_pc(branch.target) : stmt.pc + 1;
        for (int i = 0; i < WARP_SIZE; i++) {
            if (threads[i].is_active) {
                threads[i].pc = target_pc;
            }
        }
        
        PTX_DEBUG_EMU("[SIMT:BRANCH] Non-divergent, all lanes to PC=%d", target_pc);
    }
}
```

### 4.3 Reconvergence 检查

```cpp
bool WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return true;
    
    SIMTStackEntry& top = simt_stack.back();
    
    // Check if all lanes from return_mask have reached reconvergence_pc
    bool all_converged = true;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (top.return_mask & (1u << i)) {
            if (threads[i].pc != top.reconvergence_pc) {
                all_converged = false;
                PTX_DEBUG_EMU("[SIMT:RECONV] Lane %d not converged (pc=%d, expected=%d)",
                              i, threads[i].pc, top.reconvergence_pc);
                break;
            }
        }
    }
    
    if (all_converged) {
        // Pop the stack - lanes are now in lockstep
        simt_stack.pop_back();
        execution_mask = top.return_mask;
        
        PTX_DEBUG_EMU("[SIMT:RECONV] All lanes converged at PC=%d", top.reconvergence_pc);
    }
    
    return all_converged;
}
```

### 4.4 Barrier 执行流程

```cpp
void BarWarpSyncHandler::execute(ThreadContext* context, const BarrierInstr& instr) {
    WarpContext* warp = context->warp_context_;
    
    // Step 1: All participating lanes arrive at barrier
    int lane_id = context->lane_id_;
    warp->convergence_barrier.arrive(lane_id);
    
    PTX_DEBUG_EMU("[BAR:ARRIVE] Lane %d arrived at barrier %d", 
                  lane_id, instr.barrier_id);
    
    // Step 2: Check if barrier is complete
    if (!warp->convergence_barrier.is_complete()) {
        // Block this thread until barrier completes
        context->state = ThreadState::BLOCKED_BARRIER;
        return;
    }
    
    // Step 3: Barrier complete - verify memory fence
    #ifdef PTX_DEBUG
    warp->convergence_barrier.verify_memory_fence();
    #endif
    
    // Step 4: All lanes resume at reconvergence_pc
    warp->convergence_barrier.reset();
    
    for (int i = 0; i < WARP_SIZE; i++) {
        if (warp->threads[i].is_active && 
            warp->threads[i].state == ThreadState::BLOCKED_BARRIER) {
            warp->threads[i].state = ThreadState::ACTIVE;
            warp->threads[i].pc = warp->convergence_barrier.reconvergence_pc;
        }
    }
    
    PTX_DEBUG_EMU("[BAR:COMPLETE] All lanes released at PC=%d",
                  warp->convergence_barrier.reconvergence_pc);
}
```

### 4.4 Hopper/Blackwell 分歧路径执行顺序

> **重要前提：PTX 是虚拟 ISA**
>
> ```
> PTX (虚拟指令集)
>     │
>     │  ptxas 编译器
>     ▼
> SASS (实际机器码，包含 BSSY/BSYNC)
>     │
>     │  GPU 硬件执行
>     ▼
> 实际执行行为
> ```
>
> **PTX 层面不直接暴露 `BSSY`/`BSYNC`**，这些是 ptxas 编译器在生成 SASS 时自动插入的。但 PTX 规范定义了**可观测行为（observable behavior）**，我们基于此来描述执行顺序。

#### 4.4.1 PTX 指令序列示例

为方便标注，给每条指令编号：

```ptx
// ===== 公共前导 =====
I01: ld.param.u64 %rd1, [_Z27..._param_0]
I02: mov.u32 %r1, %tid.x
I03: and.b32 %r2, %r1, 31
I04: setp.lt.u32 %p1, %r2, 16
I05: @%p1 bra $L__BB0_4                    // 分歧点

// ===== Path B (fall-through, lane >= 16) =====
I06: neg.s32 %r18, %r2
I07: mov.b32 %r20, 1
I08: mov.b32 %r17, 15
    $L__BB0_2:                              // 循环入口
I09: add.s32 %r14, %r17, -14
I10: mul.lo.s32 %r20, %r14, %r20
I11: add.s32 %r18, %r18, 1
I12: add.s32 %r17, %r17, 1
I13: setp.ne.s32 %p2, %r18, -15
I14: @%p2 bra $L__BB0_2                    // 循环回边

// ===== 公共出口 =====
    $L__BB0_3:
I15: cvta.to.global.u64 %rd4, %rd1
I16: bar.sync 0
I17: mul.wide.u32 %rd5, %r1, 4
I18: add.s64 %rd6, %rd4, %rd5
I19: st.global.u32 [%rd6], %r20
I20: ret

// ===== Path A (taken, lane < 16) =====
    $L__BB0_4:
I21: add.s32 %r15, %r2, -1
I22: mul.wide.u32 %rd2, %r15, %r2
I23: shr.u64 %rd3, %rd2, 1
I24: cvt.u32.u64 %r16, %rd3
I25: add.s32 %r20, %r2, %r16
I26: bra.uni $L__BB0_3                     // 跳到公共出口
```

#### 4.4.2 ptxas 编译为 SASS 时的隐式插入

ptxas 将上述 PTX 编译为 Hopper/Blackwell SASS 时，会在关键位置插入收敛屏障指令：

```
PTX I05: @%p1 bra $L__BB0_4
                │
                │ ptxas 分析控制流图，找到 post-dominator = $L__BB0_3
                ▼
SASS 生成:
    BSSY B0, $L__BB0_3     ← 自动插入（设置收敛点）
    @P0 BRA $L__BB0_4

PTX I14 (循环最后一次不跳转后) → 下一条是 I15 ($L__BB0_3)
                │
                ▼
SASS 生成:
    @P2 BRA $L__BB0_2
    BSYNC B0                ← 自动插入（Path B 线程到达屏障）

PTX I26: bra.uni $L__BB0_3
                │
                ▼
SASS 生成:
    BSYNC B0                ← 自动插入（Path A 线程到达屏障）
    // bra.uni 被优化掉，因为 BSYNC 后直接从 target 恢复
```

#### 4.4.3 Hopper/Blackwell 实际执行顺序

**关键原则：执行顺序是非确定性的**

PTX 规范对 Hopper/Blackwell 的保证是：

1. 同一线程内的指令按程序顺序执行
2. 分歧路径的线程**最终**会在 reconvergence point 汇合
3. 不同路径的相对执行顺序**无保证**

**以 32 线程 Warp 为例，可能的执行场景：**

**场景 1：类 Pre-Volta 顺序（最常见的实际行为）**

```
时间步    活跃线程         执行的 PTX 指令         说明
─────────────────────────────────────────────────────────────
t1       T0-T31 (32)     I01: ld.param           全部线程统一执行
t2       T0-T31 (32)     I02: mov %tid.x         ↓
t3       T0-T31 (32)     I03: and                ↓
t4       T0-T31 (32)     I04: setp               ↓
t5       T0-T31 (32)     I05: @%p1 bra           分歧发生，Scheduler 选择路径
─────────────────────────── 分歧开始 ──────────────────────────
t6       T16-T31 (16)    I06: neg                Scheduler 选择执行 Path B
t7       T16-T31 (16)    I07: mov 1              ↓
t8       T16-T31 (16)    I08: mov 15             ↓
t9       T16-T31 (16)    I09: add (iter 1)       循环第 1 次
t10      T16-T31 (16)    I10: mul               ↓
t11      T16-T31 (16)    I11: add               ↓
t12      T16-T31 (16)    I12: add                ↓
t13      T16-T31 (16)    I13: setp              ↓
t14      T16-T31 (16)    I14: @%p2 bra           T16 退出循环(p2=false)
                                                   其余继续
t15      T17-T31 (15)    I09: add (iter 2)       T16 已挂起(BSYNC)
t16      T17-T31 (15)    I10: mul               ↓
...                                               （循环继续）
t??      T31 (1)         I09-I14 (iter 16)       最后一个线程完成
                                                   → BSYNC → 挂起
─────────────── Path B 全部线程到达屏障 ────────────────────────
t??+1    T0-T15 (16)     I21: add               Scheduler 切换到 Path A
t??+2    T0-T15 (16)    I22: mul.wide           ↓
t??+3    T0-T15 (16)    I23: shr               ↓
t??+4    T0-T15 (16)    I24: cvt               ↓
t??+5    T0-T15 (16)    I25: add               ↓
                          (I26: bra.uni)         → BSYNC → 挂起
─────────────── Path A 全部线程到达屏障 ────────────────────────
                          屏障 B0 释放            32 线程全部恢复
─────────────────────── 汇合 ──────────────────────────────────
t??+6    T0-T31 (32)     I15: cvta              统一执行
t??+7    T0-T31 (32)    I16: bar.sync 0         block 级同步
t??+8    T0-T31 (32)    I17: mul.wide          ↓
t??+9    T0-T31 (32)    I18: add               ↓
t??+10   T0-T31 (32)    I19: st.global         ↓
t??+11   T0-T31 (32)    I20: ret               结束
```

**场景 2：交错执行（Hopper/Blackwell 独有能力）**

```
时间步    活跃线程         执行的 PTX 指令         说明
─────────────────────────────────────────────────────────────
t1-t5    T0-T31 (32)     I01-I05                 公共前导（同场景 1）
─────────────────────────── 分歧开始 ──────────────────────────
t6-t8    T16-T31 (16)    I06-I08                 Path B 初始化
t9-t14   T16-T31 (16)    I09-I14 (iter 1)        Path B 第 1 次循环
         T16 → BSYNC                             T16 完成，挂起

★ Scheduler 决策：Path B 只剩 15 线程活跃，
   切换到 Path A 可以利用空闲的执行通道

t15      T0-T15 (16)    I21: add                 切换执行 Path A！
t16      T0-T15 (16)    I22: mul.wide           ↓
t17      T0-T15 (16)    I23: shr               ↓
t18      T0-T15 (16)    I24: cvt               ↓
t19      T0-T15 (16)    I25: add               ↓
         T0-T15 → BSYNC                         Path A 全部完成，挂起

★ 此时 T0-T15 都已到达屏障，但 T17-T31 仍在循环
   屏障未释放（需要全部 32 线程到达）

t20      T17-T31 (15)    I09-I14 (iter 2)       继续 Path B 循环
...                                               （T17 完成第 2 次后挂起）
...                                               （逐步减少活跃线程）
t??      T31 (1)         I09-I14 (iter 16)       最后一个线程
         T31 → BSYNC                             屏障条件满足！

─────────────────────── 屏障释放 ───────────────────────────────
t??+1    T0-T31 (32)     I15: cvta              统一执行
...
t??+6    T0-T31 (32)     I20: ret               结束
```

**场景 3：Path A 优先（Blackwell 启发式优化）**

Blackwell 的 Scheduler 可能检测到 Path A 更短（无循环，5 条指令），优先执行：

```
时间步    活跃线程         执行的 PTX 指令         说明
─────────────────────────────────────────────────────────────
t1-t5    T0-T31 (32)     I01-I05                 公共前导
─────────────────────────── 分歧开始 ──────────────────────────
★ Blackwell Scheduler 启发式：Path A 更短，优先调度

t6       T0-T15 (16)    I21: add                 先执行 Path A
t7       T0-T15 (16)    I22: mul.wide           ↓
t8       T0-T15 (16)    I23: shr               ↓
t9       T0-T15 (16)    I24: cvt               ↓
t10      T0-T15 (16)    I25: add               ↓
         T0-T15 → BSYNC                         Path A 完成，挂起

t11-t??  T16-T31 (16)    I06-I08, 循环           再执行 Path B（完整循环）
         T16-T31 逐步 → BSYNC

─────────────────────── 屏障释放 ───────────────────────────────
t??+1    T0-T31 (32)     I15-I20                 公共出口
```

#### 4.4.4 PTX 层面的可观测保证（Hopper/Blackwell 内存模型）

**跨路径可见性：**

```ptx
// Thread A (lane=5, Path A):
I25: add.s32 %r20, %r2, %r16       // 计算 value
     // ... 到达收敛点 ...

// Thread B (lane=20, Path B):
I10: mul.lo.s32 %r20, %r14, %r20   // 计算 value
     // ... 到达收敛点 ...

// 汇合后:
I16: bar.sync 0                     // ← 这里保证所有线程看到一致状态
I19: st.global.u32 [%rd6], %r20    // 每个线程写自己的 %r20，安全
```

**PTX 对执行顺序的规范声明：**

```
PTX ISA Spec (8.x+) 关于分歧的保证:

1. "Different execution paths in diverged threads will eventually 
    reconverge" (分歧线程最终会汇合)

2. "The order of execution of different paths is UNSPECIFIED" 
   (不同路径的执行顺序未定义)

3. "Programs must not rely on any particular execution order of 
    divergent paths" (程序不得依赖分歧路径的特定执行顺序)

4. "bar.sync guarantees that all threads in the CTA have reached 
    the barrier" (bar.sync 保证 CTA 内所有线程到达)
```

#### 4.4.5 PTX 指令到硬件行为的映射表

| PTX 指令 | Hopper/Blackwell 硬件行为 |
|----------|--------------------------|
| `@%p1 bra $target` | 若分歧：触发 Scheduler 线程分组；若无分歧：普通跳转 |
| `bra.uni $target` | 所有活跃线程统一跳转（无分歧开销）。若在分歧区域末尾，编译为 BSYNC |
| `@%p2 bra $loop` (循环) | 每次迭代可能导致部分线程退出循环（退出的线程提前到达 BSYNC） |
| `bar.sync 0` | Block 级全同步。已汇合的 Warp 正常通过；未汇合的 Warp 会在此被**强制汇合** |
| `ret` | 线程终止。注意：若 Warp 中部分线程先 ret，其他还在执行，先 ret 的线程被标记为 terminated |

#### 4.4.6 关键总结

```
┌────────────────────────────────────────────────────────────────────┐
│  在 PTX 层面，Hopper/Blackwell 的执行模型可以这样理解：            │
│                                                                     │
│  1. 公共前导 I01-I05：所有线程「同时」执行（一条接一条，锁步）     │
│                                                                     │
│  2. 分歧点 I05 之后：                                              │
│     - Path A (I21-I25+I26) 和 Path B (I06-I14循环)                 │
│     - 以「某种不确定的顺序」被 Scheduler 调度执行                   │
│     - 可能全 B 先、全 A 先、或交错                                  │
│     - PTX 程序不得假设任何特定顺序                                  │
│                                                                     │
│  3. 汇合点 $L__BB0_3 (I15)：                                      │
│     - 硬件保证两条路径的线程在此重新统一                            │
│     - 之后 I15-I20 再次「同时」执行                                 │
│                                                                     │
│  4. bar.sync (I16)：提供 Block 级同步+内存可见性保证               │
│                                                                     │
│  5. 性能提示（非功能保证）：                                        │
│     - Hopper: 倾向于按 fall-through 优先                           │
│     - Blackwell: 倾向于短路径优先 + 动态交错                       │
└────────────────────────────────────────────────────────────────────┘
```

**本质差异：PTX 不规定分歧路径的执行顺序。Pre-Volta 的 SIMT Stack 带来了确定性顺序（虽然 spec 也没保证），但 Hopper/Blackwell 的 Independent Thread Scheduling 让这种非确定性成为现实中可观测的行为。程序正确性必须依赖显式同步（`bar.sync`、`__syncwarp()`），不能依赖路径执行顺序。**

---

## 5. 与 PTX ISA 的映射

### 5.1 PTX 控制流指令映射

| PTX 指令 | v2.0 处理 | SIMT Stack 操作 |
|----------|----------|-----------------|
| `bra.cond target` | Branch | Push if divergent |
| `bar.sync id` | Convergence Barrier | Verify all lanes arrived |
| `bar.warp.sync mask` | Warp Barrier | Use participation_mask |
| `exit` | Thread Termination | Mark lane inactive |
| `ret` | Function Return | Pop call stack |

### 5.2 PTX → 内部表示

```ptx
// Example PTX
.visible .entry test_divergence() {
.reg .pred %p<1>;
.reg .b32 %r<1>;

mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 16;
@%p1 bra target;

// Path A (lane >= 16)
add.u32 %r1, %r1, 1;
bra.uni merge;

target:
// Path B (lane < 16)
sub.u32 %r1, %r1, 1;

merge:
bar.sync 0;
ret;
}
```

**内部表示**:

```
PC=0: mov.u32 %r1, %tid.x              (all lanes)
PC=1: setp.lt.u32 %p1, %r1, 16         (all lanes)
PC=2: @%p1 bra target  [reconvergence=PC=5]
      → Divergent! Push SIMT stack: {branch=2, reconvergence=5, taken_mask=0xFFFF}
PC=3: add.u32 %r1, %r1, 1              (lanes 16-31 only)
PC=4: bra.uni merge                    (lanes 16-31 only)
PC=5: ← RECONVERGENCE POINT (all lanes)
      → Pop SIMT stack
PC=6: sub.u32 %r1, %r1, 1              (lanes 0-15 only, already executed)
PC=7: bar.sync 0                       (all lanes)
      → Wbar.arrive() for all 32 lanes
      → Wait for is_complete()
      → Release all lanes to PC=8
PC=8: ret                              (all lanes)
```

---

## 6. 与 GPGPU-Sim 的对比

| 特性 | GPGPU-Sim | PTX-EMU v2.0 |
|------|-----------|--------------|
| **SIMT Stack** | ✅ Hardware stack | ✅ Hardware stack |
| **Per-Thread PC** | ✅ Implicit via stack | ✅ Explicit ThreadState |
| **Post-Dominator** | ✅ CFG analysis | ✅ CFG analysis |
| **Barrier** | ✅ Counting + membar | ✅ Convergence + membar |
| **Reconvergence** | ✅ Explicit check | ✅ Explicit check |
| **Memory Fence** | ✅ Implicit | ✅ Verified (debug) |

---

## 7. 设计决策与权衡

### 7.1 Per-Thread PC vs Per-Warp PC

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|
| **Per-Warp PC** | 简单，内存占用小 | 无法精确跟踪 divergent 执行 | ❌ |
| **Per-Thread PC** | 精确模拟硬件行为 | 内存占用大 (32x) | ✅ |

**决策理由**: Hopper/Blackwell 支持 Independent Thread Scheduling，必须实现 per-thread PC。

### 7.2 CFG 分析时机

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|------|
| **编译时 (PTX → IR)** | 运行时开销小 | 增加 parser 复杂度 | ❌ |
| **运行时 (首次执行)** | 动态适应 | 首次执行延迟 | ❌ |
| **Parser 阶段 (kernel 加载)** | 平衡复杂度和性能 | 需要 CFG 基础设施 | ✅ |

**决策理由**: Kernel 加载阶段做 CFG 分析，一次计算多次使用，无运行时开销。

### 7.3 Barrier 验证范围

| 方案 | 优点 | 缺点 | 选择 |
|------|------|------|------|------|
| **无验证** | 最快 | 无法检测语义错误 | ❌ |
| **Debug Only** | 开发友好，性能影响小 | Production 无保护 | ✅ |
| **Always On** | 完整保护 | 性能开销 (估计 5-10%) | ❌ |

**决策理由**: Barrier 验证仅在 PTX_DEBUG 模式下启用，平衡开发和生产需求。

---

## 8. 实现路线图

### Phase 1: CFG 分析基础设施
- 基本块识别
- 控制流图构建
- Post-dominator 计算
- **预计**: 2 天

### Phase 2: SIMT Stack 实现
- Stack entry 数据结构
- Push/pop 操作
- Reconvergence 检查
- **预计**: 2 天

### Phase 3: Per-Thread PC 集成
- ThreadState 重构
- WarpContext 更新
- 调度器适配
- **预计**: 2 天

### Phase 4: Barrier 增强
- Wbar 与 SIMT stack 集成
- Memory fence 验证
- Debug 模式支持
- **预计**: 1 天

### Phase 5: 测试与验证
- 单元测试
- 集成测试
- 性能基准
- **预计**: 2 天

**总计**: 9 天

---

## 9. 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CFG 构建错误 | 中 | 高 | 单元测试覆盖所有分支模式 |
| 性能回归 | 低 | 中 | Benchmark 对比，优化热点 |
| 测试失败增加 | 中 | 高 | 详细日志，快速定位 |
| 需要重构 | 低 | 高 | 分 Phase 验证，每阶段可回滚 |

---

## 10. 验收标准

### 10.1 功能验收

- [ ] `test_warp_divergence` 3/3 PASS
- [ ] `test_syncthreads` 3/3 PASS
- [ ] `test_simt_reconvergence` (new) 5/5 PASS
- [ ] `test_barrier_semantics` (new) 3/3 PASS

### 10.2 性能验收

- [ ] 无 regression (> 5% 视为 regression)
- [ ] CFG 分析开销 < 1ms per kernel
- [ ] SIMT stack 操作开销 < 1 cycle per instruction

### 10.3 代码质量

- [ ] LSP diagnostics clean
- [ ] 所有新增代码通过 clang-tidy
- [ ] 单元测试覆盖率 > 80%

---

## 11. 参考文档

1. NVIDIA PTX ISA 9.1 Documentation
2. GPGPU-Sim 3.x Source Code
3. "Control Flow Management in Modern GPUs" (arXiv:2407.02944)
4. "Stack-less SIMT Reconvergence at Low Cost" (HAL:00622654)
5. "Divergence-Aware Warp Scheduling" (MICRO-2013)

### 相关 ADR

| ADR | 标题 | 状态 |
|-----|------|------|
| [ADR-0014](../adr/0014-independent-thread-scheduling.md) | Independent Thread Scheduling (ITS) 支持 | Proposed |

---

## 附录 D: 已实现但未文档化的功能 (2026-05-05 添加)

以下功能在代码中已实现，但本文档未详细描述：

### D.1 双向同步机制

```cpp
// ThreadContext → WarpState 同步
void ThreadContext::sync_to_warp_state();

// WarpState → ThreadContext 同步
void ThreadContext::sync_from_warp_state();
```

**位置**: `thread_context.cpp:679-748`
**用途**: 保持 ThreadContext 和 WarpState 状态一致

### D.2 统一 PC 更新接口

```cpp
// 单线程 PC 更新
void WarpContext::advance_thread_pc(int lane_id, int new_pc);

// 所有活跃线程 PC 更新
void WarpContext::advance_all_threads(int new_pc);
```

**位置**: `warp_context.cpp:80-99`
**用途**: 提供统一的 PC 更新接口，替代分散的 PC 设置

### D.3 屏障后 SIMT 栈清理

`sm_context.cpp` 中对 `S_BAR` 和 `S_BAR_WARP_SYNC` 指令调用 `check_reconvergence()`，确保屏障完成后检查 SIMT 栈收敛。

**位置**: `sm_context.cpp:203-206`
**代码**:
```cpp
if (stmt->type == S_BRA || stmt->type == S_BAR ||
    stmt->type == S_BAR_WARP_SYNC) {
    next_warp->check_reconvergence();
}
```

### D.4 退出线程收敛跳过

`SIMTStackEntry::is_converged()` 会跳过 `is_exited=true` 或 `is_active=false` 的线程，不阻塞收敛检查。

**位置**: `simt_stack.cpp:7-19`
**代码**:
```cpp
if (threads[i].is_exited || !threads[i].is_active) {
    continue;  // 跳过退出线程
}
```

### D.5 SIMT 栈深度限制

防止无限嵌套分支导致栈溢出：

```cpp
static constexpr size_t MAX_DEPTH = 32;
```

**位置**: `simt_stack.h`
**限制**: 最多 32 层嵌套

### D.6 废弃 API 标记

| 废弃字段/方法 | 替代方案 | 位置 |
|-------------|---------|------|
| `WarpContext::pc_stacks` | `warp_state.threads[i].pc` | warp_context.h |
| `WarpContext::pc` | `warp_state.warp_pc` | warp_context.h |
| `update_pc_stack()` | `warp_state.threads[i].pc = new_pc` | warp_context.h |
| `handle_branch_divergence()` | `advance_thread_pc()` | warp_context.h |

---

**文档状态**: ✅ 已完成 (文档已对齐)
**最后更新**: 2026-05-05
**评审者**: PTX-EMU Architecture Team
