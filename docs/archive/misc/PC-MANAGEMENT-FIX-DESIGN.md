# PTX-EMU PC 管理架构修复设计

> **日期**: 2026-05-04
> **状态**: 设计阶段
> **Sprint**: 11.2

---

## 1. 背景

### 1.1 Sprint 11.2 进度

| Commit | 任务 | 状态 |
|--------|------|------|
| `92f7585` | 修改 ThreadContext 结构：移除 pc/next_pc 字段，使用 WarpState | ✅ |
| `bed8993` | 修复 barrier PC 回归 bug | ✅ |
| `c7d4f8b` | 添加 commit_pc() 作为正常 PC 推进的唯一入口 | ✅ |
| **本文档** | T11.2.3：修复 PC 管理架构，使其符合 SIMT 硬件行为 | 进行中 |

### 1.2 问题描述

当前 `commit_pc()` 虽然已实现，但仍存在以下问题：

1. **`BraHandler::executeBranch()` (control.cpp:25) 直接调用 `set_pc()`** — 虽然不改变结果（因为 `handle_branch()` 已正确设置 `next_pc`），但违反了"只有 `commit_pc()` 能推进 PC"的约定

2. **`BarWarpSyncHandler` barrier 完成时当前线程 PC 被错误覆盖** — PipelineHandler::ExecPipe 在 `processOperation()` 调用 `set_thread_pc()` 后又执行 `set_next_pc(saved_pc + 1)`，导致 `commit_pc()` 把当前线程的 PC 从 `reconvergence_pc` 覆盖回 `barrier_pc + 1`

3. **未实现调度器 stall 逻辑** — 如果 `commit_pc()` 未被执行，调度器应自然 stall（而不是盲目调度）

---

## 2. 硬件行为基准

### 2.1 PC 更新时机

**NVIDIA GPU** (Ampere+):
- PC 在指令执行**完成后**更新
- Fetch → Decode → Issue → Control → Execute → **[PC ← PC + 1]** → Fetch(下一条)
- Per-thread PC (Volta+)，Per-warp PC (早期)

**AMD GPU** (RDNA):
- 类似行为：指令完成后 PC + 1
- Per-wavefront PC

### 2.2 Barrier 行为

| 方面 | NVIDIA | AMD |
|------|--------|-----|
| 指令 | `bar.warp.sync`, `barrier.cta.sync` | `s_barrier` |
| Warp/Wave stall | ✅ 是 — warp stall 直到所有线程到达 | ✅ 是 — wave stall |
| 调度器行为 | 选择其他已就绪 warp | 选择其他已就绪 wavefront |
| Fetch 基于 | PC 值（未更新则 stall） | PC 值 |

### 2.3 关键结论

> **PC 更新是执行结果，不是执行前提。**
>
> 如果 `commit_pc()` 未执行：
> - warp_state.pc 未更新
> - 调度器下次取指仍取到同一 PC
> - **自然 stall**，无需手动引入 stall cycles

---

## 3. 当前代码分析

### 3.1 PC 相关方法调用关系

```
ThreadContext:
  get_pc()      → warp_state.threads[lane].pc       (读取)
  set_pc()      → warp_state.threads[lane].pc = next_pc = new_pc  (直接覆盖)
  get_next_pc() → warp_state.threads[lane].next_pc  (读取)
  set_next_pc() → warp_state.threads[lane].next_pc = new_next_pc  (设置)
  commit_pc()   → set_pc(get_next_pc())              (PC ← next_pc)

_execute_once() 流程:
  1. get_pc()           ← 读取当前 PC
  2. set_next_pc(pc+1) ← 设置 next_pc 为下一条
  3. ExecPipe()         ← 执行指令（可能修改 next_pc 或直接修改 pc）
  4. commit_pc()        ← PC ← next_pc（正常执行路径）

合法 set_pc() 调用者（仅用于初始化/同步/重置）:
  - reset()              → 初始化
  - sync_from_warp_state() → WarpState → ThreadContext 同步
  - BraHandler (control.cpp:25) → 分支重定向
  - BarWarpSyncHandler (barrier.cpp:158) → barrier 完成后设置其他线程 PC
```

### 3.2 问题 1: BraHandler 冗余调用

**文件**: `src/ptxsim/instructions/control.cpp:25`

```cpp
void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    WarpContext* warp_ctx = context->get_warp_context();
    // ...
    warp_ctx->handle_branch(...);  // 已正确设置 ALL 线程的 pc 和 next_pc
    context->set_pc(warp_ctx->get_thread_pc(context->lane_id_));  // ← 冗余！
}
```

**分析**:
- `handle_branch()` 已直接写入 `warp_state.threads[i].pc` 和 `.next_pc`
- 第 25 行 `set_pc()` 再次覆盖当前线程的 `pc` 和 `next_pc`（值相同）
- 随后 `_execute_once()` 末尾 `commit_pc()` 执行 `set_pc(get_next_pc())`
- 由于 `next_pc` 已在 `handle_branch()` 中设置为目标值，`commit_pc()` 结果正确

**修复**: 删除 `control.cpp:25` 的 `set_pc()` 调用

### 3.3 问题 2: BarWarpSyncHandler PC 覆盖

**文件**: `src/ptxsim/instruction_base.cpp:75-104`

```cpp
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int saved_pc = context->get_pc();  // 保存原始 barrier PC

    prepareOperands(...);
    executeOperation(...);  // 调用 processOperation()
      // processOperation() 调用 set_thread_pc(i, reconvergence_pc)
      // 设置 ALL 到达线程（包括当前线程）的 pc = next_pc = reconvergence_pc

    commitResults(...);

    // 问题：这里覆盖了当前线程的 next_pc！
    context->set_next_pc(saved_pc + 1);  // next_pc = barrier_pc + 1
}

void ThreadContext::_execute_once() {
    // ...
    commit_pc();  // set_pc(get_next_pc()) → pc = barrier_pc + 1（错误！）
}
```

**分析**:
- `processOperation()` 对当前线程调用 `set_thread_pc(current_lane, reconvergence_pc)`
- 当前线程: `pc = reconvergence_pc`, `next_pc = reconvergence_pc`
- PipelineHandler 第 103 行: `set_next_pc(saved_pc + 1)` → `next_pc = barrier_pc + 1`
- `commit_pc()`: `pc = next_pc = barrier_pc + 1` ❌（应该是 `reconvergence_pc`）

**影响**: 当前线程（最后一个到达 barrier 的线程）最终 PC = barrier 指令下一条，而非 reconvergence PC

**修复方案**: 见第 4 节

---

## 4. 修复方案

### 4.1 方案概述

| 步骤 | 描述 | 涉及文件 |
|------|------|----------|
| 4.2 | 删除 BraHandler 冗余的 `set_pc()` 调用 | control.cpp |
| 4.3 | 修复 BarWarpSyncHandler 的 PC 覆盖问题 | instruction_base.cpp |
| 4.4 | 实现调度器 stall 逻辑 | warp_context.cpp |
| 4.5 | 添加调试日志追踪 PC 变化 | thread_context.cpp |
| 4.6 | 验证修复正确性 | 测试 |

### 4.2 删除 BraHandler 冗余调用

**文件**: `src/ptxsim/instructions/control.cpp`

```cpp
// 修改前
void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    WarpContext* warp_ctx = context->get_warp_context();
    int target_pc = /* resolve label */;
    warp_ctx->handle_branch(predicate, predicate_negated, target_pc,
                            instr.reconvergence_pc, context->get_pc());
    context->set_pc(warp_ctx->get_thread_pc(context->lane_id_));  // ← 删除此行
}

// 修改后
void BraHandler::executeBranch(ThreadContext *context, const BranchInstr &instr) {
    WarpContext* warp_ctx = context->get_warp_context();
    int target_pc = /* resolve label */;
    warp_ctx->handle_branch(predicate, predicate_negated, target_pc,
                            instr.reconvergence_pc, context->get_pc());
    // commit_pc() 在 _execute_once() 末尾正确完成 PC 推进
}
```

**理由**: `handle_branch()` 已设置 `next_pc = target`，`commit_pc()` 会正确推进

### 4.3 修复 BarWarpSyncHandler PC 覆盖

**核心问题**: PipelineHandler::ExecPipe 在 `processOperation()` 之后覆盖 `next_pc`

**方案**: 区分处理：

1. **BarWarpSyncHandler 覆盖 PipelineHandler::ExecPipe 的行为**
   - BarWarpSyncHandler 实现自己的 `ExecPipe()`，不继承 PipelineHandler
   - 或者在 PipelineHandler::ExecPipe 中添加检查：如果 `processOperation()` 已设置 `next_pc`，则不再覆盖

2. **修改 PipelineHandler::ExecPipe 逻辑**

```cpp
// 修改 instruction_base.cpp

void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int saved_pc = context->get_pc();
    // 保存 next_pc（processOperation 可能已修改）
    int original_next_pc = context->get_next_pc();

    if (!prepareOperands(context, stmt)) { /* retry */ return; }
    if (!executeOperation(context, stmt)) { /* retry */ return; }
    if (!commitResults(context, stmt)) { /* retry */ return; }

    // 关键：如果 processOperation 已设置 next_pc（不同于 saved_pc+1），保留它
    int final_next_pc = context->get_next_pc();
    if (final_next_pc == saved_pc + 1) {
        // 正常流程：fall through（已在 _execute_once 中设置）
        // 不需要额外操作
    } else {
        // processOperation 已修改 next_pc（如 barrier 完成）
        // 保持 final_next_pc 不变
    }
}
```

但这个方案需要修改 PipelineHandler::ExecPipe 的语义，可能影响其他 handler。

**更好的方案**：修改 BarWarpSyncHandler 的行为，使其在 barrier 完成时不修改当前线程的 `next_pc`，而是让正常的 `commit_pc()` 流程处理。

```cpp
// 修改 barrier.cpp - BarWarpSyncHandler::processOperation

void BarWarpSyncHandler::processOperation(ThreadContext *context, void **operands,
                                          const Qualifier &q,
                                          const std::vector<char> *is_immediate) {
    // ...
    if (wbar.is_complete()) {
        // barrier 完成
        for (int i = 0; i < WarpContext::WARP_SIZE; ++i) {
            if ((wbar.arrived_mask & (1u << i)) && warp_state.threads[i].is_active) {
                if (i == context->lane_id_) {
                    // 当前线程：不修改 next_pc，让 commit_pc() 正确处理
                    // warp_state.threads[i].pc 已由 set_thread_pc 设置
                    // 但不覆盖 next_pc，这样 commit_pc() 会使用正确的值
                } else {
                    // 其他线程：直接设置（这些线程不执行 _execute_once）
                    warp_ctx->set_thread_pc(i, reconvergence_pc);
                }
            }
        }
    }
}
```

但 `set_thread_pc()` 同时设置 `pc` 和 `next_pc`。对于当前线程，我们需要只设置 `pc` 而不改变 `next_pc`。

**最终方案**：添加 `force_set_pc()` 方法，仅设置 `pc`（不设置 `next_pc`），用于blocked 线程恢复

### 4.4 添加 force_set_pc() 方法

**文件**: `include/ptxsim/thread_context.h`

```cpp
// 在现有 set_pc() 后添加
// 【强制写入】强制设置 PC（不修改 next_pc）
// 仅用于 warp 级操作（barrier 完成）对非当前线程的直接写入
void force_set_pc(int new_pc);
```

**文件**: `src/ptxsim/core/thread_context.cpp`

```cpp
void ThreadContext::force_set_pc(int new_pc) {
    if (!warp_context_) return;
    int lane = lane_id_;
    if (lane < 0 || lane >= 32) return;
    warp_context_->get_warp_state().threads[lane].pc = new_pc;
    // next_pc 保持不变
}
```

### 4.5 调度器 Stall 逻辑

**文件**: `src/ptxsim/core/warp_context.cpp`

```cpp
bool WarpContext::is_warp_ready_to_fetch() const {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active) continue;
        // 如果 pc != next_pc，说明 commit_pc() 未执行，当前线程 stall
        if (warp_state.threads[i].pc != warp_state.threads[i].next_pc) {
            return false;
        }
    }
    return true;
}
```

在调度器选择 warp 时调用此检查。如果返回 `false`，调度器选择其他 warp。

---

## 5. 实现计划

| 阶段 | 任务 | 涉及文件 |
|------|------|----------|
| 1 | 删除 BraHandler 冗余 `set_pc()` | control.cpp |
| 2 | 添加 `force_set_pc()` 方法 | thread_context.h, thread_context.cpp |
| 3 | 修改 `set_thread_pc()` 对当前线程使用 `force_set_pc()` | warp_context.cpp |
| 4 | 实现 `is_warp_ready_to_fetch()` | warp_context.cpp |
| 5 | 添加调试日志 | thread_context.cpp |
| 6 | 运行测试验证 | - |

---

## 6. 测试验证

### 6.1 单元测试

- `test_ptx_ld_st` — 基础内存操作
- `test_ptx_bra` — 分支指令
- barrier 相关测试 — 验证 PC 正确性

### 6.2 预期结果

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| barrier 完成后的 PC | `barrier_pc + 1`（错误） | `reconvergence_pc`（正确） |
| BraHandler 后 PC | 正确 | 正确（不变） |
| 调度器 stall | N/A | 自然 stall |

---

## 7. 参考文档

- [NVIDIA PTX ISA 9.2 - Barrier Instructions](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [NVIDIA GPU Pipeline - Fetch/Decode/Execute](https://semiiphub.com/article/analyzing-modern-nvidia-gpu-cores)
- [AMD RDNA Architecture Whitepaper](https://gpuopen.com/download/RDNA_Architecture_public.pdf)
- Sprint 11.2 回归调试报告: `docs/archive/misc/REGRESSION-DEBUGGING-GUIDE.md`
