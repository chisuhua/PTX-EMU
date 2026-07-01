# Fix 2 — S_BAR 死锁（修复记录）

> **Status (2026-07-01):** **FIXED** — 详见 [postmortem-sbar-deadlock-fix.md](./postmortem-sbar-deadlock-fix.md)
>
> - `integration_cute_rmsnorm_bar_sync_pattern` — PASS ✓
> - `integration_cta_barrier_memory_visibility` — PASS ✓ (924 assertions)
> - `integration_warp_barrier` — PASS ✓ (no more SEGFAULT)
> - `ctest -L barrier` — **25/25 PASS** ✓
>
> **前置依赖**：[postmortem-fix-1-gate-active-vs-return-mask.md](./postmortem-fix-1-gate-active-vs-return-mask.md) 已完成（门控改用 `return_mask`），但**不足以**恢复 `cute_rmsnorm`。剩余失败位于 CTA-level `bar.sync` / `synchronize_barrier` 路径，需要本文档下文所述的架构性工作。

## Goal（修复目标）
修复 S_BAR handler（`bar.sync 0`）的死锁问题：当只有部分 lane 到达 barrier 时，scheduler 应该把其他 lane 推进到 barrier PC（而不是让先到的 lane 阻塞等待）。

## Background / Root Cause

### 关键背景
- cute_rmsnorm.ptx 的 3 次 `bar.sync 0` 走 **S_BAR handler**（`barrier.cpp:333-384`），不是 S_BAR_WARP_SYNC
- S_BAR 是 **CTA-level barrier**，用 `SMContext::synchronize_barrier(barId, context)` 实现
- cute_rmsnorm 是单 warp kernel（blockDim=32），所有 32 lane 在同一 warp
- cute_rmsnorm 实际执行中：先到的 lane 阻塞在 barrier，其他 lane 永远到不了 barrier PC → **死锁**

### 根因（已定位，2026-06-16）
`src/ptxsim/instructions/barrier.cpp:333-384` 的 `BarHandler::executeBarrier`：

**当前逻辑**（行 371-383）：
```cpp
bool sync_complete = sm_context->synchronize_barrier(barId, context);
if (sync_complete) {
    context->set_next_pc(context->get_pc() + 1);  // 释放：推进 PC
} else {
    context->set_next_pc(context->get_pc());       // 等待：PC 不动
}
```

**Scheduler 行为**（`sm_context.cpp:325-345` 和 `include/ptxsim/testing/scheduler_utils.h:19-32`）：
- `get_lanes_by_pc()` 只返回 `is_active && !is_exited` 的 lane
- `step_warp` 跳过有 is_blocked lane 的整个 PC 组
- 调度器挑**最低 PC**的非阻塞组

**死锁场景**：
1. 所有 32 lane 在 PC=0（per-lane st.shared）执行完，下一 PC=1（bar.sync 0）
2. step_warp 挑 PC=1，调 `execute_warp_instruction(stmt_1, 1)`
3. 对每个 lane 调用 S_BAR handler：
   - lane 0 调用 `synchronize_barrier(0, ctx_0)` → 加入 waiting set → 不完整 → `set_next_pc(1)` → 阻塞
   - lane 1 调用 `synchronize_barrier(0, ctx_1)` → 加入 waiting set → 不完整 → `set_next_pc(1)` → 阻塞
   - ... lane 31 同上
4. 等所有 32 加入 waiting set，barrier fires，release 所有 → `set_next_pc(2)`
5. **理论上**所有 32 应该都推进到 PC=2

**问题在哪？**

如果 lane 0 先到 PC=1，调用 handler，**lane 0 进入 waiting set**。handler 返回 `sync_complete=false` → `set_next_pc(1)` → lane 0 仍在 PC=1，is_blocked=true。

然后 `execute_warp_instruction` 内的 update_active_mask 把 lane 0 标记 is_active=false（因为 is_blocked=true）。**Lane 0 离开 lanes_by_pc[1]**。

step_warp 看到 PC=1 还有 lanes 1-31（它们还没执行 PC=1）。调度器应该挑 PC=1 给 lanes 1-31 执行。

但**实际上**：可能因为某种竞态/状态机错误，lanes 1-31 没有及时推进到 PC=1。

复现测试的 trace：
```
Lane 0 executed PCs: [1,1,1,1]   ← 卡在 PC=1 多次
```

### 已定位的相关代码
1. `src/ptxsim/instructions/barrier.cpp:333-384` — S_BAR handler
2. `src/ptxsim/core/sm_context.cpp:605-700` — `synchronize_barrier` 实现
3. `src/ptxsim/core/sm_context.cpp:325-419` — 调度器（生产）
4. `include/ptxsim/testing/scheduler_utils.h:19-32` — `step_warp`（测试）
5. `src/ptxsim/core/warp_context.cpp:325-338` — `set_active_mask` 单 lane 版
6. `src/ptxsim/core/warp_context.cpp:440-448` — `force_reconvergence_at_barrier`（**候选修改点**）

### 复现测试
**已存在**（2026-06-16 编写）：
- 集成测试：`tests/integration/divergence/test_cute_rmsnorm_bar_sync_pattern.cpp`
  - ctest 名称：`integration_cute_rmsnorm_bar_sync_pattern`
  - 标签：`integration;barrier;divergence;s_bar;regression;BUG-DISPATCH-GATE-LANE0-SKIP`
  - 当前 **FAIL**（RED）— S_BAR 死锁复现
  - lane 0 卡在 PC=1，never reach PC=2/8/9/10

## Affected Files

| 文件 | 可能修改点 |
|------|----------|
| `src/ptxsim/core/warp_context.cpp:440-448` | `force_reconvergence_at_barrier` 增强为真正推进 lane 到 barrier PC |
| `src/ptxsim/instructions/barrier.cpp:333-384` | S_BAR handler：可能需要等所有 lane 到达后调 force_reconvergence |
| `src/ptxsim/core/sm_context.cpp:605-700` | `synchronize_barrier`：可能需要支持 partial-warp release |

**这是个 deeper fix**，需要架构调整。

## Test Setup

### 已有 ctest 目标
```bash
cmake -S /workspace/project/PTX-EMU -B /workspace/project/PTX-EMU/build
cmake --build /workspace/project/PTX-EMU/build --target integration_cute_rmsnorm_bar_sync_pattern

# 跑当前应 FAIL
/workspace/project/PTX-EMU/build/bin/tests/integration_cute_rmsnorm_bar_sync_pattern
```

### 关键 trace 输出
```
Lane 0 executed PCs: [1,1,1,1]      ← 死锁在 PC=1
Lane 0 is_blocked: 1
Lane 0 is_active: 0
```

### 复现模式（已硬编码在测试中）
```
PC=0:  st.shared.b32 [sdata+r_tid], r_val   ; per-lane write
PC=1:  bar.sync 0                            ; ★ S_BAR 死锁点
PC=2:  setp.ne.s32 p3, r_tid, 0
PC=3:  @p3 bra L_TID0_DVRG_W
PC=4:  bra L_BCONV
PC=5-7: ...                                   ; intermediate nops
PC=8:  st.shared.b32 [sdata+r_tid], r_rsqrt  ; lane 0 only
PC=9:  bar.sync 0                            ; broadcast barrier
PC=10: ld.shared.b32 r2, [sdata+r_tid]       ; BROADCAST READ
PC=11: ret
```

## Constraints

### ⚠️ 严格约束
1. **不要破坏现有 `unit_barrier_active_mask_preserved` 等测试**（`tests/unit/sync/test_barrier_active_mask_preserved.cpp`）
2. **CTA-level barrier 必须真正等待所有 thread**（不能简单做 per-warp barrier）
3. **不要破坏 `force_reconvergence_at_barrier` 的设计意图**（除非有充分理由）
4. **S_BAR 与 S_BAR_WARP_SYNC 行为差异必须保留**（用户可能同时用两个）

### ⚠️ 修复策略选择

**选项 A：S_BAR 改用 Wbar 机制（per-warp 屏障）**
- 优点：避免 CTA-level 等待，scheduler 推进所有 lane 到 barrier PC
- 缺点：cute_rmsnorm 是单 warp，单 CTA 行为等价。Multi-warp kernel 会破坏 CTA barrier 语义

**选项 B：调度器在 bar.sync 前 force 推进所有 lane 到 barrier PC**
- 优点：保留 CTA-level 语义，scheduler 协调推进
- 缺点：实现复杂，需要 scheduler 感知 barrier 指令

**选项 C：`synchronize_barrier` 改进，支持 partial completion**
- 优点：最小改动 CTA-level barrier 逻辑
- 缺点：可能不解决根本的"先到 lane 阻塞"问题

**建议**：先尝试**选项 B**（最小侵入 + 保留语义）。如果不行，再考虑选项 A。

### ⚠️ 注意事项
- 修复后 I-3 集成测试必须从 RED 转 GREEN
- 修复后**全量 sanity 必须通过**（不能破坏其他 barrier 测试）
- **保留必要的注释**：解释为什么采用此策略（特别是如果用选项 A，需要明确单 warp 假设）

## Step-by-Step Approach

### Step 1: 验证 I-3 当前确实 RED
```bash
/workspace/project/PTX-EMU/build/bin/tests/integration_cute_rmsnorm_bar_sync_pattern 2>&1 | tail -20
```
预期：trace 显示 `Lane 0 executed PCs: [1,1,1,1]`，断言 FAIL

### Step 2: 阅读 S_BAR handler 完整实现
```bash
sed -n '333,384p' /workspace/project/PTX-EMU/src/ptxsim/instructions/barrier.cpp
```

### Step 3: 探索修复方向
**方案 B（推荐）**：在 `execute_warp_instruction` 入口对 `S_BAR` 类型做特殊处理：
- 在 `warp_context.cpp:execute_warp_instruction` 行 214-309 的早期，检测 `stmt.type == S_BAR`
- 如果是 S_BAR，先调 `warp_ctx->force_reconvergence_at_barrier(target_pc)` （**但** 现有 `force_reconvergence_at_barrier` 是空函数，需要增强它）
- 然后正常执行

**需要先研究**：
1. 为什么 `force_reconvergence_at_barrier` 留空？注释说"不能推进线程PC"
2. 是否有其他地方依赖"先到 lane 阻塞等待"的行为
3. 修复 cute_rmsnorm（单 warp）的同时是否会破坏 multi-warp kernel

### Step 4: 实施修复（推荐方案 B）

修改 `src/ptxsim/core/warp_context.cpp` 的 `force_reconvergence_at_barrier`（行 440-448）：

修改前（空函数）：
```cpp
void WarpContext::force_reconvergence_at_barrier(int barrier_pc) {
    // 不主动推进线程PC —— 让调度器自然选择非阻塞的PC执行
    // ...
}
```

修改后：
```cpp
void WarpContext::force_reconvergence_at_barrier(int barrier_pc) {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (warp_state.threads[i].is_active &&
            !warp_state.threads[i].is_exited &&
            (int)warp_state.threads[i].pc != barrier_pc) {
            warp_state.threads[i].pc = barrier_pc;
            warp_state.threads[i].next_pc = barrier_pc;
        }
    }
}
```

**注释要求**（写在函数开头）：
```cpp
// Advance all active lanes to barrier_pc so they can all hit the barrier
// on the same step_warp iteration. Without this, single-warp CTA barriers
// (e.g., bar.sync 0 in cute_rmsnorm) deadlock: the first lane to arrive
// blocks, is removed from lanes_by_pc, and the scheduler cannot advance
// the remaining lanes to the barrier PC.
```

然后确认 `barrier.cpp:361-363` 的 `force_reconvergence_at_barrier` 调用（已有）让它真正生效。

### Step 5: 验证 I-3 转 GREEN
```bash
cmake --build /workspace/project/PTX-EMU/build --target integration_cute_rmsnorm_bar_sync_pattern
/workspace/project/PTX-EMU/build/bin/tests/integration_cute_rmsnorm_bar_sync_pattern
```
预期：trace 显示 lane 0 执行 PC=0, 1, 2, 3, 4, 8, 9, 10, 11，断言 PASS

### Step 6: 全量 sanity 验证
```bash
./scripts/sanity.sh
cd /workspace/project/PTX-EMU/build && ctest 2>&1 | tail -5
```
预期：129/129 pass

### Step 7: 跑 cute_rmsnorm E2E
```bash
cd /workspace/project/PTX-EMU/build && ctest -R cute_rmsnorm -V
```
预期：output[0] ≈ input[0] / rms（**E2E 第一次 PASS**）

## Reference Materials

### 已读过的关键文件
- `src/ptxsim/instructions/barrier.cpp:333-384` — S_BAR handler
- `src/ptxsim/instructions/barrier.cpp:340-380` — `synchronize_barrier` 调用
- `src/ptxsim/core/sm_context.cpp:605-700` — `synchronize_barrier` 实现（CTA-level）
- `src/ptxsim/core/sm_context.cpp:660-680` — barrier completion & release
- `src/ptxsim/core/warp_context.cpp:440-448` — `force_reconvergence_at_barrier`（**要修改的文件**）
- `tests/integration/divergence/test_cute_rmsnorm_bar_sync_pattern.cpp` — 复现测试
- `bench/cute/cute_rmsnorm.ptx:109-145` — cute_rmsnorm PTX 实际模式
- `bench/cute/cute_rmsnorm.cu:113-121` — cute_rmsnorm 启动参数（blockSize=32）
- `docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm"` — 完整 bug 上下文
- [postmortem-fix-1-gate-active-vs-return-mask.md](./postmortem-fix-1-gate-active-vs-return-mask.md) — **已完成**（门控改用 `return_mask`，见 postmortem 文档）

### 关键发现笔记
- S_BAR 与 S_BAR_WARP_SYNC 是两个完全不同的代码路径
- S_BAR 用 `synchronize_barrier`（CTA-level），S_BAR_WARP_SYNC 用 Wbar（per-warp）
- cute_rmsnorm 用 S_BAR（bar.sync 0），不是 S_BAR_WARP_SYNC
- 单 warp kernel + S_BAR 容易死锁（这是 PTX-EMU 模拟器的特殊问题）
- `force_reconvergence_at_barrier` 留空可能是历史决定（"不能推进线程PC" 注释）
- 但这个设计假设在单 warp + S_BAR 场景下不成立

### 顺序建议
**状态**：Fix 1 已完成（门控改用 `return_mask`），但单独不足以恢复 `cute_rmsnorm`。
Fix 2（`S_BAR` 死锁）**仍然 OPEN**，需要实施本文档 §"Step 4: 实施修复" 的方案。

## 任务完成判定

修复成功的标志：
- [x] Fix 1 已完成（门控改用 `return_mask`，见 [postmortem-fix-1](./postmortem-fix-1-gate-active-vs-return-mask.md)）
- [ ] I-3 集成测试从 RED 转 GREEN
- [ ] cute_rmsnorm E2E 测试 PASS（首次）
- [ ] 全量 `./scripts/sanity.sh` 通过
- [ ] `ctest` 显示 100% pass
- [ ] 代码 diff 仅修改 `warp_context.cpp` 一处（或必要时小改 `barrier.cpp`）
- [ ] 注释解释了为什么要推进所有 lane 到 barrier PC
