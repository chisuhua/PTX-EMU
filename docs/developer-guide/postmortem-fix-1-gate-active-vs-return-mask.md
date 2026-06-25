# Handoff Document: Fix 1 — Dispatch Gate 阻塞范围 (PARTIALLY APPLIED / SUPERSEDED)

> **Status (2026-06-25):** Partially applied. The actual gate fix uses
> `return_mask`, not `active_mask`. The original prescription below
> (`if (!(top.active_mask & ...)) continue`) was incomplete: it would
> fail to block fall-through lanes at the reconvergence point, allowing
> them to execute past `reconv_pc` before the taken path converges.
>
> - `unit_simt_stack_stale_entry_blocks_lane0` now **PASS** (after test
>   adjustments to match `return_mask` semantics).
> - `cute_rmsnorm` and `integration_cute_rmsnorm_bar_sync_pattern` still
>   **FAIL** — these are pre-existing failures that require Fix 2 /
>   deeper `S_BAR` scheduler work, not just the gate change.
>
> For the authoritative final semantics, see ADR-0006 §"三个字段的角色分工"
> and the Fix 3 postmortem.

## Goal
修复 `WarpContext::check_and_block_at_reconvergence_point()` 的 dispatch gate bug：当前实现会阻塞**所有**到达 `reconvergence_pc` 的 lane，但**应该**只阻塞属于栈顶 SIMTStackEntry 分歧组（`return_mask`）内的 lane。

> **Correction from original handoff:** 最初认为应该限制为 `active_mask`
>（只阻塞走分支的 lane）。后续 Fix 3 的收敛分析表明，门控必须阻塞
> `return_mask` 内所有到达 `reconv_pc` 的 lane，包括 fall-through lane，
> 否则会在汇聚点产生乱序执行。因此最终代码使用 `return_mask`。

## Background / Root Cause

### 关键背景
- PTX-EMU 是 C++20/CUDA 的 PTX 模拟器，ANTLR4 解析 PTX
- cute_rmsnorm.cu 是单 warp kernel，使用 `bar.sync 0`（S_BAR，不是 S_BAR_WARP_SYNC）做 reduction + broadcast
- cute_rmsnorm E2E 测试 `cute_rmsnorm`（ctest #132）当前 FAIL：output[0] = 0
- 根因分析见 `docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm — broadcast-after-barrier skipped"`

### 根因（已定位，2026-06-16；2026-06-25 修正）
`src/ptxsim/core/warp_context.cpp` 的 `check_and_block_at_reconvergence_point` 有一个 dispatch gate bug：

**Bug 实现（修复前）**：
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (!warp_state.threads[i].is_exited &&
        (int)warp_state.threads[i].pc == reconv_pc &&
        !warp_state.threads[i].is_blocked) {
        warp_state.threads[i].is_blocked = true;  // ← 阻塞所有在 reconv_pc 的 lane
        blocked_lanes.push_back(i);
    }
}
```

**问题**：阻塞循环**没有限制范围**，导致任何 PC==`reconv_pc` 的 lane 都会被阻塞，即使它根本不属于当前栈顶 SIMTStackEntry。

**最初诊断（错误）**：猜测应限制为 `top.active_mask`（只阻塞走分支的 lane）。
**后续修正（2026-06-25）**：Fix 3 的收敛分析表明，门控必须用 `top.return_mask`（整个分歧组），否则 fall-through lane 会在汇聚点越过 `reconv_pc`，造成乱序执行。

**正确逻辑（最终实现）**：只阻塞栈顶 entry 的 `return_mask` 内、且到达 `reconv_pc` 的 lane。
`return_mask` 在 `handle_branch` 中设为 `warp_state.exec_mask`（覆盖 taken + fall-through 整个分歧组），
因此 fall-through lane 在 `reconv_pc` 也会被阻塞，直到栈 pop。

### 复现测试
**已存在**（2026-06-16 编写）：
- 单元测试：`tests/unit/barrier/test_simt_stack_stale_entry_blocks_lane0.cpp`
  - ctest 名称：`unit_simt_stack_stale_entry_blocks_lane0`
  - 标签：`unit;barrier;divergence;simt;regression;BUG-DISPATCH-GATE-LANE0-SKIP`
  - 包含 2 个 TEST_CASE：
    - **U-1**（stale entry 场景）当前 **FAIL**（RED）— 这就是要修的 bug 的复现
    - **U-2**（non-stale entry 场景）当前 **PASS**（GREEN）— 锁定正确行为

**U-1 测试当前输出**（在 `build/bin/tests/unit_simt_stack_stale_entry_blocks_lane0`）：
```
step_warp picked PC=0
Lane 0 is_blocked: 1
Lane 0 is_active: 0
Trace lane 0 PCs: (空)
CHECK(lane0_dispatched) → false  ← 期望 true
CHECK(!ws.threads[0].is_blocked) → false  ← 期望 true
```

## Affected Files

| 文件 | 修改内容 |
|------|---------|
| `src/ptxsim/core/warp_context.cpp` | 修复 `check_and_block_at_reconvergence_point`：阻塞循环用 `top.return_mask`（非 `top.active_mask`）限制范围 |

**只需修改一个函数**，影响范围小。

## Test Setup

### 已有 ctest 目标
```bash
cmake -S /workspace/project/PTX-EMU -B /workspace/project/PTX-EMU/build
cmake --build /workspace/project/PTX-EMU/build --target unit_simt_stack_stale_entry_blocks_lane0

# 单独跑（应 FAIL 当前，PASS 修复后）
/workspace/project/PTX-EMU/build/bin/tests/unit_simt_stack_stale_entry_blocks_lane0

# 跑全量验证无回归
./scripts/sanity.sh
```

### 关键 helper（已存在）
- `ptxsim::testing::step_warp()` — 模拟调度器
- `ptxsim::ExecutionTracer` — 记录 (lane, PC) 对
- `w->get_simt_stack().push(entry)` — 注入 stale entry
- `w->update_active_mask()` — 重建 active_mask

## Constraints

### ⚠️ 严格约束
1. **不要修改 `set_active_mask` 的语义**（per `src/ptxsim/core/AGENTS.md` "BUG-POSTBARRIER-TWOHALVES" 文档）：handler 必须用 `get_active_mask() | arrived_mask` 在 caller 处 OR，而不是改 `set_active_mask` 自身
2. **不要修改 `update_active_mask`**：它是 self-heal 机制，bug 在 caller
3. **保留 `is_blocked=true` 状态作为信号**：可能其他地方依赖这个状态
4. **不要触碰 reduction loop 的 `force_reconvergence_at_barrier` 空函数**：它是意图设计（注释说"不能推进线程PC"），改它可能引入新 bug

### ⚠️ 注意事项
- 修复后**U-1 必须从 RED 转 GREEN**
- 修复后**U-2 必须保持 PASS**（这是正确行为的回归测试）
- **全量 sanity 必须通过**（不能破坏 `unit_broadcast_after_barrier`, `unit_post_barrier_two_halves`, `unit_barrier_divergence_reconvergence_simplegemm` 等已有测试）
- **保留必要的注释**：解释为什么阻塞 `return_mask`（而非 `active_mask`）内的 lane—— fall-through lane 也必须阻塞

## Step-by-Step Approach

### Step 1: 验证 U-1 当前确实 RED
```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh --quick
/workspace/project/PTX-EMU/build/bin/tests/unit_simt_stack_stale_entry_blocks_lane0
```
预期：U-1 FAIL, U-2 PASS

### Step 2: 阅读 `warp_context.cpp:138-170` 当前实现
```bash
sed -n '138,170p' /workspace/project/PTX-EMU/src/ptxsim/core/warp_context.cpp
```

### Step 3: 应用修复（最终正确版本）

**修改 `src/ptxsim/core/warp_context.cpp` 的 `check_and_block_at_reconvergence_point`**：

修改前：
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (!warp_state.threads[i].is_exited &&
        (int)warp_state.threads[i].pc == reconv_pc &&
        !warp_state.threads[i].is_blocked) {
        warp_state.threads[i].is_blocked = true;
        blocked_lanes.push_back(i);
    }
}
```

修改后（**最终正确版本——用 `return_mask` 而非 `active_mask`**）：
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (!(top.return_mask & (1u << i))) continue;          // 仅阻塞分歧组内 lane
    if (!warp_state.threads[i].is_exited &&
        (int)warp_state.threads[i].pc == reconv_pc &&
        !warp_state.threads[i].is_blocked) {
        warp_state.threads[i].is_blocked = true;
        blocked_lanes.push_back(i);
    }
}
```

**注释要求**（写在函数开头）：
```cpp
// BUG-DISPATCH-GATE-LANE0-SKIP (fix): only block lanes that belong to the
// top entry's divergence group (return_mask). Lanes outside return_mask are
// on an unrelated path (or have already converged past reconv_pc) and must
// continue executing. Without this guard, the gate incorrectly blocks any
// lane sitting at reconv_pc.
```

### Step 4: 验证 U-1 转 GREEN
```bash
cmake --build /workspace/project/PTX-EMU/build --target unit_simt_stack_stale_entry_blocks_lane0
/workspace/project/PTX-EMU/build/bin/tests/unit_simt_stack_stale_entry_blocks_lane0
```
预期：U-1 PASS, U-2 PASS

### Step 5: 全量 sanity 验证
```bash
./scripts/sanity.sh
```
预期：所有测试通过，**无回归**

### Step 6: 跑 ctest 全量
```bash
cd /workspace/project/PTX-EMU/build && ctest 2>&1 | tail -10
```
预期：99% pass（cute_rmsnorm 仍 FAIL — 那需要 Fix 2 + S_BAR 路径修复）

## Reference Materials

### 已读过的关键文件
- `src/ptxsim/core/warp_context.cpp:138-170` — `check_and_block_at_reconvergence_point`（**要修改的文件**）
- `src/ptxsim/core/simt_stack.cpp:7-25` — `is_converged`（参考 `active_mask` 收敛判定语义，**与门控不同**）
- `src/ptxsim/core/simt_stack.cpp:82-95` — `check_reconvergence`（参考 entry pop 逻辑）
- `include/ptxsim/simt_stack.h:12-21` — `SIMTStackEntry` 结构
- `tests/unit/barrier/test_simt_stack_stale_entry_blocks_lane0.cpp` — 复现测试
- `src/ptxsim/core/AGENTS.md` — DUAL STATE MECHANISM 文档
- `docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm"` — 完整 bug 上下文

### 关键发现笔记
- cute_rmsnorm.ptx 有 3 次 `bar.sync 0`（line 114, 129, 144），用 S_BAR handler
- 现有 `test_broadcast_after_barrier.cpp` 用 `bar.warp.sync`（S_BAR_WARP_SYNC），是**不同代码路径**
- `force_reconvergence_at_barrier`（warp_context.cpp:440-448）是空函数，注释说"不能推进线程PC"
- `check_reconvergence` 用 `while` 循环连续 pop，但只检查栈顶
- reduction loop 5 次迭代会 push 5 个 @%p10 back-edge entry，循环退出时 5 个都 pop（`return_mask=0xFFFFFFFF` 全在 reconv=loop_exit）

## 任务完成判定（最终状态）

- [x] U-1 单元测试 PASS（`unit_simt_stack_stale_entry_blocks_lane0`）
- [x] U-2 单元测试 PASS
- [x] 全量 `./scripts/sanity.sh` 通过（`cute_rmsnorm` 等仍是 baseline 失败，与本次修复无关）
- [x] 代码 diff 修改 `warp_context.cpp`（门控用 `return_mask`，`check_reconvergence` 用 `return_mask` 恢复 `exec_mask`）
- [x] 注释解释了 `return_mask` 限制的 SIMT 不变量

## Lesson Learned（保留价值所在）

### 关键教训：门控（gate）和收敛判定（is_converged）使用**相反**的字段

| 函数 | 应使用的字段 | 为什么 |
|------|--------------|--------|
| `is_converged()` 收敛判定 | `active_mask` | 只关心"走了分支的 lane 是否都到齐"——fall-through lane 本来就没分支 |
| `check_and_block_at_reconvergence_point()` 门控 | `return_mask` | 必须阻塞所有到达 `reconv_pc` 的 lane（包括 fall-through），否则 fall-through lane 会越过 `reconv_pc` 跑掉造成乱序 |
| `check_reconvergence()` 弹出后恢复 `exec_mask` | `return_mask` | 弹出后整个分歧组都应可执行，不只是 active 子集 |

### 为什么最初会猜错

最初的 bug 报告只展示了"lane 0 被不该阻塞的位置阻塞"——看起来像是"门控多阻塞了非 `active_mask` 的 lane"，所以直觉处方是"限制为 `active_mask`"。但这个处方**未考虑 fall-through lane**——它们同样需要被阻塞在 `reconv_pc`，否则会越界执行。

只有当 Fix 3 的 `is_converged` 收敛分析要求"只用 `active_mask`、不跳 `!is_active`"明确下来后，对比两个函数对 `active_mask` 的使用差异，才能意识到门控**必须用 `return_mask`** 才能同时覆盖 taken + fall-through 两条路径。

### 防错要点

- **不要在未理解"为什么两条路径都需要阻塞"前，凭直觉处方**
- **遇到 SIMT gate/converge 类 bug 时，先列出两个函数的字段使用表**——通常分歧出在"两个函数用了不同字段但看起来应该用同一个"
- **参考 ADR 0006 §"三个字段的角色分工"** 作为权威清单

### 关联修复

- **postmortem-fix-3-is-converged-skip-inactive.md** — `is_converged` 不跳 `!is_active`（合并排查）
- **open-fix-2-sbar-deadlock.md** — `cute_rmsnorm` 仍失败，需要更深层的 `S_BAR` scheduler 修复
