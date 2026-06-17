# Handoff Document: Fix 1 — Dispatch Gate 不应阻塞非 `active_mask` 的 lane

## Goal
修复 `WarpContext::check_and_block_at_reconvergence_point()` 的 dispatch gate bug：当前实现会阻塞**所有**到达 `reconvergence_pc` 的 lane，但**应该**只阻塞栈顶 SIMTStackEntry 的 `active_mask` 内的 lane。

## Background / Root Cause

### 关键背景
- PTX-EMU 是 C++20/CUDA 的 PTX 模拟器，ANTLR4 解析 PTX
- cute_rmsnorm.cu 是单 warp kernel，使用 `bar.sync 0`（S_BAR，不是 S_BAR_WARP_SYNC）做 reduction + broadcast
- cute_rmsnorm E2E 测试 `cute_rmsnorm`（ctest #132）当前 FAIL：output[0] = 0
- 根因分析见 `docs/developer-guide/KNOWN_ISSUES.md §"cute_rmsnorm — broadcast-after-barrier skipped"`

### 根因（已定位，2026-06-16）
`src/ptxsim/core/warp_context.cpp:138-170` 的 `check_and_block_at_reconvergence_point` 有一个 dispatch gate bug：

**当前实现（行 161-168）**：
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

**问题**：行 161 的循环**没有检查** `(top.active_mask & (1u << i))`。这导致任何在 PC=`reconv_pc` 的 lane 都会被阻塞，即使它根本不在该 SIMTStackEntry 的 `active_mask` 内。

**正确逻辑应该是**：只阻塞栈顶 entry 的 `active_mask` 内、且到达 `reconv_pc` 的 lane。其他 lane（不在 active_mask）应该继续执行，因为它们属于另一条分歧路径或已经"通过"了 reconvergence。

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
| `src/ptxsim/core/warp_context.cpp` | 修复 `check_and_block_at_reconvergence_point`（行 138-170）：在阻塞循环中加 `top.active_mask` 检查 |

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
- **保留必要的注释**：解释为什么只阻塞 active_mask 内的 lane（这是 SIMT 语义的不变量）

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

### Step 3: 应用修复
**修改 `src/ptxsim/core/warp_context.cpp` 行 161-168**：

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

修改后（**关键**）：
```cpp
for (int i = 0; i < WARP_SIZE; i++) {
    if (!(top.active_mask & (1u << i))) continue;
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
// BUG-DISPATCH-GATE-LANE0-SKIP (fix): only block lanes within the top
// entry's active_mask. Lanes outside active_mask are on a different
// divergence path (or have already converged past reconv_pc) and must
// continue executing.
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
- `src/ptxsim/core/simt_stack.cpp:7-20` — `is_converged`（参考 active_mask 语义）
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
- reduction loop 5 次迭代会 push 5 个 @%p10 back-edge entry，循环退出时 5 个都 pop（active_mask=0xFFFFFFFF 全在 reconv=loop_exit）

## 任务完成判定

修复成功的标志：
- [ ] U-1 单元测试从 RED 转 GREEN
- [ ] U-2 单元测试保持 PASS
- [ ] 全量 `./scripts/sanity.sh` 通过
- [ ] `ctest` 显示 99% pass（仅 cute_rmsnorm 仍 FAIL，等 Fix 2）
- [ ] 代码 diff 仅修改 `warp_context.cpp` 一处
- [ ] 注释解释了 active_mask 限制的 SIMT 不变量
