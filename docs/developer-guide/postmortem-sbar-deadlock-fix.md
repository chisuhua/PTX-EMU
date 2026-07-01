# Postmortem: S_BAR (bar.sync 0) Deadlock 修复

> **日期**: 2026-07-01 | **范围**: 3 个 barrier 测试从 FAIL → PASS | **相关**: Fix 2 (open-fix-2-sbar-deadlock)

## 修复成果

| 测试 | 修复前 | 修复后 |
|------|--------|--------|
| `integration_warp_barrier` | **SEGFAULT** | PASS |
| `integration_cta_barrier_memory_visibility` | FAILED (ret0 != 15) | PASS (924 断言) |
| `integration_cute_rmsnorm_bar_sync_pattern` | DISABLED (deadlock) | PASS (131 断言) |
| **ctest -L barrier** | 23→24→25 全通过 | ✔ |

## 问题清单 (7 bugs)

### B1: `step_warp` 缺少 PC 边界检查 → SEGFAULT
- **文件**: `include/ptxsim/testing/scheduler_utils.h`
- **症状**: `v[pick]` 访问越界，`instructionText` 是垃圾指针
- **根因**: BUG-POSTBARRIER-TWOHALVES 修复后所有 32 lanes 穿过 barrier，reconvergence_pc 超出 statements 范围
- **修复**: 添加 `pick >= v.size()` 检查，返回 -1

### B2: `run_warp_until_ret_or_stuck` 过早跳过 barrier → FAIL
- **文件**: `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp`
- **症状**: warp 直接返回 `post_barrier_pc=11`，跳过了 `bar.sync` handler
- **根因**: `all_at_barrier && any_unblocked` 在线程刚到达 barrier（未执行 handler）时就触发
- **修复**: 添加 `is_cta_barrier_complete(0)` 门控

### B3: `release_cta_barrier` 缺少 `is_active` 恢复 → 死锁
- **文件**: `src/ptxsim/barrier/barrier_module.cpp`
- **症状**: barrier 释放后 `get_lanes_by_pc()` 返回空 → warp 永久卡住
- **根因**: `update_active_mask()` 在 BAR_SYNC 后将 `is_active` 设为 false，但 release 时只设了 `is_blocked=false, status=Active`
- **修复**: 添加 `ts.is_active = true` + 对每个受影响 warp 调用 `update_active_mask()`

### B4: `CTABarrier::reset()` 破坏可重用性 → 死锁
- **文件**: `src/ptxsim/barrier/cta_barrier.cpp`
- **症状**: "arrive called on uninitialized barrier" → 第二次 `bar.sync 0` 永不完成
- **根因**: `reset()` 将 `is_initialized_` 设为 false
- **修复**: `reset()` 只清 `arrived_threads_`，保留初始化状态

### B5: 测试中 `reconvergence_pc` 指向错误位置 → 死锁
- **文件**: `tests/integration/divergence/test_cute_rmsnorm_bar_sync_pattern.cpp`
- **症状**: `check_and_block_at_reconvergence_point` 把 non-taken lanes 堵在 PC=4 等待 taken lane（永不到达 PC=4）
- **根因**: `reconv=4`（fallthrough 路径），但 taken path 跳到 PC=8
- **修复**: `reconv=PC_BAR3=9`（两个路径的真正汇聚点）

### B6: ExecutionTracer 使用 post-exec PC → 断言不匹配
- **文件**: `tests/integration/divergence/test_cute_rmsnorm_bar_sync_pattern.cpp`
- **症状**: `e.pc == PC_BROADCAST(10)` 永远不匹配
- **根因**: tracer 记录 `commit_pc()` 后的值（`ld.shared @ PC=10` → 记录 `PC=11`）
- **修复**: 广播检查改为 `e.pc == PC_RET(11)`

## 关键经验

### 诊断模式（可用于未来调试）

```
SEGFAULT? → GDB backtrace + 检查 fprintf %s 参数（语句指针是否垃圾）
  └─ PC 异常高 → 检查 step_warp / reconvergence_pc / statements 大小

warp 卡住? → 添加 [THREAD_DEBUG] / [BARRIER_DEBUG] 打印
  └─ get_lanes_by_pc() 空 → 检查 is_active / is_blocked / update_active_mask
  └─ barrier 永不完成 → 检查 CTABarrier::arrive / arrived_threads / expected_threads

分歧 + barrier 卡住? → 打印 per-lane PC → 检查 check_and_block_at_reconvergence_point
  └─ reconv_pc != 实际汇聚点 → 修正 reconvergence_pc
```

### S_BAR vs S_BAR_WARP_SYNC 状态管理差异

| 维度 | S_BAR_WARP_SYNC | S_BAR (bar.sync 0) |
|------|-----------------|---------------------|
| 状态管理 | `processOperation` 直接操作 warp_state | `executeBarrier` → `BarrierModule` |
| 释放后 `is_active` | `set_active_mask` 设置了 | **需显式设置** `ts.is_active=true` |
| 释放后 `update_active_mask` | handler 内完成 | **需显式调用** |
| barrier 可重用性 | wbar 自动重新初始化 | reset() 可能清除初始化状态 |

### 调试命令速查

```bash
# 每线程指令追踪
./build/bin/test_xxx 2>&1 | grep "\[THREAD_DEBUG\]" | grep "pc=N" | head -40

# barrier 抵达追踪
./build/bin/test_xxx 2>&1 | grep "arrive.*complete=" | head -20

# 检查 active_mask（自愈 Bug 检测）
gdb --batch -ex "b WarpContext::update_active_mask" -ex "run" --args ./build/bin/test_xxx
```

## 相关文档

- [open-fix-2-sbar-deadlock.md](./open-fix-2-sbar-deadlock.md) — 原始 issue
- [KNOWN_ISSUES.md](./KNOWN_ISSUES.md) §"Pre-P0 Baseline Red" — 原始症状
- [postmortem-fix-1-gate-active-vs-return-mask.md](./postmortem-fix-1-gate-active-vs-return-mask.md) — 相关 Fix 1
- [postmortem-fix-3-is-converged-skip-inactive.md](./postmortem-fix-3-is-converged-skip-inactive.md) — 相关 Fix 3
