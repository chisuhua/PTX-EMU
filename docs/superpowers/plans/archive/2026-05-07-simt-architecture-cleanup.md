# SIMT Architecture Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Clean up deprecated pc_stacks infrastructure and clarify active_mask/exec_mask dual-mask architecture

**Architecture:** Two-phase approach:
- Phase 1: Remove pc_stacks dead code (deprecated [[deprecated]] but still called from barrier.cpp:177)
- Phase 2: Clarify active_mask/exec_mask relationship with proper synchronization at divergence points (NOT unification — they serve different purposes: active_mask[] for scheduler is_lane_active(), exec_mask for PTX activemask instruction)

**Tech Stack:** C++20, Catch2, CMake

---

## Phase 1: pc_stacks Cleanup (P1 — Low Risk)

### Task 1: Remove update_pc_stack() call from barrier.cpp

**Files:**
- Modify: `src/ptxsim/instructions/barrier.cpp:177`
- Test: `cd build && ctest -R "barrier" -V`

**Context:** barrier.cpp lines 172-177 already set PC via new interface. Line 177's call to update_pc_stack() is redundant.

```
Lines 172-176 (new-style PC setting — already correct):
    if (i == context->lane_id_) {
        context->force_set_pc(reconvergence_pc);
        context->set_next_pc(reconvergence_pc);
    } else {
        warp_ctx->set_thread_pc(i, reconvergence_pc);
    }
Line 177 (deprecated call — REMOVE):
    warp_ctx->update_pc_stack(i, reconvergence_pc);
```

- [ ] **Step 1: Remove the deprecated update_pc_stack() call**

Edit `src/ptxsim/instructions/barrier.cpp` — remove line 177:
```cpp
// BEFORE:
        warp_ctx->set_thread_pc(i, reconvergence_pc); // ✅ 新接口
        warp_ctx->update_pc_stack(i, reconvergence_pc);  // ❌ 废弃调用，冗余
        warp_state.threads[i].is_blocked = false;

// AFTER:
        warp_ctx->set_thread_pc(i, reconvergence_pc); // ✅ 新接口
        warp_state.threads[i].is_blocked = false;
```

- [ ] **Step 2: Build to verify no compilation errors**

Run: `cmake --build build --target ptxsim 2>&1 | tail -20`
Expected: Clean build (no warnings from deprecated call)

- [ ] **Step 3: Run barrier tests to confirm no regression**

Run: `cd build && ctest -R "barrier" -V 2>&1 | tail -40`
Expected: All barrier tests PASS

- [ ] **Step 4: Commit**

```bash
git add src/ptxsim/instructions/barrier.cpp
git commit -m "fix: remove deprecated update_pc_stack call from barrier.cpp

The new-style PC setting (force_set_pc/set_thread_pc) already handles
PC update correctly. The update_pc_stack() call was redundant and
produced [[deprecated]] warnings."
```

---

### Task 2: Remove deprecated pc_stacks infrastructure from WarpContext

**Files:**
- Modify: `include/ptxsim/warp_context.h` (lines 90-96 update_pc_stack, lines 159-161 handle_branch_divergence, line 232 pc_stacks member)
- Modify: `src/ptxsim/core/warp_context.cpp` (line 122 constructor init, lines 287-288 reset(), lines 294-307 handle_branch_divergence implementation)
- Test: `cd build && ctest -R "warp|active_mask|exec_mask" -V`

**Context:** After Task 1, no live code calls update_pc_stack(). The entire pc_stacks infrastructure can now be removed safely.

#### Part A: Remove from warp_context.h

- [ ] **Step 1: Remove update_pc_stack() inline method (lines 90-96)**

```cpp
// REMOVE these lines from include/ptxsim/warp_context.h:
// [[deprecated("Use warp_state.threads[lane_id].pc = new_pc instead")]]
// void update_pc_stack(int lane_id, uint32_t new_pc) {
//     if (lane_id >= 0 && lane_id < WARP_SIZE && !pc_stacks[lane_id].empty()) {
//         pc_stacks[lane_id].back() = new_pc;
//     }
// }
```

- [ ] **Step 2: Remove handle_branch_divergence() declaration (lines 159-161)**

```cpp
// REMOVE from include/ptxsim/warp_context.h:
// [[deprecated("PC stacks replaced by SIMT stack + warp_state.threads[i].pc")]]
// void handle_branch_divergence(int lane_id, int new_pc);
```

- [ ] **Step 3: Remove pc_stacks member (line 232)**

```cpp
// REMOVE from include/ptxsim/warp_context.h:
// std::vector<int> pc_stacks[WARP_SIZE]; // 每个线程的 PC 栈，用于分支重新合并
```

#### Part B: Remove from warp_context.cpp

- [ ] **Step 4: Remove pc_stacks constructor initialization (line 122)**

```cpp
// REMOVE from WarpContext constructor in src/ptxsim/core/warp_context.cpp:
// pc_stacks[i] = std::vector<int>();
```

- [ ] **Step 5: Remove pc_stacks cleanup from reset() (lines 287-288)**

```cpp
// REMOVE from reset() in src/ptxsim/core/warp_context.cpp:
// pc_stacks[i].clear();
// pc_stacks[i].push_back(0);
```

- [ ] **Step 6: Remove handle_branch_divergence() implementation (lines 294-307)**

```cpp
// REMOVE entire function from src/ptxsim/core/warp_context.cpp:
// void WarpContext::handle_branch_divergence(int lane_id, int new_pc) {
//     if (lane_id >= 0 && lane_id < WARP_SIZE && !pc_stacks[lane_id].empty()) {
//         pc_stacks[lane_id].push_back(new_pc);
//     }
// }
```

- [ ] **Step 7: Build to verify**

Run: `cmake --build build --target ptxsim 2>&1 | tail -20`
Expected: Clean build, no undefined reference errors

- [ ] **Step 8: Run warp/active_mask/exec_mask tests**

Run: `cd build && ctest -R "warp|active_mask|exec_mask" -V 2>&1 | tail -60`
Expected: All tests PASS

- [ ] **Step 9: Commit**

```bash
git add include/ptxsim/warp_context.h src/ptxsim/core/warp_context.cpp
git commit -m "refactor: remove deprecated pc_stacks infrastructure

pc_stacks (per-thread PC stack array) was replaced by SIMT stack
+ warp_state.threads[i].pc in SIMT v2.0. All call sites removed
in prior commit. Now removing the dead infrastructure:
- update_pc_stack() inline method
- handle_branch_divergence() declaration and implementation
- pc_stacks[WARP_SIZE] member variable
- constructor initialization and reset() cleanup"
```

---

### Task 3: Full regression verification

**Files:** (no modifications — verification only)

- [ ] **Step 1: Run full test suite**

Run: `cd build && ctest -j4 2>&1 | tail -30`
Expected: All tests PASS

- [ ] **Step 2: Run PTX syntax tests**

Run: `./tests/ptx/test_all_ptx.sh 2>&1 | tail -20`
Expected: All PTX syntax tests PASS

- [ ] **Step 3: Run sanity check**

Run: `./scripts/sanity.sh 2>&1 | tail -30`
Expected: Full sanity PASS

---

## Phase 2: active_mask / exec_mask Architecture Clarification (P2 — Medium Risk)

> **Note:** These two masks serve DIFFERENT purposes and should NOT be unified:
> - `exec_mask` (WarpState): PTX activemask instruction result, SIMT stack managed
> - `active_mask[]` (WarpContext): Scheduler's is_lane_active() per-lane boolean array
>
> The correct fix is ensuring proper synchronization at divergence points, NOT removing one mask.

### Task 4: Add explicit synchronization comment and verify barrier hybrid fix

**Files:**
- Modify: `src/ptxsim/core/warp_context.cpp` (handle_branch function — add synchronization note)
- Modify: `include/ptxsim/warp_context.h` (document active_mask purpose)
- Test: No new tests needed — existing tests already cover this

**Context:** After Phase 1 cleanup, we should add documentation clarifying the dual-mask design and ensure the barrier hybrid fix (barrier.cpp:162+185) is the canonical pattern.

- [ ] **Step 1: Add architecture comment to warp_context.h explaining dual-mask design**

Find the `active_mask[]` declaration in `include/ptxsim/warp_context.h` and add a detailed comment:

```cpp
// 活跃掩码数组 — 与 warp_state.exec_mask 是两个独立机制，共同维护：
// - exec_mask: WarpState 中的 uint32_t，用于 PTX activemask 指令返回值和 SIMT stack 管理
// - active_mask[]: WarpContext 中的 bool[WARP_SIZE]，用于调度器的 is_lane_active() 判断
// 两者在以下时序点同步：
//   1. barrier 释放时：barrier.cpp 同时调用 set_exec_mask() 和 set_active_mask()
//   2. 线程退出/阻塞时：update_active_mask() 根据 is_exited/is_blocked 重建 active_mask[]
// 注意：handle_branch() 发散时只更新 exec_mask，active_mask[] 在下一 execute_warp_instruction() 周期通过 update_active_mask() 重建
std::array<bool, WARP_SIZE> active_mask;
```

- [ ] **Step 2: Add synchronization note in handle_branch() after exec_mask update**

In `src/ptxsim/core/warp_context.cpp`, after line 70 (`warp_state.exec_mask = taken_mask;`), add:

```cpp
// Note: active_mask[] is NOT updated here. It will be recalculated
// by update_active_mask() at the end of execute_warp_instruction().
// The barrier release path (barrier.cpp:162+185) handles explicit
// synchronization for that use case.
```

- [ ] **Step 3: Build and run tests**

Run: `cmake --build build --target ptxsim 2>&1 | tail -10`
Run: `cd build && ctest -R "active_mask|exec_mask|barrier" -V 2>&1 | tail -40`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add include/ptxsim/warp_context.h src/ptxsim/core/warp_context.cpp
git commit -m "docs: clarify active_mask/exec_mask dual-mask architecture

These two masks serve different purposes and must NOT be unified:
- exec_mask: PTX activemask instruction, managed by SIMT stack
- active_mask[]: scheduler is_lane_active() decision

Proper synchronization happens at barrier release (set_exec_mask +
set_active_mask) and at execute_warp_instruction boundaries
(update_active_mask). Documented the design rationale."
```

---

## Verification Summary

| Phase | Task | Risk | Tests |
|-------|------|------|-------|
| 1 | T1: Remove barrier.cpp:177 call | Low | barrier tests |
| 1 | T2: Remove pc_stacks infrastructure | Low | warp/active_mask/exec_mask tests |
| 1 | T3: Full regression | — | Full ctest + PTX syntax |
| 2 | T4: Document dual-mask design | None (docs only) | Existing tests |

## File Changes Summary

| File | Lines | Change |
|------|-------|--------|
| `src/ptxsim/instructions/barrier.cpp` | 177 | Delete 1 line |
| `include/ptxsim/warp_context.h` | 90-96, 159-161, 232 | Delete 9 lines, add 12 lines docs |
| `src/ptxsim/core/warp_context.cpp` | 122, 287-288, 294-307 | Delete ~18 lines, add 5 lines comment |
