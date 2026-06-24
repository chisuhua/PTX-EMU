# ISSUE-004: active_mask 一致性修复

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 消除 `active_mask[]`（WarpContext）和 `warp_state.threads[i].is_active`（WarpState）之间的三路不一致，确保屏障释放/线程退出/状态变更后两数据源同步。

**Architecture:** 修改 3 个函数 + 验证现有 5 个测试用例。核心改动：(1) `update_active_mask()` 统一从 `warp_state` 读写，(2) `is_lane_active()` 委托 `is_lane_schedulable()`，(3) `sync_to_warp_state(RUN)` 添加 `is_active=true`。已有 `test_active_mask_consistency.cpp`（J1-J5）验证。

**Tech Stack:** C++20, PTX-EMU ptxsim 核心模块

---

### Task 1: 修改 update_active_mask() — 统一数据源

**Files:**
- Modify: `src/ptxsim/core/warp_context.cpp:210-222`
- Read reference: `include/ptxsim/thread_state.h:58-60`（了解 is_schedulable()）

- [ ] **Step 1: 读取当前实现**

```bash
cd /workspace/project/PTX-EMU
sed -n '205,225p' src/ptxsim/core/warp_context.cpp
```

Expected: 看到当前 `update_active_mask()` 从 `threads[i]->is_exited()`（ThreadContext）和 `warp_state.threads[i].is_blocked`（WarpState）混合读取。

- [ ] **Step 2: 修改 update_active_mask() 统一从 warp_state 读写**

修改 `src/ptxsim/core/warp_context.cpp:210-222`：

```cpp
void WarpContext::update_active_mask() {
    active_count = 0;
    for (int i = 0; i < WARP_SIZE; i++) {
        if (i < threads.size() && threads[i] != nullptr) {
            // 统一从 warp_state 读取权威状态，而非混合 ThreadContext + WarpState
            bool active = warp_state.threads[i].is_active &&
                          !warp_state.threads[i].is_exited &&
                          !warp_state.threads[i].is_blocked &&
                          (warp_state.threads[i].status == ThreadStatus::Active);
            active_mask[i] = active;
            // 双向同步：同时写回 warp_state，消除双源不一致
            warp_state.threads[i].is_active = active;
            if (active) active_count++;
        }
    }
}
```

- [ ] **Step 3: 编译验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build --target ptxsim 2>&1 | tail -20
```

Expected: 编译成功，无错误。

---

### Task 2: 让 is_lane_active() 委托 is_lane_schedulable() — 消除 ISSUE-005

**Files:**
- Modify: `include/ptxsim/warp_context.h:124-126`
- Read reference: `include/ptxsim/warp_context.h:97-102`（is_lane_schedulable 定义）

- [ ] **Step 1: 读取当前 is_lane_active() 实现**

```bash
cd /workspace/project/PTX-EMU
sed -n '120,130p' include/ptxsim/warp_context.h
```

Expected: 看到 `is_lane_active()` 仅从 `active_mask[lane_id]` 读取——非权威源。

- [ ] **Step 2: 修改 is_lane_active() 委托给 is_lane_schedulable()**

```cpp
// 从：仅读取 active_mask[]
// bool is_lane_active(int lane_id) const { return lane_id >= 0 && lane_id < WARP_SIZE && active_mask[lane_id]; }

// 改为：委托给 warp_state 权威源（同时修复 ISSUE-005）
bool is_lane_active(int lane_id) const {
    return is_lane_schedulable(lane_id);
}
```

注意：`is_lane_schedulable()`（第 97-102 行）定义为：
```cpp
bool is_lane_schedulable(int lane_id) const {
    return warp_state.threads[lane_id].is_schedulable();
}
```
而 `is_schedulable()`（`thread_state.h:58-60`）为：
```cpp
bool is_schedulable() const {
    return is_active && !is_exited && !is_blocked && (status == ThreadStatus::Active);
}
```

- [ ] **Step 3: 编译验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build --target ptxsim 2>&1 | tail -20
```

Expected: 编译成功，无错误。

---

### Task 3: 在 sync_to_warp_state(RUN) 中添加 is_active=true

**Files:**
- Modify: `src/ptxsim/core/thread_context.cpp:735-742`
- Read reference: `src/ptxsim/core/thread_context.cpp:720-750`（完整 sync_to_warp_state）

- [ ] **Step 1: 读取 sync_to_warp_state RUN 分支**

```bash
cd /workspace/project/PTX-EMU
sed -n '730,750p' src/ptxsim/core/thread_context.cpp
```

Expected: 看到 RUN case 设置 `is_blocked=false` 和 `status=Active`，但未设置 `is_active=true`。

- [ ] **Step 2: 添加 is_active = true**

```cpp
case RUN:
    thread_state.status = ThreadStatus::Active;
    thread_state.is_blocked = false;
    thread_state.is_active = true;  // <-- 新增：屏障释放后标记线程为活跃
    break;
```

- [ ] **Step 3: 编译验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build --target ptxsim 2>&1 | tail -20
```

Expected: 编译成功。

---

### Task 4: 运行现有 active_mask 一致性测试验证无回归

**Files:**
- Test: `tests/test_active_mask_consistency.cpp`（J1-J5 测试）
- Read: `scripts/sanity.sh:142,158`（ISSUE-004 测试注册）

- [ ] **Step 1: 构建并运行 active_mask 测试**

```bash
cd /workspace/project/PTX-EMU && . env.sh && cmake --build build 2>&1 | tail -5
cd build && ctest -R test_active_mask_consistency -V 2>&1 | tail -30
```

Expected: 5 个测试用例全部通过（PASSED）。

- [ ] **Step 2: 运行 post_barrier_divergence 测试验证屏障相关行为**

```bash
cd /workspace/project/PTX-EMU/build && ctest -R test_post_barrier_divergence -V 2>&1 | tail -40
```

Expected: 5 个测试用例全部通过（包含 BUG-REPRODUCTION 和 FIX-VERIFICATION 测试）。

- [ ] **Step 3: 运行快速 sanity 确认无回归**

```bash
cd /workspace/project/PTX-EMU && ./scripts/sanity.sh --quick 2>&1 | tail -20
```

Expected: 所有测试通过（SUCCESS 或 0 failures）。

---

### Task 5: 清理 barrier.cpp 中的临时修复（可选，合并后无需补丁）

**Files:**
- Read: `src/ptxsim/instructions/barrier.cpp:180-185`

- [ ] **Step 1: 检查 barrier.cpp 中的 set_active_mask 调用**

由于 Task 1-3 已经让 `update_active_mask()` 统一处理了 `warp_state.threads[i].is_active`，`barrier.cpp:184` 的 `warp_ctx->set_active_mask(wbar.arrived_mask)` 可以保留（起到立即生效的作用）或者移除（让下一周期 `update_active_mask()` 统一更新）。

```bash
cd /workspace/project/PTX-EMU
sed -n '178,190p' src/ptxsim/instructions/barrier.cpp
```

建议：**保留**该调用——它在同一屏障释放周期立即更新 `active_mask[]`，避免下一周期延迟。

- [ ] **Step 2: 提交（可选）**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/core/warp_context.cpp include/ptxsim/warp_context.h src/ptxsim/core/thread_context.cpp
git commit -m "fix: ISSUE-004 active_mask consistency — unify dual sources, delegate to is_schedulable, fix barrier release is_active"
```
