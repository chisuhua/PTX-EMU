# T2-1 — active_mask 三源统一 (Tasks)

> **总工期**: 3-5 天 | **依赖**: T1-1..T1-5 ✅, integrate-barrier-module-cta-warp **必须先合并**

> **基础 plan**: `docs/superpowers/plans/2026-05-07-active-mask-consistency.md`（14 步已列）
> 本 tasks.md 是该 plan 的工程化版本（增加 file:line + 验证步骤 + commit 规范）。

> **TDD 三阶段**: 严格遵循 AGENTS.md §TDD 流程。

## Task 1: 修改 `update_active_mask()` 统一数据源

**Files:**
- Modify: `src/ptxsim/core/warp_context.cpp:316-329`（`update_active_mask()` 主体）
- Read reference: `include/ptxsim/thread_state.h:54-59`（`is_schedulable()` canonical 定义）

- [ ] **Step 1: 读取当前实现**

```bash
cd /workspace/project/PTX-EMU
sed -n '316,330p' src/ptxsim/core/warp_context.cpp
```

预期：看到当前 `update_active_mask()` 从 `warp_state.threads[i]` 4 个字段读取。

- [ ] **Step 2: 修改 update_active_mask() 统一从 warp_state 读写**

按 `docs/superpowers/plans/2026-05-07-active-mask-consistency.md:38-48` 的伪代码实现，确保：
- 从 `warp_state.threads[i].is_active` 等 4 字段读取
- 双向同步写回 `warp_state.threads[i].is_active`
- `active_count` 正确累加

- [ ] **Step 3: 编译验证**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim 2>&1 | tail -10
```

预期：编译成功。

- [ ] **Step 4: 跑现有测试不回归**

```bash
cd /workspace/project/PTX-EMU/build
ctest -R "active_mask|post_barrier|barrier_module" -V 2>&1 | tail -30
```

预期：全部通过。

- [ ] **Step 5: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/core/warp_context.cpp
git commit -m "refactor(warp): unify update_active_mask() to read from warp_state (T2-1 Task 1)"
```

---

## Task 2: 让 `is_lane_active()` 委托 `is_lane_schedulable()`

**Files:**
- Modify: `include/ptxsim/warp_context.h:124-126`（`is_lane_active()` 实现）

- [ ] **Step 1: 读取当前实现**

```bash
cd /workspace/project/PTX-EMU
sed -n '120,135p' include/ptxsim/warp_context.h
```

- [ ] **Step 2: 委托实现**

```cpp
// 从：仅读取 active_mask[]
// bool is_lane_active(int lane_id) const { return lane_id >= 0 && lane_id < WARP_SIZE && active_mask[lane_id]; }

// 改为：委托给 warp_state 权威源
bool is_lane_active(int lane_id) const {
    return is_lane_schedulable(lane_id);
}
```

- [ ] **Step 3: 编译验证 + 跑测试**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim
cd build && ctest -R "active_mask|is_lane" -V 2>&1 | tail -20
```

- [ ] **Step 4: 提交**

```bash
git add include/ptxsim/warp_context.h
git commit -m "refactor(warp): delegate is_lane_active() to is_lane_schedulable() (T2-1 Task 2)"
```

---

## Task 3: 在 `sync_to_warp_state(RUN)` 中添加 `is_active=true`

**Files:**
- Modify: `src/ptxsim/core/thread_context.cpp:735-742`（`sync_to_warp_state` RUN case）

- [ ] **Step 1: 读取 sync_to_warp_state RUN 分支**

```bash
cd /workspace/project/PTX-EMU
sed -n '730,750p' src/ptxsim/core/thread_context.cpp
```

- [ ] **Step 2: 添加 is_active = true**

```cpp
case RUN:
    thread_state.status = ThreadStatus::Active;
    thread_state.is_blocked = false;
    thread_state.is_active = true;  // <-- 新增：屏障释放后标记线程为活跃
    break;
```

- [ ] **Step 3: 编译 + 跑测试**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build --target ptxsim
cd build && ctest -R "active_mask|post_barrier_two_halves" -V 2>&1 | tail -30
```

预期：post_barrier_two_halves 不回归。

- [ ] **Step 4: 提交**

```bash
git add src/ptxsim/core/thread_context.cpp
git commit -m "fix(thread): set is_active=true in sync_to_warp_state(RUN) (T2-1 Task 3)"
```

---

## Task 4: 启用 J1-J10 active_mask 一致性测试

**Files:**
- Verify: `tests/unit/exec/test_active_mask_consistency.cpp`（234 行，J1-J10 应已存在但未启用）
- Read: `scripts/sanity.sh:142,158`（确认测试注册）

- [ ] **Step 1: 检查测试当前状态**

```bash
cd /workspace/project/PTX-EMU
cd build && ctest -N 2>&1 | grep -i active_mask
```

- [ ] **Step 2: 启用 J1-J10（如未启用）**

如测试被注释 / DISABLED：删除 `DISABLED TRUE` 属性或在 CMakeLists.txt 注册。

- [ ] **Step 3: 跑测试**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build
cd build && ctest -R "active_mask_consistency" -V 2>&1 | tail -50
```

预期：J1-J10 全部通过。

- [ ] **Step 4: 跑全量 sanity**

```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh --quick 2>&1 | tail -20
```

预期：所有测试通过（SUCCESS 或 0 failures）。

- [ ] **Step 5: 提交（如有改动）**

```bash
git add tests/unit/exec/test_active_mask_consistency.cpp tests/unit/CMakeLists.txt
git commit -m "test(active-mask): enable J1-J10 consistency tests (T2-1 Task 4)"
```

---

## Task 5: 清理 `barrier.cpp` 临时修复（可选）+ 重写 AGENTS.md

**Files:**
- Read: `src/ptxsim/instructions/barrier.cpp:180-185`（评估是否保留 `set_active_mask`）
- Modify: `src/ptxsim/core/AGENTS.md:49-60`（重写为 SINGLE SOURCE OF TRUTH）
- Modify: `include/ptxsim/warp_state.h:17-18`（`wbars` + `current_wbar_id` 标 `[[deprecated]]`，**不删**）

- [ ] **Step 1: 评估 barrier.cpp:178-190 的 set_active_mask 调用**

```bash
cd /workspace/project/PTX-EMU
sed -n '178,200p' src/ptxsim/instructions/barrier.cpp
```

建议：**保留**（OR-merge 仍由 BarrierModule 内部承担，但 caller 显式 set 可让同周期生效，避免下一周期延迟）。

- [ ] **Step 2: 重写 src/ptxsim/core/AGENTS.md:49-60**

替换 "DUAL STATE MECHANISM" 段为 "SINGLE SOURCE OF TRUTH"，明确：
- 权威源：`warp_state.threads[i].is_schedulable()`（定义于 `thread_state.h:54-59`）
- 派生：`active_mask[]`（由 `update_active_mask()` 周期重算）
- 派生：`active_count`（由 `set_active_mask()` 内部维护）
- 例外：`warp_state.exec_mask` 是 PTX `activemask` 指令的独立源（不与 `is_active` 共享）

- [ ] **Step 3: 标记 wbars / current_wbar_id 为 deprecated**

修改 `include/ptxsim/warp_state.h:17-18`：
```cpp
// 修改前
std::array<Wbar, 4> wbars;  // deprecated
int current_wbar_id = -1;

// 修改后
[[deprecated("Use BarrierModule::get_warp_barrier() instead - will be removed in T2-3")]]
std::array<Wbar, 4> wbars;
[[deprecated("Use BarrierModule state instead - will be removed in T2-3")]]
int current_wbar_id = -1;
```

- [ ] **Step 4: 编译验证（deprecated warning 应不破坏构建）**

```bash
cd /workspace/project/PTX-EMU && . env.sh
cmake --build build 2>&1 | grep -E "warning|deprecated" | head -20
```

预期：可看到 `wbars` / `current_wbar_id` 访问点的 deprecated warning，**不应**有 error。

- [ ] **Step 5: 跑全量验证**

```bash
cd /workspace/project/PTX-EMU
./scripts/sanity.sh --quick 2>&1 | tail -20
```

预期：全部通过（warning 不影响测试结果）。

- [ ] **Step 6: 提交**

```bash
cd /workspace/project/PTX-EMU
git add src/ptxsim/core/AGENTS.md include/ptxsim/warp_state.h
git commit -m "docs(agents): rewrite DUAL STATE MECHANISM section + deprecate wbars (T2-1 Task 5)"
```

---

## Phase 3 验证门禁

- [ ] J1-J10 全部通过
- [ ] `test_post_barrier_divergence` + `test_post_barrier_two_halves` 不回归
- [ ] `ctest -L "barrier;simt;exec;sync"` 全过
- [ ] `./scripts/sanity.sh --quick` 全过
- [ ] ASan 跑 e2e_barrier_warp_sync 无 LeakSanitizer
- [ ] 文档：`src/ptxsim/core/AGENTS.md` §DUAL STATE 段已重写
