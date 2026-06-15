# 04. PTX-EMU 当前 Barrier 实现 — 代码地图

> **子代理调研任务**：`bg_59c0f18c` — Map project barrier implementation  
> **调研日期**：2026-06-15  
> **主题**：本项目 barrier 系统的代码地图与已知 bug  
> **来源**：项目源码 + git history

---

## 1. `bar.warp.sync` 实现 — `BarWarpSyncHandler::processOperation()`

**文件**: `/workspace/project/PTX-EMU/src/ptxsim/instructions/barrier.cpp:108-270`

### 1.1 完整状态机（按行号拆解）

| 行号 | 阶段 | 行为 |
|------|------|------|
| 116 | 读取 static_mask | `*operands[0]` = PTX 静态参与掩码 |
| 118 | 读取 reconvergence_pc | `*operands[1]` = 屏障后跳转的 PC |
| 131-143 | **动态掩码构建**（仅在 `current_wbar_id < 0` 时） | 遍历 32 个 lane，匹配 `pc == current_pc \|\| next_pc == current_pc` |
| 145 | `get_unique_pcs()` | 当前 warp 中不同 PC 的数量 |
| 146-202 | **force_reconvergence 路径**（仅当 `unique_pcs > 1` 且 `current_wbar_id < 0`） | 触发强制重汇合 |
| 208-269 | **正常路径** | 初始化/复用 wbar、arrive、检查完成、释放 |

### 1.2 参与掩码源 — 静态 vs 动态

```cpp
// barrier.cpp:131-143 (动态掩码构建) - 仅 normal path 实际使用
uint32_t dynamic_mask = 0;
if (warp_state.current_wbar_id < 0) {
    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active) continue;
        if (warp_state.threads[i].is_exited) continue;
        if (warp_state.threads[i].pc == current_pc ||
            warp_state.threads[i].next_pc == static_cast<uint32_t>(current_pc)) {
            dynamic_mask |= (1u << i);
        }
    }
}
```

**关键设计决策**：
- **force_reconvergence 路径**（行 156）：**直接使用 `static_mask`**，因为分歧时动态掩码不完整
- **正常路径**（行 216）：`participation_mask = dynamic_mask != 0 ? (dynamic_mask & static_mask) : static_mask`

### 1.3 `arrive()` 调用时机

| 路径 | 调用位置 | 谁调用 |
|------|---------|--------|
| force_reconvergence | barrier.cpp:173 | 当前 lane（init 后立即 arrive）|
| 正常路径 | barrier.cpp:224 | 当前 lane |

### 1.4 `is_complete()` 计算

```cpp
// wbar.h:31-36
bool is_complete() const {
    if (!is_initialized || participation_mask == 0) {
        return false;
    }
    return (arrived_mask & participation_mask) == participation_mask;
}
```

### 1.5 屏障释放时 active_mask / exec_mask 操作

**两处释放点**（均含 BUG-POSTBARRIER-TWOHALVES 修复）:

| 位置 | 行号 | 行为 |
|------|------|------|
| force_reconvergence 路径 | barrier.cpp:179, 190-191 | `set_exec_mask(arrived_mask)` + `set_active_mask(get_active_mask() \| arrived_mask)` |
| 正常路径 | barrier.cpp:240, 258-259 | 同上 |

```cpp
// barrier.cpp:258-259 (修复后)
warp_ctx->set_active_mask(
    warp_ctx->get_active_mask() | wbar.arrived_mask);
```

---

## 2. `bar.sync` (CTA 级) 实现

### 2.1 `BarHandler::executeBarrier` — barrier.cpp:325-376

```cpp
// barrier.cpp:353-355 (强制重汇合决策)
if (warp_ctx->get_unique_pcs().size() > 1) {
    warp_ctx->force_reconvergence_at_barrier(context->get_pc());
}

// barrier.cpp:361
bool sync_complete = sm_context->synchronize_barrier(barId, context);

if (sync_complete) {
    context->set_next_pc(context->get_pc() + 1);   // 释放
} else {
    context->set_next_pc(context->get_pc());        // 等待
}
```

### 2.2 `SMContext::synchronize_barrier` — sm_context.cpp:605-706

**两种释放机制**:

| 机制 | 位置 | 触发条件 |
|------|------|---------|
| **批量释放** | sm_context.cpp:207-242（`exe_once` 内部） | 每个 cycle 检查 `barrier_waiting_threads[barId]` |
| **同步返回** | sm_context.cpp:660-682 | 同步调用时所有线程已到齐 |

**关键代码片段**：
```cpp
// sm_context.cpp:660-682
if (barrier_waiting_threads[barId].size() >= barrier_thread_counts[barId]) {
    for (auto waiting_thread : barrier_waiting_threads[barId]) {
        waiting_thread->set_state(RUN);
        waiting_thread->set_next_pc(waiting_thread->get_pc() + 1);
        // 清除 warp_state blocked 状态
        wc->get_warp_state().threads[lid].is_blocked = false;
        wc->get_warp_state().threads[lid].status = ptxsim::ThreadStatus::Active;
        waiting_thread->sync_to_warp_state();
    }
    barrier_waiting_threads[barId].clear();
    return true;
}
```

---

## 3. Wbar 数据结构 — `/workspace/project/PTX-EMU/include/ptxsim/wbar.h`

### 3.1 字段语义

| 字段 | 类型 | 语义 |
|------|------|------|
| `participation_mask` | uint32_t | **静态**——PTX 指令指定的应参与线程位图 |
| `arrived_mask` | uint32_t | **动态**——已调用 `arrive()` 的 lane 位图 |
| `reconvergence_pc` | int | 屏障完成后所有 lane 跳转到的 PC |
| `barrier_pc` | uint32_t | 屏障指令本身的 PC（用于调试） |
| `is_initialized` | bool | 是否已调用 `init()` |
| `expected_count` | uint32_t | `popcount(participation_mask)` |

### 3.2 `arrive()` / `is_complete()` / `count_arrived()`

```cpp
// wbar.h:46-50
void arrive(int lane_id) {
    if (lane_id >= 0 && lane_id < 32) {
        arrived_mask |= (1u << lane_id);
    }
}

// wbar.h:31-36
bool is_complete() const {
    return (arrived_mask & participation_mask) == participation_mask;
}

// wbar.h:42-44
int count_arrived() const {
    return __builtin_popcount(arrived_mask);  // 累计已到达
}
```

**注意**: `count_arrived()` 返回 `arrived_mask` 的总位数，**不限制**于 `participation_mask` 范围。`is_complete()` 才是判断所有参与者到齐的正确语义。

### 3.3 Wbar 数量

`warp_state.h:17` 声明 `std::array<Wbar, 4> wbars;` —— 每个 warp 最多 4 个 Wbar。

---

## 4. `force_reconvergence_at_barrier` — `/workspace/project/PTX-EMU/src/ptxsim/core/warp_context.cpp:440-448`

```cpp
void WarpContext::force_reconvergence_at_barrier(int barrier_pc) {
    // 不主动推进线程PC —— 让调度器自然选择非阻塞的PC执行
    // 屏障处理器会在 divergence 路径中阻塞当前线程（set is_blocked=true），
    // 调度器随后会跳过有阻塞线程的PC组，选择其他PC执行。
    // 当所有线程都到达屏障后，wbar 完成并释放所有线程。
    //
    // 注意：不能推进线程PC，否则会跳过 shared memory store 等关键指令。
    // 注释掉的代码（advance_thread_pc）曾导致 E2E 测试中共享内存数据丢失。
}
```

**实际行为**: **空操作（No-op）**！方法体只有注释，不做任何事情。
- 设计哲学: **不推进 PC** → 留给调度器自然处理
- 依赖副作用: 调用方在 force_reconvergence 后会立即执行 `is_blocked = true` 让调度器跳过
- 历史教训: 早期曾 `advance_thread_pc` → 导致 E2E 共享内存 store 丢失

**调用方**：

| 调用方 | 位置 | 后续行为 |
|--------|------|---------|
| `BarWarpSyncHandler::processOperation` | barrier.cpp:147 | 设置 `current_wbar_id = 0`，init wbar，arrive |
| `BarHandler::executeBarrier` | barrier.cpp:354 | 调用 `synchronize_barrier` |

---

## 5. 已知已修复的 Bug 总结

| Bug | 状态 | 触发场景 | 症状 | 修复 |
|-----|------|---------|------|------|
| **BUG-RETHANG** | FIXED 2026-06 | `ret` 指令发散时 | `WarpContext::is_finished()` 永远 false，warp 调度死循环 | 标记**所有 32 lane** 为 exited（`is_exited=true`, `state=EXIT`），再 `update_active_mask()` |
| **BUG-POSTBARRIER-TWOHALVES** | FIXED 2026-06 | divergent warp 两半在不同 cycle 到达同一 `bar.warp.sync` | 第二次释放覆写 `active_mask`，丢失第一次释放的 lane | 释放前 `set_active_mask(get_active_mask() \| arrived_mask)`（**caller 层 OR，不改 `set_active_mask` 全局语义**） |
| **BUG-RECONVERGENCE-SIMPLEGEMM** | FIXED 2026-06 | simpleGEMM 风格的 `bar.sync` 翻译为 `bar.warp.sync`，第一半 lane 16-31 释放后被 wbar 重新 init 抹除到达记录 | 后续到达永远无法凑齐 `participation_mask` → barrier 永不完成 → 后到 lane 永远卡在 barrier PC | wbar 已初始化时**只更新** `participation_mask` / `reconvergence_pc`，**保留 `arrived_mask`** |

---

## 6. 最近 Barrier 相关 Commits

### 6.1 `09de279` — BUG-POSTBARRIER-TWOHALVES 修复 (2026-06-14)

**文件变更**: `src/ptxsim/instructions/barrier.cpp`

**Diff 摘要** (2 处):
```diff
- warp_ctx->set_active_mask(init_wbar.arrived_mask);  // barrier.cpp:170
+ warp_ctx->set_active_mask(
+     warp_ctx->get_active_mask() | init_wbar.arrived_mask);

- warp_ctx->set_active_mask(wbar.arrived_mask);        // barrier.cpp:235
+ warp_ctx->set_active_mask(
+     warp_ctx->get_active_mask() | wbar.arrived_mask);
```

**Bug 解释**: divergent warp 的两条路径在不同 cycle 到达同一 barrier，force_reconvergence 路径为每半初始化**全新** wbar。第二次释放时 `set_active_mask(arrived_mask)` 仅含第二半，**覆写**了第一半已经释放的 lane。修复: 改为 OR 合并。

### 6.2 `5820f7e` — BUG-RECONVERGENCE-SIMPLEGEMM 修复 (2026-06-14)

**文件变更**: `src/ptxsim/instructions/barrier.cpp` (+15, -1)

**Diff 摘要** (1 处):
```diff
- init_wbar.init(participation_mask, reconvergence_pc);  // barrier.cpp:158
+ if (!init_wbar.is_initialized) {
+     init_wbar.init(participation_mask, reconvergence_pc);
+ } else {
+     init_wbar.participation_mask = participation_mask;
+     init_wbar.reconvergence_pc = reconvergence_pc;
+     init_wbar.expected_count = __builtin_popcount(participation_mask);
+     init_wbar.is_initialized = true;
+ }
```

**Bug 解释**: simpleGEMM kernel 中 `bar.sync` 翻译为 `bar.warp.sync` 后，lanes 16-31 先到达 → wbar init + arrive(16-31) → 完成 → release lanes 16-31。随后 lanes 0-15 到达 → 再次走 force_reconvergence 路径 → `wbar.init()` **重置** `arrived_mask`（因 `init()` 内部调用 `reset()`，wbar.h:62）→ arrive(0-15) → 但 `participation_mask` 全 32 lane，永远凑不齐。修复: 已 init 时只更新 mask/pc，**保留** `arrived_mask`。

### 6.3 其他相关历史 Commits

| Commit | 描述 |
|--------|------|
| `e8c9e41` | fix(sm-admission): streaming block admission (BUG-SM-ADMISSION-OVERFLOW) |
| `7405286` | refactor(ptxsim): set is_active default to false in WarpContext ctor |
| `b95ebf7` | fix(ptxsim): recalculate active_count after blocked-decrement cycle |
| `1d896df` | refactor(ptxsim): extract decrement_blocked_cycles + add E2E hang regression test |
| `3629243` | fix(scheduler): B4.1 blocked-finish cascade bug |
| `a5cd7fa` | fix(barrier): preserve blocked status in divergence path |
| `e087e4f` | ptxsim: implement warp-level divergence reconvergence at barrier |
| `c0e67ae` | feat: Phase 1.3 - integrate BsyncManager into bar.warp.sync handler |

---

## 7. 屏障相关测试

### 7.1 单元测试 (`tests/unit/barrier/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_barrier_module.cpp` | WarpBarrier / CTABarrier / BarrierModule 基础数据结构（init/reset 状态、masks、counts） |
| `test_warp_barrier.cpp` | Wbar API 直接调用（arrive/is_complete/count_arrived/reset） |
| `test_barrier_reconvergence.cpp` | barrier 与 reconvergence 交互 |
| `test_barrier_scenarios.cpp` | 各种 barrier 场景（基础） |
| `test_barrier_scenarios_integrated.cpp` | 同上，集成版（驱动 execute_warp_instruction） |
| `test_barrier_interaction_integrated.cpp` | barrier 与其他指令的交互 |
| `test_barrier_verification.cpp` | 屏障验证逻辑 |
| `test_post_barrier_two_halves.cpp` | **BUG-POSTBARRIER-TWOHALVES** 直接测试 `set_active_mask(arrived_mask)` vs OR 语义 |
| `test_barrier_divergence_reconvergence_simplegemm.cpp` | **BUG-RECONVERGENCE-SIMPLEGEMM** 在 Wbar 层面的场景重现 |

### 7.2 集成测试 (`tests/integration/barrier/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_warp_barrier_integrated.cpp` | 通过 `step_warp()` 完整驱动 warp 屏障执行（**当前失败**） |
| `test_cta_barrier_memory_visibility.cpp` | CTA 级屏障的 shared memory 可见性 |
| `test_warp_barrier_memory_visibility.cpp` | warp 级屏障的 shared memory 可见性 |
| `test_barrier_full_lifecycle.cpp` | 屏障完整生命周期（init → arrive → release → 复用） |
| `test_barrier_divergence_scheduling.cpp` | 屏障在分歧调度器下的行为 |
| `test_barrier_module_integrated.cpp` | BarrierModule 集成测试 |
| `test_barrier_verification_integrated.cpp` | 屏障验证逻辑集成测试 |

### 7.3 分歧 + 屏障集成测试 (`tests/integration/divergence/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_post_barrier_two_halves.cpp` | **BUG-POSTBARRIER-TWOHALVES** smoke test |
| `test_post_barrier_reconvergence_simplegemm.cpp` | **BUG-RECONVERGENCE-SIMPLEGEMM** 端到端 |
| `test_post_barrier_divergence.cpp` | barrier 后分歧行为（**已知问题**: 文档中标注 `synchronize_barrier()` 可能不正确更新 active_mask） |
| `test_nested_divergence.cpp` | 嵌套分歧 |
| `test_divergence_sync_convergence.cpp` | 分歧同步收敛（基础） |

---

## 8. 关键架构洞察（不变量与风险）

### 8.1 Dual State Mechanism（来自 `src/ptxsim/core/AGENTS.md`）

| 状态 | 类型 | 写入者 |
|------|------|--------|
| `active_mask[32]` (bool) | 调度器视图 | `set_active_mask()`, `update_active_mask()` |
| `warp_state.threads[i].is_active` | 源真相 | `set_active_mask()`, `update_active_mask()`, `decrement_blocked_cycles` |
| `warp_state.exec_mask` (uint32_t) | PTX `activemask` 指令返回值 | `set_exec_mask()` |

**关键不变量**: 每次 `execute_warp_instruction()` 末尾的 `update_active_mask()` 会**从 `is_active` 重建 `active_mask[]`**。因此**临时性的 `active_mask` 错误会被下一条指令自愈**。

**设计后果**：
- E2E 集成测试可能 PASS（自愈掩盖了 bug）
- 单元测试直接断言 `get_active_mask()` 立即调用后的值才能捕获 bug
- 这是为什么 BUG-POSTBARRIER-TWOHALVES 既有 unit 测试（捕获）又有 integration 测试（smoke）

### 8.2 SCOPE-OF-EFFECT 原则

影响 warp 级状态的指令 handler（`ret`、barrier、branch reconvergence）必须考虑**所有 32 个 lane**，不能只处理当前执行 lane：
- `ret` handler: 必须把所有 32 lane 都标 `is_exited=true` + `state=EXIT`
- barrier handler: 必须用 OR 合并 `active_mask`
- 调用 `update_active_mask()` 必须在 scheduler 看到之前 reconcile

### 8.3 已知未解决问题

来自 `src/ptxsim/core/AGENTS.md`:
> // synchronize_barrier() may not update active_mask correctly after barrier release
> // See: tests/integration/divergence/test_post_barrier_divergence.cpp (2 TEST_CASE documenting the issue)

**仍有 2 个 TEST_CASE 在记录 active_mask 更新问题**——属于已修复 BUG 的回归测试，但具体细节需进一步审查。

---

## 🎯 核心架构总结

1. **3 条 barrier 路径**:
   - (a) `bar.warp.sync` 正常路径（单 PC 屏障）
   - (b) `bar.warp.sync` force_reconvergence 路径（多 PC 屏障）
   - (c) `bar.sync` CTA 级路径（synchronize_barrier）

2. **Wbar 是 warp 级屏障的核心**：含 `participation_mask`（静态期望）和 `arrived_mask`（动态已到），`is_complete()` 通过位运算判断是否全员到齐。每个 warp 有 4 个 Wbar 槽。

3. **force_reconvergence_at_barrier 故意是空操作**：依赖调用方在 force_reconvergence 后立即 `is_blocked=true`，让调度器自然选择其他 PC 组。这是"lazy reconciliation"设计。

4. **Dual state 是 self-healing 机制**：`active_mask[32]` 错误会被 `update_active_mask()` 下一周期从 `is_active` 重建。

5. **两个最近修复的 bug 都围绕"两半释放"问题**：
   - BUG-POSTBARRIER-TWOHALVES: 第二次释放覆写第一次释放的 active_mask → 修复: OR 合并
   - BUG-RECONVERGENCE-SIMPLEGEMM: 第二次 force_reconvergence 重置 wbar 的 arrived_mask → 修复: 已 init 时只更新 mask/pc，保留 arrived_mask
