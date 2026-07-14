# BarrierModule 集成 Change 修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 OpenSpec change `integrate-barrier-module-cta-warp` 的 3 个 Blocker + 6 个 Major issues，使其从 REJECT 升级到 APPROVE-WITH-CONDITIONS。

**Architecture:** 三层修复：(1) 代码层修复 `BarrierModule` / `WarpBarrier` 的 2 个 bug 回归风险；(2) 文档层修复 `change` artifacts 的 6 处一致性问题；(3) 验证层加 4 道测试 gate 确保修复后无回归。

**Tech Stack:** C++20、ANTLR4 PTX、OpenSpec、Catch2、ctest、CMake

**Prerequisites:**
- 当前分支 `main`（commit `baa8c4e`）
- 已评审的 change：`openspec/changes/integrate-barrier-module-cta-warp/`
- 已加载 skill：`.opencode/skills/ptx-barrier-mechanism/`
- 项目 invariant 来源：`src/ptxsim/core/AGENTS.md`（BUG-POSTBARRIER-TWOHALVES 必须在 CALLER 实施 OR）

**评审 session ID（可续接 Momus）：** `ses_128115580ffegArLPu5lzZKpi0`

---

## 文件变更总览

| 文件 | 操作 | 责任 |
|---|---|---|
| `src/ptxsim/barrier/barrier_module.cpp` | 修改 L83-117 | Blocker 1：release_warp_barrier 加 OR 逻辑 |
| `src/ptxsim/barrier/warp_barrier.cpp` | 修改 L13-25 | Blocker 2：WarpBarrier::init 保留 arrived_mask |
| `include/ptxsim/barrier/warp_barrier.h` | 不变 | — |
| `include/ptxsim/barrier/barrier_module.h` | 不变 | — |
| `openspec/changes/integrate-barrier-module-cta-warp/tasks.md` | 新增 4 个 task、修改 2 个 task | Blocker 1/2 + Major 5/6 |
| `openspec/changes/integrate-barrier-module-cta-warp/specs/warp-barrier-unification/spec.md` | 修改 L19 | Major 3（拼写）|
| `openspec/changes/integrate-barrier-module-cta-warp/design.md` | 修改 L178 + 风险表 | Major 1 + Major 4 |
| `openspec/changes/integrate-barrier-module-cta-warp/proposal.md` | 修改 5 处行号 | Major 2 |
| `docs/adr/0008-barrier-semantics.md` | 追加 §"2026-06-17 追加" | Major 8 |
| `tests/unit/barrier/test_barrier_module.cpp` | 新增 1 个 TEST_CASE | Blocker 1 验证门 |

---

## Phase 0: Pre-flight 审计（30 min）

### Task 0.1: 创建 worktree + 基线

**Files:**
- 无（git 操作）

- [ ] **Step 1: 创建 worktree**

```bash
cd /workspace/project/PTX-EMU
git worktree add ../ptx-emu-barrier-fix -b fix/integrate-barrier-module-review
cd ../ptx-emu-barrier-fix
```

- [ ] **Step 2: 验证 worktree 与主分支同步**

```bash
git log --oneline -1 main
git log --oneline -1 HEAD
```

Expected: 两条 commit hash 一致（`baa8c4e`）

- [ ] **Step 3: 跑基线测试**

```bash
. env.sh
cmake --build build --target ptxsim 2>&1 | tail -10
cd build && ctest -L "barrier;exec_mask;simt_entry" --output-on-failure 2>&1 | tail -30
```

Expected: 全部 PASS，记录基线输出到 `/tmp/baseline_$$.txt`

---

## Phase 1: Blocker 代码修复（2 hour）

### Task 1.1: 🔴 修复 Blocker 1 — `release_warp_barrier` 加 OR 逻辑

**Files:**
- Modify: `src/ptxsim/barrier/barrier_module.cpp:83-117`
- Test: `tests/unit/barrier/test_barrier_module.cpp`

**Context:** 当前 `barrier_module.cpp:105` 用 `warp_ctx->set_exec_mask(arrived_mask)` 覆写，会回归 BUG-POSTBARRIER-TWOHALVES。修复必须遵循 `src/ptxsim/core/AGENTS.md` 的不变量：OR 逻辑必须在 CALLER，不能改 `set_active_mask` 自身。

- [ ] **Step 1: 先写失败测试（验证 BUG-POSTBARRIER-TWOHALVES 不回归）**

在 `tests/unit/barrier/test_barrier_module.cpp` 末尾追加：

```cpp
TEST_CASE("release_warp_barrier preserves active_mask via OR (BUG-POSTBARRIER-TWOHALVES)",
          "[barrier][release][warp_barrier]") {
    // Setup: 模拟"两个 divergent half 命中同一 barrier"场景
    SMContext sm(4, 128, 4096, 0);
    CTAContext cta(&sm, 0, 0);
    WarpContext warp(&cta, 0);

    // 初始 active_mask = 全 32 lanes
    warp.set_active_mask(0xFFFFFFFFu);

    BarrierModule mod;

    // 第一次 release：第一半（lane 0-15）到达
    mod.init_warp_barrier(0, 0x0000FFFFu, 21, 20);
    for (int i = 0; i < 16; ++i) {
        mod.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(mod.is_warp_barrier_complete(0));

    // 第一次 release 后，active_mask 应该是全 32 lanes（OR 逻辑）
    mod.release_warp_barrier(0, &warp);
    REQUIRE(warp.get_active_mask() == 0x0000FFFFu);  // 第一半先 OR 进来

    // 第二次 release：第二半（lane 16-31）到达（force_reconvergence 路径）
    mod.init_warp_barrier(0, 0xFFFF0000u, 21, 20);
    for (int i = 16; i < 32; ++i) {
        mod.arrive_at_warp_barrier(0, i);
    }
    REQUIRE(mod.is_warp_barrier_complete(0));

    // 关键断言：第二次 release 后 active_mask MUST = 全 32 lanes（不是覆写为 0xFFFF0000）
    mod.release_warp_barrier(0, &warp);
    CHECK(warp.get_active_mask() == 0xFFFFFFFFu);  // 关键：OR 合并，不是覆写
}
```

- [ ] **Step 2: 编译并验证测试失败（当前实现必失败）**

```bash
cmake --build build --target ptxsim
cd build && ctest -R "release_warp_barrier preserves" -V
```

Expected: 测试 FAIL，第二半 release 后 `active_mask` 被覆写为 `0xFFFF0000` 而非 `0xFFFFFFFF`

- [ ] **Step 3: 实施修复 — 修改 `release_warp_barrier`**

修改 `src/ptxsim/barrier/barrier_module.cpp:105`：

```cpp
    // BUG-POSTBARRIER-TWOHALVES fix (per src/ptxsim/core/AGENTS.md):
    // OR with existing active_mask to preserve lanes already released
    // by a prior barrier call (e.g., when a divergent warp hits the
    // same barrier in two halves).
    // MUST live in the caller — set_active_mask semantics must not change
    // globally (ret handler relies on overwrite semantics).
    warp_ctx->set_active_mask(
        warp_ctx->get_active_mask() | arrived_mask);
    warp_ctx->set_exec_mask(arrived_mask);
```

- [ ] **Step 4: 重新编译并验证测试通过**

```bash
cmake --build build --target ptxsim
cd build && ctest -R "release_warp_barrier preserves" -V
```

Expected: PASS

- [ ] **Step 5: 跑 BUG-POSTBARRIER-TWOHALVES 回归测试（确保未引入新 bug）**

```bash
cd build && ctest -R "post_barrier_two_halves" -V
cd build && ctest -R "post_barrier_reconvergence" -V
```

Expected: 全部 PASS

- [ ] **Step 6: 提交**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
git add src/ptxsim/barrier/barrier_module.cpp tests/unit/barrier/test_barrier_module.cpp
git commit -m "fix(barrier): preserve active_mask in release_warp_barrier via OR

BUG-POSTBARRIER-TWOHALVES regression guard. The CALLER must OR with
existing active_mask (per src/ptxsim/core/AGENTS.md invariant).
set_active_mask semantics must not change globally because ret
handler relies on overwrite (set_active_mask(0u) to clear)."
```

---

### Task 1.2: 🔴 修复 Blocker 2 — `WarpBarrier::init` 保留 `arrived_mask`

**Files:**
- Modify: `src/ptxsim/barrier/warp_barrier.cpp:13-25`
- Test: `tests/unit/barrier/test_barrier_module.cpp`

**Context:** 当前 `warp_barrier.cpp:15-16` 无条件 `arrived_mask_ = 0; arrived_count_ = 0;`，会丢失 force_reconvergence 路径下第一半 lanes 的 arrived 状态。`barrier-handler-bugfix/spec.md:46-55` 明确要求 re-init 保留 arrived_mask。

- [ ] **Step 1: 先写失败测试（验证 re-init 保留 arrived_mask）**

在 `tests/unit/barrier/test_barrier_module.cpp` 追加：

```cpp
TEST_CASE("WarpBarrier::init preserves arrived_mask on re-init (BUG-RECONVERGENCE-SIMPLEGEMM)",
          "[barrier][warp_barrier][init]") {
    WarpBarrier wbar;

    // 第一次 init：第一半到达
    wbar.init(0x0000FFFFu, 21, 20);
    for (int i = 0; i < 16; ++i) {
        wbar.arrive(i);
    }
    REQUIRE(wbar.get_arrived_mask() == 0x0000FFFFu);
    REQUIRE(wbar.get_arrived_count() == 16);

    // 第二次 init：force_reconvergence 路径，必须保留 arrived_mask
    wbar.init(0xFFFF0000u, 21, 20);

    // 关键断言：arrived_mask MUST 被保留（不是 reset 为 0）
    CHECK(wbar.get_arrived_mask() == 0x0000FFFFu);
    CHECK(wbar.get_arrived_count() == 16);
    // 元数据应被更新
    CHECK(wbar.get_participation_mask() == 0xFFFF0000u);
    CHECK(wbar.is_initialized());
}
```

- [ ] **Step 2: 编译并验证测试失败**

```bash
cmake --build build --target ptxsim
cd build && ctest -R "WarpBarrier::init preserves" -V
```

Expected: FAIL，re-init 后 `arrived_mask == 0`（覆写丢失）

- [ ] **Step 3: 实施修复 — 修改 `WarpBarrier::init`**

修改 `src/ptxsim/barrier/warp_barrier.cpp:13-25`：

```cpp
void WarpBarrier::init(uint32_t participation_mask, int reconvergence_pc, uint32_t barrier_pc) {
    // BUG-RECONVERGENCE-SIMPLEGEMM fix (per barrier-handler-bugfix spec):
    // If already initialized, only update metadata (participation_mask,
    // reconvergence_pc, expected_count) but PRESERVE arrived_mask and
    // arrived_count. This is required by the force_reconvergence path
    // where a divergent half re-enters the barrier with a fresh
    // participation mask.
    if (is_initialized_) {
        participation_mask_ = participation_mask;
        reconvergence_pc_ = reconvergence_pc;
        barrier_pc_ = barrier_pc;
        expected_count_ = __builtin_popcount(participation_mask);
        state_ = State::Waiting;  // Already past Initializing
        // arrived_mask_/arrived_count_ intentionally NOT reset

        PTX_DEBUG_EMU("WarpBarrier::init RE-INIT mask=0x%X reconv=%d barrier_pc=%u "
                      "expected=%d (preserved arrived=0x%X count=%d)",
                      participation_mask, reconvergence_pc, barrier_pc,
                      expected_count_, arrived_mask_, arrived_count_);
        return;
    }

    participation_mask_ = participation_mask;
    arrived_mask_ = 0;
    expected_count_ = __builtin_popcount(participation_mask);
    arrived_count_ = 0;
    reconvergence_pc_ = reconvergence_pc;
    barrier_pc_ = barrier_pc;
    is_initialized_ = true;
    state_ = State::Initializing;

    PTX_DEBUG_EMU("WarpBarrier::init mask=0x%X reconv=%d barrier_pc=%u expected=%d",
                   participation_mask, reconvergence_pc, barrier_pc, expected_count_);
}
```

- [ ] **Step 4: 重新编译并验证测试通过**

```bash
cmake --build build --target ptxsim
cd build && ctest -R "WarpBarrier::init preserves" -V
```

Expected: PASS

- [ ] **Step 5: 跑 BUG-RECONVERGENCE-SIMPLEGEMM 回归测试**

```bash
cd build && ctest -R "post_barrier_reconvergence_simplegemm" -V
```

Expected: PASS

- [ ] **Step 6: 提交**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
git add src/ptxsim/barrier/warp_barrier.cpp tests/unit/barrier/test_barrier_module.cpp
git commit -m "fix(barrier): preserve arrived_mask on WarpBarrier re-init

BUG-RECONVERGENCE-SIMPLEGEMM fix. The force_reconvergence path
re-initializes a fresh wbar for each arriving half; arrived_mask
must be preserved across re-init so the second half accumulates
onto the first half's records, not resets them."
```

---

## Phase 2: Change Artifacts 修复（1 hour）

### Task 2.1: 🔴 更新 tasks.md — 增补 Task 2.2b/2.2c 与 6.2 扩展

**Files:**
- Modify: `openspec/changes/integrate-barrier-module-cta-warp/tasks.md`

- [ ] **Step 1: 在 §2 插入 Task 2.2b（release_warp_barrier OR 逻辑）**

在 `tasks.md` §"## 2. 扩展 BarrierModule" 的 Task 2.4 之前插入新任务：

```markdown
- [ ] 2.2b 修改 `src/ptxsim/barrier/barrier_module.cpp::release_warp_barrier`：在 `set_exec_mask` 前先 `set_active_mask(get_active_mask() | arrived_mask)` 实施 OR 逻辑；遵循 `src/ptxsim/core/AGENTS.md` 不变量（"OR logic must live in the caller"）；必须同时保留 `set_exec_mask(arrived_mask)` 调用（用于 PTX `activemask` 指令）
- [ ] 2.2b.1 验证：`cd build && ctest -R "release_warp_barrier preserves" -V` PASS；`ctest -R "post_barrier_two_halves" -V` 不回归
```

- [ ] **Step 2: 在 §2 插入 Task 2.2c（WarpBarrier::init 保留 arrived_mask）**

紧接 Task 2.2b 后插入：

```markdown
- [ ] 2.2c 修改 `src/ptxsim/barrier/warp_barrier.cpp::WarpBarrier::init`：增加 `if (is_initialized_)` 分支，仅更新 metadata（participation_mask、reconvergence_pc、barrier_pc、expected_count、state=Waiting），**不**重置 arrived_mask_/arrived_count_；保持首次 init 路径不变
- [ ] 2.2c.1 验证：`cd build && ctest -R "WarpBarrier::init preserves" -V` PASS；`ctest -R "post_barrier_reconvergence_simplegemm" -V` 不回归
```

- [ ] **Step 3: 扩展 Task 6.2 — 完整删除 sm_context.cpp:200-260 周期 barrier 检查**

修改 `tasks.md` §"## 6. 旧代码清理" 的 Task 6.2：

```markdown
- [ ] 6.2 从 `src/ptxsim/core/sm_context.h` + `sm_context.cpp` 删除：
  - `synchronize_barrier()` 方法体
  - `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` 字段
  - **`sm_context.cpp:200-260` 周期 barrier 检查代码块**（`exe_once` 内的 `for (auto &[barId, waiting_threads] : barrier_waiting_threads)` 整段）：该逻辑依赖 `barrier_mutex_` 和全局 `barrier_waiting_threads` map；删除后 barrier 同步由 `CTAContext::barrier_module_` 完全接管；MUST NOT 留下孤儿 mutex
```

- [ ] **Step 4: 在 §3 新增 Task 3.5（TSan 前置验证）**

在 `tasks.md` §"## 3. 新增 CTABarrier 完整流程单元测试" 的 Task 3.1 之前插入：

```markdown
- [ ] 3.0 前置验证：检查项目 CMake 是否配置 TSan 构建目标；若未配置，Task 3.2 改为使用现有 race detector 或 skip（标注 deferred）；NOTE：当前项目 CMake 中无 `-fsanitize=thread` 配置
```

- [ ] **Step 5: 验证 tasks.md 修改完整性**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
grep -n "^### Task\|^## " openspec/changes/integrate-barrier-module-cta-warp/tasks.md
```

Expected: 看到 `2.2b`, `2.2c`, `6.2` 扩展版，`3.0` 前置任务

- [ ] **Step 6: 提交**

```bash
git add openspec/changes/integrate-barrier-module-cta-warp/tasks.md
git commit -m "docs(openspec): address Momus review blockers in tasks.md

- Add Task 2.2b: release_warp_barrier OR logic (BUG-POSTBARRIER-TWOHALVES)
- Add Task 2.2c: WarpBarrier::init preserve arrived_mask (BUG-RECONVERGENCE)
- Expand Task 6.2: full removal of sm_context.cpp:200-260 barrier check
- Add Task 3.0: TSan pre-flight check"
```

---

### Task 2.2: 修复 spec 拼写错误 + 验证 M4 硬编码 0 约束

**Files:**
- Modify: `openspec/changes/integrate-barrier-module-cta-warp/specs/warp-barrier-unification/spec.md`

- [ ] **Step 1: 修复 L19 拼写错误 `barrive_at_warp_barrier` → `arrive_at_warp_barrier`**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
```

使用 edit 工具修改 `specs/warp-barrier-unification/spec.md` L19：

**oldString:**
```
- **AND** then call `barrive_at_warp_barrier(0, lane_id)`
```

**newString:**
```
- **AND** then call `arrive_at_warp_barrier(0, lane_id)`
```

- [ ] **Step 2: 增加 Scenario 约束 `arrive_at_warp_barrier` 仅支持 index 0**

在 `specs/warp-barrier-unification/spec.md` 的 `Requirement: BarWarpSyncHandler MUST use BarrierModule::arrive_at_warp_barrier` 块末尾增加：

```markdown
#### Scenario: bar.warp.sync uses only barrier slot 0
- **WHEN** any PTX `bar.warp.sync mask, reconv_pc` is executed
- **THEN** the handler MUST pass `warp_barrier_id=0` to `arrive_at_warp_barrier`
- **AND** MUST NOT pass any other `warp_barrier_id` (PTX ISA defines a single barrier per warp)
- **AND** if `warp_barrier_id != 0` is observed, emit `PTX_ERROR_EMU` and treat as no-op
```

- [ ] **Step 3: 验证修改**

```bash
grep -n "barrive_at_warp_barrier" openspec/changes/integrate-barrier-module-cta-warp/specs/warp-barrier-unification/spec.md
```

Expected: 无输出（拼写错误已修复）

- [ ] **Step 4: 提交**

```bash
git add openspec/changes/integrate-barrier-module-cta-warp/specs/warp-barrier-unification/spec.md
git commit -m "docs(openspec): fix typo and add barrier index 0 constraint scenario"
```

---

### Task 2.3: 修复 design.md 错误引用 + 风险表补充

**Files:**
- Modify: `openspec/changes/integrate-barrier-module-cta-warp/design.md`

- [ ] **Step 1: 修复 L178 错误引用 `bsync_state.h` → `bsync_state.cpp:14-24`**

使用 edit 工具修改 `design.md` L178：

**oldString:**
```
1. **`bsync_manager_` 在 `bsync_state.h:84-90` 的 `bsync(thread_id, lane_id, pc)` 是否有 BarrierModule 不承担的副作用？** 需在 Phase 1 完整审计
```

**newString:**
```
1. **`bsync_manager_` 在 `bsync_state.cpp:14-24` 的 `bsync(thread_id, lane_id, pc)` 是否有 BarrierModule 不承担的副作用？** 需在 Phase 1 完整审计
```

- [ ] **Step 2: 在风险表补充 sm_context.cpp:204 周期检查迁移风险**

在 `design.md` 风险表的"CTAContext 新增成员影响 CTA 创建/销毁的对称性"行**之前**插入：

```markdown
| `sm_context.cpp:200-260` 周期 barrier 检查逻辑依赖 `barrier_mutex_` 与全局 `barrier_waiting_threads` map | 中 | 高 | 完整 audit `sm_context.cpp` 中所有 `barrier_mutex_` 引用；删除时同时移除 `barrier_mutex_` 字段与周期检查代码块；CTA 同步由 `CTAContext::barrier_module_` 完全接管 |
```

- [ ] **Step 3: 验证修改**

```bash
grep -n "bsync_state\.[ch]" openspec/changes/integrate-barrier-module-cta-warp/design.md
grep -n "sm_context.cpp:200" openspec/changes/integrate-barrier-module-cta-warp/design.md
```

Expected: 两条都返回修正后的引用

- [ ] **Step 4: 提交**

```bash
git add openspec/changes/integrate-barrier-module-cta-warp/design.md
git commit -m "docs(openspec): fix bsync_state file reference + add sm_context barrier check migration risk"
```

---

### Task 2.4: 修复 proposal.md 行号引用

**Files:**
- Modify: `openspec/changes/integrate-barrier-module-cta-warp/proposal.md`

- [ ] **Step 1: 先精确核查 work-around 实际行号**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
grep -n "advance_thread_pc\|work-around\|advance pc" tests/integration/barrier/test_cta_barrier_memory_visibility.cpp | head -20
```

- [ ] **Step 2: 用实际行号替换 proposal.md 中所有 "L141-184"**

使用 `replaceAll` edit：

**oldString:** `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:141-184`

**newString:** `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:L138-184`（用 Step 1 实际查到的行号替换）

- [ ] **Step 3: 同时修正 design.md L17, 105, 143 的行号引用**

```bash
grep -n "L141-184\|141-184" openspec/changes/integrate-barrier-module-cta-warp/design.md
```

对每个匹配位置用 edit 工具替换为实际行号

- [ ] **Step 4: 验证修改**

```bash
grep -rn "L141-184\|141-184" openspec/changes/integrate-barrier-module-cta-warp/
```

Expected: 无输出（全部替换）

- [ ] **Step 5: 提交**

```bash
git add openspec/changes/integrate-barrier-module-cta-warp/proposal.md openspec/changes/integrate-barrier-module-cta-warp/design.md
git commit -m "docs(openspec): correct work-around line number references"
```

---

### Task 2.5: 追加 ADR-0008 §"2026-06-17 追加"

**Files:**
- Modify: `docs/adr/0008-barrier-semantics.md`

- [ ] **Step 1: 读取 ADR-0008 末尾**

```bash
tail -30 docs/adr/0008-barrier-semantics.md
```

- [ ] **Step 2: 追加新章节**

在 ADR-0008 末尾追加：

```markdown
## 2026-06-17 追加：BarrierModule 集成与状态机扩展

### 决策
- `BarrierModule` 由 `CTAContext` 持有（每个 CTA 一个实例），替代 `SMContext` 全局 mutex + map
- `release_cta_barrier(cta_barrier_id, cta_ctx)` 新增 `CTAContext*` 参数，用于遍历线程并调用 `set_state(RUN)` + `advance_thread_pc`
- `release_warp_barrier` 必须在 `set_exec_mask` 前实施 `set_active_mask(get_active_mask() | arrived_mask)`（OR 逻辑），由 CALLER 负责（不可改 `set_active_mask` 全局语义）
- `WarpBarrier::init` re-init 时仅更新 metadata，**保留** `arrived_mask` / `arrived_count`（force_reconvergence 路径需求）

### 状态机扩展
- `WarpBarrier::State` 新增 `Waiting`（已 init 但未 complete）和 `Complete`（全部到达），删除旧的 `Uninitialized → Active → Released` 三态模型
- `CTABarrier` 沿用相同状态机

### 移除项
- `include/ptxsim/wbar.h` 旧 `Wbar` 结构体
- `warp_state.h::wbars[]` + `current_wbar_id` 字段
- `SMContext::synchronize_barrier` + `barrier_waiting_threads` map + `barrier_mutex_` 字段
- `sm_context.cpp:200-260` 周期 barrier 检查代码块

### 合规检查项
- [ ] `release_warp_barrier` 调用前 `grep -n "set_active_mask.*|.*arrived_mask" src/ptxsim/barrier/barrier_module.cpp` 必须命中
- [ ] `WarpBarrier::init` 必须有 `if (is_initialized_) return;` 分支
- [ ] `set_active_mask` 全局实现 MUST NOT 修改（ret handler 依赖覆写语义清零）
```

- [ ] **Step 3: 验证**

```bash
tail -20 docs/adr/0008-barrier-semantics.md
```

Expected: 看到 "2026-06-17 追加" 章节

- [ ] **Step 4: 提交**

```bash
git add docs/adr/0008-barrier-semantics.md
git commit -m "docs(adr): append BarrierModule integration decision to ADR-0008"
```

---

## Phase 3: 验证 Gate（30 min）

### Task 3.1: 跑全量回归

**Files:** 无

- [ ] **Step 1: 编译检查**

```bash
. env.sh
cmake --build build 2>&1 | tail -20
```

Expected: 编译成功，无 error（warning 可接受）

- [ ] **Step 2: 跑 barrier 相关测试**

```bash
cd build && ctest -L "barrier" -V 2>&1 | tail -50
```

Expected: 全部 PASS

- [ ] **Step 3: 跑 exec_mask / simt_entry 测试（与 baseline 对比）**

```bash
cd build && ctest -L "exec_mask;simt_entry" -V 2>&1 | tail -30
```

Expected: 全部 PASS

- [ ] **Step 4: 跑 ./scripts/sanity.sh --quick**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
./scripts/sanity.sh --quick 2>&1 | tail -30
```

Expected: 全部 PASS

- [ ] **Step 5: 对比 baseline**

```bash
diff /tmp/baseline_$$.txt <(cd build && ctest -L "barrier;exec_mask;simt_entry" 2>&1)
```

Expected: 无差异（基线 = 当前）

---

### Task 3.2: 重提交评审

**Files:** 无

- [ ] **Step 1: 整理评审材料清单**

```bash
cd /workspace/project/PTX-EMU/../ptx-emu-barrier-fix
git log --oneline main..HEAD
```

Expected: 看到 6-8 个 commit（Phase 1 + 2 + ADR）

- [ ] **Step 2: 准备重评审 prompt**

发送给 Momus 续接 session `ses_128115580ffegArLPu5lzZKpi0`：

```markdown
Continue plan review. The 9 conditions (C1-C9) have been addressed:
- C1 (Blocker): release_warp_barrier OR logic — implemented + tested (release_warp_barrier preserves)
- C2 (Blocker): WarpBarrier::init preserve arrived_mask — implemented + tested (WarpBarrier::init preserves)
- C3 (Blocker): sm_context.cpp:200-260 removal — expanded in Task 6.2
- C4: design.md:178 bsync_state.h → bsync_state.cpp — fixed
- C5: proposal.md / design.md line numbers — corrected to L138-184
- C6: spec.md L19 barrive_at_warp_barrier → arrive — fixed
- C7: Task 3.0 TSan pre-flight — added
- C8: ADR-0008 §"2026-06-17 追加" — appended
- C9: design.md risk table sm_context.cpp:200-260 — added

Evidence:
- Test "release_warp_barrier preserves active_mask" PASSES (verifies BUG-POSTBARRIER-TWOHALVES fix)
- Test "WarpBarrier::init preserves arrived_mask" PASSES (verifies BUG-RECONVERGENCE-SIMPLEGEMM fix)
- ctest -L "barrier;exec_mask;simt_entry" all PASS

Worktree: ../ptx-emu-barrier-fix
Branch: fix/integrate-barrier-module-review
Commits: $(git log --oneline main..HEAD | wc -l) commits

Re-verify: do the 9 fixes actually address the blockers? Are there any NEW gaps introduced by the fix code itself?
```

- [ ] **Step 3: 等待 Momus 重评审 verdict**

Expected: APPROVE 或 APPROVE-WITH-CONDITIONS（剩余 minor 条件可后续处理）

---

## Self-Review Checklist

修复完成后对照原 spec 核查：

- [ ] `cta-barrier-module/spec.md` 4 个 ADDED Requirements 都有对应任务（Task 1.x, 2.x, 4.x）
- [ ] `warp-barrier-unification/spec.md` 5 个 ADDED Requirements 都有对应任务（Task 1.1, 1.2, 5.x）
- [ ] `barrier-handler-bugfix/spec.md` 4 个 ADDED Requirements 都有对应任务（Task 1.1, 1.2, 4.4）
- [ ] 3 个 Blocker（C1, C2, C3）有 Task 1.1, 1.2, 2.1 覆盖
- [ ] 6 个 Major（C4-C9）有 Task 2.1, 2.2, 2.3, 2.4, 2.5 覆盖
- [ ] 每步有明确验证命令 + Expected 输出
- [ ] 所有 commit 独立可回退
- [ ] 修复后的 `release_warp_barrier` 代码与 `src/ptxsim/core/AGENTS.md` 不变量一致
- [ ] 修复后的 `WarpBarrier::init` 首次 init 路径行为不变（向后兼容）

---

## 风险与回退

| 风险 | 概率 | 缓解 |
|---|---|---|
| Task 1.1 OR 逻辑实现错（误用 set_active_mask 而非保留 set_exec_mask） | 中 | Step 4-5 双重验证：单元测试 + 集成测试 |
| Task 1.2 re-init 保留 arrived_mask 破坏 first init 行为 | 低 | Step 1 测试明确区分 first init 与 re-init 两个 case |
| Phase 1 编译错误 | 中 | 每步独立 commit；Step 4 验证后再继续 |
| Phase 2 修改不完整（漏掉某个 spec 行） | 低 | Self-Review Checklist 强制对照 |
| 重评审仍有未发现 issue | 中 | Momus 续接 session 复用上下文，可快速定位 |

**回退策略：**
- Phase 1 失败：`git revert HEAD~N`（回退所有 Blocker 修复 commit），保留 Phase 2 文档修复
- Phase 2 失败：单独 revert 文档 commit
- Task 1.1 单独回退：`git revert <commit-hash>` 然后保留 Task 1.2

---

## 后续工作（不在本 plan 范围）

修复通过评审后，按原 design.md 7 阶段迁移计划继续：
- Phase 3：BarHandler 切换
- Phase 4：BarWarpSyncHandler 切换
- Phase 5：旧代码清理
- Phase 6：文档同步
- Phase 7：完整验证 + 发布

这 5 个阶段仍按原 tasks.md 执行，本 plan 只解决 Momus 评审的 9 个 conditions。
