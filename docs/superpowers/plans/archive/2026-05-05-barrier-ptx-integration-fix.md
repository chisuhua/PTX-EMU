# SIMT v2.0 Barrier + PTX 解析层集成修复计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 修复 `test_divergence_sync_standalone` 失败问题（expected -847479615, got 0），完成 SIMT v2.0 架构与 PTX 解析层的集成。

**Architecture:**

1. **Bug #1 (PC 覆盖)**: `PipelineHandler::ExecPipe:103` 在 `processOperation` 返回后无条件执行 `set_next_pc(saved_pc + 1)`，覆盖了 `BarWarpSyncHandler::processOperation` 为当前线程设置的 `reconvergence_pc`。修复：引入 flag 机制，允许 handler 声明"已设置 next_pc"，ExecPipe 跳过默认推进。

2. **Bug #2 (active_mask 破坏)**: `synchronize_barrier` 在所有 warp 上将 `exec_mask`/`active_mask` 重置为 `0xFFFFFFFF`。修复：仅在 CTA 级别同步，不修改 warp 级别的 exec_mask；Wbar 路径保持正确的 `arrived_mask`。

3. **Gap (BarrierInstr 无 reconvergence_pc)**: `BarrierInstr`（S_BAR）缺少 `reconvergence_pc` 字段，导致 CFG 分析无法为 CTA 级屏障设置收敛点。修复：为 `BarrierInstr` 添加可选字段，并在 CFG 分析中处理 S_BAR。

4. **Gap (CFG 跳过 S_BAR)**: `ptx_interpreter.cpp` 仅对 S_BRA 和 S_BAR_WARP_SYNC 做 CFG 后置支配分析。修复：扩展 CFG 分析覆盖 S_BAR。

5. **Gap (check_reconvergence 残留条目)**: 分叉执行后 `check_reconvergence` 仅在所有 group 处理完后调用一次，且在 barrier 完成后被 `synchronize_barrier` 的 `exec_mask` 重置覆盖。修复：确保 barrier 完成后 SIMT stack 被正确清理。

**Tech Stack:** C++20, PTX ISA, ANTLR4, Catch2

---

## Task 1: 修复 PipelineHandler::ExecPipe PC 覆盖 Bug

**Files:**
- Modify: `src/ptxsim/instruction_base.h` — 添加 `pc_overridden_` 成员和 getter/setter
- Modify: `src/ptxsim/instruction_base.cpp:75-104` — ExecPipe 检查 flag后再决定是否覆盖
- Modify: `src/ptxsim/instructions/barrier.cpp:95-182` — `BarWarpSyncHandler::processOperation` 设置 flag
- Create: `tests/test_barrier_pc_overwrite.cpp` — 回归测试（重现 bug，修复后通过）

- [ ] **Step 1: 写回归测试**（在 `tests/` 新建 `test_barrier_pc_overwrite.cpp`）

```cpp
// test_barrier_pc_overwrite.cpp
// 验证: bar.warp.sync 完成后，当前线程的 next_pc 被正确设置为 reconvergence_pc，
// 而非被 PipelineHandler::ExecPipe 覆盖回 barrier_pc+1
TEST_CASE("bar_warp_sync_pc_not_overwritten", "[barrier][pipeline]") {
    // 1. 构造 warp 和 thread
    WarpContext warp(ctx.get(), sm);
    ThreadContext t(&warp, 0, ...);
    warp.add_thread(&t);

    // 2. 初始化 Wbar（participation_mask = 0xFFFFFFFF, reconvergence_pc = 50）
    Wbar& wbar = warp.get_warp_state().wbars[0];
    wbar.init(0xFFFFFFFF, 50);

    // 3. 让 lane 0 到达屏障
    wbar.arrive(0);

    // 4. 让其他 lanes 也到达（触发 is_complete）
    for (int i = 1; i < 32; i++) wbar.arrive(i);
    REQUIRE(wbar.is_complete() == true);

    // 5. 通过 PipelineHandler::ExecPipe 执行 BarWarpSyncHandler
    StatementContext stmt;
    stmt.type = S_BAR_WARP_SYNC;
    BarWarpSyncInstr instr;
    instr.qualifiers = {Qualifier::Q_B32};
    instr.operands.push_back(OperandContext{ImmOperand{"0xFFFFFFFF"}});
    instr.operands.push_back(OperandContext{ImmOperand{"50"}});
    stmt.data = instr;

    PipelineHandler::ExecPipe(&t, stmt);

    // 6. 验证: next_pc == 50（reconvergence_pc），而非 barrier_pc+1
    CHECK(t.get_next_pc() == 50);
}
```

- [ ] **Step 2: 验证测试失败**

Run: `cd build && ctest -R test_barrier_pc_overwrite -V`
Expected: FAIL — `got barrier_pc+1, expected 50`

- [ ] **Step 3: 修改 instruction_base.h — 添加 flag**

在 `PipelineHandler` 类中添加：

```cpp
class PipelineHandler {
protected:
    bool pc_overridden_ = false;  // 新增
    void set_pc_overridden(bool v) { pc_overridden_ = v; }
public:
    bool is_pc_overridden() const { return pc_overridden_; }
    // ... existing members ...
};
```

- [ ] **Step 4: 修改 instruction_base.cpp — ExecPipe 检查 flag**

修改 `PipelineHandler::ExecPipe` 第 102-103 行：

```cpp
// 原来（bug）:
// context->set_next_pc(saved_pc + 1);

// 修改后:
if (!pc_overridden_) {
    context->set_next_pc(saved_pc + 1);
}
pc_overridden_ = false;  // 重置 flag
```

- [ ] **Step 5: 修改 barrier.cpp — BarWarpSyncHandler 设置 flag**

在 `BarWarpSyncHandler::processOperation` 第 159-160 行之后添加：

```cpp
// 原来：
context->force_set_pc(reconvergence_pc);
context->set_next_pc(reconvergence_pc);

// 修改后：
context->force_set_pc(reconvergence_pc);
context->set_next_pc(reconvergence_pc);
set_pc_overridden(true);  // 通知 ExecPipe 不要覆盖
```

并确保 `commitResults` 不覆盖（已在 barrier.cpp:86-92 正确实现）。

- [ ] **Step 6: 验证测试通过**

Run: `cd build && ctest -R test_barrier_pc_overwrite -V`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add src/ptxsim/instruction_base.h src/ptxsim/instruction_base.cpp \
        src/ptxsim/instructions/barrier.cpp tests/test_barrier_pc_overwrite.cpp
git commit -m "fix: prevent PipelineHandler::ExecPipe from overwriting bar.warp.sync reconvergence_pc"
```

---

## Task 2: 修复 synchronize_barrier 对 active_mask/exec_mask 的破坏

**Files:**
- Modify: `src/ptxsim/core/sm_context.cpp:469-503` — synchronize_barrier 不再盲目重置所有 warp 的 exec_mask
- Create: `tests/test_barrier_active_mask_preserved.cpp` — 验证屏障释放后分歧状态被保留

- [ ] **Step 1: 写测试**

```cpp
// tests/test_barrier_active_mask_preserved.cpp
// 验证: bar.sync (CTA 级) 完成后，warp 的 exec_mask 没有被破坏
TEST_CASE("cta_barrier_preserves_exec_mask", "[barrier][active_mask]") {
    WarpContext warp(ctx.get(), sm);
    ThreadContext t0(&warp, 0, ...);
    ThreadContext t1(&warp, 1, ...);
    warp.add_thread(&t0);
    warp.add_thread(&t1);

    // 设置分歧状态: exec_mask = 0x0000FFFF (只有 t0 active)
    warp.set_exec_mask(0x0000FFFF);

    // 调用 synchronize_barrier
    bool complete = sm->synchronize_barrier(0, &t0);
    REQUIRE(complete == true);

    // 验证: exec_mask 保持 0x0000FFFF，未被重置为 0xFFFFFFFF
    CHECK(warp.get_exec_mask() == 0x0000FFFF);
}
```

- [ ] **Step 2: 验证测试失败**

Run: `cd build && ctest -R test_barrier_active_mask_preserved -V`
Expected: FAIL — exec_mask 被重置为 0xFFFFFFFF

- [ ] **Step 3: 修改 sm_context.cpp — synchronize_barrier**

修改 `synchronize_barrier` 第 477-483 行（释放路径）：

```cpp
// 原来（bug）:
for (auto waiting_thread : barrier_waiting_threads[barId]) {
    WarpContext* warp_ctx = waiting_thread->get_warp_context();
    if (warp_ctx) {
        warp_ctx->set_exec_mask(0xFFFFFFFF);   // ← 破坏分歧
        warp_ctx->set_active_mask(0xFFFFFFFF);
    }
}

// 修改后:
for (auto waiting_thread : barrier_waiting_threads[barId]) {
    WarpContext* warp_ctx = waiting_thread->get_warp_context();
    if (warp_ctx) {
        // 保留当前 exec_mask，只在需要时才更新为 arrived_mask
        // warp_ctx->set_exec_mask(...) 不再调用
        // active_mask 由各 warp 自行通过 update_active_mask() 维护
    }
}
```

**注意**: `set_exec_mask(0xFFFFFFFF)` 的调用应被删除。`active_mask` 更新应通过 warp 的 `update_active_mask()` 统一管理，而非在 CTA barrier 路径中单独设置。

- [ ] **Step 4: 验证测试通过**

Run: `cd build && ctest -R test_barrier_active_mask_preserved -V`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add src/ptxsim/core/sm_context.cpp tests/test_barrier_active_mask_preserved.cpp
git commit -m "fix: preserve exec_mask during CTA barrier release"
```

---

## Task 3: 为 BarrierInstr 添加 reconvergence_pc 并扩展 CFG 分析

**Files:**
- Modify: `include/ptx_ir/statement_context.h:71-75` — 添加 `reconvergence_pc` 字段
- Modify: `src/cudart/ptx_interpreter.cpp:603-644` — CFG 分析覆盖 S_BAR
- Modify: `src/ptx_parser/ptx_visitor_barrier.cpp:76-91` — 原始 S_BAR handling 填充 `reconvergence_pc`

- [ ] **Step 1: 写测试**（CFG 分析为 S_BAR 计算 reconvergence_pc）

```cpp
// tests/test_cfg_barrier_reconvergence.cpp
TEST_CASE("cfg_barrier_reconvergence", "[cfg][barrier]") {
    // 构建简单 kernel: if -> bar.sync -> post-barrier sum
    // 验证 CFG 正确计算 bar.sync 的 reconvergence_pc
    KernelContext kernel;
    // ... 添加 statements（含 bar.sync）...
    PtxInterpreter interpreter;
    interpreter.build_cfg(&kernel);

    // 查找 S_BAR statement 的 reconvergence_pc
    auto* stmt = find_statement_by_type(&kernel, S_BAR);
    REQUIRE(stmt != nullptr);
    auto& barrier_instr = std::get<BarrierInstr>(stmt->data);

    // 验证 reconvergence_pc 被设置（> -1）
    CHECK(barrier_instr.reconvergence_pc >= 0);
}
```

- [ ] **Step 2: 修改 statement_context.h — 添加字段**

```cpp
// 原来:
struct BarrierInstr {
    std::vector<Qualifier> qualifiers;
    std::string type;
    std::optional<int> barId;
};

// 修改后:
struct BarrierInstr {
    std::vector<Qualifier> qualifiers;
    std::string type;
    std::optional<int> barId;
    int reconvergence_pc = -1;  // 新增
};
```

- [ ] **Step 3: 修改 ptx_interpreter.cpp — CFG 分析 S_BAR**

在 CFG 分析循环中（第 613-644 行附近），添加对 S_BAR 的处理：

```cpp
// 原来:
if (stmt->type == S_BRA) {
    // ... 处理 branch ...
} else if (stmt->type == S_BAR_WARP_SYNC) {
    // ... 处理 warp barrier ...
}

// 修改后:
if (stmt->type == S_BRA) {
    // ... 处理 branch ...
} else if (stmt->type == S_BAR_WARP_SYNC) {
    // ... 处理 warp barrier ...
} else if (stmt->type == S_BAR) {
    // S_BAR (CTA barrier): 从后置支配者链计算 reconvergence_pc
    BarrierInstr& barrier = std::get<BarrierInstr>(stmt->data);
    int rp = compute_post_dominator_reconvergence(stmt, kernel);
    barrier.reconvergence_pc = rp;
}
```

- [ ] **Step 4: 修改 ptx_visitor_barrier.cpp — 原始 S_BAR handling 填充字段**

在 `VISITOR_BARRIER` 宏的第 76-91 行（原始处理分支），确保 `reconvergence_pc` 被初始化：

```cpp
BarrierInstr instr;
// ...
instr.reconvergence_pc = -1;  // 默认值，CFG 分析后会更新
stmtCtx.data = instr;
```

- [ ] **Step 5: 验证测试通过**

Run: `cd build && ctest -R test_cfg_barrier_reconvergence -V`

- [ ] **Step 6: 提交**

```bash
git add include/ptx_ir/statement_context.h src/cudart/ptx_interpreter.cpp \
        src/ptx_parser/ptx_visitor_barrier.cpp tests/test_cfg_barrier_reconvergence.cpp
git commit -m "feat: add reconvergence_pc to BarrierInstr and CFG analysis for S_BAR"
```

---

## Task 4: 修复 check_reconvergence 残留条目问题

**Files:**
- Modify: `src/ptxsim/core/sm_context.cpp:228-231` — 分叉执行后立即检查并清理 SIMT stack
- Modify: `src/ptxsim/instructions/barrier.cpp:174-175` — Wbar reset 时同步清理 PC stack

- [ ] **Step 1: 分析残留条目的来源**

运行 `test_divergence_sync_standalone` 并观察 SIMT stack 状态：
- 在 `check_reconvergence()` 添加日志：每次 pop 时打印 stack depth 变化
- 验证 barrier 完成后 stack 是否真的为空

- [ ] **Step 2: 写测试**

```cpp
// tests/test_simt_stack_barrier_cleanup.cpp
TEST_CASE("simt_stack_cleaned_after_barrier", "[simt][barrier]") {
    WarpContext warp(ctx.get(), sm);
    // 设置一个 SIMT stack 条目（模拟分歧状态）
    warp.get_simt_stack().push({...});
    REQUIRE(warp.get_simt_stack().size() == 1);

    // 模拟 bar.warp.sync 完成
    warp.get_warp_state().wbars[0].init(0xFFFFFFFF, 50);
    // ... arrive all lanes, is_complete = true ...

    // 执行 check_reconvergence
    warp.check_reconvergence();

    // 验证: SIMT stack 为空，或 exec_mask 不再被旧条目影响
    CHECK(warp.get_simt_stack().empty() || warp.get_exec_mask() == 0xFFFFFFFF);
}
```

- [ ] **Step 3: 修改 sm_context.cpp — 分叉执行后多次检查**

修改 `sm_context.cpp:228-231`，在所有 divergent group 处理后循环调用 `check_reconvergence` 直到 stack 空或稳定：

```cpp
// 原来:
if (next_warp && !next_warp->get_simt_stack().empty()) {
    next_warp->check_reconvergence();
}

// 修改后:
if (next_warp && !next_warp->get_simt_stack().empty()) {
    // 循环清理直到 SIMT stack 为空或深度不再减少
    size_t prev_depth = 0;
    while (prev_depth != next_warp->get_simt_stack().depth() && prev_depth > 0) {
        prev_depth = next_warp->get_simt_stack().depth();
        next_warp->check_reconvergence();
    }
}
```

- [ ] **Step 4: 验证测试通过**

Run: `cd build && ctest -R test_simt_stack_barrier_cleanup -V`

- [ ] **Step 5: 提交**

```bash
git add src/ptxsim/core/sm_context.cpp tests/test_simt_stack_barrier_cleanup.cpp
git commit -m "fix: loop check_reconvergence to clear residual SIMT stack entries"
```

---

## Task 5: 添加运行时检查防止类似 Bug

**Files:**
- Create: `src/ptxsim/runtime_checks.h` — 运行时检查工具
- Modify: `src/ptxsim/instruction_base.cpp` — 在 ExecPipe 中添加 PC 合理性检查
- Modify: `src/ptxsim/core/warp_context.cpp` — 在 check_reconvergence 后验证 exec_mask 合理性

- [ ] **Step 1: 创建运行时检查工具**

```cpp
// src/ptxsim/runtime_checks.h
#ifndef PTXSIM_RUNTIME_CHECKS_H
#define PTXSIM_RUNTIME_CHECKS_H

#include <cstdint>
#include <cstdio>

namespace ptxsim {

// 运行时检查: next_pc 不应小于当前 PC（允许相等，表示停留在同一指令）
inline bool check_pc_advance_valid(int current_pc, int next_pc, const char* context) {
    if (next_pc < current_pc) {
        PTX_ERROR_EMU("RUNTIME_CHECK: PC went backward in %s: pc=%d, next_pc=%d",
                       context, current_pc, next_pc);
        return false;
    }
    return true;
}

// 运行时检查: exec_mask 中置位的 lanes 应对应活跃 threads
inline bool check_exec_mask_valid(uint32_t exec_mask, uint32_t active_mask) {
    uint32_t invalid = exec_mask & ~active_mask;
    if (invalid != 0) {
        PTX_ERROR_EMU("RUNTIME_CHECK: exec_mask has lanes not in active_mask: 0x%X",
                       invalid);
        return false;
    }
    return true;
}

// 运行时检查: SIMT stack 深度异常（> 16 层通常意味着有问题）
inline bool check_simt_stack_depth(size_t depth, const char* context) {
    if (depth > 16) {
        PTX_ERROR_EMU("RUNTIME_CHECK: SIMT stack depth=%zu exceeds limit in %s",
                       depth, context);
        return false;
    }
    return true;
}

}  // namespace ptxsim
#endif
```

- [ ] **Step 2: 在 PipelineHandler::ExecPipe 中添加 PC 前进检查**

在 `instruction_base.cpp` 的 `ExecPipe` 开头添加：

```cpp
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int current_pc = context->get_pc();
    int next_pc_before = context->get_next_pc();

    // ... 现有代码 ...

    // 最后：在设置 next_pc 前，检查其合理性
    if (!pc_overridden_) {
        int final_next = saved_pc + 1;
        check_pc_advance_valid(current_pc, final_next, "PipelineHandler::ExecPipe");
    }
}
```

- [ ] **Step 3: 在 check_reconvergence 后添加 exec_mask 合理性检查**

在 `warp_context.cpp` 的 `check_reconvergence` 末尾添加：

```cpp
void WarpContext::check_reconvergence() {
    // ... 现有代码 ...

    if (simt_stack.depth() < depth_before) {
        if (simt_stack.empty()) {
            warp_state.exec_mask = 0xFFFFFFFF;
        } else {
            warp_state.exec_mask = simt_stack.top().active_mask;
        }
        // 新增: 验证 exec_mask 与 active_mask 一致性
        check_exec_mask_valid(warp_state.exec_mask, get_active_mask());
    }
}
```

- [ ] **Step 4: 写测试验证运行时检查有效**

```cpp
// tests/test_runtime_checks.cpp
TEST_CASE("runtime_check_pc_backward_detected", "[runtime_checks]") {
    // 构造一个场景: 有人错误地将 next_pc 设置为 current_pc - 1
    ThreadContext t(...);
    t.force_set_pc(10);
    t.set_next_pc(9);  // 向后移动

    // 验证检查能捕获
    bool valid = check_pc_advance_valid(10, 9, "test");
    CHECK(valid == false);  // 应返回 false 并打印错误
}
```

- [ ] **Step 5: 验证运行时检查测试通过**

Run: `cd build && ctest -R test_runtime_checks -V`

- [ ] **Step 6: 提交**

```bash
git add src/ptxsim/runtime_checks.h src/ptxsim/instruction_base.cpp \
        src/ptxsim/core/warp_context.cpp tests/test_runtime_checks.cpp
git commit -m "feat: add runtime checks to catch PC/exec_mask corruption early"
```

---

## Task 6: 端到端验证 — test_divergence_sync_standalone 通过

**Files:**
- Run: `cd build && ./bin/test_divergence_sync_standalone`
- Run: `cd build && ctest -R test_divergence_sync -V`
- Run: `./scripts/sanity.sh --quick`

- [ ] **Step 1: 运行 test_divergence_sync_standalone**

```bash
cd build && ./bin/test_divergence_sync_standalone 2>&1 | tail -20
```

Expected (修复后): `PASS`

- [ ] **Step 2: 运行相关 ctest**

```bash
cd build && ctest -R "divergence|sync|barrier" --output-on-failure
```

Expected: 全部 PASS

- [ ] **Step 3: 运行 sanity.sh --quick**

```bash
./scripts/sanity.sh --quick
```

Expected: 全部 PASS

- [ ] **Step 4: 若全部通过，提交最终状态**

```bash
git add -A
git commit -m "test: verify test_divergence_sync_standalone passes after all fixes"
```

---

## 如何避免类似 Bug 的手段

### 1. 运行时检查（已实现）

| 检查项 | 文件 | 触发条件 |
|--------|------|----------|
| PC 后退检查 | `runtime_checks.h::check_pc_advance_valid` | `next_pc < current_pc` |
| exec_mask 合理性 | `runtime_checks.h::check_exec_mask_valid` | `exec_mask` 有 lane 不在 `active_mask` |
| SIMT stack 深度 | `runtime_checks.h::check_simt_stack_depth` | `depth > 16` |

**启用方式**: 在 Debug/RelWithDebInfo build 中自动开启（`#ifdef DEBUG` 或 `PTX_DEBUG` 宏）。

### 2. PipelineHandler Flag 机制（已实现）

任何 handler 如果需要覆盖 `PipelineHandler::ExecPipe` 的默认 `next_pc = saved_pc + 1` 行为，必须显式调用 `set_pc_overridden(true)`。这使得 PC 管理意图**显式化**而非隐式。

### 3. 单元测试必须经过完整 Dispatch 路径

**规则**: 所有 barrier 相关测试必须通过 `PipelineHandler::ExecPipe` 或 `BarrierHandler::ExecPipe` 执行，不得直接调用 `processOperation`。在 `tests/AGENTS.md` 中添加：

```
## Barrier Test Rule (强制)
- 测试 bar.sync / bar.warp.sync 必须通过 ExecPipe dispatch
- 禁止直接调用 processOperation()（绕过 PipelineHandler）
- 违反此规则的测试将被标记为无效
```

### 4. 回归测试覆盖

每个修复的 bug 都必须有对应的**回归测试**，保存在 `tests/` 中：
- `test_barrier_pc_overwrite.cpp` — Bug #1
- `test_barrier_active_mask_preserved.cpp` — Bug #2
- `test_cfg_barrier_reconvergence.cpp` — Gap #3
- `test_simt_stack_barrier_cleanup.cpp` — Gap #5

### 5. 架构守护

在 `src/ptxsim/core/AGENTS.md` 中将以下条款加入 **ANTI-PATTERNS**：

```
- DO NOT 在 barrier handler 中调用 set_next_pc 后不调用 set_pc_overridden(true)
- DO NOT 在 synchronize_barrier 中重置 exec_mask 为 0xFFFFFFFF（破坏分歧状态）
- DO NOT 绕过 ExecPipe 直接调用 processOperation（除非明确测试 handler 本身）
```

---

## 任务依赖关系

```
Task 1 (PC覆盖bug)          ← 独立，可最先做
Task 2 (active_mask破坏)   ← 依赖 Task 1 的 flag 机制
Task 3 (BarrierInstr扩展)  ← 独立，CFG 分析扩展
Task 4 (SIMT stack清理)     ← 可与 Task 1/2 并行
Task 5 (运行时检查)         ← 独立工具，可在任何阶段添加
Task 6 (E2E验证)           ← 依赖 Task 1-5 全部完成
```

**推荐执行顺序**: Task 1 → Task 5 → Task 2 → Task 3 → Task 4 → Task 6
