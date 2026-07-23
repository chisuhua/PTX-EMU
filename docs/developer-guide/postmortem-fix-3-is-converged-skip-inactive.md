# Postmortem: Fix 3 — `is_converged` 不应跳过暂时不活跃的 lane

> **Status (2026-06-25):** **FIXED & VERIFIED**.
>
> - `simpleCONV-{int,float,double}` all exit 0.
> - 单元测试 `unit_simt_stack_entry`, `unit_simt_integration`,
>   `unit_handle_branch`, `unit_handle_branch_two_level_divergence` 全部 PASS。
> - 全量 sanity 通过；`cute_rmsnorm` 是 baseline 失败，与本次修复无关。

## TL;DR

> **如果你只想知道结论**，看下面三处即可，本文其余内容（调试过程、诊断模板）是**经验沉淀**而非必读：
>
> 1. **ADR-0006**：[`docs/adr/ADR-0006-simt-stack-management.md`](../adr/0006-simt-stack-management.md) §"三个字段的角色分工（Fix 1 + Fix 3 后澄清）" — `active_mask` / `return_mask` / `is_active` 各用在哪
> 2. **KNOWN_ISSUES**：[`docs/developer-guide/KNOWN_ISSUES.md`](./KNOWN_ISSUES.md) §"B4.2 — simpleCONV-{int,float,double} hang at SIMT stack reconvergence point (FIXED 2026-06-25)" — 症状、链路、验证命令
> 3. **代码位置**：`src/ptxsim/core/simt_stack.cpp:7-25` `is_converged`（核心修复）

## Goal

修复 `simpleCONV-{int,float,double}` 测试超时挂死的根因：
`SIMTStackEntry::is_converged()` 错误地把 `!is_active` 的 lane 视为"已收敛"，
导致 lane 因内存停顿暂时失活时被过早弹出 SIMT 栈条目，从而在真正的汇聚点失锁。

## Background / Root Cause

### 关键背景

- `simpleCONV.cu` 是一个 3-block × 4-warp 的卷积 kernel，每个 warp 在循环
  `@%p4 bra $L__BB0_3`（PC=44，回跳到 PC=37，`$L__BB0_4` 在 PC=45）上发生分歧
- 3 个简单变体的 E2E 测试（`simpleCONV-int/float/double`）**在 baseline 即挂死**，
  与本仓库 `KNOWN_ISSUES.md` 列出的 `simpleCONV` 系列挂死项一致
- 但根因与 `cute_rmsnorm` 的 dispatch-gate bug（见 [postmortem-fix-1](./postmortem-fix-1-gate-active-vs-return-mask.md)）**完全不同**——
  那是门控阻塞了不该阻塞的 lane；这里是收敛判定跳过了不该跳过的 lane

### 根因（已定位，2026-06-25）

`src/ptxsim/core/simt_stack.cpp:7-25` 的 `SIMTStackEntry::is_converged()`：

**Bug 行为**（修复前）：
```cpp
if (threads[i].is_exited || !threads[i].is_active) {
    continue;   // ← BUG：把内存停顿的 lane 也当作"已收敛"跳过
}
```

**触发链路**（以 simpleCONV-int 的 warp 3 lane 0 为例）：

1. **PC=44 分歧**：`@%p4 bra $L__BB0_3`，lane 0 取分支（→PC=37），lanes 1-31
   fall-through（→PC=45）。`handle_branch` 压栈：返回掩码 `0xFFFFFFFF`、
   `active_mask=0x00000001`、`reconvergence_pc=45`
2. **lane 0 在循环体内执行 `ld.global.u32`**（PC=38）。`PipelineHandler::ExecPipe`
   进入流水线，但因数据未就绪触发 **pipeline retry**——
   `update_active_mask()` 在 `execute_warp_instruction` 末尾被调用时，
   把 lane 0 暂时标记为 `is_active=false`（因为 `is_blocked=true` 之类的瞬态）
3. **调度器下一轮走到栈顶汇聚点 PC=45**：`check_and_block_at_reconvergence_point`
   工作正常（门控阻塞 lanes 1-31），然后 `check_reconvergence()` 调用 `is_converged`
4. **`is_converged` 检查 lanes 0-31**：
   - lane 0：`active_mask` 命中，`!is_active=true` → **`continue` 跳过**
   - lanes 1-31：`active_mask` 未命中，根本不检查
   - 循环结束无人返回 false → 返回 true（**假阳性**）
5. **`SIMTStack::check_reconvergence` 弹出栈条目**，并把 `is_blocked=false`
   设给 lanes 1-31（因为它们 `pc == reconv_pc`）
6. **lane 0 恢复活跃后**（load 完成）回到 PC=38，但栈已经空了、gate 不再阻塞——
   继续执行到 PC=45（被 gate "空" 时放过），最终停在 PC=46，而 lanes 1-31 永远
   停在 PC=45 → **死锁**

### 复现路径

```
[POST-LOOP] warp3 cycle=904 lane0_pc=37 lane1_pc=45 lane31_pc=45   ← 分歧正确
[RECONV-CHECK] warp3 cycle=916 depth=1 will_pop=0                   ← 正确，lane 0 还在 PC=37
[RECONV-CHECK] warp3 cycle=928 depth=1 will_pop=1 l0(pc=39 act=0)  ← BUG 触发点
[POP] warp3 cycle=928 ...                                           ← 栈被错误弹出
[GATE-45] warp=3 EMPTY                                              ← 栈已空，门控失效
```

### 关键诊断笔记

- 调试过程中最初**严重误导**的方向：以为是栈被多次 push（看到三次 `[BRANCH]`
  在不同 cycle）。后来通过 `[SCHED]` 打印 `WarpContext` 指针发现：
  同一 `warp_id==3` 在 SM 0 上有 **3 个不同的对象**（3 个 CTA 各有一个
  warp 3），它们的指针 `0x...860 / 0x...650 / 0x...4e0` 完全不同
- 用 `this=%p` 区分 CTA 是定位"为什么 PC 在 904→908 之间看似被重置"的
  关键技巧——那些 PC 重置其实是不同 CTA 的 warp 3 在独立调度
- `update_active_mask()` **每周期**对所有 warp 调用一次，所以 lane 0 的
  `is_active=false` 会在 `is_converged` 看到时被多次观察到

### 与 Fix 1 的关系

| 维度 | Fix 1（dispatch gate） | Fix 3（is_converged） |
|------|----------------------|---------------------|
| 触发症状 | lane 被不该阻塞的位置阻塞 | lane 被不该跳过的位置跳过 |
| 影响字段 | `active_mask` vs `return_mask` 在**门控**里的语义 | `is_exited` vs `!is_active` 在**收敛判定**里的语义 |
| 修复方向 | gate 必须用 `return_mask`（阻塞全部到达 reconv_pc 的 lane，包括 fall-through） | `is_converged` 只跳 `is_exited`，**不跳 `!is_active`** |
| 共同点 | 都是 SIMT stack 三个字段（`active_mask`、`return_mask`、`is_active`）的语义混淆 | 同左 |

## Affected Files

| 文件 | 修改内容 |
|------|---------|
| `src/ptxsim/core/simt_stack.cpp` | `SIMTStackEntry::is_converged()`：去掉 `!is_active` 跳过条件 |
| `src/ptxsim/core/warp_context.cpp` | `check_and_block_at_reconvergence_point()`：用 `return_mask`（而非 `active_mask`）阻塞 lane——**这条修改其实是 Fix 1 残留的回归**，本次重新确认其正确性 |
| `src/ptxsim/core/warp_context.cpp` | `check_reconvergence()` 弹出后用 `return_mask` 恢复 `exec_mask` |
| `tests/unit/simt/test_simt_stack_entry.cpp` | B2 测试：明确"active_mask=0 的条目空虚收敛"语义 |
| `tests/unit/simt/test_simt_integration.cpp` | I2 测试：`exec_mask` 在嵌套弹出后取新栈顶的 `return_mask` |
| `tests/unit/simt/test_handle_branch_two_level_divergence.cpp` | 不变（已正确） |

**关键不变式**（三处代码必须严格区分三个字段的角色）：

| 字段 | 在哪个函数里用 | 为什么 |
|------|---------------|--------|
| `active_mask` | `is_converged()` 内的循环 | 收敛判定只关心"走了分支的 lane 是否到齐" |
| `return_mask` | gate `check_and_block_at_reconvergence_point()` | 阻塞所有到达 reconv_pc 的 lane，包括 fall-through 分支 |
| `return_mask` | `check_reconvergence()` 弹出后恢复 `exec_mask` | 弹出后整个分歧组都应可执行，不只是 active 子集 |

混淆 `active_mask` 与 `return_mask` 都会引入回归——
参见 KNOWN_ISSUES § "common mistakes"。

## Test Setup

### 已有 ctest 目标（本次回归验证）

```bash
# 修复前（验证 Fix 3 前 baseline 行为）
cmake --build build --target simpleCONV-int
timeout 5 ./build/bin/simpleCONV-int     # ← 必须 hang/timeout
```

### 关键 helper

- `ptxsim::testing::step_warp()` — 集成测试用的调度模拟
- `WarpContext::get_simt_stack().push(entry)` — 单元测试可手动注入栈条目
- `WarpContext::set_thread_pc(i, pc)` — 单元测试直接设置 lane PC

### 本次调试新增的临时调试手段（提交时已全部清除）

- `fprintf(stderr, "[BRANCH] warp3 ...")` — warp 3 走分支时打印栈状态
- `fprintf(stderr, "[POST-LOOP] warp3 lane0/1/31_pc")` — 循环结束后 PC
- `fprintf(stderr, "[POST-BRANCH] ...")` — handle_branch 后 PC
- `fprintf(stderr, "[GATE-45] warp=%d")` — 门控进入/退出
- `fprintf(stderr, "[RECONV-CHECK] warp3 will_pop=... l0(pc=%d act=%d exit=%d)")` — `is_converged` 调用前完整快照
- `fprintf(stderr, "[POP] warp3 reconv=%d")` — 弹出事件
- `fprintf(stderr, "[SCHED] warp3 this=%p groups=%zu pc=%d:{...}")` — **最关键的诊断**：在 `SMContext::exe_once` 的 `get_lanes_by_pc()` 后立刻打印，揭示了 3 个不同 CTA 的 warp 3 各自独立调度

**经验教训**：临时调试打印应当
1. 用明显的 tag（如 `[GATE-45]`）便于 grep
2. 在每条 fprintf 前**注明打印位置语义**（是入口？循环内？循环后？）
3. **提交前必须清除**——`grep fprintf` 应只命中注释

## Constraints

### ⚠️ 严格约束

1. **`is_converged` 只跳 `is_exited`，绝对不要跳 `!is_active`**：
   内存停顿、barrier 等待等瞬态失活都不能算"已收敛"
2. **三个字段角色不能混淆**（见上表）——
   `active_mask` 用于收敛判定，`return_mask` 用于门控阻塞和 exec_mask 恢复
3. **不要修改 `update_active_mask` 的双向同步语义**——
   它是 self-heal 机制（per `src/ptxsim/core/AGENTS.md` "T2-1"），
   修改它会破坏 lane-activity 单源真相

### ⚠️ 注意事项

- **同栈多 CTA 调度**：SM 0 上同一 `warp_id` 可对应多个 CTA 的多个
  `WarpContext` 对象；调试时必须用 `this=%p` 区分，否则会误以为状态被重置
- **`update_active_mask()` 每周期对所有 warp 调用**：
  一个 lane 在某周期被标记 `is_active=false`，下一周期可能就恢复——
  这是正常的流水线重试，不要当作状态损坏
- **`while (next_warp->check_reconvergence())`** 只在栈顶条目满足时 pop；
  一次 pop 后下次循环再判。所以 `depth_before` 必须**每次**重新读
- **bug 复现需要精确的随机种子**：`simpleCONV-int` 用 `srand(time(NULL))`，
  不同运行可能选择不同分支路径；要稳定复现应加 `PTX_EMU_CONV_TEST_MODE`

## Step-by-Step Approach (已完成)

### Step 1: 复现 baseline 挂死

```bash
timeout 5 ./build/bin/simpleCONV-int
# 预期：超时（exit 124）
```

### Step 2: 定位到 is_converged

通过 `[SCHED]` + `[RECONV-CHECK]` + `[POST-BRANCH]` 三组调试打印，
锁定：lane 0 的 `is_active` 因 `ld.global` 流水线重试变为 `false`，
`is_converged` 错误地跳过它，弹出栈条目，lane 1-31 被错误解锁。

### Step 3: 应用修复（is_converged 单点）

```cpp
bool SIMTStackEntry::is_converged(const std::array<ThreadState, 32>& threads) const {
    for (size_t i = 0; i < 32; i++) {
        if (active_mask & (1u << i)) {
            if (threads[i].is_exited) {     // ← 仅跳 exit
                continue;
            }
            if ((int)threads[i].pc != reconvergence_pc) {
                return false;
            }
        }
    }
    return true;
}
```

注释**必须保留**——它是后续重构时防止再回归的关键。

### Step 4: 调整两个测试（语义澄清）

- `tests/unit/simt/test_simt_stack_entry.cpp` B2：
  从"active_mask=0 时不应 pop"改为"active_mask=0x0000FFFF 且全到齐 → pop"
- `tests/unit/simt/test_simt_integration.cpp` I2：
  `exec_mask` 检查从新栈顶 `return_mask` 取（先前是 `active_mask`，是错的）

### Step 5: 验证 simpleCONV 通过

```bash
timeout 60 ./build/bin/simpleCONV-int    # ← exit 0
timeout 60 ./build/bin/simpleCONV-float  # ← exit 0
timeout 60 ./build/bin/simpleCONV-double # ← exit 0
```

### Step 6: 全量 sanity 验证

```bash
./scripts/sanity.sh --full --verbose
# 70 PASS, 0 FAIL（cute_rmsnorm 是 baseline 已有失败，与本 fix 无关）
```

## Reference Materials

### 已读过的关键文件

- `src/ptxsim/core/simt_stack.cpp:7-25` — **`is_converged`（已修改）**
- `src/ptxsim/core/warp_context.cpp:108-149` — `check_reconvergence`
- `src/ptxsim/core/warp_context.cpp:160-204` — `check_and_block_at_reconvergence_point`（Fix 1 修复）
- `src/ptxsim/core/sm_context.cpp:230-265` — 调度器 `lanes_by_pc` + `while check_reconvergence`
- `include/ptxsim/simt_stack.h:12-21` — `SIMTStackEntry` 结构
- `src/ptxsim/core/AGENTS.md` — **DUAL STATE MECHANISM / T2-1 必读**
- `docs/adr/ADR-0006-simt-stack-management.md` — SIMT stack 设计决策
- `docs/developer-guide/KNOWN_ISSUES.md §"simpleCONV hang"` — bug 上下文
- [postmortem-fix-1-gate-active-vs-return-mask.md](./postmortem-fix-1-gate-active-vs-return-mask.md) — 同类问题参考（门控侧）
- [open-fix-2-sbar-deadlock.md](./open-fix-2-sbar-deadlock.md) — `S_BAR` 死锁（仍 OPEN，与本次修复无关）

### 关键发现笔记

- **`this=%p` 是同栈多 CTA 调度的救命稻草**——之前我误以为 PC 在
  cycle 904→908 之间被重置，实际是 3 个不同 CTA 的 warp 3 各自独立调度
- **`update_active_mask` 在每周期**对**所有** warp 调用——lane 0 因
  ld.global 流水线重试进入瞬态 `is_active=false`，**这是正常行为**
- **`is_converged` 用 `active_mask` 而不是 `return_mask`** 是正确的——
  收敛判定只关心"走了分支的 lane 是否到齐"，fall-through 的 lane
  本来就不需要"收敛"（它们没分支出去）
- **gate 用 `return_mask` 而不是 `active_mask`** 也是正确的——
  阻塞到达 reconv_pc 的所有 lane，**包括 fall-through**，
  否则 fall-through lane 会越过 reconv_pc 跑掉
- 这两个"正确但相反"的选择是 `is_converged` 与 gate 的本质区别，
  务必区分

### 未来类似问题的诊断模板

1. **挂死 + 多个 lane 卡在不同 PC + 栈深度异常** → 多半是 `is_converged`
   错误返回 true（跳过 lane / 字段混淆）
2. **挂死 + lane 被卡住不动** → 多半是 gate 错误阻塞（Fix 1 风格）
3. **挂死 + 栈深度不变但 PC 在跳转** → 多半是 `handle_branch` 的
   `is_divergent` 判定或 fall-through PC 计算错误
4. **第一步**：在 `SMContext::exe_once` 的 `get_lanes_by_pc()` 后立刻
   打印 `this=%p` + lanes 分组 + `simt_stack().depth()`，确认是同栈
   还是多栈

## 任务完成判定

修复成功的标志：
- [x] simpleCONV-{int,float,double} 全部 exit 0
- [x] unit_simt_stack_entry、unit_simt_integration、unit_handle_branch
  全部 PASS
- [x] 全量 sanity 70 PASS, 0 新 FAIL（cute_rmsnorm 仍 FAIL，与本 fix 无关）
- [x] 调试打印全部清除（`grep fprintf` 仅命中注释和允许的 PTX_DEBUG_*）
- [x] 代码注释解释了"为什么只跳 is_exited 而不跳 !is_active"
- [x] 三个字段（active_mask / return_mask / is_active）的角色清晰化