# PTX-EMU 开发经验沉淀（Lessons Learned）

> **来源**: `integrate-barrier-module-cta-warp` change 实施全过程（2026-06-18, commit `f033312`）+ `migrate-bar-warp-sync-to-barrier-module` 落地全过程（2026-07-03, commits `0e311566`+`f5640042`+`0bab6487`）
> **目的**: 把本次工作的关键经验教训系统化，供后续重构/迁移任务参考
> **互补文档**: [`debugging-strategy.md`](debugging-strategy.md)（问题分类与快速验证）、[`.opencode/skills/ptx-barrier-mechanism/`](../skills/)（领域知识）、[`.opencode/skills/ptx-lessons-learned/SKILL.md`](../skills/ptx-lessons-learned/SKILL.md)（快速决策树 + Checklists）

---

## 1. 跨模块间接状态翻译：迁移函数时不能只对"主逻辑"做 Diff

### 现象
`BarHandler::executeBarrier` 在 Phase 4 迁移中漏掉了一行 `context->set_state(BAR_SYNC)`。原 `SMContext::synchronize_barrier`（`sm_context.cpp:703`）看似"次要"的状态设置，实际上是 `ThreadContext::sync_to_warp_state()`（`thread_context.cpp:794-796`）的输入，被翻译为 `warp_state.threads[i].is_blocked = true`。没有这一行，调度器（`warp_context.cpp:267`）不识别 BAR_SYNC 状态，线程在 barrier 指令处死循环。

### 教训
- **跨函数的"间接翻译"是最隐蔽的 bug 来源**。`set_state(BAR_SYNC)` 在调用点看似冗余（`next_pc` 也设了），但它是另一模块的 API 契约。
- **迁移函数时必须做到"行级 Diff"**：把 baseline 函数和新实现并列，逐行检查有没有丢东西，而不仅仅检查"主要逻辑路径"。
- **强制检查清单**（迁移任何带 `ThreadContext` / `WarpState` 的函数时）：
  1. 列出 baseline 函数中所有的 `thread->set_*()` 调用
  2. 确认新实现保留了每一项，**即使是看似多余的**
  3. 对每一项 `set_*`，grep 其值的下游消费者（`grep "<state>\|is_<state>" src/...`）

### 检查工具
```bash
# 找出所有 "在某处设置 BAR_SYNC" 的位置
grep -rn "set_state(BAR_SYNC)" src/
# 找出所有 "读取 BAR_SYNC" 的位置
grep -rn "state == BAR_SYNC\|get_state() == BAR_SYNC\|is_at_barrier" src/
# 对照：调用点 vs 消费点数量应该一致
```

### 真实案例
- **bug 表现**: `integration_barrier_full_lifecycle` 测试第 336 行 `CHECK(!any_unblocked)` 失败 — 部分线程 `is_blocked == false`
- **修复**: 添加 `context->set_state(BAR_SYNC);` + 解释性注释
- **回归测试**: 已加入 `integration_barrier_full_lifecycle`（单元测试无法发现，必须用集成测试驱动调度器才能暴露）

---

## 2. 递归锁死锁：互斥量需要"使用同一锁的完整代码路径"集中审计

### 现象
`CTABarrier::arrive()` 持 `mutex_` 后调用 `is_complete()`，后者再次 `lock_guard(mutex_)`。`std::mutex` 不可重入 → 死锁。

```cpp
// 错误：持锁状态下调用同锁的其他 public 方法
bool CTABarrier::arrive(ThreadContext* thread) {
    std::lock_guard<std::mutex> lock(mutex_);   // 持锁
    arrived_threads_.insert(thread);
    if (is_complete()) { ... }                   // is_complete() 又 lock 同一 mutex
}

bool CTABarrier::is_complete() const {
    std::lock_guard<std::mutex> lock(mutex_);   // ← 死锁点
    return arrived_threads_.size() >= expected_threads_;
}
```

### 教训
- **可重入性问题通常被单元测试忽略**。类型一单元测试如果只调用一个方法，问题不会暴露；只有真实并发或循环调用才暴露。
- **死锁信号模式**："测试在断言 N+1 处挂起，N+1 通常是循环的第一次调用"。本次是第 7 个断言（`for (int i = 0; i < BLOCK_DIM; i++)` 的 i=1）。
- **诊断路径**：
  1. 看到 "测试在某点后挂起" → 检查该点附近的所有互斥量
  2. `grep -n "lock\|lock_guard\|unique_lock" <file>` 列出所有锁点
  3. 找出"同一 mutex 上的多个 lock_guard" → 这就是嫌疑点
  4. 检查是否存在"持锁方法调用其他持锁方法"的链

### 检查工具
```bash
# 列出某文件所有互斥量使用
grep -n "mutex_\|lock_guard\|unique_lock" src/ptxsim/barrier/cta_barrier.cpp
# 检查是否有"嵌套锁"模式
grep -B2 "lock_guard" src/ | grep -A1 "lock_guard"
```

### 真实案例
- **bug 表现**: `integration_barrier_module` 测试 BUG-HANDLER-PC-ADVANCE 在 6 个断言后挂起 10s+ → SIGTERM
- **修复**: 在 `arrive()` 内联完整性比较，避免调用 `is_complete()`，加解释性注释警告未来维护者

---

## 3. "可重入安全" 模式：public 方法不应该再锁

### 设计原则
对于已加锁的方法内部调用的辅助方法，应提供"内部无锁版本"或"明确不锁"的语义：

```cpp
// 方案 A：拆分两个版本（推荐）
class CTABarrier {
    bool is_complete_unsafe() const;  // 假设已持锁
    bool is_complete() const {         // public 持锁
        std::lock_guard lock(mutex_);
        return is_complete_unsafe();
    }
};

// 方案 B：内联（最小改动）
bool arrive() {
    std::lock_guard lock(mutex_);
    arrived_threads_.insert(thread);
    bool complete = arrived_threads_.size() >= expected_threads_;  // 直接比较
    if (complete) { ... }
    return false;
}
```

### 决策标准
- **方案 A**：API 表面更清晰，长期可维护性高
- **方案 B**：改动最小，但需要在注释中说明"为什么内联"

---

## 4. "复杂迁移"必须分 Phase commit，每个 Phase 独立可回退

### 现象
本次 `integrate-barrier-module-cta-warp` change 包含 5 个 Phase：
- Phase 1-3：审计 / 扩展 BarrierModule / 单元测试 ✅ 单独 commit 通过
- Phase 4：迁移 `BarHandler::executeBarrier`（bar.sync 路径）✅ 单独 commit 通过
- Phase 5：迁移 `BarWarpSyncHandler::processOperation`（bar.warp.sync 路径）❌ 引入 6 个分歧测试回归

### 教训
- **每个 Phase 必须独立 commit、独立验证**。一旦 Phase N 失败，只需要 revert 那个 commit，而不会污染其他 Phase。
- **Phase 5 的失败教训**：bar.warp.sync 涉及分歧/汇聚，单个调度器路径通过不代表全部分歧场景通过。**涉及控制流/分歧/同步的迁移，必须有"分歧场景"的测试覆盖**。
- **判定"Phase 完成"的标准**：
  1. ✅ 所有原通过的测试仍然通过
  2. ✅ 新增的测试通过
  3. ✅ 没有测试"意外"变快/变慢（可能掩盖 bug）
  4. ❌ 当任何已有测试回归时，**立即 revert 该 Phase**，不要把修复混在后续 commit

### 实际执行
- Phase 5 失败时 → `git revert --no-commit 36dbb9a` 立即回退
- 把 BAR_SYNC 修复 + 死锁修复 + Phase 5 revert 打包成新 commit `f033312`
- **关键**：commit message 详细列出 3 个独立 fix，未来 bisect 可以单独分析每个

---

## 5. 基线 worktree：任何重大重构前的"最低成本保险"

### 现象
本次工作开始时创建了 `.worktrees/baseline-check`（commit `00f698f`），让我能区分：
- **预先存在的失败**（基线也失败，不在本次范围）：`unit_simt_stack_stale_entry_blocks_lane0`、`integration_cute_rmsnorm_bar_sync_pattern`
- **本次引入的回归**（基线通过，本次失败）：**0 个** ✅

### 教训
- **基线 worktree 是免费的事实校验器**。每次大重构前花 1 分钟建立，可以节省数小时的"这个失败是基线的还是我的"争论。
- **建立时机**：在第一个有风险的 commit 之前建立（不是工作开始时，是"进入危险区"时）。
- **保留时机**：直到 change 完整合并 + 验证通过。**不要中途清理**（曾因为 `git stash` 影响而重新执行了部分测试）。

### 实测验证：worktree 里 nvcc + CUDA 编译完全正常

**担心**: worktree 里 nvcc 编译可能出问题（路径不一致、CUDA toolkit 找不到）
**实测**（在 `.worktrees/fix-pre-p0-baseline` 中跑通）:
- `which nvcc` → `/workspace/project/opt/cuda/bin/nvcc` ✓
- `integration_barrier_module` → PASS 0.29s
- `e2e_barrier_warp_sync`（含 nvcc 编译 CUDA kernel）→ PASS 0.53s

**原因**: `env.sh` 设计是路径无关的（`NVCC_PATH=$(which nvcc)`），不依赖固定路径。nvcc 不依赖 git worktree 的文件系统结构。

### 标准操作（含实测时间预算）
```bash
# === Step 1: 建立 baseline（5 分钟）===
git worktree add .worktrees/baseline-check <baseline-commit>
cd .worktrees/baseline-check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release  # cmake configure (~30s)
# 关键：必须全量编译一次，预算 15-20 分钟
cmake --build build -j$(nproc)

# === Step 2: 验证 baseline 通过（5 分钟）===
cd build
ctest -L "barrier" --output-on-failure -j$(nproc)
# 记录 baseline 输出作为对照

# === Step 3: 对比 main（增量编译快）===
cd /workspace/project/PTX-EMU
cmake --build build -j$(nproc)
cd build && ctest -L "barrier" --output-on-failure -j$(nproc)
# diff 输出，确认没有新增 FAIL

# === Step 4: 清理（验证通过后）===
git worktree remove .worktrees/baseline-check
```

### 时间预算
- baseline 首次 build：15-20 分钟（必须！否则部分 target 找不到）
- 后续增量 build：几秒-几十秒
- 单个测试 target：5-30 秒

### ⚠️ 三个常见陷阱

1. **baseline commit 选择错误**：baseline 必须在 main **之前**（不是任意旧 commit），且必须包含 main 上有的所有测试。如果 baseline 比 main 旧太多、缺测试，对比会失败。
2. **baseline 首次 build 不完整**：如果只编译了部分 target，ctest 会因"找不到可执行文件"误报失败（实测遇到）。解决方案：第一次必须全量编译。
3. **worktree 中 .git 共享**：worktree 之间共享 `.git` 目录，但 build 目录是独立的。同一时刻不能在两个 worktree 中切到同一分支。

### 推荐 baseline 选择策略
```bash
# 推荐：选择上一个稳定 release / merge commit
git log --oneline --tags -n 10
# 选 "Merge branch 'xxx'" 或 "release: vX.Y" 类型的 commit

# 不推荐：选任意旧的 fix commit
# 不推荐：选 main HEAD（那不是 baseline）
```

---

## 6. 类型一 vs 类型二 vs 类型三测试的"发现能力"差异

### 本次观察

| 测试类型 | 能发现的问题 | 不能发现的问题 |
|----------|------------|--------------|
| **类型一（单元）** | 递归锁死锁（直接调 API）、数据竞争、状态机错误 | 跨模块状态翻译缺失（因为不驱动调度器） |
| **类型二（集成）** | 跨模块状态翻译（需要驱动调度器）、PC 推进、SIMT stack | E2E 性能/内存布局问题 |
| **类型三（E2E）** | 完整 kernel 语义、用户视角行为 | 内部数据结构 bug（被 self-heal 掩盖） |

### 教训
- **状态机/API 重构的 bug 优先靠类型一测试发现**。例如递归锁死锁，只有直接调 `arrive()` 才会暴露。
- **跨模块状态翻译的 bug 必须靠类型二/三测试发现**。`set_state(BAR_SYNC)` 的缺失只在调度器 + executor + sync_to_warp_state 全链路协同时才暴露。
- **判断"测试类型选择"的方法**：问"我修改的代码是否经过 N 个中间层才到测试断言？" — 经过的层数越多，越需要类型三。

### 实际映射
| 修复 | 需要的测试类型 | 实际使用 |
|------|--------------|---------|
| 递归锁死锁 | 类型一（直接 API） | ✅ `test_barrier_module_integrated.cpp` |
| BAR_SYNC 缺失 | 类型二/三（驱动调度器） | ✅ `test_barrier_full_lifecycle.cpp` |
| BAR_SYNC 缺失 — 反例 | ❌ 类型一（不驱动调度器，会通过） | ⚠️ 需要注释说明为什么类型一测不到 |

---

## 7. `git revert` / `git stash` / `git worktree` 的状态机陷阱

### 现象
- `git revert 36dbb9a --no-commit` 创建了 staged but uncommitted 状态
- `git stash` 保存了 working tree + index
- `git stash pop` 恢复后，**revert 状态从 staged 变成 unstaged**（因为 `git stash pop` 重新应用的是 working tree 内容）
- 如果此时直接 commit，会把 revert + 当前修改混在一起

### 教训
- **复杂状态下，git 操作后必须立即 `git status` 验证**。
- **推荐的"安全回退"流程**：
  ```bash
  # 1. 验证当前状态
  git status
  # 2. 执行操作
  git revert --no-commit <commit>
  # 3. 立即验证
  git status   # 应该看到 "You are currently reverting"
  # 4. 决定：继续回退 → git revert --continue（写 message），或放弃 → git revert --abort
  # 5. 验证：git log --oneline -3
  ```
- **stash/pop 会改变 staged/unstaged 状态**。如果用 stash 保存一个 revert 状态，pop 之后需要重新 stage。

### 避免
- ❌ 不要在 revert 状态未 commit 时进行 `git stash` / `git rebase` / `git pull`
- ❌ 不要假设 `git status` 输出在多次操作后保持不变

---

## 8. 测试超时检测："per-test timeout" 优于"套件整体 timeout"

### 现象
- 第一次跑 `ctest -L barrier` 整体 120s 超时，无法定位是哪个测试
- 切换到 `ctest -R "specific_test" -V` + 10s per-test timeout → 3 秒内定位到 `integration_barrier_module` 的 BUG-HANDLER-PC-ADVANCE 死锁

### 教训
- **遇到测试套件整体超时时，立即切到 single-test 模式**。不要试图扩大搜索范围或增加 timeout。
- **标准操作**：
  ```bash
  # 步骤 1：列出该标签下所有测试
  ctest -L <label> -N
  # 步骤 2：对可疑测试用 -V + per-test timeout
  timeout 10 ctest -R <specific_test> -V
  # 步骤 3：定位后，单独运行该测试的 binary
  timeout 10 ./bin/tests/<test_binary>
  ```
- **核心 insight**：ctest 的并行/串行调度对超时表现差异很大，**永远是 single-test + timeout 优先**。

---

## 9. Commit 原子性 vs 可 bisect 性的取舍

### 本次 commit `f033312` 的组成
1. **Revert Phase 5**（移除 BarWarpSyncHandler 迁移）
2. **修复 BAR_SYNC 缺失**（BarHandler else 分支）
3. **修复 CTABarrier 死锁**（arrive() 内联完整性检查）

### 取舍分析
| 方案 | 优点 | 缺点 |
|------|------|------|
| **三个独立 commit**（本次未采用） | 每个修复可独立 revert / cherry-pick | 三个 commit 中间状态可能都不通过测试，CI 会红 |
| **一个原子 commit**（本次采用） | 永远处于"要么全好要么全坏" | 未来 bisect 看到 1 个 commit 包含 3 个 fix |

### 决策
- **采用单 commit** + 在 commit message 中详细列出 3 个 fix 及其各自的测试影响
- 理由：3 个 fix 互相依赖（revert Phase 5 是先决条件；BAR_SYNC 修复是 Phase 4 的补全；死锁是 BarrierModule 的预存 bug），单独 commit 中间状态可能不可用
- **文档化补救**：在 ADR-0008 追加"2026-06-18 postmortem"段落，列出 3 个 fix 各自的原因

### 教训
- **当多个 fix 互相依赖时，打包 commit + 详细 message 比强制拆开更安全**
- **文档化补救比 commit 拆开更重要**。后续读者通过 ADR 能看到完整故事

---

## 10. 注释的"必要性"判断标准

### 本次写的 3 个"看似可省略"的注释

#### 注释 A（`barrier.cpp` else 分支）
```cpp
// Mark thread as waiting at barrier so the executor (warp_context.cpp:267)
// recognizes BAR_SYNC and skips re-execution. Without this, sync_to_warp_state()
// keeps is_blocked=false and the scheduler spins on the barrier instruction.
// (Mirrors legacy SMContext::synchronize_barrier at sm_context.cpp:703.)
```
**必要性判断**：✅ 必要 — 解释"为什么需要 `set_state(BAR_SYNC)`"（看似冗余）。没有注释，未来维护者会因 next_pc 已设而删除它，重新引入 bug。

#### 注释 B（`cta_barrier.cpp` arrive()）
```cpp
// Inline completeness check instead of calling is_complete() — the public
// method re-locks mutex_, which would deadlock since we already hold it
// (std::mutex is non-recursive).
```
**必要性判断**：✅ 必要 — 警告死锁陷阱。没有注释，未来维护者会把 `arrived_threads_.size() >= expected_threads_` "重构"为 `is_complete()`，再次引入死锁。

#### 注释 C（`src/ptxsim/instructions/AGENTS.md`）
```markdown
**Do NOT fix `set_active_mask` semantics globally** to be additive — the
ret handler relies on overwrite semantics (`set_active_mask(0u)` to clear).
The OR logic must live in the CALLER.
```
**必要性判断**：✅ 必要 — 防止"全局修复"破坏其他 invariants。AGENTS.md 是项目级 invariant 文档，专门警告"看似合理但会破坏其他东西"的修改。

### 通用判断标准
> **写下来能否让一个 3 个月后的陌生人（也可能是 3 个月后的自己）避免犯同样的错？**
>
> - ✅ 是 → 写
> - ❌ 否 → 不写（让代码自解释）

### 排除
- ❌ 不要写"做了什么"（代码已经说清楚了）
- ❌ 不要写"为什么用这个库"（README/ADR 里有）
- ✅ 要写"为什么**不**用更直观的写法"（如注释 A、B、C）

---

## 11. 文档同步清单：改代码时必须同步更新的文档

### 本次涉及的同步项

| 代码改动 | 同步的文档 |
|---------|-----------|
| `barrier.cpp` BarHandler 改用 BarrierModule | `src/ptxsim/instructions/AGENTS.md`（"barrier/ subdirectory"） |
| `barrier_module.cpp` release_cta_barrier 新增 | `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md` |
| `Wbar` 标记 `[[deprecated]]` | `src/ptxsim/core/AGENTS.md`（"DO NOT add new uses"） |
| 整体架构变更 | `docs/adr/ADR-0008-barrier-semantics.md`（追加 2026-06-17 段落） |
| Phase 5 部分推迟 | `openspec/changes/.../tasks.md`（标记 Phase 5 为 deferred） |

### 教训
- **把"同步文档"作为 commit 模板的固定项**。例如 commit message 检查清单：
  ```
  □ 代码改动
  □ AGENTS.md 同步（如涉及 invariant / ANTI-PATTERN）
  □ ADR 追加段落（如涉及架构决策）
  □ OpenSpec tasks.md 更新（如涉及 task 状态）
  ```
- **AGENTS.md 的同步优先级最高**。它是给 AI agent 看的 invariant 文档，错一位就可能导致下次任务违反约束。

### 自动化建议
未来可以加一个 git hook：检测 `src/ptxsim/instructions/AGENTS.md` 中提到的"DO NOT"条款被违反时告警。

---

## 12. OpenSpec change 的"渐进式推进"模式

### 本次模式
OpenSpec change `integrate-barrier-module-cta-warp` 设计了 3 个 spec + 13 个 requirement。实施时拆成 5 个 Phase：

| Phase | 内容 | Commit | 状态 |
|-------|------|--------|------|
| 1 | 审计旧 API 使用 | （分析） | ✅ |
| 2 | 扩展 BarrierModule | `13b6b36` | ✅ |
| 3 | 单元测试 | `acb2311` | ✅ |
| 4 | 迁移 BarHandler | `b04cdb2` | ✅ |
| 5 | 迁移 BarWarpSyncHandler | `36dbb9a` | ❌ → revert |
| 5+ | 修复 + 回退 | `f033312` | ✅ |

### 教训
- **OpenSpec 的多 spec / 多 task 设计天然适合分阶段推进**。每个 task 都有明确的"完成定义"，可以独立 commit、独立验证。
- **任务粒度的选择**：太粗（一个 task 包含整个子系统）= 风险大；太细（每个 function 一个 task）= 进度噪声。**经验值：每个 task 是 1-3 个文件、1-3 个 commit 的工作量**。
- **当某个 task 失败时，回退到上一个 working task**。不要硬扛，避免污染后续 task。

### 决策
- **Phase 5 推迟到未来 change**，不在本次 change 中硬塞。理由：bar.warp.sync 的分歧/汇聚交互需要更深入分析，硬塞会引入更多 bug。
- **ADR 记录推迟理由**：在 ADR-0008 写明"为什么 Phase 5 推迟"，给未来读者明确指引。

---

## 13. 失败模式速查表

| 症状 | 最可能原因 | 诊断命令 |
|------|-----------|---------|
| 集成测试断言 N+1 处挂起 | 递归锁死锁 | `grep -n "lock_guard\|unique_lock" <file>` |
| `CHECK(!is_blocked)` 类似断言失败 | 跨模块状态翻译缺失 | `grep "set_state.*BAR_SYNC" src/` 对比 `grep "state == BAR_SYNC" src/` |
| 调度器在 barrier 指令处死循环 | `set_state(BAR_SYNC)` 漏掉 | `grep "set_state" src/ptxsim/instructions/barrier.cpp` |
| 分歧场景一半 lanes 卡住 | 屏障释放逻辑覆盖了已释放 lanes | 检查 `set_active_mask` 是否 OR 而非 overwrite |
| Phase N 测试通过但 Phase N+1 失败 | 跨 Phase invariant 冲突 | 用基线 worktree 隔离每个 Phase |
| `git revert` 后 `git status` 异常 | stash/pop 改变了 staged 状态 | `git status` 验证 + 必要时 `git reset` |
| ADR 与 AGENTS.md 对同一 API 推荐方向**互相矛盾** | 删除 deprecated API 后未同步文档 | `grep -rn "force_set_pc\|set_pc" docs/adr/ src/ --include="*.md"`，确认推荐方向一致 |

---

## 14. 可复用的 Checklists

### Checklist A: 函数迁移
```
□ 列出 baseline 函数中所有的 set_* / commit_* / force_* 调用
□ 列出所有 mutex_ / lock_guard / unique_lock 使用
□ 对每个 set_*，grep 其值的下游消费者
□ 对每个锁点，确认"持锁方法调用的所有其他方法"也持同一锁，或重写为无锁版本
□ 比对行级 diff（不只比对主要逻辑）
```

### Checklist B: 重构前
```
□ 建立基线 worktree (git worktree add .worktrees/baseline-check <commit>)
□ 列出本 change 的所有 Phase，决定 commit 粒度
□ 决定哪些 Phase 需要基线对比（涉及 invariant 的一定要）
□ 准备 revert 策略：每个 Phase 独立 commit，失败立即 revert
```

### Checklist C: 写注释
```
□ 这条注释能否让 3 个月后的陌生人避免犯同样的错？
□ 是 → 写
□ 否 → 不写
□ 例外：警告"不要做 X"（如 "DO NOT fix globally"）必须写
```

### Checklist D: Commit 前
```
□ 跑过 baseline worktree 对比
□ AGENTS.md 是否需要同步
□ ADR 是否需要追加
□ OpenSpec tasks.md 是否需要更新
□ commit message 列出独立的 fix 编号（如本次的 3 个 fix）
```

---

## 15. 未来 change 的建议

### 立即可做
- **修复预先存在的 2 个失败测试**（不在本次 change 范围）：
  - `unit_simt_stack_stale_entry_blocks_lane0`
  - `integration_cute_rmsnorm_bar_sync_pattern`
- **为 Phase 5（BarWarpSyncHandler 迁移）建独立 change**：需要先解决分歧/汇聚交互问题

### 中期改进
- **加 git pre-commit hook**：检测 AGENTS.md 中"DO NOT"条款被违反
- **加类型一测试覆盖到所有 CTA-level 状态变化**（防止类似本次的"set_state 漏掉"再次发生）
- **把"状态翻译"模式文档化**到 `src/ptxsim/core/AGENTS.md`：列出 `ThreadContext::state` → `warp_state.threads[i].*` 的所有翻译规则

### 长期演进
- **替换 `std::mutex` 为 `std::recursive_mutex` 或更明确的"内部无锁方法"**模式（防止未来类似死锁）
- **建立"迁移模式"工具集**：自动比对 baseline 与新实现的所有 API 调用差异

---

## 16. 类型判断只看 `qualifiers.back()` 导致 float 指令被当作整数处理

### 现象
`cute_rmsnorm` E2E 测试输出**非确定性垃圾值**（每次运行结果不同：1e-28, 1e+10, 1e-35 等）。但 `fma`、`rsqrt`、`mul.f32` 的独立单元测试全部通过。通过 handler 入口注入 `printf` 诊断发现：`MulHandler::processOperation` 中 `is_float=0`，`mul.f32` 被当作整数乘法处理，将 64 位整数乘积写入 32 位浮点寄存器，导致后续 `st.global.f32` 写出任意垃圾位模式。

### 根因
`TypeUtils::is_float_type()` (`src/ptxsim/utils/type_utils.cpp:10-11`) 只检查 `qualifiers.back()`：

```cpp
// Bug: 只检查最后一个 qualifier
bool TypeUtils::is_float_type(const std::vector<Qualifier> &qualifiers) {
    if (qualifiers.empty()) return false;
    Qualifier type = qualifiers.back();
    return (type == Qualifier::Q_F32 || type == Qualifier::Q_F64);
}
```

PTX 解析器为 `mul.f32` 生成的 qualifiers 列表是 `{Q_U32, Q_S32}`（数据类型 + 符号），Q_F32 不在最后一个位置。`qualifiers.back()` 返回 Q_S32，`is_float_type` 返回 false。Handler 走整数路径，用 `memcpy(dst, &int64_result, 8)` 把 64 位结果写入 32 位浮点寄存器 → 寄存器溢出、写坏相邻寄存器。

### 教训
- **`qualifiers.back()` 是脆弱的类型判断方式**。Qualifier 列表的最后一个元素是不确定的——它可能是数据类型、内存空间、比较运算符或修饰符。**必须遍历整个列表**检查目标类型。
- **单元测试通过 ≠ handler 在生产环境正确**。独立单元测试直接构造 `{Q_F32}` qualifiers（正确），但 PTX 解析器生成的 qualifier 列表结构不同（多个 qualifier），只有端到端测试能发现这种差异。
- **非确定性输出 = 内存/寄存器损坏的诊断信号**。本案例中每次运行得不同值（1e-28 → 1e+10 → 1e-35），是因为 `mul.f32` 将 64 位整数结果溢出写入 32 位寄存器，相邻寄存器的随机值被部分读出。
- **影响面评估**：此 bug 影响所有使用 `is_float_type()` 的 handler：`MulHandler`, `DivHandler`, `MadHandler`, `FmaHandler`, `AddHandler`, `SubHandler`, `AbsHandler`, `NegHandler`, `MinMaxHandler` 等。任何 `{Q_xxx, Q_F32}` 列表（Q_F32 在中间）都会被误判为整数。

### 修复
```cpp
// After: 遍历所有 qualifier
bool TypeUtils::is_float_type(const std::vector<Qualifier> &qualifiers) {
    if (qualifiers.empty()) return false;
    for (const auto &q : qualifiers) {
        if (q == Qualifier::Q_F32 || q == Qualifier::Q_F64 ||
            q == Qualifier::Q_F16 || q == Qualifier::Q_BF16)
            return true;
    }
    return false;
}
```

### 诊断命令
```bash
# 在 handler 入口加临时代码，观察 qualifier 列表结构
# 在 executeOperation() 中打印 qualifiers 和 is_float 判断
grep -rn "is_float_type" src/ptxsim/
# 列出所有依赖此函数的 handler — 全部受影响
```

### 检查工具
```bash
# 检查所有使用 qualifiers.back() 进行类型判断的位置
grep -rn "qualifiers.back()" src/
grep -rn "qualifiers\[" src/
# 确保没有其他脆弱的 "最后一个元素" 类型判断模式
```

### 真实案例
- **bug 表现**: `cute_rmsnorm` E2E 输出垃圾值，`cute_rmsnorm_bak` 同样失败。三次运行得三种不同的值。基础 handler 单元测试全部通过。
- **诊断过程**:
  1. 最小调试测试（N=8, 1 warp）确认 Step 1-3（fma, reduction, rsqrt）全部正确 → 缩小范围到 Step 4
  2. `st.global` handler 注入诊断 → 写出的值是垃圾，但 `ld.global.nc` 读入的值正确
  3. `mul.f32` 入口注入诊断 → `src1` (scale) 和 `src2` (input) 都正确，但 `st.global` 输出仍是垃圾
  4. `MulHandler::processOperation` 入口注入诊断 → `is_float=0`（应为 1）→ 定位到 `is_float_type()` 的 `qualifiers.back()` 错误假设
- **修复**: 将 `is_float_type()` 改为遍历所有 qualifier 检查 Q_F32/Q_F64/Q_F16/Q_BF16

---

## 17. 元信息

- **作者**: Sisyphus agent (本次 change 实施)
- **日期**: 2026-06-18
- **关联 commit**: `f033312` (fix(barrier): revert Phase 5 + fix BAR_SYNC state + CTABarrier deadlock)
- **关联 change**: `openspec/changes/integrate-barrier-module-cta-warp/`
- **关联 ADR**: `docs/adr/ADR-0008-barrier-semantics.md`（已追加 2026-06-18 postmortem）
- **关联 plan**: `docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md`
- **关联 skills**: `.opencode/skills/ptx-barrier-mechanism/`, `.opencode/skills/regression-bisect/`, `.opencode/skills/state-modification-audit/`

---

## 18. OpenSpec artifacts 提交遗漏 + Debt audit 必须 git verify

### 现象（OpenSpec artifacts commit 遗漏）

`cleanup-deprecated-barrier-apis` change 的实施过程（2026-06-20, commits `8a5573d` / `7914764` / `6ec8efd`）中：

1. 实施者在工作区修改了 `openspec/changes/cleanup-deprecated-barrier-apis/{design.md,tasks.md,specs/cleanup/spec.md}` 反映实施调整
2. 但**这些 OpenSpec artifacts 修改从未 `git add`** — 仅源码 + commit message 描述了改动
3. fast-forward merge 后，`openspec/changes/cleanup-deprecated-barrier-apis/` 目录仍是修改前的旧版本（untracked reconstructed artifacts）
4. 补救：commit `4d38772`（14:05）重建 artifacts → commit `ded4f96`（14:07）归档 change
5. **12 天后**（2026-07-02），Sisyphus agent 审计债务时基于 untracked reconstructed artifacts（而非 archive），误判 4 条 P0-A 仍为 active debt，并基于 stale audit 创建了 `barrier-migration-amendment` change 试图 amend 已归档 change — 触发本 lesson

### 教训

- **实施 OpenSpec change 时必须按 2-Phase commit 顺序**：
  1. **Phase 0**：`git add openspec/changes/<name>/` + commit "docs(openspec): <name> design adjustments" (artifacts FIRST)
  2. **Phase 1+**：实施代码 + commit
- **任何 debt audit 必须满足 2 个先决条件**：
  1. **当前 git HEAD 状态**（不是 working tree）— `git status` + `git log -- <path>` 验证
  2. **commit hash 而非文件路径** — 引用 `git log --all -- <path>` 结果，不引用 working tree 内容
- **OpenSpec archive = change 终态**：归档后任何修补需求应新建 `fix-*` 或 `refactor-*` change，并 `Ref: archive/<date>-<name>/`，不要 amend 已归档 change
- **审计撰写时若 working tree 与 git HEAD 不一致，必须明确标注**："基于 working tree 状态，可能与 HEAD 不一致"

### 检查工具

```bash
# 1. 验证 change 是否已归档
git log --all --oneline -- "openspec/changes/<change-name>/"
# 应包含 archive commit（如 ded4f96 chore(openspec): archive ...）

# 2. 验证 change 实施状态
git log --all --oneline -- <实施文件路径>
# 列出所有改动该文件的 commits

# 3. 验证 artifacts 是否 tracked（实施后必须）
git ls-files openspec/changes/<change-name>/
# 不应为空 — artifacts 应在 git 中

# 4. 审计前自检
git status openspec/changes/  # 工作区未提交修改警告
git diff HEAD openspec/changes/  # 应无差异（若审计基于 HEAD）
```

### 真实案例

- **触发**: `barrier-migration-amendment` change (2026-07-02) — 试图 amend 已于 2026-06-20 归档的 `cleanup-deprecated-barrier-apis`
- **错位审计**: `.opencode/notes/debt-audit-2026-07-02.md` §1.1 P0-A1~A4 误标为 active debt — 实际已通过 commits `8a5573d`/`7914764`/`6ec8efd` 解决
- **修复**:
  1. 删除 `openspec/changes/barrier-migration-amendment/` 和 `openspec/changes/cleanup-deprecated-barrier-apis/` (untracked 重构副本)
  2. 更新 `docs/audits/debt-audit-2026-07-02.md` §1.1 标记 P0-A1~A4 为 RESOLVED（引用 commits）
  3. 本 lesson #18 + 配套 Checklists E/F/G（见 `.opencode/skills/ptx-lessons-learned/SKILL.md`）
- **回归保护**:
  - 任何 future OpenSpec change 实施时 apply Checklist E（artifacts commit FIRST）
  - 任何 future debt audit 撰写时 apply Checklist F（git verify FIRST）

---

## 19. 成功应用 §1 跨模块状态翻译：BarWarpSyncHandler 迁移到 BarrierModule API

### 现象

`migrate-bar-warp-sync-to-barrier-module` change（commits `0e311566`+`f5640042`+`0bab6487`，2026-07-03）将 `BarWarpSyncHandler::processOperation`（路径 A force_reconvergence + 路径 B 正常 barrier）从直接操作 `warp_state.wbars[0]` + `sm_ctx->bsync_manager_` 迁移到 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` API。

之前 `commit 36dbb9a` 失败 + `f033312` revert，原因正是 §1 "跨模块间接状态翻译" — 漏掉了 release 路径的 `is_blocked=false` + `status=Active` + `is_active=true` 翻译以及 `set_pc_overridden(true)` PC 防双推进。

### 教训

**1. §1 的 fix 是这次成功的关键**: 迁移 `release_warp_barrier` 时严格按行级 diff 把 5 件事完整迁移：
  - `set_active_mask(get_active_mask() | arrived_mask)` —— OR 逻辑保留（BUG-POSTBARRIER-TWOHALVES）
  - `is_blocked=false` —— 解除阻塞
  - `status=Active` —— 重新激活调度
  - `is_active=true` —— 派发级 active
  - `context->set_pc_overridden(true)` —— release 调用方的责任（防止 `commit_pc()` 二次推进）

 每一项都有对应的下游消费者（`grep "<state>\|is_<state>" src/...` 可查），看似"次要"的都不能丢。

**2. Path A vs Path B 守卫条件对称翻译**: `current_wbar_id < 0` → `!init_wbar->is_initialized()`；`current_wbar_id >= 0` → `init_wbar->is_initialized()`。在两条路径中**保持一致** —— 不一致的守卫条件会破坏 force_reconvergence 重新进入的语义（BUG-RECONVERGENCE-SIMPLEGEMM）。

**3. P0-A5 Wbar 删除 = 单一来源**: 删除 `wbar.h`、`warp_state.wbars[]`、`current_wbar_id`、`get_wbar()` compat shim 后，所有 barrier 状态集中到 `CTAContext::barrier_module_`。代码层面 grep 零残留（`grep -rn "wbar\.h\|warp_state\.wbars\|current_wbar_id\|get_wbar(" include/ src/ tests/` → 仅剩注释引用）。

### 真实案例

- **触发 commit**: `0e311566`（feat）+ `f5640042`（chore, P0-A5 删除 Wbar）+ `0bab6487`（tests 修复）
- **§1 fix 的应用**: `src/ptxsim/barrier/barrier_module.cpp:111-134` `release_warp_barrier` —— OR 逻辑、is_blocked/status/is_active 全部在一个函数体内集中
- **§1 fix 的迁移验证**: 23/23 barrier 测试包括 `unit_post_barrier_two_halves`（BUG-POSTBARRIER-TWOHALVES）、`unit_barrier_reconvergence_simplegemm`（BUG-RECONVERGENCE-SIMPLEGEMM）、`e2e_barrier_warp_sync` 全部 PASS
- **ADR 更新**: `docs/adr/ADR-0008-barrier-semantics.md` §2026-07-03 追加完整 postmortem（含 Phase 3/Phase 7/Phase 7b 三 commit 拆分 + lessons §1/§2/§4 的应用证据）
- **关键决策**: "OR logic 是单一所有者（single owner）"，不再分散到所有 barrier caller —— `BarrierModule::release_warp_barrier` 是唯一拥有 OR 语义的函数

### 与 §1 §2 §4 的对比

| 教训 | 之前（commit 36dbb9a 失败） | 之后（commits 0e311566+ 成功） |
|------|---------------------------|-------------------------------|
| §1 跨模块状态翻译 | 漏掉 release 路径的 OR + is_blocked/status/is_active 翻译 | 严格行级 diff 在 `release_warp_barrier` 内完整保留 5 件事 |
| §2 递归锁 | 未触发（migration handler 未涉及 lock） | 同上 |
| §4 复杂迁移分 Phase | 单个 commit 36dbb9a 同时改 + 跑测试 | Phase 3 (commit 0e311566) + Phase 7 (commit f5640042) + Phase 7b tests (commit 0bab6487) 三个独立 commit，失败独立可 revert |

### 实战 checklist（apply 到任何"从 handler 直接状态字段迁移到 Module API"的工作）

- [ ] **行级 diff 列出所有 baseline 函数中的 `set_*()` 调用**（不仅 main logic）
- [ ] **确认新 Module API 内包含了所有翻译**（grep 对照调用点 vs 消费点数量应一致）
- [ ] **检查 Path A vs Path B 守卫条件的对称性**（force_reconvergence 不应该有非对称的 guard）
- [ ] **确认 release/cleanup path 由 caller 控制 PC 翻转**（`set_pc_overridden(true)`）
- [ ] **删除 compat shim 前先 grep "全项目零匹配"**（`grep -rn "<old_name>\|<deprecated_field>" include/ src/ tests/`）
- [ ] **独立 commit 拆分**（migrate 单 commit + delete 单 commit + tests 修复单 commit）

---

## 20. Pre-implementation Review：实施 OpenSpec change 前必须跑 Metis 子代理审计（2026-07 新增）

### 现象（OpenSpec scope 漂移）

`fix-cvt-strategy-actual-split` change（2026-07-04/05，commits `e8db807`/`f3ef891`/`43edf55`）原 proposal 基于 **未实证** 的假设撰写：

- 假设"`GeneralCvtStrategy::convert()` 919 行 switch 块**未拆分**"——计划 6 Phase 拆分出 5 个新 Strategy 类
- 假设"`select_strategy()` 返回 `unique_ptr<ConversionStrategy>`"（与代码现状矛盾）
- 假设"94 个 integration 测试为 oracle"（实际为 14 个）
- 假设"`.worktrees/fix-pre-p0-baseline` 可复用"（实际 worktree 目录为空）
- 假设"修复 `.opencode/notes/debt-audit-2026-07-02.md` §P0-C1"（实际文件在 `docs/audits/`）

**Metis pre-implementation review**（Phase 0 后，apply 前）实证发现：4 个活 Strategy 类早已通过 commits `fc3c352`/`9837d44`/`d6123e0` 部署，`select_strategy()` 持续 dispatch，`GeneralCvtStrategy` 是**死代码**（grep 0 external callers）。

若按原 plan 硬实施会：
1. 浪费时间把已在用的 Strategy 类搬到新文件
2. 引入 `CvtSatStrategy` 双重饱和 bug（4 个 Strategy 内部已处理 `.sat`，wrapper 会 double-saturate）
3. 改 `select_strategy()` 接口破坏 Non-Goal（"不修改 `cvt_strategy.h`"）
4. 引用 94 个虚构测试做 oracle（误判基线）

### 教训

**1. OpenSpec proposal 必须基于实证而非"目录/文件存在性推断"**：archive README 中的"✅ COMPLETED"标记 + 文件未删除 = 不等于"已完整实施"。必须 `git log -- <path>` + `grep <api>` + `wc -l <file>` 实证。

**2. 必须区分"已实施但未清理"与"未实施"两种状态**：
| 表象 | 实际状态 | 错误判断 | 正确判断 |
|------|---------|---------|---------|
| `tasks.md` 标记 ✅ + 文件未删除 | 已实施 + 死代码残留 | "未实施，需要拆分" | "已实施，应清理死代码" |
| 文件 1061 行 + 注释声称"待拆分" | 拆分已部署到其他文件 | "switch 块未拆分" | "原位置是死代码，新位置才是活代码" |

**3. 引用 "工作已完成" 作为 oracle 时需格外小心**：archive 中的 "94 个 integration tests" 可能是未来目标（如 P1-4.1 fix 启用后），不是现状。验证 oracle 必须 `ctest -N -L "<label>"` 实际查询。

**4. Metis 子代理审计产出 ⚠️ CONDITIONAL 必须 5 项 MUST-RESOLVE**：本 case 的 5 项 MUST-RESOLVE 全是实施前的隐形炸弹（scope 错误 + 接口矛盾 + 测试数量虚构 + worktree 不存在 + 路径错误），由 Metis 一次 review 全部揭示。

### 检查工具

```bash
# 实施 OpenSpec change 前必跑（Metis 子代理 prompt 模板）
# 1. 列出 baseline 文件的关键状态
wc -l <file>                              # 当前行数
git log --oneline -10 -- <file>          # 实施历史
git log --all --oneline -- "<change-dir>" # 归档状态

# 2. 验证 proposal 引用的关键 API 确实按描述存在
grep -rn "<symbol>" src/ include/ tests/  # 0 matches = API 不存在 = 假设错误

# 3. 验证 oracle 测试数量真实
ctest -N -L "<label>" 2>&1 | tail -5      # Total Tests 应等于 proposal 引用数

# 4. 验证提到的文件/路径/工具真存在
ls <worktree-path> 2>/dev/null            # empty = 不存在 = 不要假装"复用现有"
test -f <path> && echo exists || echo missing
```

### 真实案例

- **触发 change**: `fix-cvt-strategy-actual-split`（commits `e8db807`+`f3ef891`+`43edf55`，2026-07-05）
- **Metis audit 触发**: 用户进入 Phase 0 后调用 Metis 子代理审计 artifacts
- **5 项 MUST-RESOLVE before apply**:
  1. Change scope 错误（实际是 dead code 删除 + 文档同步，非 6 Phase 拆分）
  2. `CvtSatStrategy` 双重饱和（架构缺陷）
  3. `select_strategy()` 返回 `const ref` vs `unique_ptr` 接口矛盾
  4. 测试数量 94 → 14（oracle 虚构）
  5. `.worktrees/fix-pre-p0-baseline` 不存在（worktree 引用虚构）
- **修订后**:
  - 6 Phase → 3 Phase（Phase 0 artifacts + Phase 1 delete dead code + Phase 2 doc sync）
  - 预估工作量：~500 行拆分 → 实际 60 行（pure deletion + 文档同步）
  - 实施 commits 全部 0 回归（14 CVT + 33 PTX + e2e GEMM + 全套 178 ctest PASS）
- **沉淀位置**:
  - `openspec/changes/fix-cvt-strategy-actual-split/{proposal,design,tasks}.md` §Scope 修订说明
  - `docs/adr/ADR-0015-cvt-strategy-pattern.md` §2026-07 Fix 段（含 lessons-learned §20 案例沉淀）
  - `.opencode/skills/ptx-lessons-learned/SKILL.md`（新增 §20 + Checklist H）
  - 本 lessons-learned.md §20（本节）

### 与 §18 的对比

| 维度 | §18 案例（cleanup-deprecated-barrier-apis） | §20 案例（fix-cvt-strategy-actual-split） |
|------|-------------------------------------------|---------------------------------------------|
| 失败模式 | artifacts 提交遗漏，archive 后误判 active debt | proposal scope 错误，引用未实证的假设 |
| 检测时机 | archive 后 12 天（reactive） | apply Phase 0 后（proactive） |
| 检测手段 | debt audit 工作流（passive） | **Metis pre-implementation review**（proactive） |
| 修复方式 | 重建 artifacts + debt audit RESOLVED | 重写 proposal/design/tasks + scope 修订 |
| 工作量浪费 | commit `4d38772` 重建（~50 行） | 6 Phase 计划 → 3 Phase（避免 ~500 行无效迁移） |

### 实战 checklist（apply 到任何"实施 OpenSpec change"的工作）

- [ ] **实施前**：调用 Metis - Plan Consultant 子代理审计 proposal/design/tasks.md
- [ ] **Metis 输出 CONDITIONAL 时**：5 项 MUST-RESOLVE 全部完成后才能 apply
- [ ] **实施中**：每次 git log commit 后用 `git grep "<api>"` 验证假设持续成立
- [ ] **实施后**：debt audit 的"基于 HEAD <hash>" 标注（Checklist F）
- [ ] **归档时**：调用 openspec-archive-change skill（含 postmortem 选项）

---

## 21. 重大功能交付必须包含根 README 同步（root README sync after major feature delivery）

### 现象（README 滞后于实现 1 个月）

`implement-wmma-tensor-core-tcgen05` change（2026-07-04 archived，含 26 commits + ADR-0016 Blackwell-only + 5 个 tcgen05 指令完整实施 + e2e GEMM kernel）完整交付后，**根 `README.md` 未同步更新**：

- `README.md:48` 仍声称 "**WMMA / Tensor Core**：是 stub"
- `README.md:47` 仍硬编码 "核心 ISA ~67%"
- `README.md:50` 仍硬编码 "CUDA Toolkit：11.4.4 测试通过"
- `README.md:3` 仍声称 "SIMT v2.0 (Phase 10 进行中)"

**根因**：archive commit `79fc236` 仅标记 tcgen05 实施完成（per Checklist G），但未检查根 README.md 是否仍准确描述实现状态。这是 lessons-learned §6 "stale artifact" 案例的 README 子集 — 实施 + archive 完美，但描述文档滞后 30 天。

### 教训

**1. "重大功能交付" = 代码 + 单元测试 + e2e + README 同步（4 项缺一不可）**：

| 项 | 责任方 | 验证 |
|---|--------|------|
| 代码 | 实施者 | `git log` 验证 |
| 单元测试 | 实施者 | `ctest -L unit` |
| e2e | 实施者 | `ctest -L e2e` |
| **README 同步** | **实施者 / archive reviewer** | **`grep` 关键术语（如 "WMMA / stub"）应在根 README 找不到** |

**2. Archive commit 必须包含"README 状态同步"检查**（per Checklist G lifecycle 约束）。本 case 的 fix 是新建 `sync-readme-after-tcgen05` change（不 amend 已归档 change）+ `Ref:` 链接，符合 OpenSpec Checklist G。

**3. 根 README 是"对外第一印象"**：新开发者读 README.md 找方向，错误描述会立即误导（"WMMA 是 stub" → "项目还在非常早期" → 不会查 docs/adr/）。修复越晚，误导人数越多。

### 检查工具

```bash
# 任何 archive commit 前必跑：根 README "已知限制" / "状态" 章节 grep
grep -n "stub\|TODO\|FIXME\|不实现\|未完成" README.md
# 应: 0 matches (除非有明确 TODO 标注 + 修复计划)

# 验证状态描述对齐实际
grep -n "进行中\|完成\|TODO" README.md
# 与 docs/README.md Phase 表格 + AGENTS.md 状态对比应一致

# 验证 PTX 指令覆盖数字（如有）
grep -nE "[0-9]+%|第[一二三]" README.md
# 硬编码数字应替换为自动统计链接（参考 docs/audits/）
```

### 真实案例

- **触发 change**: `sync-readme-after-tcgen05`（commits `8427829` + `80271cd` + `91aeef2` + `4b8cb6b` + `746d083` + `cee527f`，2026-07-05）
- **延迟**: implement-wmma-tensor-core-tcgen05 (2026-07-04) → sync-readme (2026-07-05) = **1 天延迟**（本应在 archive 时同步）
- **修复量**: README.md +15/-5 行（5 个 commits：1 artifacts + 1 revision + 3 README Fix #1-#3 + 1 archive）
- **提交顺序**: Phase 0 artifacts FIRST（artifacts tracked → 修订 → 实施 → 验证 → 归档）
- **Lessons-learned 集成**: 严格遵守 §6 (artifacts-first) + §19 (跨模块状态翻译) + §20 (Pre-implementation Review) 三个 lessons
- **关联 postmortem**: tasks.md Phase 4.5 沉淀（per openspec-archive-change skill 强制 postmortem prompt）

### 与 §6 / §18 / §20 的对比

| 维度 | §6 案例（cleanup-deprecated-barrier-apis） | §18 案例（barrier-migration-amendment） | §20 案例（fix-cvt-strategy-actual-split） | **§21 案例（sync-readme-after-tcgen05）** |
|------|---|---|---|---|
| 失败模式 | artifacts 提交遗漏 | debt audit 误判已解决 debt | proposal scope 错误 | **根 README 同步遗漏** |
| 检测时机 | archive 后 12 天 | debt audit 时 | apply Phase 0 后 | **archive 后 1 天** |
| 修复方式 | 重建 artifacts | debt audit RESOLVED | 重写 proposal/design/tasks | **新建 sync-* change + Ref 链接** |
| 工作量 | 重建 ~50 行 | audit 修正 + lesson #18 | 6 Phase → 3 Phase 修订 | **15 行 README 同步** |
| Lessons | #18 stale artifact | #18 同上 | #20 pre-impl review | **#21 README 同步清单** |

### 实战 checklist（apply 到任何"重大功能交付"的工作）

- [ ] **实施阶段**：根 README.md "状态" / "已知限制" 章节随代码同步更新（不延后到 archive）
- [ ] **Archive commit 前**：grep 检查 "stub / TODO / FIXME / 未实现 / 硬编码百分比" 在 README.md 中应为空（或有明确 TODO 标注 + 修复 plan）
- [ ] **任何 `feat-*/implement-*` change 归档前**：必跑本 checklist
- [ ] **新 `fix-*/sync-*` change 处理已归档案例**：通过 Ref 链接 + 不 amend（per Checklist G）
- [ ] **postmortem 沉淀**：在 lessons-learned.md 追加 §N（本节作为 §21 模板），同步 .opencode/skills/ptx-lessons-learned/SKILL.md Checklist I

---

## 22. multi-PTX cubin 静默截断 + Metis pre-impl review 完整落地（2026-07 新增）

### 触发 change

`parser-completeness`（commits `eafc70f`/`918891d`/`aed66e9`，2026-07-05）— 3-Phase 清理 + 1 P0 fix。

### 现象

PTX-EMU 解析 multi-section PTX cubin 时，`src/ptx_parser/ptx_parser.cpp:60` 的 `ptx_code = of_ptx.str();` 覆盖语义导致仅保留最后 section 的 PTX 代码，前 N-1 section 全部丢失，且**无任何 warning 告知用户**。同期 `src/utils/cubin_utils.cpp` 已正确实现 append-all 行为（c5 Fix #3），parser 层与 cubin_utils 行为不一致。

### 教训

1. **跨模块行为对齐**（§1 延伸）：parser 层与 cubin_utils 层对同一概念（"multi-section PTX"）采用不同语义（覆盖 vs append），导致数据丢失。修复必须使两层行为一致。
2. **静默失败是最危险的失败模式**：无 warning 输出意味着用户不知道数据丢失。修复引入 `PTX_WARN_EMU` + section_count 计数器，确保多 section 时主动告知。
3. **Metis pre-impl review + 3-Phase scope 修订 = 成功模式**：
   - 原提案"10 条债务 + 6 Phase"被 Metis 一次 review 揭示为"12 条债务（其中仅 1 条真 P0）+ 死代码删除优先"
   - 修订后的 3-Phase 落地：Phase 1 死代码清理（0 行为变更）→ Phase 2 P0 fix（multi-PTX warning）→ Phase 3 文档同步
   - 实测耗时：~2h 实施 + oracle test（含 `__VA_ARGS__` 宏嵌套坑） + AGENTS.md 同步

4. **`__VA_ARGS__` 宏嵌套陷阱**（新增）：Catch2 `REQUIRE_NOTHROW(PTX_WARN_EMU("fmt", args))` 因 `__VA_ARGS__` 嵌套展开失败编译。修复：用 lambda 包装 `auto warn_call = []() { PTX_WARN_EMU(...); }; REQUIRE_NOTHROW(warn_call());`。此 workaround 必须在测试文件中保留注释解释，避免未来 reader 移除 lambda。

5. **oracle test 缺失是 scope 通胀信号**（§20 延伸）：Metis MR-4 揭示 `tests/unit/parser/` 不存在（proposal 声称"5+ oracle 测试"是假设）。处理：创建最小 oracle test（5 个 TEST_CASE 覆盖 multi-section / single-section / empty / smoke / regression），用 lambda 包装避免宏嵌套。

### 真实案例

- **OpenSpec change**: `openspec/changes/parser-completeness/`
- **Fix commits**:
  - `eafc70f` docs(openspec): add parser-completeness artifacts
  - `918891d` refactor(parser): delete dead code + update 5 stale comments (Fix #1)
  - `aed66e9` fix(parser): multi-PTX cubin warning + 累加语义 (Fix #2)
- **测试**: `tests/unit/parser/test_multi_ptx.cpp` (5 TEST_CASE) — 注册为 `unit_multi_ptx`
- **AGENTS.md**: line 510 "Multi-PTX cubins" 描述更新为"累加所有 sections + warning + 风险提示"
- **关联 change**: `archive/2026-07-05-fix-cvt-strategy-actual-split/`（同 Metis pre-impl 模式先例）

### 检查工具

```bash
# 1. 验证 multi-PTX 累加行为（已含 regression test）
grep -n "ptx_code += of_ptx.str\|ptx_code = of_ptx.str" src/ptx_parser/ptx_parser.cpp
# 期望：仅 += ，无裸 =

# 2. 验证 PTX_WARN_EMU 触发条件
grep -A1 "section_count > 1" src/ptx_parser/ptx_parser.cpp
# 期望：包含 PTX_WARN_EMU 调用

# 3. 验证 oracle test 存在 + PASS
ctest -R "unit_multi_ptx" --output-on-failure
# 期望：1/1 Passed

# 4. 验证 AGENTS.md 描述一致
grep -A1 "Multi-PTX cubins" AGENTS.md
# 期望：描述包含"累加" + "PTX_WARN_EMU" + "潜在风险"
```

---

## 23. OpenSpec artifacts 内部一致性强制检查（proposal/design/spec/tasks）（2026-07 新增）

### 触发 change

`docs-cuda-docs-and-openspec-orphan-sync`（2026-07-06，审查阶段发现）— 纯文档债务清理 change，但 proposal / design / tasks / specs 间存在 3 处隐性内部不一致，全部需要 MUST-RESOLVE。

### 现象

OpenSpec change 的 4 个 artifacts 单独看都"形式合规"，但**互相之间存在内部冲突**时仍会通过 self-review。具体表现：

1. **范围不一致**：proposal.md §What Changes 声称"D-5 删除 3 个过期副本"，但 design.md §Decision 3 / tasks.md Phase 3 / spec.md Requirement 2 都明确删除 **4 个**（含 `three-mode-testing/` 已禁用副本）。
2. **设计决策 vs spec Scenario 直接冲突**：design.md §Decision 1 写"归档目录内添加 README.md 段落引用 retroactive design.md"，但 spec.md Scenario "Retroactive design.md 不修改归档内容" 强制 "归档目录内所有文件的 git hash SHALL 保持不变"。两者**字面冲突**（一个允许修改 README.md，一个禁止修改任何文件）。
3. **任务路径与设计路径不一致**：design.md §Decision 1 路径示例 `openspec/changes/archive/<date>-<name>.design.md`（与归档子目录并列），但 tasks.md Phase 2.6 验证命令检查 `$d/design.md`（在归档子目录内）— **路径策略完全不匹配**。

### 教训

1. **OpenSpec artifacts 的 4 个文件本质是同一份文档的 4 种视图**（per `openspec-propose` skill 设计）。任何 artifact 内部决策必须在另外 3 个 artifact 中**对称出现**，否则视为决策未完成。
2. **"逐文件审查"无法捕获内部冲突**：审查者逐文件看 proposal ✓ → design ✓ → specs ✓ → tasks ✓，但**缺少"跨文件同一概念"的对齐检查**。这是 single-file review 的天然盲点。
3. **Checklist G 的字面引用 vs 实质合规**：design.md 表面引用了 Checklist G（"不 amend 已归档 change"），但实质上"添加 README.md 段落引用"违反 Checklist G 的精神。引用不等于遵循 — 必须做**路径 + 操作**级别的合规。
4. **方案选择模糊时优先严格约束**：当 design.md 在"修改归档 README" 和"完全不修改归档" 之间含糊时，**应直接选严格方案**（完全不修改）。这避免后续争论，也容易 spec.md 验证。

### 真实案例

- **OpenSpec change**: `openspec/changes/docs-cuda-docs-and-openspec-orphan-sync/`
- **审查发现**: 3 处 MUST-RESOLVE（1. D-5 范围不完整 / 2. design Decision 1 vs spec Scenario 冲突 / 3. tasks Phase 2.6 路径错误）
- **审查后修复**: 
  - proposal.md +7/-1 行（补全 `three-mode-testing/`）
  - design.md +8/-6 行（Decision 1 改为"禁止修改归档"、Risks 表新增"误改归档"项）
  - tasks.md +10/-7 行（路径策略块、新增 2.7 "验证归档未变"）
  - 总计 +24/-15 行，仅文档范围（按设计"不修改任何 .cpp/.h"约束）
- **审查产物**: `docs-cuda-docs-and-openspec-orphan-sync-review-report.md`（完整审查报告 + 3 个 MUST-RESOLVE 修复建议）
- **关联教训**: 与 §6（OpenSpec artifacts 提交遗漏）+ §20（Pre-impl review）三者协同 — 三者都强调"artifacts 间一致性"，但 §23 专攻"artifacts 内部冲突"维度

### 检查工具

```bash
# 1. D-系列范围对齐检查（proposal/design/tasks/spec 同一债务项的范围数字一致）
for term in "D-1\|D-4\|D-5\|D-6"; do
  echo "=== $term range consistency ==="
  for f in proposal.md design.md tasks.md specs/*/spec.md; do
    grep -c "$term" "openspec/changes/<name>/$f" 2>/dev/null
  done
done
# 期望：proposal/design/tasks/spec 中同一债务项的数字/对象列表一致

# 2. design 决策 vs spec Scenario 一致性检查
echo "=== design Decision 1 路径 vs spec Scenario 1 路径 ==="
grep -A1 "Decision 1" openspec/changes/<name>/design.md | head -5
echo "---"
grep -A3 "Scenario.*不修改\|hash SHALL" openspec/changes/<name>/specs/*/spec.md
# 期望：design 与 spec 中描述的路径 + 操作语义一致

# 3. tasks 路径策略 vs design 路径示例对齐
echo "=== tasks path vs design path ==="
grep -E "\\.design\\.md" openspec/changes/<name>/tasks.md | head -3
echo "---"
grep -E "\\.design\\.md" openspec/changes/<name>/design.md | head -3
# 期望：tasks 中的 `test -f` 命令路径 = design.md 路径示例

# 4. "禁止修改" / "禁止 amend" / "不修改" 关键词全文检索
grep -rE "禁止|不修改|不可 amend" openspec/changes/<name>/ 2>/dev/null
# 期望：每个约束都明确写出，且 spec.md Scenario 提供可执行的验证命令

# 5. 审查产物登记（项目级实践）
ls .opencode/notes/<name>-review-report.md 2>/dev/null
# 期望：每个 ≥30 commits 影响范围的 change 都有审查报告
```

---

## 24. docs-* change 实施经验：retroactive subagent、inline edit、__pycache__ 陷阱（2026-07 新增）

### 触发 change

`docs-cuda-docs-and-openspec-orphan-sync`（commits `c913bf3`/`d80088e`/`9c553bc`/`5ffc72f`/`6205a1d`，2026-07-06）— 5-Phase 实施闭环 6 条 D-系列债务（D-1~D-6）。

### 现象

实施纯文档 OpenSpec change 时，3 个非显而易见的陷阱：

1. **Retroactive artifact 合成的 subagent 数量决策**：5 个孤儿 change 缺 design.md，需要合成 5 个 retroactive design.md。决策"5 个并行 subagent 各写一个" vs "1 个 subagent 写全部"。
2. **12 个 inline 勘误标记的 edit 精度**：ERRATA 8 项（E1-E8）需要合并到主审计，但 ERRATA.md "官方说明"明确"主审计原文保持不变"。需要在 12 个位置精确添加 inline `**[勘误: ...]**` 标记而不修改原文。
3. **`git rm -r` 不会清理 .gitignored 的子目录**：`docs/skills/three-mode-testing/__pycache__/` 被 `.gitignore` 排除（`__pycache__/` 规则），`git rm -r three-mode-testing/` 仅删除 tracked 文件（SKILL.md + generate_tests.py），空 `__pycache__/` 子目录残留文件系统上。

### 教训

1. **Retroactive artifact 合成：1 个 subagent > N 个并行**：
   - 5 个 design.md 共享模板（"Retroactive synthesis from git log" 标注 + 7 段结构）+ 共享 ADR 引用规则 + 共享 commit hash 验证协议
   - 1 个 subagent：1 套上下文、5 次 Write 调用、模板一致性自然保证、commit hash 验证只跑 1 次
   - 5 个并行：5 套上下文、5 份模板各自维护、commit hash 验证重复 5 次、风险是某个 subagent 漏掉 synthesis 标注
   - **判定标准**：N 个相关任务的合成（共享模板/上下文/验证协议）→ 1 个 subagent；N 个独立任务的合成（不同模板/不同上下文）→ N 个并行

2. **Inline edit 策略 = 9 个 edit 操作，分批执行**：
   - 12 个 inline 标记分 9 个 edit：E1 在 §0.2 + §1.2 = 2 edits，E2 在 §0.2 + §2.2.1 = 2 edits，E3-E8 各 1 edit
   - 每个 edit 用 Read 验证目标段落的精确字符串 → Edit 添加 `[勘误 ...]` 标记 → 不修改原文
   - **关键约束**：ERRATA.md 顶部"官方说明"写"原审计作为 commit `baa8c4e` 的历史快照保持不变"——这是 ERRATA 文件本身的契约，违反会导致审计作为"历史快照"的可信度受损

3. **`git rm -r` + `.gitignore` 的盲区**：
   - `.gitignore` 中 `__pycache__/` 规则使目录内文件 untracked
   - `git rm -r three-mode-testing/` 仅删除 tracked 文件（SKILL.md + generate_tests.py），不动 untracked 文件
   - 但 `git rm -r` 会**报错**（"pathspec did not match any files"）如果目录已完全 untracked
   - **正确流程**：`git rm -r <dir>/`（删除 tracked 部分）→ 检查残留 → `rm -rf <dir>/<untracked-subdir>`（清理 untracked）
   - **替代方案**：`git clean -fd <dir>/`（删除所有 untracked，但**不可逆**——需先 dry-run `git clean -fdn`）

### 真实案例

- **OpenSpec change**: `openspec/changes/docs-cuda-docs-and-openspec-orphan-sync/`
- **实施 commits**:
  - `c913bf3` docs(openspec): fix 3 internal inconsistencies（审查修复）
  - `d80088e` docs(openspec): synthesize 5 retroactive design.md（Phase 2 + 1 个 writing subagent）
  - `9c553bc` docs(skills): remove 4 expired/disabled skill copies（Phase 3 + __pycache__ 清理）
  - `5ffc72f` docs(audits): inline 12 ERRATA markers E1-E8（Phase 4 + 9 edits）
  - `6205a1d` docs(roadmap): mark D-1 + D-3 RESOLVED（Phase 5.6 + 实证对齐）
- **subagent session**: `ses_0c9a60fecffeZQ7lEYex0Du0pz`（Phase 2，6m22s 完成，1 个 writing subagent）
- **关联 lessons**: §6（artifacts git-tracked）+ §20（Pre-impl review）+ §23（artifacts 内部一致性）三者的实施侧补充

### 检查工具

```bash
# 1. Retroactive subagent 决策辅助（共享模板检测）
n=$(ls openspec/changes/<name>/specs/*/spec.md 2>/dev/null | wc -l)
test "$n" -ge 3 && echo "CONSIDER_1_AGENT" || echo "PARALLEL_OK"
# 期望：>=3 共享模板 → 1 个 subagent

# 2. Inline edit 后原文未变检查（diff-based）
diff <(git show HEAD~1:docs/audits/main.md | grep -v '勘误') \
     <(git show HEAD:docs/audits/main.md | grep -v '勘误')
# 期望：无输出（仅有 inline 追加）

# 3. __pycache__ + .gitignore 残留检查
for d in $(find docs/skills -type d -name "__pycache__"); do
    echo "RESIDUAL: $d"
done
# 期望：无输出（Phase 3 后）

# 4. archive 后 git status 验证（Checklist G）
git status openspec/changes/archive/<change-name>/
# 期望：nothing to commit（archive/ 目录内容全部 unchanged）

# 5. delta spec 同步状态（archive 前必跑）
for spec in openspec/changes/<name>/specs/*/; do
    cap=$(basename "$spec")
    test -f "openspec/specs/$cap/spec.md" && echo "SYNCED: $cap" || echo "MISSING: $cap"
done
# 期望：全 SYNCED
```

---

## 25. ANTLR4 lexer 禁止定义 bare string token 与 ID 规则冲突（2026-07 新增）

### 现象

`commit ad808e3` 在 `src/grammar/ptxLexer.g4` line 406-407 新增了 bare string tokens：

```antlr
TCGEN_F16  : 'f16';   // ❌ bare，与 ID 规则冲突
TCGEN_BF16 : 'bf16';  // ❌ bare，与 ID 规则冲突
```

ANTLR4 lexer 平局规则（first-defined-wins）：`f16` 字符串既匹配 `TCGEN_F16`（line 406）又匹配 `ID`（line 512）→ 长度相同，先定义的 `TCGEN_F16` 胜出。结果：`%f16` 寄存器名被错误识别为 `TCGEN_F16` token，触发解析失败：

- `cute_rmsnorm`: `mismatched input 'f16' expecting ID`
- `simpleGEMM-float`: `mismatched input '%' expecting {'.v2', '.v4', ID}`
- 衍生 SEGFAULT: `2Dentropy`, `e2e_blackwell_gemm`, `cute_rmsnorm_debug`

**总影响**：5 个 ctest 失败 + 7 个 tcgen05 fixture LL(*) prediction 失败。

### 教训

- **ANTLR4 lexer 禁止定义 bare string token 与 ID 规则冲突**。任何 `TOKEN : 'bare_string'` 中 `bare_string` 必须**不**匹配 `ID : [a-zA-Z_$][a-zA-Z_0-9$]*`。
- **修复路径（任选其一）**：
  1. 带点前缀（`.f16`）— 复用现有 dot-prefixed `F16` / `BF16` token
  2. 用 lexer mode 隔离上下文 — 进入 `KIND COLONCOLON` 后切换 mode
  3. 删除冗余 token — 让 `ID` 自动接管，parser 用 semantic predicate 验证
- **声称 "X/X PASS" 必须用真实 kernel PTX 验证**。`ad808e3` 声称 "36/36 + 123/123 PASS"，但 `cute_rmsnorm` / `simpleGEMM` 不在测试覆盖 → "自证"测试漏掉了真实场景。修复前应将 bench/ 下的真实 kernel PTX 复制到 `tests/ptx/regression_*.ptx`。
- **一个 lexer 修复可同时解决 Kleene star 预测冲突**：原 `fix-tcgen05-antlr-prediction-bug` WIP change 声称要解决 LL(*) 冲突，但实际根因是 lexer token 抢占 → 5 行 lexer/parser 修复同时恢复 `test_all_ptx.sh` 47/47 PASS（之前 40/47）。

### 检查工具

```bash
# 1. 列出所有 bare string lexer tokens（不带点前缀的 token）
grep -nE "^\w+\s*:\s*'[a-zA-Z]" src/grammar/ptxLexer.g4

# 2. 对每个 bare token，验证其字符串模式是否与 ID 规则冲突
# ID 规则 (line ~512)：[a-zA-Z_$][a-zA-Z_0-9$]*
# 冲突条件：bare token 字符串只含 [a-zA-Z_0-9$]（无 . 或其他非 ID 字符）
grep -A1 "^[A-Z_]\+\s*$" src/grammar/ptxLexer.g4 | grep -E ":\s*'[a-zA-Z_][a-zA-Z_0-9_]*'"

# 3. 实施 ANTLR lexer 修改后必须用真实 kernel PTX 验证
# 复制 bench/cute/*.ptx 到 tests/ptx/regression_*.ptx
cp bench/cute/cute_rmsnorm.ptx tests/ptx/regression_cute_rmsnorm_f16_register.ptx
bash ./tests/ptx/test_all_ptx.sh

# 4. 跑全 ctest 套件确认无回归
cd build && ctest --output-on-failure
```

### 真实案例

- **bug 表现**: 5 个 ctest 失败（`simpleGEMM-float`, `2Dentropy`, `e2e_blackwell_gemm`, `cute_rmsnorm`, `cute_rmsnorm_debug`）+ 7 个 tcgen05 fixture LL(*) 失败
- **根因 commit**: `ad808e3 fix(grammar): resolve tcgen05 LL(*) prediction conflict (ADR-0016, Change-1 MR-3)` 引入 bare `TCGEN_F16` / `TCGEN_BF16` tokens
- **修复**: 5 行 lexer/parser diff（commit `55e216a`）— 删除 bare tokens + parser `tcgen05Qual` 加 `ID` fallback + `tcgen05Dtype` 用 dot-prefixed `F16` / `BF16`
- **测试**: `tests/ptx/regression_cute_rmsnorm_f16_register.ptx`（commit `e92f1c1`，包含 8 个 `%f1N` 寄存器名）→ `./tests/ptx/test_all_ptx.sh` 47/47 PASS
- **教训沉淀**: 本节（§25）+ `.opencode/skills/ptx-lessons-learned/SKILL.md` §核心经验 #9 + Checklist J
- **后续**: `fix-tcgen05-antlr-prediction-bug` WIP change 归档（commit chore(openspec)），因为根因已由更简单的修复解决

---

## 26. tcgen05 5-core-handler 交付 + handler dispatch 修复（2026-07-08 新增）

### 现象

`implement-tcgen05-handlers-core`（commit `df6dde7`，2026-07-07）实现了 5 个 `processTcgen05Xxx` 自由函数（MMA/LD/ST/COMMIT/WAIT），位于 `src/ptxsim/instructions/tcgen05.cpp:311-540`，并通过了 `ctest`（170/170）。但**所有含 `tcgen05.*` 指令的 PTX 实际从未执行 tcgen05 路径**——任何 Blackwell kernel 在 PTX-EMU 中立即崩溃（per-lane EXIT）。

**根因（fix-tcgen05-handler-dispatch 揭示）**:
1. `S_TCGEN05_*` 枚举值定义在 `ptx_types.h:28-38`，**在 X-Macro 循环之后**
2. `ptx_op.def:129-136` 显式注释排除 `S_TCGEN05_*` 在 X-Macro 之外
3. `InstructionFactory::initialize()` 仅从 `ptx_op.def` 注册 handler
4. `grep -rn "processTcgen05" src/ptxsim/ | grep -v tcgen05.cpp` 返回 0 结果
5. `ThreadContext::_execute_once()` 第 143 行 `get_handler` 返回 `nullptr` → `set_state(EXIT)`

**修复（3 commits, 2026-07-08）**:
- `3a30da8` — `ptx_op.def` 恢复 11 个 `S_TCGEN05_*` X-Macro + `Tcgen05PipelineHandler` stub
- `d3afaf5` — `Tcgen05Handler::processTcgen05Operation` 统一 dispatch 入口(`switch` on `instr.op_kind`)
- `cc49ae7` — wire dispatch 测试 + archive change

### 教训

- **功能实现 ≠ 可执行路径**: 写了 handler 函数 ≠ dispatch 管道接好。OpenSpec change 必须明确验收标准为 "PTX 实际执行 handler"，而非 "handler 编译通过"。
- **dead-code coverage test 是警示信号**: `fix-tcgen05-test-coverage-gaps`（`fd74261`）中 5 个测试用 `&ptxsim::processTcgen05Mma` 函数指针直接调用——这是因为 handler **不能从生产路径触达**，否则应该走 `step_warp` 间接调用（per `tests/integration/AGENTS.md` 原则 #3）。
- **跨 change 的 commit 拆分规则**: handler 实施（`df6dde7`）+ dispatch 接入（`cc49ae7`）应该**同一 change**完成。两者拆分为 2 个 change 留下死代码 1 天。
- **X-Macro 注册完整性**: 任何新 enum + handler 必须经过 `InstructionFactory::initialize()` 验证 `get_handler != nullptr`,而非仅看编译通过。
- **OpenSpec change "完成" 定义**: proposal 必须包含 "E2E kernel 执行含此指令" 作为 hard gate,而非仅 "单元测试覆盖"。

### 检查工具

```bash
# 检查 handler 是否真正接入 dispatch
grep -rn "processTcgen05\|processWmma" src/ptxsim/ | grep -v tcgen05.cpp
# 期望：包含 dispatch entry 的引用（如 instruction_factory.cpp）

# 检查 X-Macro 注册
grep "S_TCGEN05_" include/ptx_ir/ptx_op.def
# 期望：11 个 X 条目 + dispatch entry

# 检查 E2E 是否实际执行新指令
ctest -L "e2e;tcgen05" -V 2>&1 | grep -E "PASS|FAIL"
# 期望：PASS（之前为隐式 EXIT,可能假 PASS — 必须看输出 kernel 内容）
```

### 真实案例

- **bug 表现**: `tests/e2e/kernel/test_blackwell_gemm.cu`（含 `tcgen05.*` 指令）在 `df6dde7` 后仍 PASS,但 host 端 reference 对比显示 GEMM 输出未执行 tcgen05 路径(部分输出为 lane EXIT 后的零值)。原因是 `get_handler` 返回 `nullptr`,lance 直接 `set_state(EXIT)` 而非执行。
- **修复**: `Tcgen05Handler::processTcgen05Operation` 统一 dispatch,`fix-tcgen05-test-coverage-gaps` 的 dead-code coverage test 在 dispatch 修好后变为生产路径覆盖测试。
- **教训沉淀**: 本节（§26）+ `.opencode/skills/ptx-lessons-learned/SKILL.md` §核心经验 #10 + Checklist K
- **后续**: `implement-tcgen05-handlers-extended`（6 extended op_kind 实施）按 "handler + dispatch 同 change" 模式

## 27. tcgen05.mma.ws handler via qualifier routing (2026-07-09 新增, Oracle 2026-07-08 review)

### 现象

Phase 3 of `implement-tcgen05-handlers-extended` 计划写 `processTcgen05MmaWs` 独立函数 + 在 `Tcgen05Handler::processTcgen05Operation` dispatch 表加 `case Tcgen05OpKind::MMA_WS`。Oracle review 时发现：

- **grammar 把 `.ws` 当作 `Q_TCGEN_WS` qualifier 在 MMA sub-op 上**（`src/grammar/ptxInstructions.g4:436-447` `tcgen05SubOp` 只有 `MMA`/`LD`/`ST`/.../`FENCE`，**没有 `MMA_WS` sub-op**）
- 真实 PTX `tcgen05.mma.ws.kind::f16.cta_group::1 ...` 解析为 `op_kind=MMA + qualifiers={Q_TCGEN_WS, Q_F16, Q_TCGEN_CTA_GROUP}`
- 我原本计划的 `case MMA_WS:` dispatch **永远不会从真实 PTX 进**,只能由测试手动构造 `Tcgen05Instr{op_kind=MMA_WS, ...}` 触达（dead path）
- spec.md `Scenario: weight-stationary mma.ws handler` 用了 `.warpspecialized::1` 词汇,但 grammar 只有 `.ws` 裸 token,**spec 词汇与 grammar 脱节**
- `Tcgen05Instr` 便捷字段（`cta_group`/`dtype`/`num_regs`/`has_block_scale`）在 `ptx_visitor.cpp:841-885` `visitTcgen05Inst` 中**根本没被填充**,默认值永远生效

### 教训

- **dispatch case 写之前必 grep grammar 确认 sub-op 真存在**: `grep -nE "tcgen05SubOp|MMA_WS" src/grammar/ptxInstructions.g4`。如果 grammar 只有修饰符语法（如 `.ws` 是 qualifier），dispatch case 就是 dead path — 必须改用 handler 内部 qualifier scan + 路由
- **Spec/Design 词汇脱节会污染所有下游文档**: spec.md 写了 `.warpspecialized::1`，design.md/tasks.md/AGENTS.md 全部抄过去，但 grammar 永远不会产生这个 token。设计阶段必跑 `grep -nE "warpspecialized|TCGEN_WARPSPECIALIZED" src/grammar/` 验证词汇对齐
- **IR 便捷字段是承诺 ≠ 实现**: `Tcgen05Instr::cta_group` 等字段在 `ptx_visitor.cpp` 中从未被赋值（visitor 只填 `op_kind/qualifiers/operands/instructionText`），所以 handler 检查 `instr.cta_group == 1` 永远成立。Handler 检查便捷字段前必 grep visitor 验证提取路径，否则改用 `instr.qualifiers` 扫描对应 token
- **Oracle review 是发现死代码路径的关键**: 自我审查看不到 dispatch case 是否真的可达（看上去语法、编译、ctest 都对），Oracle 的"真 PTX 走哪条路"反向验证才能识破

### 检查工具

```bash
# 1. 检查 grammar sub-op 是否真存在
grep -nE "tcgen05SubOp|MMA_WS" src/grammar/ptxInstructions.g4
# 期望：MMA_WS 列出 OR 显式不在

# 2. 检查 spec 词汇是否在 grammar 中存在
grep -nE "warpspecialized|TCGEN_WARPSPECIALIZED" src/grammar/
# 期望：空（如果 spec 用了 .warpspecialized::N 但 grammar 只有 .ws 裸 token,脱节）

# 3. 检查 IR 便捷字段是否真被 visitor 填充
grep -nE "instr\.cta_group|instr\.dtype|instr\.num_regs|instr\.has_block_scale" src/ptx_parser/
# 期望：assignments (如果不是,字段就是默认值 — handler 检查无效)

# 4. 验证 handler dispatch 真可达
ctest -R "<handler_test>" -V 2>&1 | grep -E "case.*MMA_WS|op_kind"
# 期望：dispatch trace 显示 op_kind=MMA(非 MMA_WS)走 handler
```

### 真实案例

- **bug 表现**: `processTcgen05Mma` 计划加 `case MMA_WS:` 独立 dispatch 分支,但 grammar `ptxInstructions.g4:436-447` 的 `tcgen05SubOp` 只列出 10 个 sub-op,**没有 `MMA_WS`**。`ptx_visitor.cpp:846` `if (ctx->tcgen05SubOp()->MMA()) op_kind = Tcgen05OpKind::MMA;` — `.ws` 被当作 qualifier 消耗掉,op_kind 永远是 MMA。
- **修复**: Phase 3 commit `f4b6d58` 删除 `case MMA_WS:` 独立 throw,改为在 `processTcgen05Mma` 内部 scan `instr.qualifiers` for `Q_TCGEN_WS`,Q3-A 范围检查（要求 `Q_F16` 必备），然后调 `tcgen05_fragment_mma_f16` helper（Phase 2.5 抽出,DRY）。
- **设计文档同步**: spec.md Scenario 改写为 qualifier-based routing 描述；design.md D3 加 "Phase 3 实施修订" 注释解释 grammar 现实与 spec 假设的脱节；tasks.md §4 全部标记 `[x]` 含 Oracle A-path 决策
- **教训沉淀**: 本节（§27）+ `.opencode/skills/ptx-lessons-learned/SKILL.md` §失败模式速查表 3 行（dispatch dead path / spec vocab desync / IR field unwired）
- **后续**: Phase 4 fence + Phase 5 doc sync + Phase 6 archive 按 Oracle A-path 模式继续

---

## 28. Helper 累加器 "single-warp execution" 是脆弱假设 (2026-07-11)

### 现象

`fix-tcgen05-mma-accumulator-and-f32-storage`（commits `d3be589`/`df1f6de`/`f97863c`，2026-07-11）新增 `accumulate` 参数到 `tcgen05_fragment_mma_f16` 时，Oracle 2026-07-11 审计发现 **C4 BLOCKER**：`c_slot = 64 + lane_id` 硬编码假设调用方保证 single-warp 执行（per SM scheduler sequential）。FlashAttention 多 warp 协作时 warp 0 和 warp 1 都写 slot 64 → 数据竞争。

Helper header `tcgen05_helpers.h` 无任何 "single-warp assumption" 标注。

### 教训

- **"Currently safe because SM scheduler runs one warp at a time" 这种注释是已知 debt 的标记**，必须在 helper header 显式标 `[SINGLE-WARP ASSUMPTION]`
- 新增累加路径时必须同时考虑多 warp 影响：要么扩展 helper 接受 `warp_id` 参数，要么显式拒绝多 warp（throw）
- 单元测试用 `SMContext(1 warp, 32, 1 cta)` 配置是 single-warp 测试，多 warp 必须独立测试

### 检查工具

```bash
# 找出所有 "single-warp" / "one warp at a time" 注释
grep -rn "single-warp\|one warp at a time\|sequential execution" src/ include/
```

### 修复模板

见 FU-4 `fix-tcgen05-multi-warp-fragment` — `c_slot = warp_id * 32 + 64 + lane_id`。

### 真实案例

`fix-tcgen05-mma-accumulator-and-f32-storage` Oracle 2026-07-11 审计 C4 BLOCKER。

---

## 29. TcQueue wait() 必须先检查 commit_group_counter (2026-07-11)

### 现象

`fix-tcgen05-mma-accumulator-and-f32-storage` Phase 1 B2 integration test `commit_wait_sequence` 首次运行时，`tc_queue().pending_count() == 0` 断言在 `commit(1)` → `wait(warp, 0, 1)` 序列后失败。排查发现 `TcQueue::wait()` 实现顺序问题：

```cpp
// src/ptxsim/async/tc_queue.cpp — 当前实现
void wait(WarpContext* warp, lane_id_t lane_id, group_id_t group_id) {
    pending_waiters_.push_back({warp, lane_id, group_id, completion_pc_++});  // ❌ 先 push
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this, group_id] { return commit_group_counter_ >= group_id; });  // 再 check
    // → waiter 一直 pending，即使 counter 已满足
}
```

### 教训

- TcQueue 状态机设计: commit bumps counter, wait checks counter — 但 wait 必须**先** check counter 再 push，否则 false positive in `pending_count()`
- Integration test 第一次跑 B2 commit/wait 序列时 `pending_count() == 0` 断言失败 → root cause 是 wait() 实现顺序问题
- 本 change 不修 TcQueue 本身（属于 FU-1 scope），但 B2 test 必须调整 assertion 适应现状

### 检查工具

```bash
# 看 TcQueue::wait 实现
grep -n "wait\|pending_waiters_" src/ptxsim/async/tc_queue.cpp
# 找所有 commit/wait 序列测试
grep -rn "tc_queue().commit\|tc_queue().wait" tests/
```

### 修复模板 (FU-1 `fix-tcgen05-commit-wait-group` 范围)

```cpp
// AFTER (correct): wait 先 check counter 再 push
void wait(WarpContext* warp, lane_id_t lane_id, group_id_t group_id) {
    {
        std::lock_guard lock(mutex_);
        if (commit_group_counter_ >= group_id) return;  // already satisfied
        pending_waiters_.push_back({warp, lane_id, group_id, completion_pc_++});
    }
    std::unique_lock lock(mutex_);
    cv_.wait(lock, [this, group_id] { return commit_group_counter_ >= group_id; });
}
```

### 真实案例

`fix-tcgen05-mma-accumulator-and-f32-storage` Phase 1 B2 test 暴露。

---

## 30. PTX §9.7.16 `f16×f16→f32` 不变量 — storage format 必须硬件对齐 (2026-07-11)

### 现象

`fix-tcgen05-mma-accumulator-and-f32-storage` Phase 2（commit `f97863c`）将 `c_frag` 类型从 `uint16_t` 改为 `float` 时，**所有 readback site 未同步迁移**。`commit_wait_sequence` 测试自己的 `require_c_slot_matches` 函数仍用 f16 pattern 读取 f32 storage，导致：

- f32 bits 被 reinterpret_cast 为 f16 → 垃圾值（silent corruption，无 assertion 失败）
- `Catch::Approx` 默认 epsilon (1.19e-5) 不敏感 — expected 和 actual 都是同样的 garbage

**根因**：Golden header `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6` 注释声明 "32 f32 elements" 但实际 storage 是 f16，readback 用 `f16_to_f32` 转换掩盖了不一致。

### 教训

- **Helper 输出 dtype 是 hardware contract**，golden value header 注释必须**显式声明 storage format**（不是仅声明 output dtype）
- `grep "c_buf[idx * 2]" tests/` 是 readback 残留的快速检测 — f16 readback 模式是 `idx * 2`（一对 f16 元素），f32 readback 模式是 `memcpy(c_arr, ...)`
- `Catch::Approx` 默认 epsilon (1.19e-5) 对 single op 足够，但 storage format 错误时**assertion 仍通过**（因为 expected 和 actual 都是同样的 garbage）

### 检查工具

```bash
# 1. helper 类型变更后必跑 grep 验证 readback 全部迁移
grep -rn "f16_to_f32\|c_buf\[idx \* 2\]" tests/integration/tcgen05/
# 2. 验证 helper body 内部无 f32_to_f16 残留
grep -n "f32_to_f16" src/ptxsim/instructions/tcgen05_helpers.cpp
```

### 真实案例

`fix-tcgen05-mma-accumulator-and-f32-storage` Phase 2 — `commit_wait_sequence.cpp` 自己的 `require_c_slot_matches` 漏改 readback，被 ctest 捕获（5 个测试发现 2 个失败）。

---

## 31. ANTLR `extractQualifiersFromContext` 丢失 IMMEDIATE 值 (2026-07-11)

### 现象

`fix-tcgen05-mma-accumulator-and-f32-storage` Oracle 2026-07-11 审计时发现 **C3 BLOCKER**：所有 commit/wait 调用硬编码 `commit(1)` / `wait(warp, 0, 1)`。根因是 `extractQualifiersFromContext`（`src/ptx_parser/ptx_visitor.cpp:155-183`）遍历 parse tree 时只把 terminal token 映射到 `Qualifier` enum 值，**`IMMEDIATE` 节点被 `tokenToQualifier` 返回 `Q_UNKNOWN` 后静默丢弃**。结果：`.cta_group::N` 的 `::N` 值永远进不了 `Tcgen05Instr.cta_group` 字段（defaults to 1），handler 全部用硬编码 `1`。

**影响放大**：该函数被 **19 个 call sites** 调用（`ptx_visitor_atom.cpp:81`, `ptx_visitor_branch.cpp:30`, `ptx_visitor_barrier.cpp:86,97`, `ptx_visitor_call.cpp:23,44`, `ptx_visitor_generic.cpp:14`, `ptx_visitor_memory.cpp:16,29,42,55,68`, `ptx_visitor_special.cpp:16,24,32,45,58`, `ptx_visitor_warp.cpp:24,46`, `ptx_visitor.cpp:858`）。改返回类型会破坏所有 caller。

### 教训

- 19 个 call sites 共享此函数是因为它们只需要 enum 值；需要 IMMEDIATE 的 caller（commit/wait/lane_id 等）必须**单独 walk parse tree**
- 这种"通用 helper 丢失上下文信息"模式是 ANTLR visitor 实现的常见 trap — 一个"便利函数"因为被太多 caller 共享而无法添加新功能
- `Tcgen05Instr` 便捷字段（`cta_group`/`dtype`/`num_regs`/`has_block_scale`）全是默认值 — 这是 §27 的延伸案例

### 检查工具

```bash
# 1. 列出所有 Q_UNKNOWN 位置（IMMEDIATE 被丢弃的证据）
grep -n "Q_UNKNOWN" src/ptx_parser/ptx_visitor.cpp
# 2. 检查所有 extractQualifiersFromContext call sites
grep -rn "extractQualifiersFromContext" src/
# 3. 检查 Tcgen05Instr 便捷字段哪些从未被填充
grep -n "cta_group\|num_regs\|has_block_scale" src/ptx_parser/ptx_visitor.cpp
```

### 修复模板 (FU-1 `fix-tcgen05-commit-wait-group` 范围)

```cpp
// Option (b) 推荐: 单独 walk, 不改 extractQualifiersFromContext 返回类型
// 在 visitTcgen05Inst 内, extractQualifiersFromContext 调用之后:
uint32_t cta_group = 1;  // default per statement_context.h:186
if (ctx->tcgen05QualList()) {
    for (auto* qualCtx : ctx->tcgen05QualList()->tcgen05Qual()) {
        if (qualCtx->TCGEN_CTA_GROUP() && qualCtx->IMMEDIATE()) {
            cta_group = static_cast<uint32_t>(
                std::stoul(qualCtx->IMMEDIATE()->getText()));
        }
    }
}
```

### 真实案例

`fix-tcgen05-mma-accumulator-and-f32-storage` Oracle 2026-07-11 审计 C3 BLOCKER（commit/wait 硬编码 group_id=1）+ FU-1 follow-up。

---

## 32. CppTLM D1-Full MemoryBridge 集成经验 (2026-07-15)

### 背景

`cpptlm-d1-full` change 实现 CppTLM F12b-LD MemoryBridge 集成（ADR-0021），涉及 6 项架构决策（D-PTX-1~6）+ 3 个握手信号（HSK-1/2/3）+ 5 个 Phase 代码实施。

### 教训

1. **类型定义冲突**：`cpptlm_bridge.h` 定义 `cudaStream_t = struct CUstream_st*` 与 `cudart_intrinsics.h` 的 `typedef void* cudaStream_t` 冲突。修复：使用 `#if defined(__CUDACC_RUNTIME_H__)` 条件编译 + `#elif !defined(CUDA_STREAM_T_DEFINED)` fallback 到 `void*`，与 `cudart_intrinsics.h` 保持一致。

2. **迭代器失效修复**：`cudaStreamSynchronize` / `cudaDeviceSynchronize` 中 range-for 遍历 `g_pending_kernels` 时调用 `erase()` 会触发 UB。修复模式：先收集 `completed_ids` 到 vector，循环外统一 `erase()`。

3. **ExternalProject_Add vs add_subdirectory**：HSK-3 选项 1 选择 `ExternalProject_Add` 直接拉取 CppTLM 仓库，而非本地 `add_subdirectory`。优点：版本 pin（`GIT_TAG`）+ build 隔离 + 零 ABI 漂移。缺点：首次 build 需网络访问。

4. **nullptr fallback 字节级兼容**：`g_cpptlm_bridge == nullptr` 时所有操作必须走原有同步路径（字节级相同）。这是 ADR-0021 D-PTX-1 的核心约束，确保现有 600+ 测试零回归。

### 检查工具

```bash
# 1. 验证 bridge 全局指针状态
grep -n "g_cpptlm_bridge" src/cudart/cudart_sim.cpp

# 2. 检查迭代器安全模式
grep -n "completed_ids" src/cudart/cudart_sim.cpp

# 3. 验证 CMake ExternalProject_Add 配置
grep -n "ExternalProject_Add" CMakeLists.txt
```

### 真实案例

`cpptlm-d1-full` change Phase 2-7 实施（commits `f5ddb618` → `7724dc7f`），4 个测试文件 26 assertions 全 PASS。

---

## 33. ABI 头声明 ↔ .cpp 实现必须配对提交（2026-07-16）

### 现象

`cpptlm-d1-full` change Phase 1 commit（`9be56f8f`）将 `cpptlm_bridge.h` 落地为 161 行完整头文件，包含 5 个纯虚方法 ABI 声明 + `CPPTLMBRIDGE_VERSION=1` 宏 + `static_assert(sizeof(cudaStream_t))` + `g_cpptlm_bridge` `extern` 声明。**但 `cpptlm_attach_bridge(CppTLMBridge*)` 和 `cpptlm_detach_bridge()` 两个 ABI 入口函数只有声明、没有定义**。CppTLM 加载 `libcpptlm_cudart.so` 调用这两个符号时链接期 undefined reference，导致 F12b-LD 集成无法实际启用。

**Metis 二审**（2026-07-16 第一轮）发现此问题为 `❌ NO-GO BLOCKER (B1)`。

### 教训

- **"header 已 commit" ≠ "ABI 可用"**：只声明函数签名而不在同 PR 提供实现 = 链路口号承诺。运行 `nm -D build/lib/libcudart.so | grep <symbol>` 是验证 ABI 真值源活性的**唯一**手段
- **跨 so 可见性必须在 ABI 头文件同步规定**：仅 `extern` 不够，必须配 `__attribute__((visibility("default")))` 或 MSVC `__declspec(dllexport)` 宏，否则 `libcpptlm_cudart.so` 无法定位 PTX-EMU libcudart.so 中的符号
- **ABI 头文件声明 + 源文件实现应视为原子单元**：拆为两 commit 是隐式危害，必须在同一 Phase 同时落地或显式声明 ABI 暂时不可用
- **"first-cuda-call 懒初始化" 模式不替代"符号必须存在"**：D-PTX-1 选择静态指针 + 懒初始化仅指**赋值时机**，不指**符号存在性**

### 检查工具

```bash
# 1. 列出 ABI 头声明的所有 extern "C" / 跨 so 入口
grep -nE "extern.*PTXEMU_BRIDGE_API|extern \"C\"" include/cudart/cpptlm_bridge.h

# 2. 验证每个入口在 build 的 .so 中已导出符号（活函数，非死声明）
nm -D build/lib/libcudart.so | grep cpptlm_attach_bridge
nm -D build/lib/libcudart.so | grep cpptlm_detach_bridge
# 期望：T cpptlm_attach_bridge + T cpptlm_detach_bridge（非 "U" 未定义）

# 3. ABI 头声明与 .cpp 实现配对验证（手工或 grep）
for sym in $(grep -E "^extern.*PTXEMU_BRIDGE_API|extern \"C\"" include/cudart/cpptlm_bridge.h | grep -oE "[a-z_]+ ?\(?[a-zA-Z_*]*\)?$" | sort -u); do
  echo "Checking symbol: $sym"
  grep -rn "$sym" src/ || echo "  ⚠️ WARNING: declaration without definition"
done

# 4. 编译期测试：尝试 link 一个最小 libcpptlm_cudart.so consumer
g++ -shared -fPIC -xc++ - -olibtest_consumer.so <<< '
extern "C" void cpptlm_attach_bridge(void*);
int main(){ cpptlm_attach_bridge(nullptr); return 0; }
'
# 期望：链接 PASS（无 undefined reference）
```

### 真实案例

`fix(cpptlm-d1-full/cudart_sim): implement cpptlm_attach_bridge + cpptlm_detach_bridge (B1)` commit `de016f79` — Metis 二审找出 B1 后修复。新增 `tests/unit/cpptlm/test_cpptlm_attach_bridge.cpp`（3 个测试）TDD：先 RED（编译期 undefined reference），后 GREEN（链接期符号导出）。`nm -D` 验证 `00000000000d8390 T cpptlm_attach_bridge` + `00000000000d8560 T cpptlm_detach_bridge`。

---

## 34. uintptr_t reinterpret_cast 编码陷阱：stream handle 不能 `delete`（2026-07-16）

### 现象

`cpptlm-d1-full` change commit `44b54cf2` 实现 `cudaStreamCreate/Destroy`：

```cpp
cudaError_t cudaStreamCreate(cudaStream_t *stream) {
    uint64_t stream_id = generate_kernel_id();
    g_active_streams.insert(stream_id);
    *stream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(stream_id));  // 把整数编进指针类型
}

cudaError_t cudaStreamDestroy(cudaStream_t stream) {
    if (stream) {
        delete reinterpret_cast<int *>(stream);  // ⚠️ UB：stream 是 uint64_t，不是堆指针！
    }
    return cudaSuccess;
}
```

`unit_cuda_stream_handle` 测试触发了 SIGSEGV。**根因**：把整数 reinterpret_cast 为指针后，`delete` 操作假定存在堆分配 — 这是 C++ UB。**同时**该函数缺少 `g_active_streams.erase(stream_id)`，导致 `cudaStreamCreate` + `cudaStreamDestroy` 反复调用时 `g_active_streams` 单调增长（泄漏）。

### 教训

- **整数 ↔ 指针 reinterpret 是 encoding，不是 allocation**：`*out = reinterpret_cast<T*>(static_cast<uintptr_t>(id))` 仅在**调用方**字段能容纳 `T*` 时合法，**不**代表 `out` 拥有堆分配
- **delete-from-pointer 假设永远不成立 for encoded handles**：任何形如 `delete reinterpret_cast<int*>(uintptr_value);` 都是 UB，应该 grep 禁止
- **`Create` + `Destroy` 必须对称清理所有记账资源**：`g_active_streams.insert(...)` 必须在 `Destroy` 中 `erase`；审计对称性是 `state-modification-audit` skill 的核心
- **测试覆盖 SIGSEGV 路径**：单元测试必须包含反向调用（创建后销毁）才能捕获"无清理"的隐式内存泄漏
- **AGENTS.md cudart 章节应明文禁止"stream handle 即指针"反模式**

### 检查工具

```bash
# 1. grep 所有 delete-from-pointer 反模式（必须为空，否则命中 UB）
grep -rnE "delete reinterpret_cast<.*>\(" src/cudart/

# 2. 列出所有 Create/Destroy 对的资源清理对称性
for create_fn in $(grep -lE "cuda.*Create\(" src/cudart/cudart_sim.cpp); do
  echo "=== $create_fn ==="
  create_body=$(sed -n '/cuda.*Create(/,/^}/p' "$create_fn")
  # 提取 insert/insert/add/add/increment
  echo "$create_body" | grep -E "\.(insert|push|emplace|add|increment)" || echo "  (no clear insert)"
done

# 3. 验证 g_active_streams 在 Create 和 Destroy 中严格对称
grep -n "g_active_streams" src/cudart/cudart_sim.cpp
# 期望：insert + erase 数量相同

# 4. 运行单元测试 under AddressSanitizer 捕获 UB
cmake -S . -B build_asan -DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS="-fsanitize=address -fno-omit-frame-pointer"
cmake --build build_asan --target cudart
cd build_asan && ctest -L "unit;cudart" --output-on-failure
```

### 真实案例

`fix(cpptlm-d1-full/cudart_sim): cudaStreamDestroy — remove UB delete + add g_active_streams cleanup (B3)` commit `6cbdcc4c` — 修复后 `unit_cuda_stream_handle` PASS。修复后 `cudaStreamDestroy` 仅做 `g_active_streams.erase(stream_id)`，复用 `g_pending_kernels_mutex` 的 mutex。

---

## 35. 同步语义 ≠ 单次 poll：`cudaStreamSynchronize` 必须阻塞到完成（2026-07-16）

### 现象

`cudaStreamSynchronize` 第一个 bridge-aware 实现（commit `44b54cf2`）：

```cpp
if (g_cpptlm_bridge) {
    std::vector<uint64_t> completed_ids;
    for (const auto& [id, pk] : g_pending_kernels) {
        if (pk.stream_id == stream_id && !pk.completed) {
            uint64_t remaining = g_cpptlm_bridge->poll_kernel(id);
            if (remaining == 0) completed_ids.push_back(id);
        }
    }
    // erase...
    return cudaSuccess;  // ⚠️ 即使有 remaining > 0 的 kernel 也立即返回！
}
```

**CUDA 语义约束**：`cudaStreamSynchronize` 必须阻塞直到 stream 上**所有 work**完成。单次 poll 仅标记"现在完成"的 kernel，对 `remaining > 0` 的正在执行 kernel 直接放弃。

### 教训

- **"scan + return" 实现是"主动监测"语义，与 CUDA sync 不兼容**：实现同步函数时必须严格查规范 — 是要求"立即返回当前状态"还是"阻塞到完成"
- **CUDA 同步原语是一个 LOOP，不是 snapshot**：任何 `cudaStreamSynchronize` / `cudaDeviceSynchronize` / `cudaEventSynchronize` 必须含 `while` 或 `wait`
- **poll kernel 返回值 3 类语义必须全分支**：`remaining == 0` = 完成；`remaining > 0` = 等待中（继续 poll）；`UINT64_MAX` = 未知 kernel_id（spec 决策 = erase）
- **空挂起检测是 while 循环的退出条件**：内层 `g_pending_kernels.empty() || stream_has_pending == false` 都必须 explicit 检查，避免死循环
- **`std::this_thread::yield()` 防忙转**：while 循环内对 host 单线程 PTX-EMU 必须主动让出，否则 CPU 100%

### 检查工具

```bash
# 1. 列出所有 cudaSync 函数实现，必须包含 while(true) 或 wait
grep -nE "cudaStreamSynchronize|cudaDeviceSynchronize|cudaEventSynchronize" src/cudart/cudart_sim.cpp
# 对每个函数 grep 内层循环：
for fn in $(grep -E "cuda.*Synchronize\(" src/cudart/cudart_sim.cpp | grep -oE "cuda[A-Za-z]+"); do
  echo "=== $fn ==="
  grep -A30 "^cudaError_t $fn" src/cudart/cudart_sim.cpp | grep -E "while|for.*pending"
done
# 期望：每个 sync 函数体内有 while 或 for 遍历 pending kernels

# 2. 验证 UINT64_MAX sentinel 处理
grep -B2 -A2 "UINT64_MAX" src/cudart/cudart_sim.cpp
# 期望：poll_kernel 返回后分支处理 UINT64_MAX 与 0

# 3. 测试用 completion fence 验证不返回过早
# 模板：在 poll 返回值固定 large number 时，sync 必须阻塞直到 bridge 更新
```

### 真实案例

`fix(cpptlm-d1-full/cudart_sim): cudaStreamSynchronize/DeviceSynchronize real polling loop (B2)` commit `5dcccf40` — 单次 poll 替换为 `while (true) { ... }`。新增 `tests/unit/cudart/test_stream_sync_loop.cpp` 4 个契约测试（空 map 退出、不再 poll 完成的 kernel、`remaining > 0` 时继续循环）。

---

## 36. HSK 文档状态必须与 ADR 生命周期同步（2026-07-16）

### 现象

`cpptlm-d1-full` change 在 OpenSpec artifacts `hsk-1.md` / `hsk-2.md` / `hsk-3.md` 中：

- hsk-1.md page top: `> **状态**: ✅ **已发出**（commit hash 锁定，待 CppTLM 团队 rebase 确认）`
- hsk-1.md footer (line 148-149): `**HSK-1 commit `8dc000ec` 已 push 到 origin main，可立即发送**`
- hsk-2.md: `> **状态**: ✅ **可立即发送**` + footer 同样声称已 push
- hsk-3.md: 状态 `✅ 可立即发送` 但 `CPPTLM_COMMIT_HASH` 仍 `TBD`
- tasks.md 验收标准: `3 个 Handshake（HSK-1/2/3）已发出`

**真实状态**：ADR-0021 当前是 `Proposed`；OpenSpec 14/68 任务未完成；`8dc000ec` 是 history commit hash 而非 current commit hash；`origin main` 没有这些 commit（仅本地 12 commits ahead）。

3 份 OpenSpec artifact 互相 + 与 ADR 状态 + 与 git 现状矛盾，违反 **Checkpoint J（artifacts 内部一致性）**。

### 教训

- **HSK 状态必须与 ADR lifecycle 绑死**：ADR Proposed 期间禁止任何 artifact 声称"已发出"；ADR Active 后才逐步发送 HSK-1/2/3
- **OpenSpec artifact 顶部 + 底部状态必须一致**：page-top 状态 + footer claim + 验收标准三处任一不一致 = artifact 不可信
- **fictional commit hash 必须替换为 `TBD (实际发出时锁定)`**：未真实发生的 commit 不允许写具体 hash
- **Checklist J 是 artifacts 完整性的最低线**：4+ 个 OpenSpec artifact 任一项不一致 → 整个 change 必须返工
- **HSK 状态机应在 ADR 主文中明文定义**：ADR `§HSK 状态机` 强制表列出 HSK-1/2/3 触发时机 + 验证命令 + 禁止在 ADR 状态 < Active 时发出的硬约束

### 检查工具

```bash
# 1. 检查 OpenSpec artifact 顶部 vs 底部状态一致性
for f in openspec/changes/<change>/*.md; do
  top=$(grep -E "^> \*\*状态" "$f" | head -1)
  bottom=$(grep -E "已发出|已 push|可立即发送" "$f" | tail -3)
  if [ -n "$bottom" ]; then
    echo "=== $f ==="
    echo "top:    $top"
    echo "bottom: $bottom"
  fi
done
# 期望：无输出（所有 artifact 状态统一）

# 2. 验证 ADR 当前 lifecycle vs artifacts 声称
grep -n "状态" docs/adr/<adr>.md | head -1
grep -nE "已发出|可立即发送" openspec/changes/<change>/hsk-*.md
# 期望：ADR Proposed 时无任何"已发出"声称

# 3. 验证所有 fictional commit hash 都被 TBD 替换
for f in openspec/changes/<change>/*.md; do
  git_hash=$(grep -oE "[a-f0-9]{40}" "$f" | head -1)
  if [ -n "$git_hash" ]; then
    if ! git cat-file -t "$git_hash" 2>/dev/null | grep -q commit; then
      echo "⚠️ $f 引用 fictional commit $git_hash"
    fi
  fi
done
# 期望：无 warning
```

### 真实案例

`fix(cpptlm-d1-full): unify HSK status in hsk-1.md footer + tasks.md 验收 (B4)` commit `c38c31e4` + 同期 `77302f0b`（hsk-2 footer 一致化）。统一为：

> **状态**: ⏳ **待发出（ADR Accepted 后启用）**

并新增 `§HSK 状态机` 在 ADR-0021，列出 HSK-1/2/3 触发时机与验证命令。

---

## 37. 文档 vs 实现发散：git log -- <file> 是 source of truth（2026-07-16）

### 现象

`cpptlm-d1-full` change `design.md §7.1`（lines 307-317）与 `spec.md` 描述 CMake 集成：

> `find_package(cpptlm) + add_subdirectory(src/cudart/cpptlm_bridge) + target_link_libraries(ptxemu_runtime PRIVATE cpptlm::core)`

但实际 `CMakeLists.txt:122-152`（commit `d0803a09`）使用：

```
ExternalProject_Add(cpptlm ...)
...
target_link_libraries(cudart PRIVATE <cpptlm_targets>)
```

实施 commit `d0803a09` 后文档未同步，5+ 天过去没人发现 — 因为新开发者读 `design.md` 会照搬 find_package 写法而编译失败。

### 教训

- **git log -- <file> 是 source of truth**：每次修改 CMakeLists.txt / 源文件时，必须同步 grep 引用该文件的 ADR / design / spec
- **CMake 是 OpenSpec change 的"无声发散点"**：CMakeLists.txt 改动往往不被 OpenSpec "Capability" 列表捕获，但 ABI 行为完全由它定义
- **Checklist J 强调 design ↔ spec 范围路径必须一致**：包括"路径示例"（`$d/design.md` vs `$d.design.md`）和"操作语义"（"添加 README" vs "git hash 不变"）
- **design.md 中的"完整实现示例"必须 bit-equal 实际 commit**：`target_link_libraries(... cpptlm::core)` 这种具体 target 名错误是**最常见的 ABI 协作失败模式**
- **设置 doc-drift 守护**：考虑 `pre-commit` hook 验证 `design.md` / `spec.md` 中任何 "Example CMake invocation" 必须与 `CMakeLists.txt` 在 grep 下正则匹配

### 检查工具

```bash
# 1. 列出所有引用 CMake target / function 的 OpenSpec artifact，验证 vs 实际 CMakeLists.txt
grep -rnE "find_package\(|add_subdirectory\(|ExternalProject_Add|target_link_libraries" openspec/changes/ | grep -E "\.md:"
# 与 CMakeLists.txt grep 对比
grep -nE "find_package\(|add_subdirectory\(|ExternalProject_Add" CMakeLists.txt
# 不一致 → Checkpoint J 失败

# 2. 当 commit 改 CMakeLists.txt 时必同步改 OpenSpec artifact（pre-commit hook）
# .git/hooks/pre-commit 草稿：
# ! git diff --cached --name-only | grep -q CMakeLists.txt && {
#     git diff --cached --name-only | grep -q "openspec/changes/" || {
#       echo "ERROR: CMakeLists.txt modified but no OpenSpec artifact updated"
#       exit 1
#     }
#   }

# 3. 验证 ADR "实施时间表" 中的 commit hash 必须真实存在 git
for h in $(grep -oE "[a-f0-9]{8,12}" docs/adr/<adr>.md); do
  git cat-file -t "$h" >/dev/null 2>&1 || echo "⚠️ fictional: $h"
done
```

### 真实案例

`fix(cpptlm-d1-full): align CMake docs with actual ExternalProject_Add approach (B5)` commit `0456418e` — 更新 `design.md §7.1` 和 `spec.md` 场景，与 commit `d0803a09` 的实际 `CMakeLists.txt` 实现一致。修复后 `grep "ExternalProject_Add" CMakeLists.txt` × 3、`design.md` × 6、`spec.md` × 5 完全匹配。

---

## 元教训：本次 cpptlm-d1-full 变更的 5 条 anti-pattern 综合教训

### 1. "Implementation Done" 不等于 "Change Done"
- 55/68 任务 `[x]` 已勾选 + 11 commits 已 push 不代表 change 完成
- **强制钩子**：每次 OpenSpec change 实施前跑 Metis - Plan Consultant 子代理（`ptx-lessons-learned §7`）

### 2. "两轮 Metis review 才看到 code-level blocker"
- 第一轮发现 25+ doc-level 错误（A1/A2/A4/S1/HSK-I 等）
- 第二轮才发现 code-level 阻塞（B1-B5）
- **强制钩子**：Metis 二审时**必须**通过 `nm -D build/lib/lib<lib>.so | grep <symbol>` 验证 ABI 真值源活性；必须读 `*.cpp` 而非仅 ADR

### 3. "OpenSpec artifact 状态机脱钩"
- tasks.md `[x]` 是"完成"语义，OpenSpec status 是另一字段
- **强制钩子**：每次 `[x]` 操作跑 `openspec validate --change <name>` 反查

### 4. "Documentation 延迟更新" 是技术债务
- design.md vs 实际 CMakeLists.txt 5+ 天发散没人发现
- **强制钩子**：每次 commit `CMakeLists.txt` 必同步 `openspec/changes/<current>/design.md`（可 pre-commit 守护）

### 5. "Pre-impl review" 应以"concrete symbols" 为单位
- 文件存在 ≠ 符号可链
- **强制钩子**：每次修改 ABI 头文件 = (header 改 + source 改 + `nm -D` 验证 + symbol 列表进 lessons-learned) 4 件套

---

**本批 §33-37 更新日期**: 2026-07-16  
**关联 commit hash**: de016f79 / 6cbdcc4c / 5dcccf40 / c38c31e4 / 0456418e / 88d5962e  
**ADR postmortem ref**: docs/adr/ADR-0021-cpptlm-d1-full-integration.md §2026-07-16 Postmortem  
**OpenSpec change ref**: openspec/changes/cpptlm-d1-full/

---

## 38. "byte-identical fallback" 契约必须由测试锁定（2026-07-17）

### 现象

Commit `367fd6a5`（feat: exe_once 3-step injection）引入 `step_b_set_blocked_cycles`，注释写 "All three nullptr = use existing InstructionLatencyTable only"，commit message 声称 "0 regression"。下一个 commit `5b292a91`（fix）发现 `unit_simt_integration` 2 个断言失败：
- `pc_groups_after == 1 got 0`
- `threads[0].pc == 1 got 0`

### Root Cause

`step_b_set_blocked_cycles` 在两个 injector 都为 `nullptr` 时**不是 no-op**，而是 fallthrough 到 `ptxsim::getLatency()` + `set_blocked_cycles_for_active()`。这是 `exe_once()` 历史上从未有过的行为--`blocked_cycles` 设置此前仅在 `LdHandler` 路径（`memory.cpp:47,71,139`）中出现。

注释承诺 no-op，实现未兑现。fallthrough 也是调用，"代码看起来没显式调用" ≠ no-op。

### 教训

1. **"byte-identical fallback" / "nullptr = no-op" 契约必须由直接单元测试锁定**，不能仅靠注释
2. **fallthrough 也是调用** -- "代码看起来没显式调用" ≠ no-op
3. **注释承诺 no-op 时，必须明确列出"pre-change 路径中该状态变量由谁设置"**（否则注释本身就是错的，如本案注释说"use existing InstructionLatencyTable only"，但 pre-change `exe_once()` 根本没用过 InstructionLatencyTable）
4. **匿名命名空间内的 file-local 函数无法直接测试** -> 需提取为 public static 方法（与 `is_tensor_core_instruction` 等 helper 一致的可测试性先例）
5. **commit message 声称 "0 regression" 不等于真 0 回归** -- 必须跑 `./scripts/sanity.sh`（含 `unit_simt_integration`，在 Tier 5）

### 检查工具

```bash
# 1. 查找所有 "byte-identical" / "no-op" 契约注释
grep -rn "byte-identical\|no-op\|nullptr = " src/ptxsim/ include/ptxsim/

# 2. 对每个 no-op 契约，验证是否有测试锁定
grep -rn "no-op\|byte-identical" tests/

# 3. 验证 file-local 函数可测试性（匿名命名空间 = 不可直接测试）
grep -B2 "namespace {" src/ptxsim/core/*.cpp
```

### 修复模板

```cpp
// BEFORE (buggy): 注释承诺 no-op，但 nullptr 路径 fallthrough
inline void step_b(IPipeline* p, ITc* tc, Warp* w, const Stmt& s) {
    // nullptr = byte-identical fallback  <-- 注释承诺
    uint32_t latency = 0;
    if (p) { /* ... */ }
    if (latency == 0 && tc) { /* ... */ }
    if (latency == 0) latency = getLatency(s);  // <-- nullptr 时也执行！
    if (latency > 0) w->set_blocked_cycles(latency);
}

// AFTER (fixed): 显式 early return + 测试锁定 4 条分支
inline void step_b(IPipeline* p, ITc* tc, Warp* w, const Stmt& s) {
    if (!p && !tc) return;  // both nullptr = no-op (TESTED)
    // ... priority chain ...
}
// + tests/unit/sm/test_step_b_set_blocked_cycles.cpp:
//   Case 1: both nullptr -> blocked_cycles_remaining == 0 (no-op)
//   Case 2: pipeline returns 2.5 -> blocked_cycles_remaining == 3 (ceil)
//   Case 3: tc + TC instruction -> blocked_cycles_remaining == tc->get_latency()
//   Case 4: fallback -> blocked_cycles_remaining == getLatency(stmt.type).cycles
```

### 真实案例

- `commit 367fd6a5` (feat) 引入 `step_b_set_blocked_cycles`，声称 "0 regression"
- `commit 5b292a91` (fix) 修复回归：`unit_simt_integration` 2 个断言失败
- 回归在 feat 提交后**下一个提交**即被发现，说明 TDD 纪律有效，但 feat 的 "0 regression" 声明不准确（可能未跑完整 `sanity.sh --quick`，或仅跑 subset）
- 修复后补 `tests/unit/sm/test_step_b_set_blocked_cycles.cpp` 4 个 case 直接锁定 4 条分支
- 函数从匿名命名空间提取为 `SMContext::step_b_set_blocked_cycles`（public static），与 `is_tensor_core_instruction` 等 3 个 helper 一致

### 关联

- **Skill**: `.opencode/skills/ptx-lessons-learned/SKILL.md` §14（同步新增）
- **回归 commit**: `367fd6a5`（feat，引入 bug）
- **修复 commit**: `5b292a91`（fix，回归修复）
- **测试 commit**: 本次（TDD 补测，4 case 锁定 4 分支）
- **spec 契约**: cpptlm-phase8b-injection-points design.md §2.4 "byte-identical fallback"
- **ADR**: ADR-0020（CppTLM D1-Full injection points）

---

**§38 更新日期**: 2026-07-17  
**关联 commit hash**: 367fd6a5 / 5b292a91  
**Skill ref**: `.opencode/skills/ptx-lessons-learned/SKILL.md` §14

---

## 39. Step B（延迟查询）必须在 execute 之后执行（2026-07-18）

### 教训

在 `exe_once()` 中新增注入点时，**延迟/阻塞操作必须在指令执行之后设置**。

### 失败模式

```cpp
// ❌ WRONG: Step B before execute — set_blocked_cycles_for_active 设置
// is_blocked=true 后，execute_warp_instruction 的 is_lane_active() 返回 false，
// 所有 lane 被跳过 → 指令永不执行。
set_blocked_cycles_for_active(latency);  // blocks all active threads
execute_warp_instruction(stmt, pc);       // all lanes skipped!

// ✅ CORRECT: Step B after execute
execute_warp_instruction(stmt, pc);       // instruction executes normally
set_blocked_cycles_for_active(latency);   // THEN block for latency
```

### 为什么静默

所有现有测试使用 nullptr injector。当 `pipeline_provider_` 和 `tensor_core_timing_` 均为 nullptr 时，`step_b_set_blocked_cycles` 直接 return（no-op），不触发 `set_blocked_cycles_for_active`。只有注入真实 CppTLM timing 模型（非 nullptr）才会暴露。

### 发现过程

2026-07-18 Oracle 审查 `cpptlm-phase8b-injection-points` 的已提交代码（PTX-1~6）。审计 `step_b_set_blocked_cycles` 调用点（`sm_context.cpp` fast path + divergent path）时发现 Step B 在 `execute_warp_instruction` **之前**。调用链：`set_blocked_cycles_for_active` → 设置 `is_blocked=true, blocked_cycles_remaining=N` → `is_lane_schedulable()` 返回 false → `is_lane_active()` 返回 false → `execute_warp_instruction` 的 lane 活性检查跳过所有线程。结果：当 injectors 非 nullptr 时，**模拟器完全停止执行指令**。

设计文档（`design.md §7.1`）本身也指定 Step B 在 execute 之前 — 属于设计层面缺陷。

### 诱因

- **设计假设错误**：认为 "查询延迟 → 设置 blocked_cycles → 执行指令" 等价于流水线仿真。但 PTX-EMU 的执行模型是"指令立即执行（组合逻辑），然后线程阻塞 N 周期（模拟结果延迟）"。
- **nullptr 掩盖**：默认 nullptr 使 Step B 成为 no-op，隐藏了顺序敏感性问题。
- **无非 nullptr 测试**：Phase 1-4 实现期间未编写注入真实 Mock 的测试（Phase 5 才计划）。

### 量化影响

- **修复范围**: `sm_context.cpp` 2 处（fast path line 354→after 365, divergent path line 446→after 458）
- **验证**: `ctest -E e2e_divergence$` → 210/210 pass（2 个新增测试 + 0 回归）
- **预防**: PTX-7a Test 4/5/6 注入非 nullptr Mock，验证指令执行后 blocked_cycles 正确设置

### 预防规则

1. 任何新增的**线程状态修改**（`is_blocked`, `is_active`, `blocked_cycles_remaining`）必须在 `execute_warp_instruction` 之后执行
2. Phase 1-4 实现阶段必须至少有一个**非 nullptr 注入点测试**（不能仅靠 Phase 5 才覆盖）
3. Oracle 审查应关注调用**顺序**，不只是调用**存在性**

### 关联

- **修复 commit**: `290ebf88`
- **设计文档**: design.md §7.1
- **审查 commit**: `fb990cb3`（design.md §7.1 control flow fix — 修复了其他问题但保留了 Step B 顺序错误）
- **ADR**: ADR-0020

**§39 更新日期**: 2026-07-18  
**关联 commit hash**: 290ebf88  
**Skill ref**: `.opencode/skills/ptx-lessons-learned/SKILL.md` §39（待同步）

## §40：Proposal 当前状态声明必须工具验证

### 现象

`add-cudart-unit-test-coverage` 的 proposal 声称 `tests/unit/cudart/` "零直接单元测试"，但实际已有 3 个测试文件（248 行代码）。这导致 C4 的 Stream API 范围需要从"新建"调整为"互补"。究其原因是 proposal 依赖审计文档的间接描述，未用 `find`/`ls` 工具直接验证目录状态。

### 教训

- Proposal 中任何关于"当前文件/目录/测试等存在性"的声称，必须用工具直接验证
- 审计文档是二级来源，可能滞后或描述不精确
- `find <dir> -type f | wc -l` 和 `ls <dir>` 的 5 秒验证成本可避免数小时的返工

### 真实案例

- **表现**: C4 Stream API 测试范围需要修正
- **修复**: 承认已有 3 个测试，新增 Stream 测试仅覆盖互补场景（唯一性、recreate、nullptr）
- **日期**: 2026-07-18

### 关联

- **change**: `openspec/changes/add-cudart-unit-test-coverage/`
- **ADR**: ADR-0010 §2026-07-18 Postmortem

## 41. 静态库 weak symbol 跨 .so 覆盖失败（2026-07-19）

### 现象

PTX-EMU 定义 `__attribute__((weak)) cpptlm_set_driver()`，CppTLM 静态库 `libcpptlm_core.a`
定义同名的强符号。两者链接到同一 `libcudart.so` 后，`nm` 输出显示 `W`（weak），
CppTLM 的强定义未被采纳。运行时 `initialize_environment()` 调用的是 PTX-EMU 的弱 no-op。

### 教训

- **静态库链接规则**：链接器只在符号**未定义**时才从 `.a` 拉入对象文件。
  weak 定义已满足符号需求 → 静态库中的强定义被跳过。
- **修复**：`target_link_libraries(cudart -Wl,--whole-archive cpptlm_core -Wl,--no-whole-archive)`
  强制纳入所有对象，编译期覆盖。
- **备选方案**：`dlsym(RTLD_NEXT, "func")` 运行时查找（更灵活但有运行时开销），或
  将 CppTLM 编译为 `.so`（标准动态链接路径）。
- **add_subdirectory 外部项目**：需在 `add_subdirectory` 前 `set(CMAKE_POSITION_INDEPENDENT_CODE ON)`
  并在完成后恢复原值，否则 `.a → .so` 链接时报 `R_X86_64_TPOFF32` 重定位错误。

### 真实案例

- **bug 表现**: `nm build/lib/libcudart.so | grep cpptlm_set_driver` → `W`（预期 `T`）
- **修复**: `97539fdb` — `--whole-archive` + PIC save/restore
- **验证**: `nm` 输出 `T cpptlm_set_driver`，ctest 212/213 PASS

### 关联

- **change**: `openspec/changes/archive/2026-07-19-cpptlm-p1-ptxemu-shim/`
- **ADR**: ADR-0021 §2026-07-19 Postmortem

## 42. Auto-advance 机制天然解决单次 exe_once() 的 admit+judge 竞态（2026-07-21）

### 现象

`GPUContext::exe_once()` 在 bridge 路径下 admit kernel（`execute_kernel_internal` 设置 SM=RUN）后立即判 `all_warps_finished()`，同一调用内存在"admit kernel → SM=RUN → `sm->exe_once()` 可能不调度 warp → 判 EXIT"的竞态。此前受此影响的 bridge-path 测试通过 attach-bridge-after-launch（走同步路径）规避。

### 教训

- **while-loop 驱动执行天然避免单次调用的时序问题**：`PtxEmuDriverShim::advance(max_cycles)` 在 while 循环中反复调用 `exe_once()`，首次 admit → SM=RUN，后续调用执行 → EXIT。这天然分离了 admit 与 judge，不需要显式的"just_admitted flag"或 `exe_once()` 内部重构
- **auto-advance at sync point 是正确的架构模式**：标准 CUDA 程序的 `cudaDeviceSynchronize` 是最高效的 advance 触发点（用户已显式表示"等待完成"），比在 `cudaLaunchKernel` 中 auto-advance（破坏异步语义）更优
- **环境变量 ceiling 是防止死锁的简单有效机制**：`PTX_EMU_MAX_ADVANCE_CYCLES`（默认 10M）防止病态 kernel 永久挂起，比硬编码超时更灵活

### 真实案例

- **bug 表现**: `test_cosim_vector_add.cu` 在 bridge-attached-before-launch 场景下输出全零
- **根因**: 单次 `exe_once()` 内 admit→可能不执行→判 EXIT
- **修复**: `cudaDeviceSynchronize` / `cudaStreamSynchronize(0)` 中先 `advance(max_cycles, actual)` 再 poll，while-loop 反复 `exe_once()` 直至 EXIT
- **验证**: `e2e_cosim_vector_add` ON 模式 64/64 golden match + 零回归

### 诊断命令

```bash
# 验证 advance 实际执行了 PTX 指令
grep -n "advance" src/cudart/cudart_sim.cpp
# 检查 advance ceiling 配置
echo $PTX_EMU_MAX_ADVANCE_CYCLES
```

### 关联

- **change**: `openspec/changes/auto-co-sim-standalone/`
- **commit**: `10e8ad38`
- **ADR**: ADR-0021 §2026-07-21 Fix Record
