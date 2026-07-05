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
| 整体架构变更 | `docs/adr/0008-barrier-semantics.md`（追加 2026-06-17 段落） |
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
- **关联 ADR**: `docs/adr/0008-barrier-semantics.md`（已追加 2026-06-18 postmortem）
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
- **ADR 更新**: `docs/adr/0008-barrier-semantics.md` §2026-07-03 追加完整 postmortem（含 Phase 3/Phase 7/Phase 7b 三 commit 拆分 + lessons §1/§2/§4 的应用证据）
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
  - `docs/adr/0015-cvt-strategy-pattern.md` §2026-07 Fix 段（含 lessons-learned §18 案例沉淀）
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
