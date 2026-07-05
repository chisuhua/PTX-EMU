---
name: "ptx-lessons-learned"
description: "PTX-EMU 项目经验沉淀 — 跨模块状态翻译、递归锁、分 Phase commit、基线 worktree、qualifier 类型判断等具体失败模式与可复用 checklist。来自 2026-06 barrier module 迁移 + 2026-07 cute_rmsnorm float 类型判断实战"
when_to_use: |
  PTX-EMU 项目中遇到以下场景时强制使用：

  实施新功能/重构前：
  - "迁移/重构现有函数"、"API 切换"、"重写 handler"
  - 涉及 ThreadContext / WarpState / Barrier / SIMT 栈的状态修改
  - 涉及互斥量、锁、并发原语
  - 涉及多 Phase 推进（OpenSpec change 实施）
  - 涉及类型系统修改（qualifier 解析、is_float/is_signed 判断）

  调试失败时：
  - "测试在断言 N+1 处挂起" → 递归锁死锁
  - "集成测试 is_blocked 失败" → 跨模块状态翻译缺失
  - "测试套件整体超时" → 用 per-test timeout 定位
  - "Phase N 通过但 Phase N+1 失败" → 分 Phase commit 纪律
  - "E2E 测试输出非确定性但单元测试通过" → qualifier.back() 类型判断 bug

  实施完成后：
  - 提交前必查：checklist D（AGENTS.md 同步、ADR 追加、commit message 列出 fix 编号）

skills_required: []
---

# PTX-EMU 经验沉淀

> **来源**: 2026-06-18 commit `f033312` (integrate-barrier-module-cta-warp) 实战
> **完整文档**: [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md)
> **互补文档**: [`docs/dev-process/debugging-strategy.md`](../../docs/dev-process/debugging-strategy.md)（问题分类与快速验证）、[`.opencode/skills/ptx-barrier-mechanism/`](./ptx-barrier-mechanism/)（领域知识）

## ⚡ 快速决策树

```
任务是什么？
├─ 迁移/重构函数 → 查"Checklist A: 函数迁移"
├─ 重大重构前   → 查"Checklist B: 重构前"
├─ 写注释      → 查"Checklist C: 写注释"
├─ Commit 前   → 查"Checklist D: Commit 前"
└─ 调试失败    → 查"失败模式速查表"
```

---

## 📚 核心经验（最常用 5 条）

### 1. 跨模块间接状态翻译（最隐蔽的 bug 来源）

**问题模式**: 迁移函数时漏掉看似冗余的 `set_state(BAR_SYNC)`，因为下一模块的 `sync_to_warp_state()` 才把它翻译为 `is_blocked = true`。没有这行，调度器死循环。

**关键经验**：
- 跨函数的"间接翻译"是最隐蔽的 bug 来源
- 迁移函数必须做到"行级 Diff"，不只比对主要逻辑
- `set_state` 看似冗余（next_pc 已设），但它是另一模块的 API 契约

**诊断命令**：
```bash
# 找出所有 set_state 调用点
grep -rn "set_state(" src/ptxsim/ include/ptxsim/ | grep -v "test"
# 找出所有读取 state 的位置
grep -rn "state == BAR_SYNC\|get_state()\|is_at_barrier" src/ptxsim/
# 比对：每个 set_state 应该有对应的翻译规则
```

**修复模板**（详见 `debugging-strategy.md` §"跨模块状态翻译检查"）：
```cpp
} else {
    // Mark thread as waiting at <event> so the executor (<file>:<line>)
    // recognizes <STATE> and skips re-execution. Without this,
    // sync_to_warp_state() keeps is_<field>=false and the scheduler
    // spins on the instruction. (Mirrors legacy <old-code> at <file>:<line>.)
    context->set_state(<STATE>);
    context->set_next_pc(context->get_pc());
}
```

### 2. 递归锁死锁（互斥量需要集中审计）

**问题模式**: `arrive()` 持 `mutex_` 后调用 `is_complete()`，后者再次 `lock_guard(mutex_)`。`std::mutex` 不可重入 → 死锁。

**关键经验**：
- 可重入性问题通常被单元测试忽略
- 死锁信号模式："测试在断言 N+1 处挂起，N+1 通常是循环的第一次调用"
- public 方法不应该再锁（应提供 internal unsafe 版本）

**诊断命令**：
```bash
grep -n "lock_guard\|unique_lock" <file>           # 列出所有锁点
grep -B2 "lock_guard" src/ | grep -A1 "lock_guard"  # 找出嵌套锁
```

**修复模式**：
```cpp
// 方案 A：拆分两个版本
class Foo {
    bool is_complete_unsafe() const;  // 假设已持锁
    bool is_complete() const {        // public 持锁
        std::lock_guard lock(mutex_);
        return is_complete_unsafe();
    }
};

// 方案 B：内联（最小改动）
bool arrive() {
    std::lock_guard lock(mutex_);
    arrived_threads_.insert(thread);
    bool complete = arrived_threads_.size() >= expected_threads_;  // 直接比较
}
```

### 3. 复杂迁移必须分 Phase commit

**关键经验**：
- 每个 Phase 独立 commit、独立验证，失败立即 revert
- 涉及控制流/分歧/同步的迁移，必须有"分歧场景"测试覆盖
- 单 PC 测试通过 ≠ 全部分歧场景通过

**判定"Phase 完成"的标准**：
- ✅ 所有原通过的测试仍然通过
- ✅ 新增的测试通过
- ✅ 没有测试"意外"变快/变慢
- ❌ 任何已有测试回归 → **立即 revert 该 Phase**

### 4. 基线 worktree 是最低成本保险

**关键经验**：
- 任何重大重构前花 1 分钟建立基线 worktree
- 节省数小时的"这个失败是基线的还是我的"争论
- 保留直到 change 完整合并 + 验证通过

**实测验证**: worktree 里 nvcc + CUDA 编译完全正常（实测 baseline worktree 中 `e2e_barrier_warp_sync` PASS 0.53s）。`env.sh` 是路径无关的（`NVCC_PATH=$(which nvcc)`），不依赖固定路径。

**标准操作**：
```bash
# Step 1: 建立 baseline（全量 build 预算 15-20 分钟）
git worktree add .worktrees/baseline-check <baseline-commit>
cd .worktrees/baseline-check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)  # 必须全量！否则部分 target 找不到

# Step 2: 验证 baseline 通过
cd build && ctest -L <label> --output-on-failure

# Step 3: 对比 main
cd <main-build> && ctest -L <label> --output-on-failure

# Step 4: 清理（实施完成且验证通过后）
git worktree remove .worktrees/baseline-check
```

**⚠️ 三个常见陷阱**:
1. baseline commit 必须**包含** main 上所有测试（不能选任意旧 commit）
2. baseline 首次 build 必须**全量**（partial build 导致 ctest 报"找不到可执行文件"）
3. worktree 共享 `.git` 但 build 独立（同一时刻不能在两个 worktree 切到同一分支）

**时间预算**:
- baseline 首次 build：15-20 分钟
- 增量 build：几秒-几十秒
- single target：5-30 秒

### 5. 类型判断依赖 "最后一个 qualifier" 导致 float 指令被当作整数

**问题模式**: `TypeUtils::is_float_type()` 只检查 `qualifiers.back()`。PTX 解析器生成的 qualifier 列表中 Q_F32 不在末尾（如 `{Q_U32, Q_S32}`），导致 `mul.f32` 等所有 float 指令被当作整数处理 → 64 位整数乘积写入 32 位浮点寄存器 → 非确定性垃圾输出。

**关键经验**：
- `qualifiers.back()` 是脆弱的类型判断 — 最后一个元素可能是数据类型、内存空间或修饰符
- 必须遍历**整个列表**检查目标类型
- 单元测试通过 ≠ handler 正确：单元测试直接构造 `{Q_F32}`，PTX 解析器生成的是多元素列表
- 非确定性输出 + handler 单元测试通过 = 类型判断层 bug 的强信号

**诊断命令**：
```bash
# handler 入口注入临时代码确认 is_float 标志
# 在 executeOperation 中打印 qualifiers 列表
grep -rn "is_float_type" src/ptxsim/  # 列出所有受影响 handler
grep -rn "qualifiers.back()\|\\.back()" src/  # 检查其他脆弱的 back() 使用
```

**修复模板**：
```cpp
// BEFORE (buggy): 只检查最后一个
bool is_float_type(const auto &qualifiers) {
    return !qualifiers.empty() && 
           (qualifiers.back() == Q_F32 || qualifiers.back() == Q_F64);
}

// AFTER (correct): 遍历所有 qualifier
bool is_float_type(const auto &qualifiers) {
    for (const auto &q : qualifiers) {
        if (q == Q_F32 || q == Q_F64 || q == Q_F16 || q == Q_BF16)
            return true;
    }
    return false;
}
```

---

### 6. OpenSpec artifacts 提交遗漏 + Debt audit 必须 git verify（2026-07 新增）

**问题模式**: 实施 OpenSpec change 时，工作区修改了 `openspec/changes/<name>/{design.md,tasks.md,spec.md}` 反映实施调整，但**这些 artifacts 修改从未 `git add`** — 仅源码 + commit message 描述了改动。fast-forward merge 后，OpenSpec 状态与代码不一致，需 commit `reconstruct` 补救。

**触发场景**: `cleanup-deprecated-barrier-apis` (2026-06-20) — 3 个实施 commits (`8a5573d`/`7914764`/`6ec8efd`) 合并时未追踪 artifacts；12 天后 (`barrier-migration-amendment` 2026-07-02) 基于 untracked reconstructed artifacts 误判 4 条 P0-A 为 active debt。

**关键经验**：
- **实施 OpenSpec change 必须 2-Phase commit**：
  1. Phase 0：`git add openspec/changes/<name>/` + commit "docs(openspec): <name> design adjustments" (artifacts FIRST)
  2. Phase 1+：实施代码 + commit
- **Debt audit 必须满足 2 个先决条件**：
  1. 当前 git HEAD 状态（不是 working tree）— `git status` + `git log -- <path>` 验证
  2. 引用 commit hash 而非文件路径
- **OpenSpec archive = change 终态**：归档后任何修补需求应新建 `fix-*`/`refactor-*` change + `Ref: archive/<date>-<name>/`，不要 amend 已归档 change

**诊断命令**：
```bash
# 验证 change 是否已归档
git log --all --oneline -- "openspec/changes/<change-name>/"
# 应包含 archive commit（如 ded4f96 chore(openspec): archive ...）

# 验证 artifacts 是否 tracked（实施后必须）
git ls-files openspec/changes/<change-name>/
# 不应为空

# 审计前自检
git status openspec/changes/  # 未提交修改警告
```

**真实案例**:
- `barrier-migration-amendment` (2026-07-02) — 试图 amend 已于 2026-06-20 归档的 `cleanup-deprecated-barrier-apis`
- `.opencode/notes/debt-audit-2026-07-02.md` §1.1 P0-A1~A4 误标为 active debt — 实际已通过 commits `8a5573d`/`7914764`/`6ec8efd` 解决

**修复模板**：
```bash
# 1. 删除 obsolete untracked dirs
rm -rf openspec/changes/<obsolete-amendment>/ openspec/changes/<already-archived-as-reconstructed>/

# 2. 更新 debt audit 标记为 RESOLVED
# docs/audits/<audit>.md §1.1 添加"状态"列 + 引用 commits

# 3. 沉淀 lesson
# docs/dev-process/lessons-learned.md 添加新条目（按现象/教训/检查工具/真实案例结构）
# .opencode/skills/ptx-lessons-learned/SKILL.md 添加 §核心经验 + Checklists E/F/G
```

### 7. Pre-implementation Review：实施 OpenSpec change 前必须跑 Metis 审计（2026-07 新增）

**问题模式**: 实施 OpenSpec change 时，proposal 基于"目录/文件存在性推断"而非实证撰写。archive README 的"✅ COMPLETED" + 文件未删除 ≠ "已完整实施"。

**真实案例**: `fix-cvt-strategy-actual-split`（commits `e8db807`/`f3ef891`/`43edf55`，2026-07-05）原 6-Phase 计划基于错误前提：
- ❌ 假设 919 行 switch 块未拆分 → 实证：4 个 Strategy 类已部署（`fc3c352`/`9837d44`/`d6123e0`）
- ❌ 假设 `.worktrees/fix-pre-p0-baseline` 可复用 → 实证：worktree 目录为空
- ❌ 假设 94 个 integration 测试为 oracle → 实证：仅 14 个
- ❌ 假设 `select_strategy()` 返回 `unique_ptr` → 实证：返回 `const ConversionStrategy&`

**关键经验**：
- **OpenSpec proposal 必须基于实证**：`git log -- <file>` + `grep <api>` + `wc -l <file>`，**禁止用"存在/未存在"推断状态**
- **必须区分"已实施但未清理"与"未实施"**：4 类表象对照见 `docs/dev-process/lessons-learned.md §20`
- **实施 OpenSpec change 之前必须跑 Metis pre-implementation review**：本 case 的 5 项 MUST-RESOLVE 全是实施前的隐形炸弹（scope + 接口 + 测试 + worktree + 路径），由 Metis 一次 review 全部揭示
- **archive "已归档"状态不可 amend**：任何修补必须新建 `fix-*` change + `Ref: archive/<date>-<name>/`（与 §6 互补）

**诊断命令**：
```bash
# 1. 验证 proposal 引用的 API 真实存在
grep -rn "<symbol>" src/ include/ tests/

# 2. 验证 oracle 测试数量真实
ctest -N -L "<label>" 2>&1 | tail -5

# 3. 验证提到的工作目录/路径/工具真存在
ls <worktree-path> 2>/dev/null
test -f <path> && echo exists || echo missing

# 4. 列出 baseline 文件的关键状态
wc -l <file>
git log --oneline -10 -- <file>
git log --all --oneline -- "<change-dir>"
```

**修复模板**：
1. 立即调用 Metis - Plan Consultant 子代理审计 4 个 OpenSpec artifacts
2. 检查 Metis 输出 ⚠️ CONDITIONAL 决策的 MUST-RESOLVE 列表（≥3 项阻塞 apply）
3. 重写 proposal/design/tasks.md 反映真实 scope（示例：6 Phase 拆分 → 3 Phase 死代码清理）
4. 在每个 artifact §Ref 段加 "Metis pre-implementation review" 链接
5. 沉淀到 lessons-learned.md §20 + ADR-0015 §2026-07 Fix 段

---

## ✅ 可复用 Checklist

### Checklist A: 函数迁移

```
□ 列出 baseline 函数中所有的 set_* / commit_* / force_* 调用
□ 列出所有 mutex_ / lock_guard / unique_lock 使用
□ 对每个 set_*，grep 其值的下游消费者
□ 对每个锁点，确认"持锁方法调用的所有其他方法"也持同一锁，或重写为无锁版本
□ 比对行级 diff（不只比对主要逻辑）
□ 对所有 GenericPipelineHandler 子类，验证 is_float/is_signed 遍历全部 qualifier 而非只看 back()
```

### Checklist B: 重构前

```
□ 建立基线 worktree
□ 列出本 change 的所有 Phase，决定 commit 粒度
□ 决定哪些 Phase 需要基线对比（涉及 invariant 的一定要）
□ 准备 revert 策略：每个 Phase 独立 commit，失败立即 revert
```

### Checklist C: 写注释

```
□ 这条注释能否让 3 个月后的陌生人避免犯同样的错？
□ 是 → 写
□ 否 → 不写
□ 例外：警告"不要做 X"必须写
```

### Checklist D: Commit 前

```
□ 跑过 baseline worktree 对比
□ AGENTS.md 是否需要同步
□ ADR 是否需要追加
□ OpenSpec tasks.md 是否需要更新
□ commit message 列出独立的 fix 编号
```

### Checklist E: OpenSpec change 实施后（2026-07 新增）

```
□ 所有 OpenSpec artifacts (design.md / tasks.md / spec.md / proposal.md) 已 git-tracked
  - 验证：git ls-files openspec/changes/<name>/ 不应为空
□ commit message 列出独立 fix 编号（如 Fix #1, Fix #2）
□ 每个 commit 独立可 revert（git revert HEAD 后编译通过）
□ 实施 commits 合并后立即 git-tracked artifacts（避免 working tree 遗漏）
□ 归档前 grep 验证 artifacts 与代码一致（无过期 task 编号）
```

### Checklist F: Debt audit 撰写（2026-07 新增）

```
□ 审计前 git log --since=<audit-date-1> -- <path> 验证所有引用 change 的实施状态
  - 验证：git log --all --oneline -- openspec/changes/<change-name>/ 不应仅含 archive commit
□ 引用 commit hash 而非文件路径
  - 示例："P0-A1 RESOLVED by commit 8a5573d" 而非 "P0-A1 当前 design.md 已修复"
□ 区分 "active debt"（影响实施）vs "stale debt"（已解决但审计未更新）
□ 每次审计标注 "基于 HEAD <hash>" 而非 "当前状态"
□ 审计撰写时若 working tree 与 git HEAD 不一致，必须明确标注：
  - "基于 working tree 状态，可能与 HEAD 不一致"
```

### Checklist G: OpenSpec lifecycle 约束（2026-07 新增）

```
□ Proposed: 未实施，artifacts 可修改
□ Accepted: 已批准实施，artifacts 可修改但需说明理由
□ Active: 实施中，artifacts 可修改（带 progress 标记）
□ Archived: 终态，artifacts 不可修改
  - 若需修补 → 新建 fix-* / refactor-* change
  - 引用方式：Ref: archive/<date>-<change-name>/
  - 禁止 amend 已归档 change（违反 OpenSpec 生命周期）
```

### Checklist H: Pre-implementation Review 强制项（2026-07 新增）

```
□ 实施 OpenSpec change 前：调用 Metis - Plan Consultant 子代理
  - 提供 4 个 artifacts 路径 + 真实文件路径/行号引用要求
  - 要求输出：Hidden Intentions / Ambiguities / AI Failure Points / Missing Context
  - 要求给出 GO / ⚠️ CONDITIONAL / ❌ NO-GO 决策
□ Metis 输出 ⚠️ CONDITIONAL 时：5 项 MUST-RESOLVE 全部完成才能 apply
□ 验证 proposal 的关键假设（实证）：
  - wc -l 验证声称的"X 行"真实存在
  - git log 验证 archive 中的"已实施 commits"真存在
  - grep 验证引用的 API 真存在（0 matches = 假设错误）
  - ctest -N 验证 oracle 数量（如 94 → 14 是常见偏差）
  - ls 验证 worktree/路径真存在（空目录 ≠ "可复用现有"）
□ 区分"已实施但未清理" vs "未实施"（4 类表象对照 §20）
```

---

## 🔍 失败模式速查表

| 症状 | 最可能原因 | 诊断命令 |
|------|-----------|---------|
| 集成测试断言 N+1 处挂起 | 递归锁死锁 | `grep -n "lock_guard\|unique_lock" <file>` |
| `CHECK(!is_blocked)` 类似断言失败 | 跨模块状态翻译缺失 | `grep "set_state.*BAR_SYNC" src/` 对比 `grep "state == BAR_SYNC" src/` |
| 调度器在 barrier 指令处死循环 | `set_state(BAR_SYNC)` 漏掉 | `grep "set_state" src/ptxsim/instructions/barrier.cpp` |
| 分歧场景一半 lanes 卡住 | 屏障释放逻辑覆盖了已释放 lanes | 检查 `set_active_mask` 是否 OR 而非 overwrite |
| Phase N 通过但 Phase N+1 失败 | 跨 Phase invariant 冲突 | 用基线 worktree 隔离每个 Phase |
| `git revert` 后 `git status` 异常 | stash/pop 改变了 staged 状态 | `git status` 验证 + 必要时 `git reset` |
| 注释"看似冗余" | 跨模块间接状态翻译 | 查"经验 1"，看是否需要保留 |
| `ctest -L <label>` 整体超时 | 某个测试死锁 | 切到 single-test + per-test timeout |
| **E2E 测试输出非确定性（每次不同值），但 handler 单元测试通过** | **`is_float_type()` 只看 `qualifiers.back()`，Q_F32 不在末尾时被误判为整数** | **handler 入口加 printf: `is_float` 标志；`grep "is_float_type" src/ptxsim/` 列出受影响 handler** |

---

## 🔗 与 OpenSpec 流程的集成点

| OpenSpec 阶段 | 使用此 skill 的方式 |
|---------------|------------------|
| `openspec-propose`（设计阶段）| 强制调用本 skill 的 "Checklist A" 验证函数迁移完整性 |
| `openspec-apply-change`（实施阶段）| 强制调用本 skill 的 "Checklist D" 提交前验证 |
| `openspec-archive-change`（归档阶段）| 自动 prompt 询问"是否生成 postmortem"，引用 ADR 追加段落 |
| `adr-compliance-check`（合规检查）| 自动 cross-check 本 skill 的失败模式速查表 |

---

## 📊 经验沉淀的元规则

> **新经验的加入流程**:
> 1. 在实施中发现 bug 模式
> 2. 修复后，写入本 skill + `lessons-learned.md` + 相关 ADR postmortem
> 3. 同步更新相关 skill（如 `ptx-barrier-mechanism` 涉及屏障问题时）
> 4. 在 `skills/README.md` 更新调用关系图

> **本 skill 与 `lessons-learned.md` 的关系**:
> - **本 skill**: agent 主动加载，提供快速决策树 + checklist + 速查表
> - **lessons-learned.md**: 完整文档（具体案例、代码片段、长篇解释）
> - **互补关系**: 加载 skill 后能快速判断"我遇到了哪类问题"，再 deep-dive 到 lessons-learned.md 看完整案例

---

## 🔗 相关资源

- **完整经验文档**: [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md)（16 章节，~600 行）
- **调试策略**: [`docs/dev-process/debugging-strategy.md`](../../docs/dev-process/debugging-strategy.md)（含"跨模块状态翻译检查"章节）
- **Postmortem 案例**: [`docs/adr/0008-barrier-semantics.md`](../../docs/adr/0008-barrier-semantics.md) §2026-06-18 Postmortem
- **原 change 计划**: [`docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md`](../../docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md)
- **领域知识**: [`.opencode/skills/ptx-barrier-mechanism/`](./ptx-barrier-mechanism/)、[`.opencode/skills/ptx-instruction-pipeline/`](./ptx-instruction-pipeline/)
- **状态审计**: [`.opencode/skills/state-modification-audit/`](./state-modification-audit/)（与本 skill 经验 1 配套使用）
- **回归定位**: [`.opencode/skills/regression-bisect/`](./regression-bisect/)（与本 skill 经验 3-4 配套使用）
