---
name: openspec-apply-change
description: Implement tasks from an OpenSpec change. Use when the user wants to start implementing, continue implementation, or work through tasks.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.4.1"
---

Implement tasks from an OpenSpec change.

**Input**: Optionally specify a change name. If omitted, check if it can be inferred from conversation context. If vague or ambiguous you MUST prompt for available changes.

**Steps**

1. **Select the change**

   If a name is provided, use it. Otherwise:
   - Infer from conversation context if the user mentioned a change
   - Auto-select if only one active change exists
   - If ambiguous, run `openspec list --json` to get available changes and use the **AskUserQuestion tool** to let the user select

   Always announce: "Using change: <name>" and how to override (e.g., `/opsx-apply <other>`).

2. **Check status to understand the schema**
   ```bash
   openspec status --change "<name>" --json
   ```
   Parse the JSON to understand:
   - `schemaName`: The workflow being used (e.g., "spec-driven")
   - `planningHome`, `changeRoot`, and `actionContext`: planning scope and edit constraints
   - Which artifact contains the tasks (typically "tasks" for spec-driven, check status for others)

3. **Get apply instructions**

   ```bash
   openspec instructions apply --change "<name>" --json
   ```

   This returns:
   - `contextFiles`: artifact ID -> array of concrete file paths (varies by schema - could be proposal/specs/design/tasks or spec/tests/implementation/docs)
   - Progress (total, complete, remaining)
   - Task list with status
   - Dynamic instruction based on current state

   **Handle states:**
   - If `state: "blocked"` (missing artifacts): show message, suggest using openspec-continue-change
   - If `state: "all_done"`: congratulate, suggest archive
   - Otherwise: proceed to implementation

   **Workspace guard:** If status JSON reports `actionContext.mode: "workspace-planning"` and `allowedEditRoots` is empty, explain that full workspace apply is not supported in this slice. Treat linked repos and folders as read-only context, ask the user to select an affected area through an explicit implementation workflow, and STOP before editing files.

4. **Read context files**

   Read every file path listed under `contextFiles` from the apply instructions output.
   The files depend on the schema being used:
   - **spec-driven**: proposal, specs, design, tasks
   - Other schemas: follow the contextFiles from CLI output

5. **Show current progress**

   Display:
   - Schema being used
   - Progress: "N/M tasks complete"
   - Remaining tasks overview
   - Dynamic instruction from CLI

6. **Implement tasks (loop until done or blocked)**

   For each pending task:
   - Show which task is being worked on
   - Make the code changes required
   - Keep changes minimal and focused
   - Mark task complete in the tasks file: `- [ ]` → `- [x]`
   - Continue to next task

   **Pause if:**
   - Task is unclear → ask for clarification
   - Implementation reveals a design issue → suggest updating artifacts
   - Error or blocker encountered → report and wait for guidance
   - User interrupts

7. **On completion or pause, show status**

   Display:
   - Tasks completed this session
   - Overall progress: "N/M tasks complete"
   - If all done: suggest archive
   - If paused: explain why and wait for guidance

**Output During Implementation**

```
## Implementing: <change-name> (schema: <schema-name>)

Working on task 3/7: <task description>
[...implementation happening...]
✓ Task complete

Working on task 4/7: <task description>
[...implementation happening...]
✓ Task complete
```

**Output On Completion**

```
## Implementation Complete

**Change:** <change-name>
**Schema:** <schema-name>
**Progress:** 7/7 tasks complete ✓

### Completed This Session
- [x] Task 1
- [x] Task 2
...

All tasks complete! Ready to archive this change.
```

**Output On Pause (Issue Encountered)**

```
## Implementation Paused

**Change:** <change-name>
**Schema:** <schema-name>
**Progress:** 4/7 tasks complete

### Issue Encountered
<description of the issue>

**Options:**
1. <option 1>
2. <option 2>
3. Other approach

What would you like to do?
```

**Guardrails**
- Keep going through tasks until done or blocked
- Always read context files before starting (from the apply instructions output)
- If task is ambiguous, pause and ask before implementing
- If implementation reveals issues, pause and suggest artifact updates
- Keep code changes minimal and scoped to each task
- Update task checkbox immediately after completing each task
- Pause on errors, blockers, or unclear requirements - don't guess
- Use contextFiles from CLI output, don't assume specific file names

**Fluid Workflow Integration**

This skill supports the "actions on a change" model:

- **Can be invoked anytime**: Before all artifacts are done (if tasks exist), after partial implementation, interleaved with other actions
- **Allows artifact updates**: If implementation reveals design issues, suggest updating artifacts - not phase-locked, work fluidly

---

## 🔧 Implementation-Time Lessons Integration (强制)

> **加载 skill**: [`ptx-lessons-learned`](./ptx-lessons-learned/)
> **适用**: 所有涉及代码改动、API 迁移、状态修改的 OpenSpec change 实施
> **来源**: 2026-06-18 commit `f033312` 实战经验沉淀

### 必查项（实施前）

在开始实施 **任何 Phase 之前**，**必须**执行以下检查：

#### 1. 建立基线 worktree（如涉及重大重构）

**判定标准**（满足任一即需要基线）：
- 涉及多个文件、多个 invariant
- 涉及状态机重构（ThreadContext / WarpState / 屏障）
- 预计 ≥ 3 commit

**标准操作**：
```bash
# 1. 找到当前 baseline commit
git log --oneline -1
# 2. 建立基线 worktree
git worktree add .worktrees/baseline-check <baseline-commit>
# 3. 验证基线测试通过
cd .worktrees/baseline-check
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
cd build && ctest -L <relevant-label>  # 确认基线通过
```

#### 2. 加载相关领域 skill

根据本次实施的领域，加载对应 skill：

| 实施领域 | 加载的 skill |
|---------|------------|
| 屏障 / barrier | `ptx-barrier-mechanism` + `ptx-lessons-learned` |
| 指令 pipeline | `ptx-instruction-pipeline` + `ptx-lessons-learned` |
| PTX 语法 | `ptx-grammar-modification` + `ptx-lessons-learned` |
| 状态修改 | `state-modification-audit` + `ptx-lessons-learned` |
| 回归定位 | `regression-bisect` + `ptx-lessons-learned` |

### 必查项（实施中 - 每个 Phase）

#### A. 函数迁移时执行 Checklist A

**每个 Phase 开始前**：
```bash
# 1. 列出 baseline 函数所有 set_* / commit_* / force_* / lock_* 调用
grep -n "set_state\|set_pc\|set_next_pc\|commit_pc\|force_set_pc\|lock_guard\|unique_lock" <baseline-file>
# 2. 列出所有下游消费者
grep -rn "state == BAR_SYNC\|is_at_barrier\|is_blocked" src/ include/
# 3. 写行级 diff 计划到 commit message
```

**每个 Phase 完成后**：
- 验证 checklist 中每一项都已在新实现中保留
- 写"行级 diff 已确认"到 commit message

#### B. 测试失败诊断（按失败模式速查表）

| 症状 | 行动 |
|------|------|
| 集成测试断言 N+1 处挂起 | 查 `ptx-lessons-learned` 失败模式速查表 → 递归锁死锁 |
| `CHECK(!is_blocked)` 失败 | 查 → 跨模块状态翻译缺失 |
| `ctest -L <label>` 整体超时 | 立即切到 single-test + per-test timeout |
| Phase N 通过但 Phase N+1 失败 | 查 → 跨 Phase invariant 冲突 |
| `git revert` 后 `git status` 异常 | 查 → stash/pop 改变了 staged 状态 |

**任何失败**：先查 `ptx-lessons-learned` 失败模式速查表，再继续诊断。

#### C. 失败处理纪律

**任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit**：

```bash
# 1. 暂存当前 working tree 改动（如果有未完成的工作）
git stash
# 2. Revert 失败的 Phase commit
git revert --no-commit <failing-commit>
# 3. 验证 revert 后测试通过
cmake --build build && ctest -L <relevant-label>
# 4. 决定：放弃该 Phase 或重新设计
#    - 放弃：git revert --continue + 在 tasks.md 标记 [~] DEFERRED
#    - 重新设计：reset 改动，修改 design.md，重新提案
```

**绝对禁止**：
- ❌ 把"修复失败"的 commit 混入后续 commit（污染历史）
- ❌ 删除失败测试以"通过"
- ❌ 用 `as any` / `@ts-ignore` 抑制类型错误
- ❌ 跳过基线 worktree 直接修改代码

#### D. 注释纪律

每写一个 `set_state` / `set_pc` / `lock_guard` / 任何状态修改，**必须**写注释说明：
1. **消费者位置**（`grep` 出的所有下游）
2. **失败模式**（没设置会怎样）
3. **跨引用**（旧代码对照 / ADR 引用）

**判断标准**：
> 这条注释能否让 3 个月后的陌生人避免犯同样的错？
> - 是 → 写
> - 否 → 不写
> - 例外：警告"不要做 X"必须写

### 必查项（实施后 - 每个 commit 前）

#### Checklist D: Commit 前

```
□ 跑过 baseline worktree 对比（确认非本次引入的失败）
□ AGENTS.md 是否需要同步
□ ADR 是否需要追加
□ OpenSpec tasks.md 是否需要更新（标记完成的 Phase）
□ commit message 列出独立的 fix 编号
□ 如果是复合 fix（多个 bug 在一个 commit），在 commit message 详细列出每个 fix 的原因和影响
```

**commit message 模板（修复 bug 时）**：
```
fix(<area>): <one-line summary>

Three fixes bundled:

1. <Fix 1 title>:
   - Root cause: <specific file:line>
   - Symptom: <which test, what assertion>
   - Fix: <specific change>

2. <Fix 2 title>:
   ...

Test results (vs baseline <commit>):
- e2e: N/M pass
- integration: N/M pass (X pre-existing failures on baseline)
- unit: N/M pass
- Zero regressions introduced
```

### 自动化检查（在每个 commit 前）

```bash
# 1. 对比基线
cd .worktrees/baseline-check/build && ctest -R <affected-test>
cd <main-build> && ctest -R <affected-test>
# 对比输出，确认没有新增 FAIL

# 2. 检查 AGENTS.md 同步
grep -n "DO NOT\|MUST" src/ptxsim/<modified-area>/AGENTS.md

# 3. 检查 commit message 包含 fix 编号
git log -1 --format="%B" | grep "^[0-9]\."
```

### 与其他 skill 的关系

- **`ptx-lessons-learned`**: 本 skill 的源，提供所有 checklist 和失败模式
- **`adr-compliance-check`**: 在实施完成后使用，验证 ADR 合规
- **`ptx-barrier-mechanism`** / **`ptx-instruction-pipeline`**: 领域知识，实施时必读
- **`regression-bisect`**: 当测试回归时使用，定位根因
- **`state-modification-audit`**: 当怀疑状态被意外修改时使用

### 📋 实施日志模板

在实施每个 Phase 时，建议创建 `docs/superpowers/plans/<date>-<change-name>-fix.md` 记录：
1. 每个 Phase 的 commit hash + 状态
2. 失败时的诊断过程
3. 基线对比结果
4. 推迟的 Phase 原因

（参考 `docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md`）
