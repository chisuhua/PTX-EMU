---
name: openspec-archive-change
description: Archive a completed change in the experimental workflow. Use when the user wants to finalize and archive a change after implementation is complete.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.4.1"
---

Archive a completed change in the experimental workflow.

**Input**: Optionally specify a change name. If omitted, check if it can be inferred from conversation context. If vague or ambiguous you MUST prompt for available changes.

**Steps**

1. **If no change name provided, prompt for selection**

   Run `openspec list --json` to get available changes. Use the **AskUserQuestion tool** to let the user select.

   Show only active changes (not already archived).
   Include the schema used for each change if available.

   **IMPORTANT**: Do NOT guess or auto-select a change. Always let the user choose.

2. **Check artifact completion status**

   Run `openspec status --change "<name>" --json` to check artifact completion.

   Parse the JSON to understand:
   - `schemaName`: The workflow being used
   - `planningHome`, `changeRoot`, `artifactPaths`, and `actionContext`: path and scope context
   - `artifacts`: List of artifacts with their status (`done` or other)

   If status reports `actionContext.mode: "workspace-planning"`, explain that workspace archive is not supported in this slice and STOP. Do not move workspace changes into repo-local archives or edit linked repos.

   **If any artifacts are not `done`:**
   - Display warning listing incomplete artifacts
   - Use **AskUserQuestion tool** to confirm user wants to proceed
   - Proceed if user confirms

3. **Check task completion status**

   Read the tasks file (typically `tasks.md`) to check for incomplete tasks.

   Count tasks marked with `- [ ]` (incomplete) vs `- [x]` (complete).

   **If incomplete tasks found:**
   - Display warning showing count of incomplete tasks
   - Use **AskUserQuestion tool** to confirm user wants to proceed
   - Proceed if user confirms

   **If no tasks file exists:** Proceed without task-related warning.

4. **Assess delta spec sync state**

   Use `artifactPaths.specs.existingOutputPaths` from status JSON to check for delta specs. If none exist, proceed without sync prompt.

   **If delta specs exist:**
   - Compare each delta spec with its corresponding main spec at `openspec/specs/<capability>/spec.md`
   - Determine what changes would be applied (adds, modifications, removals, renames)
   - Show a combined summary before prompting

   **Prompt options:**
   - If changes needed: "Sync now (recommended)", "Archive without syncing"
   - If already synced: "Archive now", "Sync anyway", "Cancel"

   If user chooses sync, use Task tool (subagent_type: "general-purpose", prompt: "Use Skill tool to invoke openspec-sync-specs for change '<name>'. Delta spec analysis: <include the analyzed delta spec summary>"). Proceed to archive regardless of choice.

5. **Perform the archive**

   Create an `archive` directory under `planningHome.changesDir` if it doesn't exist:
   ```bash
   mkdir -p "<planningHome.changesDir>/archive"
   ```

   Generate target name using current date: `YYYY-MM-DD-<change-name>`

   **Check if target already exists:**
   - If yes: Fail with error, suggest renaming existing archive or using different date
   - If no: Move `changeRoot` to the archive directory

   ```bash
   mv "<changeRoot>" "<planningHome.changesDir>/archive/YYYY-MM-DD-<name>"
   ```

6. **Display summary**

   Show archive completion summary including:
   - Change name
   - Schema that was used
   - Archive location
   - Whether specs were synced (if applicable)
   - Note about any warnings (incomplete artifacts/tasks)

**Output On Success**

```
## Archive Complete

**Change:** <change-name>
**Schema:** <schema-name>
**Archived to:** the archive path derived from `planningHome.changesDir`/YYYY-MM-DD-<name>/
**Specs:** ✓ Synced to main specs (or "No delta specs" or "Sync skipped")

All artifacts complete. All tasks complete.
```

**Guardrails**
- Always prompt for change selection if not provided
- Use artifact graph (openspec status --json) for completion checking
- Don't block archive on warnings - just inform and confirm
- Preserve .openspec.yaml when moving to archive (it moves with the directory)
- Show clear summary of what happened
- If sync is requested, use openspec-sync-specs approach (agent-driven)
- If delta specs exist, always run the sync assessment and show the combined summary before prompting

---

## 📝 Archive-Time Lessons Integration (强制)

> **加载 skill**: [`ptx-lessons-learned`](./ptx-lessons-learned/)
> **适用**: 所有归档阶段的 OpenSpec change
> **目的**: 防止"经验教训随 change 归档而消失"，把实战经验沉淀到永久文档

### 必查项（归档前 - 强制 Prompt）

在执行归档（mv change 到 archive 目录）之前，**必须**询问用户：

> "在归档此 change 前，是否要生成 postmortem 段落（推荐）？
>
> **会做什么**:
> 1. 列出本次实施中遇到的所有 bug 和它们的根因
> 2. 在对应 ADR 追加 'Postmortem' 段落（如 `docs/adr/NNNN-*.md`）
> 3. 把新发现的失败模式加入 [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md) 和 [`ptx-lessons-learned`](./ptx-lessons-learned/) skill
> 4. 推迟的任务标记为 `[~] DEFERRED`（在 tasks.md）并附引用
>
> **不做什么**:
> - 不修改任何代码
> - 不影响归档本身
>
> 选 [是] / [否] / [推迟]"

### Postmortem 模板（如用户同意）

在对应 ADR 文件追加段落（参考 `docs/adr/ADR-0008-barrier-semantics.md` §"2026-06-18 Postmortem"）：

```markdown
## YYYY-MM-DD Postmortem：<change-name> 实施回顾

### 实施回顾

| Phase | 内容 | Commit | 状态 |
|-------|------|--------|------|
| 1 | ... | `<hash>` | ✅ / ❌ → revert |
| 2 | ... | ... | ... |

### 推迟原因（如有）

<哪些 Phase 推迟，原因是什么，给未来实施者的指引>

### 同期发现的独立 bug（如有）

#### Bug 1: <title>
- **位置**: `<file>:<line>`
- **根因**: <具体代码 + 为什么是 bug>
- **修复**: <具体改动>
- **教训**: <沉淀到 lessons-learned.md 的哪一节>
- **测试**: <哪个测试暴露>

#### Bug 2: ...

### 验证结果（最终 commit）

| 类别 | 通过 | 失败（基线也失败）|
|------|------|------------------|
| e2e | N/M | 0 |
| integration | N/M | <pre-existing> |
| unit | N/M | <pre-existing> |
| **本次引入的回归** | **0** | — |

### 相关链接

- [docs/dev-process/lessons-learned.md](../../docs/dev-process/lessons-learned.md) — 完整经验沉淀
- [docs/superpowers/plans/<date>-<change-name>-fix.md](../../docs/superpowers/plans/) — 修复计划
- [openspec/changes/<change-name>/](../../openspec/changes/) — change artifacts
```

### Lessons-Learned 更新模板（如发现新的失败模式）

在 [`docs/dev-process/lessons-learned.md`](../../docs/dev-process/lessons-learned.md) 追加章节：

```markdown
## N. <new-lesson-title>

### 现象
<具体 bug 表现 + 最小重现>

### 教训
- <要点 1>
- <要点 2>

### 真实案例
- **bug 表现**: <测试名 + 断言>
- **修复**: <改动>
- **commit**: <hash>
```

并同步更新：
- `ptx-lessons-learned` skill 的"快速决策树"和"失败模式速查表"
- `ptx-lessons-learned` skill 的"checklist X"（如适用）
- 相关的领域 skill（如涉及屏障则更新 `ptx-barrier-mechanism`）

### tasks.md 推迟标记模板

```markdown
## N. <Phase 标题>

> **⚠️ DEFERRED (YYYY-MM-DD)**: 实施 commit `<hash>` 引入 N 个回归（基线通过），已在 commit `<hash>` 中 revert。详细原因见 [`<ADR-file>` §YYYY-MM-DD Postmortem](../../../docs/adr/...) 和 [`lessons-learned.md`](../../docs/dev-process/lessons-learned.md)。
>
> **失败模式**: <具体表现>
>
> **未来实施**: 单独建 change `<new-change-name>`，<下一步指引>。

- [~] N.1 <task> — **DEFERRED**
- [~] N.2 <task> — **DEFERRED**
- [x] N.M **回退**: commit `<hash>` revert Phase N 实施；<恢复旧路径>
```

### 与其他 skill 的关系

- **`ptx-lessons-learned`**: 提供 postmortem 模板和 lessons-learned 章节模板
- **`adr-compliance-check`**: 在实施完成后使用，确保 ADR 合规
- **本 skill 的 archive 阶段**: 是 postmortem 沉淀的"强制钩子"，确保经验不丢失

### 为什么必须在 archive 阶段做？

1. **避免经验丢失**: 实施完就归档，细节容易随时间淡忘
2. **强制沉淀**: 询问 prompt 让用户主动思考"我学到了什么"
3. **建立文档网络**: ADR ↔ lessons-learned ↔ tasks.md 的交叉引用形成知识网
4. **降低未来相似 bug 概率**: 未来 agent 加载 `ptx-lessons-learned` 时看到具体案例

### 反例（不应做）

- ❌ 跳过 postmortem prompt 直接归档
- ❌ 只写"实施完成"而不总结教训
- ❌ 把"经验"写在 PR description 而不是永久文档
- ❌ 推迟的任务不标记 `[~] DEFERRED`，导致未来重复实施
