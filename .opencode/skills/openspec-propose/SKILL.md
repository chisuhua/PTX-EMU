---
name: openspec-propose
description: Propose a new change with all artifacts generated in one step. Use when the user wants to quickly describe what they want to build and get a complete proposal with design, specs, and tasks ready for implementation.
license: MIT
compatibility: Requires openspec CLI.
metadata:
  author: openspec
  version: "1.0"
  generatedBy: "1.4.1"
---

Propose a new change - create the change and generate all artifacts in one step.

I'll create a change with artifacts:
- proposal.md (what & why)
- design.md (how)
- tasks.md (implementation steps)

When ready to implement, run /opsx-apply

---

**Input**: The user's request should include a change name (kebab-case) OR a description of what they want to build.

**Steps**

1. **If no clear input provided, ask what they want to build**

   Use the **AskUserQuestion tool** (open-ended, no preset options) to ask:
   > "What change do you want to work on? Describe what you want to build or fix."

   From their description, derive a kebab-case name (e.g., "add user authentication" → `add-user-auth`).

   **IMPORTANT**: Do NOT proceed without understanding what the user wants to build.

2. **Create the change directory**
   ```bash
   openspec new change "<name>"
   ```
   This creates a scaffolded change in the planning home resolved by the CLI with `.openspec.yaml`.

3. **Get the artifact build order**
   ```bash
   openspec status --change "<name>" --json
   ```
   Parse the JSON to get:
   - `applyRequires`: array of artifact IDs needed before implementation (e.g., `["tasks"]`)
   - `artifacts`: list of all artifacts with their status and dependencies
   - `planningHome`, `changeRoot`, `artifactPaths`, and `actionContext`: path and scope context. Use these instead of assuming repo-local paths.

4. **Create artifacts in sequence until apply-ready**

   Use the **TodoWrite tool** to track progress through the artifacts.

   Loop through artifacts in dependency order (artifacts with no pending dependencies first):

   a. **For each artifact that is `ready` (dependencies satisfied)**:
      - Get instructions:
        ```bash
        openspec instructions <artifact-id> --change "<name>" --json
        ```
      - The instructions JSON includes:
        - `context`: Project background (constraints for you - do NOT include in output)
        - `rules`: Artifact-specific rules (constraints for you - do NOT include in output)
        - `template`: The structure to use for your output file
        - `instruction`: Schema-specific guidance for this artifact type
        - `resolvedOutputPath`: Resolved path or pattern to write the artifact
        - `dependencies`: Completed artifacts to read for context
      - Read any completed dependency files for context
      - Create the artifact file using `template` as the structure and write it to `resolvedOutputPath`
      - Apply `context` and `rules` as constraints - but do NOT copy them into the file
      - Show brief progress: "Created <artifact-id>"

   b. **Continue until all `applyRequires` artifacts are complete**
      - After creating each artifact, re-run `openspec status --change "<name>" --json`
      - Check if every artifact ID in `applyRequires` has `status: "done"` in the artifacts array
      - Stop when all `applyRequires` artifacts are done

   c. **If an artifact requires user input** (unclear context):
      - Use **AskUserQuestion tool** to clarify
      - Then continue with creation

5. **Show final status**
   ```bash
   openspec status --change "<name>"
   ```

**Output**

After completing all artifacts, summarize:
- Change name and location
- List of artifacts created with brief descriptions
- What's ready: "All artifacts created! Ready for implementation."
- Prompt: "Run `/opsx-apply` or ask me to implement to start working on the tasks."

**Artifact Creation Guidelines**

- Follow the `instruction` field from `openspec instructions` for each artifact type
- The schema defines what each artifact should contain - follow it
- Read dependency artifacts for context before creating new ones
- Use `template` as the structure for your output file - fill in its sections
- **IMPORTANT**: `context` and `rules` are constraints for YOU, not content for the file
  - Do NOT copy `<context>`, `<rules>`, `<project_context>` blocks into the artifact
  - These guide what you write, but should never appear in the output

**Guardrails**
- Create ALL artifacts needed for implementation (as defined by schema's `apply.requires`)
- Always read dependency artifacts before creating a new one
- If context is critically unclear, ask the user - but prefer making reasonable decisions to keep momentum
- If a change with that name already exists, ask if user wants to continue it or create a new one
- Verify each artifact file exists after writing before proceeding to next

---

## 🎯 Design-Time Lessons Integration (强制)

> **加载 skill**: [`ptx-lessons-learned`](./ptx-lessons-learned/)
> **适用**: 所有涉及 API 迁移、状态修改、多 Phase 推进的 OpenSpec change
> **来源**: 2026-06-18 commit `f033312` 实战经验沉淀

### 必查项（proposal 阶段）

在生成 `proposal.md` 和 `design.md` 之前，**必须**回答以下问题（回答写入 proposal.md 的 "## Risks" 章节或 design.md 的相应位置）：

#### A. 是否涉及函数/类/API 迁移？

**判定**: 任务中包含"迁移"、"重写"、"切换到新 API"、"替换旧实现"等关键词。

**如果是**，必须在 `design.md` 的 "Migration Plan" 章节提供：
1. **Baseline 函数清单**: 所有要迁移的函数 + 完整方法签名
2. **逐行 diff 计划**: 列出每个函数中所有 `set_*` / `commit_*` / `force_*` / `lock_*` 调用，逐一映射到新实现
3. **跨模块状态翻译表**: 如果函数操作 `ThreadContext::state` / `WarpState` / 互斥量，列出所有下游消费者
4. **回退策略**: 每个 Phase 独立 commit、独立可 revert

**检查命令**（自动验证设计完整性）:
```bash
# 列出 baseline 函数所有 set_state / set_pc 调用
grep -n "set_state\|set_pc\|set_next_pc\|commit_pc\|force_set_pc" <baseline-file>
# 列出所有锁点
grep -n "lock_guard\|unique_lock\|lock()\|unlock()" <baseline-file>
# 列出所有下游消费者
grep -rn "state == BAR_SYNC\|is_at_barrier\|is_blocked" src/ include/
```

#### B. 是否涉及状态修改（ThreadContext / WarpState / 屏障 / 互斥量）？

**判定**: 任务涉及"修改状态"、"同步"、"屏障"、"锁"、"atomic"。

**如果是**，必须在 proposal 列出：
1. **状态翻译路径**: `ThreadContext::state` → `warp_state.threads[i].*` 的完整翻译规则（参考 `src/ptxsim/core/AGENTS.md` DUAL STATE MECHANISM）
2. **invariant 清单**: 哪些全局不变式会被影响（如 `BUG-POSTBARRIER-TWOHALVES`、`BUG-RECONVERGENCE-SIMPLEGEMM`）
3. **回归测试计划**: 类型一/二/三测试各覆盖哪些 invariant

#### C. 是否涉及多 Phase 推进（预计 ≥ 3 commit）？

**如果是**，必须在 tasks.md 提供：
1. **Phase 拆分**: 每个 Phase 的独立 commit + 独立可回退性
2. **基线 worktree 计划**: `git worktree add .worktrees/baseline-check <baseline-commit>`
3. **失败处理策略**: 任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit

#### D. 文档同步清单

任何代码改动必须列出同步的文档：

| 代码改动 | 同步的文档 |
|---------|-----------|
| 修改 handler / 指令 | `src/ptxsim/instructions/AGENTS.md` |
| 修改核心数据结构 | `src/ptxsim/core/AGENTS.md`（如涉及 invariant） |
| 架构变更 | 对应 ADR 追加段落 |
| OpenSpec task 状态变更 | `openspec/changes/.../tasks.md` |

### 自动化检查（在 propose 阶段验证）

**proposal 完成后**，运行以下检查确保 design-time 教训已应用：

```bash
# 1. 检查 proposal 是否列出基线对比
grep -i "baseline\|worktree" openspec/changes/<name>/proposal.md

# 2. 检查 design.md 是否列出逐行 diff 计划
grep -i "set_state\|lock_guard\|set_pc" openspec/changes/<name>/design.md

# 3. 检查 tasks.md 是否分 Phase
grep -c "^## [0-9]" openspec/changes/<name>/tasks.md  # 应该 >= 3

# 4. 检查是否引用 ptx-lessons-learned
grep -i "lessons-learned\|ptx-lessons-learned" openspec/changes/<name>/*.md
```

如果以上检查失败，**必须**更新 proposal 后再继续。

### 与其他 skill 的关系

- **`ptx-lessons-learned`**: 本 skill 的源，提供详细 checklist 和失败模式速查表
- **`adr-compliance-check`**: 在实施完成后使用，确保实现符合 ADR
- **`ptx-barrier-mechanism`** / **`ptx-instruction-pipeline`**: 领域知识，propose 阶段必读

---

## 📋 Proposal 模板增强

**新增章节**（在标准 OpenSpec proposal 模板中追加）：

```markdown
## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性（如适用）
- [ ] Baseline 函数所有 set_*/commit_*/force_*/lock_* 调用已列出
- [ ] 逐行 diff 计划已写入 design.md
- [ ] 跨模块状态翻译路径已文档化

### 多 Phase 推进（如适用）
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明
- [ ] 基线 worktree 命令已记录
- [ ] 失败处理策略（revert 该 Phase，不混入后续 commit）已说明

### 文档同步
- [ ] AGENTS.md 同步项已列出
- [ ] ADR 追加段落已规划
- [ ] tasks.md Phase 状态变更已说明
```
