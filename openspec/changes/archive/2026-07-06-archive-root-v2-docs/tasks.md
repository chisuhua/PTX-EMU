# Tasks: Archive Root v2.0 Docs

> **Type**: 3-Phase 文档归档 change（纯位置移动）
> **HEAD baseline**: `c338f12`
> **Risk**: 🟢 极低（git mv 保留历史）
> **Lessons-learned**: §21 + Checklist E/F/G

---

## Phase 0: Artifacts Git-Tracking

- [ ] 0.1 创建工作分支
  ```bash
  git checkout -b docs/archive-root-v2-docs
  ```
- [ ] 0.2 git-tracked artifacts
  ```bash
  git add openspec/changes/archive-root-v2-docs/
  git ls-files openspec/changes/archive-root-v2-docs/
  ```
- [ ] 0.3 commit artifacts
  ```bash
  git commit -m "docs(openspec): add archive-root-v2-docs artifacts"
  ```
- [ ] 0.4 验证 0 调用方（per Checklist A 防误删）
  ```bash
  grep -rn "workflow-state\.md\|task_plan\.md\|BUILD-VERIFICATION-v2\.0\.md\|RELEASE-CHECKLIST-v2\.0\.md\|PTX_PARSING_FIX_REPORT\.md" \
    README.md AGENTS.md docs/ src/ tests/ openspec/ \
    --include="*.md" --include="*.txt" --include="*.cmake" --include="*.cpp" --include="*.h" \
    2>/dev/null
  # 期望: 0 或 1 行（仅引用文件名的 1 行）
  ```

---

## Phase 1: git mv 5 个文档 + 创建 archive README（Fix #1）

> **Risk**: 🟢 极低（git mv）

- [ ] 1.1 创建实施 worktree
  ```bash
  cd /workspace/project/PTX-EMU
  git worktree add .worktrees/archive-v2-docs-impl docs/archive-root-v2-docs
  cd .worktrees/archive-v2-docs-impl
  ```
- [ ] 1.2 **创建 archive 子目录**
  ```bash
  mkdir -p docs/archive/2026-04-simt-v2
  ```
- [ ] 1.3 **git mv 5 个文档**（用 git mv 保留历史）
  ```bash
  git mv workflow-state.md docs/archive/2026-04-simt-v2/
  git mv task_plan.md docs/archive/2026-04-simt-v2/
  git mv BUILD-VERIFICATION-v2.0.md docs/archive/2026-04-simt-v2/
  git mv RELEASE-CHECKLIST-v2.0.md docs/archive/2026-04-simt-v2/
  git mv PTX_PARSING_FIX_REPORT.md docs/archive/2026-04-simt-v2/
  ```
- [ ] 1.4 **创建 archive README.md**
  ```bash
  cat > docs/archive/2026-04-simt-v2/README.md << 'EOF'
  # SIMT v2.0 Era Archives (2026-04)

  > **归档原因**: 这些文档自 2026-04-11 (SIMT v2.0 早期) 以来未更新，与当前项目状态严重脱节（90 条技术债务 + >739 测试 + 多个已归档 OpenSpec changes）。
  >
  > **当前位置**: 根目录的文档已移到本目录；历史决策参考价值保留，git log + git blame 可追溯。

  ## 归档文档清单

  | 文件 | 最后更新 | 归档原因 |
  |------|----------|----------|
  | `workflow-state.md` | 2026-05-25 | v4 工作流状态，引用已删除文件 + 已归档 changes |
  | `task_plan.md` | 2026-04-11 | 旧调试会话的 PTX 语法修复计划 |
  | `BUILD-VERIFICATION-v2.0.md` | 2026-04-11 | 声称"零技术债务"，与当前 90 条不符 |
  | `RELEASE-CHECKLIST-v2.0.md` | 2026-04-11 | 声称"38 测试通过"，与实际 >739 测试不符 |
  | `PTX_PARSING_FIX_REPORT.md` | 2026-04-11 | 旧报告，已关闭缺陷 |

  ## 当前状态参考

  - 最新审计: `docs/audits/debt-audit-2026-07-02.md`
  - 剩余债务: `docs/roadmap/post-phase3-debt-roadmap.md`
  EOF
  ```
- [ ] 1.5 **更新 docs/archive/README.md**（添加新子目录索引）
- [ ] 1.6 **验证 0 broken references**
  ```bash
  grep -E "workflow-state|task_plan|BUILD-VERIFICATION|RELEASE-CHECKLIST|PTX_PARSING_FIX" README.md AGENTS.md
  # 期望: 无匹配
  ```
- [ ] 1.7 **Phase 1 验证**（不需要 ctest）
  ```bash
  # 检查根目录
  ls *.md 2>/dev/null
  # 期望: README.md AGENTS.md GEMINI.md QODER.md（不包含已归档的 5 文件）
  ```
- [ ] 1.8 **Commit Fix #1**
  ```bash
  git commit -am "docs(archive): move 5 root v2.0 docs to docs/archive/2026-04-simt-v2/ (Fix #1)

  Moved (git mv, history preserved):
  - workflow-state.md (last updated 2026-05-25)
  - task_plan.md (2026-04-11)
  - BUILD-VERIFICATION-v2.0.md (2026-04-11)
  - RELEASE-CHECKLIST-v2.0.md (2026-04-11)
  - PTX_PARSING_FIX_REPORT.md (2026-04-11)

  Created:
  - docs/archive/2026-04-simt-v2/README.md (explains archival reason + file list)

  Updated:
  - docs/archive/README.md (added new subdirectory)

  Verified:
  - Root README.md + AGENTS.md have no broken references (grep check)

  Per lessons-learned §21 + docs/README.md archival rules.
  "
  ```

---

## Phase 2: 文档同步（Fix #2）

- [ ] 2.1 更新 docs/audits/debt-audit-2026-07-02.md
  - 标记根目录 v2.0 docs RESOLVED（引用 Phase 1 commit hash）
- [ ] 2.2 更新 docs/roadmap/post-phase3-debt-roadmap.md
  - 从剩余债务列表移除相关条目
- [ ] 2.3 Commit Fix #2
  ```bash
  git commit -am "docs(archive): sync debt audit + roadmap post-Fix #1 (Fix #2)"
  ```

---

## Phase 3: Archive + Merge

- [ ] 3.1 Archive change
  ```bash
  openspec archive archive-root-v2-docs --yes
  ```
- [ ] 3.2 清理 worktree
  ```bash
  git worktree remove .worktrees/archive-v2-docs-impl
  ```
- [ ] 3.3 Merge to main
  ```bash
  git checkout main
  git merge --no-ff docs/archive-root-v2-docs
  ```

---

## 风险缓解矩阵

| 风险 | 缓解任务 | 验证 |
|------|---------|------|
| R1: 隐藏引用 | 0.4 + 1.6 | grep 检查 |
| R2: git mv 误用 | 1.3 | 必须用 git mv |
| R3: 未来读者不知道归档原因 | 1.4 | README.md 解释 |