# 实施任务

## Phase 1: 删除冗余文件

- [x] Task 1.1: `git rm GEMINI.md`
- [x] Task 1.2: `git rm QODER.md`
- [x] Task 1.3: 确认无其他文件引用 GEMINI.md 或 QODER.md（grep -rn "GEMINI\|QODER" --include="*.md" | grep -v openspec/changes）

## Phase 2: AGENTS.md 去重

- [x] Task 2.1: 删除 AGENTS.md 行 40-56（与行 10-26 完全重复的 OpenSpec 流程章节）
- [x] Task 2.2: 确认删除后 AGENTS.md 结构完整，无引用断裂

## Phase 3: 归档过期文档

- [x] Task 3.1: `git mv docs/PROJECT-COMPLETION-SUMMARY.md docs/archive/2026-07-18-project-completion-summary.md`
- [x] Task 3.2: 确认 docs/README.md 中无指向该文件的链接（或更新链接为 archive 路径）

## Phase 4: 验证

- [x] Task 4.1: git diff --stat 确认净 -3 文件, -55 行
- [x] Task 4.2: 确认根目录 .md 文件仅剩 AGENTS.md + README.md
