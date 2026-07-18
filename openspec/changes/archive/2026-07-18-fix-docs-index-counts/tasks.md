# 实施任务

## Phase 1: 修正 docs/README.md §文档导航 表格

- [x] Task 1.1: 修正 adr/ 行 — 15→22, 描述 "ADR-0001~0015"→"ADR-0001~0021（0017 缺失）"
- [x] Task 1.2: 修正 skills/ 行 — 8→3, 描述加注 "人类可读导航，可加载技能在 .opencode/skills/"
- [x] Task 1.3: 修正 superpowers/ 行 — 6→25, 描述加 "hsk-drafts"
- [x] Task 1.4: 修正 archive/ 行 — 50+→56
- [x] Task 1.5: 修正 roadmap/ 行 — 1→2
- [x] Task 1.6: 修正 dev-process/ 行 — 2→3, 描述加 "post-tcgen05-roadmap"
- [x] Task 1.7: 修正 audits/ 行 — 4→5, 描述加 "tcgen05-infra-audit"
- [x] Task 1.8: 第 63 行 "自动统计" 声明加注 "注: 当前校验脚本仅检查子目录名，文件数需手动维护"

## Phase 2: 修正 docs/adr/README.md

- [x] Task 2.1: 第 99 行 "ADR 总数: 19"→"20"
- [x] Task 2.2: 索引表补 ADR-0013 (statement-factory-test-unification) 条目
- [x] Task 2.3: 更新 Active/Accepted/Proposed 分类计数（如 ADR 状态已变更）

## Phase 3: 修正 docs/archive/README.md

- [x] Task 3.1: 修正子目录表格 — phase-plans 8→14, code-reviews 12→6, ptx-instruction-reference 19→13, misc 12→14
- [x] Task 3.2: 表格补 2026-04-simt-v2 子目录条目（6 文件）

## Phase 4: 验证

- [x] Task 4.1: 确认修改后 docs/README.md 中 16 个目录计数与 `find -type f | wc -l` 一致
- [x] Task 4.2: 确认 docs/adr/README.md ADR 索引表条目数 = 20
- [x] Task 4.3: git diff --stat 确认仅 3 个 .md 文件被修改
