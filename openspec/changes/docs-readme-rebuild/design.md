## Context

PTX-EMU 在 2026-06 文档重组（commit 历史中 `docs/DOCUMENTATION-REORGANIZATION-SUMMARY.md` 记录）后陆续新建多个子目录，但 `docs/README.md` 索引未同步。`openspec/specs/pc-api/` 是 v1.4.1 升级后唯一 active spec，文档维护目前无正式 spec 约束。当前 `docs/audits/debt-audit-2026-07-02.md` 已识别 7 条 P1-P2 文档债（docs/README.md 索引遗漏 9/16 子目录、统计数据全面过时、6 个 OpenSpec 孤儿 change 等）。

本次 change 是 docs/ 维护流程的首次形式化：将"README 应覆盖全部子目录"从隐式约定提升为可测试 spec（`docs-discoverability`），后续任何 docs/ 子目录新增必须同步更新 README。

**当前状态**（基于 git HEAD `3f46a3e`）：
- `docs/README.md` 171 行 36 表格行 → 覆盖 6/17 子目录
- `docs/skills/README.md` 列 9 技能 → `.opencode/skills/` 实际 18 个
- `tests/archive/` 不存在但 AGENTS.md 引用
- 6 个 OpenSpec archive change 缺 README.md
- 2 个旧文档（HEALTH-AUDIT, PROJECT-COMPLETION-SUMMARY）需 banner

**目标状态**：
- `docs/README.md` 覆盖全部 17 子目录，每个 1-2 句功能描述
- 统计信息自动生成（`grep -r "TEST_CASE" tests/ | wc -l`）
- 6 个 OpenSpec 孤儿 change 各自有 README 引用实施 commit
- 所有 README 内部链接通过 `markdown-link-check` 验证

## Goals / Non-Goals

**Goals:**
- 新人 5 分钟内能从 `docs/README.md` 找到 ADR/审计/PTX 参考/任何子目录
- 任何 `docs/<subdir>/` 新建/删除/重命名时，`docs/README.md` 自动同步（约定而非工具强制）
- OpenSpec 归档的孤儿 change 有可追溯的历史指针
- HEALTH-AUDIT 等含勘误/过时的文档有清晰 banner 指向权威版本

**Non-Goals:**
- 不实现 docs/ 自动同步工具（仅约定）
- 不重写任何子目录内容（仅重写 README 索引 + 加 banner）
- 不清理 docs/archive/ 内容（仅 README.md 索引数量同步）
- 不修改 OpenSpec archive 目录结构（仅添加缺失的 README.md）
- 不影响 C++ 代码、CMake 构建、PTX 解析、模拟器执行

## Decisions

### Decision 1: docs/README.md 索引组织方式

**选择**: 按**功能类别分组**（不是字母序）

**组织结构**（从上到下）:
1. **核心文档**：AGENTS.md (项目根), README.md (项目根)
2. **架构与设计**：adr/, architecture/, technical_design/
3. **开发与测试**：developer-guide/, testing/, plans/
4. **报告与审计**：reports/, audits/
5. **流程与技能**：dev-process/, skills/, superpowers/
6. **PTX 与解析**：ptx/
7. **roadmap**：roadmap/
8. **历史归档**：archive/

**为什么不字母序**:
- 字母序在 17 个子目录时已超出"扫一眼"长度，分组后每组 2-3 个，认知负担低
- 功能分组反映"开发者从入门到深入"的路径：架构 → 开发 → 报告 → 流程 → PTX → roadmap → 归档
- 字母序隐含"这些目录是平等的"，但 adr/ vs archive/ 的重要性差异巨大

**考虑的替代方案**:
- 纯字母序：简单但认知负担高
- 单一平铺表格：17 行过长
- **采用：7 个分组**，每组 2-3 个子目录

### Decision 2: 6 个 OpenSpec 孤儿 change 的处理

**选择**: 每个孤儿 change 添加 `README.md`，内容包含：
- 1 句话目的（来自 proposal.md）
- 实施 commit 哈希（`git log --all --oneline -- <change-name>/` 验证）
- 关联 ADR 编号（如有）
- 已知偏差/教训链接

**为什么不修改 OpenSpec CLI**:
- 6 个孤儿都是历史归档，OpenSpec v1.4.1 已修复 `openspec new` 流程
- 修改 CLI 是过度工程
- 仅缺 README 不影响 OpenSpec 工作流

**考虑的替代方案**:
- 删除孤儿 change：会丢失历史决策记录，违反 ADR-0013 原则
- 补充 design.md：原文已不存在，无法重建
- 改用 git 历史作为唯一参考：但新人不会想到查 git log
- **采用：每个加 README.md 引用 commit**

### Decision 3: tests/archive/ 路径冲突解决

**选择**: 创建 `tests/archive/.gitkeep` + AGENTS.md 引用说明

**理由**:
- 删除 AGENTS.md 引用会丢失"已归档测试"的概念，对未来扩展不友好
- 创建 `.gitkeep` 是最低成本方式（2 字节文件，保留目录作为未来归档位）
- 与项目其他 `.gitkeep` 模式一致

**考虑的替代方案**:
- 完全删除 AGENTS.md 引用：未来想归档测试时需重新建立约定
- 把 `tests/archive/three_mode_testing/` 实际迁移：原文已不存在（迁至 integration/）
- **采用：建 .gitkeep + 加注释说明用途**

### Decision 4: Banner 添加方式

**选择**: 在文档顶部（`# Title` 后第一行）添加 `> **⚠️ ...**` 引用块

**格式**:
```markdown
# Document Title

> **⚠️ 已过期 (2026-07)** — 此文档描述 v2.0 完成状态，但项目仍在 Phase 3 债务修复。
> 最新状态参考 [docs/audits/debt-audit-2026-07-02.md](../audits/debt-audit-2026-07-02.md)
>
> **勘误**: 8 个事实错误已修正，详见 [HEALTH-AUDIT-2026-06-21-ERRATA.md](HEALTH-AUDIT-2026-06-21-ERRATA.md)

(content...)
```

**为什么不修改内容**:
- 修改内容会丢失历史视角
- banner 是非侵入式警告
- 与 `cleanup-barrier-review.md` 已用的 banner 模式一致

## Risks / Trade-offs

| Risk | Impact | Mitigation |
|------|--------|-----------|
| **R1**: README 索引表格链接断链 | 新人无法跳转 | 提交前用 `markdown-link-check` 验证 |
| **R2**: 6 个孤儿 change 引用错误的 commit hash | 历史追溯失败 | 每个 hash 用 `git log --all --oneline -- <path>` 双验证 |
| **R3**: 删除手写统计信息被认为是"信息丢失" | 反对声音 | 提供自动生成脚本（Makefile target）保留同样信息 |
| **R4**: docs/ 子目录未来新增时 README 不同步 | 债务复现 | spec `docs-discoverability` 添加"任何子目录新增必须更新 README"约定 |
| **R5**: 修改 AGENTS.md 对 `tests/archive/` 引用 | 文档不一致 | Decision 3 决策记录 + 同步 .opencode/skills/ptx-lessons-learned/ |
| **R6**: 17 个子目录分组可能不符合未来 20+ 目录场景 | 扩展性 | 分组规则写入 spec，扩展时按规则重新分组即可 |

## 影响范围

| 组件 | 影响类型 |
|------|---------|
| `docs/README.md` | 重写（索引表格从 6 子目录 → 17 子目录） |
| `docs/skills/README.md` | 重写（9 技能 → 18 技能 + 标注 disabled） |
| `AGENTS.md` | 小改（tests/archive/ 引用说明） |
| `docs/audits/HEALTH-AUDIT-2026-06-21.md` | 加 banner（不修改内容） |
| `docs/PROJECT-COMPLETION-SUMMARY.md` | 加 banner（不修改内容） |
| `openspec/changes/archive/2026-06-24-phase3-*/README.md` | 新建 5 个 |
| `openspec/changes/archive/2026-06-24-integrate-barrier-module-cta-warp/README.md` | 新建 1 个（说明 superseded by 2026-06-20） |
| `tests/archive/.gitkeep` | 新建 |
| `openspec/specs/docs-discoverability/spec.md` | 新建（v1.4.1 spec） |
| `scripts/check-docs-index.sh` | 新建（自动验证脚本，optional） |

## Migration Plan

**前置**:
1. 基线 worktree 检查（参考 `.worktrees/fix-pre-p0-baseline` 可复用）
2. `git status` 确认工作区干净

**Phase 1**（独立 commit）: docs/README.md 重写
- 1.1 重写索引表格
- 1.2 验证所有链接（`markdown-link-check docs/README.md`）
- 1.3 commit: `docs(readme): expand index to 17 subdirs (Fix #1)`

**Phase 2**（独立 commit）: docs/skills/README.md 同步
- 2.1 扩展 9 技能 → 18 技能
- 2.2 标注 three-mode-testing 为 disabled
- 2.3 commit: `docs(skills): sync to 18 skills + mark three-mode-testing disabled (Fix #2)`

**Phase 3**（独立 commit）: tests/archive/ + AGENTS.md
- 3.1 创建 `tests/archive/.gitkeep`
- 3.2 修改 AGENTS.md 添加用途说明
- 3.3 commit: `chore(tests): create tests/archive/ as historical test archive (Fix #3)`

**Phase 4**（独立 commit）: 6 个 OpenSpec 孤儿 README
- 4.1 5 个 phase3-* change 加 README
- 4.2 integrate-barrier-module-cta-warp 加 superseded 说明
- 4.3 commit: `docs(openspec): add READMEs to 6 orphan archive changes (Fix #4)`

**Phase 5**（独立 commit）: 2 个 banner
- 5.1 HEALTH-AUDIT 添加勘误 banner
- 5.2 PROJECT-COMPLETION-SUMMARY 添加过期 banner
- 5.3 commit: `docs(audits): add outdated/errata banners to 2 stale docs (Fix #5)`

**Phase 6**（独立 commit）: spec + 验证脚本
- 6.1 创建 `openspec/specs/docs-discoverability/spec.md`（建立 v1.4.1 spec）
- 6.2 创建 `scripts/check-docs-index.sh`（自动验证脚本）
- 6.3 commit: `feat(docs): add docs-discoverability spec + check script (Fix #6)`

**回退策略**:
- 每个 Phase 独立可 revert（文档改动 git revert 风险低）
- 若 Phase 1 README 索引引发强烈反对，revert 后可重新设计分组规则

## Open Questions

1. **Q1**: 是否需要强制自动化（pre-commit hook 验证 README 覆盖所有子目录）？
   - 当前选择：约定 + 可选脚本（不强制 hook）
   - 待实施后根据反馈决定是否升级到 hook

2. **Q2**: docs/skills/README.md 是否应删除，仅保留 .opencode/skills/README.md 单一来源？
   - 当前选择：保留 docs/skills/README.md 作为"对开发者的技术参考导航"
   - 理由：与"可加载技能"分离，更适合人类阅读

3. **Q3**: OpenSpec 6 个孤儿 change 的 README 中是否应包含 lessons-learned 引用？
   - 当前选择：仅 commit hash + 1 句话目的，详细 lessons-learned 留在 `docs/dev-process/lessons-learned.md`
   - 待定：实施时如发现需要可补充
