## Context

PTX-EMU 根目录的 5 个过时文档自 2026-04-11（SIMT v2.0 早期）以来未更新，与当前项目状态严重脱节。

### 文档内容审查

| 文件 | 主要内容 | 当前问题 |
|------|---------|----------|
| `workflow-state.md` | v4 OpenSpec 工作流状态 | 引用已删除文件（`test_shortest_path_first.cpp` 等）+ 已归档 changes（`add-sm90-100-bsync-interleave` 等），声称 3 个 changes "已归档（未开始）"实际早已实施 |
| `task_plan.md` | PTX 语法修复计划 | 11 个失败测试 "当前状态" 已全部修复 + 新增 90 条债务 |
| `BUILD-VERIFICATION-v2.0.md` | SIMT v2.0 构建验证 | 声称"零技术债务"，实际有 90 条 |
| `RELEASE-CHECKLIST-v2.0.md` | SIMT v2.0 Release Checklist | 声称"38 测试通过" + "v2.0.0 Ready for Release"，实际 >739 测试 + v2.0.0 从未发布 |
| `PTX_PARSING_FIX_REPORT.md` | dummy-condition.1 测试失败修复 | 已关闭缺陷，但描述误导当前读者以为 PTX 解析仍有问题 |

### 归档规则（per docs/README.md）

> 满足以下条件之一应归档到 `archive/`:
> - ✅ Phase 计划（已执行）
> - ✅ 审批请求（已完成）
> - ✅ 代码审查（已合并）
> - ✅ 过时设计文档
> - ✅ 外部规范副本（如 PTX ISA reference）

5 个文档全部满足"过时设计文档" + "Phase 计划（已执行）" 条件。

### Metis Review

本 change 无需 Metis pre-impl review：
- 范围明确（git mv 文件）
- 0 决策点（无新设计）
- 0 风险（git mv 保留历史）

## Goals / Non-Goals

### Goals

1. **移动 5 个根 .md 到 `docs/archive/2026-04-simt-v2/`**
2. **创建 `docs/archive/2026-04-simt-v2/README.md`** 解释归档原因
3. **更新 `docs/archive/README.md`** 添加新子目录索引
4. **同步 `docs/audits/debt-audit-2026-07-02.md`**：标记 RESOLVED
5. **同步 `docs/roadmap/post-phase3-debt-roadmap.md`**：移除相关条目

### Non-Goals（明确排除）

1. ❌ **删除文档**：保留历史决策参考价值（git mv 保留历史）
2. ❌ **修改文档内容**：纯位置移动，不编辑内容
3. ❌ **更新 README.md**：当前 README.md 已无这些文档的引用（2026-07-05 同步过）

## Decisions

### Decision 1: 归档 vs 删除

**Choice**: 归档

**Rationale**：
- lessons-learned §21 + docs/README.md "归档规则" 要求归档过时文档
- git mv 保留历史 + git blame 可追溯
- 5 个文档可能对未来理解 2026-04 决策有价值

### Decision 2: 子目录命名

**Choice**: `docs/archive/2026-04-simt-v2/`

**Rationale**：
- 时间戳 + 时代标识便于理解
- 与其他 archived changes 一致（如 `archive/2026-06-24-phase3-...`）
- 允许未来添加其他 v2.0 时代文档

### Decision 3: 文件名加日期前缀

**Choice**: 不加日期前缀（如 `workflow-state-2026-05-25.md`）

**Rationale**：
- 子目录已含日期
- README.md 表格列出每个文件的最后更新时间
- 避免长文件名污染

## Risks / Trade-offs

| 风险 | 影响 | 缓解措施 |
|------|------|---------|
| **R1: 隐藏文档引用被打断** | 🟢 极低（grep 验证 README 已无引用）| (1) grep 检查 (2) Phase 1.4 验证 |
| **R2: git mv 误用（cp + rm）** | 🟢 极低 | 必须用 `git mv` 保留历史 |
| **R3: 未来读者不知道归档原因** | 🟢 低 | README.md 解释 + commit message |

## Migration Plan

### Phase 0: Artifacts Git-Tracking

```bash
git checkout -b docs/archive-root-v2-docs
git add openspec/changes/archive-root-v2-docs/
git commit -m "docs(openspec): add archive-root-v2-docs artifacts"
```

### Phase 1: git mv 5 个文档 + 创建 archive README（Fix #1）

```bash
git worktree add .worktrees/archive-v2-docs-impl docs/archive-root-v2-docs
cd .worktrees/archive-v2-docs-impl

# 1.1 创建 archive 子目录
mkdir -p docs/archive/2026-04-simt-v2

# 1.2 git mv 5 个文档
git mv workflow-state.md docs/archive/2026-04-simt-v2/
git mv task_plan.md docs/archive/2026-04-simt-v2/
git mv BUILD-VERIFICATION-v2.0.md docs/archive/2026-04-simt-v2/
git mv RELEASE-CHECKLIST-v2.0.md docs/archive/2026-04-simt-v2/
git mv PTX_PARSING_FIX_REPORT.md docs/archive/2026-04-simt-v2/

# 1.3 创建 archive README.md
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

# 1.4 更新 docs/archive/README.md
# 添加引用 docs/archive/2026-04-simt-v2/README.md

# 验证：grep 检查根 README.md 不引用这些文档
grep -E "workflow-state|task_plan|BUILD-VERIFICATION|RELEASE-CHECKLIST|PTX_PARSING_FIX" README.md AGENTS.md
# 期望: 无匹配

# Commit
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

### Phase 2: 文档同步（Fix #2）

```bash
# 更新 docs/audits/debt-audit-2026-07-02.md
# 标记根目录 v2.0 docs RESOLVED

# 更新 docs/roadmap/post-phase3-debt-roadmap.md
# 从剩余债务列表移除

git commit -am "docs(archive): sync debt audit + roadmap post-Fix #1 (Fix #2)"
```

### Phase 3: Archive

```bash
openspec archive archive-root-v2-docs --yes
git checkout main
git merge --no-ff docs/archive-root-v2-docs
```

### Rollback Strategy

```bash
# git mv 是可逆的
git revert HEAD  # 反向 git mv
# 或：
git mv docs/archive/2026-04-simt-v2/*.md ./
```