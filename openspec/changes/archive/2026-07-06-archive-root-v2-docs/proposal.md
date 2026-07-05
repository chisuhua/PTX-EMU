## Why

PTX-EMU 根目录有 5 个 2026-04 时代（SIMT v2.0 早期）的过时文档，自 2026-04-11 起未更新，包含：

| 文件 | 最后更新 | 主要问题 |
|------|----------|----------|
| `workflow-state.md` | 2026-05-25 | v4 工作流状态，引用已删除的文件 + 已归档的 OpenSpec changes |
| `task_plan.md` | 2026-04-11 | 旧调试会话的 PTX 语法修复任务计划，已超 94% 完成 |
| `BUILD-VERIFICATION-v2.0.md` | 2026-04-11 | 声称"零技术债务"，与当前 90 条债务不符 |
| `RELEASE-CHECKLIST-v2.0.md` | 2026-04-11 | 声称"38 测试通过"，与当前 >739 测试不符 |
| `PTX_PARSING_FIX_REPORT.md` | 2026-04-11 | 旧报告，描述已关闭的修复 |

这些文档：
1. **误导新开发者**（README 描述项目"零技术债务"，实际 90 条）
2. **占用根目录空间**（5 个文档污染项目根目录视图）
3. **违反 lessons-learned §21**（"重大变更必须同步 README"，但 README 已更新到 2026-07，根目录 stale 文档未清理）

按 `docs/README.md` "归档规则"：
- ✅ Phase 计划（已执行）
- ✅ 审批请求（已完成）
- ✅ 过时设计文档

应归档到 `docs/archive/2026-04-simt-v2/` 子目录（保留历史决策参考价值）。

## What Changes

- **移动 5 个根 .md 到 `docs/archive/2026-04-simt-v2/`**：
  - `workflow-state.md` → `docs/archive/2026-04-simt-v2/workflow-state-2026-05-25.md`
  - `task_plan.md` → `docs/archive/2026-04-simt-v2/task_plan-2026-04-11.md`
  - `BUILD-VERIFICATION-v2.0.md` → `docs/archive/2026-04-simt-v2/BUILD-VERIFICATION-v2.0.md`
  - `RELEASE-CHECKLIST-v2.0.md` → `docs/archive/2026-04-simt-v2/RELEASE-CHECKLIST-v2.0.md`
  - `PTX_PARSING_FIX_REPORT.md` → `docs/archive/2026-04-simt-v2/PTX_PARSING_FIX_REPORT.md`
- **在 `docs/archive/2026-04-simt-v2/` 添加 README.md** 解释归档原因 + 列出原文件最后更新时间
- **在 `docs/archive/README.md` 更新索引**包含新子目录
- **同步 `docs/audits/debt-audit-2026-07-02.md`**：标记 D 系列文档清理条目 RESOLVED

**BREAKING**: 无 — 文档位置变更不影响功能

## Capabilities

### New Capabilities

- `root-v2-docs-archive`: 归档根目录 v2.0 时代过时文档到 `docs/archive/2026-04-simt-v2/`

### Modified Capabilities

无 — 不影响任何 spec 级行为。

## Impact

**受影响的代码/文件**：

| 文件 | 改动 | 影响 |
|------|------|------|
| `workflow-state.md` | git mv 到 archive | 1 文件 |
| `task_plan.md` | git mv 到 archive | 1 文件 |
| `BUILD-VERIFICATION-v2.0.md` | git mv 到 archive | 1 文件 |
| `RELEASE-CHECKLIST-v2.0.md` | git mv 到 archive | 1 文件 |
| `PTX_PARSING_FIX_REPORT.md` | git mv 到 archive | 1 文件 |
| `docs/archive/2026-04-simt-v2/README.md` | 新建 | 新文件 |
| `docs/archive/README.md` | 更新索引 | ≤10 行 |
| `docs/audits/debt-audit-2026-07-02.md` | 标记 RESOLVED | 1 行 |

**受影响的 ADR**：
- 无直接 ADR 影响

**测试覆盖**：
- 现有测试无回归（纯文件移动）
- 不需要 ctest（文档 change）

**回归风险**：
- 🟢 极低：纯文件移动，git mv 保留历史

**Lessons-learned 集成**：
- ✅ Checklist E（artifacts 必 tracked）
- ✅ Checklist F（git verify）
- ✅ Checklist G（lifecycle）
- ✅ §21（重大变更必须同步 README）