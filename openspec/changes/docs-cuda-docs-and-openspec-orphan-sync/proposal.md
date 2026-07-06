## Why

`docs/` 与 `openspec/` 目录存在 5 项文档同步债务（D-1/D-3/D-4/D-5/D-6），源自 2026-07-02 债务审计（commit `07dfd48`）。D-1 与 D-3 的计数偏差在 parser-completeness 系列 change 后已部分修复但需 re-verify；D-4 遗留 5 个已归档 OpenSpec change 缺 `design.md`；D-5 `docs/skills/` 残留 3 个已迁移至 `.opencode/skills/` 的过期副本；D-6 8 项事实错误仍以独立 ERRATA 存在未合并回主审计。修复这些债务可消除新开发者的导航误导，并使审计数据可信。

## What Changes

- **D-1**: Re-verify `docs/README.md` 索引覆盖全部 16 个子目录（当前声称 16，实证 `ls -d docs/*/` = 16，无遗漏 — 若实证一致则仅添加验证注释）
- **D-4**: 为 5 个缺失 `design.md` 的已归档 OpenSpec change **新建** retroactive `design.md`（`Ref: archive/<date>-<name>/` — 不 amend 已归档 change，符合 Checklist G 约束）。第 6 个（`integrate-barrier-module-cta-warp`）已有 `design.md`。
- **D-5**: 从 `docs/skills/` 删除 3 个过期技能副本（`ptx-debug/`、`ptxir-serialization/`、`ptx-grammar-modification.md`）— 这些技能已在 `.opencode/skills/` 中维护
- **D-6**: 将 `HEALTH-AUDIT-2026-06-21-ERRATA.md` 的 8 项事实修正内联到主审计 `HEALTH-AUDIT-2026-06-21.md`，保留 ERRATA 为历史参考

## Capabilities

### New Capabilities
- `docs-index-verify`: `docs/README.md` 索引 SHALL 覆盖全部 `docs/` 子目录，标题中的数字 SHALL 与 `ls -d docs/*/ | wc -l` 一致
- `openspec-orphan-design`: 已归档 OpenSpec change 若缺 `design.md`，SHALL 通过 retroactive 合成补充（不 amend 已归档 change，通过 `Ref:` 链接引用）
- `skills-dir-cleanup`: `docs/skills/` SHALL 仅包含不可加载的技术参考文档与人类可读导航 README，过期技能副本 SHALL 移除
- `audit-errata-merge`: 审计勘误 SHALL 内联到主审计正文，ERRATA 文件 SHALL 保留为历史参考

### Modified Capabilities
<!-- None — 所有修改仅影响文档，不改变 spec 级行为 -->

## Impact

| 受影响的组件 | 影响类型 |
|-------------|---------|
| `docs/README.md` | 内容验证 + 可能补注释（D-1） |
| `openspec/changes/archive/2026-06-24-phase3-*/` | 新建 `design.md`（5 个 change，不 amend 已归档）（D-4） |
| `docs/skills/ptx-debug/` | 删除过期副本（D-5） |
| `docs/skills/ptxir-serialization/` | 删除过期副本（D-5） |
| `docs/skills/ptx-grammar-modification.md` | 删除过期副本（D-5） |
| `docs/audits/HEALTH-AUDIT-2026-06-21.md` | 内联 8 项 ERRATA 修正（D-6） |
| `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` | 添加"已合并"标记（D-6） |