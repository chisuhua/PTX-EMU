## Why

项目根目录存在 3 个冗余/过期文件：GEMINI.md 与 QODER.md 内容完全相同（38 行，无项目内容）；docs/PROJECT-COMPLETION-SUMMARY.md 正文仍声明"100% Complete"/"v2.0.0 Ready for Release"，仅靠过期 banner 缓解。AGENTS.md 中 OpenSpec 流程章节被重复粘贴两次（行 10-26 = 行 40-56）。这些文件增加了维护负担并可能误导新开发者。对应 debt-audit-2026-07-02.md 的 P0-D3、P0-D4。

## What Changes

- **删除 GEMINI.md**: 与 QODER.md 内容完全相同（md5: f958ec4c7420f4a7ac9aea5567a46ef1），AGENTS.md 已是唯一 AI agent 入口
- **删除 QODER.md**: 同上
- **AGENTS.md**: 删除行 40-56（与行 10-26 "OpenSpec 流程 + 经验沉淀" 章节完全重复）
- **归档 PROJECT-COMPLETION-SUMMARY.md**: 从 docs/ 移至 docs/archive/，彻底解决 P0-D4 "虚假声明"

## Capabilities

### New Capabilities
- `root-md-cleanup`: 确保根目录 .md 文件无冗余、无过期误导内容

### Modified Capabilities
<!-- none -->

## Impact

- 删除: GEMINI.md (-38 行)、QODER.md (-38 行)
- 移动: docs/PROJECT-COMPLETION-SUMMARY.md → docs/archive/2026-07-18-project-completion-summary.md
- 编辑: AGENTS.md (-17 行，删除重复段)
- 净: -3 文件, -55 行
- 代码: 零影响
- 测试: 零影响
- 风险: 低 — 纯文件清理，无内容变更
- 相关: debt-audit-2026-07-02.md §P0-D3, P0-D4