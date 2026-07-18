# 清理根目录冗余/过期 MD 文件

## Context

项目根目录存在 3 个问题文件：GEMINI.md 与 QODER.md md5 完全一致（无项目内容）；docs/PROJECT-COMPLETION-SUMMARY.md 声称"100% 完成"但项目仍在 Phase 3。AGENTS.md 中 OpenSpec 流程章节被复制粘贴两次。

## Goals / Non-Goals

- **Goals**: 删除冗余文件，去重 AGENTS.md，归档过期完成声明
- **Non-Goals**: 不新增文件，不修改内容逻辑

## Decisions

1. **GEMINI.md / QODER.md**: 直接删除 — AGENTS.md 已是唯一入口
2. **PROJECT-COMPLETION-SUMMARY.md**: git mv 至 docs/archive/ — 保留历史但不再误导
3. **AGENTS.md 去重**: 保留行 10-26（第一个出现），删除行 40-56

## Risk

- 极低：纯文件操作，无代码路径影响
