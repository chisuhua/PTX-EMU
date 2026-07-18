# 修正 docs/ 索引表文档计数

## Context

docs/README.md 导航表 10/16 目录文档数失实；docs/adr/README.md ADR 总数错误且漏 ADR-0013；docs/archive/README.md 4 个子目录计数错且漏列一个子目录。发现于 2026-07-18 项目治理审计。

## Goals / Non-Goals

- **Goals**: 修正 docs/README.md、docs/adr/README.md、docs/archive/README.md 中的文档计数，使其与实际文件系统一致
- **Non-Goals**: 不重写文档内容，不新增/删除文档，不修改 scripts/check-docs-index.py 逻辑

## Decisions

1. **计数字段**: 使用 `find <dir> -type f | wc -l` 结果（与 docs/README.md 第 63 行声明的统计方式一致）
2. **README.md 第 63 行声明**: 保留声明但加注 "注: 当前校验脚本仅检查子目录名，文件数需手动维护"，不修改脚本逻辑（属于独立改进项）
3. **ADR 范围描述**: "ADR-0001~0015" → "ADR-0001~0021（0017 缺失）"

## Migration / Rollback

- 无迁移 — 纯文档修��
- Rollback: `git revert`
