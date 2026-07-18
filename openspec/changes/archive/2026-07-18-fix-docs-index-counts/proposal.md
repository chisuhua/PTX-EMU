## Why

项目治理审计（2026-07-18）发现 docs/README.md 导航表中 10/16 目录的文档数量与实际不符（62.5% 偏差率），docs/adr/README.md 的 ADR 总数和索引表滞后，docs/archive/README.md 的 4 个子目录文件数错误且漏列一个子目录。这些错误已持续数周，对新开发者的 onboarding 和文档可信度造成直接影响。对应 debt-audit-2026-07-02.md 的 P1-D1、D2、D3。

## What Changes

- **docs/README.md §文档导航 表格**: 10 行文档数修正（adr/ 15→22, skills/ 8→3, superpowers/ 6→25, archive/ 50+→56, roadmap/ 1→2, dev-process/ 2→3, audits/ 4→5）+ ADR 范围描述 "0001~0015"→"0001~0021（0017 缺失）"+ 删除或修正第 63 行"脚本自动生成"声明
- **docs/adr/README.md**: 第 99 行 "ADR 总数: 19"→"20"；索引表补 ADR-0013（statement-factory-test-unification）
- **docs/archive/README.md**: 子目录表格 phase-plans 8→14, code-reviews 12→6, ptx-instruction-reference 19→13, misc 12→14；补 2026-04-simt-v2 条目（6 文件）

## Capabilities

### New Capabilities
- `docs-index-verify`: 确保 docs/README.md、docs/adr/README.md、docs/archive/README.md 索引表的文件计数与实际文件系统一致

### Modified Capabilities
<!-- none -->

## Impact

- 文件: docs/README.md、docs/adr/README.md、docs/archive/README.md（仅编辑，无新增/删除）
- 代码: 零影响
- 测试: 零影响
- 风险: 无 — 纯文档校正
- 相关: debt-audit-2026-07-02.md §P1-D1/D2/D3, openspec/specs/docs-index-verify/