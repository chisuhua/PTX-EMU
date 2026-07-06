## ADDED Requirements

### Requirement: 审计勘误应内联到主审计

`HEALTH-AUDIT-2026-06-21-ERRATA.md` 中的事实修正 SHALL 以 inline 标记形式合并到 `HEALTH-AUDIT-2026-06-21.md` 的对应段落，不修改审计原文。

#### Scenario: 受影响段落获得勘误标记

- **WHEN** 审计原文的某段落包含 ERRATA 中记录的事实错误
- **THEN** 该段落 SHALL 附加 `**[勘误: <正确值>]**` inline 标记

#### Scenario: 审计原文不被修改

- **WHEN** 合并 ERRATA 后
- **THEN** `HEALTH-AUDIT-2026-06-21.md` 的原文数字/声称 SHALL 保持不变（仅添加勘误标记）

### Requirement: ERRATA 文件保留并标记已合并

`HEALTH-AUDIT-2026-06-21-ERRATA.md` SHALL 在顶部添加合并状态标记，标注合并日期和负责 change。

#### Scenario: ERRATA 状态标记

- **WHEN** 合并完成后
- **THEN** ERRATA.md 顶部 SHALL 包含 `**[2026-07-06 已合并到主审计 by change docs-cuda-docs-and-openspec-orphan-sync]**` 或等效标记

### Requirement: 受影响的审计章节列表

以下 `HEALTH-AUDIT-2026-06-21.md` 章节 SHALL 获得勘误标记（per ERRATA §1.1-1.8）：

#### Scenario: 8 个勘误项被内联

- **WHEN** 验证合并结果
- **THEN** E1（ThreadContext 字段数）SHALL 在 §0.2 §1.2 §10.1 有标记
- **AND** E2（Symtable 泄漏）SHALL 在 §0.2 §2.2.1 有标记
- **AND** E3（拼写引用数）SHALL 在 §1.2 有标记
- **AND** E4（H2 严重度）SHALL 在 §1.2 有标记
- **AND** E5（membar 工作量）SHALL 在 §0.4 §3.5 有标记
- **AND** E6（Phase 1 顺序）SHALL 在 §8 有标记
- **AND** E7（cudaStream_t）SHALL 在 §2.2.1 有标记
- **AND** E8（PTX 8.7+ 占位）SHALL 在 §3.5 §9.1 有标记