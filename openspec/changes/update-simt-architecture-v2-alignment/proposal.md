# Proposal: update-simt-architecture-v2-alignment

## Why

`docs/architecture/SIMT-ARCHITECTURE-V2.md` (v2.0, 2026-04-09) 描述了 SIMT 架构设计，但存在以下问题：

1. **文档-实现不一致**: 部分实现细节与文档描述不匹配
2. **关键实现未文档化**: 多个重要函数（如 `advance_thread_pc()`, `sync_from_warp_state()`）未记录
3. **状态过时**: 文档最后更新于 2026-05-05，但仍有信息未同步

## What

更新 `SIMT-ARCHITECTURE-V2.md` 文档，确保与实际实现完全对齐：

1. 记录已移除的字段 (`simt_stack_depth`, `pc_stack`)
2. 记录新发现的实现细节
3. 更新关键 API 表格
4. 验证所有设计决策的实现状态

## Capabilities

1. **文档准确性**: 文档与代码完全一致
2. **新员工 onboarding**: 准确的架构文档降低学习曲线
3. **未来维护**: 避免因文档过时导致的错误修改
4. **ADR 合规**: 与 ADR-0014 等决策保持同步

## Impact

| 文件 | 影响类型 |
|------|---------|
| `docs/architecture/SIMT-ARCHITECTURE-V2.md` | 更新文档 |

## References

- ADR-0014: Independent Thread Scheduling
- Skill: state-modification-audit