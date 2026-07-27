# Approved Proposals (已批准提案)

本文件记录经 `guide-arch` Phase 5.5 审查通过的改进提案。批准后由 `guide-plan` Phase 2 propose 消费创建 OpenSpec changes。

## 批准记录

| 提案 | 优先级 | 批准日期 | 审批者 | 备注 |
|------|--------|----------|--------|------|
| [god-class-refactor-sm-context](improvements/god-class-refactor-sm-context.md) | P2 | 2026-07-26 | Oracle Round-3 | C-2 sm_context 拆分，10-12h，依赖 C-18 |
| [split-ptx-visitor-god-class](improvements/split-ptx-visitor-god-class.md) | P2 | 2026-07-26 | Oracle Round-3 | C-17 ptx_visitor 残余提取，2-3h，含 :922 bug 修复 |
| [refactor-warp-context](improvements/refactor-warp-context.md) | P2 | 2026-07-26 | Oracle Round-3 | C-18 WarpContext 拆分，6h，冻结 public API |

## 提案间依赖

```
C-17（独立）→ C-18（冻结 WarpContext API）→ C-2（依赖 C-18 冻结的 API）
```

## 评审链

- **Round-1 (Oracle)**: 初步评审，发现行数基线过期
- **Round-2 (Oracle)**: 发现 C-17 核心 scope 已实施 + C-1 重名归档
- **Round-3 (Oracle)**: 重写所有 3 个提案 + 显式边界声明

## 备注

- C-1 (god-class-refactor-thread-context) 已 REJECTED，由 ADR-0019 Phase 3 接管
- C-2/C-18 必须串行执行（API 冻结约束）
- C-17 可独立优先执行
