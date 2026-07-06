## Context

Change-1 (archived) 建立了独立 tcgen05 命名空间,handler 实施(Change-3b)前需先**审计** 4 个底层子系统。本 change 是 pure read-only 审计,无源码改动(Metis MR-3 修复),任何发现通过独立 `fix-*` change 处理。

当前状态(per Change-1 Metis MR-4 修正):TmaDescriptor 36 TEST_CASE、Tmem 19、Cluster 16、TcQueue 15、跨子系统集成 2(覆盖率不足)。29 处 `// UNVERIFIED-AGAINST-HARDWARE — PTX ISA §9.7.13`(TmaDescriptor 17 .h + 12 .cpp)。

目标:1 commit 输出 `docs/audits/2026-07-XX-tcgen05-infra-audit.md` 报告,标注每子系统 readiness + UNVERIFIED 分级,不修改任何源码。

## Goals / Non-Goals

**Goals**: 4 子系统审计 + 29 UNVERIFIED 注释分级 + cross-subsystem pipeline 覆盖空白识别。

**Non-Goals**: 不修改任何源码(per Metis MR-3)、不实施任何 fix(独立 `fix-*` change)、不实施 handler(Change-3b)、不修改 wmma.cpp(Change-4)。

## Decisions

### D1: 审计策略 — 阅读 + 跑测试 + 跑 `state-modification-audit` skill,无实验

**采纳**: 4 子系统的 `.h`/`.cpp` 静态阅读 + 跑 `ctest` baseline + `state-modification-audit` skill 验证 TcQueue ↔ WarpState invariant。

**拒绝**: 不需要真实 GPU 实验(无访问,`cuobjdump -xptx` 不可用)。

### D2: 报告结构 — per 子系统章节 + readiness 等级 + UNVERIFIED 分级 + cross-subsystem pipeline

**采纳**: 4 章节(per 子系统) + 1 章节(cross-subsystem pipeline) + 1 章节(发现的问题 + 推荐 `fix-*` change)。

### D3: 1 个 commit(纯文档,无需分 Phase)

**采纳**: 1 个 commit = 审计报告。

**理由**: per Metis F.1,本 change 极简,避免 Change-1 的 "3 atomic commits 变 6 个" 反模式。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| **R1**: 审计发现架构级 bug(需跨文件修改) | 报告标注为 "需独立 `fix-*` change",本 change 不修 |
| **R2**: 报告过长(>500 LoC) | 控制在 400 LoC 内,详细数据用表格 |

## Migration Plan

1. 跑 baseline `ctest -L "unit;memory|unit;barrier|unit;cluster|unit;async" --output-on-failure` 记录
2. 跑 `state-modification-audit` skill(per ptx-lessons-learned §1)
3. 写 `docs/audits/2026-07-XX-tcgen05-infra-audit.md`
4. `git add docs/audits/` + commit
5. `openspec archive` + commit archive

### 回退策略

`git revert HEAD` 回到 good state。

## Open Questions

无(纯 read-only 审计,无新决策)。
