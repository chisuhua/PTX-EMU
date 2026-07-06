## Context

Change-1/2/3a/3b archive 后,根 `AGENTS.md`、`src/grammar/AGENTS.md`、`src/ptxsim/instructions/AGENTS.md`、ADR-0016、lessons-learned 仍描述旧 wmma 路径状态。本 change 同步文档到 post-implementation 状态(per `ptx-lessons-learned` Checklist I "重大功能交付清单")。

无源码改动,纯 docs sync + archive 整理。**4-change 路线图终点**。

## Goals / Non-Goals

**Goals**: 同步 5 个 AGENTS.md + ADR + lessons-learned + 最终 archive。

**Non-Goals**: 不修改源码、测试、handler、grammar(全部已 archive)、不删除 wmma.cpp(Change-4 scope)。

## Decisions

### D1: 1 commit(纯 docs,无需分 Phase)

**采纳**: 1 commit = docs sync + final archive。

### D2: lessons-learned §24 内容

**必含**: 4-change 路线图回顾 + Metis 5 MR + Change 拆分价值
**拒绝**: 逐 commit 复述(冗余)

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| R1: docs 误改破坏 git blame 链 | 1 commit 集中改,易 revert |
| R2: lessons-learned §24 过长 | 控制在 100 LoC 内 |

## Migration Plan

1. 同步 5 个 AGENTS.md
2. 追加 ADR-0016 更新记录
3. 追加 lessons-learned §24
4. `openspec archive`
5. commit

## Open Questions

无。
