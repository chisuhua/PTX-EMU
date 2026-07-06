# Tasks: Audit Blackwell tcgen05 Infrastructure

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec in [specs/](specs/)
> **范围**: 1 commit(pure read-only 审计,无源码改动)
> **Lessons-learned**: Checklist E + Checklist G

## 0. Pre-Implementation Review

- [ ] 0.1 跑 Metis 验证:
  - [ ] 0.1.1 `wc -l src/ptxsim/memory/tma_descriptor.{h,cpp}`(约 168+204)
  - [ ] 0.1.2 `wc -l src/ptxsim/memory/tmem.{h,cpp}`(约 50+61)
  - [ ] 0.1.3 `wc -l src/ptxsim/cluster/cluster_context.{h,cpp}`(读后填)
  - [ ] 0.1.4 `wc -l src/ptxsim/async/tc_queue.{h,cpp}`(读后填)
  - [ ] 0.1.5 `grep -c "UNVERIFIED-AGAINST-HARDWARE" src/ptxsim/memory/tma_descriptor.{h,cpp}`(期望 29)
  - [ ] 0.1.6 `grep -c "TEST_CASE" tests/unit/memory/test_tma_descriptor.cpp tests/unit/memory/test_tmem.cpp tests/unit/cluster/test_cluster_mode.cpp tests/unit/async/test_tc_queue.cpp`(分别 36/19/16/15)

- [ ] 0.2 基线 worktree:`git worktree add .worktrees/baseline-tcgen05-audit -b feat/extend-blackwell-tcgen05-infra main`

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/extend-blackwell-tcgen05-infra`
- [ ] 1.2 `git add openspec/changes/extend-blackwell-tcgen05-infra/`
- [ ] 1.3 `git commit -m "docs(openspec): add extend-blackwell-tcgen05-infra artifacts (ADR-0016)"`

## 2. Phase A: 审计(1 commit)

### 2.1 跑 baseline

- [ ] 2.1.1 `cmake --build build` 验证编译
- [ ] 2.1.2 `ctest -L "unit;memory" --output-on-failure` 记录 baseline
- [ ] 2.1.3 `ctest -L "unit;cluster" --output-on-failure` 记录 baseline
- [ ] 2.1.4 `ctest -L "unit;async" --output-on-failure` 记录 baseline

### 2.2 阅读 4 个子系统

- [ ] 2.2.1 读 `src/ptxsim/memory/tma_descriptor.h`,记录 17 处 UNVERIFIED 位置
- [ ] 2.2.2 读 `src/ptxsim/memory/tma_descriptor.cpp`,记录 12 处 UNVERIFIED 位置
- [ ] 2.2.3 读 `src/ptxsim/memory/tmem.h`,记录 invariant
- [ ] 2.2.4 读 `src/ptxsim/cluster/cluster_context.h`,记录 arrive/wait 语义
- [ ] 2.2.5 读 `src/ptxsim/async/tc_queue.h`,记录 commit-group counter 原子性

### 2.3 `state-modification-audit` skill

- [ ] 2.3.1 跑 `state-modification-audit` skill 验证 TcQueue ↔ WarpState invariant(per ADR-0016 Decision 7)

### 2.4 写审计报告

- [ ] 2.4.1 写 `docs/audits/2026-07-XX-tcgen05-infra-audit.md`(约 400 LoC)
  - [ ] §1 概述(范围、基线、术语)
  - [ ] §2.1 TmaDescriptor(readiness 等级 + 29 UNVERIFIED 分级 P0/P1/P2)
  - [ ] §2.2 Tmem(readiness + invariant)
  - [ ] §2.3 ClusterContext(readiness + arrive/wait 语义)
  - [ ] §2.4 TcQueue(readiness + commit-group 原子性)
  - [ ] §2.5 cross-subsystem pipeline(TmaDescriptor → Tmem → TcQueue 覆盖空白)
  - [ ] §3 发现的问题(需独立 `fix-*` change)
  - [ ] §4 推荐 follow-up changes

### 2.5 验证(无源码改动)

- [ ] 2.5.1 `git diff --stat main..HEAD` 验证仅 `docs/audits/` 改动(无 `src/` diff)
- [ ] 2.5.2 `ctest --output-on-failure` 验证 baseline 仍绿

### 2.6 Commit

- [ ] 2.6.1 `git add docs/audits/`
- [ ] 2.6.2 `git commit -m "docs(audit): add tcgen05 infrastructure audit report (ADR-0016)"`

## 3. Phase B: Archive(per Checklist G)

- [ ] 3.1 `openspec archive extend-blackwell-tcgen05-infra --yes`
- [ ] 3.2 `ctest --output-on-failure` 最终验证
- [ ] 3.3 `git add openspec/changes/archive/`
- [ ] 3.4 `git commit -m "chore(openspec): archive extend-blackwell-tcgen05-infra (ADR-0016)"`

## Final Validation

- [ ] 4.1 `git log --oneline | head -3` 显示 3 个 commits(artifacts + audit + archive)
- [ ] 4.2 `git diff --stat main..HEAD` 仅 3 类文件:proposal/design/tasks/specs + audit + archive
- [ ] 4.3 `openspec list` 确认 change 已 archive

## Risks Recap

| Risk | Mitigation |
|------|------------|
| R1: 审计发现架构级 bug | 报告标注为 `fix-*` change,本 change 不修 |
| R2: 报告过长 | 控制在 400 LoC 内 |
