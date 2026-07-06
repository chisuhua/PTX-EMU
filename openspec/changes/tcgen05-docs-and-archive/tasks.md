# Tasks: tcgen05 Documentation Sync + Archive

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 changes**(必须 archive): Change-1, 2, 3a, 3b(3d 可选)
> **范围**: 1 commit(纯 docs)

## 0. Pre-Implementation Review

- [ ] 0.1 `openspec list` 确认 Change-1/2/3a/3b 已 archive
- [ ] 0.2 `cd build && ctest --output-on-failure` 确认 baseline 全绿

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/tcgen05-docs-and-archive`
- [ ] 1.2 `git add openspec/changes/tcgen05-docs-and-archive/`
- [ ] 1.3 `git commit -m "docs(openspec): add tcgen05-docs-and-archive artifacts (ADR-0016)"`

## 2. Phase 1: 文档同步(commit 2)

- [ ] 2.1 根 `AGENTS.md`:更新已知限制表 — tcgen05 5 core handler 已实现(per ADR-0016)
- [ ] 2.2 `src/grammar/AGENTS.md`:更新 lexer/parser 规则说明,标注 `tcgen05Inst` 替代 `wmmaInst`
- [ ] 2.3 `src/ptxsim/instructions/AGENTS.md`:添加 `tcgen05.cpp` 说明,标注 `wmma.cpp` 保留 pre-Blackwell 路径
- [ ] 2.4 `docs/adr/0016-blackwell-only-tcgen05.md`:在"更新记录"追加 4-5 个 archive commit 引用
- [ ] 2.5 `docs/dev-process/lessons-learned.md`:追加 §24 "重大功能交付清单"(100 LoC):
  - 4-change 路线图回顾
  - Metis 5 MR 关键教训
  - Change 拆分的价值
  - 真实案例:change-1 → 6 commits
- [ ] 2.6 验证:`grep -n "tcgen05\|pre-Blackwell" AGENTS.md src/grammar/AGENTS.md src/ptxsim/instructions/AGENTS.md`
- [ ] 2.7 `git add AGENTS.md src/grammar/AGENTS.md src/ptxsim/instructions/AGENTS.md docs/`
- [ ] 2.8 `git commit -m "docs: sync AGENTS + ADR + lessons-learned for tcgen05 4-change completion (ADR-0016)"`

## 3. Phase 2: Archive(commit 3,per Checklist G)

- [ ] 3.1 `openspec archive tcgen05-docs-and-archive --yes`
- [ ] 3.2 `ctest --output-on-failure` 最终验证
- [ ] 3.3 `git add openspec/changes/archive/`
- [ ] 3.4 `git commit -m "chore(openspec): archive tcgen05-docs-and-archive (ADR-0016)"`

## Final Validation

- [ ] 4.1 `git log --oneline | head -3` 显示 3 commits
- [ ] 4.2 `openspec list` 确认 change 已 archive
- [ ] 4.3 **5-change 路线图结束** — Change-4 (cleanup wmma) 仍可独立 propose
