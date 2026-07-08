# Tasks: tcgen05 Documentation Sync + Archive

> **依赖**: [proposal.md](proposal.md) + [design.md](design.md) + 1 spec
> **前置 changes**: Change-1 ✅ / Change-2 ✅ / Change-3a ✅ / Change-3b ✅ (3d 可选)
> **范围**: 2 commits(1 docs + 1 archive)
>
> ## 2026-07-07 状态更新
>
> - ✅ Change-3a (`fix-tcgen05-grammar-mr3`) — archived
> - ✅ Change-3b (`implement-tcgen05-handlers-core`, commit `df6dde7`) — archived,5 handler 已实施
> - ⚠️ archive `fix-tcgen05-antlr-prediction-bug` 的 proposal.md 声称 "Kleene star 根因",但 lessons-learned §25 修正为 **lexer bare token vs ID conflict**。本 change 的 preamble 中需显式引用 §25 作为权威 override。

## 0. Pre-Implementation Review

- [ ] 0.1 `openspec list` 确认 Change-1/2/3a/3b 已 archive(预期 4/4 ✅)
- [ ] 0.2 `cd build && ctest --output-on-failure` 确认 baseline 170/170 PASS
- [ ] 0.3 `./tests/ptx/test_all_ptx.sh` 确认 47/47 PASS

## 1. Artifacts Tracking(commit 1)

- [ ] 1.1 `git checkout -b feat/tcgen05-docs-and-archive`
- [ ] 1.2 `git add openspec/changes/tcgen05-docs-and-archive/`
- [ ] 1.3 `git commit -m "docs(openspec): add tcgen05-docs-and-archive artifacts (ADR-0016)"`

## 2. Phase 1: 文档同步(commit 2)

- [ ] 2.1 根 `AGENTS.md`:更新已知限制表 — tcgen05 5 core handler 已实现(per ADR-0016)
- [ ] 2.2 `src/grammar/AGENTS.md`:更新 lexer/parser 规则说明,标注 `tcgen05Inst` 替代 `wmmaInst`
- [ ] 2.3 `src/ptxsim/instructions/AGENTS.md`:添加 `tcgen05.cpp` 说明,标注 `wmma.cpp` 保留 pre-Blackwell 路径
- [ ] 2.4 `docs/adr/0016-blackwell-only-tcgen05.md`:在"更新记录"追加 4-5 个 archive commit 引用
- [ ] 2.5 `docs/dev-process/lessons-learned.md`:追加 §26 "tcgen05 5-core-handler 交付"(ref §25 根因)
  - 4-change 路线图回顾 + df6dde7 关键设计决策(op_kind dispatch)
  - 跨 change 并发风险:handler 签名变更与 god-class refactor 的串行约束
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
