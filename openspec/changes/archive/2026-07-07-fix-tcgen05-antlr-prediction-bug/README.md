# Archive Cross-Reference: fix-tcgen05-antlr-prediction-bug

> **一致性声明 (2026-07-08)**: 本 archive 的 `proposal.md`、`design.md`、`handoff.md` 声称修复 "ANTLR LL(*) 预测冲突",但**真正根因**是 **ANTLR4 lexer bare string token 与 ID 规则冲突**。
>
> **权威来源**: [`docs/dev-process/lessons-learned.md` §25](../../../../docs/dev-process/lessons-learned.md#25-antlr4-le)
>
> **背景**:
> - 本 archive 的提案由 `commit ad808e3` 引入的 `TCGEN_F16` / `TCGEN_BF16` bare lexer token 触发(lex 抢占 → Kleene star 失败表象)
> - 真正修复是 `commit 55e216a` 的 5 行 lexer/parser diff:删除 bare tokens + `tcgen05Qual` 加 `ID` fallback + `tcgen05Dtype` 用 dot-prefixed `F16/BF16`
> - 提案中的 "Kleene star 修复 / 递归重写 `tcgen05QualList` 规则" 方案**未实施**——真正修复是 lexer 层面,提案的 grammar 改造在 archive 后**没有 commit 落地**
>
> **按 `ptx-lessons-learned §6 + Checklist G` 铁律**: 已归档 change 不 amend。本 README 作为永久 cross-reference,引导后续 maintainer 优先阅读 §25 的真相描述。
>
> **未来 audit 引用**: 任何 `fix-*/refactor-*/feat-*` change 涉及 ANTLR grammar,必须引用 §25 作为失败模式检查项(per `.opencode/skills/ptx-grammar-modification/SKILL.md` §Checklist L)。

---

## Archive Index (原 change artifacts)

- `proposal.md` — 原始提案(根因描述已过期,见上方声明)
- `design.md` — 设计文档(同上)
- `tasks.md` — 任务清单
- `handoff.md` — 实施状态报告(描述已过期)
- `specs/tcgen05-qualifier-permutations/spec.md` — 资格顺序规范

## 关联 commits

| Commit | 说明 |
|--------|------|
| `ad808e3` | **引入** bare `TCGEN_F16` / `TCGEN_BF16` tokens(根因) |
| `55e216a` | **真正修复**: 5 行 lexer/parser diff |
| `e92f1c1` | 添加 `regression_cute_rmsnorm_f16_register.ptx` 守卫 |
| `0ff1d38` | lessons-learned §25 沉淀 |
| `1d1f01a` | archive 本 change |
