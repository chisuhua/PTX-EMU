# STATUS: reduce-thread-context-includes (v1, ghost)

> **状态**: Doc-only archive. NO code change was ever applied.
>
> **依据**: `tasks.md` 30 个 task 全部 `[ ]` 未勾选,无任何 git commit 涉及此 change。
>
> **当前 include 数量**: `include/ptxsim/thread_context.h` 仍为 25 个
> (与 v1/v2 design.md 声称的"基线 25"完全一致,目标 ≤15 从未达成)。

## 与 v2 的关系

- **v1** (本目录): 2026-07-29 — proposal/design/tasks 起草后未执行,直接 archive
- **v2** ([`2026-08-04-reduce-thread-context-includes-v2`](../2026-08-04-reduce-thread-context-includes-v2/)):
  重做尝试,proposal.md 由 v1 改写而来,但同样未执行任何 task,doc-only archive。
  唯一改动是 commit [`edb9302e`](https://github.com/.../commit/edb9302e)
  `refactor(ptxsim): document forward declarations in thread_context.h` —
  仅添加 forward declaration 注释,未实际精简 include。

## 结论

此目录保留为历史 design artifact,**不应被视为"已完成的 change"**。
如需重做实际 include 精简,需新建 v3 change 并走完整的
`/guide-plan` → `propose` → `tasks.md` (所有 task 必须 `[x]`) → `archive` 流程。
