# STATUS: reduce-thread-context-includes-v2 (ghost, v2 of v1)

> **状态**: Doc-only archive. NO code change was ever applied.
>
> **依据**: `tasks.md` 29 个 task 全部 `[ ]` 未勾选,无任何 git commit
> 对 `include/ptxsim/thread_context.h` 执行 include 精简。
>
> **当前 include 数量**: `include/ptxsim/thread_context.h` 仍为 25 个
> (与 v2 design.md 声称的"基线 25"完全一致,目标 ≤15 从未达成)。

## 唯一相关 commit

[`edb9302e refactor(ptxsim): document forward declarations in thread_context.h`](https://github.com/.../commit/edb9302e)
— 仅添加 forward declaration 注释块,未实际移除任何 include。

## 与 v1 的关系

- **v1** ([`2026-07-29-reduce-thread-context-includes`](../2026-07-29-reduce-thread-context-includes/)):
  同样 ghost (32/32 task 未勾选)
- **v2** (本目录): 重做尝试,同样未执行

两版的 design.md 都正确识别了 25 个 include 中可前向声明的候选,
但提案被 archive 时**所有 task 都未勾选**,说明 author 意识到工作未完成
但仍选择 archive 而非继续。

## 结论

此目录保留为历史 design artifact,**不应被视为"已完成的 change"**。
如需重做实际 include 精简,需新建 v3 change 并走完整的
`/guide-plan` → `propose` → `tasks.md` (所有 task 必须 `[x]`) → `archive` 流程。
