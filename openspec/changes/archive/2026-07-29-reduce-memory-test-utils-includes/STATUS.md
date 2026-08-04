# STATUS: reduce-memory-test-utils-includes (v1, ghost)

> **状态**: Doc-only archive. NO code change was ever applied.
>
> **依据**: `tasks.md` 32 个 task 全部 `[ ]` 未勾选,无任何 git commit 涉及此 change。

## 与 v2 的关系

- **v1** (本目录): 2026-07-29 — proposal/design/tasks 起草后未执行,直接 archive
- **v2** ([`2026-08-04-reduce-memory-test-utils-includes-v2`](../2026-08-04-reduce-memory-test-utils-includes-v2/)):
  真正执行了 include 精简,commit [`0810cd0f`](https://github.com/.../commit/0810cd0f)
  把 `memory_test_utils.h` 从 18 个 include 减到 12 个。
  v2 引入一个隐式依赖 bug,2026-08-04 通过
  [`0969a275 fix(sm): resolve dangling stmts pointer in SM/CTA unit tests`](https://github.com/.../commit/0969a275)
  修复。

## 结论

此目录保留为历史 design artifact,不应被视为"已完成的 change"。
v1 设计的"include 18 → 12"目标实际由 v2 完成。
