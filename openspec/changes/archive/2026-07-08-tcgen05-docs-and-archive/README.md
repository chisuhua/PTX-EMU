# Archive Cross-Reference: tcgen05-docs-and-archive

> **一致性声明 (2026-07-08)**: 本 archive 的 `proposal.md` 在实施过程中存在 **3 项历史偏差**,但实际文档产出通过 `tasks.md` 修订 + `lessons-learned.md §26` 的实质补充完成了正确交付。
>
> **权威来源**: [`docs/dev-process/lessons-learned.md` §26](../../../../docs/dev-process/lessons-learned.md#26-tcgen05-5-core-handler--handler-dispatch-2026-07-08-)
>
> **背景**:
> - **`proposal.md` 前置 change 列表不完整**: §4-9 列出 4 个前置 change (Change-1/2/3a/3b),遗漏 2 个 `fix-*` change:
>   - `fix-tcgen05-test-coverage-gaps` (commit `fd74261`) — 提供 5 integration parse 测试 + 1 E2E GEMM kernel + f16×f16→f32 golden value
>   - `fix-tcgen05-handler-dispatch` (commit `cc49ae7`) — wire `Tcgen05Handler::processTcgen05Operation` dispatch 入口(本 README 标题指向的"§26 5-core-handler 交付"实质内容正是此 change 的根因分析)
> - **`proposal.md` 引用 §24 vs 实际产出 §26**:
>   - `proposal.md` §20 / §35 / §77 / §195 承诺追加"§24 重大功能交付清单"(per Checklist I)
>   - `tasks.md` §2.5 在实施前已修正为 §26(per Checklist J + lessons-learned §6 artifacts-first 原则)
>   - 实际提交 `95d9d65` 添加的是 **§26 "tcgen05 5-core-handler 交付 + handler dispatch 修复"**(内容从"清单"变为"根因分析 + 5 教训 + 检查工具 + 真实案例",这正是 §26 的实质价值)
> - **`proposal.md` 承诺"1 个 commit"vs 实际 3 个 commit**:
>   - `proposal.md` §60-61 / §159 承诺"仅 1 个 commit(纯文档,无需分 Phase)"
>   - 实际 3 个 commit:`a211016` (artifacts tracking) + `95d9d65` (docs sync) + `332bacd` (archive) — 拆分合理但与提案文本不符
>
> **按 `ptx-lessons-learned §6 + Checklist G` 铁律**: 已归档 change 不 amend。本 README 作为永久 cross-reference,引导后续 maintainer 优先阅读 §26 的实质内容,而非被 `proposal.md` 的过期引用误导。
>
> **未来 audit 引用**: 任何 `sync-*` / `docs-*` change 涉及 4-change roadmap 终端文档 sync,必须:
>   1. 引用 §26 作为"功能交付 + dispatch 路径教训"的实质来源
>   2. 引用本 README 作为"proposal/tasks/spec 内部不一致"的失败模式案例
>   3. 遵循 `tasks.md` 已修正的版本(§26 + 3 commits)而非 `proposal.md` 原始承诺
>
> **OpenSpec artifacts 一致性检查**:本 archive 的 artifacts 内部存在 deviation(tasks.md 已修正,proposal.md 未同步),但因 §26 实质补充已落地,**审计判定为 RESOLVED**。详见 `docs/dev-process/lessons-learned.md` Checklists G(OpenSpec lifecycle)+ J(artifacts 内部一致性)+ K(docs-* change 陷阱)。

---

## Archive Index (原 change artifacts)

- `proposal.md` — 原始提案(§4-9 前置 change 列表 / §20 §35 §77 §195 §24 引用 / §60-61 §159 1-commit 承诺 已过期,见上方声明)
- `design.md` — 设计文档
- `tasks.md` — 任务清单(§2.5 已正确修正为 §26 引用,作为实施侧权威)
- `specs/tcgen05-docs-sync/spec.md` — 文档同步规范

## 实质补充来源

- [`docs/dev-process/lessons-learned.md` §26](../../../../docs/dev-process/lessons-learned.md#26-tcgen05-5-core-handler--handler-dispatch-2026-07-08-) — tcgen05 5-core-handler 交付 + handler dispatch 修复
  - 现象:170/170 ctest 通过但 `tcgen05.*` 指令 per-lane EXIT
  - 根因(5 点):`ptx_op.def:129-136` 注释排除 / `InstructionFactory::initialize()` 仅从 def 注册 / `ThreadContext::_execute_once():143` 返回 nullptr
  - 修复:`3a30da8` + `d3afaf5` + `cc49ae7` 三个 commit 组成
  - 5 教训 + 3 检查工具 bash + 真实案例(`tests/e2e/kernel/test_blackwell_gemm.cu` 假 PASS)
- [`docs/adr/ADR-0016-blackwell-only-tcgen05.md` §Phase 1-2 完成记录](../../../../docs/adr/ADR-0016-blackwell-only-tcgen05.md) — 5 core handler 测试覆盖 (`fd74261`) + handler dispatch 管道接入 (`cc49ae7`)
- [`openspec/changes/archive/2026-07-07-fix-tcgen05-antlr-prediction-bug/README.md`](../2026-07-07-fix-tcgen05-antlr-prediction-bug/README.md) — 姊妹 cross-reference:ANTLR4 lexer bare token 根因(per §25)

## 关联 commits

| Commit | 说明 |
|--------|------|
| `a211016` | `docs(openspec): track tcgen05-docs-and-archive artifacts (ADR-0016, all prerequisites archived, lessons-learned §25 override)` — 工件追踪 commit(proposal/tasks 偏差未在此 commit 中体现) |
| `95d9d65` | `docs: sync AGENTS + ADR-0016 + lessons-learned §26 for tcgen05 4-change completion (ADR-0016)` — 文档同步 commit,**实际产出 §26** 而非 proposal 承诺的 §24 |
| `332bacd` | `chore(openspec): archive tcgen05-docs-and-archive (ADR-0016)` — archive commit |
| `5457aca` | `merge: tcgen05-docs-and-archive (ADR-0016)` — 合并 commit |
| `fd74261` | `fix-tcgen05-test-coverage-gaps` (ADR-0016, **遗漏的前置 change**) — 5 integration parse 测试 + 1 E2E GEMM kernel + golden value |
| `cc49ae7` | `fix-tcgen05-handler-dispatch` (ADR-0016, **遗漏的前置 change**) — wire dispatch + 11 S_TCGEN05_* X-Macro |
| `df6dde7` | `implement-tcgen05-handlers-core` (ADR-0016) — 5 core handler (mma/ld/st/commit/wait) + wmma cleanup |