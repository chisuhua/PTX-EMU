# 2026-08-10-ptxir-cubin-cleanup (Archived)

> **⚠️ Archive metadata only** — 原始 change 归档时缺 `design.md`，仅保留 `proposal.md` + `tasks.md`。本 README 由 `docs-index-fix-pre-existing-orphans` 补齐。

## Purpose

PTXIR-Embedded CUBIN 工具链 Phase 12.2 收尾：补齐 `__cudaRegisterFatBinary` legacy front door 中缺失的 PTXIR dispatch 分支 (per ADR-0024 v1.1)。`ptxir_embed` / `ptxir_extract` CLI 工具已 ship，但运行时不会走 PTXIR 路径 — 用户感知不到 PTXIR-Embedded CUBIN 工具链的端到端价值。R1: PTXIRLoader 测试覆盖补齐, R2: INI `[ptxir] mode = off` 段集成到 `initialize_environment()`。

## Implementation

- **Proposal**: `proposal.md` (Why/What Changes 段，描述 PTXIR legacy front door 集成缺口 + 修复范围)
- **Tasks**: `tasks.md` (TDD 三阶段 + Phase 12.2 收尾 ship gate)
- **Implementation commit**: `aec5d80e` — `Merge feat/ptxir-cubin-cleanup: Phase 12.2 收尾 ship` (verify: `git show aec5d80e --stat`)
- **Artifact commit**: `20ad752b` — `docs(openspec): 2026-08-10-ptxir-cubin-cleanup change skeleton`
- **Archive commit**: `e0d2a93e` — `chore(openspec): archive 2026-08-10-ptxir-cubin-cleanup`

## Related

- **ADR**: ADR-0024 v1.1 (PTXIR-Embedded CUBIN 格式 + footer-layout detection + `PTXIR_EMBED_MAGIC`) + ADR-0026
- **Predecessor**: `archive/2026-08-07-implement-ptxir-cubin-embed-extension/` (PTXIR embed 工具实现 commit 1-4)
- **Components**: `ptxir_embed` / `ptxir_extract` CLI + `PTXIRLoader` + `PtxContextAdapter` + `config::isPTXIRModeEnabled` + INI `[ptxir] mode`

---

**Status**: ✅ RESOLVED (by `aec5d80e`, 2026-08-10)
**Added by**: `docs-index-fix-pre-existing-orphans` (2026-08-23)