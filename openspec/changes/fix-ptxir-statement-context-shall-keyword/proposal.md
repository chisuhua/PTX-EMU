## Why

`openspec validate --specs` reports `ptxir-statement-context-change-protocol` as failing with `requirements.4.text: Requirement must contain SHALL or MUST keyword`. Requirement #4 (Pre-commit hook warning) uses `SHOULD` in both the header and body, while its 2 Scenarios already use `MUST` — the keyword is internally inconsistent and violates the OpenSpec validator's normative requirement rule.

Root cause: the original delta spec (`openspec/changes/archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md`) was authored with `SHOULD` for Requirement #4 — likely a copy-paste oversight during the `ptxir-format-compliance` change proposal (commits `05c2a6c3` + `1c424360`).

This fix is needed so `openspec validate --specs` reports green and the spec's normative contract is unambiguous. Future re-archival of `ptxir-format-compliance` should not re-introduce the `SHOULD` regression.

## What Changes

- **Live spec upgrade (already done, commit `6a85a778`)**: `openspec/specs/ptxir-statement-context-change-protocol/spec.md` Requirement #4 upgraded from `SHOULD` → `MUST` (header + body, 2 line edits).
- **New fix-* change proposal** (this artifact): documents the upgrade intent via `## MODIFIED Requirements` delta, references the archived change per OpenSpec lifecycle.
- **Archived delta left untouched**: per OpenSpec lifecycle (Checklist G), archived changes are immutable. Any future `openspec archive` of a re-derived `ptxir-format-compliance` will need to re-author with `MUST` from the start.

## Capabilities

### New Capabilities

(none)

### Modified Capabilities

- `ptxir-statement-context-change-protocol`: Requirement #4 ("Pre-commit hook warns on StatementContext changes without PTXIR sync") upgraded from `SHOULD` to `MUST` to satisfy OpenSpec validator's `Requirement must contain SHALL or MUST keyword` rule and align with the existing `MUST` Scenarios.

## Impact

- **Spec layer**: 1 live spec file edited (commit `6a85a778`).
- **Validation**: `openspec validate --specs` total goes from 43 passed / 11 failed → 44 passed / 10 failed (this change fixes 1 regression; the other 10 failures are pre-existing and unrelated).
- **Archive layer**: archived delta `openspec/changes/archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md` still uses `SHOULD` (immutable per Checklist G). Documented in `design.md` as known inconsistency with rationale.
- **Code layer**: no code changes — this is a documentation/spec-quality fix only.
- **AGENTS.md / ADR**: no synchronization needed (no code or architecture change).

## Ref

- Archived change: `archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md`
- Live spec: `openspec/specs/ptxir-statement-context-change-protocol/spec.md` (Requirement #4, line 46-47 after commit `6a85a778`)
- Implementing commit: `6a85a778 fix(specs): upgrade SHOULD to MUST in ptxir-statement-context-change-protocol`
- Related lessons: `ptx-lessons-learned` Checklist G (OpenSpec lifecycle Archived 终态约束)

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- N/A — no code/API migration; pure spec keyword upgrade.

### 多 Phase 推进
- N/A — single trivial edit (2 line replacements), 1 commit. No Phase split needed.
- Pre-implementation review: not required (per Checklist H scope: only for OpenSpec changes that involve new code/architecture; this is a documentation-only fix).

### 文档同步
- [x] No `AGENTS.md` update needed (no code change).
- [x] No ADR update needed (no architectural decision change).
- [x] `tasks.md` Phase 状态: live spec upgrade already implemented (Phase 1 complete); remaining Phase 2 is verification + archive tracking.