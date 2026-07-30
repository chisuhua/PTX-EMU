## Context

The `openspec validate --specs` validator enforces that every Requirement's text contains at least one `SHALL` or `MUST` keyword (RFC 2119 normative language). This rule ensures spec contracts are unambiguous: a "should" or "may" requirement is non-binding and cannot be enforced.

`ptxir-statement-context-change-protocol` (live spec at `openspec/specs/ptxir-statement-context-change-protocol/spec.md`) is one of 5 capabilities introduced by the `ptxir-format-compliance` change (commits `05c2a6c3` ... `1f7379d1`, archived `2026-07-30`). Requirement #4 ("Pre-commit hook warns on StatementContext changes without PTXIR sync") was authored in the original delta spec using `SHOULD` for both the requirement header and body text. The two associated Scenarios correctly use `MUST`, making the keyword choice internally inconsistent.

State before fix:
- Live spec: failing validation (`requirements.4.text: Requirement must contain SHALL or MUST keyword`).
- Archived delta: also uses `SHOULD` (immutable per OpenSpec lifecycle).
- Live spec was edited directly (commit `6a85a778`), but without a corresponding `fix-*` change proposal to document the upgrade rationale and reference the archived source.

Stakeholders:
- Future developers reading the spec: need a clear normative contract.
- OpenSpec archive workflow: should not re-introduce the `SHOULD` regression if `ptxir-format-compliance` is ever re-derived.
- Spec auditor (`openspec validate --specs`): needs to report green.

## Goals / Non-Goals

**Goals:**
- Make `ptxir-statement-context-change-protocol` Requirement #4 pass `openspec validate --specs`.
- Document the upgrade in a `fix-*` change proposal so future readers understand WHY `SHOULD` was changed to `MUST`.
- Reference the archived `ptxir-format-compliance` change as the source of the original `SHOULD` keyword.
- Establish a clear pattern: any future spec keyword normalization follows the `fix-*` + `Ref: archive/...` + no-amend-archived workflow.

**Non-Goals:**
- Modifying the archived delta `openspec/changes/archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md` — explicitly forbidden by OpenSpec lifecycle (Checklist G).
- Fixing the other 10 pre-existing `openspec validate --specs` failures (audit-errata-merge, auto-co-simulation, cmake-cleanup-commented-tests, cudart-unit-test, docs-discoverability, docs-index-verify, openspec-orphan-design, pc-api, root-md-cleanup, skills-dir-cleanup) — those are out of scope for this single-keyword fix.
- Code changes — this is a documentation/spec-quality fix only.

## Decisions

### Decision 1: Edit live spec, leave archived delta untouched

**Choice**: Edit `openspec/specs/ptxir-statement-context-change-protocol/spec.md` (live spec, mutable); leave `openspec/changes/archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md` (archived delta, immutable) unchanged.

**Rationale**:
- OpenSpec lifecycle (Checklist G) explicitly forbids amending archived changes: "Archived: 终态, artifacts 不可修改; 若需修补 → 新建 fix-*/refactor-* change; 引用方式: Ref: archive/<date>-<name>/; 禁止 amend 已归档 change (违反 OpenSpec 生命周期)".
- Real case precedent (`ptx-lessons-learned` 经验 6): the `barrier-migration-amendment` change (2026-07-02) tried to amend the archived `cleanup-deprecated-barrier-apis` change, causing 4 P0-A debt misclassifications that took 12 days to detect. The current rule exists precisely to prevent this class of bug.
- The live spec is the authoritative contract for current behavior — fixing it is what actually matters for `openspec validate`.

**Alternatives considered**:
- *Amend archived delta*: rejected — would violate Checklist G and create the `barrier-migration-amendment` failure mode.
- *Rewrite the entire `ptxir-format-compliance` change*: rejected — too disruptive; the only real issue is one keyword.

### Decision 2: Use `fix-` prefix (not `refactor-`)

**Choice**: Change name `fix-ptxir-statement-context-shall-keyword`.

**Rationale**:
- OpenSpec convention: `fix-*` for bug fixes (correctness/regression), `refactor-*` for restructuring without behavior change. This change corrects a normative-language bug, so `fix-*` is the correct prefix.
- The change name explicitly names the affected spec and the keyword upgrade, making it greppable for future audits.

### Decision 3: Document upgrade via `## MODIFIED Requirements` (not just `## ADDED Requirements`)

**Choice**: In `specs/ptxir-statement-context-change-protocol/spec.md`, use `## MODIFIED Requirements` section that points to Requirement #4 with the `SHOULD → MUST` diff.

**Rationale**:
- The intent is "this requirement exists, but its normative strength is being strengthened", which is precisely what `MODIFIED Requirements` encodes.
- `ADDED Requirements` would imply a brand-new requirement, which is misleading.

## Risks / Trade-offs

- **Risk**: Archived delta retains `SHOULD`, while live spec uses `MUST` → state inconsistency between the two.
  - **Mitigation**: This inconsistency is documented in `tasks.md` Phase 3 ("Document known delta-vs-sync drift"). The archived delta is intentionally left untouched per Checklist G. Future re-archival should re-author with `MUST`.
- **Risk**: If `openspec archive` is re-run on `ptxir-format-compliance`, the `SHOULD` keyword might re-enter the live spec.
  - **Mitigation**: This `fix-*` change's `tasks.md` records the regression-prevention pattern: when authoring future `openspec archive` workflows, always re-validate `openspec validate --specs` post-archive and apply the `fix-*` workflow if any keyword regression is detected.
- **Risk**: Other archived deltas may have similar `SHOULD`-keyword regressions, hidden by being untracked or pre-archived.
  - **Mitigation**: Out of scope for this change. A separate `audit-archived-deltas-for-shall-keywords` improvement could be proposed via `add-improve` workflow if desired.
- **Trade-off**: Direct live-spec edit (commit `6a85a778`) preceded the formal `fix-*` proposal creation. This is a minor sequencing deviation from the canonical "proposal first, implement second" OpenSpec flow, but acceptable because (a) the edit is trivial, (b) `git log -- openspec/specs/...` records the change with full provenance, and (c) the `fix-*` proposal retroactively documents the rationale.

## Migration Plan

No migration needed — pure documentation/spec-quality fix, no code or data model changes.

Rollback strategy:
- If future review determines the live spec should revert to `SHOULD`: `git revert 6a85a778` (live spec) + archive this `fix-*` change as "not implemented". The archived delta remains unaffected.

## Open Questions

(none)