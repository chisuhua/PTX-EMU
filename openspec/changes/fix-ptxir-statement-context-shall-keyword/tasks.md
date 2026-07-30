## 1. Artifacts Setup (Phase 0 — per Checklist E)

- [ ] 1.1 Create `openspec/changes/fix-ptxir-statement-context-shall-keyword/{proposal.md,design.md,specs/ptxir-statement-context-change-protocol/spec.md,tasks.md}` — DONE
- [ ] 1.2 Verify `openspec status --change fix-ptxir-statement-context-shall-keyword` reports all artifacts ready
- [ ] 1.3 Commit artifacts in a separate commit (`docs(openspec): add fix-ptxir-statement-context-shall-keyword proposal`) — per Checklist E to avoid the "实施 commits 合并时未追踪 artifacts" 陷阱

## 2. Live Spec Upgrade (Phase 1)

- [ ] 2.1 Edit `openspec/specs/ptxir-statement-context-change-protocol/spec.md` Requirement #4 — upgrade `SHOULD` → `MUST` in header + body — DONE (commit `6a85a778`)
- [ ] 2.2 Run `openspec validate ptxir-statement-context-change-protocol --type spec` — expect "Specification ... is valid" — DONE
- [ ] 2.3 Run `openspec validate --specs` — expect total 44 passed, 10 failed (was 43/11) — DONE
- [ ] 2.4 Confirm commit `6a85a778` in git history with diff showing 2-line `SHOULD → MUST` replacement — DONE

## 3. Documentation & Drift Tracking

- [ ] 3.1 Document known delta-vs-sync drift in `openspec/changes/fix-ptxir-statement-context-shall-keyword/design.md` §Risks — DONE
- [ ] 3.2 Record regression-prevention pattern in `openspec/changes/fix-ptxir-statement-context-shall-keyword/design.md` §Risks — DONE
- [ ] 3.3 Note (in commit message of 1.3) that the archived delta is intentionally not amended per Checklist G

## 4. Verification

- [ ] 4.1 `git log --all --oneline -- openspec/specs/ptxir-statement-context-change-protocol/spec.md` — verify the live spec history
- [ ] 4.2 `git log --all --oneline -- "openspec/changes/archive/2026-07-30-ptxir-format-compliance/specs/ptxir-statement-context-change-protocol/spec.md"` — verify archived delta is unmodified after this change
- [ ] 4.3 `openspec list --change fix-ptxir-statement-context-shall-keyword --json` — confirm change is in active changes
- [ ] 4.4 `openspec validate --specs` — confirm 44 passed / 10 failed (the 10 remaining failures are pre-existing)

## 5. Archive

- [ ] 5.1 Run `openspec archive fix-ptxir-statement-context-shall-keyword --yes` once all phases complete (after 1.3 + 2.1-2.4 + 3.1-3.3 + 4.1-4.4 all checked)
- [ ] 5.2 Verify archived change appears in `openspec/changes/archive/<date>-fix-ptxir-statement-context-shall-keyword/`
- [ ] 5.3 Confirm live spec `openspec/specs/ptxir-statement-context-change-protocol/spec.md` Requirement #4 still uses `MUST` post-archive