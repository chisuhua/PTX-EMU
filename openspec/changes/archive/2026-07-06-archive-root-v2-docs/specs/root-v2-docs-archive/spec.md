# root-v2-docs-archive Specification

## Purpose
TBD - created by archiving change archive-root-v2-docs. Update Purpose after archive.
## Requirements
### Requirement: Root-V2-Docs-Archived MUST

The PTX-EMU root directory MUST NOT contain the following 5 v2.0-era documents:
- `workflow-state.md`
- `task_plan.md`
- `BUILD-VERIFICATION-v2.0.md`
- `RELEASE-CHECKLIST-v2.0.md`
- `PTX_PARSING_FIX_REPORT.md`

These documents MUST be moved to `docs/archive/2026-04-simt-v2/` (using `git mv` to preserve history).

#### Scenario: Root-Directory-Clean
- **WHEN** `ls *.md` is executed in the root directory
- **THEN** the 5 archived documents MUST NOT be listed
- **AND** only currently-maintained documents (README.md, AGENTS.md, etc.) MUST remain

#### Scenario: Archive-Location-Correct
- **WHEN** checking `docs/archive/2026-04-simt-v2/`
- **THEN** all 5 documents MUST be present
- **AND** a `README.md` MUST exist explaining the archival reason

### Requirement: V2-Docs-Broken-References-Absent MUST

No production documentation, code, or configuration file MUST reference the 5 archived documents at their old root-directory locations.

#### Scenario: No-Broken-Reference
- **WHEN** grep searches README.md, AGENTS.md, docs/, src/, tests/, openspec/ for references to the 5 archived document names
- **THEN** zero matches MUST be returned

#### Scenario: Archive-Index-Updated
- **WHEN** reading `docs/archive/README.md`
- **THEN** the new `2026-04-simt-v2/` subdirectory MUST be indexed
- **AND** a brief description MUST explain why these documents were archived

### Requirement: V2-Docs-Archival-Verified MUST

The archival MUST be verified by:
1. `git log --follow <file>` MUST show the original commit history (git mv preserves history)
2. `git blame docs/archive/2026-04-simt-v2/<file>` MUST show original authors
3. The new location MUST be discoverable via `docs/archive/2026-04-simt-v2/README.md`

#### Scenario: Git-History-Preserved
- **WHEN** running `git log --follow docs/archive/2026-04-simt-v2/workflow-state.md`
- **THEN** the original commit history (including 2026-05-25 last update) MUST be visible

#### Scenario: Documentation-Synced
- **WHEN** reading `docs/audits/debt-audit-2026-07-02.md`
- **THEN** the root-directory v2.0 docs debt item MUST have a "✅ FIXED by commit <hash>" annotation
- **AND** `docs/roadmap/post-phase3-debt-roadmap.md` MUST NOT list this debt as remaining