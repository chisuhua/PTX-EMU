## ADDED Requirements

### Requirement: Pre-commit enforcement
When ANY file under `docs/` is staged for commit (via `git add`), a pre-commit hook SHALL execute `bash scripts/check-docs-index.sh` and block the commit if any of Check 1, 2, 3, 4, 5, or 6 fails.

#### Scenario: docs/ change triggers validator
- **WHEN** a developer runs `git commit -m "..."` with at least one staged file matching `docs/.*`
- **THEN** `.git/hooks/pre-commit` runs `bash scripts/check-docs-index.sh` before allowing the commit
- **AND** the commit is blocked (exit code 1) if any check fails
- **AND** the developer sees the failing check's output

#### Scenario: non-docs commit skips validator
- **WHEN** a developer commits changes only to `src/`, `include/`, `tests/`, or other non-docs paths
- **THEN** `.git/hooks/pre-commit` does NOT run `scripts/check-docs-index.sh` (saves ~1s per commit)

### Requirement: CI enforcement (GitHub Actions)
For any pull request to the main branch that modifies `docs/**`, a GitHub Actions workflow SHALL execute `scripts/check-docs-index.sh` and fail the check if any of the 6 mandatory checks fails.

#### Scenario: PR with docs/ changes fails CI on broken check
- **WHEN** a pull request opens with changes to `docs/`
- **THEN** `.github/workflows/docs-validate.yml` runs `bash scripts/check-docs-index.sh` in CI
- **AND** the CI check fails if any of Check 1-6 reports `FAIL`
- **AND** the PR is blocked from merge until the failure is fixed

#### Scenario: PR without docs/ changes skips CI docs-validate
- **WHEN** a pull request opens with NO changes to `docs/`
- **THEN** `.github/workflows/docs-validate.yml` is skipped (`paths` filter excludes)

### Requirement: Validator unit test coverage
The `scripts/check-docs-index.py` validation script SHALL be covered by unit tests that verify each check's behavior independently of the actual project state.

#### Scenario: Check 1 PASS for matching subdirs
- **WHEN** the unit test runs the validator against a temporary `docs/` containing 16 subdirs and a `docs/README.md` indexing all 16
- **THEN** Check 1 exits PASS

#### Scenario: Check 1 FAIL for missing subdir entry
- **WHEN** the unit test runs the validator against a `docs/` containing 17 subdirs but a `docs/README.md` indexing only 16
- **THEN** Check 1 exits FAIL with `NOT_INDEXED: <missing-dir-name>` printed

#### Scenario: Check 2 PASS for resolving links
- **WHEN** the unit test runs the validator against a `docs/README.md` containing only valid internal links
- **THEN** Check 2 exits PASS

#### Scenario: Check 2 FAIL for broken link
- **WHEN** the unit test runs the validator against a `docs/README.md` containing a link to a non-existent file
- **THEN** Check 2 exits FAIL with `BROKEN: <link>` printed

#### Scenario: Check 3 FAIL for hand-edited statistics (post-Tier-2)
- **WHEN** the unit test runs the validator against a `docs/README.md` containing a table row with `38 测试`
- **THEN** Check 3 exits FAIL (not just WARN)

#### Scenario: Check 4 PASS for orphan with README
- **WHEN** the unit test creates an orphan change directory under `openspec/changes/archive/` containing `proposal.md` and `README.md`
- **THEN** Check 4 detects it as an orphan with README, increments `orphan_ok` count

#### Scenario: Check 4 FAIL for orphan without README
- **WHEN** the unit test creates an orphan change directory containing only `proposal.md` (no `README.md`)
- **THEN** Check 4 exits FAIL with `MISSING: <change-name>` printed

#### Scenario: Check 5 PASS for bannered document
- **WHEN** the unit test runs the validator against a `docs/audits/HEALTH-AUDIT-2026-06-21.md` whose first 5 lines after title contain the substring `⚠️ 8 个事实错误已修正`
- **THEN** Check 5 exits PASS

#### Scenario: Check 5 FAIL for stale document without banner
- **WHEN** the unit test runs the validator against a stale document file (declared in EXPECTED_BANNERED dict) whose first 5 lines after title do NOT contain the expected banner pattern
- **THEN** Check 5 exits FAIL with `MISSING_BANNER: <file>` printed

#### Scenario: Check 6 PASS for skills in sync
- **WHEN** the unit test creates 17 directories under `.opencode/skills/` and 1 under `.opencode/skills.disable/`, and a `docs/skills/README.md` listing all 18 (17 active + 1 disabled marker)
- **THEN** Check 6 exits PASS

#### Scenario: Check 6 FAIL for new skill not reflected
- **WHEN** the unit test creates a new directory `.opencode/skills/new-skill/` but `docs/skills/README.md` does not mention `new-skill`
- **THEN** Check 6 exits FAIL with `MISSING_IN_DOCS: new-skill` printed

#### Scenario: Check 6 FAIL for removed skill still in docs
- **WHEN** the unit test creates `.opencode/skills/` without `old-skill/` directory but `docs/skills/README.md` still lists `old-skill`
- **THEN** Check 6 exits FAIL with `STALE_IN_DOCS: old-skill` printed
