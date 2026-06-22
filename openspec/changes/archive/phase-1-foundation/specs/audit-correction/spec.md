## ADDED Requirements

### Requirement: Audit Errata publication
The repository MUST contain a file `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` that lists all verified factual errors and omissions in `docs/audits/HEALTH-AUDIT-2026-06-21.md` (the audit document at commit `baa8c4e`).

#### Scenario: Errata lists every factually wrong claim with evidence
- **WHEN** reader opens `HEALTH-AUDIT-2026-06-21-ERRATA.md`
- **THEN** for each factually wrong claim the file MUST contain: the original claim, the actual value, file:line evidence (verified by grep or static analysis), and the recommended correction text

#### Scenario: Errata lists every critical omission
- **WHEN** reader opens `HEALTH-AUDIT-2026-06-21-ERRATA.md`
- **THEN** for each item the audit missed entirely the file MUST contain: the missing item, its impact on subsequent roadmap tasks, and the priority level it should have been assigned

#### Scenario: Errata preserves audit history snapshot integrity
- **WHEN** Errata is published
- **THEN** the original `HEALTH-AUDIT-2026-06-21.md` MUST remain unchanged so that `git checkout baa8c4e -- docs/audits/HEALTH-AUDIT-2026-06-21.md` reproduces the audit as-of that commit

### Requirement: Errata version discipline
Each quarterly audit review MUST produce either a new Errata document (Errata v2, v3, ...) or an Errata amendment, never modifying the previous Errata file, so that historical corrections remain independently verifiable through git history.

#### Scenario: New quarterly review appends new Errata
- **WHEN** a quarterly audit review (e.g., 2026-09-21) discovers new factual errors
- **THEN** a new file `HEALTH-AUDIT-YYYY-MM-DD-ERRATA-vN.md` MUST be created rather than modifying the existing 2026-06-21 Errata

#### Scenario: Errata document has fixed schema
- **WHEN** any Errata document is published
- **THEN** it MUST contain the following sections in order: (1) header with date and audit reference, (2) factual errors table with claim / actual / evidence / correction, (3) critical omissions table with item / impact / priority, (4) priority adjustment recommendations, (5) decision log entries adopted

### Requirement: Errata accessibility from audit
The original audit document `HEALTH-AUDIT-2026-06-21.md` MUST contain a footer link to the Errata, so that readers discovering the audit in isolation are aware that corrections exist. The Errata MUST also link back to the audit.

#### Scenario: Audit footer references Errata
- **WHEN** reader reads the end of `HEALTH-AUDIT-2026-06-21.md`
- **THEN** the document MUST contain a link to `HEALTH-AUDIT-2026-06-21-ERRATA.md` in the report metadata section or footer

#### Scenario: Errata links back to audit
- **WHEN** reader opens `HEALTH-AUDIT-2026-06-21-ERRATA.md`
- **THEN** the document MUST contain a back-reference to the original audit (`HEALTH-AUDIT-2026-06-21.md`) at commit `baa8c4e`