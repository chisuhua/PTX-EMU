## ADDED Requirements

### Requirement: AGENTS.md SHALL reflect post-implementation tcgen05 state
After Change-1/2/3a/3b archive, the root `AGENTS.md` known-limitations
table SHALL be updated to mark Blackwell tcgen05 as implemented
(rather than "permanently throws") and reference ADR-0016.

#### Scenario: root AGENTS.md updated
- **WHEN** `cat AGENTS.md | grep "tcgen05"` is run after change archive
- **THEN** the line refers to "Blackwell tcgen05 handler implemented (per ADR-0016)"
- **AND** "pre-Blackwell WMMA permanently throws" line remains

#### Scenario: ADR-0016 updated record
- **WHEN** `tail -50 docs/adr/0016-blackwell-only-tcgen05.md` is run
- **THEN** a new "更新记录" entry exists with the change's commit hash and date

### Requirement: openspec/specs/ SHALL have final spec state
All 3 published specs from Change-1 (`tcgen05-grammar`, `tcgen05-ir-types`,
`tcgen05-parse-tests`) SHALL be marked as "implemented" in their headers
or in a final sync note.

#### Scenario: spec headers reflect implementation status
- **WHEN** `ls openspec/specs/tcgen05-*/spec.md` is run
- **THEN** each spec has an "Status: implemented" line at the top
