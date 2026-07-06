## ADDED Requirements

### Requirement: Blackwell tcgen05 infrastructure audit SHALL be published
The system SHALL provide a read-only audit report `docs/audits/2026-07-XX-tcgen05-infra-audit.md` covering 4 subsystems (TmaDescriptor, Tmem, ClusterContext, TcQueue), grading each with a readiness level (L1=working, L2=needs-attention, L3=blocks Change-3b).

#### Scenario: audit report file exists
- **WHEN** `ls docs/audits/2026-07-*-tcgen05-infra-audit.md` is run
- **THEN** the audit file exists

#### Scenario: audit covers 4 subsystems
- **WHEN** the audit file is read
- **THEN** it contains 4 sections (TmaDescriptor, Tmem, ClusterContext, TcQueue) + cross-subsystem pipeline
- **AND** each section has a readiness level (L1/L2/L3)

#### Scenario: UNVERIFIED comments graded P0/P1/P2
- **WHEN** the audit file is read
- **THEN** each `// UNVERIFIED-AGAINST-HARDWARE` comment is graded P0 (blocks handler correctness) / P1 (precision) / P2 (edge case)

#### Scenario: no source code modified
- **WHEN** `git diff --stat main..feat/extend-blackwell-tcgen05-infra` is run
- **THEN** only `docs/audits/2026-07-XX-tcgen05-infra-audit.md` is changed under `src/` (zero diff)
