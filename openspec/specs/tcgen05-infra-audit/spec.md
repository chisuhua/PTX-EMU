# tcgen05-infra-audit Specification

## Purpose
TBD - created by archiving change extend-blackwell-tcgen05-infra. Update Purpose after archive.
## Requirements
### Requirement: Blackwell tcgen05 infrastructure audit SHALL be published
The system SHALL provide a read-only audit report `docs/audits/2026-07-XX-tcgen05-infra-audit.md` covering **5 subsystems** (TmaDescriptor, Tmem, ClusterContext, TcQueue, wmma.cpp handlers L320-565 per Change-2 MR-5 scope expansion), grading each with a **documented readiness level** defined by **Decision 4 (L1=working / L2=needs-attention / L3=blocks)** with explicit criteria (all-tests-green, P0 UNVERIFIED count thresholds, fix-* backlog presence).

#### Scenario: audit report file exists
- **WHEN** `ls docs/audits/2026-07-*-tcgen05-infra-audit.md` is run
- **THEN** the audit file exists

#### Scenario: audit covers 5 subsystems (per MR-5)
- **WHEN** the audit file is read
- **THEN** it contains **5 sections** (TmaDescriptor, Tmem, ClusterContext, TcQueue, wmma.cpp handlers) + cross-subsystem pipeline
- **AND** each section has a readiness level (L1/L2/L3) with all-tests-green status + P0 count

#### Scenario: UNVERIFIED comments graded P0/P1/P2 (per Decision 5)
- **WHEN** the audit file is read
- **THEN** each `// UNVERIFIED-AGAINST-HARDWARE` comment **at implementation level** is graded P0 (blocks handler correctness) / P1 (precision) / P2 (edge case)
- **AND** reference data tables (e.g. `wmma.cpp:62-317` 256-entry fragment reference) are explicitly excluded via Decision 5 exclusion rule

#### Scenario: aggregate readiness determines Change-3 readiness (per Decision 4)
- **WHEN** the audit file is read
- **THEN** an **aggregate readiness** is computed as min-rule across 5 subsystems
- **AND** the report explicitly states "Change-3 (handlers) requires aggregate ≥ L2"

#### Scenario: NO set_state(BAR_SYNC) contract verified (per MR-1)
- **WHEN** the audit file is read
- **THEN** it contains a section verifying `tc_queue.h:16-17` / `tc_queue.cpp:13-14` `NO set_state(BAR_SYNC)` design contract
- **AND** `wmma.cpp` handler calls (`cta->tc_queue().commit(1)` at L523, `cta->tc_queue().wait(warp, 0, 1)` at L556) do NOT route through `set_state(BAR_SYNC)`

#### Scenario: no source code modified
- **WHEN** `git diff --stat main..feat/extend-blackwell-tcgen05-infra` is run
- **THEN** only `docs/audits/2026-07-XX-tcgen05-infra-audit.md` is changed under `src/` (zero diff)

