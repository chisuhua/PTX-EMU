# tcgen05-parse-tests Specification

## Purpose
TBD - created by archiving change implement-tcgen05-syntax-ir. Update Purpose after archive.
## Requirements
### Requirement: PTX Syntax Test Suite MUST Include tcgen05 Fixtures

The system SHALL provide 12 new `tests/ptx/tcgen05_*.ptx` files, one per Blackwell tcgen05 instruction family, that test end-to-end PTX text parsing. Each fixture MUST be a valid PTX snippet per NVIDIA PTX ISA 8.6 §9.7.16. The fixtures MUST be added to `./tests/ptx/test_all_ptx.sh` so that running the script validates all 12.

The 12 fixtures are:
1. `tcgen05_alloc.ptx` — `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [smem_addr], num_cols;`
2. `tcgen05_dealloc.ptx` — `tcgen05.dealloc.cta_group::1.sync.aligned.b32 tmem_addr, num_cols;`
3. `tcgen05_relinquish.ptx` — `tcgen05.relinquish_alloc_permit.cta_group::1.sync.aligned;`
4. `tcgen05_ld.ptx` — `tcgen05.ld.sync.aligned.32x32b.x4.b32 {r0,r1,r2,r3}, [tmem];`
5. `tcgen05_st.ptx` — `tcgen05.st.sync.aligned.32x32b.x2.b32 [tmem], {r0,r1};`
6. `tcgen05_cp.ptx` — `tcgen05.cp.cta_group::1.128x256b [tmem], sdesc;`
7. `tcgen05_cp_multicast.ptx` — `tcgen05.cp.cta_group::1.128x256b.multicast::cluster [tmem], sdesc, mask;`
8. `tcgen05_mma.ptx` — `tcgen05.mma.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc;`
9. `tcgen05_mma_block_scale.ptx` — `tcgen05.mma.cta_group::1.kind::f8f6f4.block_scale [d_tmem], a_desc, b_desc, idesc;`
10. `tcgen05_mma_ws.ptx` — `tcgen05.mma.ws.cta_group::1.kind::f16 [d_tmem], a_desc, b_desc, idesc;`
11. `tcgen05_commit.ptx` — `tcgen05.commit.cta_group::1.mbarrier::arrive::one.shared::cluster.b64 [mbar];`
12. `tcgen05_wait.ptx` — `tcgen05.wait.load [mbar_addr], phase;`
13. `tcgen05_fence.ptx` — `tcgen05.fence::before_thread_sync;`

#### Scenario: all-13-ptx-fixtures-parse
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after fixture additions
- **THEN** all 13 new tcgen05 fixtures parse successfully (100% pass)
- **AND** the test suite output shows PASS for each new fixture

#### Scenario: fixtures-use-real-ptx-syntax
- **WHEN** the fixture files are inspected
- **THEN** each fixture contains syntactically correct PTX per PTX ISA 8.6 §9.7.16
- **AND** the syntax matches examples from the NVIDIA official spec

#### Scenario: fixtures-cover-all-12-instruction-families
- **WHEN** the list of fixture files is enumerated
- **THEN** at least one fixture per Blackwell tcgen05 instruction family exists
- **AND** the families are: alloc, dealloc, relinquish, ld, st, cp, mma, mma.ws, mma.block_scale, commit, wait, fence

### Requirement: Unit Tests MUST Cover tcgen05 IR Types

The system SHALL provide 5 new unit test files in `tests/unit/ptx_ir/` that directly exercise Tcgen05Instr and Qualifier parsing without full grammar parse pipeline. The tests MUST use Catch2 patterns matching existing `tests/unit/ptx_ir/` style.

The 5 unit test files are:
1. `test_tcgen05_qualifier.cpp` — verify ~25 Q_* enum values are correctly defined and string-convertible
2. `test_tcgen05_opkind.cpp` — verify 11 Tcgen05OpKind enum values
3. `test_tcgen05_dtype.cpp` — verify 10 Tcgen05Dtype enum values
4. `test_tcgen05_statement_factory.cpp` — verify makeTcgen05Instr creates correct StatementContext
5. `test_tcgen05_instr_struct.cpp` — verify Tcgen05Instr field layout and default values

#### Scenario: unit-test-qualifier-coverage
- **WHEN** `cd build && ctest -R "unit_tcgen05_qualifier" -V` is run
- **THEN** the test PASSES
- **AND** all ~25 Q_* qualifiers are validated

#### Scenario: unit-test-opkind-coverage
- **WHEN** `cd build && ctest -R "unit_tcgen05_opkind" -V` is run
- **THEN** the test PASSES
- **AND** all 11 Tcgen05OpKind values are validated

#### Scenario: unit-test-statement-factory
- **WHEN** `cd build && ctest -R "unit_tcgen05_statement_factory" -V` is run
- **THEN** the test PASSES
- **AND** `makeTcgen05Instr` is verified to create correct StatementContext

### Requirement: Integration Tests MUST Cover End-to-End Grammar-to-IR Pipeline

The system SHALL provide 5 new integration test files in `tests/integration/parser/` that exercise the full ANTLR parse → IR pipeline for tcgen05 instructions. The tests MUST use the `ptx_parser` and `ptx_visitor_tcgen05` modules to parse real PTX snippets and verify the resulting `StatementContext` has correct `S_TCGEN05_*` type and `Tcgen05Instr` data.

The 5 integration test files are:
1. `test_tcgen05_mma_parse.cpp` — parse `tcgen05.mma.cta_group::1.kind::f16 [...]` → verify S_TCGEN05_MMA
2. `test_tcgen05_ld_parse.cpp` — parse `tcgen05.ld.sync.aligned.32x32b.x4.b32 {...}` → verify S_TCGEN05_LD with num_regs=4
3. `test_tcgen05_st_parse.cpp` — parse `tcgen05.st.sync.aligned.32x32b.x2.b32 [...]` → verify S_TCGEN05_ST
4. `test_tcgen05_commit_parse.cpp` — parse `tcgen05.commit.cta_group::1.mbarrier::arrive::one...` → verify S_TCGEN05_COMMIT
5. `test_tcgen05_wait_parse.cpp` — parse `tcgen05.wait.load [...]` → verify S_TCGEN05_WAIT

#### Scenario: integration-test-mma-parse
- **WHEN** `cd build && ctest -R "integration_tcgen05_mma_parse" -V` is run
- **THEN** the test PASSES
- **AND** the parsed statement has `type = S_TCGEN05_MMA`
- **AND** the parsed `Tcgen05Instr` has `op_kind = MMA`, `cta_group = 1`, `dtype = F16`

#### Scenario: integration-test-ld-num-regs
- **WHEN** the test parses `tcgen05.ld.32x32b.x4.b32 {r0,r1,r2,r3}, [tmem];`
- **THEN** the parsed `Tcgen05Instr` has `op_kind = LD`, `num_regs = 4`

#### Scenario: integration-test-all-5-pass
- **WHEN** all 5 integration tests are run
- **THEN** all PASS (no parser errors, all fields correctly populated)

### Requirement: Existing tcgen05 Tests MUST Be Migrated to S_TCGEN05_*

The system SHALL migrate the existing 2 integration test files in `tests/integration/tcgen05/` (`test_tcgen05_mma_sync.cpp` and `test_tcgen05_ld_st_commit.cpp`) to use `S_TCGEN05_*` enums and `Tcgen05Instr` struct, replacing all `S_WMMA` and `WmmaInstr` references. The migrated tests MUST continue to PASS with the same logic (only IR-level changes, no test logic changes).

#### Scenario: old-tcgen05-mma-test-migrated
- **WHEN** `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` is read
- **THEN** no `S_WMMA` references exist
- **AND** all `makeWmmaInstr` calls are replaced with `makeTcgen05Instr`
- **AND** `cd build && ctest -R "integration_tcgen05_mma_sync" -V` PASSES

#### Scenario: old-tcgen05-ld-st-test-migrated
- **WHEN** `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` is read
- **THEN** no `S_WMMA` references exist
- **AND** all `makeWmmaInstr` calls are replaced with `makeTcgen05Instr`
- **AND** `cd build && ctest -R "integration_tcgen05_ld_st_commit" -V` PASSES

### Requirement: Test CMakeLists MUST Register All New Tests

The system SHALL update `tests/unit/CMakeLists.txt` and `tests/integration/CMakeLists.txt` to register all new test files with proper CTest targets (per project convention `unit_*` / `integration_*` prefix). Each test target MUST have appropriate `LABELS` (e.g., `unit;tcgen05` or `integration;tcgen05;grammar`).

#### Scenario: all-new-tests-discoverable-by-ctest
- **WHEN** `cd build && ctest -N -L "unit;tcgen05"` is run
- **THEN** 5 new test targets are listed

#### Scenario: all-new-tests-discoverable-by-ctest-integration
- **WHEN** `cd build && ctest -N -L "integration;tcgen05"` is run
- **THEN** 5 new + 2 migrated = 7 test targets are listed

#### Scenario: ctest-test-all-passes
- **WHEN** `cd build && ctest --output-on-failure` is run after all changes
- **THEN** 100% of tests pass (no regression)
- **AND** the new 12 tests (5 unit + 5 integration + 2 migrated) all PASS

