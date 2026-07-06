## ADDED Requirements

### Requirement: Old integration tests SHALL migrate to S_TCGEN05_* namespace
The 2 existing `tests/integration/tcgen05/test_tcgen05_*.cpp` files
SHALL be migrated from `S_WMMA`/`makeWmmaInstr`/`WmmaType` to
`S_TCGEN05_*`/`makeTcgen05Instr`/`Tcgen05OpKind` after the grammar fix
lands. Migrated tests SHALL continue to PASS (behavior unchanged, only
IR namespace updated).

#### Scenario: test_tcgen05_mma_sync migrated
- **WHEN** `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` is read
- **THEN** no `S_WMMA` references exist
- **AND** all `makeWmmaInstr` calls are replaced with `makeTcgen05Instr(Tcgen05OpKind::MMA, ...)`
- **AND** `cd build && ctest -R "integration_tcgen05_mma_sync" -V` PASSES

#### Scenario: test_tcgen05_ld_st_commit migrated
- **WHEN** `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` is read
- **THEN** no `S_WMMA` references exist
- **AND** all `makeWmmaInstr` calls are replaced with `makeTcgen05Instr`
- **AND** `cd build && ctest -R "integration_tcgen05_ld_st_commit" -V` PASSES

#### Scenario: Q_TCGEN05_* stubs deleted
- **WHEN** `grep "Q_TCGEN05_LD\|Q_TCGEN05_ST\|Q_TCGEN05_COMMIT\|Q_TCGEN05_WAIT" include/ptx_ir/ptx_qualifier.def` is run
- **THEN** zero output (4 stubs deleted after wmma.cpp is updated)
