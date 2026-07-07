## ADDED Requirements

### Requirement: Old integration tests SHALL add S_TCGEN05_* compile-time aliases alongside existing WMMA paths
SHALL update the 2 existing `tests/integration/tcgen05/test_tcgen05_*.cpp` files
to add `makeTcgen05Instr(Tcgen05OpKind::*, ...)` calls as **compile-time verification
aliases** alongside the existing `makeWmmaInstr`/`WmmaType` calls (per `design.md` D3
additive strategy). The new aliases SHALL NOT be inserted into the step_warp
execution vector — they exist solely to verify that `makeTcgen05Instr` factory
compiles and that the B2 op_kind→StatementType switch produces the correct
`S_TCGEN05_*` type (validated via `static_assert`). Both old WMMA paths and new
Tcgen05 compile-time aliases SHALL coexist — all tests SHALL PASS because only
the old paths are executed. Full runtime replacement is deferred to
`implement-tcgen05-handlers-core` when the independent `Tcgen05PipelineHandler`
is registered.

#### Scenario: test_tcgen05_mma_sync aliased (compile-time only)
- **WHEN** `tests/integration/tcgen05/test_tcgen05_mma_sync.cpp` is read
- **THEN** both `makeWmmaInstr(WmmaType::WMMA_MMA, ...)` calls remain in the execution vector
- **AND** equivalent `makeTcgen05Instr(Tcgen05OpKind::MMA, ...)` compile-time aliases are added OUTSIDE the execution vector
- **AND** a `static_assert` verifies the alias statement type is `S_TCGEN05_MMA`
- **AND** `cd build && ctest -R "integration_tcgen05_mma_sync" -V` PASSES (old paths execute unchanged)

#### Scenario: test_tcgen05_ld_st_commit aliased (compile-time only)
- **WHEN** `tests/integration/tcgen05/test_tcgen05_ld_st_commit.cpp` is read
- **THEN** all 5 existing `makeWmmaInstr` calls remain in the execution vector
- **AND** equivalent `makeTcgen05Instr(Tcgen05OpKind::*, ...)` compile-time aliases are added OUTSIDE the execution vector
- **AND** `static_assert` verifies each alias statement type matches the expected `S_TCGEN05_*`
- **AND** `cd build && ctest -R "integration_tcgen05_ld_st_commit" -V` PASSES (old paths execute unchanged)

#### Scenario: Q_TCGEN05_* stubs retained (deferred to implement-tcgen05-handlers-core)
- **WHEN** `grep "Q_TCGEN05_LD\|Q_TCGEN05_ST\|Q_TCGEN05_COMMIT\|Q_TCGEN05_WAIT" include/ptx_ir/ptx_qualifier.def` is run
- **THEN** 4 stubs are still present (deferred deletion until handler creation; see design.md D4)
- **AND** the stubs remain functional for wmma.cpp dispatch (8 references) and 1 test file (4 references)