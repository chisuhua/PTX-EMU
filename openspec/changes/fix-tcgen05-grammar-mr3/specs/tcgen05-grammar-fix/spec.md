## ADDED Requirements

### Requirement: tcgen05 grammar LL(*) conflict SHALL be fixed
The ANTLR grammar in `src/grammar/ptxInstructions.g4` SHALL resolve the
LL(*) prediction conflict that causes `tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32`
to fail parsing with `mismatched input '.all' expecting ':'`. The fix
SHALL NOT regress the 2 existing PTX fixtures (`tests/ptx/tcgen05_alloc.ptx`,
`tests/ptx/tcgen05_mma.ptx`).

#### Scenario: 2 existing fixtures PASS after fix
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after grammar fix
- **THEN** `tcgen05_alloc.ptx` and `tcgen05_mma.ptx` parse successfully
- **AND** no parse error is raised

#### Scenario: ANTLR generation succeeds
- **WHEN** `cmake --build build --target GenerateParser` is run
- **THEN** ANTLR generates C++ parser source without errors
- **AND** only pre-existing warnings (BANG/F4 implicit, empty optional blocks) appear

#### Scenario: ctest zero regression
- **WHEN** `cd build && ctest --output-on-failure` is run
- **THEN** 100% of existing tests pass
- **AND** no test gap is introduced
