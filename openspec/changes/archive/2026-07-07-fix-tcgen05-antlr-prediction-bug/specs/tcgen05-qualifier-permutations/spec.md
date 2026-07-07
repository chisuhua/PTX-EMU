## ADDED Requirements

### Requirement: tcgen05 grammar SHALL support qualifier ordering permutations
The ANTLR grammar in `src/grammar/ptxInstructions.g4` MUST resolve the
LL(*) prediction bug in the `tcgen05Inst` rule's qualifier list Kleene star.
The fix SHALL ensure that qualifiers can appear in any order per PTX ISA
§9.7.16 specification, without requiring fixture reordering as a workaround.

The fix SHALL NOT regress:
- 2 existing PTX fixtures from Change-3a (`tests/ptx/tcgen05_alloc.ptx`,
  `tests/ptx/tcgen05_mma.ptx`)
- 10 new PTX fixtures from Change-3a
- 4 workaround fixtures (those using reordered qualifiers to bypass the bug)
- All 34 pre-existing PTX fixtures (e.g., `test_syncthreads_simple.ptx`)

#### Scenario: previously failing qualifier orders now parse
- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after grammar fix
- **THEN** all 4 previously-failing qualifier orderings parse successfully:
  - `tcgen05.ld.sync.aligned.32x32b.shared::cta.b32`
  - `tcgen05.st.sync.aligned.32x32b.shared::cta.b32`
  - `tcgen05.cp.sync.aligned.128x128b.shared::cta.b32`
  - `tcgen05.mma.cta_group::1.kind::f16`
- **AND** all 12 Change-3a fixtures still parse successfully
- **AND** total fixture count grows from 46 to 50 (4 repro + 8 permutations)

#### Scenario: ANTLR generation succeeds with no new warnings
- **WHEN** `cmake --build build --target GenerateParser` is run
- **THEN** ANTLR generates C++ parser source without errors
- **AND** no new warnings appear beyond pre-existing `BANG`/`F4` implicit warnings
- **AND** generated code size does not increase by more than 10% (Kleene star → recursive rewrite should be similar)

#### Scenario: ctest zero regression
- **WHEN** `cd build && ctest -L "unit|integration" --output-on-failure` is run
- **THEN** 100% of existing 123 tests pass
- **AND** no test gap is introduced

#### Scenario: full sanity validation passes
- **WHEN** `./scripts/sanity.sh` is run
- **THEN** all tiers pass (mini, integration, e2e, ptx, shared_memory, etc.)
- **AND** PTX syntax test reports 50/50 PASS (46 existing + 4 repro fixtures, post-fix)
