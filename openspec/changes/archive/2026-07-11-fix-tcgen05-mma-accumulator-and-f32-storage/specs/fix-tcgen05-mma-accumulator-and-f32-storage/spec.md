## MODIFIED Requirements

### Requirement: tcgen05_fragment_mma_f16 helper SHALL support `+=` accumulator mode (Oracle H1 fix)

The system SHALL modify `tcgen05_fragment_mma_f16(Tmem&)` helper to accept an
optional `bool accumulate` parameter. When `accumulate=true`, the helper
SHALL read the existing C slot contents, convert from f16 to f32, accumulate
with the new sum, and write back. When `accumulate=false` (default), the
helper SHALL preserve its current overwrite semantics.

#### Scenario: accumulate=false preserves overwrite semantics (default backward compatibility)
- **WHEN** `tcgen05_fragment_mma_f16(tmem)` is called with the default `accumulate=false`
- **THEN** the helper overwrites the C slot with `A * B` (current behavior, no read of existing C slot)
- **AND** all existing tests pass without modification (except persistence T1, which is intentionally inverted per design.md D3)

#### Scenario: accumulate=true reads and accumulates into existing C slot
- **WHEN** `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/true)` is called
- **THEN** the helper reads the existing C slot (128 bytes at slot 64+lane_id per `tcgen05_helpers.cpp:23`)
- **AND** converts existing f16 values to f32, accumulates with new `A * B` sum
- **AND** writes back f16 storage (Phase 1) or f32 storage (Phase 2)
- **AND** C slot is preserved across multiple mma calls (FlashAttention QK^T/PV accumulator pattern)

#### Scenario: processTcgen05Mma preserves overwrite semantics
- **WHEN** `processTcgen05Mma` is called with a Tcgen05Instr (any qualifier set)
- **THEN** it explicitly invokes `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false)` per `tcgen05.cpp:383`
- **AND** no behavior change for callers of `processTcgen05Mma`

#### Scenario: caller must ensure single-warp execution for accumulate path
- **WHEN** accumulate=true path reads-then-writes C slot
- **THEN** the helper header comment SHALL warn that callers must ensure single-warp execution (per current SM scheduler sequential execution model)

### Requirement: tcgen05_fragment_mma_f16 helper SHALL store C output as f32 per PTX ISA §9.7.16 (Oracle H2 fix)

The system SHALL modify `tcgen05_fragment_mma_f16` helper to store the C
fragment output as `float` (f32) instead of `uint16_t` (f16), matching
PTX ISA §9.7.16 specification that `tcgen05.mma` produces `f16×f16→f32`
output. The 32 f32 elements per lane SHALL fill the entire 128-byte TMEM
slot.

#### Scenario: C output stored as native f32
- **WHEN** `tcgen05_fragment_mma_f16` writes the C slot (slot 64+lane_id per `tcgen05_helpers.cpp:23`)
- **THEN** `c_frag` array type SHALL be `std::array<float, ROWS * COLS_B>` (32 elements × 4 bytes = 128 bytes)
- **AND** `memcpy(c_buf.data(), c_frag.data(), c_frag.size() * sizeof(float))` fills the entire 128-byte slot
- **AND** no `f32_to_f16` conversion is performed (removed from `tcgen05_helpers.cpp:50`)

#### Scenario: readback tests use memcpy<float> instead of f16_to_f32
- **WHEN** integration tests read back the C slot
- **THEN** the readback pattern SHALL be `float val; std::memcpy(&val, &c_buf[idx * 4], sizeof(float));`
- **AND** `c_buf[idx * 2] | (c_buf[idx * 2 + 1] << 8)` + `f16_to_f32` pattern SHALL be removed from all readback sites
- **AND** `grep -rn "c_buf\[idx \* 2\]" tests/` SHALL return no matches (mechanical verification)

#### Scenario: golden value header documents f32 storage
- **WHEN** `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h:6-7` is read
- **THEN** the comment SHALL state "Storage format: f32 (per PTX ISA §9.7.16)"
- **AND** SHALL reference the change that introduced f32 storage (fix-tcgen05-mma-accumulator-and-f32-storage Phase 2)

#### Scenario: golden value numerical contents unchanged
- **WHEN** the 32 golden f32 values (1.0..32.0) are compared
- **THEN** they are identical to the previous f16-storage version (1..32 are exactly representable in both f16 and f32)
- **AND** `Catch::Approx` tolerance is sufficient for both storage formats

### Requirement: persistence test T1 SHALL validate accumulator behavior (Oracle H1 consequence)

The system SHALL modify `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp:184-203`
to validate the new accumulator semantics, and SHALL add a new TC that
validates that `accumulate=false` preserves overwrite behavior.

#### Scenario: T1 inverted to test accumulator (2nd mma yields 2× golden)
- **WHEN** `processTcgen05Mma` is called twice with identical A, B inputs (no intermediate cp)
- **THEN** the C slot SHALL equal `2 × GOLDEN_MMA_F16_F16_F32` (accumulation semantics)
- **AND** the test TC name SHALL be updated to `"processTcgen05Mma called twice with identical A,B accumulates into C (2nd mma yields 2× golden)"`
- **AND** the comment SHALL explain that H1 implementation reverses the previous overwrite expectation

#### Scenario: T1_overwrite validates explicit accumulate=false
- **WHEN** `tcgen05_fragment_mma_f16(tmem, /*accumulate=*/false)` is called twice with identical A, B inputs
- **THEN** the C slot SHALL equal `1 × GOLDEN_MMA_F16_F16_F32` (overwrite semantics preserved)
- **AND** the test TC name SHALL be `"processTcgen05Mma with accumulate=false leaves C unchanged (overwrite preserved)"`
- **AND** this TC serves as a regression guard that the optional `accumulate` parameter does NOT silently default to `true`

#### Scenario: T2 (mma → cp) and T3 (mma → cp → mma) unchanged
- **WHEN** persistence test T2 (`mma → cp` preserves C) and T3 (full chain) run
- **THEN** they SHALL pass without modification
- **AND** `accumulate=false` default in `processTcgen05Mma` ensures T2/T3 semantics match original design

## ADDED Requirements

### Requirement: tcgen05 fragment helper SHALL document semantic gap with PTX ISA idesc.accumulate (Oracle ADR-0016 debt)

The system SHALL record in ADR-0016 that the `accumulate` parameter in
`tcgen05_fragment_mma_f16` is a simulator-internal decision and does NOT
parse the real PTX `idesc.accumulate` bit from PTX ISA §9.7.16 standard
syntax `tcgen05.mma [taddr], adesc, bdesc, idesc, pred`. A follow-up
change SHALL be proposed to parse `idesc.accumulate` from the grammar
down to the helper.

#### Scenario: ADR-0016 documents idesc semantic gap
- **WHEN** `docs/adr/0016-blackwell-only-tcgen05.md` is read
- **THEN** a "2026-07-11 Postmortem: H1+H2 fix" section SHALL exist
- **AND** it SHALL explain: "Helper `accumulate` parameter is simulator 内部决策，不解析真实 PTX `idesc.accumulate` bit. 完整修复需要 grammar + parser + visitor + handler 全栈修改."
- **AND** it SHALL reference the follow-up change placeholder: `fix-tcgen05-idesc-parsing` (to be proposed)

#### Scenario: follow-up change placeholder is created
- **WHEN** the archive of `fix-tcgen05-mma-accumulator-and-f32-storage` completes
- **THEN** a follow-up change directory `openspec/changes/fix-tcgen05-idesc-parsing/` MAY be created as a placeholder (but not necessarily active)
- **AND** `.opencode/notes/` SHALL contain a TODO entry if not created

## REMOVED Requirements

### Requirement: f16 storage pattern in tcgen05 fragment helper (removed by Oracle H2 fix)

The system SHALL NOT store the C fragment output as `uint16_t` (f16) in
`tcgen05_fragment_mma_f16`. The previous implementation used `f32_to_f16`
conversion (per `tcgen05_helpers.cpp:50` pre-H2-fix) which contradicted
PTX ISA §9.7.16 and wasted 50% of TMEM slot capacity (64 bytes of 128-byte
slot were always zero).

#### Scenario: no f32_to_f16 in tcgen05_fragment_mma_f16
- **WHEN** `src/ptxsim/instructions/tcgen05_helpers.cpp` is read
- **THEN** `grep "f32_to_f16" src/ptxsim/instructions/tcgen05_helpers.cpp` SHALL return no matches within `tcgen05_fragment_mma_f16` function body
- **AND** `c_frag[i * COLS_B + j] = sum;` (direct f32 assignment) replaces `c_frag[i * COLS_B + j] = f32_to_f16(sum);`

#### Scenario: readback tests no longer use f16_to_f32 in tcgen05 readback sites
- **WHEN** `tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` and `tests/integration/tcgen05/test_tcgen05_mma_persistence.cpp` are read
- **THEN** `grep "f16_to_f32" tests/integration/tcgen05/test_tcgen05_mma_ws.cpp` SHALL return no matches in C slot readback contexts
- **AND** the readback pattern SHALL be `float val; std::memcpy(&val, &c_buf[idx * 4], sizeof(float));`

## Cross-Reference

- Oracle 2026-07-10 session `ses_0b3791d78ffewb52428kJJ2Irz` (H1 + H2 HIGH confidence blockers)
- Oracle 2026-07-10 API 审查 session `ses_0b026333bffePgrqVq7PDJNeR1` (idesc=RegOperand, accumulate=false default, alignas(16) memcpy readback, load_c_slot helper)
- Metis pre-implementation review session `ses_0b1a0cdb1ffenbhbciQ1n0x236` (3 MUST-RESOLVE 全部采纳)
- Ref: [`archive/2026-07-10-implement-tcgen05-handlers-extended/`](../../../archive/2026-07-10-implement-tcgen05-handlers-extended/)
- Step 1 commit `d3be589 test(tcgen05): add multi-op TMEM persistence integration test`
- [ADR-0016 §2026-07-11 Postmortem: H1+H2 fix](../../../docs/adr/0016-blackwell-only-tcgen05.md) (added in this change)
- [ptx-lessons-learned](../../../.opencode/skills/ptx-lessons-learned/SKILL.md) §3, §4, §6, §7
- [proposal.md](../../proposal.md), [design.md](../../design.md), [tasks.md](../../tasks.md)