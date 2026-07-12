## ADDED Requirements

### Requirement: processTcgen05Mma SHALL read accumulate bit from idesc register at handler time (Oracle C1 fix)

The system SHALL modify `processTcgen05Mma` (`src/ptxsim/instructions/tcgen05.cpp:355-393`) to read the accumulate bit from the `idesc` register operand at handler time, rather than hardcoding `accumulate=false`. This enables real PTX `mma.accumulate::x` semantics to flow through to the helper, which is required for FlashAttention QK^T/PV matrix multiplication `C += A*B` K-loop validation.

The idesc operand (PTX ISA §9.7.16, operand[3]) is a `RegOperand` referencing a register whose runtime `uint32_t` value contains the accumulate bit at a (TBD) bit position. The handler SHALL extract this bit and pass it as the `accumulate` parameter to the fragment helper.

#### Scenario: idesc register value with accumulate bit set triggers helper accumulate path

- **WHEN** `processTcgen05Mma` is called with a `Tcgen05Instr` whose `operands[3]` is a `RegOperand` (e.g., `%r5`) and the thread's `register_bank_[%r5]` value has the accumulate bit set
- **THEN** the handler extracts `accumulate = true` from the idesc register
- **AND** calls `tcgen05_fragment_mma_f16(tmem, warp_id, /*accumulate=*/true)`
- **AND** the helper reads the existing C slot contents, accumulates with the new `A * B` sum, and writes back the result
- **AND** a subsequent `processTcgen05Mma` call with identical inputs (and accumulate bit set) SHALL result in `C == 2 × GOLDEN_MMA_F16_F16_F32` (validation scenario, FlashAttention QK^T iteration 1 → 2 pattern)

#### Scenario: idesc register value with accumulate bit cleared triggers helper overwrite path

- **WHEN** `processTcgen05Mma` is called with a `Tcgen05Instr` whose `operands[3]` register value has the accumulate bit cleared (initial state or explicitly cleared)
- **THEN** the handler extracts `accumulate = false` from the idesc register
- **AND** calls `tcgen05_fragment_mma_f16(tmem, warp_id, /*accumulate=*/false)`
- **AND** the helper overwrites C slot with `A * B` (current overwrite semantics)
- **AND** a subsequent `processTcgen05Mma` call with identical inputs (and accumulate bit cleared) SHALL result in `C == 1 × GOLDEN_MMA_F16_F16_F32` (regression guard that bit clearing correctly produces overwrite, not stale value)

#### Scenario: helper signature extended to accept warp_id (per FU-4 API sync)

- **WHEN** `tcgen05_fragment_mma_f16` is declared in `include/ptxsim/instructions/tcgen05_helpers.h:51`
- **THEN** its signature SHALL be `tcgen05_fragment_mma_f16(Tmem& tmem, int warp_id, bool accumulate = false)`
- **AND** `warp_id` is used to compute per-warp C slot offset: `c_slot = static_cast<size_t>(warp_id) * 32 + static_cast<size_t>(64) + static_cast<size_t>(lane_id)` (single-warp callers passing `warp_id=0` preserve prior layout)
- **AND** the helper header comment SHALL document the single-warp execution requirement and the warp_id parameter's purpose (per FU-4 future work)

#### Scenario: accumulate bit position calibration procedure

- **WHEN** the initial bit mask in handler reads `accumulate = (idesc_val & 0x1u)` (placeholder bit 0)
- **THEN** the T4/T5 test cases SHALL serve as calibration fixtures
- **AND** if T4 (idesc=1 → 2× GOLDEN) or T5 (idesc=0 → 1× GOLDEN) fails, the developer SHALL adjust the bit mask (e.g., `0x2u`, `0x4u`) until both tests pass
- **AND** the final bit mask position SHALL be recorded in ADR-0016 Postmortem section

#### Scenario: handler does not modify grammar or qualifier enum (per Non-Goals)

- **WHEN** this change is implemented
- **THEN** no modifications to `src/grammar/ptxLexer.g4`, `src/grammar/ptxInstructions.g4`, `include/ptx_ir/ptx_qualifier.def` SHALL occur (idesc is parsed as a `RegOperand` operand, not a qualifier)
- **AND** no new `Tcgen05Instr` qualifier enum value SHALL be added for accumulate

#### Scenario: handler signature preserves PTX ISA §9.7.16 single-warp semantics

- **WHEN** `processTcgen05Mma` is called from a warp with `warp_id = N` (`N >= 0`)
- **THEN** `warp_id` passed to helper SHALL equal `N`
- **AND** the helper SHALL compute `c_slot = N * 32 + 64 + lane_id`
- **AND** for single-warp callers passing `warp_id = 0`, the layout SHALL be identical to the prior formula `64 + lane_id` (backward compatibility)

#### Scenario: thread register accessor API availability (per OpenQuestion OQ1)

- **WHEN** this change is implemented
- **THEN** `thread.read_reg_32(reg)` accessor SHALL be available with signature `uint32_t ThreadContext::read_reg_32(const RegOperand& reg) const`
- **AND** if this accessor does not exist, a minimal accessor SHALL be added with unit test coverage for register-value retrieval by name and index

### Requirement: pt-xtest fixture for `.accumulate::x` PTX syntax SHALL be added (per Phase 2)

The system SHALL add a PTX syntax test fixture at `tests/ptx/tcgen05_mma_with_accumulate.ptx` that contains the `.accumulate::x` qualifier on a `tcgen05.mma` instruction. This fixture SHALL be parseable (not executable end-to-end — execution requires handler change per main requirement).

#### Scenario: PTX fixture parseable by ANTLR

- **WHEN** `tests/ptx/tcgen05_mma_with_accumulate.ptx` is parsed by the ANTLR4 PTX parser (`src/grammar/ptxParser.g4`)
- **THEN** no ANTLR parse errors SHALL occur
- **AND** the parsed `Tcgen05Instr` SHALL have `op_kind == Tcgen05OpKind::MMA`
- **AND** the parsed `Tcgen05Instr` SHALL carry the idesc `RegOperand` in `operands[3]` (the `.accumulate::x` semantic is NOT yet a qualifier — it remains internal to idesc per PTX ISA §9.7.16)

#### Scenario: PTX fixture SHALL be included in `tests/ptx/test_all_ptx.sh`

- **WHEN** `./tests/ptx/test_all_ptx.sh` is run after the fixture is added
- **THEN** the script SHALL include the new fixture in its discovery loop
- **AND** the fixture SHALL parse successfully with no regressions to existing fixtures (per lessons-learned §L ANTLR modification discipline)
