## MODIFIED Requirements

### Requirement: cp handler SHALL route TMEM destination through instruction-specified slot (Oracle C2 fix)

The system SHALL provide `processTcgen05Cp` in
`src/ptxsim/instructions/tcgen05_cp.cpp` that reads from shared memory
and writes to TMEM, where the destination TMEM slot MUST come from
`Tcgen05Instr.tmem_slot` (per Oracle C2 fix in
`fix-tcgen05-ld-st-slot-routing`). The previous implementation used a
hardcoded `constexpr size_t kDestSlot = 0` constant which made the
cp instruction unable to feed data into the same slots that
subsequent mma instructions consume.

#### Scenario: cp handler reuses smem address resolution and writes to instruction-specified TMEM slot
- **WHEN** `tcgen05.cp.cta_group::1.shared::cta [tmem_dst=N], [smem_src], shape` is dispatched with `instr.tmem_slot = N`
- **THEN** the handler reads from shared memory via `SharedMemoryManager` (existing path, no new `SmemDescriptor` abstraction)
- **AND** writes 128 bytes to TMEM slot `N` via `tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize)` (NOT hardcoded `kDestSlot = 0`, per Oracle C2 fix)
- **AND** if `N >= kSlotCount`, the handler throws `std::out_of_range` with message containing "tmem_slot"

#### Scenario: cp to default slot 0 preserves backward compatibility
- **WHEN** `tcgen05.cp` is dispatched with `instr.tmem_slot = 0` (default field value)
- **THEN** the handler writes to TMEM slot 0 (identical behavior to pre-C2-fix hardcoded implementation)
- **AND** existing cp tests (`test_tcgen05_cp.cpp`, `tests/integration/tcgen05/test_tcgen05_cp.cpp`) continue to PASS

#### Scenario: cp to slot 32 enables downstream mma consumption (FlashAttention data flow)
- **WHEN** `processTcgen05Cp(tmem_slot=32)` is called to load a tile of data
- **AND** then `processTcgen05Mma` is called (which reads TMEM slots 0..63 by default per `tcgen05_helpers.cpp:21`)
- **THEN** the mma result reflects the tile loaded into slot 32 (assuming the kernel places slot 32 data in the A/B slot range that mma reads)
- **AND** the test `tests/integration/tcgen05/test_tcgen05_cp_data_flow.cpp` (added by FU-5 `tcgen05-flashattention-coverage`) SHALL verify this end-to-end

## REMOVED Requirements

### Requirement: Hardcoded `kDestSlot = 0` constant in tcgen05_cp.cpp
**Reason**: The hardcoded `constexpr size_t kDestSlot = 0` in `tcgen05_cp.cpp:130` made the cp instruction unable to feed data into mma-consumed slots, breaking FlashAttention's data flow. The Oracle C2 audit (2026-07-11) identified this as a HIGH-confidence BLOCKER.

**Migration**: Per `fix-tcgen05-ld-st-slot-routing` change:
- Remove the `constexpr size_t kDestSlot = 0;` declaration from `tcgen05_cp.cpp:130`
- Replace `tmem.write(kDestSlot, tmp, Tmem::kSlotSize)` (line 138) with `tmem.write(instr.tmem_slot, tmp, Tmem::kSlotSize)`
- Ensure `Tcgen05Instr` is the source of slot — no fallback constants allowed
- Existing callers using default `tmem_slot=0` retain identical behavior
