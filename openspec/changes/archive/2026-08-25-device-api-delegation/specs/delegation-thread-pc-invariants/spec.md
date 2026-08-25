# delegation-thread-pc-invariants spec

## ADDED Requirements

### Requirement: Delegated `set_next_pc` propagates to thread PC visible to `execute_warp_instruction`

After `IPtxEmuDevice::set_next_pc(sm_id, warp_id, lane_id, pc)` is called, the next invocation of `WarpContext::execute_warp_instruction()` on the same `(sm_id, warp_id)` MUST observe `pc` as the target PC for `lane_id` (per ptx-barrier-mechanism §3 PC synchronization model). This invariant ensures the public delegation path integrates correctly with the SIMT execution pipeline without requiring internal-state bypass.

#### Scenario: Delegated PC observed in subsequent warp execution

- **WHEN** `device.set_next_pc(0, 0, 0, 42)` is called
- **AND** then `WarpContext::execute_warp_instruction()` runs for `(sm=0, warp=0)`
- **THEN** thread `(sm=0, warp=0, lane=0)` fetches its instruction from PC=42
- **AND** the SIMT-PC manager records the dispatch
- **AND** no internal state (e.g., `set_active_mask`, scoreboard) is corrupted by the delegation

### Requirement: Delegated `set_active_mask` overwrite is observable in subsequent barrier execution

After `IPtxEmuDevice::set_active_mask(sm_id, warp_id, mask)` is called with overwrite semantics, the next invocation of `BarrierModule::arrive` / `BarrierModule::release_warp_barrier` on the same `(sm_id, warp_id)` MUST observe `mask` as the active mask for that warp. This guards against the **BUG-POSTBARRIER-TWOHALVES** regression vector (per ptx-lessons-learned §1): if delegation uses OR-merge instead of overwrite, post-barrier warp state would be inconsistent.

#### Scenario: Overwrite observable in barrier release

- **WHEN** warp 0's current `active_mask_` is `0xFF` (all lanes active)
- **AND** `device.set_active_mask(0, 0, 0x01)` is called (overwrite to lane 0 only)
- **AND** then `BarrierModule::release_warp_barrier(0, 0)` runs
- **THEN** the barrier release observes only lane 0 as active
- **AND** the released thread set equals `{lane=0}`, NOT `{lane=0, 1, 2, 3, 4, 5, 6, 7}` (which would indicate OR-merge regression)