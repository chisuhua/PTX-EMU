# delegation-thread-pc-invariants spec (delta for phase-2-2-1-3-1-followup)

## MODIFIED Requirements

### Requirement: Delegated `set_next_pc` propagates to thread PC visible to `execute_warp_instruction`

After `IPtxEmuDevice::set_next_pc(sm_id, warp_id, lane_id, pc)` is called, the next invocation of `WarpContext::execute_warp_instruction()` on the same `(sm_id, warp_id)` MUST observe `pc` as the target PC for `lane_id` (per ptx-barrier-mechanism §3 PC synchronization model). This invariant ensures the public delegation path integrates correctly with the SIMT execution pipeline without requiring internal-state bypass.

> **PHASE 2.2.1 ADDITION** (delta): `device.get_thread_state(...)` MUST reflect the post-execution state (per `device-api-delegation-completion/spec.md` §get_thread_state). This is the e2e validation hook: after `set_next_pc` + `warp_exe_once`, `get_thread_state` MUST NOT return hardcoded `kIdle`.

#### Scenario: Delegated PC observed in subsequent warp execution

- **WHEN** `device.set_next_pc(0, 0, 0, 42)` is called
- **AND** then `device.warp_exe_once(0, 0)` is called
- **THEN** thread `(sm=0, warp=0, lane=0)` fetches its instruction from PC=42 (or subsequent PC)
- **AND** `device.get_thread_state(0, 0, 0)` reflects the post-execution state (not hardcoded `kIdle`)
- **AND** `device.get_warp_status(0, 0).lanes[0].pc == 42` (or subsequent PC after delegation)
- **AND** no internal state corruption

#### Scenario: Delegated set_next_pc followed by warp_exe_once observable

- **WHEN** `device.set_next_pc(0, 0, 0, 42)` is called
- **AND** then `device.warp_exe_once(0, 0)` is called
- **THEN** thread `(sm=0, warp=0, lane=0)` fetches its instruction from PC=42 (or subsequent PC)
- **AND** `device.get_thread_state(0, 0, 0)` reflects the post-execution state
- **AND** `device.get_warp_status(0, 0)` reflects the post-execution warp 0 state (PC, active_count, blocked_cycles, lanes[].pc)

### Requirement: Delegated `set_active_mask` overwrite is observable in subsequent barrier execution

After `IPtxEmuDevice::set_active_mask(sm_id, warp_id, mask)` is called with overwrite semantics, the next invocation of `BarrierModule::arrive` / `BarrierModule::release_warp_barrier` on the same `(sm_id, warp_id)` MUST observe `mask` as the active mask for that warp. This guards against the **BUG-POSTBARRIER-TWOHALVES** regression vector (per ptx-lessons-learned §1): if delegation uses OR-merge instead of overwrite, post-barrier warp state would be inconsistent.

> **PHASE 2.2.1 ADDITION** (delta): `device.get_warp_status(0, 0).active_count` MUST reflect the overwrite (per `device-api-delegation-completion/spec.md` §get_warp_status). After overwrite to lane 0 only, `active_count == 1`.

#### Scenario: Overwrite observable in barrier release

- **WHEN** warp 0's current mask is `0xFF` (all lanes active, via `device.set_active_mask(0, 0, 0xFF)` or warp initialization)
- **AND** `device.set_active_mask(0, 0, 0x01)` is called (overwrite to lane 0 only)
- **AND** then `BarrierModule::release_warp_barrier(0, 0)` runs
- **THEN** the barrier release observes only lane 0 as active
- **AND** the released thread set equals `{lane=0}`, NOT `{lane=0, 1, 2, 3, 4, 5, 6, 7}` (which would indicate OR-merge regression)
- **AND** `device.get_warp_status(0, 0).active_count == 1` (verifies snapshotter reflects overwrite via `WarpState::count_active_lanes()`)

#### Scenario: Overwrite observable in subsequent warp_exe_once

- **WHEN** warp 0's current mask is `0xFF` (all lanes active)
- **AND** `device.set_active_mask(0, 0, 0x01)` is called (overwrite to lane 0 only)
- **AND** then `device.warp_exe_once(0, 0)` is called
- **THEN** only lane 0 executes
- **AND** `device.get_warp_status(0, 0).active_count == 1`
- **AND** `device.get_warp_status(0, 0).lanes[1..31]` reflects non-execution state (`kIdle` or unchanged from pre-warp_exe_once)