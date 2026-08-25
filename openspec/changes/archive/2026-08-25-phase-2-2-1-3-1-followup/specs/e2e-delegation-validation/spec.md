# e2e-delegation-validation spec

> **REVISION HISTORY** (per Metis session `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`):
> - Initial spec referenced `result.lanes[i] == ThreadState::kRun` as if `lanes` was `std::array<ThreadState, 32>`; **Corrected**: `lanes` is `std::vector<LaneStatus>` where each entry has `lane_id` + `state` + `pc` fields (per `include/ptxemu/device_api.h:62-66`)

## ADDED Requirements

### Requirement: `tests/integration/warp/test_device_api_delegation_e2e.cc` validates thread PC reflects delegated state changes via `execute_warp_instruction`

The integration test `tests/integration/warp/test_device_api_delegation_e2e.cc` MUST be driven via `WarpContext::execute_warp_instruction` to verify that delegated state changes (via `IPtxEmuDevice::set_next_pc` + `set_active_mask`) are reflected in the SIMT execution pipeline's observable thread PC and warp state. The test MUST follow the `test-coverage-enforcer` skill's e2e test pattern.

> **Setup**: GPUContext with at least 1 SM + 1 warp containing 32 threads, kernel loaded with simple PTX (e.g., `add` instruction) to provide measurable execution state.

#### Scenario: Delegated set_next_pc observed in subsequent warp execution

- **WHEN** `device.set_next_pc(0, 0, 0, 42)` is called
- **AND** then `device.warp_exe_once(0, 0)` is called
- **THEN** thread `(sm=0, warp=0, lane=0)` fetches its instruction from PC=42 (or subsequent PC after 42)
- **AND** `device.get_thread_state(0, 0, 0)` reflects the post-execution state (NOT hardcoded `kIdle`)
- **AND** `device.get_warp_status(0, 0).lanes[0].pc == 42` (or subsequent PC)

#### Scenario: Delegated set_active_mask observable in subsequent warp execution

- **WHEN** warp 0's current mask is `0xFF` (all lanes active, via `device.set_active_mask(0, 0, 0xFF)` or warp initialization)
- **AND** `device.set_active_mask(0, 0, 0x01)` is called (overwrite to lane 0 only)
- **AND** then `device.warp_exe_once(0, 0)` is called
- **THEN** only lane 0 executes
- **AND** `device.get_warp_status(0, 0).active_count == 1` (matches `warp_active_mask::count_active_lanes()` after overwrite)
- **AND** `device.get_warp_status(0, 0).lanes[1..31]` reflects non-execution state (`kIdle` or unchanged)

#### Scenario: warp_exe_once advances single warp (not all warps)

- **WHEN** warp 0 has pending instructions and warp 1 does not
- **AND** `device.warp_exe_once(0, 0)` is called
- **THEN** warp 0's PC advances (or stays kernel-finished)
- **AND** warp 1's PC is unchanged
- **AND** `device.get_warp_status(0, 0)` reflects the post-execution warp 0 state
- **AND** `device.get_warp_status(0, 1)` reflects unchanged warp 1 state

### Requirement: `tests/integration/warp/test_warp_status_snapshot.cpp` validates `get_warp_status` semantics

The unit test `tests/integration/warp/test_warp_status_snapshot.cpp` MUST validate `IPtxEmuDevice::get_warp_status` returns correctly populated 5-field `WarpStatus` for various warp states:

- All lanes active (`active_count == 32`, all `lanes[i].state == kRun`)
- No lane active (`active_count == 0`, all `lanes[i].state == kExit` or `kIdle`)
- Mixed active/inactive
- All lanes finished (all `lanes[i].state == kExit`)
- Blocked threads contribute to `blocked_cycles` field

#### Scenario: All lanes active snapshot

- **WHEN** warp 0 has all 32 lanes active (`threads[i].is_active && !threads[i].is_exited`)
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.active_count == 32`
- **AND** `result.lanes.size() == 32`
- **AND** `result.lanes[i].lane_id == i` for all `i ∈ [0, 32)`
- **AND** `result.lanes[i].state == ThreadState::kRun` (or `kBarSync` for blocked threads) for all `i`

#### Scenario: All lanes finished snapshot

- **WHEN** warp 0 has all 32 threads in `is_exited == true`
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.active_count == 0`
- **AND** `result.lanes[i].state == ThreadState::kExit` for all `i`

#### Scenario: No lane active snapshot

- **WHEN** warp 0 has all lanes inactive (mask = `0x0`)
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.active_count == 0`
- **AND** `result.lanes[i].state` reflects per-thread state (might be `kIdle` or `kExit`)

#### Scenario: Blocked threads contribute to blocked_cycles

- **WHEN** warp 0 has threads 0-3 with `blocked_cycles_remaining == 10` each (others 0)
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.blocked_cycles == 40` (sum of blocked_cycles_remaining across all threads)

#### Scenario: warp_id and sm_id fields

- **WHEN** `device.get_warp_status(5, 7)` is called (sm_id=5, warp_id=7)
- **THEN** `result.warp_id == 7`
- **AND** `result.sm_id == 5`