# device-api-delegation-completion spec

> **REVISION HISTORY** (per Metis session `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`):
> - Initial spec proposed `WarpStatus` 4 fields (std::array<ThreadState, 32> + active_mask + blocked_cycles_remaining + finished)
> - **Corrected**: `WarpStatus` is 5-field struct at `include/ptxemu/device_api.h:69-75` (warp_id + sm_id + lanes[vector<LaneStatus>] + active_count + blocked_cycles). No layout change allowed per HSK-8 spec §Decision 5.
> - Initial spec proposed `thread->state` field access; **Corrected**: correct API is `thread->get_state()` returning `EXE_STATE` (per `include/ptxsim/thread_context.h:205`)

## ADDED Requirements

### Requirement: `IPtxEmuDevice::warp_exe_once` advances a single warp via `WarpContext::execute_warp_instruction`

The `IPtxEmuDevice::warp_exe_once(uint32_t sm_id, uint32_t warp_id)` method MUST delegate to `SMContext::get_warp(warp_id)->execute_warp_instruction()` for the given `(sm_id, warp_id)`. This advances **one** warp (per-warp scheduler semantics, NOT all warps via `g_gpu_context->exe_once()`). The method MUST return `0` on success, `-1` if `sm_id`/`warp_id` is invalid.

> **CRITICAL invariant**: This is **state-mutating** (advances PC, registers, scoreboard). Per `ptx-instruction-pipeline` skill, must NOT bypass any barrier/scoreboard invariant. Per `ptx-barrier-mechanism`, must respect `BarrierModule::release_warp_barrier` overwrite semantics (BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES guard).

#### Scenario: Valid sm_id + warp_id

- **WHEN** `device.warp_exe_once(0, 0)` is called on a GPU with SM 0 containing warp 0
- **THEN** the method delegates to `WarpContext::execute_warp_instruction()` for warp 0
- **AND** returns `0`

#### Scenario: Invalid sm_id

- **WHEN** `device.warp_exe_once(invalid_sm_id, 0)` is called
- **THEN** the method returns `-1` without crashing
- **AND** no warp state is modified

#### Scenario: Invalid warp_id

- **WHEN** `device.warp_exe_once(0, invalid_warp_id)` is called
- **THEN** the method returns `-1` without crashing
- **AND** no warp state is modified

### Requirement: `IPtxEmuDevice::get_thread_state` reads from `ThreadContext::get_state`

The `IPtxEmuDevice::get_thread_state(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id)` method MUST read from `warp->get_thread(lane_id)->get_state()` (per `include/ptxsim/thread_context.h:205`), which returns `EXE_STATE`. The `EXE_STATE` value MUST then be mapped to `ptxemu::ThreadState` via the existing `map_state` helper in `device_api_impl.cc:45-53`. The method MUST NOT mutate any state.

> **CRITICAL invariant**: Per HSK-8 spec §Decision 6, the `ThreadState` enum MUST correspond 1:1 to `ptxsim::EXE_STATE`. Returning hardcoded `ThreadState::kIdle` (the pre-fix stub behavior) violates this invariant.

#### Scenario: Valid thread state read

- **WHEN** `device.get_thread_state(0, 0, 0)` is called on a GPU with thread (sm=0, warp=0, lane=0) in `EXE_STATE::RUN`
- **THEN** the method returns `ThreadState::kRun`

#### Scenario: Thread state in BAR_SYNC

- **WHEN** `device.get_thread_state(0, 0, 0)` is called on a thread in `EXE_STATE::BAR_SYNC`
- **THEN** the method returns `ThreadState::kBarSync`

#### Scenario: Thread exited

- **WHEN** `device.get_thread_state(0, 0, 0)` is called on a thread in `EXE_STATE::EXIT`
- **THEN** the method returns `ThreadState::kExit`

#### Scenario: Invalid sm_id returns kIdle (graceful degradation)

- **WHEN** `device.get_thread_state(invalid_sm_id, 0, 0)` is called
- **THEN** the method returns `ThreadState::kIdle` (conservative default)
- **AND** no warp state is modified (read-only access)

#### Scenario: Invalid warp_id returns kIdle (graceful degradation)

- **WHEN** `device.get_thread_state(0, invalid_warp_id, 0)` is called
- **THEN** the method returns `ThreadState::kIdle`

#### Scenario: Invalid lane_id returns kIdle (graceful degradation)

- **WHEN** `device.get_thread_state(0, 0, invalid_lane_id)` is called
- **THEN** the method returns `ThreadState::kIdle`

### Requirement: `IPtxEmuDevice::get_warp_status` populates existing 5-field `WarpStatus` struct

The `IPtxEmuDevice::get_warp_status(uint32_t sm_id, uint32_t warp_id)` method MUST populate the existing 5-field `WarpStatus` struct at `include/ptxemu/device_api.h:69-75`:

- `warp_id`: equal to the input `warp_id` parameter
- `sm_id`: equal to the input `sm_id` parameter
- `lanes`: `std::vector<LaneStatus>` of size 32, where each entry `lanes[i] = {i, map_thread_status(threads[i].status), threads[i].pc}` (using `warp->get_warp_state().threads[i]` per `include/ptxsim/warp_state.h:14`)
- `active_count`: from `WarpState::count_active_lanes()` (per `include/ptxsim/warp_state.h:40`)
- `blocked_cycles`: sum of `threads[i].blocked_cycles_remaining` (per `include/ptxsim/thread_state.h:40`), cast to `int32_t` (clamped at `INT32_MAX` if sum overflows)

The method MUST NOT mutate any state (read-only snapshotter per `state-modification-audit`).

> **PUBLIC ABI**: The `WarpStatus` 5-field struct layout is FROZEN at `PTXEMU_API_VERSION=1` (per HSK-8 spec §Decision 5 "sizeof visibility is mandatory, pure data only"). This requirement does NOT introduce new public fields or modify existing field semantics.
>
> **LaneStatus 3-field struct** at `include/ptxemu/device_api.h:62-66`:
> ```cpp
> struct LaneStatus {
>     uint32_t lane_id = 0;
>     ThreadState state = ThreadState::kIdle;
>     uint32_t pc = 0;
> };
> ```

#### Scenario: Valid warp status snapshot

- **WHEN** `device.get_warp_status(0, 0)` is called on a GPU with warp 0 (sm_id=0)
- **THEN** the returned `WarpStatus.warp_id == 0`
- **AND** `WarpStatus.sm_id == 0`
- **AND** `WarpStatus.lanes.size() == 32`
- **AND** `WarpStatus.lanes[i].lane_id == i` for all `i ∈ [0, 32)`
- **AND** `WarpStatus.lanes[i].pc == warp->get_warp_state().threads[i].pc`
- **AND** `WarpStatus.active_count == warp->get_warp_state().count_active_lanes()`
- **AND** `WarpStatus.blocked_cycles == sum(warp->get_warp_state().threads[i].blocked_cycles_remaining)` (clamped at INT32_MAX)

#### Scenario: All lanes active snapshot

- **WHEN** warp 0 has all 32 lanes active (`threads[i].is_active && !threads[i].is_exited`)
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.active_count == 32`
- **AND** `result.lanes[i].state == ThreadState::kRun` (or `kBarSync` for blocked threads) for all `i`

#### Scenario: All lanes finished snapshot

- **WHEN** warp 0 has all 32 threads in `is_exited == true`
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.active_count == 0`
- **AND** `result.lanes[i].state == ThreadState::kExit` for all `i`

#### Scenario: Blocked threads contribute to blocked_cycles

- **WHEN** warp 0 has threads 0-3 with `blocked_cycles_remaining == 10` each (others 0)
- **AND** `device.get_warp_status(0, 0)` is called
- **THEN** `result.blocked_cycles == 40`

#### Scenario: Invalid sm_id returns default WarpStatus

- **WHEN** `device.get_warp_status(invalid_sm_id, 0)` is called
- **THEN** the method returns default-constructed `WarpStatus{}` (all 5 fields at default values)
- **AND** no warp state is modified

#### Scenario: Invalid warp_id returns default WarpStatus

- **WHEN** `device.get_warp_status(0, invalid_warp_id)` is called
- **THEN** the method returns default-constructed `WarpStatus{}`

### Requirement: `map_thread_status(ptxsim::ThreadStatus)` mapping helper

The implementation MUST include a pure mapping function `map_thread_status(ptxsim::ThreadStatus ts)` parallel to the existing `map_state(EXE_STATE)` helper (per `device_api_impl.cc:45-53`). The mapping MUST be:

- `ThreadStatus::Active` → `ThreadState::kRun`
- `ThreadStatus::Blocked` → `ThreadState::kBarSync`
- `ThreadStatus::Exited` → `ThreadState::kExit`
- `ThreadStatus::Yielded` → `ThreadState::kIdle` (conservative default; `ThreadState` enum frozen at 4 values per HSK-8 spec §Decision 6)

The function MUST be a pure function (no side effects).

#### Scenario: Mapping Active to kRun

- **WHEN** `map_thread_status(ThreadStatus::Active)` is called
- **THEN** the function returns `ThreadState::kRun`

#### Scenario: Mapping Blocked to kBarSync

- **WHEN** `map_thread_status(ThreadStatus::Blocked)` is called
- **THEN** the function returns `ThreadState::kBarSync`

#### Scenario: Mapping Exited to kExit

- **WHEN** `map_thread_status(ThreadStatus::Exited)` is called
- **THEN** the function returns `ThreadState::kExit`

#### Scenario: Mapping Yielded to kIdle

- **WHEN** `map_thread_status(ThreadStatus::Yielded)` is called
- **THEN** the function returns `ThreadState::kIdle`