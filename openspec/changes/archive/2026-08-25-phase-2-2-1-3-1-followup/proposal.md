# phase-2-2-1-3-1-followup

> **REVISION HISTORY**:
> - **2026-08-25 (initial)**: First proposal — based on incorrect assumptions about `WarpStatus` struct layout and `WarpContext` internal API
> - **2026-08-25 (post-Metis REVISE)**: Corrected per Metis MUST-RESOLVE #1-3 (see session `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`)
>   - `WarpStatus` struct ACTUALLY has 5 fields at `include/ptxemu/device_api.h:69-75` (not 4 fields as initially designed)
>   - `thread->state` DOES NOT exist; correct API is `thread->get_state()` returning `EXE_STATE` (per `include/ptxsim/thread_context.h:205`)
>   - WarpContext thread access via `warp->get_warp_state().threads[]` (WarpState at `include/ptxsim/warp_state.h:10-72`)
>   - **Public ABI surface UNCHANGED** (per HSK-8 spec §Decision 5 "sizeof visibility is mandatory, pure data only")

## Why

`device-api-delegation` (archived 2026-08-25, commit `183a6ada`) implemented 4 of 12 `IPtxEmuDevice` methods, but **3 methods remain stub bodies** in `src/ptxemu/device_api_impl.cc` (`warp_exe_once` L90-94 / `get_thread_state` L114-118 / `get_warp_status` L156-160) and **1 e2e test** was DEFERRED. The `set_active_mask` overwrite integration test (`tests/integration/simt/test_set_active_mask_overwrite.cpp`) was committed with a `WARN + early-return when no warp exists` guard — full overwrite verification was deferred per commit `183a6ada` message.

The 3 deferred stubs cause functional gaps for CppTLM consumers (per HSK-8 spec §CppTLM 端接受条件 #1):
1. `warp_exe_once` — CppTLM scheduler cannot advance individual warps
2. `get_thread_state` / `get_warp_status` — return hardcoded `ThreadState::kIdle` / default-constructed `WarpStatus`, defeating HSK-8 spec §Decision 6

This change completes Phase 2.2.1/2.3.1 follow-up per HSK-8 follow-up plan §Phase 3 Task 3.1-3.2.

## What Changes

- **Modify** `src/ptxemu/device_api_impl.cc`:
  - `warp_exe_once` (L90-94): delegate to `SMContext::get_warp(warp_id)->execute_warp_instruction()` (instance method)
  - `get_thread_state` (L114-118): read `warp->get_thread(lane_id)->get_state()` (per `include/ptxsim/thread_context.h:205`) + map `EXE_STATE` → `ptxemu::ThreadState` via existing `map_state` helper (L45-53)
  - `get_warp_status` (L156-160): populate the existing 5-field `WarpStatus` struct at `include/ptxemu/device_api.h:69-75`:
    - `warp_id` / `sm_id` from input parameters
    - `lanes` as `std::vector<LaneStatus>` of size 32, each entry `{lane_id, map_thread_status(threads[i].status), threads[i].pc}` (using `warp->get_warp_state().threads[i]` per `include/ptxsim/warp_state.h:14`)
    - `active_count` from `WarpState::count_active_lanes()` (warp_state.h:40)
    - `blocked_cycles` as sum of `threads[i].blocked_cycles_remaining` (thread_state.h:40), capped to `int32_t` range
- **Add** `map_thread_status(ThreadStatus)` helper function (parallel to existing `map_state`) for `ptxsim::ThreadStatus` → `ptxemu::ThreadState` mapping
- **Modify** `tests/integration/simt/test_set_active_mask_overwrite.cpp`: remove `WARN + early-return` guard, add proper warp setup
- **Add** `tests/integration/warp/test_device_api_delegation_e2e.cc`: e2e test driven via `WarpContext::execute_warp_instruction`
- **Add** `tests/integration/warp/test_warp_status_snapshot.cpp`: unit test for `get_warp_status` semantics
- **Modify** `.github/workflows/drift_check.yml`: REMOVE Invariant 6 exemption for the 3 deferred stubs (now real implementations)

**Out of scope**:
- `cpp-tlm-consumes-ptxemu-device` (HSK-9 gated)
- New IPtxEmuDevice methods (would require HSK-9)
- `attach_timing` reverse-direction consumer wiring (separate concern)
- **Public ABI surface UNCHANGED** — `WarpStatus` 5-field struct at device_api.h:69-75 preserved; no sizeof changes; `PTXEMU_API_VERSION=1` frozen

## Capabilities

### New Capabilities

- `device-api-delegation-completion`: implements 3 deferred `IPtxEmuDevice` methods + e2e test + `WarpStatus` 5-field snapshotter (preserving public struct layout)
- `e2e-delegation-validation`: e2e test infrastructure for delegated state changes via `execute_warp_instruction` (per `test-coverage-enforcer`)
- `thread-status-mapping`: new mapping helper `map_thread_status(ThreadStatus)` for `ptxsim::ThreadStatus` → `ptxemu::ThreadState` conversion

### Modified Capabilities

- `delegation-thread-pc-invariants`: ADD scenario validating `get_thread_state` reflects post-execution state (no stale `kIdle`)
- `ci-drift-check`: REMOVE Invariant 6 deferred-stubs exemption for the 3 methods (now real implementations)

## Impact

**Source code**: ~80 LOC across 3 methods + 1 new helper (~20 LOC) + ~50 LOC test extensions.

**Build artifact**: `libptxemu_device.so` and `libcudart.so` unchanged (public ABI frozen at `PTXEMU_API_VERSION=1`, `WarpStatus` struct layout preserved).

**Test impact**: ctest grows from 249 to ~251 (adds: 1 unit warp_status_snapshot + 1 integration e2e delegation; set_active_mask overwrite test is a modification of existing test, not a new entry). All existing tests preserved.

**CI impact**: drift_check Invariant 6 exemption list shrinks from 3 to 0.

**HSK impact**: None. Public ABI surface unchanged. No new public methods.

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性 (CRITICAL — applies)
- `get_warp_status` populates **existing 5-field WarpStatus struct** (device_api.h:69-75), preserving public ABI:
  - `warp_id` (uint32_t) / `sm_id` (uint32_t) / `lanes` (std::vector<LaneStatus>) / `active_count` (uint32_t) / `blocked_cycles` (int32_t)
  - No new public types added; no existing public fields changed; no sizeof changes (HSK-8 spec §Decision 5 satisfied)
- **MUST** use `warp->get_warp_state().threads[]` for per-thread access (per `include/ptxsim/warp_state.h:14`)
- **MUST** use `thread->get_state()` for thread state read (per `include/ptxsim/thread_context.h:205`)

### 状态修改 (CRITICAL — applies)
- `warp_exe_once` **ADVANCES STATE** (PC, registers, scoreboard). Per `ptx-instruction-pipeline` skill, must NOT bypass barrier/scoreboard invariants; must respect `BarrierModule::release_warp_barrier` overwrite semantics (BUG-RETHANG guard).
- `get_thread_state` / `get_warp_status` are **READ-ONLY**; must NOT mutate state (audit per `state-modification-audit`)

### 多 Phase 推进 (N/A — single phase acceptable)
- 2 commits:
  - Commit 1: `warp_exe_once` + `get_thread_state`
  - Commit 2: `get_warp_status` + `map_thread_status` helper + e2e test + drift_check exemption removal

### 文档同步 (Checklist I)
- [ ] `include/ptxemu/AGENTS.md`: update `IPtxEmuDevice` method status table (12/12)
- [ ] `README.md` §已实现功能: update IPtxEmuDevice bullet
- [ ] `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md`: add Phase 2.2.1/2.3.1 note
- [ ] AGENTS.md HSK chain: add Phase 2.4 entry
- [ ] No change to `include/ptxemu/device_api.h` (struct fields preserved)

## Reference

- **Parent HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` §Phase 3 Task 3.1-3.2
- **Archived change** (this completes): `openspec/changes/archive/2026-08-25-device-api-delegation/`
- **HSK-8 spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md`
- **HSK-8 audit**: `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md`
- **Metis REVISE session**: `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`
- **Stub locations**: `src/ptxemu/device_api_impl.cc:90-94` / `:114-118` / `:156-160`
- **Public types (preserved)**: `include/ptxemu/device_api.h:62-66` (LaneStatus) + `:69-75` (WarpStatus 5 fields)
- **Internal APIs**:
  - `SMContext::get_warp(uint32_t) → WarpContext*`
  - `WarpContext::execute_warp_instruction()`
  - `WarpContext::get_thread(int lane_id) → ThreadContext*`
  - `ThreadContext::get_state() const → EXE_STATE` (`include/ptxsim/thread_context.h:205`)
  - `WarpContext::get_warp_state() → WarpState&`
  - `WarpState::threads[32]` (warp_state.h:14)
  - `WarpState::count_active_lanes()` (warp_state.h:40)
  - `ThreadState::blocked_cycles_remaining` (thread_state.h:40)
- **Public API frozen**: `include/ptxemu/device_api.h:117` `static_assert(PTXEMU_API_VERSION == 1, ...)`
- **Skills referenced**:
  - `ptx-lessons-learned` §1 + §3 + §4 + §21
  - `ptx-barrier-mechanism` (BUG-RETHANG guard)
  - `ptx-instruction-pipeline` (warp_exe_once hot path)
  - `state-modification-audit` (read-only verification)
  - `test-coverage-enforcer` (e2e test)