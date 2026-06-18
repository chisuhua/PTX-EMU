# Capability: warp-barrier-unification

Warp 级屏障从旧 `Wbar` 结构体迁移到新 `WarpBarrier` 类——所有 warp barrier 操作（包括 `bar.warp.sync` handler、force_reconvergence 路径、release 路径）必须通过 `BarrierModule::init/arrive/release_warp_barrier` API。

## ADDED Requirements

### Requirement: BarWarpSyncHandler MUST use BarrierModule::arrive_at_warp_barrier

The system MUST route `bar.warp.sync` execution through `BarrierModule::arrive_at_warp_barrier` rather than directly manipulating `warp_state.wbars[wbar_id]`.

#### Scenario: Normal path uses arrive_at_warp_barrier
- **WHEN** a lane executes `bar.warp.sync mask, reconv_pc` with single PC (no divergence)
- **THEN** `BarWarpSyncHandler::processOperation` MUST call `warp_ctx->get_cta_context()->get_barrier_module().arrive_at_warp_barrier(0, lane_id)`
- **AND** MUST NOT call `wbar.arrive(lane_id)` directly on `warp_state.wbars[]`

#### Scenario: force_reconvergence path uses BarrierModule init
- **WHEN** the warp is divergent at the barrier PC (`get_unique_pcs().size() > 1`)
- **THEN** the handler MUST call `barrier_module.init_warp_barrier(0, participation_mask, reconv_pc, barrier_pc)`
- **AND** then call `arrive_at_warp_barrier(0, lane_id)`

#### Scenario: bar.warp.sync uses only barrier slot 0
- **WHEN** any PTX `bar.warp.sync mask, reconv_pc` is executed
- **THEN** the handler MUST pass `warp_barrier_id=0` to `arrive_at_warp_barrier`
- **AND** MUST NOT pass any other `warp_barrier_id` (PTX ISA defines a single barrier per warp)
- **AND** if `warp_barrier_id != 0` is observed, emit `PTX_ERROR_EMU` and treat as no-op

### Requirement: warp_state MUST NOT contain Wbar fields

After migration, `WarpState` MUST NOT contain `std::array<Wbar, 4> wbars` or `int current_wbar_id`. All warp barrier state MUST live in `BarrierModule::warp_barriers_`.

#### Scenario: No wbar fields in warp_state.h
- **WHEN** migration is complete
- **THEN** `warp_state.h` MUST NOT contain `wbars` or `current_wbar_id` members (verified by `grep -n "Wbar\|wbar\|current_wbar_id" include/ptxsim/warp_state.h`)

#### Scenario: No direct wbar access from handler
- **WHEN** integration is complete
- **THEN** `barrier.cpp` MUST NOT reference `warp_state.wbars` or `wbar.arrrive/init/reset` directly (verified by `grep -n "warp_state.wbars\|wbar\." src/ptxsim/instructions/barrier.cpp`)

### Requirement: include/ptxsim/wbar.h MUST be deleted

The legacy `Wbar` struct MUST be entirely removed.

#### Scenario: Header file removed
- **WHEN** migration is complete
- **THEN** the file `include/ptxsim/wbar.h` MUST NOT exist
- **AND** `#include "ptxsim/wbar.h"` MUST NOT appear in any source file

### Requirement: BarrierModule::release_warp_barrier MUST handle BUG-POSTBARRIER-TWOHALVES

The release path MUST OR `arrived_mask` with existing `active_mask` (preserving lanes already released by a prior barrier call), matching the behavior fixed in commit `09de279`.

#### Scenario: Two-half release preserves lanes
- **WHEN** two divergent halves of a warp arrive at the same barrier in different cycles
- **THEN** after both halves are released, `warp_ctx->get_active_mask()` MUST contain all 32 lanes (or all originally-active lanes)
- **AND** MUST NOT have been overwritten by the second half's `arrived_mask` alone

### Requirement: force_reconvergence preserved as no-op design

The current behavior where `force_reconvergence_at_barrier` is intentionally empty (no-op) MUST be preserved—the caller is responsible for setting `is_blocked=true` immediately after.

#### Scenario: Empty method body preserved
- **WHEN** a code reviewer inspects `WarpContext::force_reconvergence_at_barrier`
- **THEN** the method body MUST remain comment-only (no `advance_thread_pc` calls)
- **AND** the design rationale comment MUST be updated to reference the new `BarrierModule` ownership
