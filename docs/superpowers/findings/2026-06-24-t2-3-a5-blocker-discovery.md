# T2-3 A5 Discovery: True Blocker is barrier.cpp Handler

> **Date**: 2026-06-24
> **Status**: Blocker identified, PoC partial, A5 reopened
> **Owner**: T2-3 god class split change

## Summary

T2-3 A5 (physical removal of deprecated `wbars[]`/`get_wbar()`) is **more complex than initially scoped**. The blocker is not test fixture refactoring — it is the **barrier handler itself** still using legacy state for its main control flow.

## What Was Done

1. **PoC commit `421eec9`** (pushed to origin):
   - Migrated `tests/integration/barrier/test_barrier_module_integrated.cpp` from `warp->get_wbar(0)` shim to direct `BarrierModule` API
   - Pattern: local `BarrierModule mod;` + `mod.init_warp_barrier()` + `mod.arrive_at_warp_barrier()` + `wbar->is_complete()`
   - Result: pure BarrierModule unit test passes, build green, sanity quick green

2. **Failed migration attempt** (reverted, not committed):
   - Tried to migrate `tests/integration/pc/test_pc_management_integrated.cpp`
   - Replaced `warp->get_wbar(0).is_complete()` with `warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)->is_complete()`
   - **Test failed** at line 78: legacy shim returned `true`, new direct call returned `false`

## Root Cause

`BarWarpSyncHandler::processOperation` (`src/ptxsim/instructions/barrier.cpp:140-280`) uses **legacy `warp_state.wbars[]` and `warp_state.current_wbar_id` for its main control flow**:

```cpp
// barrier.cpp:145, 157, 160-161, 184, 198, 215-263
if (warp_state.current_wbar_id < 0) { ... }
ptxsim::Wbar& init_wbar = warp_state.wbars[0];
if (init_wbar.is_complete() && warp_state.current_wbar_id >= 0) { ... }
warp_state.current_wbar_id = -1;
ptxsim::Wbar& wbar = warp_state.wbars[wbar_id];
```

The BarrierModule (`cta_ctx->get_barrier_module()`) is only referenced in **one sub-function** at line 358:
```cpp
BarrierModule& bm = cta_ctx->get_barrier_module();
```

The handler updates the **legacy `Wbar` struct** (via direct `wbar.arrrive(lane_id)` calls and field writes), but does NOT propagate state to `BarrierModule::warp_barriers_[N]`.

Therefore:
- `warp->get_wbar(0)` returns the **legacy struct** which is updated → `is_complete() == true` ✓
- `warp->get_cta_context()->get_barrier_module().get_warp_barrier(0)` returns the **BarrierModule** which is NOT updated → `is_complete() == false` ✗

## Revised Scope for T2-3 A5

**Original scope** (master plan): "Migrate ~60 test call sites from `get_wbar()` to `get_barrier_module().get_warp_barrier()`"

**Actual scope** (discovered 2026-06-24): Migrate **`BarWarpSyncHandler::processOperation`** itself (~140 lines) to use BarrierModule as source of truth, including:
- 14+ legacy field accesses (`warp_state.wbars[idx]`, `warp_state.current_wbar_id`)
- 2 BUG workaround paths that depend on legacy semantics:
  - **BUG-RECONVERGENCE-SIMPLEGEMM**: divergent warp hit barrier with 2 unique PCs
  - **BUG-CUTE-RMSNORM-BROADCAST-SKIP**: `current_wbar_id < 0` check after release

Then migrate test call sites. Then remove deprecated API surface.

**Estimated effort**: 3-5 days manual, exceeds autonomous agent capability.

## Recommended Next Steps

### Option 1: Reopen T2-3 A5 as separate Phase 4 work
- Document blocker in `openspec/changes/phase3-t2-3-god-class-split/tasks.md`
- Archive current change with A5 in "deferred to Phase 4" status
- Create new change `phase4-t2-3-a5-barrier-handler-migration` for the actual work

### Option 2: Accept deprecation as terminal state
- Keep `[[deprecated]]` markers indefinitely
- Document migration map in `wbar.h` (already done)
- Document this discovery as known limitation

### Option 3: Incremental migration with parallel safety
- Migrate `BarWarpSyncHandler` to **dual-write** (update both legacy `Wbar` and `BarrierModule`)
- Migrate tests one at a time
- Once all tests migrated, remove legacy path

## Migration Pattern (Validated by PoC)

For pure BarrierModule tests (like `test_barrier_module_integrated.cpp`):
```cpp
BarrierModule mod;
WarpBarrier* wbar = mod.init_warp_barrier(/*id=*/0, /*mask=*/0x0000FFFF, /*reconv_pc=*/1, /*barrier_pc=*/0);
REQUIRE(wbar->get_expected_count() == 16);
REQUIRE(!wbar->is_complete());

for (int i = 0; i < 15; i++) {
    mod.arrive_at_warp_barrier(0, i);
    REQUIRE(!wbar->is_complete());
}
mod.arrive_at_warp_barrier(0, 15);
REQUIRE(wbar->is_complete());
```

For tests using the full barrier handler (like `test_pc_management_integrated.cpp`):
- **DO NOT migrate** until `BarWarpSyncHandler` is updated to use BarrierModule
- These tests will fail if migrated prematurely

## Files Referenced

- `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` — master plan
- `openspec/changes/phase3-t2-3-god-class-split/tasks.md` — T2-3 task list
- `include/ptxsim/wbar.h` — migration map (complete)
- `include/ptxsim/barrier/warp_barrier.h` — WarpBarrier API
- `include/ptxsim/warp_state.h` — deprecated `wbars[]` (line 17-19), `current_wbar_id` (line 22-24)
- `include/ptxsim/warp_context.h:225-227` — deprecated `get_wbar()`
- `src/ptxsim/core/warp_context.cpp:510-524` — `get_wbar()` shim
- `src/ptxsim/instructions/barrier.cpp:140-280` — **TRUE BLOCKER**: legacy field uses
- `tests/integration/barrier/test_barrier_module_integrated.cpp` — PoC migrated (commit 421eec9)
- `tests/integration/pc/test_pc_management_integrated.cpp` — migration attempted, reverted