# cleanup-deprecated-barrier-apis

**Phase 6 partial cleanup**: Remove `BsyncManager` + `SMContext::synchronize_barrier` + SM-level global barrier state. **Preserve `Wbar` struct and `warp_state.wbars[]` field** until Phase 5 independent change (`migrate-bar-warp-sync-to-barrier-module`).

## Scope

- **Delete**: `include/ptxsim/bsync_state.h`, `src/ptxsim/core/bsync_state.cpp`, `SMContext::synchronize_barrier()`, SM-level fields (`barrier_waiting_threads`, `barrier_thread_counts`, `barrier_mutex_`), periodic barrier check (`sm_context.cpp:204-242`)
- **Replace**: `warp_context.cpp:292` BAR_SYNC fallback - migrate from `synchronize_barrier` to `BarrierModule::arrive_at_cta_barrier`
- **Update**: `barrier.cpp` - remove 3 `bsync_manager_.bsync/release` call sites; remove obsolete comments
- **Document**: Update `ADR-0008` + `AGENTS.md` (3 files) + spec/design/tasks consistency
- **Delete test**: `tests/unit/sync/test_bsync_state.cpp`, `test_barrier_active_mask_preserved.cpp`, `test_barrier_scenarios.cpp`

## NOT in Scope (Deferred to Phase 5)

- `include/ptxsim/wbar.h` (Wbar struct) - **PRESERVED**
- `include/ptxsim/warp_state.h` `wbars[]` + `current_wbar_id` fields - **PRESERVED**
- `BarWarpSyncHandler::processOperation` main logic - **UNCHANGED**
- 19 test files including `ptxsim/wbar.h` - **PRESERVED** (no migration needed)
- `tests/integration/divergence/test_post_barrier_divergence.cpp` known-bug regression test - **PRESERVED**

## Implementation Plan

3 commits, each independently revert-able:

1. **Commit 1** (`8a5573d`): Delete `BsyncManager` + sync `barrier.cpp` call sites + delete 3 test files
2. **Commit 2** (`7914764`): Delete SM-level barrier state + replace `warp_context.cpp:292` BAR_SYNC fallback
3. **Commit 3** (`6ec8efd`): Sync `ADR-0008` + 3 `AGENTS.md` files + OpenSpec artifacts

## Review

Complete review report: `.opencode/notes/cleanup-barrier-review.md` (2026-06-20)

Key findings:
- Decision 1 vs Decision 3 design conflict resolved by preserving Wbar struct
- `warp_context.cpp:283-296` BAR_SYNC fallback (originally omitted from proposal) migrated to `BarrierModule`
- BAR_SYNC state has 2 production setters + 1 translator (not dead code as initially analyzed)
- 19 test files preserve Wbar references (no migration needed)

## Verification Results (post-merge on main)

- Full build (`cmake --build build -j$(nproc)`): **100% PASS** (only Wbar deprecation warnings)
- `ctest`: 3 pre-existing failures verified on `origin/main` (f033312) - **no new regressions**:
  - `unit_simt_stack_stale_entry_blocks_lane0` (lessons-learned §15)
  - `integration_cute_rmsnorm_bar_sync_pattern` (lessons-learned §15)
  - `cute_rmsnorm` (newly verified pre-existing on origin/main)
- PTX syntax tests `tests/ptx/test_all_ptx.sh`: **33/33 PASS**

## Implementation Discovered Issues (handled in-flight)

1. **`bsync_state.h` had double responsibility**: contained both `BsyncManager` (to delete) and `DivergenceExecutionMode` enum (to keep). **Resolution**: Moved `DivergenceExecutionMode` to `warp_scheduler.h`.

2. **`sm_context.cpp:200-260` line range error**: lines 244-260 were warp scheduler maintenance code (`decrement_blocked_cycles` + `update_active_mask`), NOT barrier code. **Correction**: Only delete lines 204-242.

3. **`get_barrier_module()` returns reference (not pointer)**: Use `.` not `->`.

4. **`thread->bar_id` is always 0** (BLOCKING gate 1.4): No production code sets the field; `arrive_at_cta_barrier(0, thread)` is semantically equivalent.

5. **2 test files called `synchronize_barrier`**: `test_barrier_active_mask_preserved.cpp` + `test_barrier_scenarios.cpp` deleted as part of Commit 1.
