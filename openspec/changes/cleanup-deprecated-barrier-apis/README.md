# cleanup-deprecated-barrier-apis

**Phase 6 partial cleanup**: Remove `BsyncManager` + `SMContext::synchronize_barrier` + SM-level global barrier state. **Preserve `Wbar` struct and `warp_state.wbars[]` field** until Phase 5 independent change (`migrate-bar-warp-sync-to-barrier-module`).

## Scope

- **Delete**: `include/ptxsim/bsync_state.h`, `src/ptxsim/core/bsync_state.cpp`, `SMContext::synchronize_barrier()`, SM-level fields (`barrier_waiting_threads`, `barrier_thread_counts`, `barrier_mutex_`), periodic barrier check (`sm_context.cpp:204-242 (仅 `barrier_mutex_` lock + `barrier_waiting_threads` 循环;lines 244-260 是 warp 调度维护,必须保留)`)
- **Replace**: `warp_context.cpp:292` BAR_SYNC fallback - migrate from `synchronize_barrier` to `BarrierModule::arrive_at_cta_barrier`
- **Update**: `barrier.cpp` - remove 3 `bsync_manager_.bsync/release` call sites; remove obsolete comments
- **Document**: Update `ADR-0008` + `AGENTS.md` (3 files) + spec/design/tasks consistency
- **Delete test**: `tests/unit/sync/test_bsync_state.cpp`

## NOT in Scope (Deferred to Phase 5)

- `include/ptxsim/wbar.h` (Wbar struct) - **PRESERVED**
- `include/ptxsim/warp_state.h` `wbars[]` + `current_wbar_id` fields - **PRESERVED**
- `BarWarpSyncHandler::processOperation` main logic - **UNCHANGED**
- 19 test files including `ptxsim/wbar.h` - **PRESERVED** (no migration needed)
- `tests/integration/divergence/test_post_barrier_divergence.cpp` known-bug regression test - **PRESERVED**

## Implementation Plan

3 commits, each independently revert-able:

1. **Commit 1**: Delete `BsyncManager` + sync `barrier.cpp` call sites + delete `test_bsync_state.cpp`
2. **Commit 2**: Delete SM-level barrier state + replace `warp_context.cpp:292` BAR_SYNC fallback
3. **Commit 3**: Sync `ADR-0008` + 3 `AGENTS.md` files + OpenSpec artifacts

## Review

Complete review report: `.opencode/notes/cleanup-barrier-review.md` (2026-06-20)

Key findings:
- Decision 1 vs Decision 3 design conflict resolved by preserving Wbar struct
- `warp_context.cpp:283-296` BAR_SYNC fallback (originally omitted from proposal) must be migrated to `BarrierModule`
- BAR_SYNC state has 2 production setters + 1 translator (not dead code as initially analyzed)
- 19 test files preserve Wbar references (no migration needed)
