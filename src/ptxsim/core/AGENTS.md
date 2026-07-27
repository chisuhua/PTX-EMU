# PTX-SIM Core Contexts

**Parent**: [AGENTS.md](../AGENTS.md)

## OVERVIEW
GPU simulation hierarchy: GPU → SM → CTA → Warp → Thread contexts.

## EXECUTION HIERARCHY
```
GPUContext
  └── SMContext (warp scheduler, barriers)
        └── CTAContext (warps, shared memory)
              └── WarpContext (32 threads, divergence)
                    └── ThreadContext (registers, PC)
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Per-thread PC management | `simt_pc_manager.cpp` | `get_pc()`, `commit_pc()`, `set_pc()` — Phase 1 extraction |
| Register access & lookup | `register_access_layer.cpp` | `acquire_register()`, `register_bank_manager_` — Phase 2 extraction |
| Warp scheduling | `warp_scheduler.cpp` | `is_warp_ready_to_fetch()` |
| Barrier sync | `cta_context.cpp` | `cta_context->get_barrier_module()` |
| Thread execution | `thread_context.cpp` | `execute_thread_instruction()` — delegates to sub-classes |

## KEY FILES
| File | Purpose |
|------|---------|
| `gpu_context.cpp` | Top-level: creates SMs, manages memory |
| `sm_context.cpp` | Warp scheduler, barrier management |
| `warp_context.cpp` | 32-thread warp, SIMT stack, divergence |
| `thread_context.cpp` | **Per-thread orchestrator (delegation hub)** |
| `simt_pc_manager.cpp` | **PC + execution state (Phase 1 extract)** |
| `register_access_layer.cpp` | **Register lookup + bank manager (Phase 2 extract)** |
| `cta_context.cpp` | CTA-level shared memory |
| `warp_scheduler.cpp` | RoundRobin/Greedy warp selection |

## CONVENTIONS
- PC access via `get_pc()` / `get_next_pc()` — read from warp_state directly
- Normal PC advancement: `set_next_pc(pc+1)` → execute → `commit_pc()`
- Barrier completion: `set_pc(reconvergence_pc)` — writes both pc and next_pc
- `is_warp_ready_to_fetch()` — warp selected only if all active threads have `pc == next_pc`

## ANTI-PATTERNS
- DO NOT call `ThreadContext` methods from `WarpContext` without proper locking
- DO NOT modify `active_mask` directly without barrier synchronization
- DO NOT use `force_set_pc()` — use `set_pc()` for init/sync/reset, `commit_pc()` for normal advancement
- `.bak` files in this directory are committed artifacts — do not edit

### WarpContext sub-module layout (refactor-warp-context C-18, 2026-07)

The 558-line `warp_context.cpp` has been split into focused sub-modules via
the helper namespace pattern. Each helper namespace is friend-declared in
`WarpContext` for direct member access (avoids per-instruction hot-path overhead).

| Sub-module | Responsibility | Public helper API |
|------------|----------------|-------------------|
| `warp_context.cpp` | Parent file: handle_branch, check_and_block_at_reconvergence_point, add_thread, get_lanes_by_pc, is_finished, is_warp_ready_to_fetch, reset, force_reconvergence_at_barrier, decrement_blocked_cycles, set_blocked_cycles_for_active, public API wrappers |
| `warp_context_active_mask.{h,cpp}` | `warp_active_mask::` — set_active_mask (lane + u32), update_active_mask, get_active_mask_u32 | Active mask management with T2-1 overwrite semantics preserved |
| `warp_context_simt.{h,cpp}` | `warp_simt::` — check_reconvergence | SIMT stack pop/reconvergence orchestration |
| `warp_context_dispatch.{h,cpp}` | `warp_dispatch::` — execute_warp_instruction | Per-warp instruction dispatch (124 lines extracted) |

**Friend declarations** in `WarpContext` (include/ptxsim/warp_context.h) grant the
helper namespaces direct access to private members (`active_mask[]`, `warp_state`,
`simt_stack`, `cta_context_`, `sm_context_`) without per-access getter overhead.
This is critical for `update_active_mask()` which is called at the end of every
`execute_warp_instruction()`.

**API freeze invariant**: WarpContext public API (used by sm_context.cpp:379/:461/:468/:583/:590) is unchanged. The 5+ call sites compile without modification (`sm_context.cpp zero diff` verified).

### SMContext sub-module layout (god-class-refactor-sm-context C-2, 2026-07)

The 965-line `sm_context.cpp` has been partially split into focused sub-modules
via the helper namespace pattern. Each helper namespace is friend-declared in
`WarpContext` for direct member access.

| Sub-module | Responsibility |
|------------|----------------|
| `sm_context.cpp` | Parent (862 lines vs 965 baseline, -10.7%). Contains: init, add_block, try_admit_pending_blocks, **exe_once (226 lines, monolithic)**, update_state, cleanup_finished_blocks, is_idle, resource stats, print functions |
| `sm_context_reconvergence.{h,cpp}` | `sm_reconvergence::` — drain_simt_and_update_active (dedup of :455-490 / :580-623) |
| `sm_context_cpptlm_inject.{h,cpp}` | `sm_cpptlm_inject::` — step_b_set_blocked_cycles (ADR-0020 injection) |

**Friend declarations** in `WarpContext` (include/ptxsim/warp_context.h) grant
`sm_reconvergence::` direct access to private `simt_stack` / `warp_state` / `update_active_mask()` without per-access getter overhead.

**API freeze invariant**: sm_context.cpp:379 `update_active_mask()` (lessons-learned §1) is preserved.

**Future work**: Reaching the <250 line target requires restructuring `exe_once()` (226 lines) — the per-cycle main loop is highly cohesive and not safely extractable in a single session. Future change should target exe_once() decomposition with explicit ownership transfer for scheduler / barrier / warp_lifecycle.

## KNOWN ISSUES

### SINGLE SOURCE OF TRUTH (T2-1, 2026-06)

Lane-activity state has **ONE authoritative source** + two derived caches:

| Field | Role | Authority | Notes |
|-------|------|-----------|-------|
| `warp_state.threads[i].is_schedulable()` | **AUTHORITATIVE** | `thread_state.h:54-59` | `is_active && !is_exited && !is_blocked && (status == Active) && blocked_cycles_remaining == 0` |
| `active_mask[]` (bool[32]) | derived cache | `warp_context.cpp:316-329` `update_active_mask()` | Recomputed at end of every `execute_warp_instruction()` |
| `active_count` (int) | derived counter | `warp_context.cpp:331` `set_active_mask(int,bool)` | Maintained by `set_active_mask()` for fast `is_finished()` check |
| `warp_state.exec_mask` (uint32_t) | **INDEPENDENT** | PTX `activemask` instruction | NOT unified with `is_active` — these are semantically different concepts |

**Key invariants (T2-1 contract):**
1. **`is_lane_active(lane_id)` delegates to `is_lane_schedulable(lane_id)`** — reads `warp_state` directly, no cache lag (ISSUE-005 fix). All scheduler/gate decisions see warp_state mutations immediately.
2. **`update_active_mask()` is bidirectional**: reads from `warp_state.threads[i]` AND writes back `is_active` after computing the `active` bool. This keeps `active_mask[]` and `warp_state.is_active` synchronized even when callers mutate warp_state directly (barrier release, etc.).
3. **`sync_to_warp_state(RUN)` sets `is_active = true`** — barrier release correctly marks threads active in warp_state before `is_lane_active()` reads.
4. **`set_active_mask(uint32_t mask)` is overwrite semantics** — ret handler at `src/ptxsim/instructions/call.cpp:29` uses `set_active_mask(0u)` to clear all lanes after `ret`. **DO NOT change to OR-merge** — OR logic is encapsulated in `BarrierModule::release_warp_barrier()` (`src/ptxsim/barrier/barrier_module.cpp`), now the single owner of `arrived_mask | existing_active_mask` semantics. Caller (`BarWarpSyncHandler`) just invokes release; the ret handler's overwrite semantics stay intact.

**Exception: `warp_state.exec_mask`** is PTX `activemask` instruction's independent source. It is not unified with `is_active` — these are semantically different concepts (exec_mask is set by PTX code, is_active is set by barrier/exit/divergence machinery). See `get_exec_mask()` / `set_exec_mask()`.

**Why the dual-source pattern was removed (T2-1):** Before T2-1, `is_lane_active()` read from `active_mask[]` which lagged warp_state mutations until the next `update_active_mask()` cycle. This caused subtle bugs (e.g., BUG-POSTBARRIER-TWOHALVES where post-barrier lanes were missed because `is_lane_active()` saw stale `active_mask[]`). The T2-1 delegation makes `is_lane_active()` immediate.

### BUG-RETHANG (FIXED 2026-06)
`RetHandler::processOperation` only set `state=EXIT` for the currently executing lane. Inactive lanes (stalled on divergent paths) kept `state != EXIT`, so `WarpContext::is_finished()` returned false and the scheduler looped forever.

**Fix**: Mark ALL lanes in the warp as exited (`is_exited=true`, `is_active=false`, `status=Exited`, `ThreadContext::state=EXIT`) and call `update_active_mask()`.

**Critical detail**: `ThreadContext::is_exited()` checks `state == EXIT`, NOT `warp_state.threads[i].is_exited`. Both must be set.

Regression test: `tests/unit/exec/test_ret_handler_divergent.cpp` (3 cases, 141 assertions).

### BUG-POSTBARRIER-TWOHALVES (FIXED 2026-06, refined 2026-07)
When a divergent warp splits into two halves that hit the same `bar.warp.sync` at different times, the second barrier release overwrote `active_mask` with only the currently arrived half, losing lanes already released by the first release.

**Original fix (2026-06, BarrierModule pre-migration)**: OR `arrived_mask` with existing `active_mask` BEFORE calling `set_active_mask` at both barrier completion sites in `barrier.cpp`.

**Refined fix (2026-07, commit `0e311566`)**: OR logic migrated INTO `BarrierModule::release_warp_barrier()` — single owner of `arrived_mask | existing_active_mask` semantics. `BarWarpSyncHandler::processOperation` calls `release_warp_barrier(wbar_id, warp_ctx)` and the OR is encapsulated. Handler no longer needs to know the OR pattern.

**Do NOT fix in `set_active_mask` itself** — changing its semantics globally breaks the ret handler which uses `set_active_mask(0u)` to explicitly clear. The fix lives in `BarrierModule::release_warp_barrier` — single owner, single test point.

Regression tests: `tests/unit/barrier/test_post_barrier_two_halves.cpp` (unit) + `tests/integration/divergence/test_post_barrier_two_halves.cpp` (smoke).

### HISTORICAL
// Original synchronize_barrier() had a known issue where it did not call update_active_mask() correctly
// after barrier release. The fix was committed in f033312 (lessons-learned §1 cross-module state translation).
// The synchronize_barrier method itself was removed in commit 7914764
// (cleanup-deprecated-barrier-apis, 2026-06-20); CTA-level barriers now go through BarrierModule.
// See: tests/integration/divergence/test_post_barrier_divergence.cpp (preserved as regression coverage)

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build core simulation
```