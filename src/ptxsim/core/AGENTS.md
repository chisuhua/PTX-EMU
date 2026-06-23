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
| Per-thread PC management | `thread_context.cpp` | `get_pc()`, `commit_pc()`, `force_set_pc()` |
| Warp scheduling | `warp_scheduler.cpp` | `is_warp_ready_to_fetch()` |
| Barrier sync | `cta_context.cpp` | `cta_context->get_barrier_module()` |
| Thread execution | `thread_context.cpp` | `execute_thread_instruction()` |

## KEY FILES
| File | Purpose |
|------|---------|
| `gpu_context.cpp` | Top-level: creates SMs, manages memory |
| `sm_context.cpp` | Warp scheduler, barrier management |
| `warp_context.cpp` | 32-thread warp, SIMT stack, divergence |
| `thread_context.cpp` | Per-thread registers, PC, condition codes |
| `cta_context.cpp` | CTA-level shared memory |
| `warp_scheduler.cpp` | RoundRobin/Greedy warp selection |

## CONVENTIONS
- PC access via `get_pc()` / `get_next_pc()` — read from warp_state directly
- Normal PC advancement: `set_next_pc(pc+1)` → execute → `commit_pc()`
- Barrier completion: `force_set_pc(reconvergence_pc)` + `set_next_pc(reconvergence_pc)`
- `is_warp_ready_to_fetch()` — warp selected only if all active threads have `pc == next_pc`

## ANTI-PATTERNS
- DO NOT call `ThreadContext` methods from `WarpContext` without proper locking
- DO NOT modify `active_mask` directly without barrier synchronization
- DO NOT use `set_pc()` — use `commit_pc()` or `force_set_pc()`
- `.bak` files in this directory are committed artifacts — do not edit

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
4. **`set_active_mask(uint32_t mask)` is overwrite semantics** — ret handler at `src/ptxsim/instructions/call.cpp:29` uses `set_active_mask(0u)` to clear all lanes after `ret`. **DO NOT change to OR-merge** — barrier handlers in `src/ptxsim/instructions/barrier.cpp` already OR externally before calling (`warp_ctx->set_active_mask(get_active_mask() | arrived_mask)`).

**Exception: `warp_state.exec_mask`** is PTX `activemask` instruction's independent source. It is not unified with `is_active` — these are semantically different concepts (exec_mask is set by PTX code, is_active is set by barrier/exit/divergence machinery). See `get_exec_mask()` / `set_exec_mask()`.

**Why the dual-source pattern was removed (T2-1):** Before T2-1, `is_lane_active()` read from `active_mask[]` which lagged warp_state mutations until the next `update_active_mask()` cycle. This caused subtle bugs (e.g., BUG-POSTBARRIER-TWOHALVES where post-barrier lanes were missed because `is_lane_active()` saw stale `active_mask[]`). The T2-1 delegation makes `is_lane_active()` immediate.

### BUG-RETHANG (FIXED 2026-06)
`RetHandler::processOperation` only set `state=EXIT` for the currently executing lane. Inactive lanes (stalled on divergent paths) kept `state != EXIT`, so `WarpContext::is_finished()` returned false and the scheduler looped forever.

**Fix**: Mark ALL lanes in the warp as exited (`is_exited=true`, `is_active=false`, `status=Exited`, `ThreadContext::state=EXIT`) and call `update_active_mask()`.

**Critical detail**: `ThreadContext::is_exited()` checks `state == EXIT`, NOT `warp_state.threads[i].is_exited`. Both must be set.

Regression test: `tests/unit/exec/test_ret_handler_divergent.cpp` (3 cases, 141 assertions).

### BUG-POSTBARRIER-TWOHALVES (FIXED 2026-06)
When a divergent warp splits into two halves that hit the same `bar.warp.sync` at different times, the second barrier release overwrote `active_mask` with only the currently arrived half, losing lanes already released by the first release.

**Fix**: In `barrier.cpp`, at both barrier completion sites, OR the new `arrived_mask` with the existing `active_mask` BEFORE calling `set_active_mask`:
```cpp
warp_ctx->set_active_mask(
    warp_ctx->get_active_mask() | arrived_mask);
```

**Do NOT fix in `set_active_mask` itself** — changing its semantics globally breaks the ret handler which uses `set_active_mask(0u)` to explicitly clear. The fix must be in the CALLER.

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