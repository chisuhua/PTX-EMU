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

### DUAL STATE MECHANISM (critical invariant)
`WarpContext` maintains **TWO parallel state representations** for lane activity:

| State | Type | Read by | Written by |
|-------|------|---------|------------|
| `active_mask[]` (bool[32]) | scheduler mask | `is_lane_active()` → scheduler | `set_active_mask()`, `update_active_mask()` |
| `warp_state.threads[i].is_active` | per-thread | `update_active_mask()` source | `set_active_mask()`, `update_active_mask()` |
| `warp_state.exec_mask` (uint32_t) | activemask instr | PTX `activemask` | `set_exec_mask()` |

**Key invariant**: `update_active_mask()` at the END of every `execute_warp_instruction()` recomputes `active_mask[]` from `warp_state.threads[i].is_active`. This means **bugs that temporarily corrupt `active_mask[]` (e.g., BUG-POSTBARRIER-TWOHALVES where `set_active_mask(arrived_mask)` overwrites) are SELF-HEALED by the next instruction**.

**Consequence**: End-to-end integration tests may PASS even with subtle bugs, because the self-healing masks the issue at the warp-finish level. **Unit tests that directly assert `get_active_mask()` after a specific call catch these bugs; integration tests that only check `is_finished()` do NOT.**

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