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
| Barrier sync | `sm_context.cpp` | `synchronize_barrier()` |
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
- `synchronize_barrier()` may not update `active_mask` correctly after barrier release
- `test_post_barrier_divergence.cpp` documents this as a known bug (BUG-REPRODUCTION test)

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build core simulation
```