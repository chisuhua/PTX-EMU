# PTX Instruction Handlers

**Parent**: [AGENTS.md](../AGENTS.md)

## OVERVIEW
PTX instruction implementations (arithmetic, memory, control flow, barrier, etc.).

## STRUCTURE
```
src/ptxsim/instructions/
├── arithmetic.cpp    # add, sub, mul, mad, etc.
├── bitwise.cpp       # and, or, xor, shf, etc.
├── comparison.cpp    # setp, slt, sgt, etc.
├── conversion.cpp     # cvt, cvta
├── control.cpp       # bra, ret, call, exit
├── barrier.cpp       # bar.warp.sync, bar.sync
├── memory.cpp        # ld, st, atom, etc.
├── mov.cpp           # mov, shf, prmt, etc.
├── wmma.cpp          # (stub) mma, wmma
├── atomic.cpp        # (stub) atom operations
└── tensor.cpp       # (stub) tensor operations
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add instruction | `include/ptx_ir/ptx_op.def` | X-Macro entry |
| Implement handler | `src/ptxsim/instructions/` | snake_case function |
| Dispatch | `instruction_handlers.cpp` | X-Macro dispatch |
| Barrier reconvergence | `barrier.cpp` | BarWarpSyncHandler |

## KEY FILES
| File | Purpose |
|------|---------|
| `barrier.cpp` | Warp/barrier sync (critical for divergence) |
| `control.cpp` | Branch, ret, exit |
| `arithmetic.cpp` | ALU ops |

## CONVENTIONS
- Handlers use `process_<instruction>(ThreadContext*, void**, qualifiers, operand_is_immediate)`
- PTX instruction names are all lowercase (e.g., `add`, `ld.global`)
- `InstructionFactory::initialize()` registers handlers via X-Macro
- `commit_pc()` is the only normal PC advancement - never call `set_pc()` directly

## ANTI-PATTERNS
- DO NOT use `set_pc()` — use `commit_pc()` or `force_set_pc()`
- DO NOT modify `active_mask` without barrier synchronization
- DO NOT call `ThreadContext` methods from `WarpContext` without locking

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build instruction handlers
```

## KNOWN STUBS
- `wmma.cpp` — MMA/WMMA instructions not implemented
- `atomic.cpp` — Atomic operations are stubs
- `tensor.cpp` — Tensor operations not implemented

## KNOWN ISSUES

### BUG-RETHANG: ret handler must mark ALL lanes exited (FIXED 2026-06)
`RetHandler::processOperation` must mark the entire warp as exited, not just the
executing lane. A divergent warp that reaches `ret` has many lanes stalled on
different paths; only the active lane was getting `state=EXIT`, so
`ThreadContext::is_exited()` (`state == EXIT`) was false for the rest and
`WarpContext::is_finished()` never returned true.

**Rule**: Any instruction handler that semantically ends the kernel (ret, exit)
must update BOTH `warp_state.threads[i]` fields AND `ThreadContext::state` for
all 32 lanes, then call `update_active_mask()`.

### BUG-POSTBARRIER-TWOHALVES: barrier handler must OR arrived_mask (FIXED 2026-06)
When a divergent warp hits a barrier in two halves at different times, the
second release would overwrite `active_mask` with only the second half, losing
lanes released by the first. Fix: at both barrier completion sites, call
`set_active_mask(get_active_mask() | arrived_mask)` instead of
`set_active_mask(arrived_mask)`.

**Rule**: Handler functions that set `active_mask` from partial-warp data must
OR with the existing mask, not overwrite. This is because other lanes may
have been released by a prior handler call (e.g., the force_reconvergence path
re-initializes a fresh wbar for each arriving half).

**Do NOT fix `set_active_mask` semantics globally** to be additive — the
ret handler relies on overwrite semantics (`set_active_mask(0u)` to clear).
The OR logic must live in the CALLER.

### SCOPE-OF-EFFECT PRINCIPLE
Instruction handlers that affect warp-level state (ret, barrier, branch
reconvergence) must consider ALL lanes, not just the executing one. The
scheduler's `update_active_mask()` will self-heal `active_mask[]` from
`warp_state.threads[i].is_active`, but handler logic that reads `active_mask`
mid-instruction may see stale state. Pattern: after modifying per-thread state,
call `update_active_mask()` to reconcile before any scheduler-visible call.