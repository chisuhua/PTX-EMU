# PTX-SIM Execution Engine

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
PTX simulation execution engine - dispatches instructions, manages GPU/SM/CTA/Warp/Thread hierarchy.

## STRUCTURE
```
src/ptxsim/
├── core/           # GPUContext, SMContext, CTAContext, WarpContext, ThreadContext
├── barrier/        # BarrierModule + WarpBarrier + CTABarrier (unified barrier state, 2026-06)
├── instructions/    # PTX instruction handlers (arithmetic, memory, control flow)
├── debug/          # Debug utilities, tracer
└── utils/          # Helper utilities
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Instruction dispatch | `instruction_handlers.cpp` | InstructionFactory dispatches |
| Thread execution | `ThreadContext::execute_thread_instruction()` | Per-thread PTX execution |
| Warp scheduling | `WarpContext` | 32-thread warp management |
| Memory operations | `src/memory/` | SimpleMemory, SharedMemoryManager |
| Register bank | `src/register/` | RegisterBankManager |
| **Barrier state** | **`src/ptxsim/barrier/`** | **`BarrierModule` + `WarpBarrier` + `CTABarrier`; owned by `CTAContext`** |

## KEY FILES
| File | Purpose |
|------|---------|
| `core/GPUContext.cpp` | Top-level GPU simulation |
| `core/ThreadContext.cpp` | Per-thread register/PC/execution |
| `barrier/barrier_module.cpp` | **Unified barrier state machine** (per-CTA, called from `bar.cpp` handler) |
| `barrier/warp_barrier.cpp` | Per-warp barrier state (State enum, init/arrive/is_complete) |
| `barrier/cta_barrier.cpp` | Per-CTA barrier state (mutex + arrived thread set) |
| `instructions/` | PTX instruction implementations |

## CONVENTIONS (this dir)
- Instruction handlers use snake_case (e.g., `process_add`)
- X-Macro dispatch via `ptx_op.def`
- ThreadContext holds per-thread state
- **Barrier handlers** (`BarHandler`, `BarWarpSyncHandler`) MUST route through `BarrierModule` API — never directly manipulate `warp_state.wbars[]` or `SMContext::synchronize_barrier` (legacy path)

## ANTI-PATTERNS
- DO NOT call ThreadContext methods from WarpContext without proper locking
- DO NOT modify active_mask directly without barrier synchronization
- **DO NOT call `set_active_mask` to OR with arrived_mask globally** — OR logic must live in `BarrierModule::release_warp_barrier` (caller layer). The ret handler relies on overwrite semantics (`set_active_mask(0u)` to clear).
- **DO NOT add new uses of `Wbar` struct** (`include/ptxsim/wbar.h`) — it is `[[deprecated]]`. Use `BarrierModule` + `WarpBarrier`.

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build execution engine
```
