# PTX-SIM Execution Engine

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
PTX simulation execution engine - dispatches instructions, manages GPU/SM/CTA/Warp/Thread hierarchy.

## STRUCTURE
```
src/ptxsim/
├── core/           # GPUContext, SMContext, CTAContext, WarpContext, ThreadContext
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

## KEY FILES
| File | Purpose |
|------|---------|
| `core/GPUContext.cpp` | Top-level GPU simulation |
| `core/ThreadContext.cpp` | Per-thread register/PC/execution |
| `instructions/` | PTX instruction implementations |

## CONVENTIONS (this dir)
- Instruction handlers use snake_case (e.g., `process_add`)
- X-Macro dispatch via `ptx_op.def`
- ThreadContext holds per-thread state

## ANTI-PATTERNS
- DO NOT call ThreadContext methods from WarpContext without proper locking
- DO NOT modify active_mask directly without barrier synchronization

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build execution engine
```
