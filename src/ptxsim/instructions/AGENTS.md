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