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
├── control.cpp       # bra, ret, call, exit
├── barrier.cpp       # bar.warp.sync, bar.sync
├── memory.cpp        # ld, st, atom, etc.
├── mov.cpp           # mov, shf, prmt, etc.
├── atomic.cpp        # (stub) atom operations
├── wmma.cpp          # WmmaHandler::processWmmaOperation throws
│                      #   UnsupportedInstructionException for all wmma.*
│                      #   (per ADR-0016, tcgen05 handlers migrated to tcgen05.cpp).
├── tcgen05.cpp        # 5 processTcgen05Xxx handlers (mma/ld/st/commit/wait)
│                      #   extracted from wmma.cpp (ADR-0016 Phase 1-2).
└── cvt/              # CVT 策略模式（per ADR-0015）
   ├── cvt_strategy.{h,cpp}            # dispatcher (133 行) + ConversionStrategy 接口
   ├── cvt_float_to_float.{h,cpp}      # FloatToFloatStrategy   (f32↔f64↔f16)
   ├── cvt_float_to_int.{h,cpp}        # FloatToIntStrategy     (含 .sat/5 rounding/.ftz)
   ├── cvt_int_to_float.{h,cpp}        # IntToFloatStrategy
   ├── cvt_int_to_int.{h,cpp}          # IntToIntStrategy
   └── cvt_helpers.{h,cpp}             # 4 helper 函数 (round_half_to_even 等)

   2026-07 fix-cvt-strategy-actual-split 移除 ~920 行 GeneralCvtStrategy 死代码，
   详见 ADR-0015 §2026-07 Fix 段 + debt-audit §P0-C1 (RESOLVED)。
```

## WHERE TO LOOK
| Task | Location | Notes |
|------|----------|-------|
| Add instruction | `include/ptx_ir/ptx_op.def` | X-Macro entry |
| Implement handler | `src/ptxsim/instructions/` | snake_case function |
| Dispatch | `instruction_handlers.cpp` | X-Macro dispatch |
| Barrier reconvergence | `barrier.cpp` (dispatch) → `src/ptxsim/barrier/barrier_module.cpp` (state) | `BarHandler` + `BarWarpSyncHandler` route through `BarrierModule` API |

## KEY FILES
| File | Purpose |
|------|---------|
| `barrier.cpp` | **Dispatch entry**: `BarWarpSyncHandler::processOperation` + `BarHandler::executeBarrier`. Calls `BarrierModule` API for state. |
| `control.cpp` | Branch, ret, exit |
| `arithmetic.cpp` | ALU ops |

## CONVENTIONS
- Handlers use `process_<instruction>(ThreadContext*, void**, qualifiers, operand_is_immediate)`
- PTX instruction names are all lowercase (e.g., `add`, `ld.global`)
- `InstructionFactory::initialize()` registers handlers via X-Macro
- `commit_pc()` is the only normal PC advancement - never call `set_pc()` directly

## ANTI-PATTERNS
- DO NOT use `force_set_pc()` — use `set_pc()` for init/sync/reset, `commit_pc()` for normal advancement
- DO NOT modify `active_mask` without barrier synchronization
- DO NOT call `ThreadContext` methods from `WarpContext` without locking

## COMMANDS
```bash
cmake --build build --target ptxsim     # Build instruction handlers
```

## KNOWN STUBS
- `atomic.cpp` — Cross-warp atomicity (Phase 2 of
  `implement-atomic-cas-and-true-atomicity`) is now backed by a global
  atomic mutex (`include/ptxsim/atomic/atomic_mutex.h`). All atomic
  operations (CAS plus the existing add/and/or/xor/exch/min/max/inc/dec)
  serialize through this mutex, satisfying the "no real atomicity"
  debt-audit A-9 issue at the cross-warp level. Per-warp scheduling
  (sm_context.cpp:225-260) continues to provide intra-warp ordering.
- `wmma.cpp` (WmmaHandler) — Blackwell `tcgen05.*` real fragment arithmetic
  implemented (Phase 1-3 of `implement-wmma-tensor-core-tcgen05`).
  pre-Blackwell `wmma.*` / `mma.*` permanently throws
  `UnsupportedInstructionException` per ADR-0016.

## TCGEN05 HANDLER DISPATCH (2026-07, fix-tcgen05-handler-dispatch)

- `tcgen05.cpp` — `Tcgen05Handler::processTcgen05Operation` dispatches
  on `instr.op_kind` to the 5 per-op free functions (kept for backward
  compat with `fix-tcgen05-test-coverage-gaps` dead-code coverage test).
- 11 `S_TCGEN05_*` X-Macro entries (`ptx_op.def`) all share the single
  `Tcgen05Handler` class. `Tcgen05PipelineHandler` (3-stage pipeline:
  prepare/execute/commit, mirrors `WmmaPipelineHandler`) is the
  InstructionHandler base.
- 6 deferred op_kinds (ALLOC/DEALLOC/RELINQUISH/CP/MMA_WS/FENCE) throw
  `UnsupportedInstructionException` (per ADR-0016 §C5 fix #1) — lands
  in `implement-tcgen05-handlers-extended`.

## TCGEN05 HANDLER TEST COVERAGE (2026-07, fix-tcgen05-test-coverage-gaps)

- `tcgen05.cpp` — 5 `processTcgen05Xxx` handlers (mma/ld/st/commit/wait)
  + forward declaration header `include/ptxsim/instructions/tcgen05.h`.
- Test coverage status:
  - **5 integration parse tests** (mma/ld/st/commit/wait) — verify
    `Tcgen05Instr` IR fields via `ptxir::factory::makeTcgen05Instr`
  - **1 unit golden value test** — `tests/reference/ptx_tcgen05/tcgen05_mma_golden.h`
    (8×4 f16×f16→f32 hand-computed values, marked UNVERIFIED-AGAINST-HARDWARE)
  - **1 E2E kernel** — Priority 3 f32 fallback (ptxas 13.0 lacks sm_100
    tcgen05 support; pure CUDA kernel mirrors `test_blackwell_gemm.cu`)
- **Status**: Handlers are now **wired to dispatch** (via
  `fix-tcgen05-handler-dispatch`): 5 core op_kinds route through
  `Tcgen05Handler::processTcgen05Operation`. Direct invocation test
  promoted from dead-code coverage to real-path coverage.

## TCGEN05 ALLOC-FAMILY HANDLERS (2026-07, implement-tcgen05-handlers-extended Phase 1)

- `tcgen05_alloc.cpp` — 3 alloc-family handlers (alloc/dealloc/relinquish_alloc_permit)
  added to dispatch table in `tcgen05.cpp:574-583`. Implements per-CTA TMEM
  slot allocation via `TmemAllocator` (256-slot first-fit, `std::bitset` tracking).
- **cta_group::2** throws `UnsupportedInstructionException` referencing
  ADR-0018 (cluster abstraction deferred).
- Per-warp allocate_permit checked in `alloc`; relinquished by
  `tcgen05.relinquish_alloc_permit`; restored on `WarpState::reset()`
  (CTA teardown per PTX ISA §9.7.16).
- **Status (Phase 1.x)**: TmemAllocator read-only methods now hold
  `mu_` to prevent data races (per Oracle review 2026-07-09).
  `kSlotCount` consistency enforced via `static_assert` against `Tmem`.
  3 handler-level integration tests added
  (`tests/integration/tcgen05/test_alloc_dealloc_relinquish.cpp`,
  12 TEST_CASEs / 28 assertions).

## TCGEN05 REMAINING DEFERRED (CP/MMA_WS/FENCE)

- **3 deferred op_kinds** (CP/MMA_WS/FENCE) still throw
  `UnsupportedInstructionException` (per ADR-0016 §C5 fix #1) —
  Phase 2/3/4 in `implement-tcgen05-handlers-extended`.

## ATOMIC HANDLER (Phase 1+2, 2026-07)

Implements `atomic.compare_and_swap` (a.k.a. `atomic.cas`) plus the
9 existing atom ops, with cross-warp atomicity guarantee via the
global `ptxsim::AtomicMutex`:

```cpp
atom.global.cas.u32 dst, [addr], cmp, val;
```

- Reads `*addr` into a local variable
- If loaded value equals `cmp`, writes `val` to `*addr`
- Always writes the originally-loaded value to `dst`

Operands are packed as `[dst, addr, cmp, val]` by `ptx_visitor_atom.cpp`
(opcount=3 + visitor loop pushes the optional 4th operand). The
handler holds `global_atomic_mutex()` for the duration of every
atomic read-modify-write sequence, satisfying concurrent multi-warp
contention. The mutex is non-recursive (`std::mutex`); no public
method on the handler re-enters atomic work under the same lock,
matching the cta_barrier.cpp:47 pattern (lessons-learned §2).

Tests: `integration_ptx_atom_global_cas` (3 cases — match / mismatch /
mixed) and `integration_ptx_atom_global_cas_multiwarp` (2 cases — match
contention, all-mismatch no-op). Concurrent-warp correctness is
verified end-to-end via the multi-warp case.

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