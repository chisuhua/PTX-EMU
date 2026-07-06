## Context

Phase 3 of the `god-class-refactor-thread-context` effort, following Phase 1+2 (archived at `archive/2026-07-06-god-class-refactor-thread-context/`). `ThreadContext` is now 471 lines after `SimtPcManager` (PC + state) and `RegisterAccessLayer` (register lookup) extractions. The remaining ~280 lines mix memory address resolution, control flow orchestration, and operand collection.

### Current state of `ThreadContext` (471 lines)

| 子系统 | 关键方法 | 行数估 | 耦合点 |
|--------|---------|--------|--------|
| **内存访问** | `get_memory_addr()` (260 lines), `acquire_operand()`, `initialize_shared_memory()`, `set_local_memory_space()`, `mov()`, `mov_data()` | ~280 行 | `shared_mem_space` (public), `local_mem_space` (public), `name2Sym`/`name2Share`/`cta_context_`/`SHMEMADDR` |
| **控制流/执行编排** | `_execute_once()` (~50 lines), `execute_thread_instruction()`, `collect_operands()`, `commit_operand()`, `init()`, `reset()`, `clear_temporaries()`, `isIMMorVEC()`, `dump_state()`, `prepare_breakpoint_context()`, `trace_status()`, `print_instruction_status()` | ~190 行 | `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`, `dst_operand_reg_name_`, `call_stack` |
| **Phase 1+2 委托层** | All PC/state/register methods as inline forwarders | ~50 行 | 无 |

Total: ~520 lines (some methods are counted in their own category above)

### Why Phase 3.1 + 3.2 Were Cancelled

**Phase 3.1 MemoryAccessor** (attempted 2026-07-06):
- `MemoryAccessor` extracted with its own `shared_mem_space_` and `local_mem_space_`
- `ThreadContext::get_memory_addr()` delegated via `std::function` callback
- 9 test regressions caused by state divergence: external code (CTAContext, etc.) sets `ThreadContext::shared_mem_space` directly, but `MemoryAccessor::shared_mem_space_` was never synced
- Root cause: `shared_mem_space` and `local_mem_space` are **public data members** on `ThreadContext`, settable from anywhere

**Phase 3.2 InstructionPipeline** (never attempted, analyzed):
- `_execute_once()` calls `handler->ExecPipe(this, statement)` where `this` is `ThreadContext*`
- Extracting `_execute_once()` to `InstructionPipeline` would require `this` to be `InstructionPipeline*`
- This breaks all 40+ handler signatures across `src/ptxsim/instructions/`
- Direct mechanical change but invasive

### Constraints

- **Zero external API breakage**: instruction handler `ExecPipe` signature change is the ONLY external-facing change
- **Per-thread state ownership clear**: `MemoryAccessor` owns shared/local memory; `InstructionPipeline` owns operand buffers; `ThreadContext` owns coordination only
- **Phase must be independently revert-able** (lessons-learned §3)
- **Prerequisites isolated**: public→private member conversion is a separate commit, handler signature change is a separate commit

## Goals / Non-Goals

**Goals:**
- Phase 3.1: Extract `MemoryAccessor` after converting public members to private with setters
- Phase 3.2: Change handler `ExecPipe` signature, then extract `InstructionPipeline`
- Phase 3.3: Delete legacy PODs, add ADR-0017
- Each Phase independent commit + independent verification
- Final `thread_context.cpp` ≤ 250 lines

**Non-Goals:**
- No new functionality (pure structural refactor)
- No new instructions or features
- No churn of handler semantics (only signature mechanical change)
- No aggressive optimization of extraction

## Decisions

### Decision 1: Phase 3.1 Prerequisite — Convert Public Memory Members to Private

**选择**: Before extracting `MemoryAccessor`, convert `shared_mem_space`, `local_mem_space`, and `name2Sym`/`name2Share`/`cta_context_` from public data members to private members with public setter methods. The setters call into `MemoryAccessor` to keep state synchronized.

**理由**: The Phase 3.1 cancellation's root cause was public members being settable from external code without MemoryAccessor awareness. The precondition eliminates this race.

**替代方案**:
- Use `std::shared_ptr<void>` for memory spaces — introduces reference counting overhead
- Use `std::reference_wrapper` — requires all existing direct accesses to be migrated
- **Decision**: Convert to private with public setters. Simple, matches the delegation pattern used in Phase 1+2.

### Decision 2: Phase 3.2 Handler Signature — Add `InstructionPipeline*` Alongside `ThreadContext*`

**选择**: Change `IInstructionHandler::ExecPipe(ThreadContext*, ...)` to `IInstructionHandler::ExecPipe(ThreadContext*, InstructionPipeline*, ...)`. Update all 40+ handlers to pass through the pipeline pointer. The pipeline stays default-constructed on ThreadContext.

**理由**: Keeps backward-compat with `ThreadContext*` (handlers already use it), adds the pipeline pointer for delegated control flow operations.

**替代方案**:
- Introduce new `IExecutionContext` interface — replaces `ThreadContext*` everywhere, biggest change
- Use `dynamic_cast` on `this` — fragile, adds runtime cost
- **Decision**: Add parameter. Mechanical 40-file change, low risk.

### Decision 3: Phase 3.2 InstructionPipeline Holds Operand Buffers, Not Call Stack

**选择**: `InstructionPipeline` owns `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`, `dst_operand_reg_name_`. `call_stack` stays on `ThreadContext` (it's a control flow semantic, not an execution buffer).

**理由**: Operand buffers are per-execution-step state. `call_stack` spans the entire kernel lifetime and is needed for barrier/sync coordination on `ThreadContext`.

### Decision 4: Phase 3.1 MemoryAccessor Owns Symbol Table References

**选择**: `MemoryAccessor` stores non-owning pointers to `name2Sym`, `name2Share`, and `cta_context_`. Setter methods update them whenever `ThreadContext::init()` or external code rebinds them.

**理由**: Symbol tables live in `CTAContext` and survive across kernels; `MemoryAccessor` is a pure resolution layer.

## Risks / Trade-offs

| Risk | 概率 | 缓解措施 |
|------|------|---------|
| Public→private member conversion breaks external callers | 中 | grep `shared_mem_space =` / `local_mem_space =` in `src/` first; update all direct assignments to use setter |
| Handler signature change breaks handler invocation in X-Macro dispatch | 中 | After all 40+ handlers updated, run full test suite. If regression, revert specific handler update |
| MemoryAccessor shared memory ownership causes nullptr deref if not initialized | 低 | Make MemoryAccessor robust to nullptr shared_mem_space — already does (fallback in get_memory_addr) |
| Phase 3.3 POD deletion breaks old debugging tools | 低 | grep `exec_state_` readers first; delete only if zero readers (Phase 1 verified zero readers in src/include/tests) |
| InstructionPipeline operand buffer lifetime confusion | 中 | Document Phase 3.2 commit message with state diagram; add unit test type-1 |

## Migration Plan

### Phase 3.0 Prerequisites (~1h)

1. Convert `shared_mem_space`, `local_mem_space`, `name2Sym`, `name2Share`, `cta_context_` from public to private
2. Add public setter methods on `ThreadContext` that delegate to `MemoryAccessor`
3. Verify all external direct assignments migrated to setters

### Phase 3.1 MemoryAccessor Extraction (~3h)

1. Create `include/ptxsim/memory_accessor.h` + `.cpp` (~200 lines)
2. Migrate `get_memory_addr()`, `mov_data()`, `mov()`, `initialize_shared_memory()`, `set_local_memory_space()`
3. Migrate `shared_mem_space`, `local_mem_space`, `name2Sym`, `name2Share`, `cta_context_`, `SHMEMADDR` to MemoryAccessor (private members)
4. ThreadContext retains delegation wrappers
5. **Independent commit + full test verification**

### Phase 3.2 InstructionPipeline Extraction (~4h)

1. Sub-step 3.2.0 (precondition): Change `IInstructionHandler::ExecPipe` signature in `instruction_handlers.h` — add `InstructionPipeline*` parameter (must be nullptr initially since pipeline doesn't exist yet)
2. Sub-step 3.2.1: Update all 40+ handler signatures — mechanical pass-through
3. Sub-step 3.2.2: Update X-Macro dispatch in `instruction_handlers.cpp` to pass `nullptr` for pipeline
4. Sub-step 3.2.3: Verify tests still pass after signature change
5. Sub-step 3.2.4: Create `InstructionPipeline` class
6. Sub-step 3.2.5: Migrate `_execute_once()`, `execute_thread_instruction()`, `collect_operands()`, etc.
7. Update X-Macro dispatch to pass pipeline pointer
8. **Each sub-step has its own commit**

### Phase 3.3 Cleanup + Documentation (~1h)

1. Delete `exec_state_`, `reg_pred_`, `memory_`, `program_ref_` PODs from `ThreadContext`
2. Add `docs/adr/0017-pc-management-extraction.md` recording 3-Phase history
3. Update `src/ptxsim/core/AGENTS.md` for new classes
4. Final verification + commit

### Rollback Strategy

Each Phase (3.0, 3.1, 3.2.0-7) has its own commit and can be reverted individually:
```bash
git revert <phase-commit>
```
Revert cascade order: 3.3 → 3.2 (last sub-step first) → 3.1 → 3.0.

## Open Questions

- Q1: Should `MemoryAccessor::get_memory_addr` take a `std::function` callback or a `IInstructionHandler*` + virtual method? Callback is simpler for current architecture but virtual dispatch is more testable.
- Q2: After Phase 3.2, should `ThreadContext` still expose `operand_collected` (it's read by some handlers) or migrate those reads to `pipeline->operand_collected`? Suggests batching into a follow-up cleanup change.
- Q3: Should `execute_thread_instruction()` move to `InstructionPipeline` entirely, or stay on `ThreadContext` as a delegation entry point that calls `pipeline->execute()`? The latter keeps `ThreadContext` API stable but bloats it slightly.

## Metis + Oracle Pre-Implementation Review (July 2026)

**Predecessor reviews**:
- Metis session `ses_0ca442a15ffeTGnLE1vdxlOw45` (Phase 1) — 5 MUST-RESOLVE all resolved in Phase 1+2
- Oracle session `ses_0ca36539effe2432w8IEjv4OQS` (Phase 1) — 3 Oracle gaps all resolved in Phase 1+2

**This phase-specific review pending**: Phase 3's handler signature change (Q2 + Q3) is fundamentally new and warrants a fresh Metis review before implementation begins. Recommend calling Metis after Phase 3.0 prerequisites are committed but before Phase 3.1 begins.
