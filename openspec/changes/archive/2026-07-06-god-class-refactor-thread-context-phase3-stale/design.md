## Context

Phase 3 of the `god-class-refactor-thread-context` effort, following Phase 1+2 (archived at `archive/2026-07-06-god-class-refactor-thread-context/`). Phase 1+2 extracted `SimtPcManager` (PC + execution state) and `RegisterAccessLayer` (register lookup) from `ThreadContext`. The current state of `ThreadContext` (verified 2026-07-14):

- `src/ptxsim/core/thread_context.cpp`: **727 lines**
- `include/ptxsim/thread_context.h`: **324 lines**
- ctest baseline: **198 tests**

The remaining responsibilities mix memory address resolution, control flow orchestration, and operand collection. Phase 3.1 + 3.2 were previously cancelled; this design documents the corrected approach based on Metis review `ses_0a11eea61ffe0HTZX5uQEUvP7L` (2026-07-14).

### Current state of `ThreadContext` (727 lines)

| 子系统 | 关键方法 | 行数估 | 耦合点 |
|--------|---------|--------|--------|
| **内存访问** | `get_memory_addr()` (~280 行 `thread_context.cpp:300-560` 估), `mov()`, `mov_data()`, `initialize_shared_memory()`, `set_local_memory_space()` | ~340 行 | `shared_mem_space` (public, `thread_context.h:74`), `local_mem_space` (public, `thread_context.h:77`), `name2Sym`/`name2Share` (public, `thread_context.h:40-42`), `cta_context_` (private, set in `init()`) |
| **控制流/执行编排** | `_execute_once()` (`thread_context.cpp:101-150`, 50 行), `execute_thread_instruction()`, `collect_operands()`, `commit_operand()`, `isIMMorVEC()`, `dump_state()`, `prepare_breakpoint_context()`, `trace_status()`, `print_instruction_status()` | ~190 行 | `operand_collected` (public, `thread_context.h:151-152`), `operand_is_immediate_` (public, `thread_context.h:167`), `vecOp_phy_addrs` (public, `thread_context.h:67`) |
| **Phase 1+2 委托层** | `get_pc()`, `set_pc()`, `commit_pc()`, `acquire_register()`, `read_reg_32()`, `trace_status()` 模板 | ~100 行 | `simt_pc_mgr_`, `reg_access_` (private) |
| **静态/全局状态** | `static uint64_t SHMEMADDR = 0` (`thread_context.cpp:25`) | 1 行 | file-static, must migrate to `MemoryAccessor` |
| **遗留 POD 回填** | `exec_state_`、`reg_pred_`、`memory_`、`program_ref_` 仍在 `init()` (lines 79-91) 和 `reset()` 中被回填 | ~30 行 | 待 Phase 3.3 审计后删除 |

Total: ~660 行 (line-by-line 算入 init/reset/dump_state 等跨类别的函数), 实测 727 行。**Phase 3 前的提案把 727 误算为 471** — 这个差异解释了为什么之前估算的"≤ 250 行"目标过于激进；本设计把目标调整为 **≤ 300 行**（保留 ~50 行内联委托、~50 行 init/reset、~50 行 trace/dump、PC/寄存器 委托方法 ~150 行）。

### Why Phase 3.1 + 3.2 Were Cancelled

**Phase 3.1 MemoryAccessor** (attempted 2026-07-06):
- `MemoryAccessor` extracted with its own `shared_mem_space_` and `local_mem_space_`
- `ThreadContext::get_memory_addr()` delegated via `std::function` callback
- 9 test regressions caused by state divergence: external code (CTAContext, etc.) sets `ThreadContext::shared_mem_space` directly (`cta_context.cpp:320`), but `MemoryAccessor::shared_mem_space_` was never synced
- Root cause: 5 memory-related fields are **public data members** on `ThreadContext` (lines 40, 41, 74, 77 of `thread_context.h`; `cta_context_` is private), settable from anywhere

**Phase 3.2 InstructionPipeline** (never attempted, analyzed — REDESIGNED in this proposal):
- Previous design proposed adding `InstructionPipeline*` as a second `ExecPipe` parameter
- Real handler base signature is `void ExecPipe(ThreadContext*, StatementContext&)` (`include/ptxsim/instruction_base.h:21`); the spec for Phase 3.2 was written for an imagined signature with `void** args` / `std::vector<Qualifier>` / `std::vector<OperandContext>&` — these parameters do not exist
- Direct operand-buffer reads only happen in **4 places**: `src/ptxsim/instruction_base.cpp:172-173` (GenericPipelineHandler), `:200` (AtomicPipelineHandler), `:231` (Tcgen05PipelineHandler), and `src/ptxsim/instructions/barrier.cpp:92-93` (BarWarpSyncHandler). All other handlers receive `void **operands` from `processOperation` and never touch `operand_collected` directly
- The accessor approach (this design) is the minimum-invasive path: keep `ExecPipe` signature stable, expose operand buffers via `ThreadContext` accessors, migrate the 4 direct readers one at a time

### Constraints

- **Zero external API breakage**: `ExecPipe(ThreadContext*, StatementContext&)` signature is **NOT changed** in any phase
- **Per-thread state ownership clear**: `MemoryAccessor` owns shared/local memory; `InstructionPipeline` owns operand buffers; `ThreadContext` owns coordination only
- **Phase must be independently revert-able** (lessons-learned §3)
- **Prerequisites isolated**: public→private member conversion is a separate commit; each accessor + base-class migration is a separate commit
- **PC lifecycle invariant preserved**: `set_next_pc(current_pc + 1)` → `handler->ExecPipe(this, statement)` → `commit_pc()` sequence is byte-identical, only location changes (AGENTS.md §CONVENTIONS, lessons-learned §1)

## Goals / Non-Goals

**Goals:**
- Phase 3.0: Convert public memory members to private with setters
- Phase 3.1: Extract `MemoryAccessor` (file scope: `get_memory_addr`, `mov_data`, `mov`, `initialize_shared_memory`, `set_local_memory_space`, `SHMEMADDR`)
- Phase 3.2: Extract `InstructionPipeline` via accessor approach — no handler signature change
- Phase 3.3: Delete legacy PODs, add ADR-0019
- Each sub-step has its own commit + independent verification
- Final `thread_context.cpp` ≤ 300 lines
- All 198 existing tests pass; 6 new type-1 unit tests added

**Non-Goals:**
- No new functionality (pure structural refactor)
- No new instructions or features
- No churn of handler semantics
- No change to `ExecPipe` signature
- No `dst_operand_reg_name_` (verified absent from current code by grep)
- No `IExecutionContext` interface extraction (out of scope)
- No aggressive optimization of extraction (preserve current behavior byte-identical)

## Decisions

### Decision 1: Phase 3.0 Prerequisite — Convert Public Memory Members to Private

**选择**: Before extracting `MemoryAccessor`, convert `shared_mem_space`, `local_mem_space`, `name2Sym`, `name2Share`, `cta_context_` from public data members to private members with public setter/getter pairs. `ThreadContext` itself does not yet own a `MemoryAccessor` — the setters store into `ThreadContext`'s own private fields. Phase 3.1 then creates `MemoryAccessor` and `ThreadContext`'s setters additionally delegate to it.

**理由**: The Phase 3.1 cancellation's root cause was public members being settable from external code without awareness of state synchronization. The precondition eliminates this race **without** introducing a new class in the same commit (per lessons-learned §3, prerequisite changes must be isolated for clean revert).

**替代方案**:
- Use `std::shared_ptr<void>` for memory spaces — introduces reference counting overhead
- Use `std::reference_wrapper` — requires all existing direct accesses to be migrated
- **Decision**: Convert to private with public setters. Minimal change, no API churn.

**外部直接赋值审计（来自 `grep` 验证 2026-07-14）**:
- `src/ptxsim/core/cta_context.cpp:89` — `name2Share[s->name] = std::move(s)` (赋值给 `name2Share` map, 经由 `CTAContext` 引用 — 不是直接给 `ThreadContext` 字段, 需进一步定位)
- `src/ptxsim/core/cta_context.cpp:224` — `name2Sym, label2pc, &name2Share, this` 作为 `init()` 参数传入
- `src/ptxsim/core/cta_context.cpp:320` — `thread->shared_mem_space = shared_mem_space` (**直接给 ThreadContext 字段赋值**, 必迁)
- Phase 3.0 必须把 line 320 改为 `thread->set_shared_memory_space(shared_mem_space)`

### Decision 2: Phase 3.2 Handler Signature — STABLE; Use Accessor Approach

**选择**: `IInstructionHandler::ExecPipe` signature is **unchanged**: `void ExecPipe(ThreadContext *context, StatementContext &stmt)`. `InstructionPipeline` is exposed through two new `ThreadContext` accessors (`get_operand_collected()` returning `std::vector<void*>&` and `get_operand_is_immediate()` returning `std::vector<char>&`). Initially the accessors return `ThreadContext`'s own private fields; after Phase 3.2.3 they forward to `instruction_pipeline_->...`. Only the 4 sites in `instruction_base.cpp` and `barrier.cpp` that currently read `context->operand_collected[...]` or `context->operand_is_immediate_[...]` are migrated.

**理由**: The previous "add `InstructionPipeline*` parameter" design would have touched every concrete handler (40+ signature call sites in `instruction_handlers.cpp` X-Macro dispatch) for **zero behavioral benefit** — only 4 sites actually read operand buffers. The accessor approach has:
- 4 caller changes (the actual readers), not 40+
- No change to `ExecPipe` virtual table layout
- No change to the X-Macro dispatch
- No risk of breaking handlers in user-defined instruction sets
- `InstructionPipeline` can still be unit-tested in isolation by constructing a `ThreadContext` and calling setters

**替代方案**:
- Add `InstructionPipeline*` parameter to `ExecPipe` (previous design) — invasive, breaks all 40+ handlers, no benefit
- Introduce `IExecutionContext` interface — replaces `ThreadContext*` everywhere, biggest change, out of scope
- Use `dynamic_cast` on `this` — fragile, adds runtime cost
- **Decision**: Accessor approach. 4-site mechanical change, zero handler signature change.

### Decision 3: Phase 3.2 InstructionPipeline Holds Operand Buffers, Not Call Stack

**选择**: `InstructionPipeline` owns `operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`. `call_stack` stays on `ThreadContext` (it spans the entire kernel lifetime and is needed for barrier/sync coordination on `ThreadContext`).

**理由**: Operand buffers are per-execution-step state (reset every `execute_thread_instruction()` call); `call_stack` is per-kernel state. The lifetime asymmetry is the deciding factor.

### Decision 4: Phase 3.1 MemoryAccessor Owns Symbol Table References

**选择**: `MemoryAccessor` stores non-owning pointers to `name2Sym`, `name2Share`, and `cta_context_`. Setter methods (`set_name2sym`, `set_name2share`, `set_cta_context`) update them whenever `ThreadContext::init()` or external code rebinds them. The `cta_context_->name2Local` access is preserved (current code reads this in `get_memory_addr`).

**理由**: Symbol tables live in `CTAContext` and survive across kernels; `MemoryAccessor` is a pure resolution layer. Owning the tables would require duplicating lifetime management.

### Decision 5: `get_memory_addr` Uses Virtual Method, Not `std::function` Callback

**选择**: `MemoryAccessor::get_memory_addr` is a regular method on `MemoryAccessor`. The register lookup is `thread_->acquire_register()` (an existing `ThreadContext` method). `MemoryAccessor` holds a non-owning `ThreadContext* thread_` pointer set in its constructor or via a `set_thread()` setter.

**理由**: The previous design's `std::function` callback (introduced during the cancelled Phase 3.1) adds type-erasure overhead and complicates testing (the lambda's capture list is not visible in the API). The current code's pattern — `MemoryAccessor` holds a `ThreadContext*` and calls `acquire_register()` directly — is the minimum-change path.

**替代方案**:
- `std::function<void*(...)> acquire_reg` parameter (cancelled design) — overhead, opaque API
- Make `MemoryAccessor` a friend of `ThreadContext` — leaks encapsulation
- **Decision**: `MemoryAccessor` holds `ThreadContext* thread_`; `get_memory_addr` calls `thread_->acquire_register()` directly. No `std::function`.

### Decision 6: SHMEMADDR Ownership

**选择**: `static uint64_t SHMEMADDR = 0;` in `thread_context.cpp:25` becomes `static uint64_t MemoryAccessor::SHMEMADDR_ = 0;` — a **class static member** in `MemoryAccessor`. Behavior preserved: all `MemoryAccessor` instances in the program share one SHMEMADDR (matching current behavior, since all threads in a CTA share the same shared memory base).

**理由**:
- File-static cannot survive in a new translation unit (`memory_accessor.cpp`) without `extern` declarations
- Per-instance member would change semantics (each `MemoryAccessor` would have its own SHMEMADDR, breaking the "one base per CTA" invariant)
- Class static is the closest match to current behavior, has clear ownership, and is testable (test can reset via a public test-only method or by setting the same value twice and checking the duplicate-detection throws)

**Linkage**:
- Declaration in `include/ptxsim/core/memory_accessor.h`: `static uint64_t SHMEMADDR_;`
- Definition in `src/ptxsim/core/memory_accessor.cpp`: `uint64_t MemoryAccessor::SHMEMADDR_ = 0;`
- `initialize_shared_memory` moves to `MemoryAccessor`; same atomicity guarantee (single-threaded SC, but the throw-on-duplicate-mismatch check stays in `MemoryAccessor`)

### Decision 7: ADR-0019 (Not ADR-0017)

**选择**: The new ADR is `docs/adr/ADR-0019-pc-management-extraction.md`. `0017` is missing from the ADR directory (jumps 0016 → 0018); `0018` is taken by `tcgen05-cta-group-restriction`. Using `0019` is the next free number.

**理由**:
- ADR numbering convention is sequential
- `0017` is missing and reserving it would create confusion
- `0019` is unambiguous and requires only an update to `docs/adr/README.md`

### Decision 8: Unit Tests for New Classes

**选择**: `MemoryAccessor` and `InstructionPipeline` are new classes and require type-1 unit tests per `AGENTS.md` "新数据结构 → 类型一单元测试必须" rule.

**单元测试计划** (file paths under `tests/unit/core/`):

- `tests/unit/core/test_memory_accessor.cpp` (3 cases):
  1. `set_shared_memory_space` / `get_shared_memory_space` round-trip (verifies private storage is reached via setter)
  2. `initialize_shared_memory` first-set succeeds; second-set with same value succeeds; second-set with different value throws `InvalidMemoryAccessException` (covers Decision 6 semantics)
  3. `get_memory_addr` with a special register (`%tid.x`) returns the correct host address via `thread_->acquire_register`; with a symbol in `name2Sym` returns the symbol's absolute address

- `tests/unit/core/test_instruction_pipeline.cpp` (3 cases):
  1. `collect_operands` sets `operand_collected[i]` to `operands[i].operand_phy_addr` and `operand_is_immediate_[i]` to the IMM flag
  2. Multi-VEC push-must-pair-with-pop: after `collect_operands` for a V4 instruction, `vecOp_phy_addrs` has exactly one entry (matches the BUGFIX comment at `thread_context.cpp:63-66` about per-ThreadContext stack semantics)
  3. `execute_thread_instruction` PC lifecycle: `get_pc()` before, `set_next_pc(current_pc+1)` then `commit_pc()` after → `get_pc()` returns `current_pc+1`. This locks the AGENTS.md §CONVENTIONS invariant in a test

- `tests/integration/pc/test_pc_lifecycle_invariant.cpp` (1 case): drives a real handler through `execute_warp_instruction` and asserts `pc` advances by exactly 1 for non-branch non-barrier instructions

## Risks / Trade-offs

| Risk | 概率 | 缓解措施 |
|------|------|---------|
| Public→private member conversion breaks external callers | 中 | grep `shared_mem_space =` / `local_mem_space =` / `name2Sym =` / `name2Share =` / `cta_context_ =` in `src/` first; update all direct assignments to use setter |
| Handler signature change breaks handler invocation in X-Macro dispatch | **N/A** | The accessor approach does **not** change `ExecPipe` signature; this risk is eliminated |
| `MemoryAccessor` shared memory ownership causes nullptr deref if not initialized | 低 | `get_memory_addr` is robust to nullptr `shared_mem_space_` — already does (fallback in current code) |
| `InstructionPipeline` operand buffer lifetime confusion | 中 | Unit test (Decision 8.2) locks push-must-pair-with-pop semantics; TDD Red phase before implementation |
| PC lifecycle drift in `_execute_once` migration | 中 | Unit test (Decision 8.3) + integration test (Decision 8.b) lock the `set_next_pc → ExecPipe → commit_pc` sequence byte-identical; line-level diff in tasks.md Phase 3.2.4 verification |
| `MemoryAccessor::SHMEMADDR_` class static duplicates a stale value across test runs | 低 | `tests/unit/core/test_memory_accessor.cpp` case 2 covers the duplicate-detection throw; `initialize_shared_memory` is the only writer |
| Phase 3.3 POD deletion breaks old debugging tools | 低 | grep `exec_state_` / `reg_pred_` / `memory_` / `program_ref_` readers in `src/`/`include/`/`tests/` after Phase 3.2; delete only if zero readers |
| Working tree has unrelated uncommitted changes | 中 | `git status` shows `AGENTS.md` modified, `openspec/changes/archive/2026-06-24-integrate-barrier-module-cta-warp/` files deleted, and untracked `openspec/changes/cleanup-deprecated-barrier-apis/` + `openspec/changes/migrate-bar-warp-sync-to-barrier-module/` + `docs/superpowers/plans/2026-06-18-integrate-barrier-module-cta-warp-fix.md` — these must be committed, reverted, or stashed before Phase 3 begins to avoid mixing refactor commits with unrelated changes (lessons-learned §3 + §6) |

## Migration Plan

### Phase 3.0 Prerequisites (~1h)

1. In `include/ptxsim/thread_context.h`:
   - `shared_mem_space` → `private void *shared_mem_space_ = nullptr;` with public `void set_shared_memory_space(void*)` and `void *get_shared_memory_space() const`
   - `local_mem_space` → `private void *local_mem_space_ = nullptr;` with public `void set_local_memory_space(void*)` and `void *get_local_memory_space() const`
   - `name2Sym` (line 40) → `private` with `set_name2sym()` / `get_name2sym()`
   - `name2Share` (line 41) → `private` with `set_name2share()` / `get_name2share()`
   - `cta_context_` → `private` with `set_cta_context()` / `get_cta_context()`
2. In `src/ptxsim/core/thread_context.cpp`:
   - `init()` stores into private fields via the new setters
   - `set_local_memory_space()` existing implementation is kept as the setter (already a setter)
3. In `src/ptxsim/core/cta_context.cpp`:
   - Line 320 `thread->shared_mem_space = shared_mem_space` → `thread->set_shared_memory_space(shared_mem_space)`
   - Lines 89, 224 (passing `name2Sym`/`name2Share` into `init()`) — verify `init()` parameter signature is unchanged; the parameter is stored via the new setters inside `init()`
4. Run all 198 tests
5. **Independent commit**: `refactor(core): make shared/local memory private on ThreadContext (Phase 3.0)`

### Phase 3.1 MemoryAccessor Extraction (~3h)

1. **TDD Red**: Write `tests/unit/core/test_memory_accessor.cpp` (3 cases per Decision 8). Verify they fail (no `MemoryAccessor` class yet).
2. Create `include/ptxsim/core/memory_accessor.h` and `src/ptxsim/core/memory_accessor.cpp` (~250 lines):
   - Members: `void *shared_mem_space_`, `void *local_mem_space_`, `std::map<...> *name2Sym_`, `std::map<...> *name2Share_`, `CTAContext *cta_context_`, `ThreadContext *thread_`, `static uint64_t SHMEMADDR_`
   - Public methods: `get_memory_addr`, `mov`, `mov_data`, `initialize_shared_memory`, `set_shared_memory_space`, `get_shared_memory_space`, `set_local_memory_space`, `get_local_memory_space`, `set_name2sym`, `set_name2share`, `set_cta_context`, `set_thread`
3. Update `src/CMakeLists.txt` and `src/ptxsim/core/CMakeLists.txt` to add `memory_accessor.cpp`
4. Migrate `thread_context.cpp` methods to delegate: `get_memory_addr`, `mov_data`, `mov`, `initialize_shared_memory`, `set_local_memory_space`. Each becomes a one-line inline forwarder (e.g., `return mem_access_->get_memory_addr(op, qualifiers);`)
5. `ThreadContext::init()` creates `mem_access_ = std::make_unique<MemoryAccessor>(this)` and forwards the new setters
6. **TDD Green**: Run unit tests + all 198 integration/e2e tests
7. **Line-level diff verification** (tasks.md Phase 3.1.4): `get_memory_addr` and `initialize_shared_memory` byte-comparison between pre-Phase-3.1 and post-Phase-3.1 — except for the function body now being a delegation
8. **Independent commit**: `refactor(core): extract MemoryAccessor from ThreadContext (Phase 3.1)`

### Phase 3.2 InstructionPipeline Extraction (~4h, REDESIGNED — accessor approach)

**Sub-step 3.2.0 — Add accessors (precondition)**

1. In `include/ptxsim/thread_context.h`, add:
   ```cpp
   std::vector<void*> &get_operand_collected();
   const std::vector<void*> &get_operand_collected() const;
   std::vector<char> &get_operand_is_immediate();
   const std::vector<char> &get_operand_is_immediate() const;
   ```
2. Initial implementation: return `operand_collected` / `operand_is_immediate_` (which are still public fields at this point).
3. **Independent commit**: `refactor(core): add operand-buffer accessors to ThreadContext (Phase 3.2.0)`
4. Verify all 198 tests pass (no behavior change)

**Sub-step 3.2.1 — Migrate 4 PipelineHandler base classes**

1. In `src/ptxsim/instruction_base.cpp`:
   - Line 172-173 (GenericPipelineHandler::executeOperation): `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
   - Line 200 (AtomicPipelineHandler::executeOperation): same change
   - Line 231 (Tcgen05PipelineHandler::executeOperation): same change
2. **Independent commit**: `refactor(core): route PipelineHandler base classes through operand accessors (Phase 3.2.1)`
3. Verify all 198 tests pass

**Sub-step 3.2.2 — Migrate BarWarpSyncHandler**

1. In `src/ptxsim/instructions/barrier.cpp:92-93`:
   - `&context->operand_is_immediate_` → `&context->get_operand_is_immediate()`
2. **Independent commit**: `refactor(core): route BarWarpSyncHandler through operand accessor (Phase 3.2.2)`
3. Verify all 198 tests pass

**Sub-step 3.2.3 — Create `InstructionPipeline` class**

1. **TDD Red**: Write `tests/unit/core/test_instruction_pipeline.cpp` (3 cases per Decision 8). Verify they fail.
2. Create `include/ptxsim/core/instruction_pipeline.h` and `src/ptxsim/core/instruction_pipeline.cpp` (~350 lines):
   - Members: `std::vector<void*> operand_collected_`, `std::vector<char> operand_is_immediate_`, `std::vector<std::vector<void*>> vecOp_phy_addrs_`, `ThreadContext *thread_`
   - Public methods: `collect_operands`, `commit_operand`, `clear_temporaries`, `isIMMorVEC`, `dump_state`, `prepare_breakpoint_context`, `trace_status`, `print_instruction_status`, plus the `execute_thread_instruction` entry point
3. Update `src/CMakeLists.txt` and `src/ptxsim/core/CMakeLists.txt` to add `instruction_pipeline.cpp`
4. In `thread_context.h`:
   - Make `operand_collected` (line 151) and `operand_is_immediate_` (line 167) `private` (still storage; now delegated)
   - The accessors from 3.2.0 now forward: `return instruction_pipeline_->get_operand_collected();`
   - Add `std::unique_ptr<InstructionPipeline> instruction_pipeline_` member
5. `ThreadContext::init()` creates `instruction_pipeline_ = std::make_unique<InstructionPipeline>(this)`
6. **TDD Green**: Run unit tests + all 198 tests
7. **Independent commit**: `refactor(core): extract InstructionPipeline from ThreadContext (Phase 3.2.3)`

**Sub-step 3.2.4 — Migrate `_execute_once` and friends**

1. Migrate `ThreadContext::_execute_once()` body to `InstructionPipeline::_execute_once()`. **The `set_next_pc(current_pc + 1)` → `handler->ExecPipe(this, statement)` → `commit_pc()` sequence stays byte-identical** (AGENTS.md §CONVENTIONS, lessons-learned §1).
2. Migrate `execute_thread_instruction`, `collect_operands`, `commit_operand`, `clear_temporaries`, `isIMMorVEC`, `dump_state`, `prepare_breakpoint_context`, `trace_status`, `print_instruction_status` similarly
3. `ThreadContext` retains all these methods as inline forwarders (one-liner: `return instruction_pipeline_->method(...);`)
4. **Line-level diff verification** (tasks.md Phase 3.2.4): `commit_pc()` call site, `set_next_pc` call site, `ExecPipe` call site — same line numbers, same arguments
5. **Independent commit**: `refactor(core): migrate control-flow methods to InstructionPipeline (Phase 3.2.4)`
6. Verify all 198 tests pass

### Phase 3.3 Cleanup + Documentation (~1h)

1. grep `exec_state_` / `reg_pred_` / `memory_` / `program_ref_` in `src/` / `include/` / `tests/` — confirm zero readers (Phase 1+2 already verified; re-verify after Phase 3.2)
2. Stop the back-fill in `init()` (lines 79-91) and `reset()` (lines 225-226)
3. Compile + test
4. Delete the 4 POD fields from `ThreadContext`
5. Create `docs/adr/ADR-0019-pc-management-extraction.md` recording 3-Phase history
6. Update `docs/adr/README.md` to add ADR-0019 link
7. Update `src/ptxsim/core/AGENTS.md` `WHERE TO LOOK` and `KEY FILES` tables for `MemoryAccessor` and `InstructionPipeline`
8. **Final verification**: `wc -l src/ptxsim/core/thread_context.cpp` ≤ 300; all 198 + 6 new tests pass
9. **Independent commit**: `refactor(core): delete legacy PODs and add ADR-0019 (Phase 3.3)`

### Rollback Strategy

Each Phase (3.0, 3.1, 3.2.0–3.2.4, 3.3) has its own commit and can be reverted individually:
```bash
git revert <phase-commit>
```
Revert cascade order: 3.3 → 3.2.4 → 3.2.3 → 3.2.2 → 3.2.1 → 3.2.0 → 3.1 → 3.0.

**纪律**: 任何已有测试回归 → 立即 revert 该 Phase，不得混入后续 commit（lessons-learned §3）。

## Open Questions

All previously-open questions are now resolved:

- **Q1** (was: `get_memory_addr` callback vs virtual method): **RESOLVED — Decision 5** uses `MemoryAccessor::get_memory_addr` virtual method; `MemoryAccessor` holds `ThreadContext* thread_` and calls `thread_->acquire_register()` directly. **No `std::function`.**
- **Q2** (was: should `ThreadContext` still expose `operand_collected`?): **RESOLVED — Decision 2** uses accessor approach. `ThreadContext::get_operand_collected()` is the single entry point; initially returns internal fields, post-Phase-3.2.3 forwards to `InstructionPipeline`. `operand_collected` becomes `private` on `ThreadContext` after Phase 3.2.3.
- **Q3** (was: should `execute_thread_instruction()` move to `InstructionPipeline`?): **RESOLVED — Decision 3**. `execute_thread_instruction()` is migrated to `InstructionPipeline` in sub-step 3.2.4. `ThreadContext` retains a one-line inline forwarder to preserve the public API.

## Metis + Oracle Pre-Implementation Review (July 2026)

**Predecessor reviews**:
- Metis session `ses_0ca442a15ffeTGnLE1vdxlOw45` (Phase 1) — 5 MUST-RESOLVE all resolved in Phase 1+2
- Oracle session `ses_0ca36539effe2432w8IEjv4OQS` (Phase 1) — 3 Oracle gaps all resolved in Phase 1+2

**This phase review**:
- Metis session `ses_0a11eea61ffe0HTZX5uQEUvP7L` (Phase 3 pre-impl, 2026-07-14) — 10 blocking findings, all addressed in this revision:
  1. ✅ `thread_context.cpp` line count corrected (471 → 727)
  2. ✅ `ExecPipe` signature corrected in spec/control-flow/spec.md
  3. ✅ SetpHandler `get_dst_operand_reg_name()` fabricated scenario removed
  4. ✅ Phase 3.2 redesigned from signature-change to accessor approach
  5. ✅ MemoryAccessor + InstructionPipeline unit tests added (Decision 8)
  6. ✅ ADR-0017 → ADR-0019 (Decision 7)
  7. ✅ Baseline worktree command corrected (tasks.md tied to Phase 3.0 commit, not `HEAD~1`)
  8. ✅ SHMEMADDR ownership clarified (Decision 6: class static)
  9. ✅ Q1/Q2/Q3 resolved (this section)
  10. ✅ TDD Red phase added to Phase 3.1 and 3.2.3

Recommend fresh Oracle session for the `_execute_once` migration in Phase 3.2.4 (PC lifecycle is the highest-risk area; Oracle can review the byte-level diff before commit).
