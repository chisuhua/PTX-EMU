## Why

Phase 1+2 of `god-class-refactor-thread-context` (archived at `archive/2026-07-06-god-class-refactor-thread-context/`) extracted `SimtPcManager` (PC + execution state) and `RegisterAccessLayer` (register lookup) from `ThreadContext`. The current `ThreadContext` (verified 2026-07-14) is **727 lines** in `src/ptxsim/core/thread_context.cpp` and **324 lines** in `include/ptxsim/thread_context.h`, ctest reports **198 tests**. Phase 3.1 (MemoryAccessor) and Phase 3.2 (InstructionPipeline) were cancelled during implementation due to deep coupling with `ThreadContext`'s public state.

This change picks up the remaining responsibilities that still mix memory address resolution, control flow orchestration, and operand collection into one class. Two technical blockers from the previous phase must be resolved first:

1. `shared_mem_space`, `local_mem_space`, `name2Sym`, `name2Share`, and `cta_context_` are **public data members** on `ThreadContext` (`include/ptxsim/thread_context.h:40-77`). External code in `src/ptxsim/core/cta_context.cpp:320` directly assigns `thread->shared_mem_space = shared_mem_space`, with `name2Sym`/`name2Share` being assigned at `cta_context.cpp:89` and `cta_context.cpp:224`. The previous MemoryAccessor extracted its own copy and diverged, causing 9 test regressions. **Required precondition (Phase 3.0)**: convert these to private members with public setters that update both `ThreadContext`'s state and any future `MemoryAccessor` field.
2. The previous Phase 3.2 attempt proposed adding a second `InstructionPipeline*` parameter to `IInstructionHandler::ExecPipe`. The current real signature (`include/ptxsim/instruction_base.h:21`) is:
   ```cpp
   virtual void ExecPipe(ThreadContext *context, StatementContext &stmt) = 0;
   ```
   Changing it would touch every concrete handler (40+ signature call sites), even though only 4 base classes (`PipelineHandler`, `GenericPipelineHandler`, `AtomicPipelineHandler`, `Tcgen05PipelineHandler`) actually read operand buffers. **Required precondition (Phase 3.2 — REDESIGNED)**: keep the handler signature stable; instead, expose operand buffers via `ThreadContext` accessors that delegate to `InstructionPipeline` (the **accessor approach**). `PipelineHandler` base classes are migrated first, in isolation, behind a single accessor pair.

## What Changes

- **Phase 3.0 (~1h)**: Convert memory-related public members to private with public setters
  - `shared_mem_space` (line 74) → `private void *shared_mem_space_` with `set_shared_memory_space(void*)` and `get_shared_memory_space() const`
  - `local_mem_space` (line 77) → `private void *local_mem_space_` with `set_local_memory_space(void*)` and `get_local_memory_space() const`
  - `name2Sym` (line 40) and `name2Share` (line 41) → move access through setters/getters; `init()` keeps the parameter signature but stores into private fields
  - `cta_context_` → private with `set_cta_context()` setter
  - Migrate all external direct assignments (audit list below) to setter calls

- **Phase 3.1 (~3h)**: Extract `MemoryAccessor` class
  - File scope: `get_memory_addr()`, `mov_data()`, `mov()`, `initialize_shared_memory()`, `set_local_memory_space()`, plus the new `shared_mem_space_`/`local_mem_space_`/`name2Sym`/`name2Share`/`cta_context_` storage
  - New files: `include/ptxsim/core/memory_accessor.h` + `src/ptxsim/core/memory_accessor.cpp`
  - `MemoryAccessor::SHMEMADDR` is **a class static member** (NOT a file-static global): see [MemoryAccessor#SHMEMADDR ownership](#shmemaddr-ownership) in design.md
  - `ThreadContext` retains delegation wrappers as inline forwarders (zero external API breakage)
  - **`std::function` callback is REJECTED** for `get_memory_addr`; the register lookup is a virtual call on the `MemoryAccessor` that internally calls `thread_->acquire_register()` (Decision 5 in design.md)

- **Phase 3.2 (~4h, REDESIGNED)**: Extract `InstructionPipeline` via the **accessor approach** — no handler signature change
  - Sub-step 3.2.0: Add `ThreadContext::get_operand_collected()` and `get_operand_is_immediate()` accessors that return references to internal buffers
  - Sub-step 3.2.1: Migrate the 4 `PipelineHandler` base classes (`PipelineHandler::ExecPipe`, `GenericPipelineHandler`, `AtomicPipelineHandler`, `Tcgen05PipelineHandler`) in `src/ptxsim/instruction_base.cpp` to call the accessors instead of `context->operand_collected[0]` directly. Verify all 198 tests pass with this change alone.
  - Sub-step 3.2.2: Add `BarWarpSyncHandler::processOperation` access via accessor (it reads `operand_is_immediate_` at `src/ptxsim/instructions/barrier.cpp:92-93`).
  - Sub-step 3.2.3: Create `InstructionPipeline` class. `ThreadContext` holds `std::unique_ptr<InstructionPipeline> instruction_pipeline_`. Operand buffers (`operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`) move into `InstructionPipeline` as `private` members; `ThreadContext` accessors forward to `pipeline_->...`.
  - Sub-step 3.2.4: Migrate `_execute_once()`, `execute_thread_instruction()`, `collect_operands()`, `commit_operand()`, `init()`, `reset()`, `clear_temporaries()`, `isIMMorVEC()`, `dump_state()`, `prepare_breakpoint_context()`, `trace_status()`, `print_instruction_status()` into `InstructionPipeline`. `ThreadContext` retains inline forwarders.
  - **PC lifecycle invariant preserved** (lessons-learned §1, AGENTS.md §CONVENTIONS): the `set_next_pc(current_pc + 1)` → `handler->ExecPipe(this, statement)` → `commit_pc()` sequence stays byte-identical; only the location of the call moves.
  - `call_stack` stays on `ThreadContext` (Decision 3).
  - `dst_operand_reg_name_` is **NOT part of this change**: it is not present in current `thread_context.h` (verified by grep) and is therefore out of scope.

- **Phase 3.3 (~1h)**: Delete legacy PODs (`exec_state_`, `reg_pred_`, `memory_`, `program_ref_`) and add ADR-0019
  - POD deletion is gated on Phase 1+2's "zero readers" grep verification, plus a fresh grep of the post-Phase-3.2 tree
  - The 4 PODs are still being back-filled in `init()` (`thread_context.cpp:79-91`) and `reset()` (`thread_context.cpp:225-226` per Metis audit); Phase 3.3 first stops the back-fill, verifies no readers, then deletes the fields
  - Add `docs/adr/0019-pc-management-extraction.md` recording 3-Phase decision history (0017 is missing from `docs/adr/`, 0018 is taken by `tcgen05-cta-group-restriction`; 0019 is the next free number)

- **Final target**: `thread_context.cpp` ≤ 300 lines (revised from 250 — current code is 727, with Phase 1+2 delegations adding ~200 more lines than the prior estimate accounted for). After Phase 3.3, `thread_context.cpp` will be a pure delegation hub.

## Capabilities

### New Capabilities

- `memory-access-extraction`: Phase 3.1 `MemoryAccessor` class — encapsulates per-thread memory address resolution and data movement, owns shared/local memory spaces and symbol table references. **New class — requires type-1 unit tests per `AGENTS.md`**.
- `control-flow-extraction`: Phase 3.2 `InstructionPipeline` class — encapsulates per-instruction execution orchestration and operand collection, owns operand buffers (`operand_collected`, `operand_is_immediate_`, `vecOp_phy_addrs`). **New class — requires type-1 unit tests**.
- `operand-buffer-accessor`: Phase 3.2 precondition — `ThreadContext` exposes `get_operand_collected()` and `get_operand_is_immediate()` as inline forwarders (initially returning internal fields, post-Phase-3.2 forwarding to `InstructionPipeline`).

### Modified Capabilities

<!-- No existing spec-level requirements are changing; this is a pure refactor. -->

## Impact

| 受影响组件 | 影响类型 |
|-----------|---------|
| `src/ptxsim/core/thread_context.cpp` (727 → ~300 行) | 加 ~50 行内联委托，删 400+ 行实现；新增 `MemoryAccessor` + `InstructionPipeline` 内部调用 |
| `include/ptxsim/thread_context.h` (324 → ~280 行) | 删 5 个 public 内存成员，加 5 个 setter + 2 个 accessor |
| `src/ptxsim/core/memory_accessor.{h,cpp}` (新) | Phase 3.1 提取 (~250 行) |
| `src/ptxsim/core/instruction_pipeline.{h,cpp}` (新) | Phase 3.2 提取 (~350 行) |
| `src/ptxsim/instruction_base.cpp` (4 个 PipelineHandler 基类) | Phase 3.2.1 — 通过 accessor 读取 operand buffer（不改 handler 签名） |
| `src/ptxsim/instructions/barrier.cpp` (BarWarpSyncHandler) | Phase 3.2.2 — 通过 accessor 读取 `operand_is_immediate_` |
| `src/ptxsim/core/cta_context.cpp` (lines 89, 224, 320) | Phase 3.0 — 外部直接赋值改为 setter 调用 |
| `docs/adr/0019-pc-management-extraction.md` (新) | 决策记录（替代原计划的 ADR-0017） |
| 测试 | **新增 ~6 个类型一单元测试**（MemoryAccessor 3 个 + InstructionPipeline 3 个）；198 个已有测试需保持全绿 |

## Prerequisites from Previous Phase

Before implementing Phase 3.1:
- [ ] Phase 3.0 committed: `shared_mem_space`/`local_mem_space`/`name2Sym`/`name2Share`/`cta_context_` are private on `ThreadContext`, with public setters that store into private fields
- [ ] All external direct assignments in `src/ptxsim/core/cta_context.cpp` (lines 89, 224, 320) and any others identified by grep are migrated to setter calls
- [ ] The 198 existing tests pass after Phase 3.0

Before implementing Phase 3.2:
- [ ] Sub-step 3.2.0 committed: `ThreadContext::get_operand_collected()` and `get_operand_is_immediate()` exist as inline forwarders
- [ ] Sub-step 3.2.1 committed: all 4 `PipelineHandler` base classes read through accessors; all 198 tests pass
- [ ] Sub-step 3.2.2 committed: `BarWarpSyncHandler` reads through accessor; all 198 tests pass

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性
- [ ] 列出 Phase 3.1 迁移的 set_* / commit_* 调用 (`set_shared_memory_space`, `set_local_memory_space`, `set_name2sym`, `set_name2share`, `set_cta_context`)
- [ ] 行级 diff 计划：`get_memory_addr()` + `mov_data()` + `mov()` + `initialize_shared_memory()`
- [ ] 行级 diff 计划：`_execute_once()` + `execute_thread_instruction()` + `collect_operands()` + `commit_operand()`
- [ ] **PC 生命周期保留**：`set_next_pc(current_pc+1)` → `handler->ExecPipe(this, stmt)` → `commit_pc()` 顺序与原实现字节对齐（AGENTS.md §CONVENTIONS）

### 多 Phase 推进
- [ ] Phase 拆分方案 + 独立 commit 粒度已说明（3.0、3.1、3.2.0–3.2.4、3.3 各一个 commit）
- [ ] 基线 worktree 命令已记录（绑定到 Phase 3.0 commit，不是 `HEAD~1`）
- [ ] 失败处理策略（revert 该 Phase 单独 commit，lessons-learned §3）

### Prerequisite 隔离
- [ ] Phase 3.0（public→private + setter 转换）单独 commit
- [ ] Phase 3.2.0/3.2.1/3.2.2（accessor + 基类迁移）单独 commit
- [ ] 每个 prerequisite commit 各自通过 198 个测试

### TDD 纪律
- [ ] Phase 3.1：先写 `tests/unit/core/test_memory_accessor.cpp` 3 个测试，再实现 `MemoryAccessor`（Red-Green-Refactor）
- [ ] Phase 3.2：先写 `tests/unit/core/test_instruction_pipeline.cpp` 3 个测试 + 1 个 PC 生命周期集成测试，再实现 `InstructionPipeline`

### 文档同步
- [ ] `src/ptxsim/core/AGENTS.md`: 新增 `MemoryAccessor` + `InstructionPipeline` 条目（更新 `WHERE TO LOOK` 与 `KEY FILES` 表）
- [ ] `docs/adr/0019-pc-management-extraction.md` 新建：Phase 1+2 决策 + Phase 3.1 取消教训 + Phase 3.2 accessor 方案
- [ ] `docs/adr/README.md` 追加 ADR-0019 链接
- [ ] `tasks.md` Phase 状态变更已说明

## References

- Archived predecessor: `openspec/changes/archive/2026-07-06-god-class-refactor-thread-context/`
- Predecessor commits: `9ce8e93`, `3082e80`, `bc1a0aa`, `3fa5e5f`, `a2d7ab0`
- Metis review of this proposal: session `ses_0a11eea61ffe0HTZX5uQEUvP7L` (2026-07-14) — 10 blocking findings, all addressed in this revision
- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1)
- Lessons-learned integration: §1 (cross-module state translation), §3 (multi-Phase commit), §4 (baseline worktree), §6 (artifacts-first), §7 (pre-impl Metis review)
- Project conventions: `src/ptxsim/core/AGENTS.md` §CONVENTIONS (PC lifecycle), `src/ptxsim/AGENTS.md` §CONVENTIONS (handler dispatch)
