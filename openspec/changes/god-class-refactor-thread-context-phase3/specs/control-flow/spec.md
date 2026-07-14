## ADDED Requirements

### Requirement: Handler `ExecPipe` signature is stable

The base `IInstructionHandler::ExecPipe` SHALL remain:

```cpp
virtual void ExecPipe(ThreadContext *context, StatementContext &stmt) = 0;
```

(verified from `include/ptxsim/instruction_base.h:21`).

This change does **NOT** add a second `InstructionPipeline*` parameter to `ExecPipe`. Handler signatures are **NOT** modified in any phase. The X-Macro dispatcher in `src/ptxsim/instruction_handlers.cpp` SHALL continue to call `handler->ExecPipe(context, statement)` without changes.

#### Scenario: All 40+ concrete handler signatures remain unchanged

- **WHEN** any concrete handler (e.g. `SetpHandler`, `LdHandler`, `AddHandler`, `BraHandler`) is inspected
- **THEN** its `ExecPipe` override SHALL match the base signature `void ExecPipe(ThreadContext*, StatementContext&)` — no additional parameters, no `void** args` / `std::vector<Qualifier>` / `std::vector<OperandContext>&` parameters (these do not exist in the real signature)

#### Scenario: X-Macro dispatcher is unchanged

- **WHEN** `instruction_handlers.cpp` dispatches a statement via `handler->ExecPipe(context, statement)`
- **THEN** the call site SHALL NOT be modified by any sub-step of Phase 3.2

### Requirement: `ThreadContext` exposes operand-buffer accessors

`ThreadContext` SHALL expose two accessor pairs for operand buffers, with the following semantics across Phase 3.2 sub-steps:

```cpp
std::vector<void*> &get_operand_collected();
const std::vector<void*> &get_operand_collected() const;
std::vector<char> &get_operand_is_immediate();
const std::vector<char> &get_operand_is_immediate() const;
```

- After sub-step 3.2.0: accessors return `ThreadContext`'s own private fields `operand_collected_` and `operand_is_immediate_`
- After sub-step 3.2.3: `operand_collected_` and `operand_is_immediate_` are removed from `ThreadContext`; accessors forward to `instruction_pipeline_->get_operand_collected()` and `instruction_pipeline_->get_operand_is_immediate()`

The accessor return type is `std::vector<...>&` (not `const&` for the non-const overload) so that pipeline handlers which currently write through `&(context->operand_collected[0])` can continue to do so through the accessor.

#### Scenario: Sub-step 3.2.0 accessors return `ThreadContext` fields

- **WHEN** sub-step 3.2.0 is committed
- **THEN** `ThreadContext::get_operand_collected()` SHALL return a reference to the same buffer that was previously accessed via `context->operand_collected` (line 151-152 of `thread_context.h`)

#### Scenario: Sub-step 3.2.3 accessors forward to pipeline

- **WHEN** sub-step 3.2.3 is committed
- **THEN** `ThreadContext::get_operand_collected()` SHALL forward to `instruction_pipeline_->get_operand_collected()`
- **AND** the buffer SHALL be allocated in `InstructionPipeline::InstructionPipeline()` with `MAX_OPERANDS_PER_INSTR` (4) slots, matching the current allocation in `ThreadContext::init()` lines 51-52

### Requirement: PipelineHandler base classes read operand buffers via accessors

After sub-step 3.2.1, the four direct reads of `context->operand_collected[...]` and `context->operand_is_immediate_` in `src/ptxsim/instruction_base.cpp` SHALL go through the new accessors:

- Line 172-173 (`GenericPipelineHandler::executeOperation`): `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`; `&context->operand_is_immediate_` → `&context->get_operand_is_immediate()`
- Line 200 (`AtomicPipelineHandler::executeOperation`): `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
- Line 231 (`Tcgen05PipelineHandler::executeOperation`): `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`

This is the **only** change in sub-step 3.2.1. Handler behavior is byte-identical.

#### Scenario: GenericPipelineHandler::executeOperation uses accessor

- **WHEN** a generic instruction (e.g. `add`) is dispatched
- **THEN** `GenericPipelineHandler::executeOperation` SHALL call `processOperation(context, &(context->get_operand_collected()[0]), qualifiers, &context->get_operand_is_immediate())`
- **AND** `processOperation` SHALL receive a pointer to the same buffer that was passed before sub-step 3.2.1 (no behavioral change)

#### Scenario: Tcgen05PipelineHandler::executeOperation uses accessor

- **WHEN** a tcgen05 instruction is dispatched
- **THEN** `Tcgen05PipelineHandler::executeOperation` SHALL call `processTcgen05Operation(context, &(context->get_operand_collected()[0]), qualifiers, instr)`
- **AND** `processTcgen05Operation` SHALL receive a pointer to the same buffer that was passed before sub-step 3.2.1 (no behavioral change, including the zero-operand skip at line 217-219)

### Requirement: BarWarpSyncHandler reads operand_is_immediate via accessor

After sub-step 3.2.2, the direct read in `src/ptxsim/instructions/barrier.cpp:92-93` SHALL go through the accessor:

- `&(context->operand_collected[0])` → `&(context->get_operand_collected()[0])`
- `&context->operand_is_immediate_` → `&context->get_operand_is_immediate()`

#### Scenario: BarWarpSyncHandler::processOperation uses accessor

- **WHEN** a `bar.warp.sync` instruction is dispatched
- **THEN** `BarWarpSyncHandler::processOperation` SHALL pass `&(context->get_operand_collected()[0])` and `&context->get_operand_is_immediate()` to the base `PipelineHandler::processOperation`
- **AND** the barrier release via `BarrierModule::release_warp_barrier` SHALL be invoked with the same arguments as before (Decision 7 of `archive/2026-07-03-migrate-bar-warp-sync-to-barrier-module/design.md`)

### Requirement: `InstructionPipeline` owns per-instruction execution state

The `InstructionPipeline` class SHALL own the per-instruction execution state previously held on `ThreadContext`:

- `std::vector<void*> operand_collected_` (private)
- `std::vector<char> operand_is_immediate_` (private)
- `std::vector<std::vector<void*>> vecOp_phy_addrs_` (private)
- `ThreadContext *thread_` (private, non-owning pointer to owner)

The `call_stack` SHALL remain on `ThreadContext` (control-flow-spanning state, not per-instruction). `dst_operand_reg_name_` is **NOT** part of this change (verified absent from current `thread_context.h` by grep 2026-07-14).

The `InstructionPipeline` class SHALL provide the following public methods (currently on `ThreadContext`):

- `void _execute_once()` — single instruction execution
- `void execute_thread_instruction()` — public API entry
- `void collect_operands(StatementContext&, const std::vector<OperandContext>&, const std::vector<Qualifier>*)`
- `void commit_operand(StatementContext&, const OperandContext&, const std::vector<Qualifier>&)`
- `void clear_temporaries()`
- `bool isIMMorVEC(OperandContext&)`
- `void dump_state(std::ostream&) const`
- `void prepare_breakpoint_context(std::unordered_map<std::string, std::any>&)`
- `void trace_status(...)` template
- `void print_instruction_status(StatementContext&)`
- `std::vector<void*>& get_operand_collected()`
- `std::vector<char>& get_operand_is_immediate()`

`ThreadContext` SHALL retain all of the above methods as inline forwarders (one-line: `return instruction_pipeline_->method(...);`) — zero external API breakage.

#### Scenario: Handler accesses operand buffer through accessor (no direct `operand_collected`)

- **WHEN** any code path (after Phase 3.2) reads `context->operand_collected` directly
- **THEN** the access SHALL go through `context->get_operand_collected()`
- **AND** the field `operand_collected` SHALL be `private` on `ThreadContext` (or removed entirely after sub-step 3.2.3)

#### Scenario: collect_operands preserves push-must-pair-with-pop semantics

- **WHEN** `collect_operands(stmt, operands, qualifier)` is called for a V4 (vec4) instruction
- **THEN** `vecOp_phy_addrs_` SHALL have exactly **one** new entry pushed (per the BUGFIX comment at `thread_context.cpp:63-66` documenting the move from `std::queue` to per-`ThreadContext` stack semantics)
- **AND** the buffer SHALL be popped by the matching `releaseAllOperands` call in `PipelineHandler::commitResults` (or its subclasses)

#### Scenario: Multi-IMM operands populate operand_is_immediate_ independently

- **WHEN** an instruction has 3 operands where operands[0] and operands[2] are immediate and operand[1] is a register
- **THEN** `operand_is_immediate_[0]` and `operand_is_immediate_[2]` SHALL be non-zero
- **AND** `operand_is_immediate_[1]` SHALL be zero
- **AND** `operand_collected[0]` and `operand_collected[2]` SHALL point to immediate values
- **AND** `operand_collected[1]` SHALL point to the register's host address

#### Scenario: `collect_operands` populates buffers before `ExecPipe` runs

- **WHEN** `PipelineHandler::ExecPipe` calls `prepareOperands` (which calls `collect_operands`)
- **THEN** the buffers SHALL be populated before `processOperation` is called by `executeOperation`
- **AND** `processOperation` SHALL observe the populated buffers via the accessor

### Requirement: `InstructionPipeline` preserves PC lifecycle invariant

`_execute_once()` SHALL preserve the `set_next_pc(current_pc + 1)` → `handler->ExecPipe(this, statement)` → `commit_pc()` sequence byte-identically (`AGENTS.md` §CONVENTIONS, `docs/adr/0003-commit-pc-pattern.md`). After migration to `InstructionPipeline`, the only change is the **location** of the call (`pipeline_->_execute_once()` instead of `this->_execute_once()`); the **sequence, arguments, and line numbers** SHALL match.

#### Scenario: PC advances by 1 for non-branch non-barrier instructions

- **WHEN** `_execute_once()` is called and the dispatched instruction is not a branch or barrier
- **THEN** `get_pc()` after `_execute_once()` returns SHALL equal `pc_before + 1`
- **AND** `get_next_pc()` after `_execute_once()` returns SHALL equal `pc_before + 1`
- **AND** `set_next_pc(current_pc + 1)` is called BEFORE `handler->ExecPipe(this, statement)` (matches the pre-refactor line ordering at `thread_context.cpp:121-149`)

#### Scenario: Branch instructions override next_pc

- **WHEN** a branch instruction is dispatched and `set_pc(target_pc)` or `set_next_pc(target_pc)` is called inside `ExecPipe`
- **THEN** `commit_pc()` SHALL commit the overridden value (not the default `current_pc + 1`)
- **AND** `get_pc()` after `_execute_once()` returns SHALL equal the branch target

#### Scenario: Barrier completion sets pc via set_pc, not commit_pc path

- **WHEN** a barrier instruction completes and `simt_pc_mgr_->set_pc(reconvergence_pc)` is called
- **THEN** the new pc SHALL be the reconvergence PC (matches the BARRIER COMPLETION convention in `AGENTS.md` §CONVENTIONS)

### Requirement: `InstructionPipeline` holds non-owning `ThreadContext*`

`InstructionPipeline` SHALL hold a non-owning `ThreadContext* thread_` pointer set in the constructor (or via a `set_thread()` setter) for the following purposes:

- Accessing `ThreadContext::simt_pc_mgr_` (for barrier state queries during `_execute_once`)
- Delegating cross-subsystem queries (e.g., `thread_->reset()` coordinating pipeline reset)
- Accessing `ThreadContext::acquire_register()` (used by `MemoryAccessor::get_memory_addr` via `thread_->acquire_register`)

`ThreadContext` SHALL hold `std::unique_ptr<InstructionPipeline> instruction_pipeline_` for ownership. The `unique_ptr` is created in `ThreadContext::init()` AFTER `warp_id_` and `lane_id_` are computed (per the MR-5 comment at `thread_context.cpp:63-65`).

#### Scenario: Pipeline queries thread for simt state

- **WHEN** `_execute_once()` needs to read the current PC
- **THEN** `InstructionPipeline` SHALL call `thread_->get_pc()` (via the `simt_pc_mgr_` accessor)
- **AND** SHALL NOT have its own copy of `simt_pc_mgr_`

#### Scenario: Pipeline constructed after warp_id_/lane_id_

- **WHEN** `ThreadContext::init()` is called
- **THEN** `instruction_pipeline_` SHALL be constructed AFTER `warp_id_` and `lane_id_` are set (lines 57-58 of `thread_context.cpp`)
- **AND** the constructor SHALL receive `this` as the `thread_` argument
