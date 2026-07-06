## ADDED Requirements

### Requirement: IInstructionHandler::ExecPipe accepts InstructionPipeline parameter

The base `IInstructionHandler::ExecPipe` method SHALL accept a second pointer parameter of type `InstructionPipeline*` in addition to the existing `ThreadContext*` parameter:

```cpp
virtual EXE_STATE ExecPipe(ThreadContext* thread, InstructionPipeline* pipeline,
                           void** args, std::vector<Qualifier> qualifier,
                           std::vector<OperandContext>& operand,
                           const std::vector<Qualifier>* stmt_qualifier);
```

All 40+ instruction handler implementations SHALL update their signatures to match. The X-Macro dispatcher in `instruction_handlers.cpp` SHALL pass through both pointers.

#### Scenario: All handlers updated to accept pipeline pointer
- **WHEN** the X-Macro dispatcher calls `handler->ExecPipe(thread, ...)`
- **THEN** the call SHALL be updated to `handler->ExecPipe(thread, pipeline, ...)` where `pipeline` is either `thread->get_instruction_pipeline()` (Phase 3.2+) or `nullptr` (Phase 3.1 - Phase 3.2 transition)

#### Scenario: Phase 3.1 transition uses nullptr
- **WHEN** Phase 3.1 is complete but Phase 3.2 has not yet begun
- **THEN** the dispatcher SHALL pass `nullptr` for the pipeline parameter, and handlers SHALL NOT call any pipeline methods (they continue to use only `ThreadContext` methods)

### Requirement: InstructionPipeline owns per-instruction execution state

The `InstructionPipeline` class SHALL own the per-instruction execution state previously held on `ThreadContext`:

- `std::vector<void*> operand_collected`
- `std::vector<char> operand_is_immediate_`
- `std::vector<std::vector<void*>> vecOp_phy_addrs`
- `std::string dst_operand_reg_name_`

The `call_stack` SHALL remain on `ThreadContext` (control-flow-spanning state, not per-instruction).

#### Scenario: Handler accesses operand buffer through pipeline
- **WHEN** an instruction handler reads `context->operand_collected[i]`
- **THEN** after Phase 3.2, handlers SHALL access `pipeline->operand_collected[i]` via the new `pipeline` parameter

#### Scenario: dst_operand_reg_name moves to pipeline
- **WHEN** `SetpHandler` reads `context->get_dst_operand_reg_name()`
- **THEN** `ThreadContext::get_dst_operand_reg_name()` SHALL forward to `pipeline->dst_operand_reg_name()`

### Requirement: InstructionPipeline provides control-flow methods

The `InstructionPipeline` class SHALL provide the following methods (currently on `ThreadContext`):

- `void _execute_once()` — single instruction execution
- `EXE_STATE execute_thread_instruction()` — public API entry
- `void collect_operands(stmt, operands, qualifier)` — collect operand addresses from registers
- `void commit_operand(stmt, operand, qualifier)` — write back result
- `void clear_temporaries()` — debug/cleanup helper
- `bool isIMMorVEC(OperandContext&)` — IMM or VEC operand check

`ThreadContext` SHALL retain delegation wrappers as inline forwarders.

#### Scenario: _execute_once preserves already_blocked guard
- **WHEN** `_execute_once()` calls `sync_to_warp_state()` via pipeline
- **THEN** `sync_to_warp_state` SHALL be invoked on the same `SimtPcManager` (via pipeline → thread delegation) to preserve the already_blocked guard invariant (lessons-learned §1)

#### Scenario: collect_operands preserves operand buffer semantics
- **WHEN** `collect_operands(stmt, operands, qualifier)` is called
- **THEN** `pipeline->operand_collected[i]` SHALL be set to `operands[i].operand_phy_addr` for each operand (preserving pre-refactor behavior, including the per-ThreadContext push-must-pair-with-pop buffer)

### Requirement: InstructionPipeline owns ThreadContext reference

The `InstructionPipeline` class SHALL hold a non-owning pointer to its owning `ThreadContext` for:

- Accessing `ThreadContext::simt_pc_mgr_` (for barrier sync during `_execute_once`)
- Delegating cross-subsystem queries (e.g., `thread->reset()` coordinating pipeline reset)

`ThreadContext` SHALL hold `std::unique_ptr<InstructionPipeline> instruction_pipeline_` for ownership.

#### Scenario: Pipeline notifies thread on barrier state changes
- **WHEN** `_execute_once()` calls `pipeline->sync_to_warp_state()`
- **THEN** `InstructionPipeline` SHALL forward to `thread->simt_pc_mgr_->sync_to_warp_state()` via its owner reference
