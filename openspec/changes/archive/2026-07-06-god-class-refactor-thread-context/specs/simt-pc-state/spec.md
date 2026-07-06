## ADDED Requirements

### Requirement: SimtPcManager provides per-thread PC read/write

The `SimtPcManager` class SHALL provide `get_pc()`, `set_pc(int)`, `get_next_pc()`, `set_next_pc(int)`, and `commit_pc()` methods that delegate to `WarpState.threads[lane_id]` via the injected `WarpContext*`.

`commit_pc()` SHALL set `pc = next_pc` by calling `set_pc(get_next_pc())`.

`set_pc(int)` SHALL write both `pc` and `next_pc` synchronously (used for init/reset/barrier completion).

#### Scenario: Normal PC advancement via commit_pc
- **WHEN** `_execute_once()` sets `set_next_pc(current_pc + 1)` and the instruction handler completes
- **THEN** `commit_pc()` SHALL set `pc = next_pc`, advancing the thread to the next instruction

#### Scenario: Barrier completion sets PC via set_pc
- **WHEN** barrier handler completes and calls `set_pc(reconvergence_pc)`
- **THEN** both `pc` and `next_pc` SHALL be set to `reconvergence_pc`

#### Scenario: reset() zeroes PC through SimtPcManager
- **WHEN** `ThreadContext::reset()` is called
- **THEN** `set_pc(0)` and `set_next_pc(0)` SHALL be called through `SimtPcManager`

### Requirement: SimtPcManager stores and manages execution state

The `SimtPcManager` class SHALL store `EXE_STATE state` and provide `get_state()`, `set_state(EXE_STATE)`, `is_active()`, `is_exited()`, and `is_at_barrier()` methods.

`is_active()` SHALL return `state != EXIT`.
`is_exited()` SHALL return `state == EXIT`.
`is_at_barrier()` SHALL return `state == BAR_SYNC`.

#### Scenario: Thread exits after ret instruction
- **WHEN** the ret handler sets `state = EXIT`
- **THEN** `is_exited()` SHALL return `true` and `is_active()` SHALL return `false`

#### Scenario: Thread enters barrier wait
- **WHEN** the barrier handler sets `state = BAR_SYNC`
- **THEN** `is_at_barrier()` SHALL return `true`

### Requirement: SimtPcManager syncs bidirectionally with warp_state

The `SimtPcManager` class SHALL provide `sync_from_warp_state()` and `sync_to_warp_state()` methods.

`sync_from_warp_state()` SHALL read `warp_state.threads[lane_id].status` and translate to `EXE_STATE`:
- `ThreadStatus::Active` → `RUN`
- `ThreadStatus::Blocked` → `BAR_SYNC`
- `ThreadStatus::Exited` → `EXIT`

`sync_to_warp_state()` SHALL translate `EXE_STATE` to `warp_state` fields and SHALL NOT overwrite `is_blocked=true` or `status=Blocked` if the thread is already blocked (the `already_blocked` guard).

#### Scenario: sync_to_warp_state preserves already-blocked status
- **WHEN** `sync_to_warp_state()` is called and `thread_state.is_blocked` is already `true`
- **THEN** only `next_pc` SHALL be written; `status` and `is_blocked` SHALL remain unchanged

#### Scenario: sync_to_warp_state sets RUN to Active
- **WHEN** `sync_to_warp_state()` is called with `state == RUN` and `is_blocked == false`
- **THEN** `thread_state.status` SHALL be set to `Active`, `is_blocked` to `false`, `is_active` to `true`

### Requirement: SimtPcManager validates and queries program counters

The `SimtPcManager` class SHALL provide `is_valid_pc()`, `is_valid_pc(int)`, `statements_size()`, `get_statement_at(int)`, and `get_current_statement()` methods.

These methods SHALL delegate to the `statements` vector (owned by `ThreadContext`, accessed via pointer/reference passed at construction).

#### Scenario: get_current_statement returns statement at current PC
- **WHEN** `get_current_statement()` is called on a valid PC
- **THEN** it SHALL return a pointer to `(*statements)[pc]`

#### Scenario: is_valid_pc rejects invalid index
- **WHEN** `is_valid_pc()` is called and PC is outside `[0, statements->size())`
- **THEN** it SHALL return `false`

### Requirement: ThreadContext delegates PC and state to SimtPcManager transparently

`ThreadContext` SHALL retain all existing public method signatures (`get_pc()`, `set_pc()`, `commit_pc()`, `get_next_pc()`, `set_next_pc()`, `get_state()`, `set_state()`, `is_active()`, `is_exited()`, `is_at_barrier()`, `is_valid_pc()`, `is_valid_pc(int)`, `statements_size()`, `get_statement_at(int)`, `get_current_statement()`, `sync_from_warp_state()`, `sync_to_warp_state()`).

Each method SHALL forward to the corresponding `SimtPcManager` method via `simt_pc_mgr_->method_name(args)`.

No instruction handler or caller outside `ThreadContext` SHALL require modification.

#### Scenario: handler calls get_pc() through ThreadContext unchanged
- **WHEN** any instruction handler calls `context->get_pc()`
- **THEN** `ThreadContext::get_pc()` SHALL return the value from `simt_pc_mgr_->get_pc()`, functionally identical to pre-refactor behavior

#### Scenario: sync_from_warp_state called from WarpContext dispatch unchanged
- **WHEN** `WarpContext::execute_warp_instruction()` calls `thread->sync_from_warp_state()` before each instruction
- **THEN** `ThreadContext::sync_from_warp_state()` SHALL forward to `simt_pc_mgr_->sync_from_warp_state()`, preserving the exact same warp_state → EXE_STATE translation

### Requirement: SimtPcManager does NOT manage call_stack or bar_id

The `SimtPcManager` class SHALL NOT own, read, or modify `call_stack` or `bar_id`. Both remain on `ThreadContext` until Phase 3.

`SimtPcManager` has no dependency on `call_stack` or `bar_id` — the `sync_to_warp_state()` / `sync_from_warp_state()` methods operate solely on `warp_state.threads[lane_id]` via the injected `WarpContext*`.

#### Scenario: SimtPcManager ignores call_stack during PC management
- **WHEN** `SimtPcManager::commit_pc()` advances PC
- **THEN** `call_stack` on `ThreadContext` remains unchanged and independent

### Requirement: exec_state_ POD state field stays consistent with SimtPcManager

After Phase 1, `exec_state_.state` SHALL be set by reading back from `simt_pc_mgr_->get_state()` during `init()`. The `exec_state_.state` field SHALL NOT be independently mutated — it is a read-back cache of the authoritative `SimtPcManager::state_`.

Phase 3 SHALL remove `exec_state_.state` entirely once ThreadContext's remaining subsystems migrate to independent classes.

#### Scenario: exec_state_.state reflects SimtPcManager state after init
- **WHEN** `ThreadContext::init()` finishes
- **THEN** `exec_state_.state` SHALL equal `simt_pc_mgr_->get_state()`, both `RUN`

### Requirement: set_warp_context fans out to SimtPcManager

`ThreadContext::set_warp_context()` SHALL call `simt_pc_mgr_->set_warp_context(warp_ctx)` before updating its own `warp_context_` field.

`SimtPcManager::set_warp_context()` SHALL update its internal `warp_context_` pointer to the provided value.

#### Scenario: set_warp_context keeps SimtPcManager synchronized
- **WHEN** `ThreadContext::set_warp_context(new_ctx)` is called
- **THEN** both `ThreadContext::warp_context_` and `simt_pc_mgr_->get_warp_context()` SHALL point to `new_ctx`