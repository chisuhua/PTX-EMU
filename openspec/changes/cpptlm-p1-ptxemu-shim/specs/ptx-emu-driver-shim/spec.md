## ADDED Requirements

### Requirement: PtxEmuDriverShim implements IPtxEmuDriver

The system SHALL provide `PtxEmuDriverShim` class that implements the CppTLM-side `IPtxEmuDriver` pure virtual interface (from `include/tlm/gpu/ptx_emu_driver.hh`).

- SHALL be located at `src/cudart/cpptlm_bridge/PtxEmuDriverShim.{h,cpp}`
- SHALL hold a non-owning `GPUContext*` pointer
- SHALL maintain a thread-safe `completion_map_` (`std::unordered_map<uint64_t, bool>`) for tracking kernel completion
- SHALL maintain per-SM resource ownership via `vector<unique_ptr<IScoreboard>>`, `vector<unique_ptr<IPipelineLatencyProvider>>`, `vector<unique_ptr<ITensorCoreTiming>>`

#### Scenario: advance() drives GPUContext::exe_once()

- **WHEN** `advance(max_cycles, actual_cycles)` is called with a valid `GPUContext*`
- **THEN** SHALL call `ctx_->exe_once()` up to `max_cycles` times
- **THEN** SHALL set `actual_cycles` to the number of executed cycles
- **THEN** SHALL return `AdvanceResult::Executed` if `actual_cycles > 0`
- **THEN** SHALL return `AdvanceResult::NoOp` if no cycles executed (kernel not started or completed)

#### Scenario: advance() returns KernelComplete on EXIT

- **WHEN** `advance()` detects `ctx_->get_state() == EXIT` after an `exe_once()` call
- **THEN** SHALL mark all registered kernel IDs as complete in `completion_map_`
- **THEN** SHALL return `AdvanceResult::KernelComplete`

#### Scenario: advance() returns Error on exception

- **WHEN** `exe_once()` throws an exception
- **THEN** SHALL catch the exception and return `AdvanceResult::Error`
- **THEN** SHALL NOT crash or propagate unhandled exceptions

#### Scenario: advance() returns Error on null context

- **WHEN** `GPUContext*` is nullptr
- **THEN** SHALL return `AdvanceResult::Error` immediately

#### Scenario: inject_scoreboard assigns to SMContext

- **WHEN** `inject_scoreboard(sm_id, unique_ptr<IScoreboard>)` is called with valid `sm_id`
- **THEN** SHALL call `scoreboard->reset()` to clear stale entries from previous kernel
- **THEN** SHALL call `ctx_->get_sm(sm_id)->set_scoreboard(ptr)` to assign to SMContext
- **THEN** SHALL move the unique_ptr into the internal `vector<unique_ptr>` to retain ownership

#### Scenario: inject_scoreboard ignores out-of-range sm_id

- **WHEN** `inject_scoreboard(sm_id, ...)` is called with `sm_id >= num_sms()`
- **THEN** SHALL return without modifying any SMContext
- **THEN** the unique_ptr SHALL be released (destroyed)

#### Scenario: inject_pipeline assigns to SMContext

- **WHEN** `inject_pipeline(sm_id, unique_ptr<IPipelineLatencyProvider>)` is called
- **THEN** SHALL forward to `ctx_->get_sm(sm_id)->set_pipeline_latency_provider(ptr)`
- **THEN** SHALL retain ownership in internal vector

#### Scenario: inject_tensor_core assigns to SMContext

- **WHEN** `inject_tensor_core(sm_id, unique_ptr<ITensorCoreTiming>)` is called
- **THEN** SHALL forward to `ctx_->get_sm(sm_id)->set_tensor_core_timing(ptr)`
- **THEN** SHALL retain ownership in internal vector

#### Scenario: is_kernel_complete returns completion status

- **WHEN** `is_kernel_complete(kernel_id)` is called
- **THEN** SHALL return `true` if `kernel_id` exists in completion_map_ and is marked complete
- **THEN** SHALL return `false` if `kernel_id` exists but is not yet complete
- **THEN** SHALL return `false` if `kernel_id` is unknown

#### Scenario: mark_complete sets kernel completion flag

- **WHEN** `mark_complete(kernel_id)` is called
- **THEN** SHALL update the entry in completion_map_ to `true`
- **THEN** SHALL be thread-safe (protected by mutex)

#### Scenario: num_sms returns GPUContext SM count

- **WHEN** `num_sms()` is called
- **THEN** SHALL return `ctx_->get_num_sms()`

#### Scenario: shim is C++17 compatible

- **WHEN** `PtxEmuDriverShim` is compiled
- **THEN** SHALL only use `uint32_t`, `uint64_t`, `unique_ptr`, and vendored pure virtual interfaces from CppTLM
- **THEN** SHALL NOT include any CppTLM C++ implementation files (only headers via the install directory)