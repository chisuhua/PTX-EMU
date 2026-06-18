# Capability: cta-barrier-module

CTA 级屏障统一管理模块——通过 `BarrierModule` API 让 `BarHandler::executeBarrier`（CTA 级 `bar.sync`）走新设计路径，消除 `SMContext::synchronize_barrier` 的全局 mutex + map 实现。

## ADDED Requirements

### Requirement: CTA barrier handler MUST use BarrierModule API

The system MUST route `bar.sync` (CTA-level barrier) execution through `BarrierModule::arrive_at_cta_barrier` rather than `SMContext::synchronize_barrier`. Each CTA MUST own exactly one `BarrierModule` instance, scoped to the CTA's lifetime.

#### Scenario: BarHandler calls arrive_at_cta_barrier
- **WHEN** a thread executes `bar.sync N` PTX instruction
- **THEN** `BarHandler::executeBarrier` MUST call `cta_ctx->get_barrier_module().arrive_at_cta_barrier(N, thread_context)`
- **AND** MUST NOT call `sm_context->synchronize_barrier()`

#### Scenario: CTA owns BarrierModule instance
- **WHEN** a CTA is created (via `cudaLaunchKernel`)
- **THEN** `CTAContext` MUST initialize a new `BarrierModule` in its constructor
- **AND** when the CTA is destroyed, the `BarrierModule` MUST be destroyed via `unique_ptr` automatic cleanup

### Requirement: BarrierModule MUST support 16 named CTA barriers

The `BarrierModule` MUST provide 16 CTA barrier slots (indices 0-15), aligned with NVIDIA hardware `HardwareMaxNumNamedBarriers = 16`.

#### Scenario: All 16 barrier slots are accessible
- **WHEN** any code requests `barrier_module.get_cta_barrier(N)` for `N` in `[0, 15]`
- **THEN** it MUST return a valid `CTABarrier*`
- **AND** `init_cta_barrier(N, ...)` MUST succeed without error

#### Scenario: Invalid barrier slot rejected
- **WHEN** code requests `barrier_module.get_cta_barrier(16)` or negative index
- **THEN** it MUST return `nullptr` and emit `PTX_ERROR_EMU`

### Requirement: release_cta_barrier MUST advance thread PC

`BarrierModule::release_cta_barrier(cta_barrier_id, cta_ctx)` MUST, for every thread in `arrived_threads_`, call `set_state(RUN)` AND `advance_thread_pc(lane, post_barrier_pc)` so the scheduler can resume execution without manual PC patching.

#### Scenario: All waited threads released to post-barrier PC
- **WHEN** `release_cta_barrier` is called and `is_complete() == true`
- **THEN** every `ThreadContext*` in `arrived_threads_` MUST have `state == RUN`
- **AND** every corresponding `warp_state.threads[lane].pc == post_barrier_pc` (advancing past the barrier)

#### Scenario: No PC advance if barrier not complete
- **WHEN** `release_cta_barrier` is called and `is_complete() == false`
- **THEN** it MUST emit `PTX_ERROR_EMU` and return without modifying any thread state

### Requirement: SMContext MUST NOT maintain global barrier state

After integration, `SMContext` MUST NOT hold `barrier_waiting_threads`, `barrier_thread_counts`, or `barrier_mutex_`. All CTA barrier state MUST live in `CTAContext::barrier_module_`.

#### Scenario: No synchronize_barrier method on SMContext
- **WHEN** the integration is complete
- **THEN** `SMContext::synchronize_barrier()` MUST NOT exist in the codebase (verified by `grep -rn "synchronize_barrier" src/`)
- **AND** `SMContext::barrier_mutex_` and related map fields MUST be removed
