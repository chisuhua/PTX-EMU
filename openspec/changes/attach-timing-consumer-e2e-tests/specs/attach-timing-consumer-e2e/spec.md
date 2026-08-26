# attach-timing-consumer-e2e Specification

## Purpose
PTX-EMU-side integration test coverage for `IPtxEmuDevice::attach_timing` end-to-end behavior — verifies namespace bridge round-trip identity + downstream query path participation in `SMContext::exe_once()`. Closes the asymmetric test gap left by CppTLM commit `d909407` (2026-08-25) which only verified consumer-side flow.

> **Scope**: This spec covers 4 `TEST_CASE` (G1-G4) in a single file `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp`, plus a backward-compatible extension to `WarpExecutorTestFixture`.

## Requirements

### Requirement: Integration tests verify `IPtxEmuDevice::attach_timing` namespace bridge round-trip identity

The PTX-EMU test suite MUST contain integration tests that verify `IPtxEmuDevice::attach_timing(IScoreboard*, IPipelineLatencyProvider*, ITensorCoreTiming*)` correctly injects the HSK-4 vendored interfaces into `SMContext` via the `static_cast<void*>` namespace bridge (per HSK-8 spec Decision 6). After injection, `sm->get_scoreboard()`, `sm->get_pipeline_latency_provider()`, and `sm->get_tensor_core_timing()` MUST return pointers equal to the original mock instances (round-trip identity preserved across the bridge).

#### Scenario: Scoreboard round-trip identity via public API
- **WHEN** a test creates a `::IScoreboard` mock instance and calls `device->attach_timing(ptxemu_sb, nullptr, nullptr)` (where `ptxemu_sb` is the `ptxemu::IScoreboard*` obtained via `static_cast<ptxemu::IScoreboard*>(static_cast<void*>(&mock))`)
- **THEN** `sm->get_scoreboard()` returns a pointer equal to `&mock`
- **AND** no copy or move occurred across the bridge

#### Scenario: Pipeline round-trip identity via public API
- **WHEN** a test creates a `::IPipelineLatencyProvider` mock instance and calls `device->attach_timing(nullptr, ptxemu_pipe, nullptr)`
- **THEN** `sm->get_pipeline_latency_provider()` returns a pointer equal to `&mock`

#### Scenario: TensorCore round-trip identity via public API
- **WHEN** a test creates a `::ITensorCoreTiming` mock instance and calls `device->attach_timing(nullptr, nullptr, ptxemu_tc)`
- **THEN** `sm->get_tensor_core_timing()` returns a pointer equal to `&mock`

### Requirement: Integration tests verify `IScoreboard` queried by `SMContext` step_a / step_c injection paths

The PTX-EMU test suite MUST contain an integration test that verifies the injected `IScoreboard` is queried during `SMContext::exe_once()` via the `step_a_scoreboard_check` and `step_c_release_scoreboard` call sites (`src/ptxsim/core/sm_context.cpp:273`, `:309`, `:316`, `:368`, `:395`, `:402`). The mock MUST record `allocate` and `release` call counts, and the test MUST assert both counts are > 0 after `sm.exe_once()` runs with at least one warp executing one instruction.

#### Scenario: scoreboard.allocate called by step_a
- **WHEN** `IScoreboard` is injected via `IPtxEmuDevice::attach_timing` AND `sm.exe_once()` runs with one warp executing one schedulable instruction
- **THEN** the mock's `allocate` counter is > 0 (proves step_a queried the scoreboard)

#### Scenario: scoreboard.release called by step_c
- **WHEN** `IScoreboard` is injected AND `sm.exe_once()` runs with one warp executing one schedulable instruction
- **THEN** the mock's `release` counter is > 0 (proves step_c queried the scoreboard)

### Requirement: Integration tests verify `IPipelineLatencyProvider` queried by step_b for non-TC instructions

The PTX-EMU test suite MUST contain an integration test that verifies the injected `IPipelineLatencyProvider` is queried during `step_b_set_blocked_cycles` (`src/ptxsim/core/sm_context_cpptlm_inject.cpp:22`) via the `get_fractional_cycles_by_type(int, PipelineId)` call. The mock MUST record call counts, and the test MUST assert the count is > 0 after `step_b_set_blocked_cycles` runs with a non-TC instruction (e.g., `S_FMA` mapped to `PipelineId::P0_INT_FP32`).

#### Scenario: pipeline.get_fractional_cycles_by_type called for S_FMA
- **WHEN** `IPipelineLatencyProvider` is injected via `IPtxEmuDevice::attach_timing` AND `SMContext::step_b_set_blocked_cycles(pipeline, nullptr, warp, S_FMA_stmt)` is called
- **THEN** the mock's `get_fractional_cycles_by_type` counter is > 0
- **AND** the call's `PipelineId` argument equals `PipelineId::P0_INT_FP32`

### Requirement: Integration tests verify `ITensorCoreTiming` queried by step_b for TC instructions

The PTX-EMU test suite MUST contain an integration test that verifies the injected `ITensorCoreTiming` is queried during `step_b_set_blocked_cycles` fallback path (`src/ptxsim/core/sm_context_cpptlm_inject.cpp:28-32`) when the pipeline provider returns 0 AND the instruction is a tensor core instruction. The mock MUST record call counts, and the test MUST assert the count is > 0 after `step_b_set_blocked_cycles` runs with `S_TCGEN05_MMA` and a pipeline mock returning 0.

#### Scenario: tensor_core.get_latency called for S_TCGEN05_MMA with pipeline returning 0
- **WHEN** `ITensorCoreTiming` is injected via `IPtxEmuDevice::attach_timing` AND pipeline mock returns 0 AND `SMContext::step_b_set_blocked_cycles(pipeline_zero, tc, warp, S_TCGEN05_MMA_stmt)` is called
- **THEN** the mock's `get_latency` counter is > 0
- **AND** the call's `TcPrecision` argument is `TcPrecision::FP16` (default from `map_instruction_to_tc_precision` for unqualified TC instructions)

### Requirement: End-to-end test verifies all 3 interfaces queried in single `sm.exe_once()` run

The PTX-EMU test suite MUST contain an end-to-end integration test that injects all 3 HSK-4 interfaces simultaneously via `IPtxEmuDevice::attach_timing`, runs `sm.exe_once()` with one warp executing one S_FMA instruction, and asserts that:
1. `scoreboard.allocate_calls > 0` AND `scoreboard.release_calls > 0` (step_a + step_c)
2. `pipeline.get_fractional_cycles_by_type_calls > 0` (step_b pipeline path, since S_FMA is non-TC)
3. `tensor_core.get_latency_calls == 0` (step_b TC path NOT triggered, since S_FMA is non-TC — this is the correctness assertion that the pipeline path takes priority)

#### Scenario: e2e with S_FMA — pipeline queried, TC not queried
- **WHEN** all 3 interfaces injected AND `sm.exe_once()` runs one warp with S_FMA
- **THEN** `scoreboard.allocate_calls > 0` AND `scoreboard.release_calls > 0`
- **AND** `pipeline.get_fractional_cycles_by_type_calls > 0`
- **AND** `tensor_core.get_latency_calls == 0` (S_FMA is non-TC, pipeline path takes priority)

### Requirement: Tests reuse `WarpExecutorTestFixture` for GPU/SM/warp setup

The integration tests MUST use the existing `WarpExecutorTestFixture` (`include/ptxemu/testing/warp_executor_test_fixture.h:42-100`) for `g_gpu_context` + `init()` + `sm_->add_block(CTAContext)` setup. Tests MUST NOT re-implement warp setup via friend namespaces or direct `sm->warps[]` manipulation. The fixture's RAII pattern MUST guarantee:
1. Before each test: a fresh `GPUContext` is installed in `g_gpu_context` (with `init()` called)
2. After each test: the previous `g_gpu_context` is restored (or set to nullptr)
3. SM 0 contains at least 1 warp added via `add_block` (production registration path, correctly registered with `warp_scheduler`)

#### Scenario: WarpExecutorTestFixture installs initialized GPUContext
- **WHEN** a `WarpExecutorTestFixture` instance is constructed at test entry
- **THEN** `g_gpu_context` points to an initialized `GPUContext`
- **AND** `gpu()->get_sm(0)` returns a valid `SMContext` pointer (non-null)

#### Scenario: WarpExecutorTestFixture restores g_gpu_context on destruction
- **WHEN** a `WarpExecutorTestFixture` instance is destroyed at test exit
- **THEN** `g_gpu_context` is restored to its pre-construction value (or nullptr if previously absent)

### Requirement: `WarpExecutorTestFixture` supports optional `statements` parameter (backward-compatible)

`WarpExecutorTestFixture` constructor MUST accept an optional `std::vector<StatementContext> statements = {}` parameter. The parameter MUST be passed through to the `CTAContext::init(...)` call (instead of constructing an empty local vector). Default value `{}` MUST preserve existing behavior — 3 existing fixture-using tests compile and run without source changes.

#### Scenario: Default empty statements preserves existing behavior
- **WHEN** `WarpExecutorTestFixture` is constructed without arguments (3 existing tests)
- **THEN** the constructor behaves identically to the pre-change implementation (empty statements vector passed to `block->init`)
- **AND** all 3 existing fixture-using tests still pass without modification:
  - `tests/integration/simt/test_set_active_mask_overwrite.cpp`
  - `tests/integration/warp/test_warp_status_snapshot.cpp`
  - `tests/integration/warp/test_device_api_delegation_e2e.cc`

#### Scenario: Custom statements enable S_FMA scheduling
- **WHEN** `WarpExecutorTestFixture({make_ffma("%f0", "%f1", "%f2", "%f3")})` is constructed
- **THEN** the warp created by `add_block` contains the FFMA statement at PC=0
- **AND** `warp->get_lanes_by_pc()` returns non-empty (PC=0 schedulable)
- **AND** `sm.exe_once()` can execute the FFMA statement (guard at `sm_context.cpp:266-268` passes)

### Requirement: No public ABI change

The change MUST MUST NOT modify `include/ptxemu/device_api.h` (frozen per HSK-8 spec Decision 3), MUST NOT bump `PTXEMU_API_VERSION` (frozen at 1), and MUST NOT add new methods to `IPtxEmuDevice`. The drift_check 7 invariants MUST remain green.

#### Scenario: device_api.h unchanged
- **WHEN** the change is complete
- **THEN** `git diff HEAD~N..HEAD -- include/ptxemu/device_api.h` is empty
- **AND** `PTXEMU_API_VERSION` macro value is still 1

#### Scenario: drift_check 7 invariants PASS
- **WHEN** `drift_check.yml` workflow runs
- **THEN** all 7 invariants (no public ABI change, no VERSION bump, no namespace pollution, ANTLR4 path, etc.) remain green

### Requirement: Test isolation (no global state leak)

Each `TEST_CASE` MUST be independent — no global state from one test affects another. The fixture's RAII MUST restore `g_gpu_context` to its pre-construction state on destruction, ensuring test order independence.

#### Scenario: Tests run in any order produce same result
- **WHEN** `integration_attach_timing_consumer_e2e` ctest target runs (4 TEST_CASE)
- **THEN** all 4 cases pass regardless of execution order
- **AND** running the target multiple times produces identical results