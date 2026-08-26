# ptxemu-device-api-delegation Delta Spec

## ADDED Requirements

### Requirement: `IPtxEmuDevice::attach_timing` MUST be covered by integration tests verifying downstream query path participation

The PTX-EMU test suite MUST contain integration tests verifying that `IPtxEmuDevice::attach_timing` results in `SMContext` actually querying the injected HSK-4 vendored interfaces during `SMContext::exe_once()`. The existing requirement `IPtxEmuDevice::attach_timing injects HSK-4 vendored interfaces into SMContext` only verifies injection (round-trip identity via `sm->get_scoreboard()` etc.) — this requirement extends verification to the **downstream query path**: after injection, `step_a_scoreboard_check` MUST query `scoreboard.allocate` and `step_c_release_scoreboard` MUST query `scoreboard.release`, AND `step_b_set_blocked_cycles` MUST query `pipeline.get_fractional_cycles_by_type` for non-TC instructions and `tc.get_latency` for TC instructions when pipeline returns 0.

#### Scenario: Integration test runs `sm.exe_once()` and verifies all 3 mocks are queried
- **WHEN** the PTX-EMU test suite's `integration_attach_timing_consumer_e2e` ctest target runs
- **THEN** all 4 `TEST_CASE` in `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp` PASS
- **AND** the test that runs `sm.exe_once()` with attached mocks records `scoreboard.allocate_calls > 0` AND `scoreboard.release_calls > 0` AND `pipeline.get_fractional_cycles_by_type_calls > 0` (for S_FMA) AND `tensor_core.get_latency_calls == 0` (correctly not triggered for non-TC S_FMA)

#### Scenario: Tests use public API `IPtxEmuDevice::attach_timing`, not direct `sm->set_*`
- **WHEN** the integration tests inject mocks
- **THEN** the injection path is `ptxemu::create_device()->attach_timing(...)` (not `sm->set_scoreboard(...)` etc.)
- **AND** the namespace bridge `static_cast<void*>` round-trip is exercised
- **AND** any regression in `src/ptxemu/device_api_impl.cc::attach_timing` (`device_api_impl.cc:299-310`) is caught