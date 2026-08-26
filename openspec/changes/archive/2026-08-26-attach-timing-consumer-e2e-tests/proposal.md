## Why

CppTLM commit `d909407` (2026-08-25) added 7 unit tests in `test/test_cuda_core_adapter_timing.cc` that verify CppTLM-side `CudaCoreAdapterMVP` correctly injects HSK-4 timing modules (`IScoreboard` / `IPipelineLatencyProvider` / `ITensorCoreTiming`) via `facade.attach_timing(...)`. However, PTX-EMU side has **no corresponding reverse-direction integration tests** verifying that (a) `IPtxEmuDevice::attach_timing` correctly injects the modules into `SMContext`, AND (b) `SMContext::exe_once()` actually queries the injected modules during instruction execution.

This creates an asymmetric test gap: a regression in `src/ptxemu/device_api_impl.cc::attach_timing` (the `static_cast<void*>` round-trip bridge per HSK-8 spec Decision 6) or in `src/ptxsim/core/sm_context.cpp` step_a/step_b/step_c injection points would not be caught by PTX-EMU's CI, only by CppTLM-side end-to-end tests. **Without PTX-EMU side coverage, the HSK-8 contract lacks defensive verification on the producer side.**

This change adds 4 integration tests on PTX-EMU side that go through the **public `IPtxEmuDevice::attach_timing` API** and verify the injected interfaces are queried by `SMContext` execution paths.

## What Changes

- **NEW** `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp`: 4 `TEST_CASE` covering G1 (scoreboard queried by step_a/c), G2 (pipeline queried by step_b for S_FMA), G3 (tensor_core queried by step_b for S_TCGEN05_MMA), G4 (e2e — `sm.exe_once()` queries all 3 injected interfaces)
- **MODIFY** `include/ptxemu/testing/warp_executor_test_fixture.h`: add backward-compatible optional `std::vector<StatementContext> statements = {}` parameter to `WarpExecutorTestFixture` constructor (default empty → 3 existing fixture-using tests unaffected)
- **MODIFY** `tests/integration/CMakeLists.txt`: register `integration_attach_timing_consumer_e2e` ctest target

**NOT a breaking change**:
- `include/ptxemu/device_api.h` untouched, `PTXEMU_API_VERSION=1` frozen, drift_check 7 invariants unchanged
- `WarpExecutorTestFixture` default parameter → existing 3 fixture-using tests compile and run without changes (`tests/integration/simt/test_set_active_mask_overwrite.cpp`, `tests/integration/warp/test_warp_status_snapshot.cpp`, `tests/integration/warp/test_device_api_delegation_e2e.cc`)

## Capabilities

### New Capabilities

- `attach-timing-consumer-e2e`: PTX-EMU-side integration test coverage for `IPtxEmuDevice::attach_timing` end-to-end behavior — verifies namespace bridge round-trip identity + downstream query path participation in `SMContext::exe_once`. Reuses existing `WarpExecutorTestFixture` (`include/ptxemu/testing/warp_executor_test_fixture.h:42-100`) for warp setup; only change is a backward-compatible optional parameter to inject custom statements for G4.

### Modified Capabilities

- `ptxemu-device-api-delegation`: extends Requirement `IPtxEmuDevice::attach_timing injects HSK-4 vendored interfaces into SMContext` to also require integration test coverage for downstream query path participation (new Requirement + Scenarios).

## Impact

**Affected code**:
- `include/ptxemu/testing/warp_executor_test_fixture.h` — +3 lines (default param in constructor signature + 1 line change to use the parameter in CTAContext::init)
- `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp` — NEW, ~250 lines
- `tests/integration/CMakeLists.txt` — +5 lines (add_catch_test registration)

**Affected APIs**:
- Public device API: NO change (`include/ptxemu/device_api.h` untouched, `PTXEMU_API_VERSION=1` frozen)
- Test fixture: `WarpExecutorTestFixture` gains optional `statements` parameter with `{}` default → existing tests unaffected (source-compatible change)
- Production code (SMContext, GPUContext, WarpScheduler, etc.): NO change

**Affected tests**:
- Baseline: 251/251 ctest PASS → target: 252/252 ctest PASS (+1 integration target with 4 TEST_CASE)
- drift_check: 7 invariants PASS unchanged (Invariant 7 added by `antlr4-path-hardcoding-fix` commit `2148e15c` is unrelated to this change)

**Affected HSK protocol**: NONE — internal refactor only.

**Affected ADRs**: NONE — does not alter HSK-8 contract or any architectural decision.

**Design-Time Checklist (per ptx-lessons-learned §open OpenSpec)**:
- ✅ Section A (API migration): N/A — no migration; only test addition with fixture parameterization
- ✅ Section B (state modification): N/A — test fixture parameter is the only state; no production state modification
- ✅ Section C (multi-phase): 2 phases (instead of 3 — `sm_test_access` friend namespace eliminated per Path A pivot), each independently revertible (see tasks.md)
- ✅ Section D (docs sync): no AGENTS.md change needed; no ADR change needed (drift_check doesn't add new invariant)

**Path A Pivot Decision** (rejected Path B friend namespace approach per Oracle consultation):
- Metis pre-impl review found 4 BLOCKERs in original design (warp_scheduler dual registration, missing init() call, wrong CMakeLists path, empty operands)
- Oracle consultation recommended Path A: leverage existing `WarpExecutorTestFixture` instead of re-inventing `sm_test_access::add_warp` (per `ptx-lessons-learned` §1 cross-module state translation: test helpers that bypass production paths create translation drift)
- Path A eliminates 2 of 4 BLOCKERs (fixture already handles `init()` and warp scheduler dual registration via `add_block → sm_warp_lifecycle::update_state`)