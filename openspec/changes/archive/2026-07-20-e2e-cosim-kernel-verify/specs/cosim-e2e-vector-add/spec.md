## ADDED Requirements

### Requirement: CUDA vectorAdd kernel end-to-end co-simulation

The system SHALL include an E2E test that exercises the CppTLM bridge callback chain (mock `submit_kernel` + `poll_kernel`) and verifies CUDA kernel execution results match CPU golden values.

**Implementation note**: The test attaches the mock bridge AFTER kernel launch (not before), exercising the synchronous execution path (`launchPtxInterpreter` + `wait_for_completion`) while the bridge mock is used by `cudaDeviceSynchronize`'s polling loop. Bridge-path dual-enqueue correctness is deferred to follow-up change `fix-bridge-path-2-cycle-exit` (see known-limitation Scenario below).

#### Scenario: kernel compiles and launches; bridge mock is attached and exercised

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON` and the test calls `vectorAdd<<<...>>>()`
- **THEN** the kernel SHALL compile, register, and launch via the synchronous execution path
- **THEN** `cpptlm_attach_bridge(&mock)` SHALL succeed and `g_cpptlm_bridge == &mock`
- **THEN** `cudaDeviceSynchronize()` SHALL invoke the bridge's `poll_kernel` callback
- **NOTE** For bridge-path dual-enqueue (`g_cpptlm_bridge` set BEFORE launch) see the known-limitation Scenario below.

#### Scenario: PTX instructions execute correctly via test-driven advance

- **WHEN** the test calls `g_ptx_emu_driver_shim->advance(N, actual)` to drive `GPUContext::exe_once()`
- **THEN** LD/ST instructions SHALL read/write correct device memory addresses
- **THEN** ADD instruction SHALL compute correct sum

#### Scenario: output matches golden value

- **WHEN** `cudaDeviceSynchronize` returns after bridge polling loop drains `g_pending_kernels`
- **THEN** device memory output SHALL equal CPU-computed golden value (`REQUIRE(output[i] == golden[i])` for all i)
- **THEN** the combination of (a) `cudaDeviceSynchronize` returning without error and (b) golden value matching SHALL be accepted as indirect proof that kernel execution completed correctly

#### Scenario: test target not created when bridge not enabled

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=OFF`
- **THEN** the test target SHALL not be created (`ctest -R e2e_cosim_vector_add` returns "No tests were found")

#### Scenario: g_ptx_emu_driver_shim is accessible from test code

- **WHEN** the test includes `PtxEmuDriverShim.h`
- **THEN** `g_ptx_emu_driver_shim` SHALL be accessible as an `extern` symbol (no longer `static` in `cudart_sim.cpp`)

#### Scenario: test executes via synchronous path (bridge attached AFTER launch) — known limitation

- **WHEN** the E2E test calls `vectorAdd<<<...>>>()` BEFORE `cpptlm_attach_bridge(&mock)`
- **THEN** `cudaLaunchKernel` takes the synchronous path (`launchPtxInterpreter` + `wait_for_completion`), NOT the bridge-path dual-enqueue
- **NOTE** This is a deliberate test design to avoid the known bridge-path 2-cycle completion bug (see `Scenario: bridge-path dual-enqueue is known-broken` below). The bridge mock is attached AFTER launch and is used only by `cudaDeviceSynchronize`'s polling loop.
- **NOTE** Dual-enqueue correctness is NOT validated by this change. It is deferred to follow-up change `fix-bridge-path-2-cycle-exit`.

#### Scenario: bridge-path dual-enqueue is known-broken (deferred to follow-up change)

- **WHEN** `g_cpptlm_bridge` is non-null at `cudaLaunchKernel` time (bridge attached BEFORE launch)
- **THEN** the dual-enqueue (`submit_kernel` + `prepareKernelLaunchRequest()`) is invoked, BUT `GPUContext::exe_once()` admits the kernel and judges `EXIT` in the same call, producing 2-cycle completion with zero PTX output
- **THEN** this scenario is documented as a known limitation deferred to `fix-bridge-path-2-cycle-exit` follow-up change
- **NOTE** Root cause hypothesis: `GPUContext::exe_once()` (`src/ptxsim/core/gpu_context.cpp:246-336`) processes the task queue and executes SMs in the same call. When the kernel is admitted via bridge path, SM state transitions cause immediate `all_warps_finished()` true, leading to `gpu_state = EXIT` without actual PTX execution. Synchronous path does NOT exhibit this bug because `launchPtxInterpreter` + `wait_for_completion` use the same `exe_once` loop but have different SM state initialization context.
- **NOTE** See [ADR-0021 §2026-07-XX Postmortem](docs/adr/ADR-0021-cpptlm-d1-full-integration.md) for full diagnostic context and follow-up change references.
