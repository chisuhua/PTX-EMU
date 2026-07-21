## ADDED Requirements

### Requirement: Standard CUDA program zero-code co-simulation

The system SHALL automatically enable co-simulation mode for any standard CUDA program when `BUILD_LIB_CPPTLM_CUDART=ON`, without requiring the program to include PTX-EMU-specific APIs (`cpptlm_attach_bridge`, `PtxEmuDriverShim`, `advance()`, etc.).

#### Scenario: Standard CUDA vectorAdd works as co-simulation

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON`
- **AND** the program is a plain CUDA `.cu` file containing only `cudaMalloc`, `cudaMemcpy`, `vectorAdd<<<>>>`, `cudaDeviceSynchronize`, and `cudaMemcpy` back
- **THEN** `g_cpptlm_bridge` SHALL be automatically set to a valid `StubBridge` instance during environment initialization (no manual `cpptlm_attach_bridge` call)
- **THEN** `cudaLaunchKernel` SHALL invoke the bridge path dual-enqueue (`submit_kernel` + `prepareKernelLaunchRequest` + `submit_kernel_request`)
- **THEN** `cudaDeviceSynchronize` SHALL call `g_ptx_emu_driver_shim->advance()` to drive PTX execution before the `poll_kernel` loop
- **THEN** device memory output SHALL match CPU golden value for all elements
- **THEN** the program SHALL NOT contain any PTX-EMU-specific includes or API calls

#### Scenario: Auto-attach StubBridge at initialization

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON`
- **AND** `initialize_environment()` runs during `__cudaRegisterFatBinary`
- **THEN** a `StubBridge` singleton SHALL be created and `g_cpptlm_bridge` SHALL point to it
- **THEN** `StubBridge::submit_kernel` SHALL capture `kernel_id` and return 0
- **THEN** `StubBridge::poll_kernel` SHALL return 0 (kernel complete — `advance()` has already run)
- **THEN** `StubBridge::global_access` SHALL return 0 (zero-latency stub)

#### Scenario: Auto-advance in cudaDeviceSynchronize and cudaStreamSynchronize

- **WHEN** `g_cpptlm_bridge != nullptr` and `cudaDeviceSynchronize` or `cudaStreamSynchronize(0)` is called
- **THEN** before entering the `poll_kernel` loop, `g_ptx_emu_driver_shim->advance(max_cycles, actual)` SHALL be called to drive PTX execution, where `max_cycles` is read from `PTX_EMU_MAX_ADVANCE_CYCLES` env var (default 10,000,000)
- **THEN** if GPUContext was already in EXIT state (e.g., kernel already completed via a prior sync call), `actual` may be 0 and the call SHALL return `cudaSuccess` immediately
- **THEN** if GPUContext was not EXIT (first advance or kernel in progress), `actual > 0` SHALL be true (kernel executed at least one cycle)
- **THEN** the existing `poll_kernel` loop SHALL then drain `g_pending_kernels`
- **THEN** `cudaDeviceSynchronize` / `cudaStreamSynchronize` SHALL return `cudaSuccess`
- **NOTE**: Auto-advance only applies to `cudaStreamSynchronize(0)` (default stream) — non-zero stream sync is not supported in this change. Standard CUDA programs using non-default streams must use `cudaDeviceSynchronize` to trigger advance, or create streams on the default stream specifically for this mode.

#### Scenario: advance ceiling prevents hang on pathological kernels

- **WHEN** a kernel has an infinite PTX loop or barrier deadlock such that `gpu_state` never reaches `EXIT`
- **AND** `advance(max_cycles)` exhausts the ceiling without completion
- **THEN** `cudaDeviceSynchronize` SHALL log `PTX_ERROR_EMU` and return `cudaErrorUnknown`
- **AND** SHALL NOT hang forever (advance ceiling safety mechanism)
1. The env var `PTX_EMU_MAX_ADVANCE_CYCLES` allows tuning the ceiling

#### Scenario: cpptlm_attach_bridge override preserves testability

- **WHEN** a test calls `cpptlm_attach_bridge(&mock)` BEFORE `__cudaRegisterFatBinary`
- **THEN** `g_bridge_user_override` SHALL be set to `true`
- **THEN** `initialize_environment()` SHALL skip StubBridge auto-attach (because `g_bridge_user_override` is true)
- **THEN** `g_cpptlm_bridge` SHALL remain pointing to the user's mock bridge
- **WHEN** the test later calls `cpptlm_detach_bridge()`
- **THEN** `g_bridge_user_override` SHALL be reset to `false` and `g_cpptlm_bridge` to `nullptr`

#### Scenario: StubBridge poll_kernel returns error for unknown kernel_id

- **WHEN** `StubBridge::poll_kernel(kid)` is called with a `kid` that was never submitted
- **THEN** SHALL return `UINT64_MAX` (unknown kernel_id, per `cpptlm_bridge.h:113` ABI spec)
- **WHEN** called with a submitted `kid`
- **THEN** SHALL return `0` (kernel complete — advance() has already run)

#### Scenario: g_ptx_interpreter null at launch returns error

- **WHEN** `g_cpptlm_bridge != nullptr` and `cudaLaunchKernel` is called
- **AND** `g_ptx_interpreter == nullptr` (PTX parsing failed or not initialized)
- **THEN** `cudaLaunchKernel` SHALL return `cudaErrorUnknown` instead of silently skipping PTX enqueue
- **AND** kernel SHALL NOT be registered in `g_pending_kernels`

#### Scenario: g_ptx_emu_driver_shim remains static

- **WHEN** the test is a plain CUDA program (no `#include "PtxEmuDriverShim.h"`)
- **THEN** `g_ptx_emu_driver_shim` SHALL remain `static` in `cudart_sim.cpp` (not accessible from outside)
- **THEN** `PtxEmuDriverShim.h` SHALL NOT contain `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;`

#### Scenario: cudaLaunchKernel returns bridge submit_kernel error

- **WHEN** `g_cpptlm_bridge != nullptr` and `cudaLaunchKernel` is called
- **AND** `bridge->submit_kernel(kernel_id)` returns a non-zero error code
- **THEN** `cudaLaunchKernel` SHALL return the same error code to the caller
- **AND** the kernel SHALL NOT be registered in `g_pending_kernels`
- **AND** `prepareKernelLaunchRequest` SHALL NOT be called for this kernel

#### Scenario: g_ptx_interpreter null at initialization does not block StubBridge

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON`
- **AND** `g_ptx_interpreter == nullptr` at `initialize_environment()` (PTX parsing failed)
- **THEN** `StubBridge` SHALL still be created and `g_cpptlm_bridge` SHALL point to it (advance on empty GPUContext is a no-op)
- **THEN** `cudaLaunchKernel` SHALL return `cudaErrorUnknown` per the existing null-interpreter guard (`spec.md:60-63`)

#### Scenario: cudaDeviceSynchronize with ceiling exhaustion cleans up state

- **WHEN** a pathological kernel (infinite loop / barrier deadlock) causes `advance(max_cycles)` to exhaust the configured ceiling without reaching EXIT
- **THEN** `cudaDeviceSynchronize` SHALL clear `executing_requests` for the stuck kernel
- **THEN** corresponding entries in `g_pending_kernels` SHALL be erased
- **THEN** SM state SHALL be reset to IDLE
- **THEN** `cudaDeviceSynchronize` SHALL log `PTX_ERROR_EMU` and return `cudaErrorUnknown`
- **AND** a subsequent `cudaDeviceSynchronize` call (ceiling reset) SHALL NOT re-advance the already-cleaned kernel

#### Scenario: repeated cudaDeviceSynchronize drains all pending kernels

- **WHEN** multiple kernels are launched via the bridge path and `cudaDeviceSynchronize` is called repeatedly
- **THEN** each `advance()` SHALL respect a fresh per-call ceiling
- **THEN** the `poll_kernel` loop in each call SHALL drain `g_pending_kernels` for completed kernels
- **THEN** repeated calls SHALL eventually drain all pending kernels without leaking entries
- **NOTE**: Fire-and-forget kernels (launched without subsequent sync) SHALL retain entries in `g_pending_kernels` — this is a known limitation of the single-threaded host model.

#### Scenario: StubBridge zero-configuration

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=ON` and no external CppTLM library is loaded
- **THEN** the program SHALL still work as co-simulation using the internal `StubBridge`
- **THEN** `global_access` returns 0 (zero-latency — no NoC model)
- **THEN** `synchronize_stream` returns 0 (always synced)

### Requirement: Bridge path production-quality correctness

The bridge path execution flow SHALL produce correct PTX output identical to the synchronous path for all standard CUDA kernels.

#### Scenario: kernel with non-pointer args does not segfault

- **WHEN** the kernel signature includes non-pointer parameters (e.g., `float* A, float* B, float* C, int N`)
- **AND** `cudaLaunchKernel` enters the bridge path
- **THEN** the args deep-copy SHALL use `kernelParams.size()` from the PTX context as the authoritative arg count
- **THEN** SHALL NOT segfault due to nullptr sentinel walk beyond args array bounds

#### Scenario: kernel with all-pointer args still works

- **WHEN** all kernel params are pointers (e.g., `int*, int*, int*`)
- **AND** PTX context lookup fails (e.g., kernel name not found — `arg_count` stays `SIZE_MAX`)
- **THEN** the fallback `count_kernel_args(args)` sentinel walk SHALL be used
- **THEN** the behavior SHALL be identical to the pre-fix code

#### Scenario: multi-warp kernel via bridge path

- **WHEN** the kernel uses 128 threads (4 warps)
- **THEN** all 4 warps SHALL execute PTX instructions correctly via bridge path
- **THEN** device memory output SHALL match golden value for all elements

#### Scenario: barrier kernel via bridge path

- **WHEN** the kernel uses `__syncthreads()` / `bar.sync`
- **THEN** barrier synchronization SHALL work correctly via bridge path
- **THEN** device memory output SHALL match golden value for all elements

### Requirement: Backward compatibility when bridge is OFF

The system SHALL maintain byte-level backward compatibility when `BUILD_LIB_CPPTLM_CUDART=OFF`.

#### Scenario: OFF mode unchanged

- **WHEN** `BUILD_LIB_CPPTLM_CUDART=OFF`
- **THEN** `g_cpptlm_bridge` SHALL remain `nullptr`
- **THEN** `cudaLaunchKernel` SHALL use the synchronous path (`launchPtxInterpreter` + `wait_for_completion`)
- **THEN** all existing tests SHALL pass with zero regression