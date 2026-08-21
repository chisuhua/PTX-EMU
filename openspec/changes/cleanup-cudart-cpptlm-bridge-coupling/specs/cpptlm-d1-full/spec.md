## REMOVED Requirements

### Requirement: auto-co-simulation (entire spec `openspec/specs/auto-co-simulation/spec.md`)

**Reason**: The auto-co-simulation capability required `BUILD_LIB_CPPTLM_CUDART=ON` + `g_cpptlm_bridge` auto-attach to a `StubBridge` singleton + `g_ptx_emu_driver_shim->advance()` driving PTX execution in `cudaDeviceSynchronize` / `cudaStreamSynchronize`. All four preconditions are deleted by this change (bridge file deletion, `cpptlm_set_driver` removal, `StubBridge` removal, `EMU_COSIM` env var removal, `PTX_EMU_MAX_ADVANCE_CYCLES` env var removal). The `auto-co-simulation/spec.md` describes mechanisms (`StubBridge::submit_kernel`, `StubBridge::poll_kernel`, `g_bridge_user_override`, `cpptlm_attach_bridge` test override, advance ceiling safety) that no longer exist in PTX-EMU's `libcudart.so`. Leaving the spec un-retired would create spec/code drift — readers would query `auto-co-simulation/spec.md` and find behaviors that the post-change runtime cannot satisfy.

**Migration**: Programs that previously relied on auto-co-simulation MUST be rewritten to use synchronous launch semantics (the default after this change). Co-simulation with CppTLM is no longer supported via `libcudart.so`; downstream consumers SHOULD migrate to `libptxemu_device.so` ABI per `cudart-sync-only-runtime` spec.

### Requirement: cpptlm-bridge-interface

**Reason**: The `CppTLMBridge` virtual class and the reverse-direction ABI (`g_cpptlm_bridge` global, `cpptlm_attach_bridge`/`cpptlm_detach_bridge`, `PtxEmuDriverApi`, `cpptlm_set_driver`) are removed from PTX-EMU's `libcudart.so` as part of the bridge coupling cleanup. CppTLM's runtime co-simulation is no longer a supported mode for PTX-EMU. The `libcudart.so` is now a synchronous-only runtime shim per the `cudart-sync-only-runtime` spec.

**Migration**: External consumers that previously linked against `libcudart.so` for CppTLM bridge functionality MUST migrate to `libptxemu_device.so` ABI (`ptxemu_image_load`/`execute`/`unload`/`kernel_name`/`module_version` per `libptxemu-abi-freeze` spec). The CppTLM-side migration is tracked in a separate future change (`cpp-tlm-consumes-ptxemu-device`).

### Requirement: cudart-async-launchkernel

**Reason**: The async path in `cudaLaunchKernel` (triggered by `g_cpptlm_bridge != nullptr` check) is removed because the bridge no longer exists. The synchronous path remains — it was the default behavior before the bridge was added and is the only behavior after this change.

**Migration**: CUDA programs that relied on async launch semantics via the CppTLM bridge MUST be rewritten to use synchronous launch semantics. Most CUDA programs already work synchronously; the bridge was an opt-in co-simulation mode, not the default.

## MODIFIED Requirements

### Requirement: cudart-stream-api

The `cudaStreamCreate` and `cudaStreamDestroy` functions in `src/cudart/cudart_sim.cpp` SHALL continue to use `generate_kernel_id()` and `g_active_streams` for stream ID generation and tracking, but SHALL NOT require `g_pending_kernels_mutex` (which was used to protect `g_active_streams.erase` in the bridge path). The stream mutex was specific to the bridge async path's `PendingKernel` map; with `PendingKernel` removed, the mutex is no longer needed.

#### Scenario: Stream lifecycle works without pending_kernels mutex

- **WHEN** a CUDA program calls `cudaStreamCreate(&stream)` and later `cudaStreamDestroy(stream)` from the same thread
- **THEN** `cudaStreamCreate` inserts stream ID into `g_active_streams`
- **AND** `cudaStreamDestroy` erases stream ID from `g_active_streams` (no mutex needed)
- **AND** the operation is consistent (no use-after-free)