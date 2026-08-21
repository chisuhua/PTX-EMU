## ADDED Requirements

### Requirement: `libcudart.so` synchronous-only CUDA runtime shim

The system SHALL provide `libcudart.so` as a synchronous-only CUDA runtime shim with zero CppTLM coupling. All CUDA runtime entry points (`cudaLaunchKernel`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaStreamSynchronize`, `cudaStreamCreate`, `cudaStreamDestroy`) SHALL use only PTX-EMU's internal completion model. The `g_cpptlm_bridge` global pointer SHALL NOT exist. The `cpptlm_set_driver` / `cpptlm_attach_bridge` / `cpptlm_detach_bridge` symbols SHALL NOT be exported. Co-simulation via CppTLM SHALL be a separate downstream concern using `libptxemu_device.so` (out of scope for this spec).

#### Scenario: cudaLaunchKernel synchronous execution only

- **WHEN** a CUDA program calls `cudaLaunchKernel(func, grid, block, args, sharedMem, stream)` with default mode
- **THEN** PTX-EMU's `cudaLaunchKernel` implementation executes the kernel synchronously
- **AND** returns `cudaSuccess` only after `g_gpu_context->wait_for_completion()` returns
- **AND** does NOT branch on `g_cpptlm_bridge` (no such global exists)

#### Scenario: No cpptlm_* ABI symbols in libcudart.so

- **WHEN** `nm -D build/lib/libcudart.so | grep -E "cpptlm_set_driver|cpptlm_attach_bridge|cpptlm_detach_bridge|g_cpptlm_bridge"` is run after Phase 3
- **THEN** zero symbols are exported (all 4 ABI entries removed)

#### Scenario: Stream lifecycle works without bridge

- **WHEN** a CUDA program calls `cudaStreamCreate(&stream)` followed by `cudaLaunchKernel(..., stream)` and `cudaStreamSynchronize(stream)`
- **THEN** `cudaStreamCreate` returns a valid stream ID (via `generate_kernel_id()` counter)
- **AND** `cudaLaunchKernel` executes the kernel synchronously (calls `g_gpu_context->wait_for_completion()` before returning) — using the stream ID for tracking only
- **AND** `cudaStreamSynchronize` returns immediately (`cudaSuccess`) because the kernel already completed synchronously inside `cudaLaunchKernel`
- **AND** `cudaStreamDestroy` removes the stream ID from `g_active_streams` (lock-free erase after `g_pending_kernels_mutex` deletion; `g_active_streams` was already insert-unlocked, so this preserves the original asymmetric semantics)

### Requirement: PTX-EMU build has zero CppTLM subdirectory dependency

The system SHALL NOT include CppTLM as a subdirectory of PTX-EMU's CMake build. The `PTX-EMU/CMakeLists.txt` SHALL NOT contain `add_subdirectory(${CPPTLM_SOURCE_DIR} ...)`. The `BUILD_LIB_CPPTLM_CUDART` compile definition SHALL NOT be set on the `cudart` target. The `libcudart.so` build SHALL NOT link any CppTLM library.

#### Scenario: PTX-EMU builds standalone

- **WHEN** `cmake -S . -B build -DCMAKE_BUILD_TYPE=Release` is run with CppTLM source directory absent (or unreadable)
- **THEN** CMake configuration succeeds (no `FATAL_ERROR` from CppTLM existence check)
- **AND** `libcudart.so` builds without CppTLM symbols

#### Scenario: CppTLM presence is irrelevant

- **WHEN** `PTX-EMU/CMakeLists.txt` is parsed by CMake
- **THEN** the file does NOT reference `CPPTLM_SOURCE_DIR` variable
- **AND** does NOT call `add_subdirectory` on CppTLM
- **AND** does NOT call `include_directories` on CppTLM include path

### Requirement: Removed `cpptlm_bridge.h` static_asserts preserved in dedicated file

The 17 `static_assert`s originally in `PTX-EMU/include/cudart/cpptlm_bridge.h` (1 `cudaStream_t` width check + 6 `PipelineId` endpoint + 6 `TcPrecision` endpoint + 4 `is_same` signature checks) SHALL be preserved in a new file `PTX-EMU/include/cudart/abi_guards.h` so ABI guards remain effective after the `cpptlm_bridge.h` deletion.

#### Scenario: abi_guards.h contains all 17 assertions

- **WHEN** `cat PTX-EMU/include/cudart/abi_guards.h` is run
- **THEN** the file contains exactly 17 `static_assert` declarations
- **AND** the assertions compile successfully against `ptxsim/` vendored interfaces