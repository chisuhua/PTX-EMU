# cpptlm-d1-full Specification

## Purpose
TBD - created by archiving change cpptlm-d1-full. Update Purpose after archive.
## Requirements

> **2026-08-25 SUPERSEDED**: This spec has been largely superseded by [`cudart-sync-only-runtime`](../cudart-sync-only-runtime/spec.md). The bridge coupling cleanup change (commit `09786635` + `292022a3` + `e4d7e369`) physically removed the `g_cpptlm_bridge` global, `cpptlm_set_driver`/`cpptlm_attach_bridge`/`cpptlm_detach_bridge` ABI, `PtxEmuDriverShim`, `StubBridge`, `BUILD_LIB_CPPTLM_CUDART` macro, and GLOBAL LD/ST bridge from PTX-EMU's `libcudart.so`. The only retained artifact from this spec is the static_assert set (preserved in [`abi_guards`](../abi_guards/spec.md) file at `include/cudart/abi_guards.h`). For the current PTX-EMU runtime contract, see `cudart-sync-only-runtime/spec.md`.

### Requirement: cudart-stream-api

The `cudaStreamCreate` and `cudaStreamDestroy` functions in `src/cudart/cudart_sim.cpp` SHALL continue to use `generate_kernel_id()` and `g_active_streams` for stream ID generation and tracking, but SHALL NOT require `g_pending_kernels_mutex` (which was used to protect `g_active_streams.erase` in the bridge path). The stream mutex was specific to the bridge async path's `PendingKernel` map; with `PendingKernel` removed, the mutex is no longer needed.

#### Scenario: Stream lifecycle works without pending_kernels mutex

- **WHEN** a CUDA program calls `cudaStreamCreate(&stream)` and later `cudaStreamDestroy(stream)` from the same thread
- **THEN** `cudaStreamCreate` inserts stream ID into `g_active_streams`
- **AND** `cudaStreamDestroy` erases stream ID from `g_active_streams` (no mutex needed)
- **AND** the operation is consistent (no use-after-free)

#### Scenario: cudaStreamCreate 分配 64-bit 唯一 ID

- **WHEN** 调用 `cudaStreamCreate(&pStream)`
- **THEN** `*pStream = reinterpret_cast<cudaStream_t>(static_cast<uintptr_t>(id))`
- **AND** `id` 通过 `next_kernel_id.fetch_add(1)` 生成（64-bit atomic 计数器）
- **AND** `id` 插入 `g_active_streams`
- **AND** `id` 与现有 kernel_id / stream_id 无冲突（unique-by-construction）

