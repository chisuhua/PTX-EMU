# include/cudart/ AGENTS.md

## OVERVIEW

ABI-critical public headers for fake `libcudart.so` — consumed by PTX-EMU internals and intercepted CUDA programs.

## KEY FILES

| File | Role |
|------|------|
| `abi_guards.h` | ABI 一致性静态断言: cudaStream_t 宽度 + G-D4 12 端点 (PipelineId/TcPrecision) + 4 签名级守卫 (从 cpptlm_bridge.h 迁移, per cleanup-cudart-cpptlm-bridge-coupling) |
| `cuda_driver.h` | CudaDriver singleton — global memory pool lifecycle (malloc/free, mmap-backed) |
| `cudart_intrinsics.h` | CUDA 类型兼容层: cudaStream_t, cudaError_t, dim3, vector types, cudaDeviceProp |
| `cudart_sim.h` | CUDA runtime 入口声明: __cudaRegisterFatBinary, __cudaRegisterFunction, __cudaRegisterVar |
| `simple_memory_allocator.h` | SimpleMemoryAllocator — offset-based bump+free-list allocator (4GB default pool) |

## CONVENTIONS

- **`cudaStream_t` 宽度守卫**: `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` 在 `abi_guards.h` 中 — 防止 stream_id 字段隐式截断
- **G-D4 静态断言**: `PipelineId` (6 端点) + `TcPrecision` (6 端点) + 4 个 `std::is_same_v` 签名守卫 — 跨 .so 边界防 silent enum/签名漂移
- **同步执行模型**: `libcudart.so` 是纯同步 CUDA runtime shim, 无 CppTLM bridge 耦合; co-simulation 走 `libptxemu_device.so` ABI (ADR-0029)

## ANTI-PATTERNS

- ❌ **不要在 `cuda_driver.h` 中硬编码 `GLOBAL_SIZE`** — 由 `SimpleMemory` 实例化时传入, 见 `SimpleMemoryAllocator` 的 `init(size_t)` 设计
- ❌ **不要在 `cudart_intrinsics.h` 中重复定义 CUDA SDK 已有类型** — 5 层 `#ifndef` 守卫 ( `__CUDA_RUNTIME_H__` / `__DRIVER_TYPES_H__` 等) 防止与真实 CUDA 头文件冲突
- ❌ **不要修改 `cudaStream_t` 定义而不更新 `static_assert`** — 宽度变化影响 stream_id 字段 (见 `abi_guards.h`)
