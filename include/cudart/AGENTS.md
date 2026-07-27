# include/cudart/ AGENTS.md

## OVERVIEW

ABI-critical public headers for fake `libcudart.so` — consumed by PTX-EMU internals, intercepted CUDA programs, and the CppTLM co-simulation bridge as an `ExternalProject_Add` dependency.

## KEY FILES

| File | Role | CppTLM ABI? |
|------|------|-------------|
| `cpptlm_bridge.h` | ABI 真值源: CppTLMBridge (5 虚方法), PtxEmuDriverApi (8 函数指针), G-D4 12 端点 static_assert | **Yes** |
| `cuda_driver.h` | CudaDriver singleton — global memory pool lifecycle (malloc/free, mmap-backed) | No |
| `cudart_intrinsics.h` | CUDA 类型兼容层: cudaStream_t, cudaError_t, dim3, vector types, cudaDeviceProp | No |
| `cudart_sim.h` | CUDA runtime 入口声明: __cudaRegisterFatBinary, __cudaRegisterFunction, __cudaRegisterVar | No |
| `simple_memory_allocator.h` | SimpleMemoryAllocator — offset-based bump+free-list allocator (4GB default pool) | No |

## CONVENTIONS

- **CPPTLMBRIDGE_VERSION**: 任何接口签名变更必须先 bump 版本号 (当前 2), 再通知 CppTLM rebase
- **`cudaStream_t` 宽度守卫**: `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` 在 `cpptlm_bridge.h` 中 — 防止 stream_id 字段隐式截断
- **G-D4 静态断言**: `PipelineId` (6 端点) + `TcPrecision` (6 端点) + 3 个 `std::is_same_v` 签名守卫 — 跨 .so 边界防 silent enum/签名漂移
- **`g_cpptlm_bridge` 全局指针**: nullptr = 独立模式 (字节级兼容原同步路径); 非 null = CppTLM 异步路径; 生命周期通过 `cpptlm_attach_bridge` / `cpptlm_detach_bridge` 管理
- **`cpptlm_set_driver` 弱符号**: PTX-EMU 提供 `__attribute__((weak))` 空实现, CppTLM 强定义覆盖 — 无 CppTLM 时安全 no-op

## ANTI-PATTERNS

- ❌ **不要向 `cpptlm_bridge.h` 添加 CppTLM 头文件 include** — 接口零外部依赖, CppTLM 侧通过 `ExternalProject_Add` 引用此单头文件
- ❌ **不要静默 bump `CPPTLMBRIDGE_VERSION`** — 必须通知 CppTLM 同步, 否则 CI 编译期断言失败
- ❌ **不要在 `cuda_driver.h` 中硬编码 `GLOBAL_SIZE`** — 由 `SimpleMemory` 实例化时传入, 见 `SimpleMemoryAllocator` 的 `init(size_t)` 设计
- ❌ **不要在 `cudart_intrinsics.h` 中重复定义 CUDA SDK 已有类型** — 5 层 `#ifndef` 守卫 ( `__CUDA_RUNTIME_H__` / `__DRIVER_TYPES_H__` 等) 防止与真实 CUDA 头文件冲突
- ❌ **不要修改 `cudaStream_t` 定义而不更新 `static_assert`** — 宽度变化影响 bridge 的 `stream_id` 字段