# src/cudart/ AGENTS.md
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents cudart-specific content.

## OVERVIEW

Fake `libcudart.so` — LD_PRELOAD interception of CUDA runtime API, forwarding to PTX-EMU's instruction-level simulator. No real GPU required. Pure synchronous execution model (no CppTLM bridge since cleanup-cudart-cpptlm-bridge-coupling).

## STRUCTURE

```
src/cudart/
├── cudart_sim.cpp           # 35+ CUDA runtime entry points (sync-only)
├── ptx_interpreter.h/.cpp   # PTX parsing + kernel launch dispatch
├── cuda_driver.cpp           # CudaDriver singleton: device memory pool
├── simple_memory_allocator.cpp # Offset-based bump + free-list allocator
└── cpptlm_module.cpp         # libptxemu_device.so ABI (ptxemu_image_*) — NOT part of libcudart.so
```

## KEY FILES

| File | Role | Lines | Key Symbols |
|------|------|-------|-------------|
| `cudart_sim.cpp` | CUDA runtime intercept (sync-only) | ~1330 | `__cudaRegisterFatBinary`, `cudaLaunchKernel`, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaStreamCreate`, `cudaEventCreate`, `cudaGetDeviceProperties` |
| `ptx_interpreter.cpp` | PTX kernel dispatch | ~613 | `launchPtxInterpreter`, `prepareKernelLaunchRequest`, `funcInterpreter`, `setupLabels`, `setupConstantSymbols` |
| `cuda_driver.cpp` | Device memory pool | ~129 | `CudaDriver::instance()` (singleton), `malloc`, `free`, `malloc_managed`, `get_global_pool` |
| `cpptlm_module.cpp` | `libptxemu_device.so` ABI implementation | ~290 | `ptxemu_image_load/execute/unload/kernel_name/module_version/...` (8 extern "C" symbols) |

## CONVENTIONS

- **`extern "C"`**: All CUDA API entry points use `extern "C"` linkage — exact signature match required for LD_PRELOAD interception
- **`__cudaRegisterFatBinary` 单次调用**: `SingletonGuard` 检测重复初始化，FATAL abort（D-PTX-2）
- **同步执行模型**: `cudaLaunchKernel` 同步执行并通过 `g_gpu_context->wait_for_completion()` 等待完成; `cudaStreamSynchronize`/`cudaDeviceSynchronize` 立即返回 (kernel 已在 launch 时完成)
- **`CudaDriver` 单例**: `instance()` 静态局部变量, 内部 `std::mutex` 保护 `malloc`/`free` 路径
- **`PTX_DEBUG_*` 日志**: 每个入口点必须记录调用参数（`PTX_DEBUG_EMU`, `PTX_DEBUG_MEM`, `PTX_DEBUG_EXEC`）
- **`g_active_streams`**: stream 生命周期跟踪 (cudaStreamCreate insert / cudaStreamDestroy erase), 无锁 (insert/erase 对称)

## ANTI-PATTERNS

- ❌ **不要实现真实 CUDA 设备代码** — 所有 device 操作通过 `GPUContext::exe_once()` 循环模拟执行
- ❌ **不要假设线程安全** — `CudaDriver` 使用 `std::mutex` 保护共享状态
- ❌ **不要跳过 `SingletonGuard`** — 多实例仿真中单例重复初始化导致静默状态损坏
- ❌ **不要修改 `cudaStream_t` 定义而不更新 `static_assert`** — 宽度变化影响 stream_id 字段（见 `include/cudart/abi_guards.h`）
- ❌ **不要绕过 `CudaDriver` 直接管理全局内存** — 所有 device memory 分配/释放必须通过 `CudaDriver::instance()`
- ❌ **不要重新引入 `g_cpptlm_bridge` / `cpptlm_set_driver` / `StubBridge`** — CppTLM bridge 已移除 (cleanup-cudart-cpptlm-bridge-coupling), co-simulation 走 `libptxemu_device.so` ABI
