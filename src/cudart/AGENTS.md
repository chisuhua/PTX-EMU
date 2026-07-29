# src/cudart/ AGENTS.md
**SSOT**: Common conventions (build/test/format/conventions/anti-patterns) live in root AGENTS.md; this file only documents cudart-specific content.

## OVERVIEW

Fake `libcudart.so` — LD_PRELOAD interception of CUDA runtime API, forwarding to PTX-EMU's instruction-level simulator. No real GPU required.

## STRUCTURE

```
src/cudart/
├── cudart_sim.cpp           # 35+ CUDA runtime entry points
├── ptx_interpreter.h/.cpp   # PTX parsing + kernel launch dispatch
├── cuda_driver.cpp           # CudaDriver singleton: device memory pool
├── simple_memory_allocator.cpp # Offset-based bump + free-list allocator
├── stub_bridge.h             # Zero-latency CppTLMBridge (auto-co-sim)
└── cpptlm_bridge/            # PtxEmuDriverShim: CppTLM co-simulation IP
    ├── PtxEmuDriverShim.h
    └── PtxEmuDriverShim.cpp
```

## KEY FILES

| File | Role | Lines | Key Symbols |
|------|------|-------|-------------|
| `cudart_sim.cpp` | CUDA runtime intercept | ~1473 | `__cudaRegisterFatBinary`, `cudaLaunchKernel`, `cudaMalloc`, `cudaMemcpy`, `cudaDeviceSynchronize`, `cudaStreamCreate`, `cudaEventCreate`, `cudaGetDeviceProperties` |
| `ptx_interpreter.cpp` | PTX kernel dispatch | ~613 | `launchPtxInterpreter`, `prepareKernelLaunchRequest`, `funcInterpreter`, `setupLabels`, `setupConstantSymbols` |
| `cuda_driver.cpp` | Device memory pool | ~129 | `CudaDriver::instance()` (singleton), `malloc`, `free`, `malloc_managed`, `get_global_pool` |
| `stub_bridge.h` | Zero-latency bridge stub | 59 | `StubBridge`: 5 virtual methods (submit/poll/synchronize/global_access/version) |
| `PtxEmuDriverShim.h` | CppTLM co-sim IP | 63 | `PtxEmuDriverShim`: `advance()`, `inject_*()`, `is_kernel_complete()`, `mark_complete()` |

## CONVENTIONS

- **`extern "C"`**: All CUDA API entry points use `extern "C"` linkage — exact signature match required for LD_PRELOAD interception
- **`__cudaRegisterFatBinary` 单次调用**: `SingletonGuard` 检测重复初始化，FATAL abort（D-PTX-2）
- **`g_cpptlm_bridge` 全局指针**: nullptr = 独立模式; non-null = CppTLM 异步路径; `cpptlm_attach_bridge` / `cpptlm_detach_bridge` 管理生命周期
- **`cpptlm_set_driver` 弱符号**: PTX-EMU 提供 `__attribute__((weak))` 空实现, CppTLM 强定义覆盖 — 无 CppTLM 时安全 no-op
- **`CudaDriver` 单例**: `instance()` 静态局部变量, 内部 `std::mutex` 保护 `malloc`/`free` 路径
- **`PTX_DEBUG_*` 日志**: 每个入口点必须记录调用参数（`PTX_DEBUG_EMU`, `PTX_DEBUG_MEM`, `PTX_DEBUG_EXEC`）

## ANTI-PATTERNS

- ❌ **不要实现真实 CUDA 设备代码** — 所有 device 操作通过 `GPUContext::exe_once()` 循环模拟执行
- ❌ **不要假设线程安全** — `CudaDriver` 和 `StubBridge` 使用 `std::mutex` 保护共享状态; 未来可能被 CppTLM 多线程调用
- ❌ **不要跳过 `SingletonGuard`** — 多实例仿真中单例重复初始化导致静默状态损坏
- ❌ **不要修改 `cudaStream_t` 定义而不更新 `static_assert`** — 宽度变化影响 bridge 的 `stream_id` 字段（见 `include/cudart/AGENTS.md`）
- ❌ **不要绕过 `CudaDriver` 直接管理全局内存** — 所有 device memory 分配/释放必须通过 `CudaDriver::instance()`
- ❌ **不要静默 bump `CPPTLMBRIDGE_VERSION`** — 必须通知 CppTLM 同步, 否则 CI 编译期断言失败