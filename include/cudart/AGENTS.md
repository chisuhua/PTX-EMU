# CUDA Runtime Public Headers

**Parent**: [AGENTS.md](../../AGENTS.md)

## OVERVIEW
Public headers for fake CUDA runtime library (`libcudart.so`).

## KEY FILES
| File | Purpose |
|------|---------|
| `cpptlm_bridge.h` | **ABI 真值源** — CppTLM ↔ PTX-EMU 桥接接口（5 虚方法）|
| `cuda_driver.h` | `CudaDriver` 单例（全局内存池管理）|
| `cudart_intrinsics.h` | CUDA 类型定义（`cudaStream_t` / `cudaError_t` 等）|
| `cudart_sim.h` | CUDA runtime 入口声明 |
| `simple_memory_allocator.h` | 简单内存分配器 |

## `cpptlm_bridge.h` ABI 管理

### 版本控制
- `CPPTLMBRIDGE_VERSION` 宏（当前 = 1）
- 每次接口签名变更必须 bump 版本号
- CppTLM 端 `MemoryBridge::version()` 必须返回相同值

### Bump 流程
1. 修改 `cpptlm_bridge.h` 接口签名
2. bump `CPPTLMBRIDGE_VERSION`（如 1 → 2）
3. commit + 通知 CppTLM（HSK-1 重新发出）
4. CppTLM 通过 `ExternalProject_Add` 拉取新 commit hash → 同步 rebase

### 编译期约束
- `static_assert(sizeof(cudaStream_t) <= sizeof(uint64_t))` — 防止 stream_id 截断
- `cudaStream_t` 类型与 `cudart_intrinsics.h` 保持一致（`void*`）

## ANTI-PATTERNS
- DO NOT add CppTLM includes to `cpptlm_bridge.h`（零 CppTLM 依赖）
- DO NOT change `cudaStream_t` typedef without checking `cudart_intrinsics.h`
- DO NOT bump `CPPTLMBRIDGE_VERSION` without notifying CppTLM
