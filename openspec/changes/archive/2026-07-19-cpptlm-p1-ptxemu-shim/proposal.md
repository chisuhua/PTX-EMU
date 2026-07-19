## Why

CppTLM P1 Phase 4 (D1-Full) 完成 ScoreboardTLM/PipelineTLM/TensorCoreTLM 核心模块 + `IPtxEmuDriver` 窄接口后，PTX-EMU 端需要：
1. 实现 `PtxEmuDriverShim` 作为 CppTLM 驱动的执行后端（对接 `GPUContext::exe_once()`）
2. 修复 bridge path 使 CppTLM 提交的 kernel 能同时 enqueue 到 `GPUContext` 并设置 `on_complete` 回调
3. 新增 `cpptlm_set_driver` ABI 入口实现双端驱动注册
4. 修复构建配置使 CppTLM `cpptlm_core` 可链接为 `.so`

当前 bridge path（`cudaLaunchKernel` 中 `g_cpptlm_bridge` 路径）仅存储到 `g_pending_kernels` 但不 enqueue 到 `GPUContext::task_queue`，导致 `poll_kernel` 立即返回 0（完成），CppTLM 端无法获取真实执行延迟。同时双端协同仿真缺少驱动注册的 ABI 入口。

## What Changes

### 新增
- **新增** `src/cudart/cpptlm_bridge/PtxEmuDriverShim.{h,cpp}` — `IPtxEmuDriver` 的 PTX-EMU 端实现
- **新增** `src/cudart/cpptlm_bridge/CMakeLists.txt` — cpptlm_bridge 子目录构建
- **新增** `cpptlm_set_driver` ABI 入口（`cudart_sim.cpp`，与 `cpptlm_attach_bridge` 同层级）

### 修改
- **修改** `src/cudart/cudart_sim.cpp`：
  - `cudaLaunchKernel` bridge 路径：提交 kernel 到 `GPUContext::task_queue` + 设置 `on_complete` 回调
  - `initialize_environment()`：调用 `cpptlm_set_driver` 注册 `PtxEmuDriverShim`
  - 新增 `g_ptx_emu_driver` 全局指针
- **修改** `CMakeLists.txt`：新增 cpptlm_bridge 子目录构建 + 启用 CppTLM `cpptlm_core` PIC + pin `CPPTLM_COMMIT_HASH`
- **修改** `include/cudart/cpptlm_bridge.h`：新增 `cpptlm_set_driver` 声明（bump to VERSION=2）

## Capabilities

### New Capabilities
- `ptx-emu-driver-shim`: PtxEmuDriverShim 实现 `IPtxEmuDriver` 接口 — `advance()` 驱动 `GPUContext::exe_once()`, `inject_*()` 转发到 `SMContext` setter, `completion_map_` 跟踪 kernel 完成状态
- `cpptlm-set-driver-abi`: `cpptlm_set_driver` ABI 入口 — 跨 `.so` 边界的驱动注册，`initialize_environment()` 中调用
- `bridge-path-enqueue`: bridge 路径 kernel 提交 — `cudaLaunchKernel` 同时 enqueue 到 `GPUContext::task_queue` + 设置 `on_complete` 回调
- `cpptlm-build-fix`: CppTLM 构建修复 — `cpptlm_core` PIC, pin `CPPTLM_COMMIT_HASH`, CMake export target

## Impact

| 文件 | 类型 | 影响 |
|------|------|:----:|
| `src/cudart/cpptlm_bridge/PtxEmuDriverShim.{h,cpp}` | 新增 | ~200 LOC shim 实现 |
| `src/cudart/cudart_sim.cpp` | 修改 | +50 LOC（bridge path + ABI + init） |
| `include/cudart/cpptlm_bridge.h` | 修改 | +3 LOC（`cpptlm_set_driver` 声明, VERSION=2） |
| `CMakeLists.txt` | 修改 | +10 LOC（子目录 + PIC + pin commit） |
| `src/cudart/cpptlm_bridge/CMakeLists.txt` | 新增 | ~10 LOC 构建配置 |
| **合计** | | **~270 LOC** |