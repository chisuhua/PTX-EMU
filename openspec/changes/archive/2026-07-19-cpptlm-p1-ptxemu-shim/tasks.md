# Tasks: cpptlm-p1-ptxemu-shim — PTX-EMU 端 PtxEmuDriverShim + Co-sim Seam 闭合

> **Status**: Proposed（2026-07-18）
> **Parent change**: CppTLM `cpptlm-d1-p1-pipeline-scoreboard`（Phase 4 Wave 0）
> **Cross-project reference**: `CppTLM/openspec/changes/cpptlm-d1-p1-pipeline-scoreboard/tasks.md §4.0.1-4.0.4`
> **CppTLM 接口**: `include/tlm/gpu/ptx_emu_driver.hh`（通过 `ExternalProject_Add` `cpptlm-install/include/` 可用）
> **Vendor 接口**: `include/cudart/{scoreboard,pipeline,tensor_core}_interface.h`（已 vendor, commit `c16ff97` / `09b64b6`）

## 1. PtxEmuDriverShim 头文件

- [ ] 1.1 创建 `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` — 类声明 + 成员（`GPUContext* ctx_` / `completion_map_` / `vector<unique_ptr<IScoreboard>>` 等）
- [ ] 1.2 创建 `src/cudart/cpptlm_bridge/PtxEmuDriverShim.cpp` — `advance()` 循环调用 `exe_once()` + EXIT detection + error handling
- [ ] 1.3 实现 `inject_scoreboard()`: `reset()` + `set_scoreboard()` + ownership transfer
- [ ] 1.4 实现 `inject_pipeline()` / `inject_tensor_core()`: 转发到 `SMContext` setter + ownership
- [ ] 1.5 实现 `is_kernel_complete()` / `mark_complete()` / `num_sms()`

**NOTE**: `IPtxEmuDriver` 接口来自 CppTLM `include/tlm/gpu/ptx_emu_driver.hh`（通过 `ExternalProject_Add` `cpptlm-install/include/` 可达）。**MUST NOT** 直接 include CppTLM 仓库 `.cc` 或非头文件。

## 2. cpptlm_set_driver ABI 入口

- [ ] 2.1 修改 `include/cudart/cpptlm_bridge.h`：新增 `namespace tlm { class IPtxEmuDriver; }` forward declare + `extern "C" PTXEMU_BRIDGE_API void cpptlm_set_driver(tlm::IPtxEmuDriver* driver);`
- [ ] 2.2 bump `CPPTLMBRIDGE_VERSION` 1 → 2
- [ ] 2.3 `src/cudart/cudart_sim.cpp`：新增全局 `tlm::IPtxEmuDriver* g_ptx_emu_driver = nullptr;` + `cpptlm_set_driver()` 实现

## 3. Bridge Path Kernel 入队

- [ ] 3.1 `src/cudart/cudart_sim.cpp` `cudaLaunchKernel` bridge 路径：在 `g_cpptlm_bridge->submit_kernel()` 后，构建 `KernelLaunchRequest` 并通过 `g_gpu_context->submit_kernel_request()` enqueue
- [ ] 3.2 统一 `kernel_id`：bridge 路径与 `KernelLaunchRequest` 使用同一 `kernel_id`（确保 `on_complete` 回调查询一致性）
- [ ] 3.3 args 深拷贝单次化：bridge 路径完成深拷贝后 `std::move` 到 `KernelLaunchRequest`，消除双重深拷贝
- [ ] 3.4 `on_complete` 回调设置：`[kernel_id](){ if (g_ptx_emu_driver) g_ptx_emu_driver->mark_complete(kernel_id); }`

**NOTE**: bridge 路径原有逻辑（`g_cpptlm_bridge->submit_kernel()` + `g_pending_kernels` 注册）**MUST** 保持不变。新增的是**追加** enqueue 到 `GPUContext::task_queue`，不是替换。

## 4. initialize_environment() 集成

- [ ] 4.1 `src/cudart/cudart_sim.cpp` `initialize_environment()`：`g_gpu_context` 创建后，构造 `PtxEmuDriverShim` 并调用 `cpptlm_set_driver(shim)`
- [ ] 4.2 `PTX_DEBUG_EMU` 日志：driver 创建成功/失败日志 + 生命周期跟踪

## 5. 构建修复

- [ ] 5.1 新增 `src/cudart/cpptlm_bridge/CMakeLists.txt`：编译 `PtxEmuDriverShim.cpp` 为 `cpptlm_bridge` 对象库
- [ ] 5.2 修改 `CMakeLists.txt`：`ExternalProject_Add(cpptlm ...)` 添加 `-DCMAKE_POSITION_INDEPENDENT_CODE=ON`
- [ ] 5.3 修改 `CMakeLists.txt`：`ExternalProject_Add(cpptlm ...)` pin `GIT_TAG 73e5422`
- [ ] 5.4 修改 `CMakeLists.txt`：`include(cpptlm_bridge)` 子目录 + 链接到 `cudart` 目标

## 6. 验证

- [ ] 6.1 构建通过：`cmake -S . -B build -DBUILD_LIB_CPPTLM_CUDART=ON -DCMAKE_BUILD_TYPE=Release && cmake --build build -j$(nproc)`
- [ ] 6.2 编译期验证：CppTLM `cpptlm_core` 已启用 PIC（检查 `build/cpptlm-install/lib/libcpptlm_core.a` 位置）
- [ ] 6.3 ABI 符号验证：`nm build/lib/libcudart.so | grep cpptlm_set_driver` 确认符号导出且可见
- [ ] 6.4 `ctest --output-on-failure` 全量回归 PASS
- [ ] 6.5 功能验证：确认 bridge 路径+非bridge路径均无退化

## 验收门

- [ ] **G-W0.1** [编译] `cmake --build build -j$(nproc)` PASS（含 `BUILD_LIB_CPPTLM_CUDART=ON`）
- [ ] **G-W0.2** [ABI] `nm` 确认 `cpptlm_set_driver` 符号导出
- [ ] **G-W0.3** [回归] `ctest` 全量 PASS（bridge 和非 bridge 路径字节级兼容）
- [ ] **G-W0.4** [集成] CppTLM 端 `KernelLaunchTLM::tick()` 可调用 PTX-EMU `GPUContext::exe_once()` 返回 `AdvanceResult::Executed`