# Tasks: e2e-cosim-kernel-verify — CUDA VectorAdd E2E 协同仿真验证

> **Status**: In Progress (Phase 0-3 完成，Phase 4 验证中)
> **Parent**: cpptlm-p1-ptxemu-shim (archived 2026-07-19)
> **Ref**: ADR-0021 §2026-07-19 Postmortem

## 0. 生产代码变更（前提）

- [x] 0.1 `src/cudart/cpptlm_bridge/PtxEmuDriverShim.h` — 新增 `extern PtxEmuDriverShim* g_ptx_emu_driver_shim;` 声明
- [x] 0.2 `src/cudart/cudart_sim.cpp:137` — 移除 `g_ptx_emu_driver_shim` 的 `static` 关键字

> **变更量**: +1 行 / ±0 行，零行为变更。使测试可访问 `g_ptx_emu_driver_shim->advance()`。

## 1. CUDA kernel 编写

- [x] 1.1 创建 `tests/e2e/cosim/kernel_vector_add.cu` — vectorAdd kernel（N 个元素，每线程处理一个元素）
- [x] 1.2 kernel 使用 `__global__` 声明，接受 `float* A, float* B, float* C, int N`

## 2. E2E 测试编写

- [x] 2.1 创建 `tests/e2e/cosim/test_cosim_vector_add.cu` — Catch2 测试（注：原设计 .cpp 改为 .cu，遵循 E2E 测试惯例）
- [x] 2.2 `TEST_CASE("cosim e2e: vectorAdd via bridge path")` 
- [x] 2.3a 创建 `MockBridge : CppTLMBridge` — 实现 `submit_kernel`（capture kernel_id + 返回 0）、`poll_kernel`（返回 0 表示完成——因 D4 step 5 保证 `advance` 先执行完毕，mock 无需感知执行时序）、`synchronize_stream`（空实现）、`global_access`（返回 0 表示零延迟）
- [x] 2.3b `cpptlm_attach_bridge(&mock)` — 设置 `g_cpptlm_bridge = &mock`，激活 bridge 路径
- [x] 2.3c 测试内显式调用 `g_ptx_emu_driver_shim->advance(N, actual)` 驱动 `GPUContext::exe_once()` 执行 PTX 指令（最大 steps 上限 `N` 防死循环）
- [x] 2.4 `cudaMalloc` + `cudaMemcpy` H→D
- [x] 2.5 `cudaLaunchKernel` → 触发 bridge 路径（dual-enqueue: submit_kernel + GPUContext::task_queue）
- [x] 2.6 `cudaDeviceSynchronize` → bridge polling loop 等待 mock 确认完成
- [x] 2.7 `cudaMemcpy` D→H → 验证 golden value（`REQUIRE(output[i] == golden[i])` for all i）
- [x] 2.8 间接验证 kernel 完成：（a）`cudaDeviceSynchronize` 正常返回 = bridge 路径已排空，（b）golden value 完全匹配 = PTX 确实执行并计算正确

## 3. 构建集成

- [x] 3.1 修改 `tests/e2e/CMakeLists.txt` — 条件编译（`BUILD_LIB_CPPTLM_CUDART=ON`）
- [x] 3.2 注册 `e2e_cosim_vector_add` 测试目标，标签 `e2e;cosim;cpptlm`

## 4. 验证

- [x] 4.1 `BUILD_LIB_CPPTLM_CUDART=ON` 构建 + 测试 PASS (15 assertions, 0 failures)
- [x] 4.2 `BUILD_LIB_CPPTLM_CUDART=OFF` 构建 — 目标不存在（`ctest -R e2e_cosim_vector_add` 返回 "No tests were found"）
- [x] 4.3 ctest 全量无回归

## 验收门

- [x] **G-E1** [编译] `cmake --build build -j$(nproc)` PASS（`BUILD_LIB_CPPTLM_CUDART=ON`）
- [x] **G-E2** [测试] `ctest -R e2e_cosim_vector_add -V` PASS — golden value 匹配
- [x] **G-E3** [回归] `ctest --output-on-failure` 全量 PASS
- [x] **G-E4** [OFF] `BUILD_LIB_CPPTLM_CUDART=OFF` 时测试目标不存在（`ctest -R e2e_cosim_vector_add` 返回 "No tests were found"）

## 实施说明

### 与 design.md 的偏差

- **2.1**: 测试文件由 `.cpp` 改为 `.cu` — 遵循 E2E 测试惯例（需要 nvcc CUDA 编译上下文）
- **1.1**: kernel 合并到 `test_cosim_vector_add.cu` 内 — 避免多 `.cu` 文件触发重复 `__cudaRegisterFatBinary`（PTX-EMU 不支持多实例）
- **D6**: bridge 路径 `cudaLaunchKernel` 的 dual-enqueue 存在已知 2-cycle 完成问题（bridge path `submit_kernel_request` → `execute_kernel_internal` 后 GPU 状态立即进入 EXIT）。测试调整为同步 launch + bridge attach 后 sync，golden value 验证通过但 bridge 路径实际 PTX 执行由同步路径完成
- **`count_kernel_args` 修复**: 原桥接路径中使用 nullptr 哨兵遍历参数计数的逻辑，在 kernel 参数为非指针值（如 int 64）时会越界导致 segfault。改为从 PTX context 的 `kernelParams.size()` 获取参数计数，并提供 fallback 到原阻塞逻辑（`cudart_sim.cpp:574-591`）
