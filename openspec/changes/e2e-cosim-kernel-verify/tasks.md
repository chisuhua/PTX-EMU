# Tasks: e2e-cosim-kernel-verify — CUDA VectorAdd E2E 协同仿真验证

> **Status**: Proposed
> **Parent**: cpptlm-p1-ptxemu-shim (archived 2026-07-19)
> **Ref**: ADR-0021 §2026-07-19 Postmortem

## 1. CUDA kernel 编写

- [ ] 1.1 创建 `tests/e2e/cosim/kernel_vector_add.cu` — vectorAdd kernel（N 个元素，每线程处理一个元素）
- [ ] 1.2 kernel 使用 `__global__` 声明，接受 `float* A, float* B, float* C, int N`

## 2. E2E 测试编写

- [ ] 2.1 创建 `tests/e2e/cosim/test_cosim_vector_add.cpp` — Catch2 测试
- [ ] 2.2 `TEST_CASE("cosim e2e: vectorAdd via bridge path")` 
- [ ] 2.3 设置 `g_cpptlm_bridge` mock/real bridge
- [ ] 2.4 `cudaMalloc` + `cudaMemcpy` H→D
- [ ] 2.5 `cudaLaunchKernel` → 触发 bridge 路径
- [ ] 2.6 `cudaDeviceSynchronize` → poll + exe_once 循环
- [ ] 2.7 `cudaMemcpy` D→H → 验证 golden value
- [ ] 2.8 `is_kernel_complete(kernel_id)` 断言

## 3. 构建集成

- [ ] 3.1 修改 `tests/e2e/CMakeLists.txt` — 条件编译（`BUILD_LIB_CPPTLM_CUDART=ON`）
- [ ] 3.2 注册 `e2e_cosim_vector_add` 测试目标，标签 `e2e;cosim;cpptlm`

## 4. 验证

- [ ] 4.1 `BUILD_LIB_CPPTLM_CUDART=ON` 构建 + 测试 PASS
- [ ] 4.2 `BUILD_LIB_CPPTLM_CUDART=OFF` 构建 + 测试 SKIP
- [ ] 4.3 ctest 全量无回归

## 验收门

- [ ] **G-E1** [编译] `cmake --build build -j$(nproc)` PASS（`BUILD_LIB_CPPTLM_CUDART=ON`）
- [ ] **G-E2** [测试] `ctest -R e2e_cosim_vector_add -V` PASS — golden value 匹配
- [ ] **G-E3** [回归] `ctest --output-on-failure` 全量 PASS
- [ ] **G-E4** [OFF] `BUILD_LIB_CPPTLM_CUDART=OFF` 时测试 SKIP