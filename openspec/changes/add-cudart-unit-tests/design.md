## Context

当前 `src/cudart/cudart_sim.cpp` 的测试覆盖仅通过 E2E kernel 测试间接完成（`tests/e2e/kernel/` 目录下的 `.cu` 文件通过 `nvcc -ptx` 编译后由模拟器完整执行）。没有直接针对 cudart 入口函数（`cudaLaunchKernel`、`cudaStreamSynchronize`、`__cudaRegisterFatBinary`）的单元测试。

当修改 cudart 层行为时（如 CppTLM bridge 集成、新的 CUDA API 拦截），必须在完整 kernel 执行后才能验证，调试周期长。增加单元测试可以：
- 在 1 秒内验证 `cudaLaunchKernel` 的 kernel 注册逻辑
- 独立测试 `cudaStreamSynchronize` 的同步机制
- 隔离故障：cudart 层的问题不需要通过整个执行管道定位

## Goals / Non-Goals

**Goals:**
- 为 `cudaLaunchKernel`、`cudaStreamSynchronize` 创建直接单元测试
- 至少 5 个测试用例覆盖正常路径和错误路径
- 所有测试在 Catch2 框架下可独立运行（无需真实 CUDA 设备）
- 更新 CMakeLists.txt 使新测试加入 ctest 套件

**Non-Goals:**
- 不修改 `src/cudart/cudart_sim.cpp` production 代码
- 不重构 cudart 层的内部结构
- 不修改 E2E 测试逻辑
- 不覆盖所有 cudart 函数（仅核心入口）

## Decisions

- **测试框架**: 使用 Catch2（项目已有 `tests/catch_amalgamated.hpp`），与其他单元测试一致
- **测试粒度**: 函数级（`cudaLaunchKernel` 等每个入口独立测试），不做 mock/stub
- **测试隔离**: 每个测试用例独立构造 `GPUContext`/`SMContext` 环境，不共享全局状态
- **文件位置**: `tests/unit/cudart/test_cudart_sim.cpp`，遵循现有 `tests/unit/<area>/` 分类规范
- **CMake 目标命名**: `add_catch_test(unit_cudart_sim ...)`，带 `unit;` 前缀，标签 `cudart`

## Risks / Trade-offs

| Risk | Mitigation |
|------|-----------|
| cudart 函数依赖全局状态（`g_device_mem` 等），测试间可能相互干扰 | 每个 TEST_CASE 使用独立的 `GPUContext`/`SMContext` 实例，通过 `setUp`/`tearDown` 重置 |
| `cudaStreamSynchronize` 依赖异步 launch 状态，测试需要构造完整 kernel 路径 | 使用最小的 MockKernel 结构体模拟 `CUDAKernelLaunchParams`，不触发完整指令执行 |
| 测试覆盖可能不足以发现回归 | 保持与现有 E2E 测试互补——单元测试覆盖逻辑正确性，E2E 覆盖集成正确性 |
| CMake 构建配置需要单独处理 cudart 测试目标 | 在 `tests/unit/CMakeLists.txt` 中新增分支，添加 `add_catch_test` 并链接 `cudart` 库 |