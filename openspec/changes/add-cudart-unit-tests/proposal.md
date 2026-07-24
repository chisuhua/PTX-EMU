## Why

`src/cudart/cudart_sim.cpp`（933 行）是 PTX-EMU 的 CUDA Runtime 拦截入口，包含 `cudaLaunchKernel`、`__cudaRegisterFatBinary`、`cudaStreamSynchronize` 等核心函数。当前这些入口函数零直接单元测试——仅通过 End-to-End kernel 编译执行间接验证。E2E 测试耗时长，定位问题时需要等待完整 kernel 链路执行，无法快速获得红/绿反馈。本次为 cudart 核心入口函数添加直接单元测试，提升回归速度和问题定位效率。

来源：Debt Audit P0-C2 (2026-07-02)

## What Changes

- 为 `src/cudart/cudart_sim.cpp` 核心函数创建专用单元测试文件 `tests/unit/cudart/test_cudart_sim.cpp`
- 至少覆盖 `cudaLaunchKernel` 和 `cudaStreamSynchronize` 两个核心入口
- 使用 Catch2 测试框架，不依赖真实 CUDA 设备
- 不修改 production 代码，不重构 `cudart_sim.cpp`

## Capabilities

### New Capabilities
- `cudart-unit-tests`: 为 cudart 核心入口函数提供直接单元测试覆盖，包括正常路径和错误路径

### Modified Capabilities
<!-- No existing specs need modification — this is pure test coverage addition -->

## Impact

- **新增文件**: `tests/unit/cudart/test_cudart_sim.cpp`
- **构建**: 更新 `tests/unit/cudart/CMakeLists.txt` 添加 `add_catch_test` 目标
- **无生产代码修改**: `src/cudart/`、`include/` 均不受影响
- **ADR 关联**: Debt Audit 2026-07-02 §P0-C2