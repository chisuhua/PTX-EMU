# cudart-unit-tests

## Purpose

为 `src/cudart/cudart_sim.cpp` 核心入口函数提供直接单元测试覆盖，使回归检测无需完整 kernel 执行链路，提升问题定位效率。来源：Debt Audit P0-C2 — cudart_sim.cpp 933 行核心入口零直接单元测试。

## Requirements

### Requirement: cudart core function unit test coverage
The system SHALL provide direct unit tests for cudart_sim.cpp core entry functions, enabling rapid regression detection without full kernel execution.

#### Scenario: cudaLaunchKernel registers kernel correctly
- **WHEN** `cudaLaunchKernel` is called with a valid kernel function pointer and launch parameters
- **THEN** the kernel SHALL be registered in the GPUContext kernel registry
- **THEN** the kernel SHALL be launchable via subsequent stream synchronization

#### Scenario: cudaLaunchKernel rejects null kernel
- **WHEN** `cudaLaunchKernel` is called with a null function pointer
- **THEN** the call SHALL return `cudaErrorInvalidDeviceFunction`

#### Scenario: cudaStreamSynchronize returns success after launch
- **WHEN** `cudaStreamSynchronize` is called after a successful `cudaLaunchKernel`
- **THEN** it SHALL return `cudaSuccess`