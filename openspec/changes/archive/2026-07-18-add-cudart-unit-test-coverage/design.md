# 为 cudart_sim.cpp 添加单元测试覆盖

## Context

src/cudart/cudart_sim.cpp（1255 行）是 CUDA runtime API 拦截入口，处理 cudaLaunchKernel、__cudaRegisterFatBinary、memory/stream/event 等核心 API。当前零直接单元测试，所有回归依赖 E2E kernel 测试间接发现。自 debt-audit-2026-07-02 以来已增长 322 行。

约束: 不修改 cudart_sim.cpp 行为；测试通过 fake libcudart.so 路径或直接链接 cudart 目标。

## Goals / Non-Goals

- **Goals**: 为 Memory API（cudaMalloc/Free/Memcpy/Memset）和 Stream API（cudaStreamCreate/Synchronize）添加最小单元测试
- **Non-Goals**: 完整 API 覆盖、重构 cudart_sim.cpp、修改生产行为

## Decisions

1. **测试路径**: 直接链接 cudart 库目标（非 fake libcudart.so 加载路径）— 简单、快速、无运行时耦合
2. **测试夹具**: 最小 GPUContext + CUDA 设备初始化辅助函数放在 cudart_test_helpers.h
3. **Phase 优先级**: Memory API → Stream API（按风险排序）
4. **Mock 策略**: 最小 mock — 仅 mock GPUContext 的 return 路径，不 mock 整个执行管线

## Risks / Mitigations

- cudart_sim.cpp 的部分函数依赖全局状态（g_gpu_ctx）→ 测试夹具初始化/清理全局状态
- 不存在 stand-alone GPUContext 创建路径 → 在测试夹具中通过现有 API 构造
