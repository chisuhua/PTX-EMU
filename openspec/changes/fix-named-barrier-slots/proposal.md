## Why

ADR-0008 明确了 16 个 named barrier 槽的设计，但当前实现仅使用 `wbars[0]`，未扩展到全部 16 个槽位。CUTLASS 的 `HardwareMaxNumNamedBarriers = 16` 约束表明实际硬件支持完整 16 槽。缺少多槽支持会阻塞需要独立 barrier 组的 CUDA kernel（如多阶段 pipeline 同步），也与 Blackwell tcgen05 的 warp-group barrier 场景不兼容。

## What Changes

- 将 `WarpContext::wbars[]` 从 1 个扩展到 16 个槽位
- 修改 `barrier.cpp` 中 `BarWarpSyncHandler` 的分发逻辑，支持 `bar.sync` / `bar.arrive` 按 `bar_id` 路由到正确槽
- 更新 `WarpBarrier::init()` 接受槽索引参数
- 为多槽并发 barrier 添加集成测试（integration 级别）
- 更新 `barrier.cpp` 中遗留的 `current_wbar_id` 逻辑

## Capabilities

### New Capabilities
- `multi-barrier-slot`: 支持 16 个独立 named barrier 槽，每个槽独立管理参与掩码和状态

### Modified Capabilities
<!-- No spec-level behavior changes — this is an internal implementation expansion -->

## Impact

- `include/ptxsim/warp_barrier.h`: 无需修改接口（`wbars[16]` 替换 `wbars[1]`）
- `src/ptxsim/core/`: WarpContext 中 `wbars[]` 数组大小变更
- `src/ptxsim/barrier.cpp`: BarWarpSyncHandler 从硬编码 `wbars[0]` 改为按 `bar_id` 分发
- 所有现有 barrier 测试必须无回归（单槽场景不变）
- 与 CppTLM bridge 无交互（barrier 状态在模拟器内部管理）