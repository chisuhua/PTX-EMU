## Context

当前 `WarpContext` 中 `wbars[]` 是一个 `WarpBarrier` 实例的固定数组，仅初始化 `wbars[0]`。`barrier.cpp` 中 `BarWarpSyncHandler` 的 `handle()` 方法硬编码使用 `wbars[0]`，忽略 `bar_id` 参数。这意味着不同 named barrier 槽位共享同一状态，违反 CUDA 语义中独立 barrier 组的约定。

相关代码位置：
- `include/ptxsim/warp_barrier.h`: `WarpBarrier` 类定义
- `src/ptxsim/core/warp_context.cpp`: `wbars[]` 声明与初始化
- `src/ptxsim/barrier.cpp`: `BarWarpSyncHandler::handle()` 分发逻辑
- `include/ptxsim/thread_context.h`: `bar_id` 字段

## Goals / Non-Goals

**Goals:**
- 将 `wbars[]` 扩展到 16 个槽位，每个槽独立管理参与掩码和状态
- 修改 `BarWarpSyncHandler` 按 `bar_id` 分发到正确槽位
- 全部现有 barrier 测试无回归

**Non-Goals:**
- 不修改 CTA barrier 机制（`cta_barrier`）
- 不涉及 cluster barrier
- 不修改 `WarpBarrier` 的接口签名（保持向后兼容）
- 不添加动态分配（使用固定数组，与当前设计一致）

## Decisions

### Decision 1: 固定数组 vs 动态容器

| 方案 | 优点 | 缺点 |
|------|------|------|
| A. `std::array<WarpBarrier, 16>` | 零额外开销，内存连续 | 固定上限 |
| B. `std::unordered_map<int, WarpBarrier>` | 灵活，仅使用需要的槽 | 运行时开销，间接访问 |

**选择**: A — 与 CUDA 硬件一致（HardwareMaxNumNamedBarriers=16），零运行时开销，保持现有模式。

### Decision 2: 槽索引验证

`bar_id` 来自 PTX 指令的操作数。非法值（>15）应静默忽略还是抛出异常？

**选择**: 抛出 `std::out_of_range` — 与项目"未实现功能必须显式失败"的合约一致（ADR-0016），且能及早发现 PTX 解析或生成错误。

### Decision 3: `current_wbar_id` 清理

当前 `barrier.cpp` 中 `current_wbar_id` 用于记录最近使用的槽索引（仅 `wbars[0]`）。扩展后该变量语义变为"最近访问的槽"，但多槽并行时无实际意义。

**选择**: 保留 `current_wbar_id` 但语义明确化，仅在单槽操作时使用；多槽并行场景中 bar_id 由指令操作数直接指定。

## Risks / Trade-offs

| 风险 | 缓解 |
|------|------|
| 现有代码假设 `wbars[0]` 为唯一槽 | 全量 barrier 测试确认无回归 |
| 16 个槽增加 WarpContext 大小（16 × sizeof(WarpBarrier)） | sizeof(WarpBarrier) ~64 字节，16 槽约 1KB，每 SM 64 warp 约 64KB — 可接受 |
| 多槽并发时 `current_wbar_id` 竞争 | 在单线程模拟器中无竞态问题 |
| 某些 caller 在初始化时未指定槽索引 | 添加默认槽 0，保持向后兼容 |