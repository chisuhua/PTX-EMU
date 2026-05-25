# Proposal: fix-bug-simt-001-lowest-pc-scheduling

## Why

BUG-SIMT-001: `sm_context.cpp` 的 divergent path 用 `for` 循环在单个 cycle 内执行所有 PC 组，导致：
- Warp divergence 无时间代价（违背 SIMT 串行执行约束）
- Cycle 计数不准确
- 性能建模不可信

当前代码在 `exe_once()` 中对所有 PC 组执行循环，违反了 SIMT 模型的串行执行原则。

## What

修复 `sm_context.cpp` 中的 divergent path 调度，采用 **Lowest PC first** 策略：
- 每个 cycle 只执行一个 PC 组（选择最低 PC 的组）
- 阻止 warp 在单个 cycle 内完成多个分支路径的执行
- 正确建模 divergence 的性能开销

## Capabilities

1. **单 PC 组执行**: 每个 cycle 只选择一个 PC 组执行，而非遍历所有
2. **Lowest PC 优先调度**: 选择最低 PC 值对应的 lanes 执行
3. **Cycle 计数准确**: 每次调度操作计为 1 cycle
4. **性能建模**: divergence 现在有真实的性能代价

## Impact

| 文件 | 影响类型 |
|------|---------|
| `src/ptxsim/core/sm_context.cpp:219-257` | 修改 exe_once() divergent path |
| `src/ptxsim/core/warp_context.cpp` | 可能需要调整同步逻辑 |
| `tests/` (SIMT divergence 相关) | 更新测试断言 |

## References

- ADR-0014: Independent Thread Scheduling
- Skill: ptx-instruction-pipeline
- Skill: ptx-debug