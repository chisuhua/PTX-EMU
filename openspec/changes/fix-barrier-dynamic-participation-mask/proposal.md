# Proposal: fix-barrier-dynamic-participation-mask

## Why

当前 `bar.warp.sync` 实现存在 **动态 mask 计算错误**：

当 warp 分叉时（lanes 0-15 在 PC=A，lanes 16-31 在 PC=B），只有到达 barrier PC 的线程才真正参与。但当前实现使用静态 mask，导致：

1. Barrier 在只有部分线程到达时就错误地判定完成
2. 未参与 barrier 的线程被错误地释放
3. 分叉路径的同步语义被破坏

## What

修复 barrier 模块的 participation mask 计算：
- 动态计算 `participation_mask`（只包含实际到达 barrier PC 的线程）
- 修正 `Wbar::is_complete()` 判断逻辑
- 确保只有真正参与 barrier 的线程被释放

## Capabilities

1. **动态 Mask 计算**: 基于线程实际到达的 PC 计算 participation mask
2. **正确的完成判断**: 只有 `arrived_mask == participation_mask` 时才完成
3. **分叉安全**: divergence 情况下正确处理多个独立的 barrier
4. **Wbar 生命周期管理**: init → arrive → complete → reset

## Impact

| 文件 | 影响类型 |
|------|---------|
| `src/ptxsim/core/barrier.cpp` | 修改 wbar 逻辑 |
| `include/ptxsim/core/wbar.h` | Wbar 结构修改 |
| `src/ptxsim/core/sm_context.cpp` | 调用方修改 |

## References

- Skill: ptx-barrier-mechanism
- Skill: ptx-debug
- ADR-0014 (相关 ITS 问题分析)