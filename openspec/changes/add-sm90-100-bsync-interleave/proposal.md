# Proposal: add-sm90-100-bsync-interleave

## Summary

实现 Hopper/Blackwell (sm90/sm100) 执行模型的核心特性：BSSY/BSYNC 状态机、动态线程迁移调度器，以及三种可配置的分支分歧执行模式（sequential/interleaved/shortest_first）。

## Motivation

当前 PTX-EMU 的 warp 调度器使用简单的 "sequential" 模式（先执行 Path A，再执行 Path B），这与 NVIDIA Hopper/Blackwell 架构的动态交错执行模型不符。需要：

1. 实现 BSSY (Branch Synchronization Yield) 和 BSYNC (Branch Synchronization) 语义
2. 支持动态线程迁移 - 当一组线程在屏障等待时，调度器可切换执行另一组
3. 提供三种执行模式：sequential（当前）、interleaved（动态交错）、shortest_first（Blackwell 短路径优先）

## Scope

### 新增文件

- `include/ptxsim/bsync_state.h` - BSSY/BSYNC 状态机接口
- `src/ptxsim/core/bsync_state.cpp` - BsyncManager 实现

### 修改文件

- `src/ptxsim/instructions/barrier.cpp` - 集成 BsyncManager
- `src/ptxsim/core/warp_scheduler.cpp` - 动态交错逻辑
- `src/ptxsim/core/sm_context.cpp` - 调度策略选择
- `include/ptxsim/thread_state.h` - 完善 blocked_cycles

## Impact

- 改善 warp 分歧执行效率
- 更准确模拟现代 GPU 行为
- 为未来 sm90/100 架构支持奠定基础

## Tasks

14 tasks across 5 phases (see tasks.md)