# Design: fix-barrier-dynamic-participation-mask

## 现状问题

`bar.warp.sync` 的问题（从调试日志分析）：

```
[DEBUG] Cycle 126: SM 0 Warp 0 PC=20 [FFFF0000] bar.warp.sync.b32 1, 0;
// lanes 16-31 (mask 0xFFFF0000) 执行 barrier
[DEBUG] Lane 16 blocked at bar.warp.sync (arrived=1/16)
...
[DEBUG] Lane 31 blocked at bar.warp.sync (arrived=16/16)
[INFO] bar.warp.sync: Barrier complete, releasing 16 threads to PC=21
// 但 lanes 0-15 (mask 0000FFFF) 根本不在这个 barrier！
// 它们在 PC=25 执行完全不同的代码路径
```

**根本原因**:

1. **动态 mask 计算错误** (`barrier.cpp:129-140`)
   - 代码用 `warp_state.threads[i].pc == current_pc` 判断哪些线程"参与"
   - 这个逻辑只检查线程是否在同一个 PC，没有考虑**实际执行了 bar.warp.sync 指令**
   - 分叉时 lanes 0-15 在 PC=25，lanes 16-31 在 PC=20，只有到达 PC=20 的线程参与

2. **Wbar::is_complete() 判断不准确**
   - 当 `participation_mask = 0xFFFF0000` (期望 16 个线程)
   - `arrived_mask = 0xFFFF0000` (实际到达 16 个)
   - 计算 `(0xFFFF0000 & 0xFFFF0000) == 0xFFFF0000` → true
   - 但问题是：静态 mask 本身就错了

3. **静态 mask vs 动态 mask 混淆**
   - `bar.warp.sync` operand 是**静态 participation mask**
   - 分叉的 warp 中，只有到达 barrier 指令 PC 的线程才真正参与

## 目标状态

**正确的动态 mask 计算**:

```cpp
// 在 barrier 处理时
void handle_bar_warp_sync(ThreadContext* thread, int barId) {
    int current_pc = thread->get_pc();

    // 找到所有实际到达这个 barrier PC 的线程
    uint32_t dynamic_mask = 0;
    for (int lane = 0; lane < 32; lane++) {
        if (warp_state.threads[lane].pc == current_pc &&
            warp_state.threads[lane].exec_mask & (1 << lane)) {
            dynamic_mask |= (1 << lane);
        }
    }

    // 用 dynamic_mask 初始化或更新 wbar
    // ...
}
```

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| barrier.cpp | 修改 | wbar 处理逻辑 |
| wbar.h | 修改 | Wbar 结构 |
| sm_context.cpp | 潜在影响 | 调用方可能需要调整 |

## 实现步骤

1. **修改动态 mask 计算**: 只统计实际到达 barrier PC 的线程
2. **修正 is_complete()**: 使用动态计算的 mask
3. **添加调试日志**: 验证 mask 计算正确性
4. **更新测试**: 添加分叉场景的 barrier 测试

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| 现有 barrier 测试失败 | 高 | 中 | 提前检查测试覆盖，准备调整预期 |
| 性能轻微下降 | 低 | 低 | 动态计算有开销，但正确性优先 |