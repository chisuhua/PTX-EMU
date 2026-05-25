# Design: fix-bug-simt-001-lowest-pc-scheduling

## 现状问题

`sm_context.cpp` 的 `exe_once()` 方法处理 divergent warp 时存在问题：

```cpp
// 当前实现 (问题代码)
} else if (!lanes_by_pc.empty()) {
    // 问题：对所有 PC 组执行循环，在一个 cycle 内完成所有分支
    for (const auto& [pc, lanes] : lanes_by_pc) {
        // 执行每一组的指令
        execute_pc_group(pc, lanes);
    }
}
```

**问题分析**:
1. `for` 循环在一个 cycle 内执行所有 PC 组
2. 违背 SIMT 串行执行约束
3. Cycle 计数不准确（一个 cycle 完成多个分支）
4. 性能建模不可信

## 目标状态

每个 cycle 只执行一个 PC 组：

```cpp
// 目标实现
} else if (!lanes_by_pc.empty()) {
    // 选择最低 PC 的组
    auto it = lanes_by_pc.begin();
    int selected_pc = it->first;
    const auto& selected_lanes = it->second;

    // 执行选中的 PC 组（只执行一个组）
    execute_pc_group(selected_pc, selected_lanes);
}
```

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| sm_context.cpp | 修改 | exe_once() divergent path 逻辑 |
| warp_context.cpp | 潜在影响 | PC 同步机制可能需要调整 |
| 测试用例 | 更新 | SIMT divergence 测试的预期行为 |

## 实现步骤

1. **修改 exe_once() divergent path**: 将 for 循环改为选择最低 PC 组
2. **验证 cycle 计数**: 确保每次调度只计 1 cycle
3. **更新测试**: 调整 SIMT divergence 测试的预期
4. **运行 sanity.sh**: 确保无回归

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| 现有测试失败 | 中 | 中 | 提前检查测试覆盖，准备调整 |
| 性能下降明显 | 低 | 中 | 这是预期行为，表示正确建模 |