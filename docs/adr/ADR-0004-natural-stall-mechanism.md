# ADR-0004: 自然停顿机制 is_warp_ready_to_fetch

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-04 |
| **关联任务** | T11.2.4 |
| **作者** | PTX-EMU Team |

## 上下文

在 SIMT 执行模型中，warp scheduler 需要决定下一个 fetch 哪个 warp。之前的问题是：

- Scheduler 只检查 warp 是否有活跃线程，不检查 warp 是否"准备好 fetch"
- 如果一个 warp 的指令尚未 commit（pc != next_pc），scheduler 仍然可能选择它
- 这导致 PC 状态不一致时就被 fetch，产生错误行为

真实 GPU 的行为是：指令必须在 pipeline 中完成 commit 后，warp 才能被选中 fetch 下一条指令。这是一个自然的停顿（stall）机制。

## 决策驱动因素

1. **SIMT 硬件一致性**：真实 GPU 的 warp scheduler 不会选择尚未完成当前指令的 warp
2. **PC 状态保护**：在 commit_pc 完成前，warp 不应被选中
3. **调度器公平性**：需要一种机制让 scheduler 知道哪些 warp "准备好"

## 考虑的替代方案

### 方案 A: 在 scheduler 中直接检查 pc == next_pc

**描述**: scheduler 选择 warp 时直接遍历线程检查 PC 状态

**优点**:
- 简单直接

**缺点**:
- scheduler 需要了解 PC 内部细节，破坏封装
- 检查逻辑散落在多个 scheduler 实现中
- 难以复用和测试

### 方案 B: 添加 WarpContext::is_warp_ready_to_fetch() (✅ 选中)

**描述**: 在 WarpContext 上提供统一的 readiness 检查接口

**优点**:
- 封装良好，scheduler 只需调用一个函数
- 检查逻辑集中在 WarpContext，易于维护和测试
- 语义清晰：warp 是否准备好 fetch 下一条指令

**缺点**:
- 每次检查需要遍历 warp 中所有线程（32 次检查，性能可接受）

**选择理由**: 封装性和可维护性优先。32 次布尔检查的开销在 scheduler 上下文中可忽略。

### 方案 C: 使用状态机标记 warp 状态

**描述**: WarpContext 维护一个 ready/busy 状态标志

**优点**:
- 检查 O(1)，无需遍历

**缺点**:
- 需要在所有 PC 操作后维护状态标志，容易遗漏
- 状态标志可能与实际 PC 状态不一致
- 增加复杂性

## 决策内容

### 设计原则

1. **封装检查逻辑**：readiness 检查由 WarpContext 负责
2. **基于事实判断**：检查 pc == next_pc 的实际状态，而非维护额外标志
3. **调度器集成**：所有 scheduler 实现都集成 readiness 检查

### 实现要点

```cpp
// WarpContext 中的 readiness 检查
bool WarpContext::is_warp_ready_to_fetch() const {
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active) continue;
        // 如果任何活跃线程的 pc != next_pc，说明指令尚未 commit
        if (warp_state.threads[i].pc != warp_state.threads[i].next_pc) {
            return false;
        }
    }
    return true;
}

// Scheduler 集成（RoundRobin 示例）
for (int i = 0; i < max_warps; i++) {
    current_warp = (current_warp + 1) % max_warps;
    if (!warps[current_warp].is_active()) continue;
    if (!warps[current_warp].is_warp_ready_to_fetch()) continue;  // 新增
    return current_warp;
}
return -1;  // 没有准备好的 warp
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/warp_context.h` | 修改 | 添加 is_warp_ready_to_fetch() 声明 |
| `src/ptxsim/core/warp_context.cpp` | 修改 | 实现 readiness 检查 |
| `src/ptxsim/core/warp_scheduler.cpp` | 修改 | RoundRobin 和 Greedy 集成检查 |

## 后果

### 正面影响

- 防止在指令未 commit 时 fetch 新指令
- 与真实 GPU 的 stall 行为一致
- 调度器实现简洁（调用单一函数）

### 负面影响

- 每个 scheduler 选择 warp 时需要额外遍历 32 个线程
- 如果 warp 数量很多，轮询所有 warp 的开销增加

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| readiness 检查遗漏某些线程状态 | 低 | 高 | 回归测试覆盖多 warp 场景 |
| 所有 warp 都未准备好时死锁 | 低 | 高 | 确保 commit_pc 总是被调用，warp 最终会 ready |

## 合规检查

后续相关开发应检查：

- [ ] 所有 scheduler 实现都集成 is_warp_ready_to_fetch() 检查
- [ ] 不在 commit_pc 完成前手动修改 warp 的 ready 状态
- [ ] 新增 warp-level 操作时考虑对 readiness 的影响

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-04 | 初始版本 | PTX-EMU Team |

## 参考

- [ADR-0003: commit_pc / force_set_pc 分离](./ADR-0003-commit-pc-pattern.md)
- [GPU Pipeline 调研](../archive/misc/GPU-PIPELINE-RESEARCH.md)
