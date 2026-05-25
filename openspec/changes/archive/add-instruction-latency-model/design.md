# Design: add-instruction-latency-model

## 现状问题

当前实现中所有指令执行时间相同（1 cycle），这导致：
1. `ld.global` (100 cycle latency) 被当作 1 cycle 处理
2. 性能建模完全不可信
3. 无法实现 ITS（Independent Thread Scheduling）

## 目标状态

引入指令延迟模型：

```cpp
// 在 InstructionAttributes 或 ptx_op.def 中定义
struct InstructionLatency {
    int cycles;           // 执行周期数
    bool is_blocking;     // 是否会阻塞其他线程
};

// 典型指令 latency
ld.global    → 100 cycles, is_blocking=true
st.global    → 1 cycle,   is_blocking=false
add          → 1 cycle,   is_blocking=false
mul          → 4 cycles,   is_blocking=false
bar.sync     → 1 cycle,   is_blocking=false (同步点)
bra          → 1 cycle,   is_blocking=false
```

## 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| ptx_op.def | 修改 | 添加 latency X-Macro |
| sm_context.cpp | 修改 | exe_once() 添加 blocked 检测 |
| InstructionHandlers | 修改 | 长延迟指令标记 lane blocked |
| ThreadContext | 潜在修改 | 添加 is_blocked 状态 |

## 实现步骤

### Phase 1: 定义 Latency 表

1. 在 `ptx_op.def` 添加 latency 定义
2. 创建 `instruction_latency.h` 工具表

### Phase 2: 修改调度器

1. 在 `exe_once()` 中添加 blocked 检测
2. 选择最低 PC 的非 blocked 组
3. 如果所有组 blocked，选择 Lowest PC（被动等待）

### Phase 3: 长延迟指令处理

1. `ld.global` 等长延迟指令执行后标记 lane 为 blocked
2. 每个 cycle 检查 blocked 状态并更新

## 风险与缓解

| 风险 | 可能性 | 影响 | 缓解 |
|------|--------|------|------|
| 测试大量失败 | 中 | 高 | 先在小范围验证，逐步推广 |
| 性能下降 | 低 | 低 | 这是正确行为，非回归 |