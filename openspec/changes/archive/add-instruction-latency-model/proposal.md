# Proposal: add-instruction-latency-model

## Why

当前 PTX-EMU 对所有指令使用相同的 latency（1 cycle），无法准确建模：
- `ld.global` 等长延迟指令（~100 cycles）
- `mul` 等多周期指令（~4 cycles）
- 指令调度优化（当 Lowest PC 路径 blocked 时切换到其他 Ready 组）

这导致性能建模不准确，且无法实现真正的 Independent Thread Scheduling (ITS)。

## What

引入指令延迟模型，为每个 PTX 指令定义真实的 latency 值：

```cpp
struct InstructionAttributes {
    int latency;           // 执行周期数
    bool is_long_delay;    // 是否为长延迟指令（yield candidate）
};

// 示例 latency 值
ld.global    → latency = 100
st.global    → latency = 1
add          → latency = 1
mul          → latency = 4
bar.sync     → latency = 1 (同步点，不阻塞)
bra          → latency = 1
```

## Capabilities

1. **精确 latency 定义**: 在 `ptx_op.def` 中为每个指令定义 latency
2. **Blocked 状态建模**: 长延迟指令标记 lane 为 `is_blocked`
3. **ITS 近似支持**: 当 Lowest PC 路径 blocked 时，调度器可切换到其他 Ready 组
4. **性能分析**: 准确的 cycle 计数和性能建模

## Impact

| 文件 | 影响类型 |
|------|---------|
| `include/ptx_ir/ptx_op.def` | 添加 latency 属性 |
| `src/ptxsim/core/sm_context.cpp` | 修改调度器支持 blocked 检测 |
| `src/ptxsim/instructions/*.cpp` | 每个 handler 处理长延迟指令 |
| `include/ptx_ir/statement_context.h` | 添加 latency 查询接口 |

## References

- ADR-0014: Independent Thread Scheduling (Phase 2)
- Skill: ptx-instruction-pipeline
- Skill: ptx-debug