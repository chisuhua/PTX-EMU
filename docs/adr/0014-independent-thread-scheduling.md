# ADR-0014: Independent Thread Scheduling (ITS) 支持

## 状态
**Proposed** — 待讨论和批准

---

## 背景

### 问题 1: BUG-SIMT-001

`sm_context.cpp` 的 divergent path 用 `for` 循环在单个 cycle 内执行所有 PC 组，导致：
- Warp divergence 无时间代价（违背 SIMT 串行执行约束）
- Cycle 计数不准确
- 性能建模不可信

修复后采用 **Lowest PC first**，每个 cycle 只执行一个 PC 组。但这只是 Block Scheduling，不是真正的 ITS。

### 问题 2: Long-Delay 指令阻塞

当 Lowest PC 路径遇到 `ld.global`（延迟 ~100 cycles）时：
- 该 PC 组所有 lane 被标记 `is_blocked`
- 如果下一 cycle 仍选 Lowest PC 且该组仍 blocked，该 cycle 浪费
- 其他已 Ready 的 PC 组（如 PC=11）无法被执行

### 问题 3: Hopper ITS 的真实能力

**重要澄清**：Hopper ITS 不是 warp 间调度，而是 **warp 内多路径线程之间的独立调度**。

```
传统 SIMT (Pre-Hopper):
  Warp 内所有线程必须同时执行同一条指令
  divergence → 串行执行不同路径

Hopper ITS:
  Warp 内不同 PC 的线程可以独立调度
  divergence → 真正并行执行不同路径
  等待长延迟的线程不妨碍其他线程继续执行
```

---

## 目标

1. **修复 BUG-SIMT-001**：单 cycle 单 PC 组执行（Lowest PC first）
2. **引入指令延迟模型**：区分 1-cycle 指令和长延迟指令
3. **实现 ITS 近似**：当 Lowest PC 路径 blocked 时，切换到其他 Ready 的 PC 组
4. **为真正 ITS 预留架构**：未来可扩展到每 lane 独立调度

---

## 技术方案

### Phase 1: 修复 BUG-SIMT-001（立即）

采用 Lowest PC first，确保每 cycle 只执行一个 PC 组。

### Phase 2: 添加指令延迟模型

```cpp
// 在 StatementContext 或新指令属性中
struct InstructionAttributes {
    int latency;           // 执行周期数
    bool is_long_delay;    // 是否为长延迟指令（yield candidate）
};

// 示例 latency 值
ld.global    → latency = 100
st.global     → latency = 1
add          → latency = 1
mul          → latency = 4
bar.sync     → latency = 1 (同步点，不阻塞)
bra          → latency = 1
```

### Phase 3: Blocked 检测和调度器增强

```cpp
// sm_context.cpp exe_once() divergent path

} else if (!lanes_by_pc.empty()) {
    int selected_pc = -1;
    const auto* selected_lanes = nullptr;

    // 策略：选择最低 PC 的非 blocked 组
    for (const auto& [pc, lanes] : lanes_by_pc) {
        bool all_unblocked = true;
        for (int lane : lanes) {
            ThreadContext* t = next_warp->get_thread(lane);
            if (t && t->is_blocked()) { all_unblocked = false; break; }
        }
        if (all_unblocked) {
            selected_pc = pc;
            selected_lanes = &lanes;
            break;  // Lowest unblocked PC
        }
    }

    // 如果所有组 blocked，选择 Lowest PC（被动等待）
    if (selected_pc == -1) {
        auto it = lanes_by_pc.begin();
        selected_pc = it->first;
        selected_lanes = &it->second;
    }

    // 执行选中的 PC 组
    // ...

    // 长延迟指令：标记 lane 为 blocked
    if (stmt->is_long_delay()) {
        for (int lane : *selected_lanes) {
            ThreadContext* t = next_warp->get_thread(lane);
            if (t) t->set_blocked(true, stmt->latency);
        }
    }
}
```

### Phase 4: Unblocked 解除机制

每 cycle 检查 blocked lanes 是否到期：

```cpp
// 在 exe_once() 或 schedule_next() 中
for (int lane = 0; lane < WARP_SIZE; lane++) {
    ThreadContext* t = next_warp->get_thread(lane);
    if (t && t->is_blocked()) {
        t->decrement_blocked_cycles();
        if (t->blocked_cycles_remaining() == 0) {
            t->set_blocked(false);
        }
    }
}
```

---

## 架构影响

### 需要修改的文件

| 文件 | 修改内容 |
|------|---------|
| `include/ptxsim/statement_context.h` | 添加 `InstructionAttributes` |
| `src/ptxsim/core/sm_context.cpp` | 调度器增强（Phase 2-3） |
| `src/ptxsim/core/thread_context.cpp` | `set_blocked()`, `decrement_blocked_cycles()` |
| `src/ptxsim/core/warp_context.cpp` | `is_lane_blocked()` 查询 |
| `include/ptxsim/thread_state.h` | `is_blocked`, `blocked_cycles` 字段 |

### 不需要修改的文件

| 文件 | 原因 |
|------|------|
| `simt_stack.cpp/h` | 屏障机制独立运作 |
| `ptx_parser/*` | 解析层不感知执行调度 |
| `warp_context.cpp` 中的 `get_lanes_by_pc()` | 仅用于 PC 分组，不感知 blocked 状态 |

---

## 调度策略对比

| 策略 | 描述 | 优点 | 缺点 |
|------|------|------|------|
| **Lowest PC first (current)** | 始终选择最小 PC | 简单、可预测 | 长延迟指令阻塞其他路径 |
| **Lowest Unblocked PC** | 选择最低 PC 的非 blocked 组 | 解决阻塞问题，保持简单 | 仍非真正 ITS |
| **Round-Robin** | 各 PC 组轮流 | 公平，避免饥饿 | 收敛延迟增加 |
| **Ready-first** | 有 ready lane 的组优先 | 最大化利用率 | 可能饿死低 PC 组 |
| **Per-lane scheduling** | 每 lane 独立调度 | 真正 ITS | 需要大改，复杂度高 |

---

## 决策

**推荐**：采用 **Lowest Unblocked PC** 策略（Phase 1 + 2 + 3），原因：
1. 解决 BUG-SIMT-001（正确性）
2. 处理长延迟指令阻塞问题（功能性）
3. 保持调度器简单性
4. 为未来真正 ITS 预留接口

**不推荐**：
- **Per-lane scheduling**：复杂度高，实现成本大，当前不需要
- **Round-Robin**：引入不必要的收敛延迟

---

## 开放问题

1. **Latency 值来源**：从 PTX 指令属性定义，还是从 GPU 架构配置读取？
2. **Blocked cycles 精确性**：每 cycle decrement 是否准确反映硬件行为？
3. **Barrier 处理**：blocked 检测是否影响 barrier 同步？
4. **测试验证**：如何验证 ITS 行为正确性？

---

## 参考

- [BUG-SIMT-001: Divergent Warp 单 Cycle 执行多条不同 PC 指令](../reports/BUG-SIMT-001-divergent-warp-multiple-pc-per-cycle.md)
- `ptx-instruction-pipeline` 技能文档 — 指令执行流水线
- `ptx-barrier-mechanism` 技能文档 — 屏障机制（Wbar）
- `src/ptxsim/core/sm_context.cpp:219-257` — 当前 divergent path 实现
- `src/ptxsim/core/warp_context.cpp:327-340` — `get_lanes_by_pc()` 实现
- NVIDIA Hopper Architecture Whitepaper — ITS 描述