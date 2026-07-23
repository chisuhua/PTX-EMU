# ADR-0012: Per-Thread PC 设计（Volta+ SIMT 模型）

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 3 (Per-Thread PC) |
| **作者** | PTX-EMU Team |

## 上下文

GPU 的 SIMT 执行模型经历了演进：

- **Pre-Volta（Fermi ~ Pascal）**：Per-Warp PC，warp 内所有线程共享一个 PC
- **Volta+（Turing、Ampere、Hopper、Blackwell）**：Per-Thread PC，warp 内每个线程有独立的 PC

Per-Warp PC 的局限：
- 无法精确模拟独立线程调度（Independent Thread Scheduling）
- 无法正确处理线程提前退出（exit）的场景
- 无法模拟 Volta+ 引入的新语义（如 `bar.sync` 的精确参与线程）

## 决策驱动因素

1. **硬件准确性**：Volta+ GPU 使用 per-thread PC，模拟器应准确反映
2. **Independent Thread Scheduling**：Hopper 支持线程独立调度，需要 per-thread PC
3. **Barrier 语义正确性**：`bar.warp.sync mask` 需要精确知道哪些线程参与

## 考虑的替代方案

### 方案 A: Per-Warp PC

**描述**: Warp 内所有线程共享一个 PC

**优点**:
- 实现简单
- 内存占用小（1 个 PC vs 32 个）

**缺点**:
- 无法模拟 Volta+ 的独立线程调度
- 无法处理线程提前退出
- 无法精确跟踪 divergent 执行

### 方案 B: Per-Thread PC with Warp-Level Coordination (✅ 选中)

**描述**: 每个线程有独立 PC，但通过 WarpContext 统一管理

**优点**:
- 精确模拟硬件行为
- 支持独立线程调度
- 支持线程提前退出
- 通过 WarpContext 统一管理，避免分散

**缺点**:
- 内存占用增加（32 倍）
- 调度器需要检查所有线程的 PC

**选择理由**: Per-thread PC 是准确模拟 Volta+ GPU 的必要条件，32 倍的内存开销在模拟器场景可接受（32 * 4 bytes = 128 bytes per warp，完全可忽略）。

## 决策内容

### 设计原则

1. **ThreadState 存储 per-thread PC**：每个线程有自己的 pc 和 next_pc
2. **WarpContext 统一管理**：所有 PC 操作通过 WarpContext 的接口
3. **调度器检查 per-thread 状态**：scheduler 根据线程 PC 决定是否调度该 warp

### 实现要点

```cpp
// ThreadState - per-thread PC
struct ThreadState {
    int pc;                     // 当前指令 PC
    int next_pc;                // 下一条指令 PC
    bool is_active;             // 线程是否活跃
    ThreadStatus status;        // 线程状态（Active/Blocked/Exited）
};

// WarpState - 包含 32 个线程的状态
struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t active_mask;       // 当前活跃的线程掩码
    int warp_pc;                // Warp 级 PC（非分歧时使用）
};

// WarpContext 统一 PC 操作
class WarpContext {
    WarpState warp_state;
    
    // 单线程 PC 更新
    void advance_thread_pc(int lane_id, int new_pc) {
        warp_state.threads[lane_id].pc = new_pc;
        warp_state.threads[lane_id].next_pc = new_pc + 1;
    }
    
    // 所有活跃线程 PC 更新
    void advance_all_threads(int new_pc) {
        for (int i = 0; i < WARP_SIZE; i++) {
            if (warp_state.threads[i].is_active) {
                warp_state.threads[i].pc = new_pc;
                warp_state.threads[i].next_pc = new_pc + 1;
            }
        }
    }
    
    // 获取线程 PC（ThreadContext 委托调用）
    int get_thread_pc(int lane_id) const {
        return warp_state.threads[lane_id].pc;
    }
};

// 调度器集成
WarpContext* RoundRobinWarpScheduler::schedule_next() {
    for (int i = 0; i < warps.size(); i++) {
        current_warp_idx = (current_warp_idx + 1) % warps.size();
        auto* warp = warps[current_warp_idx];
        
        if (!warp->is_active()) continue;
        
        // 检查 warp 是否准备好 fetch（所有活跃线程 pc == next_pc）
        if (!warp->is_warp_ready_to_fetch()) continue;
        
        return warp;
    }
    return nullptr;
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/thread_state.h` | 修改 | ThreadState 包含 pc/next_pc |
| `include/ptxsim/warp_state.h` | 修改 | WarpState 包含 ThreadState 数组 |
| `include/ptxsim/warp_context.h` | 修改 | 统一 PC 操作接口 |
| `src/ptxsim/core/warp_context.cpp` | 修改 | 实现 PC 管理 |
| `src/ptxsim/core/warp_scheduler.cpp` | 修改 | 调度器集成 readiness 检查 |

## 后果

### 正面影响

- 精确模拟 Volta+ GPU 行为
- 支持独立线程调度
- 支持线程提前退出
- 与 SIMT 栈、Barrier 机制正确集成

### 负面影响

- 内存占用增加（32 倍，但绝对值很小）
- 调度器需要检查所有线程状态

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| PC 双源不一致 | 高 | 高 | 统一权威源（ADR-0002） |
| 调度器遗漏某些线程 | 低 | 高 | readiness 检查覆盖所有活跃线程 |
| 线程退出后仍被调度 | 低 | 高 | is_exited 标记 + 调度器过滤 |

## 合规检查

后续相关开发应检查：

- [ ] 不在 WarpContext 外直接修改 ThreadState::pc
- [ ] 调度器检查所有活跃线程的 readiness
- [ ] 线程退出后正确标记 is_exited
- [ ] 分支指令正确更新 taken/not-taken 线程的 PC

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [SIMT 架构文档 - 7.1 Per-Thread PC vs Per-Warp PC](../architecture/SIMT-ARCHITECTURE-V2.md#71-per-thread-pc-vs-per-warp-pc)
- [NVIDIA Volta Architecture Whitepaper](../archive/misc/sm90_100.md)
- [ADR-0002: PC 权威源统一到 WarpState](./ADR-0002-pc-unification.md)
