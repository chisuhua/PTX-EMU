# Design: add-sm90-100-bsync-interleave

## 概述

实现 sm90/sm100 (Hopper/Blackwell) 架构的 BSSY/BSYNC 分支同步机制和动态线程迁移调度器。

## 架构设计

### BsyncManager 放置位置

**决策**: BsyncManager 作为 SMContext 的成员变量，管理所有 warps 的屏障

**理由**: BSSY/BSYNC 是 warp 间协调机制，需要跨 warp 共享状态

### BsyncState 结构

```cpp
struct BsyncState {
    uint32_t barrier_id;
    uint32_t waiting_threads_mask;  // 等待中的线程位掩码
    uint32_t total_threads;         // 总线程数
    uint32_t suspended_pc;           // 挂起点 PC
    bool is_released;
};

class BsyncManager {
    std::unordered_map<uint32_t, BsyncState> barriers_;
public:
    void bssy(uint32_t barrier_id, uint32_t thread_mask);
    uint32_t bsync(uint32_t barrier_id, uint32_t lane_id, uint32_t current_pc);
    bool check_release(uint32_t barrier_id);
    void release(uint32_t barrier_id);
    BsyncState* get_state(uint32_t barrier_id);
};
```

### 三种执行模式

```cpp
enum class DivergenceExecutionMode {
    Sequential,      // 先执行 Path A，再执行 Path B（当前实现）
    Interleaved,    // 动态交错执行（场景 2）
    ShortestFirst   // 短路径优先（Blackwell 风格，场景 3）
};
```

### 动态交错调度策略

1. **Sequential (默认)**: 按顺序执行每个分支路径
2. **Interleaved**: 当一组线程到达屏障等待时，调度器随机选择另一组可执行线程
3. **ShortestFirst**: 基于启发式估算路径长度，优先调度短路径

### blocked_cycles 递减位置

**决策**: 在 SMContext::exe_once() 中集中递减

**理由**: 更容易追踪和管理，避免在多个地方重复递减逻辑

## 文件变更

### 新增文件

| 文件 | 职责 |
|------|------|
| `include/ptxsim/bsync_state.h` | BsyncState 结构体 + BsyncManager 类声明 |
| `src/ptxsim/core/bsync_state.cpp` | BsyncManager 实现 |

### 修改文件

| 文件 | 变更 |
|------|------|
| `src/ptxsim/instructions/barrier.cpp` | 集成 BsyncManager，替代 is_blocked 标记 |
| `src/ptxsim/core/warp_scheduler.cpp` | 添加 schedule_with_migration() 动态交错逻辑 |
| `src/ptxsim/core/sm_context.cpp` | 添加 divergence_execution_mode 配置和调度策略选择 |
| `include/ptxsim/thread_state.h` | 完善 blocked_cycles_remaining 字段 |

## 实现顺序

1. `bsync_state.h/cpp` - BSSY/BSYNC 核心状态机
2. `barrier.cpp` 集成 - 替换 is_blocked 为 BsyncManager
3. `warp_scheduler.cpp` 修改 - 添加动态交错逻辑
4. `sm_context.cpp` 修改 - 添加调度策略选择
5. `thread_state blocked_cycles` - 完善生命周期
6. 测试用例
7. 文档更新

## 风险与备选

| 风险 | 缓解措施 |
|------|---------|
| BsyncManager 状态复杂导致死锁 | 添加超时检测和强制释放机制 |
| 动态交错引入不确定性导致难以测试 | 通过 seed 控制随机性，提供确定性测试模式 |
| 性能下降（频繁切换） | 提供开关，默认使用顺序模式 |
## 实现状态（归档前记录 —2026-06-09）

**已完成**：

- `include/ptxsim/bsync_state.h` — `BsyncState` + `BsyncManager` + `DivergenceExecutionMode` 三模式枚举
- `src/ptxsim/core/bsync_state.cpp` — `bssy` / `bsync` / `check_release` / `release`全部实现
- `src/ptxsim/instructions/barrier.cpp` —集成 `bsync_manager_`，替换简单的 `is_blocked`标记
- `src/ptxsim/core/sm_context.cpp` — `set_divergence_execution_mode` / `get_divergence_execution_mode` / `select_next_group` / `suspend_and_switch`
- `include/ptxsim/thread_state.h` — `blocked_cycles_remaining`字段 + `is_schedulable()` 检查
- `tests/unit/sync/test_bsync_state.cpp` —单元测试覆盖生命周期

**占位实现（未完整实现，需后续 follow-up）**：

- `DivergenceExecutionMode::Interleaved` — `select_next_group` 的 Interleaved 分支当前 `return0`（fall-through 至 sequential）
- `DivergenceExecutionMode::ShortestFirst` — `select_next_group` 的 ShortestFirst 分支当前 `return0`（fall-through 至 sequential）
- `suspend_and_switch()` —注释明确写 "placeholder for future blocking implementation"

**实际可用模式**：`Sequential`（默认）— Hopper/Blackwell 的动态交错与短路径优先调度策略尚未落地。

**推荐 follow-up change**：创建新 change `add-sm90-100-scheduler-policies` 实现 Interleaved/ShortestFirst调度策略。
