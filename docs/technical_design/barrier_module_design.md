# BarrierModule 技术设计文档

**项目**: PTX-EMU Barrier 统一管理模块
**状态**: 草稿
**日期**: 2026-05-24
**作者**: Sisyphus

---

## 1. 背景与问题分析

### 1.1 当前问题

`bar.warp.sync` 在只有部分线程到达 barrier 时就错误地判定完成了。

**问题现象** (从调试日志):
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

当前实现存在三个关键缺陷：

1. **动态 mask 计算错误** (`barrier.cpp:129-140`)
   - 代码试图通过 `warp_state.threads[i].pc == current_pc` 判断哪些线程"参与"了当前 barrier
   - 但这个逻辑只检查线程是否在**同一个 PC**，没有考虑**实际执行了 bar.warp.sync 指令**
   - 当 warp 分叉时 (lanes 0-15 在 PC=25，lanes 16-31 在 PC=20)，只有到达 PC=20 的线程才参与这个 barrier

2. **Wbar::is_complete() 判断不准确** (`wbar.h:31-36`)
   ```cpp
   bool is_complete() const {
       if (!is_initialized || participation_mask == 0) {
           return false;
       }
       return (arrived_mask & participation_mask) == participation_mask;
   }
   ```
   - 当 `participation_mask = 0xFFFF0000` (32-bit，期望 16 个线程)
   - `arrived_mask = 0xFFFF0000` (实际到达 16 个线程)
   - `(0xFFFF0000 & 0xFFFF0000) == 0xFFFF0000` → true ✓
   - 但问题是：如果只有 lanes 16-31 到达，而 lanes 0-15 根本没参与这个 barrier，这是**正确的完成判断**
   - **真正的问题**：静态 mask `0xFFFF0000` 本身就错了——它应该只包含**实际到达 PC 的线程**

3. **静态 mask vs 动态 mask 的混淆**
   - `bar.warp.sync` 的 operand 是**静态 participation mask**，表示哪些线程"应该"参与
   - 但在分叉的 warp 中，只有**到达 barrier 指令 PC 的线程**才真正参与
   - 当前代码在初始化时用 `dynamic_mask & static_mask`，但 dynamic_mask 计算有误

### 1.2 现有架构分析

**当前 wbar.h 结构**:
```cpp
struct Wbar {
    uint32_t participation_mask = 0;  // 期望参与的线程掩码
    uint32_t arrived_mask = 0;         // 实际到达的线程掩码
    int reconvergence_pc = -1;
    uint32_t barrier_pc = 0;
    bool is_initialized = false;
    int expected_count = 0;            // 参与线程数（popcount of participation_mask）
    // ...
};
```

**当前 warp_state.h**:
```cpp
struct WarpState {
    std::array<ThreadState, 32> threads;
    uint32_t exec_mask = 0xFFFFFFFF;
    std::array<Wbar, 4> wbars;        // 4 个 warp-level barriers
    int current_wbar_id = -1;
    // ...
};
```

**当前 SMContext 的 CTA 级别 barrier** (`sm_context.cpp:449-532`):
```cpp
bool SMContext::synchronize_barrier(int barId, ThreadContext *thread) {
    std::lock_guard<std::mutex> lock(barrier_mutex_);
    // ...
    barrier_waiting_threads[barId].insert(thread);
    if (barrier_waiting_threads[barId].size() >= barrier_thread_counts[barId]) {
        // 所有线程到达，释放它们
    }
    // ...
}
```

---

## 2. 硬件行为调研

### 2.1 NVIDIA GPU Barrier 架构

**硬件资源** (来自 PTXAS 逆向工程):
- 每个 CTA 有 **16 个命名 barriers** (索引 0-15)
- `EIATTR_NUM_BARRIERS` 元数据告诉硬件需要初始化多少 barriers

**指令映射**:

| PTX | SASS | 行为 |
|-----|------|------|
| `bar.sync id` | `BAR.SYNC` | 等待所有 CTA 线程到达 barrier id |
| `bar.arrive id` | `BAR.ARRIVE` | 到达 barrier 不等待完成 |
| `bar.red...` | `BAR.RED` | 归约操作 |
| `bar.warp.sync mask, rpc` | `BSYNC` | Warp 级同步，配合 WARPSYNC |

**关键设计原则**:

1. **独立线程调度 (Volta+)**: 每个线程有自己的 PC，barrier 只等待实际到达的线程
2. **Convergence Barrier File**: 硬件维护每个 barrier 的状态
3. **Warp-level vs CTA-level**: `bar.warp.sync` 是 warp 内同步，`bar.sync` 是 CTA 内所有 warps 同步

### 2.2 GPGPU-sim 的做法

**Barrier Table** (在 shader_core_ctx 中):
```cpp
m_barriers(this, config->max_warps_per_shader, config->max_cta_per_core,
           config->max_barriers_per_cta, config->warp_size);
```

关键设计：
- CTA 级别的 barrier 表，跟踪所有 warps 的到达状态
- Warp 级别的 barrier 状态机
- 当 warp 到达 barrier 时，标记为 waiting，直到所有 warps 到达才释放

---

## 3. 设计方案

### 3.1 架构概述

```
┌─────────────────────────────────────────────────────────────────────┐
│                         SMContext                                   │
│  ┌─────────────────────────────────────────────────────────────────┐│
│  │                    CTABarrierManager                            ││
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐              ││
│  │  │ barrier[0] │  │ barrier[1]  │  │ barrier[15] │  (16 barriers)││
│  │  │ warp_mask  │  │ warp_mask   │  │ warp_mask   │              ││
│  │  │ arrived    │  │ arrived     │  │ arrived     │              ││
│  │  └─────────────┘  └─────────────┘  └─────────────┘              ││
│  └─────────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────────┘
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
         ┌──────────────────┐  ┌──────────────────┐
         │   WarpContext    │  │   WarpContext    │
         │  ┌────────────┐  │  │  ┌────────────┐  │
         │  │WarpBarrier │  │  │  │WarpBarrier │  │
         │  │ (per-warp)│  │  │  │ (per-warp)│  │
         │  └────────────┘  │  │  └────────────┘  │
         └──────────────────┘  └──────────────────┘
```

### 3.2 核心类设计

#### 3.2.1 WarpBarrier (Warp 级别 barrier)

```cpp
// include/ptxsim/barrier/warp_barrier.h
#pragma once

#include <cstdint>
#include <algorithm>

namespace ptxsim {

/**
 * WarpBarrier - Warp 级别的屏障状态管理
 *
 * 模拟硬件的 warp-level 同步机制 (bar.warp.sync)
 *
 * 设计原则:
 * 1. 明确区分 "期望参与者" 和 "实际到达者"
 * 2. 只有实际到达 barrier 指令 PC 的线程才算参与
 * 3. barrier 完成需要 (arrived_count == expected_count)
 */
class WarpBarrier {
public:
    static constexpr int WARP_SIZE = 32;

    enum class State {
        Uninitialized,  // 未初始化
        Initializing,   // 正在初始化 (有线程到达但未完成)
        Waiting,        // 等待中 (部分线程到达，全部未到齐)
        Complete,        // 完成 (所有期望线程已到达)
        Released         // 已释放 (线程已离开 barrier)
    };

    WarpBarrier() { reset(); }

    // 初始化 barrier
    void init(uint32_t participation_mask, int reconvergence_pc, uint32_t barrier_pc);

    // 线程到达 barrier
    void arrive(int lane_id);

    // 检查 barrier 是否完成
    bool is_complete() const;

    // 获取当前状态
    State get_state() const { return state_; }

    // 获取统计信息
    int get_expected_count() const { return expected_count_; }
    int get_arrived_count() const { return arrived_count_; }
    uint32_t get_participation_mask() const { return participation_mask_; }
    uint32_t get_arrived_mask() const { return arrived_mask_; }
    int get_reconvergence_pc() const { return reconvergence_pc_; }
    uint32_t get_barrier_pc() const { return barrier_pc_; }

    // 检查是否需要等待
    bool needs_to_wait(int lane_id) const;

    // 重置 barrier
    void reset();

    // 获取未到达的参与者
    uint32_t get_missing_mask() const {
        return participation_mask_ & ~arrived_mask_;
    }

    // 是否所有参与者都已到达
    bool all_participants_arrived() const {
        return (arrived_mask_ & participation_mask_) == participation_mask_;
    }

#ifdef PTX_DEBUG
    void dump() const;
#endif

private:
    State state_;
    uint32_t participation_mask_;  // 期望参与者的掩码
    uint32_t arrived_mask_;         // 已到达的参与者掩码
    int expected_count_;            // 期望参与者数量
    int arrived_count_;             // 已到达数量
    int reconvergence_pc_;         // 重汇聚 PC
    uint32_t barrier_pc_;           // barrier 指令的 PC
};

} // namespace ptxsim
```

#### 3.2.2 CTABarrier (CTA 级别 barrier)

```cpp
// include/ptxsim/barrier/cta_barrier.h
#pragma once

#include <cstdint>
#include <vector>
#include <set>
#include <mutex>

namespace ptxsim {

class ThreadContext;

/**
 * CTABarrier - CTA 级别的屏障状态管理
 *
 * 模拟硬件的 CTA 级同步机制 (bar.sync / __syncthreads())
 *
 * 设计原则:
 * 1. 跟踪所有 warps 的到达状态
 * 2. 只有当所有 warps 的所有线程都到达时才完成
 * 3. 线程级并发安全 (使用 mutex)
 */
class CTABarrier {
public:
    static constexpr int MAX_WARPS_PER_CTA = 16;
    static constexpr int MAX_BARRIERS_PER_CTA = 16;

    CTABarrier();
    explicit CTABarrier(int barrier_id);

    // 初始化 barrier
    void init(int barrier_id, int total_threads, int warp_count);

    // 线程到达 barrier (可能被多个 warps 调用)
    bool arrive(ThreadContext* thread);

    // 获取 barrier 状态
    bool is_complete() const;
    int get_barrier_id() const { return barrier_id_; }
    int get_expected_threads() const { return expected_threads_; }
    int get_arrived_threads() const { return arrived_threads_.size(); }
    const std::set<ThreadContext*>& get_waiting_threads() const { return arrived_threads_; }

    // 重置 barrier
    void reset();

    // 获取调试信息
    int get_warp_count() const { return warp_count_; }
    int get_expected_warps() const { return expected_warps_; }

#ifdef PTX_DEBUG
    void dump() const;
#endif

private:
    int barrier_id_;
    int expected_threads_;     // 期望的总线程数
    int warp_count_;           // CTA 中的 warps 数
    int expected_warps_;       // 期望到达的 warps 数
    std::set<ThreadContext*> arrived_threads_;
    std::mutex mutex_;
    bool is_initialized_;
};

} // namespace ptxsim
```

#### 3.2.3 BarrierModule (统一管理)

```cpp
// include/ptxsim/barrier/barrier_module.h
#pragma once

#include "warp_barrier.h"
#include "cta_barrier.h"
#include <array>
#include <memory>
#include <vector>

namespace ptxsim {

class WarpContext;
class ThreadContext;

/**
 * BarrierModule - 统一的 Barrier 管理模块
 *
 * 整合 warp-level 和 CTA-level barrier 的管理
 *
 * 设计原则:
 * 1. 集中管理所有 barrier 状态
 * 2. 清晰的接口分离
 * 3. 易于调试和测试
 */
class BarrierModule {
public:
    static constexpr int MAX_WARP_BARRIERS = 4;
    static constexpr int MAX_CTA_BARRIERS = 16;

    BarrierModule();

    // ============== Warp Barrier 接口 ==============

    // 初始化 warp barrier
    WarpBarrier* init_warp_barrier(int warp_barrier_id,
                                   uint32_t participation_mask,
                                   int reconvergence_pc,
                                   uint32_t barrier_pc);

    // 获取 warp barrier
    WarpBarrier* get_warp_barrier(int warp_barrier_id);

    // 线程到达 warp barrier
    bool arrive_at_warp_barrier(int warp_barrier_id, int lane_id);

    // 检查 warp barrier 是否完成
    bool is_warp_barrier_complete(int warp_barrier_id) const;

    // warp barrier 是否需要等待
    bool warp_barrier_needs_wait(int warp_barrier_id, int lane_id) const;

    // 释放 warp barrier (移动到 reconvergence PC)
    void release_warp_barrier(int warp_barrier_id, WarpContext* warp_ctx);

    // ============== CTA Barrier 接口 ==============

    // 初始化 CTA barrier
    CTABarrier* init_cta_barrier(int cta_barrier_id,
                                 int total_threads,
                                 int warp_count);

    // 获取 CTA barrier
    CTABarrier* get_cta_barrier(int cta_barrier_id);

    // 线程到达 CTA barrier
    bool arrive_at_cta_barrier(int cta_barrier_id, ThreadContext* thread);

    // 检查 CTA barrier 是否完成
    bool is_cta_barrier_complete(int cta_barrier_id) const;

    // 释放所有等待的线程
    void release_cta_barrier(int cta_barrier_id);

    // ============== 状态查询 ==============

    // 获取统计信息
    int get_active_warp_barrier_count() const;
    int get_active_cta_barrier_count() const;

    // 重置所有 barriers
    void reset_all();

    // 调试打印
#ifdef PTX_DEBUG
    void dump() const;
#endif

private:
    std::array<WarpBarrier, MAX_WARP_BARRIERS> warp_barriers_;
    std::array<CTABarrier, MAX_CTA_BARRIERS> cta_barriers_;
};

} // namespace ptxsim
```

---

## 4. 关键算法

### 4.1 Warp Barrier 动态 Mask 计算

**问题**: 如何正确判断哪些线程真正"参与"了一个 barrier？

**硬件行为**:
- `bar.warp.sync` 是指令级别的同步
- 只有执行到这条指令的线程才参与
- 线程必须在同一个 PC (或 next_pc) 才能算参与

**算法**:
```cpp
uint32_t WarpBarrier::compute_dynamic_participation_mask(
    WarpState& warp_state,
    uint32_t static_mask,
    int current_pc,
    int lane_id) {

    uint32_t dynamic_mask = 0;

    // 找到所有在当前 PC 或 next_pc 的线程
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp_state.threads[i].is_active) continue;
        if (warp_state.threads[i].is_exited) continue;

        // 线程在 barrier PC
        bool at_barrier_pc = (warp_state.threads[i].pc == current_pc) ||
                              (warp_state.threads[i].next_pc == current_pc);

        if (at_barrier_pc) {
            dynamic_mask |= (1u << i);
        }
    }

    // 静态 mask 和动态 mask 的交集
    return dynamic_mask & static_mask;
}
```

### 4.2 Barrier 完成判断

```cpp
bool WarpBarrier::is_complete() const {
    if (state_ == State::Complete) return true;
    if (state_ == State::Uninitialized) return false;

    // 关键: 检查 arrived_mask 是否覆盖了所有 participation_mask
    return (arrived_mask_ & participation_mask_) == participation_mask_;
}

bool WarpBarrier::needs_to_wait(int lane_id) const {
    if (state_ == State::Uninitialized) return false;  // 第一个到达需要初始化

    // 线程已经在 arrived 中，不需要等待
    if (arrived_mask_ & (1u << lane_id)) return false;

    // barrier 已完成，不需要等待
    if (state_ == State::Complete || state_ == State::Released) return false;

    return true;
}
```

---

## 5. 接口变更

### 5.1 BarWarpSyncHandler 更新

**旧实现** (barrier.cpp:108-205):
```cpp
void BarWarpSyncHandler::processOperation(...) {
    // 问题: dynamic_mask 计算不正确
    uint32_t dynamic_mask = 0;
    if (warp_state.current_wbar_id < 0) {
        uint32_t current_pc = warp_state.threads[lane_id].pc;
        for (int i = 0; i < 32; i++) {
            // ... 计算 dynamic_mask
        }
    }
    // 问题: participation_mask 可能是错的
    wbar.init(participation_mask, reconvergence_pc);
    wbar.arrive(lane_id);
    if (wbar.is_complete()) { ... }
}
```

**新实现**:
```cpp
void BarWarpSyncHandler::processOperation(...) {
    WarpContext* warp_ctx = context->warp_context_;
    WarpState& warp_state = warp_ctx->get_warp_state();

    // 使用 BarrierModule
    BarrierModule* barrier = warp_ctx->get_barrier_module();

    int wbar_id = 0;
    WarpBarrier* wbar = barrier->get_warp_barrier(wbar_id);

    if (wbar->get_state() == WarpBarrier::State::Uninitialized) {
        // 第一个到达的线程初始化 barrier
        uint32_t dyn_mask = compute_dynamic_participation_mask(
            warp_state, static_mask,
            warp_state.threads[lane_id].pc,
            lane_id);

        barrier->init_warp_barrier(wbar_id, dyn_mask, reconvergence_pc,
                                  warp_state.threads[lane_id].pc);
    }

    // 线程到达
    bool complete = barrier->arrive_at_warp_barrier(wbar_id, lane_id);

    if (complete) {
        barrier->release_warp_barrier(wbar_id, warp_ctx);
    } else {
        // 阻塞等待
        warp_state.threads[lane_id].is_blocked = true;
        warp_state.threads[lane_id].status = ThreadStatus::Blocked;
        set_pc_overridden(true);
    }
}
```

### 5.2 BarSyncHandler 更新

**旧实现** (sm_context.cpp:449-532):
```cpp
bool SMContext::synchronize_barrier(int barId, ThreadContext *thread) {
    std::lock_guard<std::mutex> lock(barrier_mutex_);
    barrier_waiting_threads[barId].insert(thread);

    if (barrier_waiting_threads[barId].size() >= barrier_thread_counts[barId]) {
        // 完成
    }
}
```

**新实现**:
```cpp
bool SMContext::synchronize_barrier(int barId, ThreadContext *thread) {
    CTAContext* cta = find_cta(thread->get_physical_block_id());
    BarrierModule* barrier = cta->get_barrier_module();

    bool complete = barrier->arrive_at_cta_barrier(barId, thread);

    if (complete) {
        barrier->release_cta_barrier(barId);
        return true;
    }

    return false;
}
```

---

## 6. 文件结构

```
include/ptxsim/barrier/
├── barrier_module.h      # 主模块
├── warp_barrier.h        # Warp barrier 实现
├── cta_barrier.h         # CTA barrier 实现
└── barrier_types.h       # 公共类型和常量

src/ptxsim/barrier/
├── barrier_module.cpp    # 实现
├── warp_barrier.cpp
└── cta_barrier.cpp

tests/barrier/
├── test_warp_barrier.cpp
├── test_cta_barrier.cpp
└── test_barrier_module.cpp
```

---

## 7. 测试计划

### 7.1 WarpBarrier 测试

1. **初始化测试**: 创建后状态为 Uninitialized
2. **单线程到达测试**: 一个线程到达，状态应为 Initializing
3. **部分到达测试**: 16 个线程到达（总共 32 个），不应完成
4. **全部到达测试**: 所有 16 个参与者到达，应变为 Complete
5. **重置测试**: reset() 应清空所有状态

### 7.2 CTABarrier 测试

1. **多 warp 到达测试**: 4 个 warps 逐步到达，应正确跟踪
2. **并发到达测试**: 多个线程同时调用 arrive()
3. **完成释放测试**: 所有线程到达后，所有等待线程应被释放

### 7.3 集成测试

1. **bar.warp.sync 单独测试**: 单 warp，无分歧
2. **bar.warp.sync 分歧测试**: 修复当前 bug
3. **bar.sync 测试**: CTA 级别同步
4. **混合测试**: bar.warp.sync + bar.sync 组合

---

## 8. 风险和缓解

### 8.1 风险

1. **破坏现有功能**: 修改核心 barrier 逻辑可能引入新 bug
2. **性能下降**: 新的 barrier 管理可能增加开销
3. **线程安全**: CTA barrier 的 mutex 可能成为瓶颈

### 8.2 缓解措施

1. **TDD**: 先写测试，再实现
2. **保持向后兼容**: 旧接口可以继续工作
3. **性能测试**: 比较新旧实现的性能
4. **详细日志**: 便于调试

---

## 9. 实施计划

### Phase 1: 基础架构 (1-2 天)
- [ ] 创建 `include/ptxsim/barrier/` 目录结构
- [ ] 实现 `WarpBarrier` 类
- [ ] 实现 `CTABarrier` 类
- [ ] 实现 `BarrierModule` 类

### Phase 2: 集成 (2-3 天)
- [ ] 更新 `BarWarpSyncHandler` 使用 `BarrierModule`
- [ ] 更新 `BarSyncHandler` 使用 `BarrierModule`
- [ ] 添加调试日志

### Phase 3: 测试 (1-2 天)
- [ ] 编写单元测试
- [ ] 修复当前 bug 的回归测试
- [ ] 运行完整测试套件

### Phase 4: 优化 (可选)
- [ ] 性能分析
- [ ] 减少 mutex 竞争

---

## 10. 附录

### A. 调试日志格式建议

```cpp
PTX_INFO_EMU("WarpBarrier[%d] state=%s mask=0x%X arrived=0x%X expected=%d arrived=%d",
              barrier_id,
              state_to_string(state_).c_str(),
              participation_mask_,
              arrived_mask_,
              expected_count_,
              arrived_count_);
```

### B. 关键文件列表

| 文件 | 当前状态 | 修改范围 |
|-----|---------|---------|
| `include/ptxsim/wbar.h` | 现有 | 保留（可能被新类替代） |
| `src/ptxsim/instructions/barrier.cpp` | 现有 | 重写 BarWarpSyncHandler |
| `src/ptxsim/core/sm_context.cpp` | 现有 | 更新 synchronize_barrier |
| `include/ptxsim/barrier/*` | 新增 | 新模块 |