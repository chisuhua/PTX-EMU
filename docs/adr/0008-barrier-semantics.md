# ADR-0008: Barrier 语义增强 - Convergence + Memory Fence

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 4 (Barrier 增强) |
| **作者** | PTX-EMU Team |

## 上下文

PTX 的 barrier 指令（`bar.sync`、`bar.warp.sync`）具有两个关键语义：

1. **同步语义**：所有参与的线程必须到达 barrier 后才能继续
2. **内存语义**：barrier 隐含 memory fence，barrier 前的所有内存写入对 barrier 后的所有线程可见

早期的简单 counting barrier 只实现了同步语义，缺少：

- 参与线程的精确跟踪（participation mask）
- Memory fence 验证
- 收敛点关联（barrier 后线程应恢复到哪个 PC）

## 决策驱动因素

1. **PTX ISA 规范要求**：PTX ISA 9.1 Section 9.7.13.1 明确 barrier 隐含 memory fence
2. **调试需求**：需要验证 barrier 语义是否被正确遵守
3. **SIMT 集成**：barrier 必须在 reconvergence point 之后，需要与 SIMT 栈协调

## 考虑的替代方案

### 方案 A: Counting Barrier（简单计数器）

**描述**: 维护一个计数器，每个到达的线程计数+1，计满后释放所有线程

**优点**:
- 实现简单
- 性能好

**缺点**:
- 无法验证参与线程是否正确
- 无法关联 reconvergence point
- 无法验证 memory fence

### 方案 B: Always-On Memory Fence 验证

**描述**: 每次 barrier 都验证所有前置内存操作已完成

**优点**:
- 完整保护
- 能检测所有语义错误

**缺点**:
- 性能开销 5-10%
- 需要跟踪所有内存操作

### 方案 C: Debug-Only Convergence Barrier (✅ 选中)

**描述**: Wbar 实现 convergence barrier + memory fence 验证，但验证仅在 PTX_DEBUG 模式下启用

**优点**:
- 开发时能检测语义错误
- 生产模式无性能影响
- 与 SIMT 栈集成良好

**缺点**:
- 生产模式无法检测 barrier 误用

**选择理由**: PTX-EMU 主要用于研究和教学，开发阶段的正确性验证比运行时性能更重要。Debug-only 验证平衡了两者。

## 决策内容

### 设计原则

1. **Participation Mask**：明确哪些线程必须参与 barrier
2. **Arrived Mask 跟踪**：记录哪些线程已到达
3. **Reconvergence PC 关联**：barrier 完成后线程恢复到的 PC
4. **Debug-Only 验证**：memory fence 验证仅在 PTX_DEBUG 模式下启用

### 实现要点

```cpp
class Wbar {
public:
    int barrier_id;
    int reconvergence_pc;           // barrier 后线程恢复的 PC
    uint32_t participation_mask;    // 必须参与的线程
    uint32_t arrived_mask;          // 已到达的线程
    
    // Memory fence 状态（仅调试用）
    bool is_initialized;
    bool memory_fence_verification_enabled;
    
    #ifdef PTX_DEBUG
    std::vector<std::pair<int, uint64_t>> pre_barrier_stores;  // barrier 前的内存操作
    #endif
    
    void init(int _reconvergence_pc, uint32_t _participation_mask) {
        reconvergence_pc = _reconvergence_pc;
        participation_mask = _participation_mask;
        arrived_mask = 0;
        is_initialized = true;
    }
    
    void arrive(int lane_id) {
        arrived_mask |= (1u << lane_id);
        #ifdef PTX_DEBUG
        // 记录 barrier 前的内存操作
        pre_barrier_stores.push_back({lane_id, get_last_store_addr(lane_id)});
        #endif
    }
    
    bool is_complete() const {
        return (arrived_mask & participation_mask) == participation_mask;
    }
    
    #ifdef PTX_DEBUG
    void verify_memory_fence() const {
        if (!memory_fence_verification_enabled) return;
        
        // 验证所有 barrier 前的 store 对参与线程可见
        for (const auto& store : pre_barrier_stores) {
            int lane_id = store.first;
            uint64_t addr = store.second;
            for (int i = 0; i < 32; i++) {
                if (participation_mask & (1u << i)) {
                    if (!is_store_visible(i, addr)) {
                        PTX_ERROR("Lane %d store to 0x%lx not visible to lane %d!",
                                  lane_id, addr, i);
                    }
                }
            }
        }
    }
    #endif
};
```

### Barrier 执行流程

```cpp
// barrier.cpp
void BarWarpSyncHandler::execute(ThreadContext* context, const BarrierInstr& instr) {
    WarpContext* warp_ctx = context->warp_context_;
    int lane_id = context->lane_id_;
    
    // 1. 当前线程到达 barrier
    warp_ctx->wbar.arrive(lane_id);
    
    // 2. 检查 barrier 是否完成
    if (!warp_ctx->wbar.is_complete()) {
        // 未完成，线程阻塞
        context->state = ThreadStatus::Blocked;
        return;
    }
    
    // 3. Barrier 完成，验证 memory fence（Debug 模式）
    #ifdef PTX_DEBUG
    warp_ctx->wbar.verify_memory_fence();
    #endif
    
    // 4. 所有线程恢复到 reconvergence PC
    for (int i = 0; i < WARP_SIZE; i++) {
        if (warp_ctx->wbar.participation_mask & (1u << i)) {
            if (i == lane_id) {
                // 当前线程使用 force_set_pc
                context->force_set_pc(warp_ctx->wbar.reconvergence_pc);
                context->set_next_pc(warp_ctx->wbar.reconvergence_pc);
            } else {
                // 其他线程通过 WarpContext 设置
                warp_ctx->set_thread_pc(i, warp_ctx->wbar.reconvergence_pc);
            }
            // 重置线程状态
            warp_ctx->set_thread_status(i, ThreadStatus::Active);
        }
    }
    
    // 5. 重置 barrier
    warp_ctx->wbar.reset();
    
    // 6. 检查 SIMT 栈收敛（barrier 可能在 reconvergence point 后）
    warp_ctx->check_reconvergence();
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/wbar.h` | 修改 | Wbar 结构增强 |
| `src/ptxsim/instructions/barrier.cpp` | 修改 | barrier 执行流程 |
| `src/ptxsim/core/sm_context.cpp` | 修改 | barrier 后调用 check_reconvergence |

## 后果

### 正面影响

- 精确验证 barrier 语义
- Debug 模式能检测 memory fence 违规
- 与 SIMT 栈正确集成

### 负面影响

- Debug 模式下 memory fence 验证有性能开销
- 需要跟踪 barrier 前的内存操作（调试用）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| participation_mask 计算错误 | 中 | 高 | 单元测试覆盖发散场景的 barrier |
| Memory fence 验证遗漏 | 低 | 高 | Debug 模式下充分测试 |
| Barrier 完成后线程状态泄漏 | 低 | 高 | 确保 barrier 后重置 status = Active |

## 合规检查

后续相关开发应检查：

- [ ] barrier 指令正确初始化 participation_mask
- [ ] 所有参与线程都调用 arrive()
- [ ] barrier 完成后验证 is_complete()
- [ ] barrier 后线程恢复到正确的 reconvergence_pc
- [ ] Debug 模式下 memory fence 验证启用

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [PTX ISA 9.1 Section 9.7.13.1 - Barrier Synchronization](../archive/ptx-instruction-reference/9.7.13_sync_comm.md)
- [SIMT 架构文档](../architecture/SIMT-ARCHITECTURE-V2.md#34-barrier-机制)
- [ADR-0006: SIMT Stack 显式控制流管理](./0006-simt-stack-management.md)
