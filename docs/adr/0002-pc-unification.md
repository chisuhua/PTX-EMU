# ADR-0002: PC 权威源统一到 WarpState

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-04 |
| **关联任务** | T11.2.1-T11.2.6 |
| **作者** | PTX-EMU Team |

## 上下文

在 PC 统一之前，ThreadContext 和 WarpState 各自维护 PC 信息：

- `ThreadContext::pc` 和 `ThreadContext::next_pc`：线程级 PC
- `WarpState::threads[i].pc` 和 `WarpState::threads[i].next_pc`：Warp 级 PC

这导致了严重的双源不一致问题：

1. **数据冗余**：两处存储相同信息，需要同步操作
2. **同步开销**：每次 warp 切换需要 `sync_to_warp_state()` 和 `sync_from_warp_state()`
3. **一致性问题**：如果忘记同步，两处 PC 会不一致
4. **Barrier 回归**：barrier 完成后，ThreadContext 的 PC 可能被 WarpState 覆盖，导致跳过指令

## 决策驱动因素

1. **单一真相源（Single Source of Truth）**：PC 只应在一个地方存储
2. **SIMT 语义正确性**：真实 GPU 的 PC 是 per-thread 的（Volta+），存储在 warp 级别的状态中
3. **消除同步开销**：不需要在两处之间来回同步
4. **修复 Barrier 回归**：barrier 后 PC 不应被意外覆盖

## 考虑的替代方案

### 方案 A: 保持 ThreadContext::pc，移除 WarpState::pc

**描述**: ThreadContext 作为权威源，WarpState 通过引用访问

**优点**:
- ThreadContext 是现有代码的主要 PC 访问点
- 修改量较小

**缺点**:
- Warp-level 操作（如 barrier reconvergence）需要修改所有线程的 ThreadContext，接口不自然
- 与 SIMT 硬件模型不符（真实 GPU 的 PC 在 warp scheduler 中维护）
- 调度器选择 warp 时需要检查所有 ThreadContext，效率低

### 方案 B: 保持 WarpState::pc，移除 ThreadContext::pc (✅ 选中)

**描述**: WarpState 作为权威源，ThreadContext 通过 warp_context_ 指针委托访问

**优点**:
- 符合 SIMT 硬件模型（warp scheduler 管理 PC）
- Warp-level 操作（barrier、reconvergence）接口自然
- 调度器可直接在 WarpState 上检查 PC 状态
- 消除同步操作，ThreadContext 直接委托给 WarpState

**缺点**:
- ThreadContext 需要持有 warp_context_ 指针（生命周期管理）
- 所有 PC 访问需要修改为委托调用

**选择理由**: 与真实 GPU 的 SIMT 执行模型一致，warp 级别的 PC 管理是硬件的实际行为。且 warp-level 操作（如 barrier reconvergence）在 WarpState 上操作更自然。

### 方案 C: 保留两处，但添加自动同步机制

**描述**: 使用属性或 getter/setter 自动保持同步

**优点**:
- 现有代码改动最小

**缺点**:
- 隐式同步容易出 bug，调试困难
- 仍然有冗余存储
- 无法从根本上解决一致性问题

## 决策内容

### 设计原则

1. **WarpState 是唯一 PC 存储**：ThreadContext 不再拥有 pc/next_pc 成员
2. **委托访问**：ThreadContext 的 get_pc()/set_pc() 委托给 WarpState
3. **生命周期绑定**：ThreadContext 通过 warp_context_ 指针引用，确保生命周期一致
4. **向后兼容**：保留 get_pc()/set_pc() 接口，内部实现改为委托

### 实现要点

```cpp
// ThreadContext 结构变更
class ThreadContext {
    // 移除: int pc_; int next_pc_;
    WarpContext* warp_context_;  // 新增
    
    // 委托实现
    int get_pc() const {
        return warp_context_->get_thread_pc(thread_id_);
    }
    void set_pc(int new_pc) {
        warp_context_->set_thread_pc(thread_id_, new_pc);
    }
    void set_next_pc(int new_next_pc) {
        warp_context_->set_thread_next_pc(thread_id_, new_next_pc);
    }
};

// WarpState 中的存储
struct ThreadState {
    int pc;           // 当前指令 PC
    int next_pc;      // 下一条指令 PC
    bool is_active;
};
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/thread_context.h` | 修改 | 移除 pc/next_pc，添加 warp_context_ |
| `src/ptxsim/core/thread_context.cpp` | 修改 | PC 访问器委托实现 |
| `include/ptxsim/warp_context.h` | 修改 | 添加 get/set thread PC 接口 |
| `src/ptxsim/core/warp_context.cpp` | 修改 | 实现 thread PC 管理 |
| `src/ptxsim/instructions/*.cpp` | 修改 | PC 访问保持不变（接口兼容） |

## 后果

### 正面影响

- 消除 PC 双源不一致问题
- 移除 sync_to/from_warp_state 同步操作
- 与 SIMT 硬件模型一致
- Barrier reconvergence 操作更自然

### 负面影响

- ThreadContext 必须绑定到 WarpContext（不能在 warp 间迁移，但这是正确的行为）
- 所有 PC 相关代码需要验证委托调用正确

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| warp_context_ 为空时访问 | 低 | 高 | 构造函数确保初始化，访问前检查 |
| 遗漏某处直接访问 ThreadContext::pc | 中 | 高 | 编译验证（移除成员后编译失败会暴露） |
| 回归 bug（如之前 test_ptx_ld_st 失败） | 中 | 高 | 充分的 PC 管理回归测试 |

## 合规检查

后续相关开发应检查：

- [ ] 不在 ThreadContext 中添加新的 PC 相关成员
- [ ] 新代码通过 get_pc()/set_pc() 委托访问，不直接访问 WarpState
- [ ] WarpContext 的 thread PC 管理接口保持线程安全

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-04 | 初始版本 | PTX-EMU Team |

## 参考

- [任务计划](../reports/task-plan.md#sprint-112-pc-权威源统一day-5-8)
- [GPU Pipeline 调研](../archive/misc/GPU-PIPELINE-RESEARCH.md)
- [PC Management Fix Design](../archive/misc/PC-MANAGEMENT-FIX-DESIGN.md)
