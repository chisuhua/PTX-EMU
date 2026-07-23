# ADR-0003: commit_pc / force_set_pc 分离

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-04 |
| **关联任务** | T11.2.2 |
| **作者** | PTX-EMU Team |

## 上下文

在 ADR-0002（PC 统一到 WarpState）之后，发现 `set_pc()` 存在语义模糊问题：

- `set_pc(new_pc)` 同时设置 `pc = new_pc` 和 `next_pc = new_pc`
- 这导致正常指令执行和 warp-level 操作（如 barrier reconvergence）使用相同的接口
- 结果：barrier 完成后，如果调用 `commit_pc()`，pc 会被设为 `next_pc`（已经被 set_pc 修改过），导致 PC 跳过指令

具体回归案例：PipelineHandler::ExecPipe 在 barrier 后调用 `set_next_pc(get_pc() + 1)`，此时 get_pc() 返回的是 reconvergence_pc，导致 next_pc = reconvergence_pc + 1，跳过了 barrier 后的第一条指令。

## 决策驱动因素

1. **语义清晰**：正常 PC 推进 vs. warp-level PC 设置应有明确区分
2. **防止误用**：不应让一个函数承担两种不同的语义
3. **可追溯性**：PC 的变更应有明确的入口点，便于调试

## 考虑的替代方案

### 方案 A: set_pc 只设置 pc，不修改 next_pc

**描述**: 修改 set_pc 语义，只设置 pc

**优点**:
- 简化 set_pc 行为

**缺点**:
- 破坏初始化/同步场景（这些场景需要 pc 和 next_pc 一致）
- 调用方需要手动设置 next_pc，容易遗漏

### 方案 B: 添加新函数，保留 set_pc (✅ 选中)

**描述**: 
- `commit_pc()`: 正常 PC 推进的唯一入口，执行 `pc = next_pc`
- `force_set_pc(new_pc)`: 设置 pc 和 next_pc 为同一值，用于 warp-level 操作
- `set_pc(new_pc)`: 保留，仅用于初始化和同步

**优点**:
- 语义清晰，每种场景有专用入口
- force_set_pc 显式表达"强制设置"的意图
- commit_pc 作为单一入口，便于审计 PC 推进

**缺点**:
- 接口数量增加
- 需要文档说明各函数的使用场景

**选择理由**: 语义分离是最安全的设计。commit_pc 作为正常推进的单一入口，force_set_pc 明确表达 warp-level 强制设置的意图。

### 方案 C: 使用参数区分行为

**描述**: `set_pc(new_pc, mode)` 通过 mode 参数区分

**优点**:
- 函数数量不增加

**缺点**:
- 布尔参数/枚举参数可读性差
- 调用时容易传错 mode
- 不如独立函数名表意清晰

## 决策内容

### 设计原则

1. **commit_pc 是正常推进的唯一入口**：每条指令执行完成后通过 commit_pc 推进 PC
2. **force_set_pc 用于 warp-level 强制设置**：barrier reconvergence、异常处理等场景
3. **set_pc 仅用于初始化和同步**：线程创建、状态恢复等场景

### 实现要点

```cpp
// ThreadContext 中的 PC 操作接口
class ThreadContext {
    // 正常 PC 推进 - 唯一入口
    void commit_pc() {
        // pc ← next_pc
        warp_context_->set_thread_pc(thread_id_, get_next_pc());
    }
    
    // Warp-level 强制设置 - 同时设置 pc 和 next_pc
    void force_set_pc(int new_pc) {
        warp_context_->set_thread_pc(thread_id_, new_pc);
        warp_context_->set_thread_next_pc(thread_id_, new_pc);
    }
    
    // 初始化/同步专用
    void set_pc(int new_pc) {
        // 同时设置 pc 和 next_pc（与 force_set_pc 相同）
        // 保留此函数是为了向后兼容初始化和同步场景
        force_set_pc(new_pc);
    }
};
```

### 使用场景对照

| 场景 | 使用的函数 | 原因 |
|------|-----------|------|
| 正常指令执行完成 | `commit_pc()` | pc ← next_pc，标准推进 |
| Barrier 完成（当前线程） | `set_pc(reconvergence_pc)` | 设置到聚合点（`force_set_pc` 已于 2026-07 移除） |
| Barrier 完成（其他线程） | `WarpContext::set_thread_pc()` | 通过 WarpContext 批量设置 |
| 分支指令 | 通过 PipelineHandler 间接处理 | 不直接操作 PC |
| 线程初始化 | `set_pc(0)` | 设置初始值 |
| 状态同步 | `set_pc()` | 恢复到指定状态 |

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/thread_context.h` | 修改 | 添加 commit_pc() 和 force_set_pc() |
| `src/ptxsim/core/thread_context.cpp` | 修改 | _execute_once 改用 commit_pc() |
| `src/ptxsim/instructions/barrier.cpp` | 修改 | barrier 完成改用 force_set_pc() |

## 后果

### 正面影响

- 消除 set_pc 语义模糊导致的回归
- PC 推进路径可审计（所有正常推进都经过 commit_pc）
- Barrier reconvergence 行为正确

### 负面影响

- 开发者需要理解三种 PC 操作的语义区别
- 需要文档和代码注释明确说明使用场景

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| 误用 set_pc 代替 commit_pc | 中 | 高 | 代码审查时检查 PC 操作的正确性 |
| 遗漏某处应使用 force_set_pc | 低 | 高 | 回归测试覆盖 barrier reconvergence 场景 |

## 合规检查

后续相关开发应检查：

- [ ] 正常指令执行完成后使用 commit_pc()，而非 set_pc()
- [ ] Warp-level PC 强制设置使用 force_set_pc()
- [ ] 不在热路径中错误调用 force_set_pc（会不必要地修改 next_pc）

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-04 | 初始版本 | PTX-EMU Team |

## 参考

- [ADR-0002: PC 权威源统一到 WarpState](./ADR-0002-pc-unification.md)
- [GPU Pipeline 调研](../archive/misc/GPU-PIPELINE-RESEARCH.md)
