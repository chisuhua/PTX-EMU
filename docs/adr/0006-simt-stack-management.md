# ADR-0006: SIMT Stack 显式控制流管理

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 2 (SIMT Stack 实现) |
| **作者** | PTX-EMU Team |

## 上下文

在 SIMT（Single Instruction Multiple Thread）执行模型中，当 warp 内的线程出现分歧（divergence）时，需要一种机制来：

1. 跟踪哪些线程走了哪个分支路径
2. 确定所有线程何时重新汇合（reconvergence）
3. 支持嵌套分支（分支内还有分支）

早期 GPU 使用隐式的 reconvergence 机制（如 hardcode 的 join point），但现代 GPU（Volta+）需要显式的控制流栈来精确管理分歧。

## 决策驱动因素

1. **PTX ISA 语义正确性**：bra 指令的 reconvergence 点由 CFG post-dominator 确定，必须在运行时跟踪
2. **嵌套分支支持**：分支内部可以再有分支，需要栈结构管理
3. **Hopper/Blackwell 独立线程调度**：per-thread PC 需要显式的 SIMT 栈来协调
4. **调试可见性**：显式栈可以在运行时打印状态，便于调试

## 考虑的替代方案

### 方案 A: 隐式 Reconvergence（早期 GPU 方式）

**描述**: 不维护显式栈，通过硬编码的规则判断 reconvergence（如所有线程到达同一 PC）

**优点**:
- 实现简单
- 内存占用小

**缺点**:
- 无法处理嵌套分支
- 无法精确跟踪 divergent 执行路径
- 与现代 GPU 硬件行为不符

### 方案 B: CFG Post-Dominator 隐式推导

**描述**: 每次执行时从 CFG 推导 reconvergence 点，不维护运行时栈

**优点**:
- 无运行时状态维护

**缺点**:
- 每次查询 CFG 开销大
- 无法跟踪 active_mask 的变化
- 嵌套分支时状态丢失

### 方案 C: 显式 SIMT Stack (✅ 选中)

**描述**: 维护一个运行时栈，每个栈条目包含 branch_pc、reconvergence_pc、active_mask、return_mask

**优点**:
- 精确模拟硬件 SIMT 栈行为
- 支持任意深度的嵌套分支
- 运行时可查询栈状态（调试友好）
- reconvergence 检查高效（O(WARP_SIZE)）

**缺点**:
- 需要维护栈状态（push/pop）
- 栈深度限制（当前 MAX_DEPTH=32）

**选择理由**: 与 NVIDIA 硬件 SIMT 栈的实现方式一致，是精确模拟现代 GPU 控制流管理的必要组件。

## 决策内容

### 设计原则

1. **栈条目包含完整分支上下文**：branch_pc、reconvergence_pc、active_mask、return_mask、return_pc
2. **深度限制防止溢出**：MAX_DEPTH=32，足够覆盖实际 kernel 的嵌套深度
3. **收敛检查只跳过真正退出线程**：仅 `is_exited` 的 lane 跳过；
   `!is_active`（内存停顿、barrier 等待等瞬态失活）**不能**跳过，
   否则会过早弹出栈条目，导致失活的 lane 在恢复后被孤立（Fix 3，2026-06-25）

### 三个字段的角色分工（Fix 1 + Fix 3 后澄清）

`SIMTStackEntry` 有三个易混淆字段，**绝对不能互换使用**：

| 字段 | 含义 | 在哪里使用 |
|------|------|-----------|
| `active_mask` | 走了分支的 lane 子集 | `is_converged()` 的收敛判定循环 |
| `return_mask` | 整个分歧组（taken + not-taken） | gate `check_and_block_at_reconvergence_point()`；`check_reconvergence()` 弹出后恢复 `exec_mask` |
| `is_active` | lane 当前是否在 warp 中活跃 | `update_active_mask()` 双向同步（self-heal） |

**为什么这样分工**：
- 收敛判定只关心"走了分支的 lane 是否都到齐"——fall-through lane 本来就没分支
- gate 阻塞必须覆盖所有到达 reconv_pc 的 lane（包括 fall-through），
  否则 fall-through lane 会越过 reconv_pc 跑掉造成乱序（Fix 1）
- `exec_mask` 弹出后取 `return_mask`（整个分歧组），这样后续 PTX
  `activemask` 指令返回值包含所有应该一起执行的 lane

### 实现要点

```cpp
struct SIMTStackEntry {
    int branch_pc;              // 分支指令的 PC
    int reconvergence_pc;       // 汇合点 PC（CFG post-dominator）
    uint32_t active_mask;       // 分支后活跃的线程掩码
    uint32_t return_mask;       // 汇合后应恢复的线程掩码
    int return_pc;              // 汇合后继续执行的 PC
    
    bool is_converged(const std::array<ThreadState, 32>& threads) const {
        for (size_t i = 0; i < 32; i++) {
            if (active_mask & (1u << i)) {
                // Only skip lanes that have exited the kernel.
                // A lane that is temporarily inactive (e.g., memory stall,
                // blocked at barrier) is still part of the active convergence
                // group and must reach reconvergence_pc before we pop.
                if (threads[i].is_exited) {
                    continue;
                }
                if ((int)threads[i].pc != reconvergence_pc) {
                    return false;
                }
            }
        }
        return true;
    }
};

class SIMTStack {
    static constexpr size_t MAX_DEPTH = 32;
    std::vector<SIMTStackEntry> entries_;
    
    void push(const SIMTStackEntry& entry) {
        if (entries_.size() >= MAX_DEPTH) {
            throw ExecutionStateException("SIMT stack overflow");
        }
        entries_.push_back(entry);
    }
    
    SIMTStackEntry pop() { ... }
    bool check_reconvergence(const std::array<ThreadState, 32>& threads);
};
```

### while 循环收敛模式

在某些场景下（如 barrier release 后），可能有多个 SIMT 栈条目同时满足收敛条件。此时需要使用 while 循环直到无条目可 pop：

```cpp
// sm_context.cpp
// Check SIMT stack reconvergence after processing all divergent groups
// Loop until no more entries are convergent (barrier may resolve multiple entries)
if (next_warp && !next_warp->get_simt_stack().empty()) {
    while (next_warp->check_reconvergence()) {
        // Keep popping until no more convergent entries
    }
}

// warp_context.cpp
bool WarpContext::check_reconvergence() {
    if (simt_stack.empty()) return false;

    size_t depth_before = simt_stack.depth();
    simt_stack.check_reconvergence(warp_state.threads);

    if (simt_stack.depth() < depth_before) {
        // An entry was popped, update exec_mask.
        // IMPORTANT: use return_mask (full divergence group), not
        // active_mask (taken-subset), so that subsequent PTX `activemask`
        // returns all lanes that should now be executing together.
        if (simt_stack.empty()) {
            warp_state.exec_mask = 0xFFFFFFFF;  // All lanes converged
        } else {
            warp_state.exec_mask = simt_stack.top().return_mask;
        }
        return true;
    }
    return false;
}
```

**为什么需要 while 循环**：
- 嵌套分支场景中，barrier release 可能同时解除多层分支的阻塞
- 单次 check_reconvergence 只 pop 最顶层的收敛条目
- while 循环确保所有收敛条目都被处理

### handle_branch 中的 PC 过滤

为防止发散线程的过时 PC (stale PC) 影响 warp-level 分支决策，`handle_branch` 只处理 `pc == current_inst_pc` 的线程：

```cpp
void WarpContext::handle_branch(const std::string& predicate,
                               bool predicate_negated,
                               int target_pc,
                               int reconvergence_pc,
                               int current_inst_pc) {
    // ...

    for (int i = 0; i < 32; i++) {
        if (!warp_state.threads[i].is_active) continue;
        // PC 过滤：只处理当前指令 PC 的线程，防止 stale PC 影响分支决策
        if (warp_state.threads[i].pc != current_inst_pc) continue;

        bool should_branch = true;
        // ... predicate evaluation ...
    }
    // ...
}
```

**为什么需要 PC 过滤**：
- Per-thread PC 模型下，不同线程可能处于不同 PC
- 发散分支中，只有当前执行到分支指令的线程才应参与分支决策
- 防止其他路径的线程（stale PC）错误地影响 taken_mask/not_taken_mask 计算

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/simt_stack.h` | 新增 | SIMTStack 和 SIMTStackEntry 定义 |
| `src/ptxsim/core/simt_stack.cpp` | 新增 | 栈操作实现 |
| `include/ptxsim/warp_context.h` | 修改 | WarpContext 包含 simt_stack 成员 |
| `src/ptxsim/instructions/control.cpp` | 修改 | bra 指令 push SIMT 栈 |
| `src/ptxsm/core/sm_context.cpp` | 修改 | barrier 后调用 check_reconvergence |

## 后果

### 正面影响

- 精确模拟硬件 SIMT 控制流
- 支持任意深度的嵌套分支
- 调试时可打印栈状态

### 负面影响

- 栈深度限制（10 层），极端情况下可能溢出
- 每次分支/收敛需要维护栈状态

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| SIMT 栈溢出 | 极低 | 高 | MAX_DEPTH=32 足够覆盖实际 kernel；溢出时抛异常 |
| reconvergence 检查遗漏 | 低 | 高 | 单元测试覆盖嵌套分支场景 |
| active_mask 计算错误 | 中 | 高 | 与 CFG post-dominator 结果交叉验证 |

## 合规检查

后续相关开发应检查：

- [ ] 分支指令执行时正确 push SIMT 栈
- [ ] reconvergence 时正确 pop SIMT 栈
- [ ] **`is_converged` 只跳 `is_exited`，绝对不跳 `!is_active`**（Fix 3）
- [ ] **`is_converged` 循环使用 `active_mask`**（不是 `return_mask`）
- [ ] **gate 使用 `return_mask`** 阻塞所有到达 reconv_pc 的 lane（Fix 1）
- [ ] **`check_reconvergence` 弹出后 `exec_mask` 取新栈顶 `return_mask`**
- [ ] 三个字段（`active_mask` / `return_mask` / `is_active`）的角色分工清晰
- [ ] 栈深度不超过 MAX_DEPTH
- [ ] barrier 后使用 while 循环处理所有收敛条目
- [ ] check_reconvergence 返回 bool 表示是否有条目被 pop
- [ ] handle_branch 中使用 PC 过滤防止 stale PC 影响分支决策

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |
| 2026-05-06 | 添加 while 循环收敛模式说明、更新合规检查项 | PTX-EMU Team |
| 2026-05-06 | 添加 handle_branch PC 过滤说明、更新合规检查项 | PTX-EMU Team |
| 2026-06-19 | 同步 MAX_DEPTH 文档与代码（10 → 32，P1-1 quickwin） | Sisyphus |
| 2026-06-25 | Fix 3：`is_converged` 不再跳 `!is_active`；澄清三个字段（active_mask/return_mask/is_active）的角色分工；`exec_mask` 弹出后取 `return_mask`；修正 is_converged / check_reconvergence 的代码示例与决策原则 | Sisyphus |

## 参考

- [SIMT 架构文档](../architecture/SIMT-ARCHITECTURE-V2.md#321-simt-stack-entry)
- [GPGPU-Sim SIMT 分析](../architecture/GPGPU-SIM-SIMT-ANALYSIS.md)
- [NVIDIA PTX ISA 9.1 - Control Flow](../archive/ptx-instruction-reference/9.7.12_control_flow.md)
