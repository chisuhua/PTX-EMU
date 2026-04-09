# Phase 9: SIMT Stack 执行流程集成

**日期**: 2026-04-11  
**阶段**: Phase 9 - SIMT Stack 执行集成  
**状态**: 开始实施  
**预计时间**: 6-8 小时

---

## 目标

将 SIMT Stack 与分支执行流程完全集成，实现正确的分支发散和收敛管理。

---

## 关键修改点

### 1. BraHandler 修改

**文件**: `src/ptxsim/instructions/control.cpp`

**当前实现**:
```cpp
void BraHandler::executeBranch(ThreadContext* context, ...) {
    // 简单跳转，无 SIMT stack 管理
    if (predicate) {
        context->pc = target_pc;
    }
}
```

**修改后实现**:
```cpp
void BraHandler::executeBranch(ThreadContext* context, ...) {
    WarpContext* warp = context->warp_context_;
    
    // 1. 评估所有 lanes 的谓词
    uint32_t taken_mask = compute_taken_mask();
    uint32_t not_taken_mask = ~taken_mask;
    
    // 2. 检查是否发散
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    
    if (is_divergent) {
        // 3. Push SIMT stack
        SIMTStackEntry entry;
        entry.branch_pc = context->pc;
        entry.reconvergence_pc = branch.reconvergence_pc;  // 使用 Phase 5 计算的值
        entry.active_mask = taken_mask;
        entry.return_mask = warp->warp_state.exec_mask;
        warp->simt_stack.push(entry);
        
        // 4. 设置 per-thread PC
        for (int i = 0; i < 32; i++) {
            if (taken_mask & (1u << i)) {
                warp->set_thread_pc(i, target_pc);
            }
        }
    } else {
        // Convergent branch - all lanes take same path
        for (int i = 0; i < 32; i++) {
            warp->set_thread_pc(i, target_pc);
        }
    }
}
```

---

### 2. WarpContext 修改

**文件**: `include/ptxsim/warp_context.h`, `src/ptxsim/core/warp_context.cpp`

**新增方法**:
```cpp
class WarpContext {
public:
    // Push SIMT stack entry for divergent branch
    void push_simt_stack(const SIMTStackEntry& entry);
    
    // Check if all lanes have reconverged
    bool check_reconvergence();
    
    // Update exec_mask based on simt_stack
    void update_exec_mask();
};
```

**修改 execute_warp_instruction**:
```cpp
void WarpContext::execute_warp_instruction(StatementContext& stmt) {
    // Check reconvergence before executing non-branch instructions
    if (stmt.type != S_BRA) {
        if (check_reconvergence()) {
            // All lanes converged - pop SIMT stack
            simt_stack.pop();
            update_exec_mask();
        }
    }
    
    // Execute for active lanes only
    for (int i = 0; i < WARP_SIZE; i++) {
        if (is_lane_active(i)) {
            threads[i]->execute_thread_instruction();
        }
    }
}
```

---

### 3. SIMTStackEntry 增强

**文件**: `include/ptxsim/simt_stack.h`

**新增字段**:
```cpp
struct SIMTStackEntry {
    int branch_pc;
    int reconvergence_pc;  // Already exists (Phase 5)
    uint32_t active_mask;  // Which lanes are active in this path
    uint32_t return_mask;  // Which lanes should be active after reconvergence
    int return_pc;         // PC to resume at reconvergence
};
```

---

## 实施步骤

### Step 1: 增强 SIMTStackEntry (30 min)
- [ ] 添加必要字段（如缺失）
- [ ] 添加构造函数
- [ ] 添加 is_converged() 方法

### Step 2: 修改 BraHandler (2 hours)
- [ ] 添加 divergence detection
- [ ] 添加 SIMT stack push logic
- [ ] 添加 per-thread PC setting
- [ ] 添加详细日志

### Step 3: 修改 WarpContext (2 hours)
- [ ] 添加 check_reconvergence() 方法
- [ ] 添加 update_exec_mask() 方法
- [ ] 修改 execute_warp_instruction()
- [ ] 添加收敛验证

### Step 4: 创建集成测试 (2 hours)
- [ ] 创建 SIMT stack 集成测试
- [ ] 创建发散执行测试
- [ ] 创建收敛验证测试
- [ ] 创建 barrier 同步测试

### Step 5: 运行验证 (1-2 hours)
- [ ] 编译验证
- [ ] 运行现有测试（确保无回归）
- [ ] 运行新集成测试
- [ ] 记录测试结果

---

## 测试计划

### 测试用例 1: Simple Divergent Branch

```ptx
@%p1 bra $L_then;  // Divergent: lanes 0-15 take, 16-31 fall through
L_then:
// Lanes should reconverge here
```

**Expected**:
- SIMT stack pushed at bra
- reconvergence_pc correctly set
- All lanes reconverge at merge point

---

### 测试用例 2: Nested Divergent Branch

```ptx
@%p1 bra $L_outer;
@%p2 bra $L_inner;
L_inner:
L_outer:
```

**Expected**:
- Two SIMT stack entries
- Correct reconvergence for each level
- Proper stack unwinding

---

### 测试用例 3: Divergent + Barrier

```ptx
@%p1 bra $L_then;
L_then:
st.shared.u32 [%r1], value;
bar.sync 0;  // Must wait for all lanes
```

**Expected**:
- All stores visible after barrier
- Barrier waits for all lanes
- No race conditions

---

## 验收标准

### 功能验收

- [ ] 发散分支正确 push SIMT stack
- [ ] reconvergence_pc 正确使用
- [ ] 收敛时正确 pop SIMT stack
- [ ] exec_mask 正确更新
- [ ] barrier 同步正确

### 回归验证

- [ ] 现有 12 测试全部通过
- [ ] 无性能回归
- [ ] 无内存泄漏

### 新增测试

- [ ] Simple divergent branch test PASS
- [ ] Nested divergent branch test PASS
- [ ] Divergent + barrier test PASS

---

## 风险评估

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| BraHandler 修改复杂 | 中 | 高 | 分步实施，逐步验证 |
| Reconvergence 检测错误 | 中 | 高 | 详细日志，单元测试 |
| exec_mask 更新错误 | 中 | 中 | 添加验证，测试覆盖 |
| 性能回归 | 低 | 中 | Phase 8 已验证 CFG 开销低 |

---

## 时间表

| 任务 | 预计时间 | 状态 |
|------|---------|------|
| Step 1: SIMTStackEntry 增强 | 30 min | ⏳ |
| Step 2: BraHandler 修改 | 2 hours | ⏳ |
| Step 3: WarpContext 修改 | 2 hours | ⏳ |
| Step 4: 集成测试创建 | 2 hours | ⏳ |
| Step 5: 验证与修复 | 1-2 hours | ⏳ |
| **总计** | **7.5-8 hours** | - |

---

**状态**: 准备开始 Step 1  
**下一步**: 增强 SIMTStackEntry
