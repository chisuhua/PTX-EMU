# SIMT v2.0 实施计划
## ——分阶段架构升级路线图

**版本**: 1.0  
**日期**: 2026-04-09  
**状态**: 待批准  
**基于**: SIMT-ARCHITECTURE-V2.md

---

## 📋 执行摘要

本实施计划详细描述了从当前架构（v1.0）升级到 SIMT v2.0 的分阶段路线图。整个升级预计需要 **9 个工作日**，分为 5 个 Phase，每个 Phase 都有明确的验收标准和回滚策略。

### 关键约束

- ✅ **向后兼容**: 每个 Phase 完成后，现有测试必须保持通过
- ✅ **可回滚**: 每个 Phase 在独立分支开发，可随时回滚
- ✅ **测试驱动**: 每个 Phase 必须有对应的测试验证
- ✅ **文档同步**: 代码和文档必须同步更新

---

## 🎯 Phase 1: CFG 分析基础设施

**预计时间**: 2 天  
**风险等级**: 中  
**依赖**: 无

### 1.1 任务分解

| ID | 任务 | 文件 | 预计 | 状态 |
|----|------|------|------|------|
| 1.1.1 | 基本块识别算法 | `src/ptx_parser/cfg_builder.h` | 4h | ⏳ |
| 1.1.2 | 控制流图构建 | `src/ptx_parser/cfg_builder.cpp` | 4h | ⏳ |
| 1.1.3 | Post-Dominator 计算 | `src/ptx_parser/cfg_builder.cpp` | 4h | ⏳ |
| 1.1.4 | 单元测试 | `tests/test_cfg_analysis.cpp` | 4h | ⏳ |

### 1.2 详细设计

#### 任务 1.1.1: 基本块识别

```cpp
// File: src/ptx_parser/cfg_builder.h
#pragma once

#include <vector>
#include <map>
#include <string>

namespace ptx {

struct BasicBlock {
    int id;
    int start_pc;
    int end_pc;                  // Exclusive
    std::string label_name;      // If this block starts with a label
    std::vector<int> successors; // IDs of successor blocks
    std::vector<int> predecessors;
    
    bool is_branch_target;       // Is this block a branch target?
    bool is_exit;                // Is this the exit block?
    
    // Helper methods
    int size() const { return end_pc - start_pc; }
    bool contains(int pc) const { return pc >= start_pc && pc < end_pc; }
};

class CFGBuilder {
public:
    struct CFG {
        std::vector<BasicBlock> blocks;
        int entry_block_id;
        int exit_block_id;
        
        BasicBlock* find_block_by_pc(int pc);
        BasicBlock* find_block_by_id(int id);
    };
    
    // Main entry point
    static CFG build(const std::vector<StatementContext>& statements);
    
private:
    // Step 1: Identify basic blocks
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<StatementContext>& statements);
    
    // Step 2: Build successor/predecessor edges
    static void buildEdges(CFG& cfg);
    
    // Helper: Find branch targets
    static std::set<int> findBranchTargets(
        const std::vector<StatementContext>& statements);
};

} // namespace ptx
```

**实现细节**:

```cpp
// File: src/ptx_parser/cfg_builder.cpp
#include "cfg_builder.h"

std::set<int> CFGBuilder::findBranchTargets(
    const std::vector<StatementContext>& statements) {
    
    std::set<int> targets;
    
    for (const auto& stmt : statements) {
        if (stmt.type == S_BRA) {
            const auto& branch = std::get<BranchInstr>(stmt.data);
            // Find PC of the target label
            int target_pc = find_label_pc(stmt.label2pc, branch.target);
            targets.insert(target_pc);
        }
    }
    
    return targets;
}

std::vector<BasicBlock> CFGBuilder::identifyBasicBlocks(
    const std::vector<StatementContext>& statements) {
    
    // Find all block boundaries
    std::set<int> boundaries;
    boundaries.insert(0);  // Entry
    boundaries.insert(statements.size());  // Exit
    
    // Branch targets are block starts
    auto targets = findBranchTargets(statements);
    boundaries.insert(targets.begin(), targets.end());
    
    // Branch instructions are block ends
    for (int i = 0; i < statements.size(); i++) {
        if (statements[i].type == S_BRA) {
            boundaries.insert(i + 1);  // Block ends after branch
        }
    }
    
    // Create basic blocks from boundaries
    std::vector<BasicBlock> blocks;
    int block_id = 0;
    int prev_boundary = 0;
    
    for (int boundary : boundaries) {
        if (boundary > prev_boundary) {
            BasicBlock block;
            block.id = block_id++;
            block.start_pc = prev_boundary;
            block.end_pc = boundary;
            block.is_branch_target = (targets.count(prev_boundary) > 0);
            block.is_exit = false;
            
            blocks.push_back(block);
        }
        prev_boundary = boundary;
    }
    
    return blocks;
}
```

#### 任务 1.1.3: Post-Dominator 计算

```cpp
// Reference: Cytron et al. "Simple and Efficient Construction of SSA"
std::map<int, int> CFGBuilder::computePostDominators(const CFG& cfg) {
    // Initialize post-dom sets
    std::map<int, std::set<int>> postDomSets;
    
    for (const auto& block : cfg.blocks) {
        if (block.id == cfg.exit_block_id) {
            postDomSets[block.id] = {block.id};
        } else {
            // Start with all blocks (will be refined)
            std::set<int> all_blocks;
            for (const auto& b : cfg.blocks) {
                all_blocks.insert(b.id);
            }
            postDomSets[block.id] = all_blocks;
        }
    }
    
    // Iterate until fixed point
    bool changed = true;
    while (changed) {
        changed = false;
        
        for (const auto& block : cfg.blocks) {
            if (block.id == cfg.exit_block_id) continue;
            
            // Post-dom = intersection of successors' post-dom sets + self
            std::set<int> newSet = {block.id};
            
            if (!block.successors.empty()) {
                for (int succ_id : block.successors) {
                    std::set<int> intersection;
                    std::set_intersection(
                        newSet.begin(), newSet.end(),
                        postDomSets[succ_id].begin(), postDomSets[succ_id].end(),
                        std::inserter(intersection, intersection.begin())
                    );
                    newSet = intersection;
                }
            }
            
            if (newSet != postDomSets[block.id]) {
                postDomSets[block.id] = newSet;
                changed = true;
            }
        }
    }
    
    // Extract immediate post-dominator
    std::map<int, int> result;
    for (const auto& block : cfg.blocks) {
        result[block.start_pc] = findImmediatePostDominator(cfg, block, postDomSets);
    }
    
    return result;
}
```

### 1.3 验收标准

```bash
# 1. 编译成功
cmake --build build --target ptx_parser

# 2. 单元测试通过
ctest -R cfg_analysis -V

# 3. 测试覆盖率 > 80%
lcov --capture --directory . --output-file coverage.info
genhtml coverage.info --output-directory coverage
# Check: lines: > 80%
```

### 1.4 回滚策略

```bash
# If Phase 1 fails, revert to backup:
git checkout main
git branch -D feature/simt-v2-phase1

# Or if using worktree:
cd ..
rm -rf ptx-phase1  # Remove worktree
```

---

## 🎯 Phase 2: SIMT Stack 实现

**预计时间**: 2 天  
**风险等级**: 中  
**依赖**: Phase 1 完成

### 2.1 任务分解

| ID | 任务 | 文件 | 预计 | 状态 |
|----|------|------|------|------|
| 2.1.1 | SIMTStackEntry 数据结构 | `include/ptxsim/simt_stack.h` | 2h | ⏳ |
| 2.1.2 | Stack 操作（push/pop） | `src/ptxsim/core/simt_stack.cpp` | 4h | ⏳ |
| 2.1.3 | Reconvergence 检查 | `src/ptxsim/core/simt_stack.cpp` | 4h | ⏳ |
| 2.1.4 | WarpContext 集成 | `include/ptxsim/warp_context.h` | 2h | ⏳ |
| 2.1.5 | 单元测试 | `tests/test_simt_stack.cpp` | 4h | ⏳ |

### 2.2 详细设计

#### 任务 2.1.1: SIMTStackEntry

```cpp
// File: include/ptxsim/simt_stack.h
#pragma once

#include <cstdint>
#include <vector>

namespace ptxsim {

struct SIMTStackEntry {
    // Branch information
    int branch_pc;              // PC of the branch instruction
    int reconvergence_pc;       // Where all lanes should converge (from CFG)
    
    // Execution masks
    uint32_t active_mask;       // Which lanes are active in this path
    uint32_t return_mask;       // Which lanes should be active after reconvergence
    
    // Reconvergence tracking
    int return_pc;              // Same as reconvergence_pc (for clarity)
    
    // Helper methods
    bool is_converged(const std::vector<ThreadState>& threads) const;
    uint32_t get_divergent_mask() const;
    
    // Debug printing
    std::string toString() const;
};

class SIMTStack {
public:
    // Stack operations
    void push(const SIMTStackEntry& entry);
    SIMTStackEntry pop();
    SIMTStackEntry& top();
    const SIMTStackEntry& top() const;
    
    bool empty() const;
    size_t depth() const;
    void clear();
    
    // Reconvergence check
    bool check_reconvergence(const std::vector<ThreadState>& threads);
    
    // Debug
    void print() const;
    
private:
    std::vector<SIMTStackEntry> entries;
};

} // namespace ptxsim
```

#### 任务 2.1.3: Reconvergence 检查

```cpp
// File: src/ptxsim/core/simt_stack.cpp
bool SIMTStackEntry::is_converged(const std::vector<ThreadState>& threads) const {
    // Check if all lanes in return_mask have reached reconvergence_pc
    for (int i = 0; i < 32; i++) {
        if (return_mask & (1u << i)) {
            if (threads[i].pc != reconvergence_pc) {
                return false;
            }
        }
    }
    return true;
}

bool SIMTStack::check_reconvergence(const std::vector<ThreadState>& threads) {
    if (entries.empty()) {
        return true;  // No pending reconvergence
    }
    
    SIMTStackEntry& top = entries.back();
    
    if (top.is_converged(threads)) {
        // All lanes converged - pop the stack
        entries.pop_back();
        
        PTX_DEBUG_EMU("[SIMT] Reconverged at PC=%d (was branch at PC=%d)",
                      top.reconvergence_pc, top.branch_pc);
        
        return true;
    }
    
    return false;
}
```

#### 任务 2.1.4: WarpContext 集成

```cpp
// File: include/ptxsim/warp_context.h
class WarpContext {
public:
    // Existing fields
    ThreadState threads[WARP_SIZE];
    uint32_t execution_mask;
    
    // NEW: SIMT Stack (Phase 2)
    SIMTStack simt_stack;
    
    // NEW: Convergence barrier
    Wbar convergence_barrier;
    
    // Methods
    void execute_warp_instruction(StatementContext& stmt);
    
private:
    // NEW: SIMT stack operations
    void push_branch(int branch_pc, int reconvergence_pc, uint32_t active_mask);
    bool check_reconvergence();
    
    // Existing
    void update_execution_mask();
};
```

### 2.3 验收标准

```bash
# 1. 单元测试通过
ctest -R simt_stack -V

# 2. 验证 stack 操作
./tests/test_simt_stack --reporters=compact

# 3. 验证 reconvergence
./tests/test_simt_reconvergence --reporters=compact
```

### 2.4 回滚策略

```bash
# Phase 2 失败时:
git checkout main
git branch -D feature/simt-v2-phase2

# 或者保留供后续参考:
git branch archive/simt-v2-phase2-failed
```

---

## 🎯 Phase 3: Per-Thread PC 集成

**预计时间**: 2 天  
**风险等级**: 高  
**依赖**: Phase 2 完成

### 3.1 任务分解

| ID | 任务 | 文件 | 预计 | 状态 |
|----|------|------|------|------|
| 3.1.1 | ThreadState 重构 | `include/ptxsim/thread_context.h` | 3h | ⏳ |
| 3.1.2 | WarpContext 执行逻辑更新 | `src/ptxsim/core/warp_context.cpp` | 5h | ⏳ |
| 3.1.3 | 调度器适配 | `src/ptxsim/core/sm_context.cpp` | 4h | ⏳ |
| 3.1.4 | Branch 指令处理 | `src/ptxsim/instructions/control.cpp` | 4h | ⏳ |
| 3.1.5 | 集成测试 | `tests/test_per_thread_pc.cpp` | 4h | ⏳ |

### 3.2 关键改动

#### 任务 3.1.2: WarpContext 执行逻辑

```cpp
// File: src/ptxsim/core/warp_context.cpp
void WarpContext::execute_warp_instruction(StatementContext& stmt) {
    // Step 1: Check reconvergence BEFORE executing
    if (check_reconvergence()) {
        // All lanes converged, execute in lockstep
        execute_lockstep(stmt);
    } else {
        // Divergent execution
        execute_divergent(stmt);
    }
}

void WarpContext::execute_lockstep(StatementContext& stmt) {
    // All lanes at same PC, execute together
    for (int i = 0; i < WARP_SIZE; i++) {
        if (threads[i].is_active && !threads[i].is_blocked) {
            threads[i].sync_from_warp_state();
            threads[i].execute_thread_instruction();
            threads[i].sync_to_warp_state();
        }
    }
    execution_mask = get_active_mask();
}

void WarpContext::execute_divergent(StatementContext& stmt) {
    // Lanes at different PCs, execute only matching lanes
    for (int i = 0; i < WARP_SIZE; i++) {
        if (threads[i].is_active && !threads[i].is_blocked) {
            if (threads[i].pc == stmt.pc) {
                // This lane should execute this instruction
                threads[i].sync_from_warp_state();
                threads[i].execute_thread_instruction();
                threads[i].sync_to_warp_state();
            }
        }
    }
    // Update execution mask based on which lanes executed
    execution_mask = compute_execution_mask_for_pc(stmt.pc);
}
```

#### 任务 3.1.4: Branch 指令处理

```cpp
// File: src/ptxsim/instructions/control.cpp
void BraHandler::executeBranch(ThreadContext* context, const BranchInstr& instr) {
    WarpContext* warp = context->warp_context_;
    
    // Step 1: Evaluate predicate for each lane
    uint32_t taken_mask = 0;
    uint32_t not_taken_mask = 0;
    
    for (int i = 0; i < WARP_SIZE; i++) {
        if (!warp->threads[i].is_active) continue;
        
        bool pred_value = context->evaluate_predicate(i, instr.predicate);
        if (instr.predicate_negated) pred_value = !pred_value;
        
        if (pred_value) {
            taken_mask |= (1u << i);
        } else {
            not_taken_mask |= (1u << i);
        }
    }
    
    // Step 2: Check for divergence
    bool is_divergent = (taken_mask != 0) && (not_taken_mask != 0);
    
    if (is_divergent) {
        // Step 3a: Push SIMT stack
        SIMTStackEntry entry;
        entry.branch_pc = context->pc;
        entry.reconvergence_pc = instr.reconvergence_pc;
        entry.active_mask = taken_mask;
        entry.return_mask = warp->execution_mask;
        entry.return_pc = instr.reconvergence_pc;
        
        warp->simt_stack.push(entry);
        
        PTX_DEBUG_EMU("[BRA:DIVERGE] branch_pc=%d reconvergence_pc=%d taken=0x%x",
                      entry.branch_pc, entry.reconvergence_pc, taken_mask);
        
        // Step 4a: Set per-thread PC
        int target_pc = context->get_label_pc(instr.target);
        for (int i = 0; i < WARP_SIZE; i++) {
            if (taken_mask & (1u << i)) {
                warp->threads[i].pc = target_pc;
            } else if (not_taken_mask & (1u << i)) {
                warp->threads[i].next_pc = context->pc + 1;
            }
        }
        
    } else {
        // Step 3b: Non-divergent
        int target_pc = (taken_mask != 0) ? context->get_label_pc(instr.target) 
                                          : context->pc + 1;
        for (int i = 0; i < WARP_SIZE; i++) {
            if (warp->threads[i].is_active) {
                warp->threads[i].pc = target_pc;
            }
        }
        
        PTX_DEBUG_EMU("[BRA:UNIFORM] all lanes to PC=%d", target_pc);
    }
}
```

### 3.3 验收标准

```bash
# 1. 现有测试保持通过
ctest -R "test_syncthreads|test_warp_barrier" -V

# 2. 新测试通过
ctest -R per_thread_pc -V

# 3. 分支测试
ctest -R branch_divergence -V
```

### 3.4 回滚策略

```bash
# Phase 3 涉及较大改动，建议完整测试后再合并
git checkout main
git branch -D feature/simt-v2-phase3

# 或者如果有问题:
git stash  # 保存当前工作
git checkout main
```

---

## 🎯 Phase 4: Barrier 增强

**预计时间**: 1 天  
**风险等级**: 低  
**依赖**: Phase 3 完成

### 4.1 任务分解

| ID | 任务 | 文件 | 预计 | 状态 |
|----|------|------|------|------|
| 4.1.1 | Wbar 与 SIMT stack 集成 | `include/ptxsim/wbar.h` | 2h | ⏳ |
| 4.1.2 | Memory fence 验证 | `src/ptxsim/instructions/barrier.cpp` | 4h | ⏳ |
| 4.1.3 | Debug 模式支持 | `configs/debug_config.ini` | 2h | ⏳ |
| 4.1.4 | Barrier 测试 | `tests/test_barrier_semantics.cpp` | 4h | ⏳ |

### 4.2 详细设计

#### 任务 4.1.2: Memory Fence 验证

```cpp
// File: src/ptxsim/instructions/barrier.cpp
#ifdef PTX_DEBUG
void BarWarpSyncHandler::verify_memory_fence(ThreadContext* context, 
                                              const BarrierInstr& instr) {
    WarpContext* warp = context->warp_context_;
    
    // Verify that all pre-barrier stores are visible
    for (int i = 0; i < WARP_SIZE; i++) {
        if (warp->convergence_barrier.participation_mask & (1u << i)) {
            // Check each lane's stores
            for (const auto& store : context->get_pre_barrier_stores(i)) {
                uint64_t addr = store.address;
                
                // Verify visibility to all participating lanes
                for (int j = 0; j < WARP_SIZE; j++) {
                    if (warp->convergence_barrier.participation_mask & (1u << j)) {
                        if (!warp->is_store_visible(i, j, addr)) {
                            PTX_ERROR("Barrier semantics violation: "
                                      "Lane %d store to 0x%lx not visible to lane %d",
                                      i, addr, j);
                        }
                    }
                }
            }
        }
    }
    
    PTX_DEBUG_EMU("[BAR:VERIFY] Memory fence verified for barrier %d", instr.barrier_id);
}
#endif

void BarWarpSyncHandler::execute(ThreadContext* context, const BarrierInstr& instr) {
    // ... existing barrier code ...
    
    // NEW: Verify memory fence (debug only)
    #ifdef PTX_DEBUG
    if (context->config.enable_barrier_verification) {
        verify_memory_fence(context, instr);
    }
    #endif
}
```

### 4.3 验收标准

```bash
# 1. Debug 模式验证通过
PTX_DEBUG=1 ctest -R barrier -V

# 2. 性能测试（无 regression）
./bench/test_syncthreads/test_syncthreads

# 3. 验证无 false positive
ctest -R barrier_semantics -V
```

### 4.4 回滚策略

```bash
# Phase 4 是可选增强，回滚简单
git checkout main
git branch -D feature/simt-v2-phase4
```

---

## 🎯 Phase 5: 测试与验证

**预计时间**: 2 天  
**风险等级**: 低  
**依赖**: Phase 1-4 完成

### 5.1 任务分解

| ID | 任务 | 文件 | 预计 | 状态 |
|----|------|------|------|------|
| 5.1.1 | 单元测试完善 | `tests/` | 4h | ⏳ |
| 5.1.2 | 集成测试 | `tests/integration/` | 4h | ⏳ |
| 5.1.3 | 性能基准 | `bench/` | 4h | ⏳ |
| 5.1.4 | 回归测试 | Full test suite | 4h | ⏳ |

### 5.2 测试用例

#### 新增测试用例

```cpp
// File: tests/test_simt_reconvergence.cpp
TEST_CASE("Branch reconvergence with divergent paths", "[simt][reconvergence]") {
    // Setup
    WarpContext warp;
    warp.init(/* ... */);
    
    // Create divergent branch
    // Lanes 0-15: Path A
    // Lanes 16-31: Path B
    
    // Execute until reconvergence
    while (!warp.simt_stack.empty()) {
        warp.execute_next_instruction();
    }
    
    // Verify all lanes at reconvergence PC
    for (int i = 0; i < 32; i++) {
        REQUIRE(warp.threads[i].pc == expected_reconvergence_pc);
    }
}

TEST_CASE("Barrier after divergent execution", "[simt][barrier]") {
    // Setup divergent paths with stores
    // Lane 0-15: store to shared[0-15]
    // Lane 16-31: store to shared[16-31]
    
    // Execute barrier
    execute_barrier();
    
    // Verify all stores visible
    for (int i = 0; i < 32; i++) {
        int value = read_shared(i);
        REQUIRE(value == expected_values[i]);
    }
}
```

### 5.3 性能基准

```bash
# Compare v1.0 vs v2.0
./scripts/benchmark.sh v1.0 > benchmark_v1.txt
./scripts/benchmark.sh v2.0 > benchmark_v2.txt

# Analyze
./scripts/compare_benchmarks.sh benchmark_v1.txt benchmark_v2.txt

# Acceptable regression: < 5%
```

### 5.4 验收标准

```bash
# All tests must pass
ctest --output-on-failure

# Coverage > 80%
lcov --summary coverage.info

# Performance regression < 5%
# (checked by compare_benchmarks.sh)
```

---

## 📊 项目追踪

### 分支策略

```bash
# Main development branch
git checkout -b feature/simt-v2

# Per-phase branches
git checkout -b feature/simt-v2-phase1
git checkout -b feature/simt-v2-phase2
# ...

# Merge back to main after each phase
git checkout main
git merge feature/simt-v2-phase1
```

### Git Worktree 建议

```bash
# Create isolated worktree for development
./zcf_git worktree add ../simt-v2-dev -b feature/simt-v2

# Work in isolation
cd ../simt-v2-dev
# ... development ...

# Back to main worktree
cd /workspace/project/PTX-EMU
./zcf_git worktree list
```

### 每日检查点

```bash
# End of each day
1. Commit all changes: git add -A && git commit -m "Day N progress"
2. Run tests: ctest --output-on-failure
3. Update progress: docs/architecture-upgrade/PROGRESS.md
4. Backup: git push origin feature/simt-v2
```

---

## ⚠️ 风险缓解

### 风险矩阵

| 风险 | 概率 | 影响 | 缓解措施 | 负责人 |
|------|------|------|---------|--------|
| CFG 构建错误 | 中 | 高 | 单元测试覆盖所有分支模式 | TBD |
| 性能回归 > 5% | 低 | 中 | 每 Phase 后 benchmark 对比 | TBD |
| 测试失败增加 | 中 | 高 | 详细日志，快速定位 | TBD |
| 需要大规模重构 | 低 | 高 | 分 Phase 开发，每阶段可回滚 | TBD |
| 时间超支 | 中 | 中 | 每日检查，及时调整 | TBD |

### 应急计划

**如果 Phase N 失败**:

```bash
1. 停止当前开发
2. 分析失败原因 (docs/architecture-upgrade/FAILURE-ANALYSIS.md)
3. 决定：
   a. 修复后重试 (estimated time: X hours)
   b. 修改设计方案 (需要架构评审)
   c. 跳过该 Phase，继续下一 Phase
   d. 回滚到 Phase N-1
```

---

## 📋 批准检查清单

在开始编码前，请确认以下内容：

- [ ] **架构文档审查**: SIMT-ARCHITECTURE-V2.md 已被审阅和批准
- [ ] **实施计划审查**: 本计划已被审阅和批准
- [ ] **备份分支**: 已创建备份分支 `backup/pre-simt-v2`
- [ ] **测试基线**: 当前所有测试已通过
- [ ] **性能基线**: Benchmark 已运行并记录
- [ ] **时间承诺**: 确认有 9 个工作日的开发时间
- [ ] **回滚计划**: 理解并准备好回滚策略

---

**批准人签名**: _________________________  
**批准日期**: _________________________  
**预计开始日期**: _________________________  
**预计完成日期**: _________________________

---

## 附录：决策日志

### 决策 1: 是否启用完整 CFG 分析？

**选项**:
- A: 完整 CFG 分析 (准确度高，复杂度中)
- B: 简化 Heuristic (准确度中，复杂度低)

**决策**: **选项 A** - 完整 CFG 分析

**理由**: 
- 简化 heuristic 可能在边缘场景失败
- CFG 分析是一次性开销（kernel 加载时），无运行时开销
- 符合 "匹配真实硬件" 的目标

### 决策 2: Barrier 验证是否总是启用？

**选项**:
- A: Always On (完整保护，性能开销 5-10%)
- B: Debug Only (开发友好，Production 无开销)
- C: 完全禁用 (最快，无保护)

**决策**: **选项 B** - Debug Only

**理由**:
- Barrier 语义错误是开发期 bug，不应出现在 production
- PTX_DEBUG 模式已足够覆盖开发和测试场景
- 性能敏感场景不应承担额外开销

### 决策 3: Per-Thread PC 实现粒度？

**选项**:
- A: 显式 ThreadState 数组 (内存占用大，访问快)
- B: 隐式通过 SIMT stack (内存占用小，访问稍慢)

**决策**: **选项 A** - 显式 ThreadState 数组

**理由**:
- Hopper/Blackwell 支持 Independent Thread Scheduling
- 显式数组更容易调试和验证
- 内存开销可接受 (32 threads * sizeof(ThreadState) ≈ 2KB per warp)

---

**文档结束**
