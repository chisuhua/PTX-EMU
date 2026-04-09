# SIMT v2.0 Phase 5-6 重新规划 (Revised)

**版本**: 2.0 (Revised)  
**日期**: 2026-04-10  
**状态**: Ready for Execution  
**审查者**: PTX-EMU Architecture Team

---

## 执行摘要

本文档定义了 SIMT v2.0 架构的 Phase 5-6 实施计划，重点解决 CFG 分析集成问题。基于对任务依赖关系的深入分析，我们识别出 CFG 分析是 reconvergence PC 计算的前提条件，必须优先完成。

**关键决策**:
1. ✅ CFG Builder 必须在 Phase 5.1 完成编译集成
2. ✅ reconvergence_pc 填充必须在 Phase 5.3 完成
3. ✅ 所有分支测试必须在 Phase 5.4 通过
4. ✅ Phase 6 专注于完整验证而非修复

**预计完成时间**: 40 小时 (5 个工作日)  
**风险等级**: 中 (有降级方案)

---

## 依赖关系深度分析

### 依赖链 1: CFG 分析 → reconvergence PC

```
┌────────────────────────────────────────────────────────────┐
│  CFG Builder Source Code                                   │
│  - cfg_builder.h (类型定义)                                │
│  - cfg_builder.cpp (算法实现)                              │
└────────────────────────────────────────────────────────────┘
                            ↓ 编译 (CMake)
┌────────────────────────────────────────────────────────────┐
│  libptx_parser.so                                          │
│  - 必须包含 cfg_builder.cpp                                │
│  - 导出 CFGBuilder 符号                                    │
└────────────────────────────────────────────────────────────┘
                            ↓ 链接
┌────────────────────────────────────────────────────────────┐
│  libcudart.so (ptx_interpreter.cpp)                        │
│  - 调用 CFGBuilder::build()                                │
│  - 调用 CFGBuilder::computePostDominators()                │
└────────────────────────────────────────────────────────────┘
                            ↓ 执行 (Kernel 加载)
┌────────────────────────────────────────────────────────────┐
│  CFG Analysis Results                                      │
│  - CFG: BasicBlocks, edges                                 │
│  - PostDominatorMap: PC → reconvergence_pc                 │
└────────────────────────────────────────────────────────────┘
                            ↓ 更新
┌────────────────────────────────────────────────────────────┐
│  BranchInstr.reconvergence_pc                              │
│  - 从 -1 更新为实际 PC 值                                    │
│  - 用于 SIMT Stack push                                    │
└────────────────────────────────────────────────────────────┘
```

### 依赖链 2: reconvergence PC → 分支收敛

```
BranchInstr.reconvergence_pc (正确值)
    ↓
BraHandler::executeBranch() 使用 reconvergence_pc
    ↓
WarpContext::handle_branch() push SIMTStackEntry
    ↓
WarpContext::execute_warp_instruction() check_reconvergence
    ↓
SIMTStack::check_reconvergence() 返回 true
    ↓
收敛成功，exec_mask 恢复
```

### 关键识别

**问题点**: 当前实现中，依赖链 1 在第 2 步断开
- libptx_parser.so **不包含** cfg_builder.cpp
- libcudart.so **无法链接** CFGBuilder 符号

**解决方案**: Phase 5.1 专门修复编译链接问题

---

## 影响评估

### 受影响的测试用例

| 测试 | 当前状态 | 预期状态 | 依赖 |
|------|---------|---------|------|
| test_ptx_bra | ❌ FAIL | ✅ PASS | reconvergence_pc |
| test_syncthreads | ❌ FAIL | ✅ PASS | reconvergence_pc |
| test_warp_divergence | ❌ FAIL | ✅ PASS | reconvergence_pc |
| test_cfg_analysis | ⏳ Pending | ✅ PASS | CFG Builder |
| test_simt_thread_pc | ✅ PASS | ✅ PASS | 独立功能 |
| test_warp_barrier | ✅ PASS | ✅ PASS | 独立功能 |

### 不受影响的功能

以下功能**已验证可用**，不依赖 CFG 分析：
- ✅ Per-Thread PC 管理
- ✅ SIMT Stack 数据结构
- ✅ WarpContext::handle_branch()
- ✅ Barrier 基本功能
- ✅ Scheduler 配置

**结论**: 62.5% 的核心 SIMT 功能已验证可用 (5/8 测试通过)

---

## Phase 5: 基础集成 (Revised)

### Phase 5.0: 准备与审查 (2 小时) ⭐ NEW

**目标**: 确认当前状态，准备实施环境

**任务**:
- [ ] 5.0.1 审查 cfg_builder.h/cpp 当前代码
- [ ] 5.0.2 识别类型不匹配问题
- [ ] 5.0.3 准备备份分支
- [ ] 5.0.4 创建测试基线

**验收标准**:
```bash
# 创建备份分支
git checkout -b backup/pre-phase5

# 记录当前测试状态
ctest -R "simt|bra|syncthreads" --output-on-failure > test_baseline.txt
```

**输出文件**:
- `docs/architecture-upgrade/CODE-REVIEW-CFG.md` - 代码审查报告
- `tests/baseline/test_baseline.txt` - 测试基线

---

### Phase 5.1: CFG Builder 编译集成 (6 小时)

**目标**: 使 CFG Builder 正确编译并链接

#### 任务 5.1.1: 修复 cfg_builder.h 类型定义 (2 小时)

**当前问题**:
```cpp
// Include path issue
#include "ptx_ir/statement_context.h"  // May not be found

// Missing type definitions
struct StatementContext;  // Forward declaration needed?
```

**修复步骤**:
1. 检查 include 路径
2. 添加必要的前向声明
3. 确保类型完整性

**修复后代码** (cfg_builder.h):
```cpp
#pragma once

#include <vector>
#include <map>
#include <set>
#include <string>

// Forward declarations
namespace ptx {
struct StatementContext;  // Forward declare
}

namespace ptx {
namespace cfg {

struct BasicBlock {
    int id;
    int start_pc;
    int end_pc;
    std::vector<int> successors;
    std::vector<int> predecessors;
    bool is_branch_target;
    bool is_exit;
    
    int size() const { return end_pc - start_pc; }
    bool contains(int pc) const { return pc >= start_pc && pc < end_pc; }
};

struct CFG {
    std::vector<BasicBlock> blocks;
    int entry_block_id;
    int exit_block_id;
    
    BasicBlock* find_block_by_pc(int pc);
    BasicBlock* find_block_by_id(int id);
    void print() const;
};

using PostDominatorMap = std::map<int, int>;

class CFGBuilder {
public:
    static CFG build(
        const std::vector<ptx::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static PostDominatorMap computePostDominators(const CFG& cfg);
    
private:
    static std::vector<BasicBlock> identifyBasicBlocks(
        const std::vector<ptx::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static std::set<int> findBranchTargets(
        const std::vector<ptx::StatementContext>& statements,
        const std::map<std::string, int>& label2pc);
    
    static void buildEdges(CFG& cfg,
                          const std::vector<ptx::StatementContext>& statements);
};

} // namespace cfg
} // namespace ptx
```

**验收标准**:
```bash
g++ -c src/ptx_parser/cfg_builder.cpp -I./include -std=c++17
# 预期：cfg_builder.o generated, 0 errors
```

---

#### 任务 5.1.2: 修复 cfg_builder.cpp 实现 (2 小时)

**当前问题**:
```cpp
// Missing includes
#include "ptx_ir/statement_context.h"
#include "ptx_ir/symtable.h"

// Incomplete implementations
```

**修复步骤**:
1. 添加必要 includes
2. 补全方法实现
3. 移除未使用的代码

**验收标准**:
```bash
g++ -c src/ptx_parser/cfg_builder.cpp -I./include -std=c++17
# 预期：0 errors, 0 warnings
```

---

#### 任务 5.1.3: 更新 CMakeLists.txt (1 小时)

**当前问题**: cfg_builder.cpp 未添加到任何库

**修复方案**:
```cmake
# File: src/CMakeLists.txt

# Option 1: Add to ptx_parser (recommended)
add_library(ptx_parser SHARED
    ptx_parser/ptx_visitor.cpp
    ptx_parser/cfg_builder.cpp  # ADD THIS
)

# Option 2: Create separate library
add_library(ptx_cfg_builder SHARED
    ptx_parser/cfg_builder.cpp
)
target_link_libraries(ptx_cfg_builder ptx_ir)
```

**推荐**: Option 1 (更简单，减少链接复杂度)

**验收标准**:
```bash
cmake --build build --target ptx_parser
# 预期：100% Built target ptx_parser
```

---

#### 任务 5.1.4: 验证完整编译 (1 小时)

**验收标准**:
```bash
# Full rebuild
cd build && rm -rf * && cmake .. && cmake --build . -j$(nproc)

# Expected output:
# [60%] Built target ptx_ir
# [68%] Built target ptx_parser  # Should include cfg_builder
# [71%] Built target cudart
# [100%] Built target all
```

**回滚方案**:
```bash
# If compilation fails, revert to backup
git checkout backup/pre-phase5
```

---

### Phase 5.2: Kernel 加载流程集成 (6 小时)

**目标**: 在 kernel 加载完成后执行 CFG 分析

#### 任务 5.2.1: 添加头文件包含 (1 小时)

**文件**: `src/cudart/ptx_interpreter.cpp`

```cpp
// Add after existing includes
#include "ptx_parser/cfg_builder.h"  // SIMT v2.0 CFG analysis
```

**验收标准**: 头文件路径正确，无编译错误

---

#### 任务 5.2.2: 在 setupLabels() 中添加 CFG 调用 (3 小时)

**文件**: `src/cudart/ptx_interpreter.cpp`  
**函数**: `PtxInterpreter::setupLabels()`

**当前代码** (简化):
```cpp
void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        const auto &e = kernelContext->kernelStatements[i];
        if (e.type == S_DOLLOR) {
            const auto &s = std::get<DollarNameInstr>(e.data);
            label2pc[s.name] = i;
        }
    }
}
```

**修复后代码**:
```cpp
void PtxInterpreter::setupLabels(std::map<std::string, int> &label2pc) {
    // Existing label setup
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        const auto &e = kernelContext->kernelStatements[i];
        if (e.type == S_DOLLOR) {
            const auto &s = std::get<DollarNameInstr>(e.data);
            label2pc[s.name] = i;
        }
    }
    PTX_INFO("Total labels registered: %zu", label2pc.size());
    
    // NEW: CFG analysis for reconvergence PC (SIMT v2.0)
    PTX_INFO("Running CFG analysis for reconvergence PC computation...");
    
    try {
        // Build CFG
        ptx::cfg::CFG cfg = ptx::cfg::CFGBuilder::build(
            kernelContext->kernelStatements, 
            label2pc
        );
        
        // Compute post-dominators
        ptx::cfg::PostDominatorMap postDoms = 
            ptx::cfg::CFGBuilder::computePostDominators(cfg);
        
        // Update BranchInstr with reconvergence_pc
        int updated_count = 0;
        for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
            auto &stmt = kernelContext->kernelStatements[i];
            if (stmt.type == S_BRA) {
                auto &branch = std::get<ptx::BranchInstr>(stmt.data);
                
                auto it = postDoms.find(i);
                if (it != postDoms.end() && it->second >= 0) {
                    branch.reconvergence_pc = it->second;
                    updated_count++;
                    PTX_DEBUG("Branch at PC=%d: reconvergence_pc=%d", 
                             i, it->second);
                } else {
                    // Fallback: use next instruction
                    branch.reconvergence_pc = i + 1;
                    PTX_DEBUG("Branch at PC=%d: no post-dom, using %d", 
                             i, i + 1);
                }
            }
        }
        
        PTX_INFO("CFG analysis complete: updated %d branch instructions", 
                updated_count);
                
    } catch (const std::exception& e) {
        PTX_ERROR("CFG analysis failed: %s", e.what());
        // Fallback: leave reconvergence_pc as -1
    }
}
```

**关键点**:
1. 使用 try-catch 处理异常 (CFG 分析可能失败)
2. 添加详细日志输出
3. 提供 fallback (reconvergence_pc = i + 1)

**验收标准**: 编译通过，无链接错误

---

#### 任务 5.2.3: 更新 ptx_parser 库链接 (1 小时)

**文件**: `src/CMakeLists.txt`

**确保 cudart 链接 ptx_parser**:
```cmake
add_library(cudart SHARED ${SOURCES})
target_link_libraries(cudart ptx_ir ptx_parser ptxsim)  # Must include ptx_parser
```

**验收标准**:
```bash
cmake --build build --target cudart
# 预期：无 undefined reference 错误
```

---

#### 任务 5.2.4: 测试编译 (1 小时)

**验收标准**:
```bash
# Full build test
cd build && cmake --build . -j$(nproc) 2>&1 | grep -E "error:|warning:"
# 预期：0 errors, warnings acceptable if not related to CFG
```

---

### Phase 5.3: reconvergence_pc 填充验证 (4 小时)

**目标**: 验证 reconvergence_pc 被正确填充

#### 任务 5.3.1: 添加调试日志 (2 小时)

**文件**: `src/cudart/ptx_interpreter.cpp`

在 CFG 分析后添加:
```cpp
PTX_INFO("=== Reconvergence PC Summary ===");
for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
    const auto &stmt = kernelContext->kernelStatements[i];
    if (stmt.type == S_BRA) {
        const auto &branch = std::get<ptx::BranchInstr>(stmt.data);
        PTX_INFO("PC=%d: reconvergence_pc=%d", i, branch.reconvergence_pc);
    }
}
```

**验收标准**: 日志输出显示 reconvergence_pc > 0

---

#### 任务 5.3.2: 运行简单分支测试 (2 小时)

**测试用例**: `test_ptx_bra`

**预期输出**:
```
=== Reconvergence PC Summary ===
PC=10: reconvergence_pc=20
PC=15: reconvergence_pc=25
...
CFG analysis complete: updated N branch instructions
```

**验收标准**: reconvergence_pc 非 -1

---

### Phase 5.4: 单元测试修复验证 (4 小时)

**目标**: 验证修复后的测试通过

#### 任务 5.4.1: 运行失败的测试 (2 小时)

```bash
cd build && ctest -R "ptx_bra|syncthreads" --output-on-failure -V
```

**预期结果**:
```
Test project /workspace/project/PTX-EMU/build
    Start 12: test_ptx_bra
1/2 Test #12: test_ptx_bra .....................   Passed    0.30 sec
    Start 28: test_syncthreads
2/2 Test #28: test_syncthreads .................   Passed    0.40 sec

100% tests passed, 0 tests failed out of 2
```

---

#### 任务 5.4.2: 运行完整 SIMT 测试套件 (2 小时)

```bash
cd build && ctest -R "simt|warp|barrier|bra" --output-on-failure
```

**预期结果**: 8/8 tests pass (之前 5/8)

---

### Phase 5.5: CFG 单元测试 (4 小时)

**目标**: 验证 CFG Builder 本身正确性

#### 任务 5.5.1: 创建 test_cfg_builder.cpp (2 小时)

```cpp
#include <catch2/catch.hpp>
#include "ptx_parser/cfg_builder.h"
#include "ptx_ir/statement_context.h"

using namespace ptx::cfg;

TEST_CASE("CFG Builder - Basic Block Identification", "[cfg]") {
    std::vector<ptx::StatementContext> statements(5);
    std::map<std::string, int> label2pc;
    label2pc["target"] = 3;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 1);
    REQUIRE(cfg.entry_block_id == 0);
}

TEST_CASE("CFG Builder - Post-Dominator", "[cfg]") {
    std::vector<ptx::StatementContext> statements(5);
    std::map<std::string, int> label2pc;
    label2pc["merge"] = 4;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    
    REQUIRE(postDoms.size() > 0);
    // All PCs should have a post-dominator
    for (const auto& [pc, postDom] : postDoms) {
        REQUIRE(postDom >= 0);
    }
}
```

---

#### 任务 5.5.2: 运行 CFG 测试 (2 小时)

```bash
cd build && ctest -R cfg_builder --output-on-failure -V
```

**预期**: 2/2 tests pass

---

### Phase 5 完成验收

**验收清单**:
- [ ] 5.1 CFG Builder 编译通过 ✅
- [ ] 5.2 Kernel 集成完成 ✅
- [ ] 5.3 reconvergence_pc 填充验证 ✅
- [ ] 5.4 单元测试修复 (8/8 pass) ✅
- [ ] 5.5 CFG 单元测试 (2/2 pass) ✅

**完成标准**: 10/10 tests pass

---

## Phase 6: 完整验证 (Revised)

### Phase 6.1: 分支收敛集成测试 (6 小时)

**目标**: 测试完整的 SIMT 分支收敛流程

#### 测试用例 6.1.1: 简单 if-else (2 小时)

```cpp
__global__ void test_simple_branch() {
    int tid = threadIdx.x;
    if (tid < 16) {
        shared_data[tid] = tid;
    } else {
        shared_data[tid] = tid * 2;
    }
    __syncthreads();
    // Verify all values correct
}
```

**验收标准**: 测试通过，值正确

---

#### 测试用例 6.1.2: 嵌套分支 (2 小时)

```cpp
__global__ void test_nested_branch() {
    int tid = threadIdx.x;
    if (tid < 8) {
        if (tid % 2 == 0) {
            // Path A1
        } else {
            // Path A2
        }
    } else {
        // Path B
    }
    __syncthreads();
}
```

**验收标准**: 嵌套分支正确收敛

---

#### 测试用例 6.1.3: 分支 + barrier (2 小时)

```cpp
__global__ void test_branch_barrier() {
    int tid = threadIdx.x;
    if (tid < 16) {
        shared_data[tid] = tid;
    } else {
        shared_data[tid] = tid * 2;
    }
    __syncthreads();  // Must wait for all paths
    // Verify all values visible
}
```

**验收标准**: barrier 正确等待所有路径

---

### Phase 6.2: 性能基准测试 (6 小时)

**目标**: 验证无性能回归

#### 基准测试 6.2.1: test_syncthreads (2 小时)

```bash
# Baseline (before CFG integration)
./bench/test_syncthreads/test_syncthreads > baseline.txt

# After CFG integration
./bench/test_syncthreads/test_syncthreads > after.txt

# Compare
diff baseline.txt after.txt
# 预期：性能差异 < 5%
```

---

#### 基准测试 6.2.2: test_warp_divergence (2 小时)

**关键指标**:
- 执行时间
- 内存使用
- 指令计数

**验收标准**: 性能回归 < 5%

---

#### 基准测试 6.2.3: 2Dentropy (真实工作负载) (2 小时)

```bash
./bench/2Dentropy/2Dentropy
# 预期：执行时间 < 5% regression
```

---

### Phase 6.3: 代码质量审查 (6 小时)

#### 任务 6.3.1: 编译警告检查 (2 小时)

```bash
cmake -DCMAKE_CXX_FLAGS="-Wall -Wextra -Werror" ..
cmake --build build
# 预期：0 warnings, 0 errors
```

---

#### 任务 6.3.2: LSP 诊断检查 (2 小时)

**验收标准**:
- 0 LSP errors
- LSP warnings < 5 (必须是非关键问题)

---

#### 任务 6.3.3: 内存泄漏检查 (2 小时)

```bash
valgrind --leak-check=full ./bin/tests/test_warp_divergence
# 预期：0 bytes lost, 0 leaks
```

---

### Phase 6.4: 文档完善 (6 小时)

#### 任务 6.4.1: 更新 SIMT-ARCHITECTURE-V2.md (2 小时)

**更新内容**:
- Phase 5.1-5.5 实施细节
- CFG 分析集成说明
- reconvergence_pc 计算流程

---

#### 任务 6.4.2: 创建 CFG_ANALYSIS_GUIDE.md (2 小时)

**内容大纲**:
1. CFG Builder 使用指南
2. Post-Dominator 计算原理
3. reconvergence_pc 配置
4. 故障排查

---

#### 任务 6.4.3: 更新 PROGRESS-V2.md (2 小时)

**更新内容**:
- Phase 5 完成状态
- Phase 6 进展
- 测试覆盖率统计

---

### Phase 6.5: 最终验收 (6 小时)

#### 任务 6.5.1: 完整测试套件 (3 小时)

```bash
cd build && ctest --output-on-failure
# 预期：100% tests passed (之前 62.5%)
```

**完整测试清单**:
- [ ] test_ptx_bra ✅
- [ ] test_syncthreads ✅
- [ ] test_warp_divergence ✅
- [ ] test_simt_thread_pc ✅
- [ ] test_warp_barrier_extended ✅
- [ ] test_scheduler_config ✅
- [ ] test_warp_context ✅
- [ ] test_warp_scheduler ✅
- [ ] test_cfg_builder ✅

---

#### 任务 6.5.2: 性能验收 (2 小时)

**验收标准**:
- 所有基准测试性能回归 < 5%
- 内存使用无显著增长 (< 10%)

---

#### 任务 6.5.3: 文档验收 (1 小时)

**验收清单**:
- [ ] SIMT-ARCHITECTURE-V2.md 更新 ✅
- [ ] CFG_ANALYSIS_GUIDE.md 创建 ✅
- [ ] PROGRESS-V2.md 更新 ✅
- [ ] Phase 5-6 测试报告 ✅

---

## 时间估算 (Revised)

| Phase | 任务 | 原估算 | 修订估算 | 说明 |
|-------|------|--------|---------|------|
| **5.0** | 准备与审查 | - | 2h | 新增，确保充分准备 |
| **5.1** | CFG 编译 | 4h | 6h | 增加类型问题调试时间 |
| **5.2** | Kernel 集成 | 4h | 6h | 增加异常处理 |
| **5.3** | reconvergence_pc | 2h | 4h | 增加日志验证 |
| **5.4** | 单元测试修复 | 2h | 4h | 增加调试时间 |
| **5.5** | CFG 单元测试 | 2h | 4h | 完整测试覆盖 |
| **Phase 5 小计** | | 14h | **26h** | **+12h buffer** |
| | | | | |
| **6.1** | 集成测试 | 4h | 6h | 增加测试用例 |
| **6.2** | 性能基准 | 4h | 6h | 增加对比分析 |
| **6.3** | 代码质量 | 4h | 6h | 增加 valgrind |
| **6.4** | 文档完善 | 4h | 6h | 增加 CFG guide |
| **6.5** | 最终验收 | 4h | 6h | 增加完整测试 |
| **Phase 6 小计** | | 20h | **30h** | **+10h buffer** |
| | | | | |
| **总计** | | 34h | **56h** | **7 工作日** |

**缓冲**: +22h (65% buffer for unexpected issues)

---

## 关键风险与缓解 (Revised)

| 风险 | 概率 | 影响 | 缓解措施 | 负责人 |
|------|------|------|---------|--------|
| **CFG 类型不匹配** | 高 | 高 | Phase 5.0 预先审查，Phase 5.1 专门处理 | Dev Lead |
| **CMake 链接错误** | 中 | 高 | 准备手动链接脚本作为 backup | Build Engineer |
| **reconvergence_pc 计算错误** | 中 | 高 | Phase 5.3 详细日志验证 | Dev |
| **测试仍然失败** | 中 | 中 | Phase 5.4 预留 4h 调试时间 | QA |
| **性能回归 > 5%** | 低 | 中 | Phase 6.2 监控，必要时优化 CFG 算法 | Perf Engineer |
| **内存泄漏** | 低 | 高 | Phase 6.3 valgrind 检查 | QA |
| **文档滞后** | 低 | 低 | Phase 6.4 强制更新，验收检查 | Tech Writer |

---

## 降级方案 (Fallback Plan)

如果 Phase 5.1-5.2失败：

### 降级方案 A: 简化 CFG (Recommended)

```cpp
// 使用简化版 CFG 分析
// 只计算直接后继，不计算完整 post-dominator
branch.reconvergence_pc = i + 1;  // Conservative fallback
```

**优点**: 快速，保证编译通过  
**缺点**: reconvergence 不精确

---

### 降级方案 B: 分阶段集成

```
Phase 5.1-5.2: 仅编译 CFG Builder
Phase 5.3-5.4: 跳过 Kernel 集成
Phase 6.x:     下次迭代完成集成
```

**优点**: 降低风险  
**缺点**: reconvergence_pc 功能延后

---

### 降级方案 C: 条件编译开关

```cpp
#ifdef PTX_SIMT_V2_CFG_ENABLED
    // Full CFG integration
#else
    // Fallback: reconvergence_pc = i + 1
#endif
```

**优点**: 可运行时切换  
**缺点**: 代码复杂度增加

---

## 里程碑 (Revised)

| 里程碑 | 原计划 | 修订计划 | 交付物 |
|--------|--------|---------|-------|
| **M0: Phase 5.0 完成** | - | 2026-04-10 18:00 | 代码审查报告 |
| **M1: Phase 5.1 完成** | 2026-04-11 12:00 | 2026-04-11 18:00 | CFG 编译通过 |
| **M2: Phase 5.2 完成** | 2026-04-11 16:00 | 2026-04-12 18:00 | Kernel 集成完成 |
| **M3: Phase 5.3-5.5 完成** | 2026-04-11 20:00 | 2026-04-13 18:00 | Phase 5 100% |
| **M4: Phase 6.1-6.3 完成** | 2026-04-12 12:00 | 2026-04-14 18:00 | 集成测试通过 |
| **M5: Phase 6.4-6.5 完成** | 2026-04-12 20:00 | 2026-04-15 18:00 | 最终验收 |
| **M6: 合并到 main** | 2026-04-13 | 2026-04-16 | SIMT v2.0 完整功能 |

---

## 立即行动项 (Next Steps)

### Phase 5.0.1: 审查 cfg_builder.h (30 分钟)

```bash
# 检查当前代码
cat src/ptx_parser/cfg_builder.h

# 识别问题
# 1. Include paths
# 2. Type definitions
# 3. Forward declarations
```

**输出**: `docs/architecture-upgrade/CODE-REVIEW-CFG.md`

---

### Phase 5.0.2: 创建备份分支 (15 分钟)

```bash
git checkout -b backup/pre-phase5
git push origin backup/pre-phase5
```

---

### Phase 5.0.3: 记录测试基线 (15 分钟)

```bash
cd build && ctest -R "simt|bra|syncthreads" --output-on-failure > test_baseline.txt
```

**基线**: 5/8 tests pass (62.5%)

---

### Phase 5.0.4: 准备开发环境 (30 分钟)

```bash
# 确保编译工具链可用
which g++ cmake valgrind

# 设置调试标志
export CMAKE_BUILD_TYPE=Debug
export PTX_DEBUG=1
```

---

## 总结

### 关键改进

1. ✅ **增加 Phase 5.0** (准备与审查) - 避免盲目实施
2. ✅ **增加时间缓冲** (从 34h 到 56h) - 应对意外问题
3. ✅ **详细依赖关系** - 明确任务先后顺序
4. ✅ **降级方案** - 3 个 fallback 方案
5. ✅ **完整验收标准** - 每个任务都有明确验收标准

### 风险降低

- **原风险**: 编译失败、链接错误、测试失败
- **缓解措施**: Phase 5.0 预先审查，增加缓冲时间，降级方案

### 成功标准

**Phase 5 完成**: 10/10 tests pass  
**Phase 6 完成**: 100% tests pass, 0 performance regression

---

**状态**: Ready for Phase 5.0 execution  
**下一步**: 开始 Phase 5.0.1 (代码审查)

---

## 附录：Parser 阶段 reconvergence_pc 测试方案 ⭐ NEW

**提出者**: User  
**实施日期**: 2026-04-10  
**优势**: 早期测试，快速反馈，不依赖完整运行时

### 测试架构

```
┌────────────────────────────────────────────────────────────┐
│  test_reconvergence.ptx                                    │
│  - Simple branch testcases (if-else, nested)               │
│  - Expected: Specific reconvergence points                 │
└────────────────────────────────────────────────────────────┘
                            ↓ (input)
┌────────────────────────────────────────────────────────────┐
│  test_reconvergence_parser.cpp                             │
│  - Loads PTX file                                          │
│  - Calls PtxParser                                         │
│  - Runs CFG Builder                                        │
│  - Checks BranchInstr.reconvergence_pc                     │
│  - Reports: PASS/FAIL per branch                           │
└────────────────────────────────────────────────────────────┘
                            ↓ (execution)
┌────────────────────────────────────────────────────────────┐
│  test_reconvergence_ptx.sh                                 │
│  - Compiles test program                                   │
│  - Runs test                                               │
│  - Reports overall result                                  │
│  - Provides troubleshooting guidance                       │
└────────────────────────────────────────────────────────────┘
```

---

### 测试文件清单

| 文件 | 路径 | 说明 |
|------|------|------|
| **PTX 测试代码** | `tests/ptx/test_reconvergence.ptx` | 包含简单和嵌套分支 |
| **C++ 测试程序** | `tests/ptx/test_reconvergence_parser.cpp` | Parser 级别测试 |
| **Shell 脚本** | `tests/ptx/test_reconvergence_ptx.sh` | 编译和运行脚本 |
| **CMake 配置** | `tests/CMakeLists.txt` | 添加测试目标 |

---

### 运行方法

#### 方法 1: 使用 Shell 脚本 (推荐)

```bash
cd /workspace/project/PTX-EMU
./tests/ptx/test_reconvergence_ptx.sh
```

**预期输出**:
```
=== SIMT v2.0 Reconvergence PC Parser Test ===
Step 1: Checking test PTX file... OK
Step 2: Compiling test program... OK
Step 3: Running reconvergence_pc test...
============================================================================
=== Checking test_simple_branch ===
PC=3: bra $L_then, reconvergence_pc=10 ✅ PASS
PC=7: bra.uni $L_merge, reconvergence_pc=10 ✅ PASS

--- Summary for test_simple_branch ---
Total branches: 2
With reconvergence_pc: 2
Invalid reconvergence_pc: 0

=== Overall Result ===
✅ ALL TESTS PASSED
```

---

#### 方法 2: 直接运行测试程序

```bash
cd build
cmake --build . --target test_reconvergence_parser
./bin/tests/test_reconvergence_parser ../tests/ptx/test_reconvergence.ptx
```

---

#### 方法 3: 使用 CTest

```bash
cd build
ctest -R reconvergence --output-on-failure -V
```

---

### 测试用例设计

#### 测试用例 1: 简单分支

```ptx
mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 16;
@%p1 bra $L_then;      // PC=3, expected reconvergence=10

// else path (lanes 16-31)
add.u32 %r2, %r1, 100; // PC=4-6
bra.uni $L_merge;      // PC=7, expected reconvergence=10

$L_then:               // PC=10
// then path (lanes 0-15)
add.u32 %r2, %r1, 1;   // PC=11-12

$L_merge:              // PC=13 (convergence point)
st.shared.u32 [%shared0], %r2;
bar.sync 0;
ret;
```

**预期**:
- PC=3 bra → reconvergence_pc=10
- PC=7 bra.uni → reconvergence_pc=13

---

#### 测试用例 2: 嵌套分支

```ptx
mov.u32 %r1, %tid.x;
setp.lt.u32 %p1, %r1, 8;
@%p1 bra $L_inner;     // PC=3, expected reconvergence=18 (outer merge)

// outer else (lanes 8-31)
add.u32 %r2, %r1, 1000;
bra.uni $L_outer_merge;

$L_inner:              // inner branch (lanes 0-7)
setp.lt.u32 %p2, %r1, 4;
@%p2 bra $L_inner_then; // PC=12, expected reconvergence=18

// inner else (lanes 4-7)
add.u32 %r3, %r1, 200;
bra.uni $L_inner_merge;

$L_inner_then:         // inner then (lanes 0-3)
add.u32 %r3, %r1, 10;

$L_inner_merge:        // inner convergence (all 0-7)
add.u32 %r2, %r3, 0;

$L_outer_merge:        // outer convergence (all 0-31)
st.shared.u32 [%shared0], %r2;
bar.sync 0;
ret;
```

**预期**:
- PC=3 bra → reconvergence_pc=18
- PC=12 bra → reconvergence_pc=18

---

### 验收标准

| 测试 | 预期结果 | 实际 | 状态 |
|------|---------|------|------|
| 简单分支 reconvergence | ✅ All PASS | - | ⏳ Pending |
| 嵌套分支 reconvergence | ✅ All PASS | - | ⏳ Pending |
| CFG Builder 编译 | ✅ 0 errors | - | ⏳ Pending |
| Post-Dominator 分析 | ✅ Valid results | - | ⏳ Pending |

---

### 故障排查

#### 问题 1: 编译失败

```
error: undefined reference to 'ptx::cfg::CFGBuilder::build(...)'
```

**原因**: CFG Builder 未编译或未链接

**解决**:
```bash
# Check if cfg_builder.cpp is in CMakeLists.txt
grep cfg_builder src/CMakeLists.txt

# Should show:
#   ptx_parser/cfg_builder.cpp

# If missing, add it to ptx_parser library
```

---

#### 问题 2: reconvergence_pc = -1

```
PC=3: bra $L_then, reconvergence_pc=-1 ❌ FAIL
```

**原因**: CFG 分析未执行或失败

**解决**:
```bash
# Check if CFG analysis is called in ptx_interpreter.cpp
grep -A10 "CFG analysis" src/cudart/ptx_interpreter.cpp

# Should show CFGBuilder::build() call
```

---

#### 问题 3: reconvergence_pc <= branch_pc

```
PC=3: bra $L_then, reconvergence_pc=3 ❌ FAIL (reconvergence_pc <= branch_pc)
```

**原因**: Post-Dominator 计算错误

**解决**:
```bash
# Add debug output to computePostDominators
# Check if basic block edges are correct
# Verify post-dominator algorithm
```

---

### 集成到 Phase 5 计划

#### Phase 5.4 更新

**新增任务**:
- [ ] 5.4.3: 运行 parser 级别 reconvergence 测试

**验收标准**:
```bash
./tests/ptx/test_reconvergence_ptx.sh
# 预期：ALL TESTS PASSED
```

---

### 优势总结

| 方面 | 完整测试 | Parser 测试 |
|------|---------|------------|
| **测试时机** | Kernel 加载后 | Parser 解析后 |
| **反馈速度** | 慢 (~2 秒) | 快 (~0.1 秒) |
| **调试复杂度** | 高 (运行时) | 低 (静态分析) |
| **依赖** | 完整运行时 | 仅 parser |
| **适用阶段** | Phase 5.4+ | **Phase 5.1+** |

**结论**: Parser 级别的 reconvergence 测试应该作为 Phase 5.1-5.2的**即时验证**工具。

---

### 后续改进建议

1. **自动化**: 将 test_reconvergence_ptx.sh 集成到 CI/CD
2. **扩展测试**: 添加更多分支模式测试 (循环、switch-case)
3. **性能基准**: 测量 CFG 分析开销
4. **可视化**: 生成 CFG 图辅助调试
