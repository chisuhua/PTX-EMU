# CFG 集成指南

**版本**: v2.0  
**最后更新**: 2026-04-11  
**适合人群**: 后端开发者，架构师

---

## 📖 概述

本指南介绍如何在 PTX-EMU 项目中集成 CFG (Control Flow Graph) 分析。

---

## 🏗️ 架构设计

### CFG Builder 组件

```
src/ptx_parser/
├── cfg_builder.h          # 接口定义
└── cfg_builder.cpp        # 实现
```

### 主要接口

```cpp
namespace ptx {
namespace cfg {

// Build CFG from PTX statements
CFG CFGBuilder::build(
    const std::vector<StatementContext>& statements,
    const std::map<std::string, int>& label2pc
);

// Compute post-dominators
PostDominatorMap CFGBuilder::computePostDominators(const CFG& cfg);

}
}
```

---

## 🔧 集成步骤

### Step 1: 包含头文件

```cpp
#include "ptx_parser/cfg_builder.h"
```

### Step 2: 准备数据

```cpp
// Kernel statements (from parser)
std::vector<StatementContext> statements = ...;

// Label to PC mapping
std::map<std::string, int> label2pc;
for (int i = 0; i < statements.size(); i++) {
    if (statements[i].type == S_DOLLOR) {
        const auto& dollar = std::get<DollarNameInstr>(statements[i].data);
        label2pc[dollar.name] = i;
    }
}
```

### Step 3: 构建 CFG

```cpp
CFG cfg = CFGBuilder::build(statements, label2pc);
```

### Step 4: 计算 Post-Dominators

```cpp
PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
```

### Step 5: 更新 reconvergence_pc

```cpp
for (size_t i = 0; i < statements.size(); i++) {
    if (statements[i].type == S_BRA) {
        auto& branch = std::get<BranchInstr>(statements[i].data);
        
        auto it = postDoms.find(i);
        if (it != postDoms.end() && it->second >= 0) {
            branch.reconvergence_pc = it->second;
        } else {
            branch.reconvergence_pc = i + 1;  // Fallback
        }
    }
}
```

---

## 📋 实际集成 (ptx_interpreter.cpp)

```cpp
void PtxInterpreter::setupLabels(std::map<std::string, int>& label2pc) {
    // 1. Register labels
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        const auto& e = kernelContext->kernelStatements[i];
        if (e.type == S_DOLLOR) {
            const auto& s = std::get<DollarNameInstr>(e.data);
            label2pc[s.name] = i;
        }
    }
    
    // 2. CFG analysis
    CFG cfg = CFGBuilder::build(kernelContext->kernelStatements, label2pc);
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    
    // 3. Update reconvergence_pc
    int updated_count = 0;
    for (int i = 0; i < kernelContext->kernelStatements.size(); i++) {
        const auto& stmt = kernelContext->kernelStatements[i];
        if (stmt.type == S_BRA) {
            auto& branch = std::get<BranchInstr>(
                kernelContext->kernelStatements[i].data);
            
            auto it = postDoms.find(i);
            if (it != postDoms.end() && it->second >= 0) {
                branch.reconvergence_pc = it->second;
                updated_count++;
            } else {
                branch.reconvergence_pc = i + 1;
            }
        }
    }
    
    PTX_INFO("CFG analysis complete: updated %d branches", updated_count);
}
```

---

## 🧪 测试验证

### 单元测试

```cpp
TEST_CASE("CFG integration", "[cfg][integration]") {
    std::vector<StatementContext> statements(10);
    std::map<std::string, int> label2pc;
    
    // Setup branch
    statements[2].type = S_BRA;
    BranchInstr branch;
    branch.target = "L_merge";
    branch.reconvergence_pc = -1;
    statements[2].data = branch;
    
    label2pc["L_merge"] = 8;
    
    // Build CFG
    CFG cfg = CFGBuilder::build(statements, label2pc);
    PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);
    
    // Verify
    REQUIRE(postDoms.count(2) > 0);
    REQUIRE(postDoms[2] == 8);  // reconvergence at L_merge
}
```

### 性能验证

```cpp
auto start = std::chrono::high_resolution_clock::now();
CFG cfg = CFGBuilder::build(statements, label2pc);
auto end = std::chrono::high_resolution_clock::now();

auto duration = std::chrono::duration_cast<std::chrono::microseconds>(
    end - start).count();

REQUIRE(duration < 100);  // <100 μs for medium kernel
```

---

## ⚠️ 常见问题

### Q1: Post-Dominator 计算慢

**原因**: 迭代次数过多

**解决**:
```cpp
// Add iteration limit
int iterations = 0;
while (changed && iterations < 100) {
    // ...
    iterations++;
}
```

### Q2: reconvergence_pc 不正确

**检查**:
1. label2pc 是否正确
2. CFG edges 是否完整
3. Post-Dominator 算法是否正确

### Q3: 内存使用高

**优化**:
```cpp
// Pre-allocate
std::vector<BasicBlock> blocks;
blocks.reserve(statements.size() / 10);
```

---

## 📚 参考文档

- [`cfg-builder-pattern.md`](../skills/)
- [`post-dominator-algorithm.md`](../skills/)
- [`PHASE5-FINAL-REPORT.md`](../reports/phase-reports/)

---

**最后更新**: 2026-04-11  
**集成状态**: ✅ Complete (Phase 5)
