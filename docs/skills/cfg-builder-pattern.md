# CFG Builder 模式

**版本**: 1.0  
**日期**: 2026-04-11  
**适用**: PTX kernel 控制流分析

---

## 📖 背景

在 PTX 模拟器中，需要分析 kernel 的控制流图 (CFG) 来计算分支收敛点 (reconvergence point)。这是 SIMT 执行模型的核心需求。

---

## 🏗️ 模式说明

### 1. BasicBlock 识别

**目标**: 将连续的 PTX 语句分割为基本块

```cpp
std::set<int> boundaries;
boundaries.insert(0);  // Entry
boundaries.insert(statements.size());  // Exit

// 添加分支目标
for (const auto& stmt : statements) {
    if (stmt.type == S_BRA) {
        boundaries.insert(label2pc[branch.target]);
    }
}

// 创建 BasicBlocks
for (int boundary : boundaries) {
    BasicBlock block;
    block.start_pc = prev_boundary;
    block.end_pc = boundary;
    block.is_branch_target = (targets.count(prev_boundary) > 0);
    blocks.push_back(block);
}
```

**关键点**:
- Entry (PC=0) 总是块开始
- Exit (PC=size) 总是块结束
- 分支目标总是新块开始
- 分支指令后总是块结束

---

### 2. CFG 构建

**目标**: 添加块之间的边 (successors/predecessors)

```cpp
void CFGBuilder::buildEdges(CFG& cfg, ...) {
    for (auto& block : cfg.blocks) {
        int last_pc = block.end_pc - 1;
        const auto& stmt = statements[last_pc];
        
        if (stmt.type == S_BRA) {
            // 1. Add fall-through edge (if exists)
            for (auto& other : cfg.blocks) {
                if (other.start_pc == block.end_pc) {
                    block.successors.push_back(other.id);
                }
            }
            
            // 2. Add branch target edge (CRITICAL FIX)
            int target_pc = label2pc[branch.target];
            for (auto& other : cfg.blocks) {
                if (other.start_pc == target_pc) {
                    block.successors.push_back(other.id);
                }
            }
        } else {
            // Sequential flow
            for (auto& other : cfg.blocks) {
                if (other.start_pc == block.end_pc) {
                    block.successors.push_back(other.id);
                }
            }
        }
    }
}
```

**关键点**:
- 分支有两路：fall-through + branch target
- 非分支只有一路：sequential flow
- **Phase 5 关键修复**: 添加 branch target 边

---

### 3. Post-Dominator 计算

**目标**: 计算每个块的立即后支配点 (immediate post-dominator)

```cpp
PostDominatorMap CFGBuilder::computePostDominators(const CFG& cfg) {
    std::map<int, std::set<int>> postDomSets;
    
    // 初始化
    for (const auto& block : cfg.blocks) {
        if (block.id == cfg.exit_block_id) {
            postDomSets[block.id] = {block.id};
        } else {
            postDomSets[block.id] = all_block_ids;
        }
    }
    
    // 迭代数据流算法
    bool changed = true;
    int iterations = 0;
    while (changed && iterations < 100) {
        changed = false;
        iterations++;
        
        for (const auto& block : cfg.blocks) {
            if (block.id == cfg.exit_block_id) continue;
            
            std::set<int> newSet = {block.id};
            for (int succ_id : block.successors) {
                // Intersection with successor's post-dom set
                newSet = intersection(newSet, postDomSets[succ_id]);
            }
            
            if (newSet != postDomSets[block.id]) {
                postDomSets[block.id] = newSet;
                changed = true;
            }
        }
    }
    
    // 提取 immediate post-dominator
    PostDominatorMap result;
    for (const auto& block : cfg.blocks) {
        result[block.start_pc] = findImmediatePostDominator(block, postDomSets);
    }
    
    return result;
}
```

**关键点**:
- 迭代算法，通常 <100 次收敛
- Exit block 的 post-dom 是自身
- 其他块初始为所有块集合
- 迭代取后继的交集

---

## 🔑 关键要点

### 1. 边界识别 (3 个原则)

1. Entry (PC=0) 总是块开始
2. Exit (PC=size) 总是块结束
3. 分支目标和分支后总是新块

### 2. 边构建 (2 种边)

1. Fall-through 边：分支的下一条指令
2. Branch target 边：分支跳转的目标

### 3. 迭代收敛 (<100 次)

- 保护迭代次数，防止无限循环
- 通常 10-20 次收敛

### 4. 错误处理

```cpp
try {
    CFG cfg = CFGBuilder::build(statements, label2pc);
    postDoms = CFGBuilder::computePostDominators(cfg);
} catch (const std::exception& e) {
    // Fallback: reconvergence_pc = branch_pc + 1
    branch.reconvergence_pc = i + 1;
}
```

---

## 📊 时间复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| BasicBlock 识别 | O(n) | n = statement 数量 |
| CFG 构建 | O(n²) | 块数量通常很小 |
| Post-Dominator | O(n × iterations) | iterations < 100 |
| **总计** | **O(n²)** | 实际很快 |

---

## 💡 适用场景

- ✅ PTX kernel 控制流分析
- ✅ Branch reconvergence 点计算
- ✅ SIMT 收敛管理
- ✅ Compiler optimization

---

## 🧪 测试覆盖

| 测试 | 文件 | 状态 |
|------|------|------|
| Basic block identification | test_cfg_edge_cases.cpp | ✅ |
| CFG build | test_cfg_builder.cpp | ✅ |
| Post-dominator | test_cfg_edge_cases.cpp | ✅ |
| Complex branches | test_nested_3levels.ptx | ✅ |

---

## 📚 参考资料

- Cytron et al. "Simple and Efficient Construction of Static Single Assignment Forms with Optimal Dominator Frontiers"
- PTX ISA 9.1 Documentation
- GPGPU-Sim CFG Implementation

---

**维护**: 持续更新  
**最后更新**: 2026-04-11  
**版本**: 1.0
