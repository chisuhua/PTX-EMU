# Post-Dominator 算法

**版本**: 1.0  
**日期**: 2026-04-11  
**来源**: Cytron et al. SSA Construction

---

## 📖 定义

**Post-Dominator**: 节点 B post-dominates 节点 A，当且仅当从 A 到 exit 的**所有**路径都必须经过 B。

**Immediate Post-Dominator**: 最接近的 post-dominator (支配所有其他 post-dominators)。

---

## 🎯 用途

在 SIMT v2.0 中，post-dominator 用于计算：
- Branch reconvergence point
- SIMT Stack reconvergence_pc
- Control flow 汇合点

---

## 🔧 算法实现

### 输入
```cpp
CFG cfg;  // 控制流图
// - blocks: BasicBlock vector
// - entry_block_id: 入口块 ID
// - exit_block_id: 出口块 ID
```

### 输出
```cpp
PostDominatorMap ipd;  // PC → immediate post-dominator PC
// 如果 PC 没有 post-dominator (如 exit)，值为 -1
```

---

### Step 1: 初始化

```cpp
std::map<int, std::set<int>> postDomSets;

for (const auto& block : cfg.blocks) {
    if (block.id == cfg.exit_block_id) {
        // Exit block 的 post-dom 是自身
        postDomSets[block.id] = {block.id};
    } else {
        // 其他块初始为所有块集合
        postDomSets[block.id] = getAllBlockIds();
    }
}
```

---

### Step 2: 迭代计算

```cpp
bool changed = true;
int iterations = 0;

while (changed && iterations < 100) {
    changed = false;
    iterations++;
    
    for (const auto& block : cfg.blocks) {
        if (block.id == cfg.exit_block_id) continue;
        
        // post-dom = {self} ∩ (∩ successors' post-dom sets)
        std::set<int> newSet = {block.id};
        
        for (int succ_id : block.successors) {
            auto it = postDomSets.find(succ_id);
            if (it == postDomSets.end()) continue;
            
            std::set<int> intersection;
            std::set_intersection(
                newSet.begin(), newSet.end(),
                it->second.begin(), it->second.end(),
                std::inserter(intersection, intersection.begin())
            );
            newSet = intersection;
        }
        
        if (newSet != postDomSets[block.id]) {
            postDomSets[block.id] = newSet;
            changed = true;
        }
    }
}
```

**收敛保护**:
- 最大迭代 100 次
- 无变化时提前退出
- 通常 10-20 次收敛

---

### Step 3: 提取 Immediate Post-Dominator

```cpp
PostDominatorMap result;

for (const auto& block : cfg.blocks) {
    int ipd = findImmediatePostDominator(block, postDomSets);
    result[block.start_pc] = ipd;
}
```

**findImmediatePostDominator** 实现:
```cpp
int findImmediatePostDominator(
    const BasicBlock& block,
    const std::map<int, std::set<int>>& postDomSets) {
    
    auto it = postDomSets.find(block.id);
    if (it == postDomSets.end()) return -1;
    
    const std::set<int>& postDoms = it->second;
    
    // 找到 immediate post-dominator
    for (int candidate : postDoms) {
        if (candidate == block.id) continue;
        
        bool isImmediate = true;
        for (int other : postDoms) {
            if (other == block.id || other == candidate) continue;
            
            auto otherIt = postDomSets.find(other);
            if (otherIt == postDomSets.end()) continue;
            
            // 如果 other post-dominates candidate, candidate 不是 immediate
            if (otherIt->second.count(candidate)) {
                isImmediate = false;
                break;
            }
        }
        
        if (isImmediate) {
            return candidate;
        }
    }
    
    return -1;
}
```

---

## 📊 示例

### 简单 if-else

```
     [PC=0]
        │
     [PC=1] bra
       /   \
      /     \
 [PC=2]   [PC=5]
   │        │
 [PC=3]   [PC=6]
   │        │
   └───┬────┘
       │
    [PC=7] ← Immediate Post-Dominator
```

**结果**:
| PC | Post-Dominator |
|----|----------------|
| 1 (bra) | 7 |
| 2-3 | 7 |
| 5-6 | 7 |

---

## ⏱️ 时间复杂度

| 阶段 | 复杂度 | 说明 |
|------|--------|------|
| 初始化 | O(n) | n = block 数量 |
| 迭代计算 | O(n × iterations × succ_count) | iterations < 100 |
| 提取 IPD | O(n²) | 两两比较 |
| **总计** | **O(n²)** | 实际很快 |

---

## 🧪 测试验证

### 测试用例 1: 简单分支

```cpp
TEST_CASE("Simple if-else post-dominator") {
    // Setup: if-else CFG
    PostDominatorMap ipd = computePostDominators(cfg);
    
    // Verify: both paths converge at merge point
    REQUIRE(ipd[branch_pc] == merge_pc);
}
```

### 测试用例 2: 嵌套分支

```cpp
TEST_CASE("Nested branches post-dominator") {
    // Setup: nested if-else CFG
    PostDominatorMap ipd = computePostDominators(cfg);
    
    // Verify: each level has correct reconvergence
    REQUIRE(ipd[outer_bra] == outer_merge);
    REQUIRE(ipd[inner_bra] == inner_merge);
}
```

---

## 💡 优化技巧

### 1. 提前退出

```cpp
if (!changed) break;  // 无变化，提前退出
```

### 2. 迭代限制

```cpp
if (iterations >= 100) {
    PTX_WARNING("Post-dominator did not converge in 100 iterations");
    break;
}
```

### 3. 集合操作优化

使用 `std::set_intersection` 代替手动循环。

---

## 📚 参考资料

1. Cytron, R., et al. "Efficiently computing SSA form..." Communications of the ACM (1991).
2. Cooper, K. D., et al. "A simple, fast dominance algorithm." (2001).
3. PTX ISA 9.1 Documentation - Control Flow.

---

**维护**: 持续更新  
**最后更新**: 2026-04-11  
**版本**: 1.0
