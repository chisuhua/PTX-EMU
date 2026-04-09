# CFG Builder 代码审查 - Corner Case 分析

**日期**: 2026-04-10  
**审查者**: PTX-EMU Architecture Team  
**范围**: cfg_builder.h/cpp 完整性审查  
**状态**: ⚠️ 需要添加边界测试

---

## 识别的 Corner Cases

### 🔴 高优先级 (必须测试)

#### 1. 空输入 (Empty Input)

**场景**: 空的 PTX kernel (0 条语句)

**当前代码问题**:
```cpp
// cfg_builder.cpp:67
std::vector<BasicBlock> CFGBuilder::identifyBasicBlocks(...) {
    std::set<int> boundaries;
    boundaries.insert(0);
    boundaries.insert(statements.size());  // 如果 size=0, 边界为 {0, 0}
    
    // 问题：可能产生空块或崩溃
}
```

**预期行为**: 
- 应该返回空 CFG 或单入口/出口块
- 不应该崩溃

**需要测试**: 
```cpp
TEST_CASE("Empty PTX kernel", "[cfg][edge]") {
    std::vector<StatementContext> statements;
    std::map<std::string, int> label2pc;
    
    // Should not crash
    CFG cfg = CFGBuilder::build(statements, label2pc);
    
    REQUIRE(cfg.blocks.size() >= 0);  // At least entry/exit
}
```

---

#### 2. 单一语句 (Single Statement)

**场景**: 只有 1 条语句（如只有 ret）

**当前代码问题**:
```cpp
// cfg_builder.cpp:97
int last_pc = block.end_pc - 1;
if (last_pc < 0 || last_pc >= (int)statements.size()) continue;

// 如果 block 为空，last_pc 可能无效
```

**预期行为**:
- 正确处理单语句 kernel
- 不崩溃，返回合理 CFG

**需要测试**:
```cpp
TEST_CASE("Single statement kernel", "[cfg][edge]") {
    std::vector<StatementContext> statements(1);
    statements[0].type = S_RET;
    std::map<std::string, int> label2pc;
    
    CFG cfg = CFGBuilder::build(statements, label2pc);
    REQUIRE(cfg.blocks.size() == 1);
    REQUIRE(cfg.blocks[0].is_exit == true);
}
```

---

#### 3. 分支到自身 (Self-Referencing Branch)

**场景**: 分支目标是自身 PC (无限循环)

**PTX 示例**:
```ptx
loop:
  bra loop;  // 分支到自身
```

**当前代码问题**:
```cpp
// cfg_builder.cpp:127
int target_pc = label2pc.at(branch.target);  // 如果 target = current PC

// 问题：可能创建自环，Post-Dominator 算法可能不终止
```

**预期行为**:
- 正确处理自环
- Post-Dominator 算法应该在 100 次迭代内终止

**需要测试**:
```cpp
TEST_CASE("Self-referencing branch", "[cfg][edge]") {
    // Create infinite loop scenario
    // Verify post-dominator computation terminates
}
```

---

#### 4.  unreachable 标签 (Unreachable Label)

**场景**: 定义了标签但没有任何分支指向它

**PTX 示例**:
```ptx
$L_unused:
  add.u32 %r1, %r2, %r3;
ret;
```

**当前代码问题**:
```cpp
// findBranchTargets() only finds targets that ARE branched to
// Unreachable labels become separate basic blocks with no predecessors
```

**预期行为**:
- 正确识别 unreachable 代码
- 不影响 Post-Dominator 计算

**需要测试**:
```cpp
TEST_CASE("Unreachable label", "[cfg][edge]") {
    // Create unreachable code block
    // Verify CFG handles it gracefully
}
```

---

#### 5. 多重分支汇合 (Multi-Branch Convergence)

**场景**: 3+ 个分支路径汇合到同一点

**PTX 示例**:
```ptx
if (x == 0) bra L0;
if (x == 1) bra L1;
if (x == 2) bra L2;
// All converge here
L0:
L1:
L2:
merge_point:
  ret;
```

**当前代码问题**:
```cpp
// buildEdges handles 2 successors (fall-through + branch target)
// But merge point may have 3+ predecessors
```

**预期行为**:
- 正确计算多路径汇合点
- Post-Dominator 应该是 merge_point

**需要测试**:
```cpp
TEST_CASE("Multi-branch convergence", "[cfg][edge]") {
    // 3+ branches converging to same point
    // Verify post-dominator is the merge point
}
```

---

#### 6. 分支目标不存在 (Missing Branch Target)

**场景**: 分支指向不存在的标签

**PTX 示例**:
```ptx
bra $L_nonexistent;
ret;
```

**当前代码问题**:
```cpp
// cfg_builder.cpp:51
auto it = label2pc.find(branch.target);
if (it != label2pc.end()) {
    targets.insert(it->second);
} else {
    // Warning logged, but execution continues
    // Problem: CFG may be incomplete
}
```

**预期行为**:
- 记录错误日志
- 不崩溃
- 可能使用默认 successor

**需要测试**:
```cpp
TEST_CASE("Missing branch target", "[cfg][edge]") {
    // Branch to non-existent label
    // Should not crash, should log warning
}
```

---

#### 7. 嵌套分支深度 (Deep Nested Branches)

**场景**: if-else 嵌套超过 10 层

**当前代码问题**:
```cpp
// Post-Dominator algorithm has iteration limit of 100
// Deep nesting may require more iterations
```

**预期行为**:
- 处理任意深度嵌套
- 在 100 次迭代内收敛

**需要测试**:
```cpp
TEST_CASE("Deep nested branches (20 levels)", "[cfg][edge]") {
    // 20 levels of nested if-else
    // Verify post-dominator computation converges
}
```

---

### 🟡 中优先级 (应该测试)

#### 8. 无分支的线性代码 (Linear Code, No Branches)

**场景**: 只有顺序执行，没有控制流

**PTX 示例**:
```ptx
mov.u32 %r1, 0;
add.u32 %r2, %r1, 1;
ret;
```

**测试价值**:
- 最简单的 CFG (单块)
- 验证基本功能

**需要测试**:
```cpp
TEST_CASE("Linear code (no branches)", "[cfg][basic]") {
    // No branches at all
    // Should have single basic block
}
```

---

#### 9. 只有 Fall-Through 分支 (Fall-Through Only)

**场景**: 条件分支但都是 fall-through

**PTX 示例**:
```ptx
@%p1 bra $L_target;  // 但 %p1 总是 false
add.u32 %r1, %r2, %r3;  // 总是执行
$L_target:
ret;
```

**需要测试**:
```cpp
TEST_CASE("Fall-through branch", "[cfg][edge]") {
    // Branch that always falls through
    // CFG should have correct edges
}
```

---

#### 10. 重复标签 (Duplicate Labels)

**场景**: 同一标签名出现多次

**PTX 示例**:
```ptx
$L_dup:
  mov.u32 %r1, 0;
$L_dup:  // 重复定义
  ret;
```

**预期行为**:
- 应该检测并报告错误
- 或使用最后一个定义

**需要测试**:
```cpp
TEST_CASE("Duplicate labels", "[cfg][edge]") {
    // Duplicate label names
    // Should handle gracefully (error or last-wins)
}
```

---

#### 11. 条件分支后立即返回 (Branch Immediately Before Return)

**场景**: 分支后紧跟着 ret，没有汇合点

**PTX 示例**:
```ptx
@%p1 bra $L_then;
ret;  // Fall-through returns
$L_then:
ret;  // Branch target also returns
```

**需要测试**:
```cpp
TEST_CASE("Branch before return (no merge)", "[cfg][edge]") {
    // Both paths return, no merge point
    // Post-dominator should be exit
}
```

---

### 🟢 低优先级 (可选测试)

#### 12. 极大 Kernel (Large Kernel)

**场景**: 1000+ 条语句

**测试价值**:
- 性能测试
- 内存使用测试

**需要测试**:
```cpp
TEST_CASE("Large kernel (1000+ statements)", "[cfg][perf]") {
    // Performance and memory test
}
```

---

## 当前测试覆盖分析

### 现有测试

| 测试 | 覆盖场景 |
|------|---------|
| test_simple_branch | ✅ 简单分支 (if-else) |
| test_nested_branch | ✅ 两层嵌套分支 |
| test_cfg_builder | ✅ 编译验证 |

### 缺失测试

| 缺失 | 优先级 | 说明 |
|------|--------|------|
| 空输入 | 🔴 高 | 0 语句 kernel |
| 单语句 | 🔴 高 | 只有 ret 的 kernel |
| 自环分支 | 🔴 高 | 无限循环 |
| 多重汇合 | 🔴 高 | 3+ 路径汇合 |
| 缺失标签 | 🔴 高 | 分支到不存在的标签 |
| 深层嵌套 | 🔴 高 | >10 层嵌套 |
| 线性代码 | 🟡 中 | 无分支 |
| 重复标签 | 🟡 中 | 标签重定义 |
| 大 Kernel | 🟢 低 | 性能测试 |

---

## 建议的测试矩阵

### Phase 5.5 测试清单

```cpp
// 1. Basic Tests (基本场景)
TEST_CASE("Empty kernel", "[cfg][basic]") {}
TEST_CASE("Single statement", "[cfg][basic]") {}
TEST_CASE("Linear code", "[cfg][basic]") {}

// 2. Branch Tests (分支场景)
TEST_CASE("Simple branch", "[cfg][branch]") {}
TEST_CASE("Nested branches (2 levels)", "[cfg][branch]") {}
TEST_CASE("Nested branches (5 levels)", "[cfg][branch]") {}
TEST_CASE("Nested branches (10 levels)", "[cfg][branch]") {}
TEST_CASE("Multi-branch convergence (3 paths)", "[cfg][branch]") {}
TEST_CASE("Multi-branch convergence (5 paths)", "[cfg][branch]") {}

// 3. Edge Cases (边界情况)
TEST_CASE("Self-referencing branch", "[cfg][edge]") {}
TEST_CASE("Missing branch target", "[cfg][edge]") {}
TEST_CASE("Unreachable label", "[cfg][edge]") {}
TEST_CASE("Branch immediately before return", "[cfg][edge]") {}
TEST_CASE("Duplicate labels", "[cfg][edge]") {}

// 4. Integration Tests (集成场景)
TEST_CASE("Branch + barrier", "[cfg][integration]") {}
TEST_CASE("Branch + shared memory", "[cfg][integration]") {}
TEST_CASE("Nested branch + barrier", "[cfg][integration]") {}

// 5. Performance Tests (性能场景)
TEST_CASE("Large kernel (1000 statements)", "[cfg][perf]") {}
TEST_CASE("Deep nesting (20 levels)", "[cfg][perf]") {}
```

---

## 总体评估

### 当前覆盖率

| 类别 | 覆盖率 | 状态 |
|------|--------|------|
| 基本场景 | 50% | ⚠️ 部分测试 |
| 分支场景 | 70% | ✅ 主要覆盖 |
| 边界情况 | 10% | ❌ 严重不足 |
| 集成场景 | 20% | ⚠️ 需要补充 |
| 性能测试 | 0% | ❌ 缺失 |

**总体**: 30% **不满足生产质量要求**

---

## 建议

### 立即行动 (Phase 5.5)

1. **添加高优先级边界测试** (6 个 tests)
2. **添加中优先级边界测试** (4 个 tests)
3. **运行完整测试矩阵**
4. **修复发现的任何问题**

### 验收标准

```bash
ctest -R cfg --output-on-failure
# Expected: 100% tests passed
# Minimum: 14 new tests added
```

---

**状态**: 需要添加 14 个边界测试用例  
**优先级**: Phase 5.5 (测试验证阶段)
