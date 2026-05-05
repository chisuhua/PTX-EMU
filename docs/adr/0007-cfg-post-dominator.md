# ADR-0007: CFG Post-Dominator 收敛分析

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 1 (CFG Builder) |
| **作者** | PTX-EMU Team |

## 上下文

在 SIMT 执行模型中，当 warp 内的线程出现分歧时，需要确定一个 reconvergence point（汇合点），所有分歧路径的线程最终都会到达这个点。

早期方案使用硬编码的 reconvergence 规则（如 branch 后的第一条公共指令），但这无法处理复杂的控制流模式，特别是：

- 多重分支（if-else if-else）
- 嵌套分支（分支内还有分支）
- 循环内的分支

## 决策驱动因素

1. **PTX ISA 规范要求**：PTX ISA 9.1 明确指出 reconvergence point 是所有路径的 post-dominator
2. **复杂控制流支持**：硬编码规则无法覆盖所有分支模式
3. **编译时计算，零运行时开销**：CFG 分析在 kernel 加载时执行一次

## 考虑的替代方案

### 方案 A: 硬编码 Reconvergence 规则

**描述**: 根据指令类型硬编码 reconvergence point（如 bra 后第二条指令）

**优点**:
- 实现简单
- 无额外分析开销

**缺点**:
- 无法处理复杂控制流
- 无法处理嵌套分支
- 与 PTX ISA 语义不符

### 方案 B: 运行时动态检测

**描述**: 执行时动态检测所有线程何时汇合

**优点**:
- 无需 CFG 分析

**缺点**:
- 运行时开销大
- 可能永远无法汇合（如线程提前退出）
- 无法预知 reconvergence point

### 方案 C: CFG Post-Dominator 分析 (✅ 选中)

**描述**: 在 kernel 加载时构建控制流图，计算 post-dominator tree，为每个分支确定 reconvergence point

**优点**:
- 与 PTX ISA 语义完全一致
- 编译时计算，零运行时开销
- 支持任意复杂控制流
- 可验证分支收敛性

**缺点**:
- CFG 构建算法复杂（迭代定点计算）
- 需要识别基本块和分支目标

**选择理由**: Post-dominator 理论是编译器领域成熟的技术，能精确确定 reconvergence point，且只需在 kernel 加载时计算一次。

## 决策内容

### 设计原则

1. **基本块识别**：以分支指令和分支目标为边界划分基本块
2. **迭代定点计算**：post-dominator 集合迭代至不动点
3. **立即后支配者**：取最近的 post-dominator 作为 reconvergence point

### 实现要点

```cpp
// 基本块结构
struct BasicBlock {
    int id;
    int start_pc;         // 起始 PC（包含）
    int end_pc;           // 结束 PC（不包含）
    std::vector<int> successors;
    std::vector<int> predecessors;
    bool is_branch_target;
    bool is_exit;
};

// CFG 构建流程
CFG CFGBuilder::build(
    const std::vector<StatementContext>& statements,
    const std::map<std::string, int>& label2pc) 
{
    // 1. 识别基本块
    auto blocks = identifyBasicBlocks(statements, label2pc);
    
    // 2. 构建边
    CFG cfg;
    cfg.blocks = blocks;
    buildEdges(cfg, label2pc, statements);
    
    return cfg;
}

// Post-Dominator 计算
PostDominatorMap CFGBuilder::computePostDominators(const CFG& cfg) {
    std::map<int, std::set<int>> postDomSets;
    
    // 初始化：exit block 的 post-dom 集合只包含自己
    // 其他 block 的 post-dom 集合包含所有 block
    for (const auto& block : cfg.blocks) {
        if (block.id == cfg.exit_block_id) {
            postDomSets[block.id] = {block.id};
        } else {
            postDomSets[block.id] = getAllBlockIds();
        }
    }
    
    // 迭代至不动点
    bool changed = true;
    while (changed) {
        changed = false;
        for (const auto& block : cfg.blocks) {
            if (block.id == cfg.exit_block_id) continue;
            
            // Post-dom set = ∩(successors' post-dom sets) ∪ {self}
            std::set<int> newSet = {block.id};
            for (int succ_id : block.successors) {
                set_intersection(newSet, postDomSets[succ_id]);
            }
            
            if (newSet != postDomSets[block.id]) {
                postDomSets[block.id] = newSet;
                changed = true;
            }
        }
    }
    
    // 提取立即后支配者
    PostDominatorMap result;
    for (const auto& block : cfg.blocks) {
        result[block.start_pc] = findImmediatePostDominator(block, postDomSets);
    }
    return result;
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `src/ptx_parser/cfg_builder.h` | 新增 | CFG 和 BasicBlock 结构定义 |
| `src/ptx_parser/cfg_builder.cpp` | 新增 | CFG 构建和 post-dominator 计算 |
| `src/ptx_parser/ptx_visitor.cpp` | 修改 | kernel 加载时调用 CFGBuilder |
| `src/ptxsim/instructions/control.cpp` | 修改 | bra 指令使用 reconvergence_pc |

## 后果

### 正面影响

- 精确确定 reconvergence point
- 支持任意复杂控制流
- 零运行时开销

### 负面影响

- CFG 构建算法复杂，需要充分测试
- 极端情况下迭代可能收敛慢（但实际 kernel 通常很快）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| CFG 构建错误 | 中 | 高 | 单元测试覆盖所有分支模式 |
| Post-dominator 计算死循环 | 低 | 高 | 设置最大迭代次数（通常 < 10 次收敛） |
| 无限循环无法确定 exit | 低 | 高 | 添加循环检测，loop header 的 post-dominator 特殊处理 |

## 合规检查

后续相关开发应检查：

- [ ] 分支指令的 reconvergence_pc 来自 CFG post-dominator
- [ ] 不硬编码 reconvergence 规则
- [ ] CFG 分析在 kernel 加载时执行一次
- [ ] 单元测试覆盖嵌套分支、循环内分支等复杂场景

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |

## 参考

- [SIMT 架构文档](../architecture/SIMT-ARCHITECTURE-V2.md#33-cfg-分析模块)
- [Cytron et al. - SSA Form with Optimal Dominator Frontiers](https://doi.org/10.1145/115372.115320)
- [PTX ISA 9.1 - Control Flow Instructions](../archive/ptx-instruction-reference/9.7.12_control_flow.md)
