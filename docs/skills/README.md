# 技能沉淀目录

本目录包含 SIMT v2.0 项目中的技术技能和模式总结。

---

## 📁 技能列表

| 技能 | 难度 | 适用场景 | 状态 |
|------|------|---------|------|
| [cfg-builder-pattern](./cfg-builder-pattern.md) | ⭐⭐⭐ | CFG 构建 | ✅ |
| [post-dominator-algorithm](./post-dominator-algorithm.md) | ⭐⭐⭐⭐ | 控制流分析 | ✅ |
| [simt-reconvergence](./simt-reconvergence.md) | ⭐⭐⭐⭐ | SIMT 收敛 | ✅ |
| [tdd-workflow](./tdd-workflow.md) | ⭐⭐ | 开发流程 | ✅ |

---

## 🎯 技能分类

### 算法类

- **CFG Builder Pattern** - 控制流图构建模式
- **Post-Dominator Algorithm** - 后支配树算法

### 架构类

- **SIMT Reconvergence** - SIMT 收敛技术

### 流程类

- **TDD Workflow** - 测试驱动开发流程

---

## 📖 学习路径

### 新开发者

```
1. TDD Workflow (⭐⭐) → 了解开发流程
2. CFG Builder Pattern (⭐⭐⭐) → 学习 CFG 构建
3. SIMT Reconvergence (⭐⭐⭐⭐) → 理解 SIMT 收敛
4. Post-Dominator Algorithm (⭐⭐⭐⭐) → 深入算法
```

### 架构师

```
1. Post-Dominator Algorithm (⭐⭐⭐⭐) → 核心算法
2. CFG Builder Pattern (⭐⭐⭐) → 构建模式
3. SIMT Reconvergence (⭐⭐⭐⭐) → 架构设计
```

---

## 🔧 实用代码

### CFG Builder 快速使用

```cpp
#include "ptx_parser/cfg_builder.h"

// Build CFG
CFG cfg = CFGBuilder::build(statements, label2pc);

// Compute post-dominators
PostDominatorMap postDoms = CFGBuilder::computePostDominators(cfg);

// Use reconvergence_pc
for (size_t i = 0; i < statements.size(); i++) {
    if (statements[i].type == S_BRA) {
        auto& branch = std::get<BranchInstr>(statements[i].data);
        branch.reconvergence_pc = postDoms[i];
    }
}
```

---

## 📊 技能掌握评估

| 技能 | 理解 | 应用 | 精通 |
|------|------|------|------|
| CFG Builder | ✅ | ✅ | ⏳ |
| Post-Dominator | ✅ | ✅ | ⏳ |
| SIMT Reconvergence | ✅ | ✅ | ✅ |
| TDD Workflow | ✅ | ✅ | ✅ |

---

## 📚 参考资料

- Cytron et al. "Efficiently computing SSA form"
- PTX ISA 9.1 Documentation
- GPGPU-Sim Implementation
- NVIDIA SIMT Architecture Papers

---

**维护**: 持续更新  
**贡献**: 提交 PR 到 `docs/skills/`  
**最后更新**: 2026-04-11
