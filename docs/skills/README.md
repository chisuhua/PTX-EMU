# 技能沉淀目录

本目录包含技术参考文档。**可加载的技能**已迁移至 `.opencode/skills/`。

## 可加载技能（`.opencode/skills/`）

| 技能 | 类别 |
|------|------|
| `ptx-debug` | 调试 |
| `regression-bisect` | 调试 |
| `state-modification-audit` | 调试 |
| `oracle-prompting` | 调试 |
| `ptx-instruction-pipeline` | PTX 仿真 |
| `ptx-barrier-mechanism` | PTX 仿真 |
| `ptx-grammar-modification` | 语法 |
| `ptxir-serialization` | 解析 |
| `three-mode-testing` | 测试 |

## 技术参考（本目录）

| 文档 | 用途 |
|------|------|
| [post-dominator-algorithm.md](./post-dominator-algorithm.md) | Post-Dominator 算法 |
| [simt-reconvergence.md](./simt-reconvergence.md) | SIMT 收敛技术 |
| [ptx-grammar-modification.md](./ptx-grammar-modification.md) | PTX 语法修改详细指南（→ 已转 skill） |

> 历史说明：原 `cfg-builder-pattern.md` 和 `tdd-workflow.md` 已删除（其内容已迁移至 `.opencode/skills/` 对应技能）

## 📖 学习路径

### 新开发者

```
1. Three-Mode Testing (⭐⭐⭐) → 学习 PTX 测试生成
2. SIMT Reconvergence (⭐⭐⭐⭐) → 理解 SIMT 收敛
3. Post-Dominator Algorithm (⭐⭐⭐⭐) → 深入算法
```

### 架构师

```
1. Post-Dominator Algorithm (⭐⭐⭐⭐) → 核心算法
2. SIMT Reconvergence (⭐⭐⭐⭐) → 架构设计
3. Three-Mode Testing (⭐⭐⭐) → 测试框架
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
| Post-Dominator | ✅ | ✅ | ⏳ |
| SIMT Reconvergence | ✅ | ✅ | ✅ |

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
