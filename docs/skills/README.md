# 技能沉淀目录

> **可加载技能**已迁移至 [`.opencode/skills/`](../../.opencode/skills/)。本目录保留：
> 1. 人类可读导航（本 README）
> 2. 非技能类技术参考文档（[post-dominator-algorithm.md](./post-dominator-algorithm.md)、[simt-reconvergence.md](./simt-reconvergence.md)）

---

## 可加载技能（17 active + 1 disabled）

> 权威来源：[`.opencode/skills/`](../../.opencode/skills/) — 本目录仅作导航副本，**禁止手写技能内容**。

### 调试（4）

| 技能 | 路径 | 用途 |
|------|------|------|
| `ptx-debug` | [`.opencode/skills/ptx-debug/`](../../.opencode/skills/ptx-debug/) | PTX 调试自动化（配置选择 + 场景化方法） |
| `regression-bisect` | [`.opencode/skills/regression-bisect/`](../../.opencode/skills/regression-bisect/) | 重构后测试回归定位（git bisect + 行级 diff） |
| `state-modification-audit` | [`.opencode/skills/state-modification-audit/`](../../.opencode/skills/state-modification-audit/) | 状态变量全项目读写交叉引用 |
| `oracle-prompting` | [`.opencode/skills/oracle-prompting/`](../../.opencode/skills/oracle-prompting/) | Oracle 子代理防虚构提示词构造 |

### PTX 仿真核心（5）

| 技能 | 路径 | 用途 |
|------|------|------|
| `ptx-instruction-pipeline` | [`.opencode/skills/ptx-instruction-pipeline/`](../../.opencode/skills/ptx-instruction-pipeline/) | 指令执行流水线全貌（调度 → handler → 屏障） |
| `ptx-barrier-mechanism` | [`.opencode/skills/ptx-barrier-mechanism/`](../../.opencode/skills/ptx-barrier-mechanism/) | 屏障机制全解（S_BAR / S_BAR_WARP_SYNC） |
| `ptx-lessons-learned` | [`.opencode/skills/ptx-lessons-learned/`](../../.opencode/skills/ptx-lessons-learned/) | 项目经验沉淀（跨模块状态翻译、递归锁等） |
| `ptx-lane-tracer` | [`.opencode/skills/ptx-lane-tracer/`](../../.opencode/skills/ptx-lane-tracer/) | Lane 级追踪调试 |
| `ptx-lane-verification` | [`.opencode/skills/ptx-lane-verification/`](../../.opencode/skills/ptx-lane-verification/) | Lane 行为验证 |

### 语法与解析（2）

| 技能 | 路径 | 用途 |
|------|------|------|
| `ptx-grammar-modification` | [`.opencode/skills/ptx-grammar-modification/`](../../.opencode/skills/ptx-grammar-modification/) | ANTLR4 语法修改流程（强制 TDD） |
| `ptxir-serialization` | [`.opencode/skills/ptxir-serialization/`](../../.opencode/skills/ptxir-serialization/) | PTXIR 二进制序列化格式 |

### OpenSpec 流程（5）

| 技能 | 路径 | 用途 |
|------|------|------|
| `openspec-propose` | [`.opencode/skills/openspec-propose/`](../../.opencode/skills/openspec-propose/) | 提议新 change（proposal + design + tasks） |
| `openspec-apply-change` | [`.opencode/skills/openspec-apply-change/`](../../.opencode/skills/openspec-apply-change/) | 实施 OpenSpec change 的 tasks |
| `openspec-archive-change` | [`.opencode/skills/openspec-archive-change/`](../../.opencode/skills/openspec-archive-change/) | 归档完成的 change（含 postmortem prompt） |
| `openspec-explore` | [`.opencode/skills/openspec-explore/`](../../.opencode/skills/openspec-explore/) | 探索模式（需求澄清 + 设计预研） |
| `openspec-sync-specs` | [`.opencode/skills/openspec-sync-specs/`](../../.opencode/skills/openspec-sync-specs/) | 同步 delta spec 到 main spec |

### 合规与测试（2）

| 技能 | 路径 | 用途 |
|------|------|------|
| `adr-compliance-check` | [`.opencode/skills/adr-compliance-check/`](../../.opencode/skills/adr-compliance-check/) | 开发完成后 ADR 合规检查 |
| `test-coverage-enforcer` | [`.opencode/skills/test-coverage-enforcer/`](../../.opencode/skills/test-coverage-enforcer/) | 测试覆盖强制（unit + integration + e2e 三层） |

### 已禁用（1）

| 技能 | 原路径 | 状态 | 原因 |
|------|--------|------|------|
| `three-mode-testing` | [`.opencode/skills.disable/three-mode-testing/`](../../.opencode/skills.disable/three-mode-testing/) | `[disabled]`（commit `14c8eeb`） | 已被 `tests/unit/` + `tests/integration/` + `tests/e2e/` 三层物理目录替代 |

---

## 技术参考（不可加载，保留在 docs/skills/）

| 文档 | 用途 |
|------|------|
| [post-dominator-algorithm.md](./post-dominator-algorithm.md) | Post-Dominator 算法详解（CFG 重建、SSA 转换） |
| [simt-reconvergence.md](./simt-reconvergence.md) | SIMT 收敛技术（branch reconvergence、implicit enforcement） |

> **历史说明**：原 `cfg-builder-pattern.md` 和 `tdd-workflow.md` 已删除（内容已迁移至 `.opencode/skills/ptx-lessons-learned/` 与 `.opencode/skills/test-coverage-enforcer/`）。

---

## 📖 学习路径

### 新开发者

1. **测试基础** → `test-coverage-enforcer`（unit + integration + e2e 三层）
2. **SIMT 基础** → [`simt-reconvergence.md`](./simt-reconvergence.md) + `ptx-instruction-pipeline`
3. **算法基础** → [`post-dominator-algorithm.md`](./post-dominator-algorithm.md)
4. **经验沉淀** → `ptx-lessons-learned`（必读，避免重蹈 16 个失败模式）

### 架构师

1. **算法** → [`post-dominator-algorithm.md`](./post-dominator-algorithm.md)
2. **架构** → [`simt-reconvergence.md`](./simt-reconvergence.md) + `ptx-instruction-pipeline`
3. **决策** → `adr-compliance-check` + 浏览 [`docs/adr/`](../adr/)
4. **流程** → `openspec-propose` + `openspec-apply-change` + `openspec-archive-change`

### 调试人员

1. **快速定位** → `ptx-debug`
2. **回归定位** → `regression-bisect`
3. **状态审计** → `state-modification-audit`
4. **Oracle 咨询** → `oracle-prompting`（防虚构）

---

## 🔧 技能加载机制

> **OpenCode 会自动发现** [`.opencode/skills/`](../../.opencode/skills/) 下所有 `SKILL.md`。技能在 `description` 字段匹配用户请求时自动加载。

**新增技能**：在 `.opencode/skills/<name>/SKILL.md` 创建 → 本 README 同步添加一行（按需分类）。

**禁用技能**：移至 `.opencode/skills.disable/<name>/` → 本 README 标记 `[disabled]`（保留可见以供历史参考）。

---

## 📊 技能统计

| 类别 | 数量 |
|------|------|
| 调试 | 4 |
| PTX 仿真核心 | 5 |
| 语法与解析 | 2 |
| OpenSpec 流程 | 5 |
| 合规与测试 | 2 |
| **Active 总计** | **17** |
| 已禁用 | 1 |
| **全部** | **18** |

> 自动统计来源：当前 `.opencode/skills/` 目录 + `.opencode/skills.disable/` 目录

---

**维护**: 与 `.opencode/skills/` 同步  
**最后更新**: 2026-07-03 (C4 docs-readme-rebuild Fix #2)
