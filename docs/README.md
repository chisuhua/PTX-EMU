# PTX-EMU 项目文档

> **当前状态**: Phase 3 结构债务修复中（参考 [docs/audits/debt-audit-2026-07-02.md](./audits/debt-audit-2026-07-02.md)）
> **SIMT 架构**: v2.0（per-thread PC + CFG-based post-dominator）
> **测试隔离**: unit / integration / e2e 三层物理目录
> **OpenSpec**: v1.4.1

---

## 📚 文档导航（16 个子目录，按功能分组）

### 1. 架构与设计

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`adr/`](./adr/) | 15 | 架构决策记录（ADR-0001~0015）— 重大设计选择的上下文与决策 |
| [`architecture/`](./architecture/) | 4 | 系统架构设计（SIMT v2、GPGPU-SIM 对比、sm_90/100） |
| [`technical_design/`](./technical_design/) | 2 | 详细技术设计（barrier module、implicit reconvergence） |

### 2. 开发与测试

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`developer-guide/`](./developer-guide/) | 18 | 开发指南（构建、测试、调试、postmortem） |
| [`testing/`](./testing/) | 2 | 测试文档（simt-complete-test-plan、TEST_DOCUMENTATION） |
| [`plans/`](./plans/) | 1 | 实施计划 |

### 3. 报告与审计

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`reports/`](./reports/) | 10 | 项目报告（Phase 报告、test 报告、bug 复现） |
| [`audits/`](./audits/) | 4 | 健康审计与债务审查（HEALTH-AUDIT + ERRATA + debt-audit） |

### 4. 流程与技能

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`dev-process/`](./dev-process/) | 2 | 开发流程（lessons-learned、debugging-strategy） |
| [`skills/`](./skills/) | 8 | 技能沉淀（人类可读导航，**可加载技能在 `.opencode/skills/`**） |
| [`superpowers/`](./superpowers/) | 6 | 超级能力沉淀（specs/plans/findings） |

### 5. PTX 与解析

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`ptx/`](./ptx/) | 1 | PTX 指令参考（**注：当前为骨架，详见 README**） |

### 6. Roadmap 与调研

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`roadmap/`](./roadmap/) | 1 | Roadmap（活跃 OpenSpec changes 即未来项） |
| [`research/`](./research/) | 8 | 外部技术调研（NVIDIA 官方 + 开源参考） |

### 7. 附录与归档

| 目录 | 文档数 | 用途 |
|------|--------|------|
| [`appendix/`](./appendix/) | 4 | 附录（CHANGELOG、GLOSSARY、REFERENCES、RELEASE-NOTES） |
| [`archive/`](./archive/) | 50+ | 历史归档（pre-simt-v2、PTX 指令副本等） |

> **自动统计**: 上述"文档数"由 `scripts/check-docs-index.sh` 生成（`find docs/<dir> -type f | wc -l`）

---

## 🚀 快速开始

### 新开发者路径

1. **理解项目** → 根 [README.md](../README.md) + [AGENTS.md](../AGENTS.md)
2. **理解 SIMT 架构** → [`architecture/SIMT-ARCHITECTURE-V2.md`](./architecture/SIMT-ARCHITECTURE-V2.md)
3. **学习 CFG Builder** → [`skills/post-dominator-algorithm.md`](./skills/post-dominator-algorithm.md)
4. **参考开发指南** → [`developer-guide/`](./developer-guide/)
5. **了解历史教训** → [`dev-process/lessons-learned.md`](./dev-process/lessons-learned.md)

### 架构师路径

1. **阅读 SIMT v2 架构** → [`architecture/`](./architecture/)
2. **查阅 ADR 决策历史** → [`adr/`](./adr/)
3. **详细技术设计** → [`technical_design/`](./technical_design/)

### 测试工程师路径

1. **测试类型与标签** → [`testing/TEST_DOCUMENTATION.md`](./testing/TEST_DOCUMENTATION.md)
2. **SIMT 完整测试计划** → [`testing/simt-complete-test-plan.md`](./testing/simt-complete-test-plan.md)
3. **测试报告** → [`reports/test-reports/`](./reports/test-reports/)

### 调试人员路径

1. **调试策略** → [`dev-process/debugging-strategy.md`](./dev-process/debugging-strategy.md)
2. **PTX 调试技能** → [`.opencode/skills/ptx-debug/`](../.opencode/skills/ptx-debug/)
3. **失败案例 postmortem** → [`developer-guide/`](./developer-guide/)

---

## 🔗 PTX 规范参考

**PTX ISA 完整规范**已在外部 cuda-ptx 技能中维护（405 个文件）：

```
~/.config/opencode/skills/cuda-ptx/references/ptx-docs/
```

**项目归档副本**: [`archive/ptx-instruction-reference/`](./archive/ptx-instruction-reference/)（13 个文件，已与外部规范重复，参考用）

**修改 PTX 语法前必读**：[`.opencode/skills/ptx-grammar-modification/`](../.opencode/skills/ptx-grammar-modification/)（强制 TDD 流程）

---

## 📈 项目里程碑（截至 2026-07）

| Phase | 名称 | 状态 | 报告 |
|-------|------|------|------|
| 0-6 | 基础架构 | ✅ 完成 | [`archive/`](./archive/) |
| 7 | Reconvergence Validation | ✅ 完成 | [`PHASE7-FINAL-REPORT.md`](./reports/phase-reports/PHASE7-FINAL-REPORT.md) |
| 8 | Performance Benchmark | ✅ 完成 | [`PHASE8-PERFORMANCE-REPORT.md`](./reports/phase-reports/PHASE8-PERFORMANCE-REPORT.md) |
| 9 | SIMT Stack Integration | ✅ 完成 | [`archive/`](./archive/) |
| 10 | Documentation & Release | ⏳ 进行中 | 当前 |
| 3-2026 | 结构债务修复 | ⏳ 进行中 | [`audits/debt-audit-2026-07-02.md`](./audits/debt-audit-2026-07-02.md) |

> **状态更新方式**: 任何 Phase 状态变更必须同步更新本表格 + 提交审计或报告

---

## 🔧 开发环境

### 构建命令

```bash
# 1. 设置环境（必须！）
. env.sh

# 2. 配置 + 构建
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build

# 3. 跑测试
cd build && ctest --output-on-failure
```

### 测试命令

```bash
# 全部测试
ctest --output-on-failure

# 按标签分组
ctest -L "unit;barrier"           # 屏障相关单元测试
ctest -L "integration;simt"      # SIMT 集成测试
ctest -L "e2e"                   # 端到端 CUDA kernel 测试

# 单个测试（verbose）
ctest -R unit_barrier_module -V
```

更多测试组织规范 → [根 AGENTS.md §TDD 开发流程](../AGENTS.md)

---

## 📋 文档维护规范

### 新增子目录规则

任何 `docs/<new-dir>/` 新建时，**必须同步**更新本文档 §文档导航 表格。

### 新增文档规则

1. 确定所属子目录
2. 使用 Markdown 格式 + 顶部元数据（标题、日期、状态）
3. 大文档添加子目录 + 自身 README
4. 更新本文档相关章节

### 归档规则

满足以下条件之一应归档到 `archive/`:
- ✅ Phase 计划（已执行）
- ✅ 审批请求（已完成）
- ✅ 代码审查（已合并）
- ✅ 过时设计文档
- ✅ 外部规范副本（如 PTX ISA reference）

### 统计信息生成

> **禁止** 在本文档手写"X 测试用例"、"Y 行代码"等数字。所有统计数据由 `scripts/check-docs-index.sh` 自动生成。

---

## 📞 联系与支持

- **项目仓库**: `github.com/chisuhua/PTX-EMU`
- **OpenSpec 活跃 changes**: [`openspec/changes/`](../openspec/changes/)（即未来 roadmap 项）
- **最新审计**: [docs/audits/debt-audit-2026-07-02.md](./audits/debt-audit-2026-07-02.md)

---

**最后更新**: 2026-07-03 (C4 docs-readme-rebuild Fix #1)
**维护者**: PTX-EMU Architecture Team
