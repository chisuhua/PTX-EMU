# PTX-EMU 项目文档

> **项目**: PTX-EMU SIMT v2.0  
> **状态**: Phase 9/10 完成 (90%)  
> **测试**: 38/38 测试通过 (100%)  
> **版本**: v2.0 (准备发布)

---

## 📚 文档导航

| 类别 | 路径 | 文档数 | 用途 |
|------|------|--------|------|
| 🏗️ **架构文档** | [`architecture/`](./architecture/) | 1 | 系统架构设计 |
| 📖 **开发指南** | [`developer-guide/`](./developer-guide/) | 1 | 开发指导手册 |
| 🛠️ **技能沉淀** | [`skills/`](./skills/) | 4 | 技术技能总结 |
| 📊 **项目报告** | [`reports/`](./reports/) | 6 | Phase 与测试报告 |
| 📎 **附录** | [`appendix/`](./appendix/) | 3 | 补充资料 (CHANGELOG, 术语表) |
| 🔬 **调研文档** | [`research/`](./research/) | 1 主题 | 外部技术调研（NVIDIA 官方 + 开源参考） |
| 🗄️ **历史归档** | [`archive/`](./archive/) | 50+ | 历史文档存档 |

---

## 🚀 快速开始

### 新开发者路径

1. **阅读 GETTING-STARTED** → `developer-guide/`
2. **理解 SIMT 架构** → `architecture/SIMT-ARCHITECTURE-V2.md`
3. **学习 CFG Builder** → `skills/post-dominator-algorithm.md`
4. **参考测试指南** → `developer-guide/TESTING-GUIDE.md` (待创建)

### 架构师路径

1. **阅读 SIMT 架构** → `architecture/SIMT-ARCHITECTURE-V2.md`
2. **查看 CFG 设计** → `skills/post-dominator-algorithm.md`
3. **参考项目报告** → `reports/phase-reports/`

### 测试工程师路径

1. **阅读测试指南** → `developer-guide/TESTING-GUIDE.md` (待创建)
2. **查看测试报告** → `reports/test-reports/`
3. **参考边界测试** → 技能文档和 Phase 报告

---

## 📋 核心文档推荐

### Top 5 必读文档

| # | 文档 | 路径 | 适合人群 |
|---|------|------|---------|
| 1 | SIMT-ARCHITECTURE-V2.md | [`architecture/`](./architecture/) | 所有人 |
| 2 | post-dominator-algorithm.md | [`skills/`](./skills/) | 架构师 |
| 4 | PERFORMANCE-BENCHMARK-REPORT.md | [`reports/phase-reports/`](./reports/phase-reports/) | 性能工程师 |
| 5 | post-dominator-algorithm.md | [`skills/`](./skills/) | 算法开发者 |

---

## 🔗 PTX 规范参考

**PTX ISA 完整规范**已在 cuda-ptx 技能中维护 (405 个文件):

```
/home/ubuntu/.config/opencode/skills/cuda-ptx/references/ptx-docs/
```

**项目归档**: PTX 指令参考文档已归档到 [`archive/ptx-instruction-reference/`](./archive/ptx-instruction-reference/)

**原因**: 避免重复维护，指向权威来源

---

## 📈 项目里程碑

| Phase | 名称 | 状态 | 报告 |
|-------|------|------|------|
| 0 | 设计与规划 | ✅ 完成 | [`archive/`](./archive/) |
| 1 | CFG Builder Core | ✅ 完成 | - |
| 2 | SIMT Stack | ✅ 完成 | - |
| 3 | Per-Thread PC | ✅ 完成 | - |
| 4 | Barrier Enhancement | ✅ 完成 | - |
| 5 | Integration & Testing | ✅ 完成 | (已归档) |
| 6 | Final Verification | ✅ 完成 | (已归档) |
| 7 | Reconvergence Validation | ✅ 完成 | [`Phase 7 Report`](./reports/phase-reports/PHASE7-FINAL-REPORT.md) |
| 8 | Performance Benchmark | ✅ 完成 | [`Phase 8 Report`](./reports/phase-reports/PHASE8-PERFORMANCE-REPORT.md) |
| 9 | SIMT Stack Integration | ✅ 完成 | (已归档) |
| 10 | Documentation & Release | ⏳ In Progress | - |

---

## 📊 代码统计

| 类型 | 行数 |
|------|------|
| 核心代码 | ~750 行 |
| 集成代码 | ~30 行 |
| 测试代码 | ~600 行 |
| 文档 | ~3,500 行 |
| 测试用例 | 38 个 |
| Git Commits | 22+ |

---

## 🔧 开发环境

### 构建命令

```bash
# Configure
cmake -S . -B build

# Build
cmake --build build

# Test
ctest --test-dir build --output-on-failure
```

### 测试命令

```bash
# Run all tests
ctest --test-dir build

# Run specific suite
ctest --test-dir build -R "cfg|simt"

# Verbose output
ctest --test-dir build -R "cfg" -V
```

---

## 📞 联系与支持

- **项目仓库**: github.com/chisuhua/PTX-EMU
- **文档版本**: v2.0
- **最后更新**: 2026-04-11

---

## 📝 文档维护指南

### 添加新文档

1. 确定文档类别 (architecture/developer-guide/skills/reports/archive)
2. 使用 Markdown 格式
3. 添加顶部元数据 (标题、日期、状态)
4. 更新相关 README 索引

### 文档版本控制

- **架构文档**: 只保留最新版本，旧版本归档
- **报告文档**: 永久保留，按 Phase 组织
- **技能文档**: 持续更新，版本号在内容中标注

### 归档规则

满足以下条件的文档应归档到 `archive/`:
- ✅ Phase 计划 (已执行)
- ✅ 审批请求 (已完成)
- ✅ 代码审查 (已合并)
- ✅ 过时设计文档
- ✅ 外部规范复制 (如 PTX ISA reference)

---

**最后更新**: 2026-04-11  
**维护者**: PTX-EMU Architecture Team  
**文档版本**: v2.0
