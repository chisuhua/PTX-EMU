# 开发指南

> **版本**: v2.0  
> **最后更新**: 2026-04-11

---

## 📁 指南列表

| 指南 | 适合人群 | 状态 |
|------|---------|------|
| [GETTING-STARTED.md](./GETTING-STARTED.md) | 新开发者 | ✅ |
| [TESTING-GUIDE.md](./TESTING-GUIDE.md) | 测试工程师 | ✅ |
| [PERFORMANCE-GUIDE.md](./PERFORMANCE-GUIDE.md) | 性能工程师 | ✅ |
| [CFG-INTEGRATION-GUIDE.md](./CFG-INTEGRATION-GUIDE.md) | 后端开发 | ✅ |
| [DEBUG-CONFIG-GUIDE.md](./DEBUG-CONFIG-GUIDE.md) | 所有开发者 | ✅ |
| [DEBUG-QUICK-REFERENCE.md](./DEBUG-QUICK-REFERENCE.md) | 所有开发者 | ✅ |
| [DEBUGGING-GUIDE.md](./DEBUGGING-GUIDE.md) | 所有开发者 | ✅ |
| [REGRESSION-DEBUGGING-GUIDE.md](./REGRESSION-DEBUGGING-GUIDE.md) | 所有开发者 | ✅ 新增 |
| [PTX-DEBUG-SKILL-USAGE.md](./PTX-DEBUG-SKILL-USAGE.md) | 所有开发者 | ✅ |
| [THREE-MODE-TESTING-GUIDE.md](./THREE-MODE-TESTING-GUIDE.md) | 测试工程师 | ✅ 四模式（含 Mode 4 PTXIR） |
| [BARRIER-PROGRAMMING-REFERENCE.md](./BARRIER-PROGRAMMING-REFERENCE.md) | 所有开发者 | ✅ Barrier 参考 |

### 🛠️ 修复 Postmortem / Open Issue 文档

**Postmortem（已修复，作为经验沉淀保留）**

| 文档 | 修复内容 | 关键结论 |
|------|---------|---------|
| [postmortem-fix-1-gate-active-vs-return-mask.md](./postmortem-fix-1-gate-active-vs-return-mask.md) | Fix 1 — gate 阻塞范围 | 门控用 `return_mask`（非 `active_mask`）；处方必须覆盖 taken + fall-through 两条路径 |
| [postmortem-fix-3-is-converged-skip-inactive.md](./postmortem-fix-3-is-converged-skip-inactive.md) | Fix 3 — `is_converged` 不应跳过暂时不活跃的 lane | `is_converged` 只跳 `is_exited`，不跳 `!is_active` |
| [postmortem-sbar-deadlock-fix.md](./postmortem-sbar-deadlock-fix.md) | Fix 2 — S_BAR 死锁修复 (7 bugs) | `release_cta_barrier` 需显式恢复 `is_active`；`CTABarrier::reset()` 不清 `is_initialized_`；`step_warp` 需边界检查 |

**Open Issue（未实现，待后续修复）**

| 文档 | 问题 | 状态 |
|------|------|------|
| [open-fix-2-sbar-deadlock.md](./open-fix-2-sbar-deadlock.md) | Fix 2 — `S_BAR` 死锁（`bar.sync 0`） | **FIXED (2026-07-01)** → 见 [postmortem-sbar-deadlock-fix.md](./postmortem-sbar-deadlock-fix.md) |

> **铁律（Fix 1 + Fix 3 后确立）**：SIMTStackEntry 的三个字段**绝对不能互换**：
> - `active_mask` — 收敛判定（`is_converged`）
> - `return_mask` — 门控阻塞（gate）+ `exec_mask` 恢复（`check_reconvergence`）
> - `is_active` — `update_active_mask` 双向同步（self-heal）
>
> 详见 [ADR-0006 §"三个字段的角色分工"](../adr/0006-simt-stack-management.md) 与 [KNOWN_ISSUES §B4.2](./KNOWN_ISSUES.md)。

---

## 🚀 快速导航

### 新开发者

**起点**: [`GETTING-STARTED.md`](./GETTING-STARTED.md)

1. 安装工具 → 构建项目 → 运行测试
2. 学习 SIMT 架构
3. 理解 CFG Builder

### 测试工程师

**起点**: [`TESTING-GUIDE.md`](./TESTING-GUIDE.md)

1. 运行测试套件
2. 编写新测试
3. 调试失败测试

### 性能工程师

**起点**: [`PERFORMANCE-GUIDE.md`](./PERFORMANCE-GUIDE.md)

1. 性能基准测试
2. 热点分析
3. 优化建议

### 后端开发者

**起点**: [`CFG-INTEGRATION-GUIDE.md`](./CFG-INTEGRATION-GUIDE.md)

1. 理解 CFG 架构
2. 集成指南
3. 测试验证

---

## 📚 相关资源

| 资源 | 路径 |
|------|------|
| 架构文档 | [`../architecture/`](../architecture/) |
| 技能沉淀 | [`../skills/`](../skills/) |
| 项目报告 | [`../reports/`](../reports/) |

---

**维护**: 持续更新  
**最后更新**: 2026-04-11
