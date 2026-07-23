# PTX-EMU Roadmap

> **维护**: PTX-EMU Architecture Team
> **当前阶段**: Phase 10 — Documentation & Release（β 完成中）+ Phase 3 结构债务修复
> **最后更新**: 2026-07-23
> **关联**: [docs/roadmap/post-phase3-debt-roadmap.md](docs/roadmap/post-phase3-debt-roadmap.md)（详细债务清单）
> **参考**: [docs/README.md](docs/README.md)（文档索引）

---

## 当前状态

| 维度 | 数据 |
|------|------|
| ADR 数 | 21（ADR-0001 ~ ADR-0022） |
| OpenSpec 已归档 | 18 个 |
| 活跃 changes | 0 |
| 测试覆盖 | unit / integration / e2e 三层物理隔离 |
| PTX 语法测试 | `./tests/ptx/test_all_ptx.sh` 45/45 |
| CppTLM 集成 | D1-Full MemoryBridge 已归档（ADR-0021） |
| 最近审计 | [debt-audit-2026-07-02](docs/audits/debt-audit-2026-07-02.md) |

## 已完成阶段

| Phase | 名称 | 状态 | 关键交付 |
|-------|------|:--:|------|
| 0-6 | 基础架构 (PTX 解析/执行/内存) | ✅ | ANTLR4 解析器 + IR + 解释执行 |
| 7 | Reconvergence Validation | ✅ | SIMT v2 收敛验证 |
| 8 | Performance Benchmark | ✅ | 基准性能数据 |
| 9 | SIMT Stack Integration | ✅ | Per-thread PC + CFG post-dominator |
| Phase 3-2026 | 结构债务修复 | ⏳ | A 系列 0 剩余；C 系列 18；D 系列 6 |

## 当前任务（Phase 10: Documentation & Release β）

### 🔴 阻塞项

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| RD-1 | 创建 root 级 roadmap.md (本文件) | ✅ 2026-07-23 | arch-done 门控 |
| RD-2 | 初始化 `.rddf/state/` + `.arch-handoff.json` | ✅ 2026-07-23 | arch-done 门控 |
| RD-3 | ADR 重命名为 `ADR-NNNN` 格式 | ✅ 2026-07-23 | rdd-workflow 合规 |
| RD-4 | CppTLM unified build ADR-0022 签署 | ✅ 2026-07-23 | CppTLM Oracle 审查 |

### 🟡 进行中

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| C-* | C 系列代码债务 (18 项) | ⏳ | [post-phase3-debt-roadmap §1.2](docs/roadmap/post-phase3-debt-roadmap.md) |
| D-* | D 系列文档债务 (6 项) | ⏳ | [post-phase3-debt-roadmap §1.3](docs/roadmap/post-phase3-debt-roadmap.md) |

### 🟢 计划

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| H5 | Hopper/Blackwell tcgen05 后续 | 📋 | [ADR-0016](docs/adr/ADR-0016-blackwell-only-tcgen05.md) |
| S1 | 符号覆盖 CI 测试 | 📋 | [ADR-0022](docs/adr/ADR-0022-cpptlm-unified-build.md) |
| S2 | cpptlm_core_minimal 拆分 | 📋 | ADR-0022 远期优化 |

## 下一步

```
arch-done → guide-plan:
  1. scan 项目状态
  2. propose 新的 OpenSpec changes（开始消耗 C/D 系列债务）
  3. deps 分析
  4. plan-done → guide-ship 执行
```

---

**维护者**: PTX-EMU Architecture Team
**日期**: 2026-07-23