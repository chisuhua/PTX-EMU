# PTX-EMU Roadmap

> **维护**: PTX-EMU Architecture Team
> **当前阶段**: Phase 10 — Documentation & Release（β 完成中）+ Phase 3 结构债务修复 + Phase 12.2 PTXIR Cubin 集成（实施中）
> **最后更新**: 2026-08-07
> **关联**: [docs/roadmap/post-phase3-debt-roadmap.md](docs/roadmap/post-phase3-debt-roadmap.md)（详细债务清单）
> **参考**: [docs/README.md](docs/README.md)（文档索引）

---

## 当前状态

| 维度 | 数据 |
|------|------|
| ADR 数 | 24（ADR-0001 ~ ADR-0024，0024 v1.1 amendment 2026-08-07） |
| OpenSpec 已归档 | 18 个 |
| 活跃 changes | 1 (`implement-ptxir-cubin-embed-extension`) |
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
| 12.1 | PTXIR 二进制格式 | ✅ 2026-07-30 | ADR-0023 + ADR-0011 升级 |
| **12.2** | **PTXIR Cubin 集成** | **📋 2026-08-07 实施中** | **ADR-0024 v1.1 + OpenSpec change** |

## Phase 12.2 PTXIR Cubin 集成（实施中）

### 目标

依据 [ADR-0024 v1.1](docs/adr/ADR-0024-ptxir-cubin-embed-extension.md) (2026-08-07 amendment)，将 PTXIR 嵌入到最终可执行文件末尾（ELF 容忍尾部 overlay data），使 PTX-EMU 能从 embed 段反序列化 PTXIR 并复用 `set_ptx_context()` 主路径，同时保留 NVIDIA 工具链兼容性（cub level 工具独立支持）。

### 关联 OpenSpec change

- `openspec/changes/implement-ptxir-cubin-embed-extension/` — 已 plan-done（2026-08-07 commit `6a531428`）

### 5 个 commits（已 plan，含 governance check）

| Commit | 内容 | 状态 |
|--------|------|:--:|
| 0 | ADR-0024 v1.1 amendment (footer layout + magic literal change) | ✅ 2026-08-07 |
| 1 | PTXIRLoader + PtxContextAdapter + config + unit tests | 📋 |
| 2 | cudart dispatch 集成 + integration tests | 📋 |
| 3 | tools/ CLI + e2e tests | 📋 |
| 4 | roadmap.md + 根 README.md 文档同步 | 📋 |

### 关键约束

- `PTXIR_MODE` 默认 OFF → 字节级兼容现状
- `PTXIR_EMBED_MAGIC = {'P','T','X','E','M','B','\x01','\x00'}` — 已 2026-08-07 ADR amendment
- byte source = `/proc/self/exe` 末尾（非 `fat_bin` 参数 — dead parameter）
- v1 显式为 single-kernel scope（PTXIR v3 限制）

## 当前任务（Phase 10: Documentation & Release β）

### 🔴 阻塞项

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| RD-1 | 创建 root 级 roadmap.md (本文件) | ✅ 2026-07-23 | arch-done 门控 |
| RD-2 | 初始化 `.rddf/state/` + `.arch-handoff.json` | ✅ 2026-07-23 | arch-done 门控 |
| RD-3 | ADR 重命名为 `ADR-NNNN` 格式 | ✅ 2026-07-23 | rdd-workflow 合规 |
| RD-4 | CppTLM unified build ADR-0022 签署 | ✅ 2026-07-23 | CppTLM Oracle 审查 |
| RD-5 | Phase 12.2 governance check (ADR-0024 magic + layout) | ✅ 2026-08-07 | §合规检查 #6 |

### 🟡 进行中

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| C-* | C 系列代码债务 (18 项) | ⏳ | [post-phase3-debt-roadmap §1.2](docs/roadmap/post-phase3-debt-roadmap.md) |
| D-* | D 系列文档债务 (6 项) | ⏳ | [post-phase3-debt-roadmap §1.3](docs/roadmap/post-phase3-debt-roadmap.md) |
| P12.2 | PTXIR Cubin 集成（5 commits） | 📋 2026-08-07 | `openspec/changes/implement-ptxir-cubin-embed-extension/` |

### 🟢 计划

| # | 任务 | 状态 | 关联 |
|---|------|:--:|------|
| H5 | Hopper/Blackwell tcgen05 后续 | 📋 | [ADR-0016](docs/adr/ADR-0016-blackwell-only-tcgen05.md) |
| S1 | 符号覆盖 CI 测试 | 📋 | [ADR-0022](docs/adr/ADR-0022-cpptlm-unified-build.md) |
| S2 | cpptlm_core_minimal 拆分 | 📋 | ADR-0022 远期优化 |
| P12.3 | PTXIR Section TOC v2（多 kernel 支持） | 📋 | 后续 change |

## 下一步

```
arch-done + design-done + plan-done (Phase 12.2) → guide-ship:
  1. 创建 worktree (推荐 .worktrees/baseline-ptxir-cubin per ptx-lessons-learned §4)
  2. 执行 5 个 commit (C0 已完成, C1-C4 待实施)
  3. 每个 commit 后跑 ctest + sanity.sh
  4. plan-done → archive + cleanup
```

---

**维护者**: PTX-EMU Architecture Team
**日期**: 2026-08-07