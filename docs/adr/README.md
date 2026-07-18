# Architecture Decision Records (ADR)

本目录记录 PTX-EMU 项目的所有重要架构决策。

## 什么是 ADR？

ADR（Architecture Decision Record）是一种轻量级的架构决策文档格式，用于记录：
- **做了什么决策**
- **为什么这样做**
- **考虑了哪些替代方案**
- **决策的上下文和约束条件**

## 为什么需要 ADR？

1. **记录决策上下文**：避免"只知其然，不知其所以然"
2. **保证演进一致性**：后续修改可以对照 ADR 检查是否违背既有决策
3. **降低知识流失风险**：新人可以通过 ADR 快速理解架构演进脉络
4. **提高评审质量**：PR 可以关联相关 ADR，评审时检查是否符合既有决策

## ADR 生命周期

```
Proposed → Accepted → Active → Superseded (被新决策替代)
                          → Deprecated (不再推荐)
```

## 目录结构

```
docs/adr/
├── README.md              # 本文件 - ADR 索引
├── template.md            # ADR 模板
├── 0001-exception-hierarchy.md
├── 0002-pc-unification.md
└── ...
```

## ADR 索引

### Active (当前有效)

| # | 标题 | 状态 | 日期 | 关联任务 |
|---|------|------|------|---------|
| [0001](./0001-exception-hierarchy.md) | 异常层次体系替代 assert | Active | 2026-05-03 | T11.1.1-T11.1.4 |
| [0002](./0002-pc-unification.md) | PC 权威源统一到 WarpState | Active | 2026-05-04 | T11.2.1-T11.2.6 |
| [0003](./0003-commit-pc-pattern.md) | commit_pc / force_set_pc 分离 | Active | 2026-05-04 | T11.2.2 |
| [0004](./0004-natural-stall-mechanism.md) | 自然停顿机制 is_warp_ready_to_fetch | Active | 2026-05-04 | T11.2.4 |
| [0005](./0005-memory-region-registration.md) | MemoryRegion 注册机制 | Active | 2026-05-03 | T11.1.5-T11.1.7 |
| [0006](./0006-simt-stack-management.md) | SIMT Stack 显式控制流管理 | Active | 2026-05-05 | Phase 2 |
| [0007](./0007-cfg-post-dominator.md) | CFG Post-Dominator 收敛分析 | Active | 2026-05-05 | Phase 1 |
| [0008](./0008-barrier-semantics.md) | Barrier 语义增强 - Convergence + Memory Fence | Active | 2026-05-05 | Phase 4 |
| [0009](./0009-xmacro-instruction-dispatch.md) | X-Macro + Weak Symbol 指令分发模式 | Active | 2026-05-05 | Phase 0-9 |
| [0010](./0010-fake-cuda-runtime.md) | Fake CUDA Runtime 拦截机制 | Active | 2026-05-05 | Phase 0 |
| [0012](./0012-per-thread-pc.md) | Per-Thread PC 设计（Volta+ SIMT 模型） | Active | 2026-05-05 | Phase 3 |
| [0015](./0015-cvt-strategy-pattern.md) | CVT 指令策略模式重构 (Composition over Inheritance) | Active | 2026-06-23 | T2-6 (Phase 3) |
| [0016](./0016-blackwell-only-tcgen05.md) | Skip pre-Blackwell WMMA, only implement Blackwell tcgen05 | Accepted | 2026-07-04 | `openspec/changes/implement-wmma-tensor-core/` |
| [0018](./0018-tcgen05-cta-group-restriction.md) | tcgen05 cta_group::2 throws UnsupportedInstructionException | Accepted | 2026-07-12 | `openspec/changes/fix-tcgen05-commit-wait-group/` |
| [0019](./0019-pc-management-extraction.md) | ThreadContext 持续瘦身：MemoryAccessor + InstructionPipeline accessor 方案 | Active | 2026-07-14 | `openspec/changes/god-class-refactor-thread-context-phase3/` |
| [0020](./0020-cpptlm-injection-points.md) | 接受 CppTLM Phase 8.B D1-Full 注入（IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming） | Accepted | 2026-07-16 | `openspec/changes/cpptlm-phase8b-injection-points/` |
| [0021](./0021-cpptlm-d1-full-integration.md) | CppTLM D1-Full MemoryBridge 集成（D-PTX-1~6 + HSK-1/2/3） | Active | 2026-07-16 | `openspec/changes/cpptlm-d1-full/` |

### Proposed (规划中)

| # | 标题 | 状态 | 日期 | 关联任务 |
|---|------|------|------|---------|
| [0011](./0011-pipeline-architecture.md) | PTX→PTXIR 多阶段 Pipeline 架构 | Proposed | 2026-05-05 | Phase 12.1 |
| [0013](./0013-statement-factory-test-unification.md) | StatementContext 测试统一模式 — statement_factory + execute_warp_instruction | Proposed | 2026-05-09 | — |
| [0014](./0014-independent-thread-scheduling.md) | Independent Thread Scheduling (ITS) 支持 | Proposed | 2026-05-25 | BUG-SIMT-001 |

### Superseded (已被替代)

| # | 标题 | 被替代为 | 日期 |
|---|------|---------|------|
| - | - | - | - |

## 使用流程

### 新建 ADR

1. 复制 `template.md` 为新文件，命名格式：`NNNN-short-title.md`
2. 填写模板内容，确保包含决策背景和替代方案分析
3. 更新本文件的索引表格
4. 在 PR 中提交 ADR 变更

### 更新 ADR

- 如果决策微调：在原 ADR 中添加"更新记录"部分
- 如果决策被推翻：创建新 ADR，将原 ADR 标记为 Superseded

### 在开发中使用 ADR

- **任务开始前**：检查是否有相关 ADR，如有则遵循
- **架构变更时**：先更新或新建 ADR，再写代码
- **PR 评审时**：检查是否符合相关 ADR 的决策

---

**维护**: PTX-EMU Architecture Team  
**最后更新**: 2026-07-16  
**ADR 总数**: 20（其中 Active 14 / Accepted 3 / Proposed 3 / Superseded 0）

## 最近更新

| 日期 | 更新内容 | 关联 ADR |
|------|---------|---------|
| 2026-07-16 | **cpptlm-d1-full 状态推进**：ADR-0021 Proposed → Active；ADR-0020 Proposed → Accepted；2 轮 Metis pre-impl review + 3 阶段 12 commits 修复所有 5 个 BLOCKER（B1 ABI 实现 / B2 sync loop / B3 stream destroy UB / B4 HSK 一致性 / B5 CMake 文档同步）+ sister spec 附录 + Postmortem 沉淀 | 0020, 0021 |
| 2026-07-15 | 添加 CppTLM D1-Full MemoryBridge 集成 ADR（D-PTX-1~6 决策 + HSK-1/2/3 握手 + cpptlm_bridge.h ABI 真值源） | 0021 |
| 2026-07-14 | 添加 CppTLM Phase 8.B D1-Full 注入点接受决策（3 个纯虚接口 + SMContext 3 setter + WarpContext 扩展 + RegisterAnalyzer 增强 + exe_once 三段式注入） | 0020 |
| 2026-07-14 | 添加 ThreadContext 持续瘦身 ADR（MemoryAccessor + InstructionPipeline accessor 方案） | 0019 |
| 2026-07-12 | 添加 tcgen05 cta_group::2 throw 语义 ADR（formalize scattered throw across 11 handlers） | 0018 |
| 2026-06-23 | 添加 CVT 策略模式重构 ADR (T2-6 完成) | 0015 |
| 2026-05-06 | 添加 pc_overridden_ 机制说明、while 循环收敛模式、Fallback 策略 | 0006, 0007, 0008 |
| 2026-05-06 | 补充 barrier 场景回归测试覆盖 | 0008 |
| 2026-05-06 | 添加 handle_branch PC 过滤说明、更新合规检查项 | 0006 |
