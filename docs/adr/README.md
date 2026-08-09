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

### Active / Accepted（当前有效）

| # | 标题 | 状态 | 日期 | 关联任务 |
|---|------|------|------|---------|
| [0001](./ADR-0001-exception-hierarchy.md) | 异常层次体系替代 assert | Active | 2026-05-03 | T11.1.1-T11.1.4 |
| [0002](./ADR-0002-pc-unification.md) | PC 权威源统一到 WarpState | Active | 2026-05-04 | T11.2.1-T11.2.6 |
| [0003](./ADR-0003-commit-pc-pattern.md) | commit_pc / force_set_pc 分离 | Active | 2026-05-04 | T11.2.2 |
| [0004](./ADR-0004-natural-stall-mechanism.md) | 自然停顿机制 is_warp_ready_to_fetch | Active | 2026-05-04 | T11.2.4 |
| [0005](./ADR-0005-memory-region-registration.md) | MemoryRegion 注册机制 | Active | 2026-05-03 | T11.1.5-T11.1.7 |
| [0006](./ADR-0006-simt-stack-management.md) | SIMT Stack 显式控制流管理 | Active | 2026-05-05 | Phase 2 |
| [0007](./ADR-0007-cfg-post-dominator.md) | CFG Post-Dominator 收敛分析 | Active | 2026-05-05 | Phase 1 |
| [0008](./ADR-0008-barrier-semantics.md) | Barrier 语义增强 - Convergence + Memory Fence | Active | 2026-05-05 | Phase 4 |
| [0009](./ADR-0009-xmacro-instruction-dispatch.md) | X-Macro + Weak Symbol 指令分发模式 | Active | 2026-05-05 | Phase 0-9 |
| [0010](./ADR-0010-fake-cuda-runtime.md) | Fake CUDA Runtime 拦截机制 | Active | 2026-05-05 | Phase 0 |
| [0011](./ADR-0011-pipeline-architecture.md) | PTX→PTXIR 多阶段 Pipeline 架构 | **Accepted** | 2026-05-05 (2026-07-30 升级) | Phase 12.1 |
| [0012](./ADR-0012-per-thread-pc.md) | Per-Thread PC 设计（Volta+ SIMT 模型） | Active | 2026-05-05 | Phase 3 |

| [0015](./ADR-0015-cvt-strategy-pattern.md) | CVT 指令策略模式重构 (Composition over Inheritance) | Active | 2026-06-23 | T2-6 (Phase 3) |
| [0016](./ADR-0016-blackwell-only-tcgen05.md) | Skip pre-Blackwell WMMA, only implement Blackwell tcgen05 | Accepted | 2026-07-04 | `openspec/changes/implement-wmma-tensor-core/` |
| [0018](./ADR-0018-tcgen05-cta-group-restriction.md) | tcgen05 cta_group::2 throws UnsupportedInstructionException | Accepted | 2026-07-12 | `openspec/changes/fix-tcgen05-commit-wait-group/` |
| [0019](./ADR-0019-pc-management-extraction.md) | ThreadContext 持续瘦身：MemoryAccessor + InstructionPipeline accessor 方案 | Active | 2026-07-14 | `openspec/changes/god-class-refactor-thread-context-phase3/` |
| [0020](./ADR-0020-cpptlm-injection-points.md) | 接受 CppTLM Phase 8.B D1-Full 注入（IScoreboard / IPipelineLatencyProvider / ITensorCoreTiming） | Accepted | 2026-07-16 | `openspec/changes/cpptlm-phase8b-injection-points/` |
| [0021](./ADR-0021-cpptlm-d1-full-integration.md) | CppTLM D1-Full MemoryBridge 集成（D-PTX-1~6 + HSK-1/2/3） | Active | 2026-07-16 | `openspec/changes/cpptlm-d1-full/` |
| [0022](./ADR-0022-cpptlm-unified-build.md) | CppTLM + PTX-EMU 统一构建链路（`--whole-archive` 替代独立 `.so` + `dlopen`） | Accepted | 2026-07-23 | `openspec/changes/cpptlm-d1-full/` |
| [0023](./ADR-0023-ptxir-binary-format.md) | PTXIR 二进制序列化格式与 7 项架构决策（扁平二进制 + Section TOC + 值枚举） | Accepted | 2026-07-30 | `openspec/changes/archive/2026-06-09-ptxir-serialization-architecture/` |
| [0024](./ADR-0024-ptxir-cubin-embed-extension.md) | Cubin 嵌入 PTXIR 的混合二进制格式（PTXIR-Embedded CUBIN，loader 决策 + extract 工具） | Accepted (v1.1 2026-08-07) | `openspec/changes/implement-ptxir-cubin-embed-extension/` |

### Proposed (规划中)

| # | 标题 | 状态 | 日期 | 关联任务 |
|---|------|------|------|---------|
| [0013](./ADR-0013-statement-factory-test-unification.md) | StatementContext 测试统一模式 — statement_factory + execute_warp_instruction | Proposed | 2026-05-09 | — |
| [0014](./ADR-0014-independent-thread-scheduling.md) | Independent Thread Scheduling (ITS) 支持 | Proposed | 2026-05-25 | BUG-SIMT-001 |
| [0025](./ADR-0025-ptxir-build-cli.md) | `ptxir_build` CLI（PTX → PTXIR 序列化命令行） | Proposed | 2026-08-08 | `openspec/changes/feat-ptxir-nvcc-toolchain/` T13.1 |
| [0026](./ADR-0026-ptxir-default-mode-auto.md) | PTXIR 调度默认模式 = auto（零配置嵌入 binary） | Proposed | 2026-08-08 | `openspec/changes/feat-ptxir-nvcc-toolchain/` T13.2 |
| [0027](./ADR-0027-ptx-nvcc-wrapper.md) | `ptx-nvcc` nvcc 兼容 wrapper 工具链 | Proposed | 2026-08-08 | `openspec/changes/feat-ptxir-nvcc-toolchain/` T13.3 |
| [0029](./ADR-0029-ptxemu-image-executor.md) | PTX-EMU Image Executor（in-memory Driver API + 2 反向依赖符号搬迁 + CudaDriver 保留理由） | Proposed | 2026-08-09 | TBD（待 propose 阶段确定） |

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
**最后更新**: 2026-08-09
**ADR 总数**: 27（当前有效 21：Active 14 / Accepted 7；Proposed 6；Superseded 0）
**预留编号**: ADR-0028 **[BLOCKING DEPENDENCY, 2026-08-09 升级]** — multi-kernel manifest + runtime selection 设计。详见 [docs/architecture/ptxir-toolchain-stack.md §11](../architecture/ptxir-toolchain-stack.md#11-related-adrs)。状态从 "预留占位" 升级为 BLOCKING，因 ADR-0025/0027/0029 的 v1 单 kernel 限制都依赖此 ADR 解除。引用文件数会随时间漂移，不在此处硬编码具体数字 per Lesson §8

## 最近更新

| 日期 | 更新内容 | 关联 ADR |
|------|---------|---------|
| 2026-08-09 | **Image Executor + 2 反向依赖搬迁**：新增 ADR-0029 Proposed（`ptxemu_image_*` C-API + `libptxemu_device.so` + `cpptlm_module.h`）— 填平 ptxir-toolchain-stack.md §11 TBD 缺口；image bytes 重 deserialize 修复 `src/cudart/ptx_interpreter.cpp:100-140` mutation bug；3-Phase 实施 + 5 byte-identical gates（含 logger→g_gpu_context 新 gate）+ D3 perf acceptance（< 10% deserialize cost 实测）；不 bump cpptlm_bridge.h ABI v2；Phase 0 Step 0 = amend ADR-0021 D-PTX-1 硬约束 | 0029 |
| 2026-08-09 | **ADR-0029 F2 跨仓评审修订 + ADR-0028 BLOCKING DEPENDENCY 升级 + 工具链栈 v1.3**：(a) ADR-0029 D8 替换为 HAL 扩展方案（UsrLinuxEmu HAL 65→68 fn-ptrs + 3 新 ioctl 编号 39/40/41 + TaskRunner `IGpuDriver` 3 方法），原 D8 直链方案保留为 D8-Alt；(b) ADR-0029 D6 标签 [SINGLE-LAUNCH] → [SINGLE-GPU-INSTANCE]；(c) §合规检查 新增 2 个 Acceptance gate（Phase 0 Step 0 HARD + D8 HAL SOFT）；(d) ADR-0028 从预留占位升 BLOCKING DEPENDENCY（影响 ADR-0025/0027/0029 三个 v1 单 kernel 限制）；(e) ADR-0027 §互斥关系 段新增 wrapper 与 in-memory 路径互斥约束；(f) `docs/architecture/ptxir-toolchain-stack.md` v1.2 → v1.3：§2 CP 端跨仓集成节点表 + §11 BLOCKING DEPENDENCY + §12 HAL extension future work | 0025, 0027, 0029, 工具链栈 |
| 2026-08-09 | **ADR-0029 F3 跨仓文档契约化**：ADR-0029 §D8.7 概要化 + 跨仓 ADR 引用分工；canonical source 落地到 UsrLinuxEmu [adr-076](../../../UsrLinuxEmu/docs/00_adr/adr-076-gpgpu-kernel-module-ioctl.md)（System C ioctl 0x27/0x28/0x29 + 结构体 + HAL fn-ptr #66/#67/#68 完整定义 + 跨仓 commit 顺序协议）；consumer-side 对偶到 TaskRunner [tadr-307](../../../UsrLinuxEmu/external/TaskRunner/docs/shared/adr/tadr-307-igpu-driver-kernel-module-extension.md)（IGpuDriver 扩展契约 + shim 改动 + MockGpuDriver 更新 + e2e 测试要求）；ioctl 编号 39/40/41 → **0x27/0x28/0x29**（System C magic 'G' 8-bit 范围修正，与现有 0x01~0x26 连续）| 0029 |
| 2026-08-08 | **PTXIR nvcc 兼容工具链**：新增 ADR-0025/0026/0027 Proposed — `ptxir_build` CLI 补齐 PTX→PTXIR、`PTXIR_MODE` default=auto 实现零配置、`ptx-nvcc` wrapper 提供端到端 NVIDIA SDK 兼容体验 | 0025, 0026, 0027 |
| 2026-08-08 | **新增架构文档** `docs/architecture/ptxir-toolchain-stack.md`：工具链栈总览（组件、build-time/runtime data flow、配置优先级、兼容性矩阵、v1 限制、v2 路线） | 0025, 0026, 0027 |
| 2026-08-07 | **ADR-0024 v1.1 amendment**: footer layout (size-after-section, ZIP-EOCD style) + magic literal `{'P','T','X','E','M','B','\x01','\x00'}` + PtxContextAdapter + tools/ 目录 + MANIFEST section | 0024 |
| 2026-08-06 | **CUBIN 嵌入 PTXIR 混合二进制格式**：新增 ADR-0024 Accepted — 作为 ADR-0023 sibling 决策，在 cubin 末尾追加 `.ptxir.section` + `.ptxir.magic`，loader 决策 + extract 工具复原纯 cubin + 复用现有 PTXIR 反序列化路径 | 0024 |
| 2026-07-30 | **PTXIR 二进制格式 + Pipeline 架构升级**：新增 ADR-0023 Accepted（PTXIR 7 项决策：扁平二进制+TOC+值枚举+字符串表末尾+Extend-Only 等）；ADR-0011 从 Proposed 升级 Accepted（引用 ADR-0023 作为格式依据） | 0023, 0011 |
| 2026-07-23 | **CppTLM 统一构建链路**：ADR-0022 Accepted — `--whole-archive` 替代独立 `.so` + `dlopen`；CppTLM Oracle 审查通过，P1/P2/P3/S1 修复已完成 | 0022 |
| 2026-07-16 | **cpptlm-d1-full 状态推进**：ADR-0021 Proposed → Active；ADR-0020 Proposed → Accepted；2 轮 Metis pre-impl review + 3 阶段 12 commits 修复所有 5 个 BLOCKER（B1 ABI 实现 / B2 sync loop / B3 stream destroy UB / B4 HSK 一致性 / B5 CMake 文档同步）+ sister spec 附录 + Postmortem 沉淀 | 0020, 0021 |
| 2026-07-15 | 添加 CppTLM D1-Full MemoryBridge 集成 ADR（D-PTX-1~6 决策 + HSK-1/2/3 握手 + cpptlm_bridge.h ABI 真值源） | 0021 |
| 2026-07-14 | 添加 CppTLM Phase 8.B D1-Full 注入点接受决策（3 个纯虚接口 + SMContext 3 setter + WarpContext 扩展 + RegisterAnalyzer 增强 + exe_once 三段式注入） | 0020 |
| 2026-07-14 | 添加 ThreadContext 持续瘦身 ADR（MemoryAccessor + InstructionPipeline accessor 方案） | 0019 |
| 2026-07-12 | 添加 tcgen05 cta_group::2 throw 语义 ADR（formalize scattered throw across 11 handlers） | 0018 |
| 2026-06-23 | 添加 CVT 策略模式重构 ADR (T2-6 完成) | 0015 |
| 2026-05-06 | 添加 pc_overridden_ 机制说明、while 循环收敛模式、Fallback 策略 | 0006, 0007, 0008 |
| 2026-05-06 | 补充 barrier 场景回归测试覆盖 | 0008 |
| 2026-05-06 | 添加 handle_branch PC 过滤说明、更新合规检查项 | 0006 |
