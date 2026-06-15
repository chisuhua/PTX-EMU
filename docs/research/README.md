# PTX-EMU Research

> 调研性文档（Research investigations）  
> 与 ADR、Developer Guide 互补：ADR 记录"做什么决策"，Developer Guide 记录"如何开发"，本目录记录"为什么这样做（基于外部证据）"。

## 📚 调研主题

### [Barrier 语义（NVIDIA Hopper/Blackwell & divergent warp）](./barrier-semantics/)

调研 NVIDIA 真实硬件语义、4 个开源模拟器实现、以及本项目当前实现的对比分析。

| # | 文档 | 主题 | 来源 |
|---|------|------|------|
| 01 | [PTX ISA 官方语义](./barrier-semantics/01-ptx-isa-official-semantics.md) | `bar.sync` / `bar.warp.sync` / named barrier / cluster barrier 的形式语义 | PTX ISA 9.3, Volta Tuning Guide, LLVM NVPTX |
| 02 | [发散 warp 硬件行为](./barrier-semantics/02-divergent-warp-hardware-behavior.md) | divergent warp 命中 barrier 的真实硬件语义 | MLIR NVVM, LLVM NVPTX PR #140615, NVIDIA dev forum |
| 03 | [Hopper/Blackwell 新特性](./barrier-semantics/03-hopper-blackwell-new-features.md) | sm_90+ 引入的 cluster barrier、async barrier、mbarrier、tcgen05 集成 | H100 Whitepaper, Hopper In-Depth, ptxas reverse engineering |
| 04 | [PTX-EMU 当前实现](./barrier-semantics/04-ptx-emu-current-implementation.md) | 本项目 barrier 系统的代码地图 | 项目源码 + git history |
| 05 | [开源模拟器对比](./barrier-semantics/05-open-source-simulator-comparison.md) | gpgpu-sim / gem5 / MIAOW / Multi2Sim 的 barrier 实现 | accel-sim, gem5, VerticalResearchGroup, multi2sim |
| 06 | [综合分析与建议](./barrier-semantics/06-synthesis-and-recommendations.md) | 综合 5 份调研的结论 + 失败测试的根因 + 推荐决策 | 全部子代理输出综合 |

## 🎯 调研目的

为以下决策提供外部证据：
- 决定 `integration_warp_barrier` 失败测试的修复方向
- 评估本项目 barrier 实现是否符合 NVIDIA 硬件语义
- 规划 Hopper/Blackwell 完整 barrier 支持的实施路线图

## 📅 调研时间

2026-06-15

## 🔗 相关文档

- [ADR-0008: Barrier 语义增强](../adr/0008-barrier-semantics.md) — 本项目的 barrier 架构决策
- [BARRIER-PROGRAMMING-REFERENCE.md](../developer-guide/BARRIER-PROGRAMMING-REFERENCE.md) — 项目内 barrier 编程参考
- [sm90_100.md](../architecture/sm90_100.md) — Hopper/Blackwell 架构文档
- [GPGPU-SIM-SIMT-ANALYSIS.md](../architecture/GPGPU-SIM-SIMT-ANALYSIS.md) — 与 gpgpu-sim 的 SIMT 对比
