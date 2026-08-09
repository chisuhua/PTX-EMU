# 架构文档目录

本目录包含 PTX-EMU 的 SIMT、GPU 架构分析，以及 PTXIR 工具链与序列化设计文档。

## 📁 文档列表

| 文档 | 行数 | 状态 | 用途 |
|------|------|------|------|
| [SIMT-ARCHITECTURE-V2.md](./SIMT-ARCHITECTURE-V2.md) | 1134 | ✅ 最新 | SIMT v2.0 完整架构设计 |
| [sm90_100.md](./sm90_100.md) | ~300 | ✅ 维护中 | Hopper/Blackwell 分歧路径执行顺序详解 |
| [GPGPU-SIM-SIMT-ANALYSIS.md](./GPGPU-SIM-SIMT-ANALYSIS.md) | 717 | ✅ 参考 | GPGPU-Sim SIMT 实现详细分析 |
| [ptxir-toolchain-stack.md](./ptxir-toolchain-stack.md) | — | 📝 Proposed | PTXIR 工具链组件、构建与运行时数据流 |
| [ptxir-serialization-gaps-gap-analysis.md](./ptxir-serialization-gaps-gap-analysis.md) | — | ✅ 正式 | PTXIR 序列化目标与当前实现的差距分析 |
| CFG-DESIGN.md | — | ⏳ 待创建 | CFG 分析详细设计（文件尚不存在） |

## 🏗️ 架构版本

| 版本 | 状态 | 位置 |
|------|------|------|
| v2.0 | ✅ 当前版本 | 本目录 |
| v1.0 | 🗄️ 历史版本 | [`../archive/`](../archive/) |

## 📖 阅读顺序

1. **入门**: [SIMT-ARCHITECTURE-V2.md](./SIMT-ARCHITECTURE-V2.md) 执行摘要
2. **执行专题**: [sm90_100.md](./sm90_100.md) 与 [GPGPU-SIM-SIMT-ANALYSIS.md](./GPGPU-SIM-SIMT-ANALYSIS.md)
3. **PTXIR 背景**: [ptxir-serialization-gaps-gap-analysis.md](./ptxir-serialization-gaps-gap-analysis.md)
4. **PTXIR 工具链**: [ptxir-toolchain-stack.md](./ptxir-toolchain-stack.md)
5. **参考**: CFG-DESIGN.md（文件尚不存在，待创建）

---

**维护**: 架构文档保留当前设计、分析与差距文档，历史版本归档到 `../archive/`
**最后更新**: 2026-08-08

## 📊 GPGPU-Sim Analysis

### 分析内容

- **SIMT 执行模型**: Warp-level execution with unified PC
- **SIMT Stack**: Hardware reconvergence stack implementation
- **Barrier**: __syncthreads() and CTA-level synchronization
- **PC Management**: Per-thread PC tracking during divergence
- **Scheduler**: Multiple scheduling policies (LRR, GTO, etc.)

**Source**: GPGPU-Sim v4.x dev branch  
**Date**: 2026-04-10
