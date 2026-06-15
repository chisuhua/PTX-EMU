# Barrier 语义 — 综合索引

> **调研主题**：NVIDIA Hopper/Blackwell 真实硬件 barrier 语义 + divergent warp 行为  
> **调研日期**：2026-06-15  
> **触发原因**：`integration_warp_barrier` 测试失败根因分析

## 🎯 一句话总结

**NVIDIA 硬件 + 4 个开源模拟器 + 本项目实现，三者完全一致：barrier 是 warp-级（32 lanes）到达计数 + 整 warp 释放，不存在 per-lane participation 计数。失败的 4 个测试用例是**测试期望值陈旧**导致，与 `step_warp` 转换（commit `ca2140f`）后实际 32-lane 驱动行为不匹配。**

## 📂 文档清单

### 上游 — NVIDIA 官方语义
1. [01-ptx-isa-official-semantics.md](./01-ptx-isa-official-semantics.md) — `bar.sync` / `bar.warp.sync` 形式语义
2. [02-divergent-warp-hardware-behavior.md](./02-divergent-warp-hardware-behavior.md) — divergent warp 行为
3. [03-hopper-blackwell-new-features.md](./03-hopper-blackwell-new-features.md) — Hopper/Blackwell 新 barrier 特性

### 下游 — 实现对比
4. [04-ptx-emu-current-implementation.md](./04-ptx-emu-current-implementation.md) — 本项目实现
5. [05-open-source-simulator-comparison.md](./05-open-source-simulator-comparison.md) — gpgpu-sim / gem5 / MIAOW / Multi2Sim 实现

### 综合
6. [06-synthesis-and-recommendations.md](./06-synthesis-and-recommendations.md) — 综合分析 + 失败测试根因 + 推荐决策

## 📊 关键发现速查

| 问题 | NVIDIA 硬件语义 | 本项目实现 | 评估 |
|------|----------------|-----------|------|
| barrier 到达计数单位 | **warp-级**（整 32 lanes 全部 arrive） | warp-级（`arrive(lane_id)` for all 32 active lanes） | ✅ 一致 |
| 释放时 PC 推进 | **整 32 lanes 全部推进到同一 reconvergence PC** | 整 32 lanes 推进（`advance_thread_pc(i, reconv_pc)` for all） | ✅ 一致 |
| divergent 两半处理 | **不释放**直到两半都到达 | **不释放**（BUG-POSTBARRIER-TWOHALVES 修复后 OR active_mask） | ✅ 一致 |
| `bar.warp.sync` membermask | **UB if 执行 lane 不在 mask 内**；mask 是参与声明不是计数限制 | 当前实现用 static_mask 当 participation | ⚠️ 部分合规（未做 UB 检查） |
| 16 个 named barrier 槽 | ✅ SM 30+ 一直支持 | ❌ 当前只用 `wbars[0]` | ❌ 未实现 |
| Cluster barrier (sm_90+) | `barrier.cluster.arrive/wait` + DSMEM | ❌ 未实现 | ❌ 未实现 |
| mbarrier (sm_70+) | 64-bit shared mem 对象 + phase parity | ❌ 未实现 | ❌ 未实现 |
| membar/fence (Volta+) | `bar.sync` 不再隐式 membar.cta，需显式 fence | ⚠️ 部分支持 | ⚠️ 待完善 |

## 🚀 推荐决策（详见文档 06）

1. **保留 BUG-POSTBARRIER-TWOHALVES 修复**（`set_active_mask(get_active_mask() | arrived_mask)`）—— 符合"整 warp release"语义
2. **保留 BUG-RECONVERGENCE-SIMPLEGEMM 修复**（wbar 已初始化时保留 arrived_mask）—— 符合"到达记录不可丢失"语义
3. **更新失败测试的期望值**——`count_arrived() == 4` 改为 `== 32` 等
4. **不修改 barrier.cpp**——其实现与 NVIDIA 硬件和 4 个开源模拟器完全一致

## 🔗 引用来源

主要权威来源：
- [PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Volta Tuning Guide](https://docs.nvidia.com/cuda/volta-tuning-guide/)
- [Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)
- [H100 Whitepaper (GTC22)](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)
- [CUDA Programming Guide §4.9](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)
- [MLIR NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)
- [PTXAS Reverse Engineering Reference](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)
- gpgpu-sim commit `6c3cf4ff32110908386d605a7034fc67666a92de`
- gem5-gpu commit `c8222cc67a399bfc01e8658dd14b30d5bfd634f9`
