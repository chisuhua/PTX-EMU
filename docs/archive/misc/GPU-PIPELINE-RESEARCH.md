# GPU Pipeline Architecture Research

> **研究日期**: 2026-05-04
> **来源**: NVIDIA PTX ISA 文档、AMD RDNA 白皮书、CUDA 编程指南
> **研究员**: librarian subagent

---

## 1. NVIDIA GPU 指令流水线 (Ampere+)

### 1.1 流水线阶段

| 阶段 | 描述 | 来源 |
|------|------|------|
| **Fetch** | 调度器选择 warp，从 L1 i-cache 取指到 Instruction Buffer | [Analyzing Modern GPU Cores](https://semiiphub.com/article/analyzing-modern-nvidia-gpu-cores) |
| **Decode** | 指令解码，存入 per-warp Instruction Buffer (通常 3 条) | 同上 |
| **Issue** | GTO (Greedy Then Oldest) 调度器选择 warp，发出指令 | 同上 |
| **Control** | 增加依赖计数器 | 同上 |
| **Allocate** | 定点指令检查寄存器文件端口可用性 | 同上 |
| **Execute** | 发射到功能单元 (FP32, INT32, FP64 等) | 同上 |

### 1.2 PC 更新机制

**关键发现**: PC 在指令执行**完成后**更新

> "In GPUs of compute capability 7.0 and later, independent thread scheduling allows full concurrency between threads, including a program counter and call stack per thread."

— [CUDA Programming Guide - Independent Thread Scheduling](https://docs.nvidia.com/cuda/cuda-programming-guide/)

**架构差异**:

| 架构 | PC 粒度 | 说明 |
|------|--------|------|
| Pre-Volta | Per-warp PC | 32 线程共享单一 PC |
| Volta+ | Per-thread PC | 每线程独立 PC，支持独立线程调度 |

### 1.3 Barrier 指令行为

**PTX ISA 描述** (`bar.warp.sync`):

> "bar.warp.sync will cause executing thread to wait until all threads corresponding to membermask have executed a bar.warp.sync with the same membermask value before resuming execution."

— [PTX ISA 9.2 - Section 9.7.13.2](https://docs.nvidia.com/cuda/parallel-thread-execution/)

**Barrier 导致 Warp Stall**:

> "A warp is considered a candidate for issuing its oldest instruction in a given cycle only if... it is not waiting on a barrier"

— [PTXAS Synchronization & Barriers Reference](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)

**关键结论**:
- `BAR.SYNC` 导致 warp stall 直到所有线程到达
- 调度器切换到其他已就绪 warp
- 不需要手动引入 stall cycles

### 1.4 调度器取指基于 PC

> "The fetch scheduler tries to fetch an instruction from the same warp that has been issued in the previous cycle... unless it detects that the number of instructions already in the Instruction Buffer plus its in-flight fetches are equal to the Instruction Buffer size."

— [Analyzing Modern GPU Cores](https://semiiphub.com/article/analyzing-modern-nvidia-gpu-cores)

**Fetch 基于**:
- Warp 的 next PC（程序计数器）
- 不是 active mask

**Issue 基于**:
- Readiness（就绪状态）
  - 不等待 barrier
  - 无 RAW hazard
  - stall counter = 0

### 1.5 相关文档

| 文档 | 章节 | 关键内容 |
|------|------|----------|
| [PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html) | 9.7.13 | Barrier 指令语义 |
| [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/) | 3.2 | SIMT Execution Model |
| [PTXAS Sync Barriers](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html) | - | Barrier 流水线行为 |
| [Analyzing Modern GPU Cores](https://semiiphub.com/article/analyzing-modern-nvidia-gpu-cores) | - | Fetch/Issue 流水线 |

---

## 2. AMD GPU 指令流水线 (RDNA)

### 2.1 流水线阶段

**Source**: [NaviSim RDNA Pipeline Model](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf) (Figure 4)

| 阶段 | 描述 | 来源 |
|------|------|------|
| **Fetch** | 指令取回 arbiter 选择 wavefront（优先 oldest） | NaviSim |
| **Issue** | Issue arbiter 选择 wavefront（指令 buffer 中有就绪指令） | 同上 |
| **Decode** | 指令解码（在 Issue 之后） | 同上 |
| **Read** | 寄存器文件读 | 同上 |
| **Execute** | SIMD/SALU/LDS 执行 | 同上 |
| **Write** | 结果写回 | 同上 |

**RDNA vs GCN 差异**:

> "GCN: 4-cycle instruction issue — SIMD16 issues 1 instruction every 4 cycles"
> "RDNA: Single-cycle instruction issue — SIMD32 issues 1 instruction every cycle"

### 2.2 Wavefront 调度器

**Source**: [ROCm Compute Profiler - Pipeline Descriptions](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/cdna/pipeline-descriptions.html)

**调度行为（每时钟周期）**:
1. 以 round-robin 方式选择 SIMD 单元
2. 从该 SIMD 的池中选择 wavefront
3. 每个 wavefront 最多发出 1 条指令
4. 每类指令最多发出 1 条（VALU, VMEM, SALU/SMEM, LDS, Branch）

**最大 IPC**: 5 instructions/cycle/CU

### 2.3 Barrier 指令行为

**`s_barrier` (wave-level barrier)**:

> "Wave32 and single-cycle issue for better latency... Dependency stalls can be filled by other waves"

— [RDNA Architecture Whitepaper](https://gpuopen.com/download/RDNA_Architecture_public.pdf)

**Barrier stall 行为**:
- `s_barrier` 导致 wave stall 直到所有 wave 在 workgroup 到达
- 其他 wavefront 继续执行（延迟隐藏）

### 2.4 相关文档

| 文档 | URL | 关键内容 |
|------|-----|----------|
| [RDNA Whitepaper](https://gpuopen.com/download/RDNA_Architecture_public.pdf) | - | Fetch-decode, WGP, instruction issue |
| [NaviSim Paper](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf) | Figure 4 | DCU pipeline model |
| [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf) | Ch.4 | Program Flow Control |
| [HIP Hardware](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html) | - | CU architecture, instruction fetch |

---

## 3. 关键结论（模拟器设计）

### 3.1 两者共同的行为

| 行为 | NVIDIA | AMD |
|------|--------|-----|
| PC 更新时机 | 指令完成后 PC + 1 | 指令完成后 PC + 1 |
| PC 粒度 | Per-thread (Volta+) / Per-warp | Per-wavefront |
| Barrier stall | ✅ Warp stall | ✅ Wave stall |
| 调度器行为 | 切换到其他已就绪 warp | 切换到其他已就绪 wavefront |
| Fetch 基于 | PC 值 | PC 值 |

### 3.2 PC 更新的硬件含义

```
Fetch → Decode → Issue → [Execute] → [PC ← PC + 1] → Fetch(下一条)
                       ↑
                指令完成后才更新 PC
                更新 PC 后才能取下一条
```

### 3.3 Barrier 的硬件含义

```
bar.sync 指令:
  Thread A: stall (等待其他线程到达)
  Thread B: stall (等待其他线程到达)
  Thread C: stall (等待其他线程到达)
  
  SM 调度器: 选择其他已就绪的 warp/wavefront 执行
  → 不需要"stall cycles"，调度器自然不选 stalled warp
```

### 3.4 模拟器设计建议

**不引入 stall cycles**，而是利用自然的 PC 机制：

```
调度器 Ready 判断:
  warp_is_ready(warp):
    if warp_state.threads[lane].pc == warp_state.threads[lane].next_pc:
      return true   // commit 已完成，可取下一条
    else:
      return false  // commit 未完成，自然 stall
```

**这样：**
- 正常指令：`commit_pc()` 执行 → `pc == next_pc` → 就绪
- Barrier 未完成：state=BLOCKED → 调度器跳过
- Barrier 完成：`set_pc(reconvergence_pc)` → `pc == next_pc` → 就绪

---

## 4. 参考链接汇总

### NVIDIA
- [PTX ISA 9.2](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)
- [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/)
- [PTXAS Sync Barriers](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)
- [Analyzing Modern GPU Cores](https://semiiphub.com/article/analyzing-modern-nvidia-gpu-cores)
- [GPU Execution Model - CICC](https://gh.evko.io/nvopen-tools/cicc/gpu-execution-model.html)

### AMD
- [RDNA Architecture Whitepaper](https://gpuopen.com/download/RDNA_Architecture_public.pdf)
- [NaviSim Paper](https://bu-icsg.github.io/publications/2022/navisim_pact_2022.pdf)
- [CDNA4 ISA](https://www.amd.com/content/dam/amd/en/documents/instinct-tech-docs/instruction-set-architectures/amd-instinct-cdna4-instruction-set-architecture.pdf)
- [HIP Hardware](https://rocm.docs.amd.com/projects/HIP/en/latest/understand/hardware_implementation.html)
- [ROCm Pipeline Docs](https://rocm.docs.amd.com/projects/rocprofiler-compute/en/develop/conceptual/cdna/pipeline-descriptions.html)
