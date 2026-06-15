# 03. Hopper/Blackwell 新 barrier 特性

> **子代理调研任务**：`bg_897b4b86` — Research Hopper sm_90 cluster barriers  
> **调研日期**：2026-06-15  
> **主题**：sm_90+ 引入的 cluster barrier、async barrier、mbarrier、tcgen05 集成  
> **来源**：H100 Whitepaper, Hopper Architecture In-Depth, ptxas reverse engineering, CUTLASS

---

## 📌 主要参考文献

| 来源 | 链接 | 类型 |
|------|------|------|
| **NVIDIA Hopper Architecture In-Depth** (GTC 2022) | https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/ | 官方博客 |
| **NVIDIA H100 Whitepaper (GTC22)** | https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf | 官方白皮书 |
| **PTX ISA 9.3** | https://docs.nvidia.com/cuda/parallel-thread-execution | 官方 ISA |
| **PTX ISA 8.8 PDF** | https://docs.nvidia.com/cuda/pdf/ptx_isa_8.8.pdf | 官方 PDF |
| **CUDA Programming Guide §4.9** | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html | 官方指南 |
| **Hopper Tuning Guide §1.4.1.3** | https://docs.nvidia.com/cuda/hopper-tuning-guide/index.html | 官方调优 |
| **HotChips 34 NVIDIA Hopper** | http://hc34.hotchips.org/assets/program/conference/day1/GPU%20HPC/HC2022.NVIDIA.Choquette.vfinal01.pdf | 官方演讲 |
| **Blackwell Architecture (CUTLASS)** | https://nvidia-cutlass-22.mintlify.app/architectures/blackwell | CUTLASS 文档 |
| **PTXAS Reverse Engineering** | https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html | 逆向工程 |
| **CUTLASS barrier.h** | https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h | 生产代码 |

---

## 📋 核心摘要

| 问题 | 答案 |
|------|------|
| **Q1**: 集群级屏障 `bar.sync.cluster` | 通过 **命名集群屏障** (`barrier.cluster.arrive/wait`) + **集群作用域 mbarrier**（`.shared::cluster`）实现；依赖 **DSMEM** + GPC 内专用 SM-to-SM 网络 |
| **Q2**: 异步屏障 | Ampere 引入 split arrive/wait；Hopper 升级 — **Waiter 睡眠而非轮询**；**新增 Asynchronous Transaction Barrier**（arrival count + tx_count 同时跟踪） |
| **Q3**: 命名屏障 `bar.sync N` 数量 | **16 个**（indices 0-15），**不是 6 个**；CUTLASS `HardwareMaxNumNamedBarriers = 16` 可证 |
| **Q4**: `bar.warp.sync` | **早在 PTX 6.0 / SM 30 (2016 Pascal) 就有**，**不是 Blackwell 特有**；映射到 SASS `WARPSYNC` opcode |
| **Q5**: 完成检测硬件机制 | mbarrier = **64-bit shared memory 对象**：{participant_count (20), pending_count (20), tx_count (20), phase (1)} |
| **Q6**: 与 `membar.*` 关系 | `bar.sync` **隐式提供 release-acquire at .cta scope**；Volta+ 上**不再隐式 membar.cta**，需显式 `membar.*` 或 `fence.*` |

---

## Q1: 集群级屏障 — 通过 DSMEM 实现

### 1.1 Thread Block Cluster 定义

> **"H100 grows the CUDA thread group hierarchy with a new level called the thread block cluster. A cluster is a group of thread blocks that are guaranteed to be concurrently scheduled onto a group of SMs... The clusters in H100 run concurrently across SMs within a GPC."**  
> — [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

- Cluster 大小：H100 上**最多 16 CTA**（非可移植）/ **8 CTA**（可移植）
- 所有 CTA 保证在同一 **GPC（GPU Processing Cluster）** 内调度
- SM 物理接近，SM-to-SM 网络延迟短

### 1.2 DSMEM（分布式共享内存） — 集群屏障的物理基础

> **"With clusters, it is possible for all the threads to directly access other SM's shared memory with load, store, and atomic operations. This feature is called distributed shared memory (DSMEM)... Compared to using global memory, DSMEM accelerates data exchange between thread blocks by about 7x."**  
> — [NVIDIA H100 Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)

DSMEM 寻址使用 generic pointer；H100 上每 CTA 最多 **227 KB**；访问应 coalesced 且 32-byte 对齐。

### 1.3 两种集群屏障形式

**形式 A — 命名集群屏障**（用于 bootstrap）：

> **"`barrier.cluster.arrive` / `barrier.cluster.wait` (PTX ISA 9.2, §9.7.13.3 'barrier.cluster') — a named barrier at cluster scope. Used for bootstrapping (before any cluster-wide mbarrier is valid, all CTAs need a synchronization point)."**  
> — [Hopper Tensor Core Programming Guide](https://hackmd.io/jB0xtWKgQzOfinOgUNVhbw)

**形式 B — 集群作用域 mbarrier**（用于生产-消费者 pipeline）：

> **"Cluster-scoped variants (PTX ISA 9.2, §9.7.13.15.8): all of the above instructions have `.shared::cluster` counterparts for mbarriers placed in the cluster-scoped shared memory proxy (DSMEM)."**

**PTX 示例**：
```ptx
mbarrier.init.shared::cluster.b64 [mbar_addr], count;
mbarrier.arrive.shared::cluster.b64 _, [mbar_addr];
barrier.cluster.arrive;   // bootstrap：让 mbarrier 初始化在所有 CTA 可见
barrier.cluster.wait;
```

### 1.4 集群拓扑保证

> **"The executing CTA has to make sure that the shared memory of the peer CTA exists before communicating with it via shared memory and the peer CTA hasn't exited before completing the shared memory operation."**  
> — [PTX ISA 9.3 §1](https://docs.nvidia.com/cuda/parallel-thread-execution)

每个 cluster CTA 通过 `%cluster_ctarank` 等特殊寄存器映射 DSMEM 中的正确位置。

---

## Q2: 异步屏障 — Ampere 引入，Hopper 增强，Blackwell 扩展

### 2.1 Ampere: Split Arrive/Wait 起源

> **"Asynchronous barriers split the synchronization process into two steps. First, threads signal 'Arrive' when they are done producing their portion of the shared data. This 'Arrive' is non-blocking so the threads are free to execute other independent work. Eventually the threads need the data produced by all the other threads. At this point they do a 'Wait', which blocks them until every thread has signaled 'Arrive'."**  
> — [NVIDIA Hopper Architecture In-Depth](https://developer.nvidia.com/blog/nvidia-hopper-architecture-in-depth/)

### 2.2 Hopper 关键升级：Waiter 睡眠

> **"New for Hopper is the ability for 'Waiting' threads to sleep until all other threads arrive. On previous chips, Waiting threads would spin on the barrier object in shared memory."**  
> — [NVIDIA H100 Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)

含义：Ampere 上 wait 主动轮询（消耗 SM 调度槽位）；Hopper 上硬件可真正释放线程。

### 2.3 Asynchronous Transaction Barrier — Hopper 新增

> **"While asynchronous barriers are still part of the NVIDIA Hopper programming model, it adds a new form of barrier called an asynchronous transaction barrier... It too is a split barrier, but instead of counting just thread arrivals, it also counts transactions. NVIDIA Hopper includes a new command for writing shared memory that passes both the data to be written and a transaction count. The transaction count is essentially a byte count. The asynchronous transaction barrier blocks threads at the Wait command until all the producer threads have performed an Arrive, and the sum of all the transaction counts reaches an expected value."**  
> — [NVIDIA H100 Whitepaper](https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf)

**关键用途**：与 **TMA（Tensor Memory Accelerator）** 和 **cp.async.bulk** 集成。

### 2.4 mbarrier 完整状态机（64-bit 对象）

> **"Each mbarrier slot carries four fields packed into a 64-bit word":**

| 字段 | 位数 | 作用 |
|------|------|------|
| `participant_count` | low **20** | 完成一个 phase 所需的总 arrival 数 |
| `pending_count` | mid **20** | phase 完成前剩余的 arrival 数 |
| `tx_count` | next **20** | 仍未到达的字节数（TMA expect-tx 变体） |
| `phase` | high **1** | 每次 phase 完成时翻转 |

来源：[Tileiras NVVM Internals](https://gh.evko.io/nvopen-tools/tileiras/dialects/nvvm/mbarrier-ops.html)

**状态转移**：

| Op | 转移 |
|----|------|
| `init` | `participant_count := N`, `pending_count := N`, `tx_count := 0`, `phase := 0` |
| `arrive` | `pending_count -= 1`; 若为 0：翻转 `phase`, `pending_count := participant_count` |
| `arrive.expect_tx` | `arrive` + `tx_count += k`（TMA 生产端） |
| `try_wait.parity` | 非阻塞：返回 `true` 若 `phase == expected_phase` |
| `test_wait` | 阻塞：自旋直到 `phase` 与 token 匹配 |

### 2.5 完整 mbarrier 指令集

来源：[PTX ISA 9.3 §9.7.14.16](https://docs.nvidia.com/cuda/parallel-thread-execution) + [PTXAS Reverse Engineering](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)：

| PTX 指令 | 用途 |
|---------|------|
| `mbarrier.init` | 初始化 shared memory 中的屏障对象 |
| `mbarrier.arrive` | 非阻塞到达信号 |
| `mbarrier.arrive_drop` | 到达并永久减少 expected count |
| `mbarrier.arrive.expect_tx` | 到达并设置预期事务字节数 |
| `mbarrier.test_wait` | 测试 phase 是否完成 |
| `mbarrier.try_wait` | 带超时等待 |
| `mbarrier.try_wait.parity` | 基于 phase parity 的等待 |
| `mbarrier.pending_count` | 查询剩余到达数 |
| `mbarrier.complete_tx` | 标记事务字节完成（由 TMA 硬件触发） |
| `mbarrier.inval` | 使屏障失效 |

**SM 适用性**：
- Pre-sm90: 无 mbarrier 伪操作（phase 是 no-op）
- **sm_90 (Hopper)**: 引入，扩展为硬件 mbarrier 指令序列
- **sm_100+ (Blackwell)**: 扩展语义，支持 `tcgen05.fence`、集群级屏障、async pipeline

---

## Q3: 命名屏障 — 16 个，不是 6 个

### 关键校正

> **"The hardware provides 16 named barriers (indices 0–15), each tracking participation counts."**  
> — [PTXAS Synchronization Reference](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)

> **"`static const uint32_t HardwareMaxNumNamedBarriers = 16;`"**  
> — [NVIDIA CUTLASS `include/cutlass/arch/barrier.h:2810`](https://github.com/NVIDIA/cutlass/blob/main/include/cutlass/arch/barrier.h)

硬件从 SM 30 (Kepler) 起就提供 **16 个命名屏障**（索引 0-15）。Ampere、Hopper、Blackwell 都保持这个数量。

### 命名屏障指令族

| PTX 指令 | SASS Opcode | 行为 |
|---------|-------------|------|
| `bar.sync N` | `BAR.SYNC` | 阻塞直到所有 CTA 线程到达屏障 N |
| `bar.sync N, count` | `BAR.SYNC` | 阻塞直到 `count` 个线程到达 |
| `bar.arrive N` | `BAR.ARV` | 非阻塞到达 |
| `bar.red.and N, p` | `BAR.RED.AND` | 屏障 + warp 级 AND reduction |
| `bar.red.or N, p` | `BAR.RED.OR` | 屏障 + warp 级 OR reduction |
| `bar.red.popc N, d` | `BAR.RED.POPC` | 屏障 + warp 级 population count |
| `barrier.cta.sync N` | `BAR.SYNC` | PTX 8.0 集群感知 CTA 屏障 |
| `barrier.cta.arrive N` | `BAR.ARV` | PTX 8.0 集群感知 CTA 到达 |
| `barrier.cluster.arrive/wait` | (扩展) | **sm_90+** 集群作用域命名屏障 |

来源：[PTXAS Sync & Warp Intrinsics](https://gh.evko.io/nvopen-tools/ptxas/intrinsics/sync-warp.html)

---

## Q4: `bar.warp.sync` — 不是 Blackwell 特有

### 4.1 历史

> **"A new instruction bar.warp.sync which allows synchronizing threads in warp."**  
> — [PTX ISA Changes (2016)](https://docs.nvidia.com/cuda/archive/8.0/parallel-thread-execution/)

实际上 `bar.warp.sync` **自 PTX 6.0 / SM 30 (Pascal) 就已存在**，**Hopper 和 Blackwell 完全沿用**，没有新增变化。

### 4.2 形式定义

```ptx
bar.warp.sync membermask;
```

**关键约束**：

> **"The behavior is undefined if the executing thread is not included in the mask... For compute capability sm_6x or below, all threads in the mask must execute the same `bar.warp.sync` instruction in convergence."**  
> — [LLVM NVVM Dialect Documentation](https://enzymead.github.io/Reactant.jl/stable/api/dialects/nvvm)

### 4.3 SASS 映射

> **`bar.warp.sync membermask` | `WARPSYNC mask` | Synchronize warp lanes specified by mask**  
> — [PTXAS Sync & Warp Intrinsics](https://gh.evko.io/nvopen-tools/ptxas/intrinsics/sync-warp.html)

### 4.4 内存顺序保证

> **"This operation also guarantees memory ordering among participating threads. Threads within the warp that wish to communicate via memory can store to memory, execute `bar.warp.sync`, and then safely read values stored by other threads in the warp."**

---

## Q5: 屏障完成检测的硬件机制

### 5.1 命名屏障 — 寄存器文件式

> **"When two DEPBAR instructions produce the same signature... allocates two fresh barrier registers (register file 6), emits a new opcode-130 for each original use in the chain."**  
> — [PTXAS Reverse Engineering](https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html)

含义：命名屏障的状态（arrival count、completion flag）保存在 **register file 6**（屏障寄存器文件），每 CTA 16 个槽位。

### 5.2 mbarrier — 64-bit shared memory 对象

```
64-bit word layout (LSB → MSB):
┌──────────────────┬──────────────────┬──────────────────┬───┐
│ participant_count│ pending_count    │ tx_count         │ ph│
│     20 bits      │     20 bits      │     20 bits      │ 1 │
└──────────────────┴──────────────────┴──────────────────┴───┘
```

**硬件检测机制**：
1. **arrive 操作**：`mbarrier.arrive` 通过原子操作更新 `pending_count`
2. **phase 完成触发**：当 `pending_count == 0`，硬件自动翻转 `phase bit`，重置 `pending_count := participant_count`
3. **tx_count 跟踪**：每次 `mbarrier.arrive.expect_tx` 时增加；每次 `mbarrier.complete_tx`（TMA/cp.async.bulk 完成时触发）时减少
4. **wait 操作**：`mbarrier.try_wait.parity` 读取 `phase bit` 与期望 parity 比较

### 5.3 收敛屏障（Volta+）

| SASS Opcode | 用途 |
|-------------|------|
| `BSSY B, target` | 推送同步屏障；`target` 是 reconvergence 点 |
| `BSYNC B` | 弹出并等待在收敛屏障 B |

> **"The BSSY/BSYNC instruction pair replaces the pre-Volta implicit reconvergence stack. The compiler must insert these pairs explicitly at divergence/reconvergence points."**

**关键事实**：Hopper 和 Blackwell 都使用 **显式 BSSY/BSYNC**，**没有硬件隐式 reconvergence stack**。

---

## Q6: 与 `membar.*` 的内存顺序关系

### 6.1 屏障同步 = 隐式 release-acquire

> **"Barrier synchronization has the same effect as release-acquire synchronization performed at .cta scope."**  
> — [A Formal Analysis of the NVIDIA PTX Memory Consistency Model, ASPLOS 2019](https://d1qx31qr3h6wln.cloudfront.net/publications/ASPLOS_2019_PTXMemoryModel.pdf)

> **"A barrier makes a set of threads wait until all of them (or a specified count) have arrived. A barrier typically includes a memory-ordering effect (release on the arrive side, acquire on the wait side), so after the barrier, all participants see all writes issued before the arrives."**  
> — [Hopper Tensor Core Programming Guide](https://hackmd.io/jB0xtWKgQzOfinOgUNVhbw)

### 6.2 Volta+ 上的重要隐患

> **"Pre-Volta, `bar.sync` doubled as an implicit `membar.cta` because all threads in the warp executed lockstep. With Independent Thread Scheduling on Volta+, `bar.sync` only guarantees control-flow convergence; loads issued after the barrier can still observe stale stores from before it unless an explicit `membar.cta` is also emitted."**  
> — [CICC Builtins Reference](https://gh.evko.io/nvopen-tools/cicc/builtins/barriers.html)

含义：Volta 之后，`bar.sync` **不再隐式提供完整的内存排序**。要保证 load 看到 store，必须额外加 `membar.cta`。

### 6.3 membar 指令族

| PTX 指令 | SASS Opcode | 作用域 |
|---------|-------------|--------|
| `membar.cta` | `MEMBAR.CTA` | Thread block (CTA) |
| `membar.gl` | `MEMBAR.GL` | Device (GPU) |
| `membar.sys` | `MEMBAR.SYS` | System (所有 agents) |

### 6.4 fence 指令族 (PTX 8.0+)

| PTX 指令 | SASS Opcode (sm100+) | 用途 |
|---------|---------------------|------|
| `fence.proxy.alias` | (inline) | generic/alias 内存访问排序 |
| `fence.proxy.async` | (inline) | async copy 完成可见性排序 |
| `fence.proxy.async.global` | (inline) | global 内存 async fence |
| `fence.sc.cta` | `FENCE_S` | SC fence, CTA 作用域 |
| `fence.sc.gpu` | `FENCE_G` | SC fence, GPU 作用域 |
| `fence.acq_rel.cta` | `FENCE_T` | acquire-release fence, CTA 作用域 |

来源：[PTXAS Sync & Warp Intrinsics](https://gh.evko.io/nvopen-tools/ptxas/intrinsics/sync-warp.html)

### 6.5 内存顺序语义 (PTX 8.0+)

| 语义 | PTX 修饰符 | 含义 |
|------|----------|------|
| `relaxed` | `.relaxed` | 仅原子性；无跨线程排序 |
| `acquire` | `.acquire` | 后续 load/store 看到 release 前的写 |
| `release` | `.release` | 之前的 load/store 在 acquire 端可见 |
| `acq_rel` | `.acq_rel` | 仅在 RMW 上合法 |
| `sc` / `seq_cst` | `.sc` | 全序 |

**作用域**：`cta` / `cluster` / `gpu` / `sys`

---

## 🆕 Blackwell (sm_100, sm_120) 特有变化

### B.1 新增 PTX 指令

> **"Beyond tcgen05, Blackwell introduces or extends several instruction families visible in the opcode dispatch: `tcgen05.*` (11 instructions), `fence_view_async`, `write_async`, `viaddmax`/`viaddmin`, BGMMA / QMMA..."**

### B.2 tcgen05 与 mbarrier 集成

> **"The commit modifier emission at `sub_35F4E30` combines tensor core commit with mbarrier synchronization:**
> - `.cta_group::1`/`.cta_group::2` — Group selection
> - `.mbarrier::arrive::one` — Mbarrier arrive modifier
> - `.shared::cluster` — Shared memory cluster scope
> - `.multicast::cluster` — Multicast cluster scope"

**PTX 示例**（Blackwell GEMM）：
```ptx
tcgen05.commit(bar_mma, space="cluster");
```

### B.3 Blackwell 屏障语义的扩展

> **"sm100+ (Blackwell): Extended mbarrier semantics for `tcgen05.fence`, cluster-level barriers, and async pipeline operations."**

### B.4 SM100 与 SM120 差异

| 特性 | SM100/103 (Datacenter) | SM120 (GeForce) |
|------|------------------------|-----------------|
| Cluster Size | 最多 **16 CTAs** | 最多 **8 CTAs** |
| Shared Memory | 227 KB | 99 KB |
| Target Arch | `sm100a` | `sm120` |

---

## 📦 示例代码：完整使用模式

### 1. 异步屏障（生产者-消费者）

来源：[CUDA Programming Guide §4.9.7](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html)

```cuda
#include <cuda/barrier>
using barrier_t = cuda::barrier<cuda::thread_scope_block>;

__device__ void produce(barrier_t ready[], barrier_t filled[], float* buffer, int buffer_len, float* in, int N) {
  for (int i = 0; i < N / buffer_len; ++i) {
    ready[i % 2].arrive_and_wait();  // 等待 buffer_(i%2) 可填充
    /* 填充 buffer */
    barrier_t::arrival_token token = filled[i % 2].arrive();
  }
}
```

### 2. 异步事务屏障 + TMA

```cuda
#include <cuda/ptx>
__global__ void track_kernel() {
  __shared__ uint64_t bar;
  if (block.thread_rank() == 0) {
    cuda::ptx::mbarrier_init(&bar, block.size());
  }
  block.sync();

  uint64_t token = cuda::ptx::mbarrier_arrive_expect_tx(
      cuda::ptx::sem_release, cuda::ptx::scope_cluster,
      cuda::ptx::space_shared, &bar, 1, 0);

  while (!cuda::ptx::mbarrier_try_wait(&bar, token)) {}
}
```

### 3. 集群 bootstrap + mbarrier pipeline

```ptx
// Producer CTA:
mbarrier.init.shared::cluster.b64 [mbar_addr], count;
barrier.cluster.arrive;
barrier.cluster.wait;
mbarrier.arrive.expect_tx.shared::cluster.b64 _, [mbar_addr], byte_count;
cp.async.bulk.tensor.shared::cluster.global.mbarrier::complete_tx::bytes
    [smem_dst], [tensor_map, {x, y}], [mbar_addr];

// Consumer CTA（在不同 SM）：
mbarrier.try_wait.parity.shared::cluster.b64 %pred, [mbar_addr], %phase;
@!%pred bra wait_loop;
```

### 4. Blackwell tcgen05 + 集群 mbarrier

```ptx
// TMA warp
mbarrier.arrive_expect_tx(mbar_l, A_STAGE + B_STAGE);
cp.async.bulk.tensor.2d.shared::cluster.global.tile.mbarrier::complete_tx::bytes
    [smem_a], [A.tma_desc(), (ki*BK, m_base)], [mbar_l];

// MMA warp
mbarrier.try_wait.parity.shared__cta.b64 ready, [mbar_l], load_phase;
@!ready bra lwait;

tcgen05.mma.cta_group::1.kind::f16 [tmem_addr], desc_a, desc_b, idesc, enable_input_d;
tcgen05.commit(bar_mma, space="cluster");
```

---

## 🎯 对 PTX 模拟器的关键建议

### 仿真模型需要追踪的状态

| 状态 | 数据结构 | 来源 |
|------|---------|------|
| **命名屏障** (16 个/CTA) | 16 个 entry：{arrival_count, expected_count, completed} | `bar.sync N` |
| **mbarrier** (任意数量/CTA) | 64-bit 对象：{participant_count(20), pending_count(20), tx_count(20), phase(1)} | `mbarrier.init` |
| **BSSY/BSYNC 收敛栈** | 显式 reconvergence stack | 编译器插入 |
| **WARPSYNC mask** | 32-bit lane mask | `bar.warp.sync` |

### 关键仿真语义

1. **`bar.sync` 在 Volta+ 后不再隐式 membar** — 必须显式 membar.cta 才能保证内存顺序
2. **mbarrier 状态机** — 必须模拟 phase 翻转、tx_count 增减、arrive 的原子性
3. **集群作用域** — 需要 GPC 拓扑模型；DSMEM 访问需要特殊路径
4. **TMA/cp.async.bulk 完成** — 由硬件触发 `mbarrier.complete_tx`，需要异步引擎跟踪
5. **Waiter 睡眠 vs 轮询** — Hopper 上 wait 不占用 SM 调度槽位；模拟器可以选择忽略这个优化

### 反模式警告

> "A hand-rolled inline-PTX `bar.sync 0` without a paired `membar` is **a real, silent reorder hazard on SM 70+**."  
> — [CICC Builtins Reference](https://gh.evko.io/nvopen-tools/cicc/builtins/barriers.html)

模拟器如果实现 `bar.sync`，**必须同时实现 `membar.cta` 的隐式副作用**，否则会与硬件行为不符。
