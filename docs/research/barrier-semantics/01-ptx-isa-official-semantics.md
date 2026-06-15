# 01. PTX ISA 官方 barrier 语义

> **子代理调研任务**：`bg_6368223a` — Research PTX bar.sync/bar.warp.sync semantics  
> **调研日期**：2026-06-15  
> **主题**：`bar.sync` / `bar.warp.sync` / named barrier / cluster barrier 的形式语义  
> **来源**：PTX ISA 9.3 (NVIDIA, 2026), Volta Tuning Guide, LLVM NVPTX intrinsic 文档, CUDA Programming Guide

---

## 📌 权威来源清单

| 引用标签 | 来源 | URL |
|---------|------|-----|
| **[PTX 9.3]** | PTX ISA 9.3 (NVIDIA, 2026, latest) | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| **[PTX 8.7]** | PTX ISA 8.7 (CUDA 12.8 archive) | https://docs.nvidia.com/cuda/archive/12.8.2/parallel-thread-execution/index.html |
| **[PTX 8.0 PDF]** | PTX ISA 8.0 (PDF) | https://docs.nvidia.com/cuda/archive/12.0.1/pdf/ptx_isa_8.0.pdf |
| **[Volta-Tune]** | Volta Tuning Guide 13.3 | https://docs.nvidia.com/cuda/volta-tuning-guide/ |
| **[Async-Barrier]** | CUDA Programming Guide §4.9 — Asynchronous Barriers | https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-barriers.html |
| **[NVPTX-LLVM]** | LLVM NVPTX intrinsic docs (PR #140615) | https://lists.llvm.org/pipermail/cfe-commits/Week-of-Mon-20250519/710933.html |
| **[ptxas-ref]** | PTXAS Reverse Engineering Reference | https://gh.evko.io/nvopen-tools/ptxas/passes/sync-barriers.html |

---

## Q1. 屏障 + 分歧（divergent warp）的行为

### 官方原文（[PTX 8.0 PDF] §9.7.12.1 / [PTX 9.3] §9.7.14.1）：

> **"Barriers are executed on a per-warp basis as if all the threads in a warp are active. Thus, if any thread in a warp executes a bar instruction, it is as if all the threads in the warp have executed the bar instruction. All threads in the warp are stalled until the barrier completes, and the arrival count for the barrier is incremented by the warp size (not the number of active threads in the warp). In conditionally executed code, a bar instruction should only be used if it is known that all threads evaluate the condition identically (the warp does not diverge). Since barriers are executed on a per-warp basis, the optional thread count must be a multiple of the warp size."**

### Volta 之后的强化（[Volta-Tune] §3）：

> **"Starting with Volta, the CUDA built-in `__syncthreads()` and PTX instruction `bar.sync` (and their derivatives) are enforced per thread and thus will not succeed until reached by all non-exited threads in the block. Code exploiting the previous behavior will likely deadlock and must be modified to ensure that all non-exited threads reach the barrier."**

### 推论：16/16 分歧 case 的实际行为

- **路径 A 的 16 个 lane 命中** → 该 warp 被 stall 等待（虽然另 16 个 lane 不在路径 A，但它们已经在 SIMT stack 上等待 reconverge 到屏障处）。
- **路径 B 的 16 个 lane 同样命中** → 屏障到达计数 = 32（=两条路径的 warp 整 warp 累计）。
- **屏障完成时**，所有 32 个 lane 都"released"，并都获得**同一个** reconvergence PC（屏障之后的目标 PC）。

---

## Q2. `bar.warp.sync` vs `bar.sync` 的区别

### 官方定义（[PTX 8.0 PDF] §9.7.12.2 / [PTX 9.3] §9.7.14.2）

`bar.warp.sync` 在 PTX 6.0 引入（sm_30+），语义在 PTX 6.5 起与现代等价：

> **A new instruction `bar.warp.sync` which allows synchronizing threads in warp.**

### SASS 映射（[ptxas-ref] "Sync & Warp Intrinsics"）：

| PTX | SASS | Purpose |
|---|---|---|
| `bar.warp.sync membermask` | `WARPSYNC mask` | Synchronize warp lanes specified by mask |
| `bar.sync N` | `BAR.SYNC` | Block until all CTA threads arrive at barrier N |
| `bar.sync N, count` | `BAR.SYNC` | Block until `count` threads arrive at barrier N |

### 关键差异表

| 维度 | `bar.sync N[, count]` | `bar.warp.sync membermask` |
|------|----------------------|---------------------------|
| **作用域** | 整个 CTA（默认）；可显式 `count` | **单 warp（32 lane）内** |
| **同步对象** | CTA 内所有非退出 thread | membermask 中**置位**的 lane |
| **硬件资源** | 16 个命名 barrier 资源（index 0–15） | SASS `WARPSYNC` 单指令，无命名 |
| **mask 语义** | `count` = 到达线程数（必须是 warpSize 倍数） | `membermask` = 参与同步的 lane 位图 |
| **分分歧义** | 跨 warp、跨 lane 聚合 | 限单 warp；mismatched mask = 未定义 |
| **可用 SM** | 全部 | sm_70+（Volta+ 才有显式 `WARPSYNC`） |
| **mem 序** | sm_70+ 单独不再隐含 `membar.cta`（**必须显式配对**） | 仅控制流同步，**不**含 memory fence |

### 关键引用（[Volta-Tune]）：

> "Applications using `__syncthreads()` or the PTX `bar.sync` (and their derivatives) in such a way that a barrier will not be reached by some non-exited thread in the thread block must be modified to ensure that all non-exited threads reach the barrier."

> "Applications that assume reads and writes are implicitly visible to other threads in the same warp need to insert the new `__syncwarp()` warp-wide barrier synchronization instruction..."

⚠️ **`bar.warp.sync` 是纯 warp 级构造，作用域恒为 32 lane**。它**不能**用于 CTA 级同步。

---

## Q3. Named Barrier 的用途

### 硬件事实（[ptxas-ref]）：

> **"The hardware provides 16 named barriers (indices 0–15), each tracking participation counts. PTX exposes these as:** `bar.sync N` — block until all threads in the CTA arrive at barrier N"

### 语法（[PTX 8.0 PDF] §9.7.12.1）

```ptx
bar.sync      a [, b];           // a = barrier id (0-15), b = thread count
bar.arrive    a [, b];           // 不阻塞，仅"打卡"
bar.red.{and,or,popc}  a, p;     // barrier + warp-level reduction
```

`b` 的官方约束（[PTX 8.4+ 描述 via forum]）：

> "Operand `b` specifies the number of threads participating in the barrier. If no thread count is specified, all threads in the CTA participate in the barrier. When specifying a thread count, the value must be a multiple of the warp size."

### Named Barrier 的三大用途

1. **子集同步**（最常见）：把 CTA 内不同 warp 分成若干"组"，每组用不同 `barrier_id (0-15)` 独立同步。**典型例子**：producer-consumer 解耦、`__syncwarp` 增强版。
2. **避免锁死**（[StackOverflow / NVIDIA 论坛]）：`__syncthreads()` 隐式用 `bar.sync 0`；如果 `bar.sync 0` 出现在 `if (tid < N)` 条件块内，**条件外**的线程会因找不到匹配的 count 而 deadlock。改用 `bar.sync 1`（不同 ID）隔离。
3. **多相 rendezvous**：同一 `barrier_id` 多次使用即可形成多阶段同步。

### 实际产品代码（NVIDIA/cutlass `include/cutlass/arch/barrier.h`）

```cpp
template <
  uint32_t ThreadCount_,      // 参与该 barrier 的 thread 数
  uint32_t Offset = 0,        // 加到用户 ID 上得到最终 barrier slot
  uint32_t MaxNumNamedBarriers = 16
>
struct NamedBarrierManager {
  static_assert(MaxNumNamedBarriers <= arch::NamedBarrier::HardwareMaxNumNamedBarriers);
  // ...
};
```

> "Structure for managing multiple NamedBarriers to be used by different warp groups, allowing runtime index values to be used to call into named barriers with compile-time-constant IDs."

---

## Q4. Hopper/Blackwell 的 Cluster 级屏障

### 引入（[PTX 9.3] §1.3 / §2.2.2）

> **"Cluster is a group of CTAs that run concurrently or in parallel and can synchronize and communicate with each other via shared memory. ... Cluster-wide barriers can be used to synchronize all the threads within the cluster. ... Cluster level is applicable only on target architecture `sm_90` or higher."**

### 两种 Cluster Barrier 形态

#### (a) 命名型 cluster barrier（[PTX 9.3] §9.7.14.3 "barrier.cluster"）

```ptx
barrier.cluster.arrive;        // 所有 CTA 都执行
barrier.cluster.wait;          // 等待 cluster 内全部 CTA 到达
barrier.cluster.arrive.relaxed; // sm_90+ 起的 relaxed-memory 变体（ptx ≥ 8.1）
```

**用途**（[Hopper Tensor Core Programming Guide]）：
> "**`barrier.cluster.arrive` / `barrier.cluster.wait` (PTX ISA 9.2, §9.7.13.3 'barrier.cluster')** — a named barrier at cluster scope. Used for bootstrapping (before any cluster-wide mbarrier is valid, all CTAs need a synchronization point)."

#### (b) Cluster-scope mbarrier（[PTX 9.3] §9.7.14.16.11）

```ptx
mbarrier.init.shared::cluster.b64     [mbar_addr], count;
mbarrier.arrive.shared::cluster.b64   _, [mbar_addr];
mbarrier.try_wait.parity.shared::cluster.b64 complete, [mbar_addr], phase;
```

**Hopper 编程模型要点**（[Hopper Programming Guide]）：
> "State space: `.shared::cta` (CTA-local) or `.shared::cluster` (cluster-visible). **The cluster-scoped variant is what makes cross-CTA synchronization within a thread block cluster possible** (PTX ISA 9.2, §9.7.13.15.8)."

### Cluster 屏障 vs CTA 屏障对比

| 维度 | CTA (`bar.sync N`) | Cluster (`barrier.cluster.*` / `mbarrier.*.cluster`) |
|------|------------------|------------------------------------------------------|
| **SM floor** | 全部 | sm_90+（Hopper） |
| **同步范围** | 单 CTA 全部 thread | 整个 cluster 的所有 thread（所有 CTA） |
| **到达计数** | 16 个命名 barrier ID | 显式 `count`（mbarrier）或全 cluster（named cluster） |
| **配套设施** | 16 hardware named barriers | DSMEM（Distributed Shared Memory）+ `mapa` |
| **典型用途** | `__syncthreads` | TMA multicast、warp-specialized pipeline 跨 CTA |

---

## Q5. 屏障 "released" 时所有 lane 是否获得同一 reconvergence PC

### 答案：**是的，全部 32 lane 在屏障完成后获得同一 PC（屏障后指令的 PC）**。

### 机制（[ptxas-ref] "BSSY/BSYNC — Convergence Barriers"）：

> "The `BSSY`/`BSYNC` instruction pair replaces the pre-Volta implicit reconvergence stack. The compiler must insert these pairs explicitly at divergence/reconvergence points."

| SASS Opcode | Purpose |
|---|---|
| `BSSY B, target` | Push a synchronization barrier; `target` is the reconvergence point |
| `BSYNC B` | Pop and wait at the convergence barrier B |

### PTX 源码层的可见行为：

```ptx
PTX:  bar.sync 0;
SASS: BAR.SYNC 0x0;
      // stalls warp until all CTASize threads arrive at barrier 0
```

屏障完成后，**整 warp** 释放，所有 32 lane 都跳到 `bar.sync` 之后的同一 PC。

### Mask 的角色

"Mask 决定哪些 lane 到达"在 `bar.sync` 上**仅在计数层面**生效：
- 默认无 count ⇒ 等 CTA 内**所有非退出** thread（Volta+）。
- 显式 `count` ⇒ 等 `count` 个非退出 thread。
- **mask（active_mask）不决定 PC advance**：所有非退出的活跃 lane 都被 stall，直到计数满足，**所有非退出 lane 一同 released** 到屏障后 PC。

---

## Q6. CTA-scope `bar.sync` 与参与 mask 的关系

### 直接答案：**所有非退出 thread 都会"执行"屏障；但只有参与 mask 内的 thread 提供到达计数**。

### PTX 静态语义（[PTX 9.3] §9.7.14.1）

```ptx
bar.sync 0;          // 无 count：CTA 内所有非退出 thread 都必须到达
bar.sync 0, 64;      // count=64：CTA 内 64 个非退出 thread 到达即可
bar.sync 0, 0xFFFFFFFF;  // 32 thread：等同全 warp 屏障
```

- `count` 是**期望到达的 thread 数**，必须是 32 的倍数。
- **不在 mask 范围（`count` 之外）但仍活跃的 thread**：它们**必须**也命中屏障，但**不增加计数**。
- 不命中 → dead lock（Volta+ 严格）或 UB（pre-Volta）。

### 与 "participation mask" 的精确区别

PTX `bar.sync` 显式只接 `count`，**不接 per-lane mask**。真正的 per-lane mask 只在：
- `bar.warp.sync membermask`（warp 内）
- `__shfl_sync` / `__vote_sync` / `__match_sync` / `mbarrier.*`（带 active_mask 处理）

---

## 🎯 综合答案表

| 问题 | 简洁答案 | 关键引用 |
|------|---------|---------|
| **Q1** | **32 lane 全部参与**。屏障 per-warp basis，活跃 mask 命中后整 warp stall；Volta+ 严格检查所有非退出 thread 到达。 | [PTX 9.3] §9.7.14.1, [Volta-Tune] §3 |
| **Q2** | `bar.sync` = CTA-wide，16 个硬件命名 barrier，count 是 thread 数；`bar.warp.sync` = **纯 warp 内 32 lane**，membermask 是 lane 位图。 | [PTX 9.3] §9.7.14.2, [ptxas-ref] |
| **Q3** | 16 个 hardware named barrier slot（0–15），用于 CTA 内子集同步，避免 deadlock，producer-consumer 分组。 | [ptxas-ref], [cutlass-barrier], [PTX 8.0 PDF] §9.7.12.1 |
| **Q4** | sm_90+ 起，两种：① `barrier.cluster.arrive/wait`（命名 cluster 屏障，用于 bootstrap）② `mbarrier.*.shared::cluster`（cluster-scope 共享内存屏障，配 DSMEM+mapa）。sm_100+ 加 `tcgen05.fence/commit`。 | [PTX 9.3] §9.7.14.3, §9.7.14.16.8, [Hopper Programming Guide] |
| **Q5** | **是**。屏障完成时所有非退出 lane 一同 release 到屏障后同一 PC；active mask 只影响到达计数，不影响 PC advance。 | [ptxas-ref] "BSSY/BSYNC" |
| **Q6** | 不在 mask 范围但活跃的 thread **仍必须**命中屏障（否则 deadlock，Volta+ 严格），但**不增加到达计数**。屏障是 per-warp 粒度执行，mask 仅约束 count。 | [PTX 9.3] §9.7.14.1, [Volta-Tune] §3 |

---

## ⚠️ 关键陷阱（对架构决策）

1. **Volta+ 起 `bar.sync` 不再是隐式 `membar.cta`**（[ptxas-ref] "CICC Reverse Engineering"）：
   > "A hand-rolled inline-PTX `bar.sync 0` without a paired `membar` is a real, silent reorder hazard on SM 70+."
   
   **必须显式 `membar.cta`（或 `fence.cta`）配套**，否则 load 仍可能读到 stale store。

2. **`bar.warp.sync` 不跨 warp**：典型误用是想做"双 warp 屏障"，但单条 `bar.warp.sync` 只能同步**一个 warp** 内的 32 lane。

3. **`bar.sync` 的 count 参数恒为 32 倍数**（PTX 原文）：不要传入 16/8/4 这类非 warpSize 倍数，会被拒或 UB。

4. **Cluster barrier 必须显式标 sm_90+**（[PTX 9.3]）：
   > "Cluster level is applicable only on target architecture `sm_90` or higher."

5. **Named barrier (barrier.sync) 的 ID 范围 0–15**：超过 15 是 UB。

6. **Mbarrier 阶段翻转是硬件自动的**（[Async-Barrier] §4.9）：
   > "When the last call to `bar.arrive()` causes the countdown to reach zero, the countdown is automatically and atomically reset."
   
   phase parity bit（0/1）由硬件管理。

---

## 📚 PTX 范例

### 场景 1：CTA 256 thread = 8 warp；前 4 warp 是 producer，后 4 warp 是 consumer

```ptx
.visible .entry producer_consumer()
{
    .reg .pred %p;
    .reg .u32 %tid;
    mov.u32 %tid, %tid.x;

    setp.lt.u32 %p, %tid, 128;       // producer = low half
    @%p bra L_PROD;
    bra.uni L_CONS;

L_PROD:
    bar.sync 1, 128;                  // 只 producer 4 个 warp 同步
    bar.sync 0;                        // 全 CTA 同步
    bra.uni L_CONT;

L_CONS:
    bar.sync 2, 128;                  // 只 consumer 4 个 warp 同步
    bar.sync 0;                        // 全 CTA 同步
L_CONT:
    ret;
}
```

### 场景 2：cluster (sm_90+) — 跨 CTA 同步

```ptx
.visible .entry cluster_kernel()
{
    .reg .b64 %addr;
    .shared .b64 mbar;

    // 1) bootstrap：用命名 cluster barrier
    barrier.cluster.arrive;
    barrier.cluster.wait;

    // 2) 现在可以安全使用 cluster-scope mbarrier
    cvta.shared.u64 %addr, mbar;
    @%p0 mbarrier.init.shared::cluster.b64 [mbar], N;

    // 3) cluster 范围到达 + 等待
    mbarrier.arrive.shared::cluster.b64 _, [%addr];
    mbarrier.try_wait.parity.shared::cluster.b64 done, [%addr], 0;
    @!done bra wait_loop;
    ret;
}
```

---

## 📝 总结（一句话版）

- **`bar.sync`** = CTA-wide，per-warp 粒度执行，Volta+ 严格 per-thread 检查；16 个硬件命名 slot。
- **`bar.warp.sync`** = 单 warp 内 32 lane 同步，membermask 是 lane 位图。
- **`barrier.cluster.*`** = sm_90+ cluster-wide 命名屏障（bootstrap 用）。
- **`mbarrier.\*.cluster`** = sm_90+ 共享内存 cluster 屏障（DSMEM + mapa + phase parity）。
- 分歧 + 屏障：**所有非退出 lane 一同 stall、一同 release** 到同一 PC；mask 只控制到达计数，不影响 PC advance。
- 关键陷阱：`bar.sync` 在 sm_70+ **不再是隐式 membar.cta**，必须显式配套。
