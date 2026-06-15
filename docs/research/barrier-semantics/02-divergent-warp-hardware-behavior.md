# 02. 发散 Warp 命中 Barrier 的硬件行为

> **子代理调研任务**：`bg_f66b8d61` — Research divergent warp barrier behavior  
> **调研日期**：2026-06-15  
> **主题**：发散 warp 命中 barrier 时的真实硬件语义  
> **来源**：MLIR NVVM Dialect, LLVM NVPTX intrinsic 文档, Volta Tuning Guide, NVIDIA dev forum, Stack Overflow 权威回答

---

## 📌 权威来源

| 引用标签 | 来源 | URL |
|---------|------|-----|
| **[MLIR-NVVM]** | MLIR NVVM Dialect 文档 | https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmbarwarpsync-nvvmsyncwarpop |
| **[NVPTX-LLVM]** | LLVM NVPTX PR #140615 | https://github.com/llvm/llvm-project/pull/140615 |
| **[PTX 9.3]** | PTX ISA 9.3 | https://docs.nvidia.com/cuda/parallel-thread-execution/index.html |
| **[Volta-Tune]** | Volta Tuning Guide | https://docs.nvidia.com/cuda/volta-tuning-guide/ |
| **[NVIDIA-Forum]** | NVIDIA dev forum #282509 (Robert_Crovella, NVIDIA 论坛版主) | https://forums.developer.nvidia.com/t/requesting-clarification-cuda-warp-level-primitives-and-thread-divergence/282509 |
| **[StackOverflow]** | Stack Overflow 权威回答 (tera, 2017) | https://stackoverflow.com/questions/44487382/what-does-thread-count-mean-for-bar-arrive-ptx-barrier-synchronization-instructi |
| **[CICC]** | CICC Reverse Engineering | https://gh.evko.io/nvopen-tools/cicc/builtins/barriers.html |

---

## 关键定义：到达计数模型

**`bar.sync a, b`** 与 **`bar.arrive a, b`** 的操作数 `b` 含义（PTX ISA 1.0 以来一直如此）：

> Operand `b` specifies the number of threads participating in the barrier.  
> *（来源：[Stack Overflow 引用 PTX ISA](https://stackoverflow.com/questions/44487382)）*

`b` 是**预期的总到达线程数**，是跨整个 CTA 的全局计数器；当某 warp 抵达 `bar.sync` 时，硬件把该 warp 的**活动 lane 数**加到到达计数器上。

---

## Q1：部分谓词关闭的 `bar.sync N, 0xFFFF`

**情景**：warp 执行 `bar.sync 0, 0xFFFF`（65535 个线程 = 32 warps × 32 线程 − 1），但当前 warp 由于谓词 `!p1` 仅 lanes 0-15 活跃（`active_mask = 0x0000FFFF`）。

### 答案：到达计数 = `popc(active_mask_at_barrier) = 16`，而非 32

**核心规则**（[Volta-Tune] §1.4.1.2）：

> "Applications using `__syncthreads()` or the PTX `bar.sync` (and their derivatives) in such a way that a barrier will not be reached by some **non-exited thread** in the thread block must be modified to ensure that all non-exited threads reach the barrier."

**含义**：
- **没有退出但未抵达 `bar.sync` 的线程 = UB / 死锁**（Volta+ 必须修改）
- 因此在 Volta+ 上，把 `bar.sync` 放在仅一半 lane 活跃的谓词分支内**本身是 UB**
- 如果确实发生：硬件仅累计**实际执行该指令的活跃 lane 数**（=16）

**Pre-Volta 行为**（[CICC] Builtins）：

> "Pre-Volta, `bar.sync` doubled as an implicit `membar.cta` because all threads in the warp executed lockstep."

Pre-Volta 时代，所有 32 个 lane 强制 lockstep，没有"predicated off"概念；`active_mask` 在 `bar.sync` 处由硬件保证为 0xFFFFFFFF。

**对模拟器的影响**：
- 如果你的模拟器把 `bar.sync 0, 0xFFFF` 的到达单位硬编码为 warp 级（每 warp 抵达加 32），则在 half-predicated 情形下到达数会被高估 2 倍
- **正确做法**：使用 `(bar.sync b) → arrive += popc(per_warp_active_mask_at_barrier)`

---

## Q2：参与掩码 `0x00000003` 但全部 32 lane 都活跃

**情景**：所有 32 个 lane 都执行 `bar.warp.sync 0x00000003`（只有 lanes 0 和 1 是"参与者"），但**实际执行**指令的 lane 是全部 32 个。

### 答案：**未定义行为（UB）**

**MLIR NVVM Dialect 文档原文**（[来源](https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmbarwarpsync-nvvmsyncwarpop)）：

> **"Important constraints:**
> - The behavior is **undefined** if the executing thread is not included in the mask (i.e., the bit corresponding to the thread's lane ID is not set)
> - For compute capability sm_6x or below, all threads in the mask must execute the same `bar.warp.sync` instruction in convergence"

> "The `mask` operand specifies the threads participating in the barrier, where each bit position corresponds to the thread's lane ID within the warp. **Only threads with their corresponding bit set in the mask participate in the barrier synchronization.**"

**关键点**：
- `bar.warp.sync` 的 `membermask` 不是参与计数，而是**谁必须参与**的声明
- 执行该指令的 lane 必须**自己是 mask 的成员**（lane 0/1 必须执行）
- 不在 mask 中的 lane 2-31 执行该指令 → **UB**
- **正确用法**：所有非 mask 成员 lane 应被谓词掉（如 `@%p my_label` 跳过整条指令），只剩 mask 内的 lane 实际执行

**等价描述**（[NVIDIA-Forum] Robert_Crovella，2024）：

> "It will cause the executing thread to wait until all warp lanes named in mask have executed a `__syncwarp()` (with the same mask) before resuming execution. **Each calling thread must have its own bit set in the mask** and all non-exited threads named in mask must execute a corresponding `__syncwarp()` with the same mask, or the result is undefined."

---

## Q3：分歧 warp 两半在不同时间到达同一屏障

**情景**：lanes 0-15 先到达 `bar.sync`，lanes 16-31 后到达（由分歧分支导致）。

### 答案：屏障在**所有活动 lane 都抵达后**才释放一次（Volta+ ITS 行为）

**核心 ISA 语义**（PTX ISA Introduction）：

> "Sequential consistency is provided by the `bar.sync` instruction. Threads wait at the barrier until all threads in the CTA have arrived."

**Volta+ 独立线程调度（ITS）**（[PTX 9.3]）：

> "Starting with the Volta architecture, **Independent Thread Scheduling** allows full concurrency between threads, regardless of warp... A schedule optimizer determines how to group active threads from the same warp together into SIMT units."

**`__syncwarp()` 行为的间接证据**（[NVIDIA-Forum] Robert_Crovella 描述 Volta+ 调度器如何处理分歧屏障）：

> "The warp scheduler in Volta and beyond can and will try to 'combine' the sync variants of shuffle ops from divergent paths, if necessary, to try and 'satisfy' the member mask. If the member mask is satisfied, then the op completes 'as if' the warp is converged at least to the extent required by the shuffle mask."

**对 `bar.sync` 的应用**：
- 半 A 先抵达 → 不释放，调度器让半 A **在屏障处阻塞/等待**
- 半 B 抵达 → 满足到达计数 → 屏障释放 → 半 A 和半 B 都通过
- **不会有"半 A 单独通过"的情况**

**对模拟器的影响**：
- Wbar 跟踪器应当：first-arrival 标记不释放屏障；必须等所有活跃 lane 都 `arrive()` 后才能 release
- 这就是 PTX-EMU 现有实现的正确做法

---

## Q4：硬件是否会"阻塞"非参与线程？

### 答案：分两种屏障指令分别回答

#### (a) `bar.sync N, count`（CTA 级屏障）：**不允许非参与**

**Volta+ 强约束**（[Volta-Tune]）：

> "Applications using `__syncthreads()` or the PTX `bar.sync` (and their derivatives) in such a way that a barrier will not be reached by some non-exited thread in the thread block must be modified to ensure that **all non-exited threads reach the barrier**."

- 不退出但跳过 `bar.sync` → **未定义行为**（实测中表现为 `cudaDeviceSynchronize()` 时报 `illegal instruction`，compute-sanitizer 报告 "Barrier error detected. Divergent thread(s) in block."）
- 来源：[Stack Overflow: CUDA: how to use barrier.sync](https://stackoverflow.com/questions/53662484/cuda-how-to-use-barrier-sync)

**硬件行为**：
- 硬件**不**主动阻塞未抵达的 lane
- 屏障只在"到达计数 ≥ 预期 count"时释放
- 如果某些非退出 lane 永远不到达，**屏障永远不释放** → 死锁

#### (b) `bar.warp.sync membermask`（warp 级屏障）：**mask 决定参与资格**

**MLIR NVVM 原文**：

> "The `mask` operand specifies the threads participating in the barrier, where each bit position corresponds to the thread's lane ID within the warp. **Only threads with their corresponding bit set in the mask participate in the barrier synchronization.**"

**LLVM NVPTX `barrier.cta.sync.aligned` 语义**（[NVPTX-LLVM]）：

> "Operand %id specifies a logical barrier resource and must fall within the range 0 through 15. When present, operand %n specifies the number of threads participating in the barrier. When specifying a thread count, the value must be **a multiple of the warp size**."

> "The '`@llvm.nvvm.barrier.cta.*`' intrinsic has an optional '`.aligned`' modifier to indicate textual alignment of the barrier. When specified, it indicates that **all threads in the CTA will execute the same** '`@llvm.nvvm.barrier.cta.*`' instruction."

**`aligned` 修饰符含义**：
- `aligned = true`（默认）→ 所有 CTA 线程都必须执行；否则 UB
- `aligned = false` → 允许非全部线程执行（用于子集屏障）

---

## Q5：`bar.warp.sync` 与 active mask 的关系

### 答案：**不等价；membermask 是显式参与声明，active_mask 是运行时收敛状态**

**MLIR NVVM 原文**：

> "**Important constraints**:
> - The behavior is undefined if the executing thread is not included in the mask
> - For compute capability sm_6x or below, **all threads in the mask must execute the same** `bar.warp.sync` instruction in convergence"

**两层关系**：

| 概念 | 来源 | 与 active_mask 关系 |
|------|------|------------------|
| **`membermask`（参与掩码）** | PTX 立即数 / 寄存器参数 | 由程序员/编译器显式声明；与运行时收敛状态无关 |
| **`active_mask`（活跃掩码）** | 硬件 ITS 运行时状态 | 反映当前周期实际执行的 lane |
| **必须交集**：mask 中的 lane 必须都执行 | sm_6x：必须收敛执行；sm_70+：调度器会尝试满足 | 通过 ITS 重新汇聚实现 |

**Volta+ 上的"组合同步"行为**（[NVIDIA-Forum] Robert_Crovella）：

> "The warp scheduler in Volta and beyond **can and will try to 'combine' the sync variants of shuffle ops from divergent paths, if necessary, to try and 'satisfy' the member mask**. If the member mask is satisfied, then the op completes 'as if' the warp is converged at least to the extent required by the shuffle mask."

> "if syncwarp is used in warp-divergent paths (what ?!?!), and the scheduler can do so, **it will cause each thread to wait at the syncwarp, until all threads, even in the divergent path, have reached the syncwarp**."

**`__activemask()` 与 `__syncwarp(mask)` 的关系**（[StackOverflow] tera 回答）：

> "__activemask() tells you what threads happen to be convergent when the function is called, **which can be different from what you want to be in the collective operation**."

**对模拟器的影响（关键）**：
- `bar.warp.sync mask` 的到达判定**不**基于 active_mask；它基于 `mask` 中所有 lane 的到达
- 即使 active_mask ⊂ mask，mask 中其余 lane 也会被调度器"拉"过来执行屏障
- 这意味着：**`bar.warp.sync` 是收敛屏障**（convergent barrier），调度器必须保证 mask 内所有 lane 都到达才能释放

---

## 综合判断：`arrive()` 应当调用给谁？

基于以上权威语义，**`arrive()` 应当调用给"参与线程"，定义如下**：

### 规则 1（CTA 级 `bar.sync N, count`）

- **必须调用给所有非退出的活跃 lane**（active_mask 中的 lane）
- Predicated-off lane（谓词为 false）不调用 arrive
- **违反**：UB / 死锁（Volta+）

### 规则 2（Warp 级 `bar.warp.sync membermask`）

- **必须调用给 membermask ∩ active_mask 中的 lane**
- 即只有 mask 内且实际执行指令的 lane 才调用 arrive
- mask 外但 active 的 lane **不应**调用 arrive（它们执行了指令本身是 UB）

### 规则 3（半 A/半 B 分歧场景）

- 两半都到达后屏障才释放
- 半 A 抵达时**不**触发 release；必须等半 B
- Wbar 的 reconvergence PC 必须在半 B 抵达后才弹出

---

## 验证 PTX-EMU 模拟器现状的关键检查点

| 检查项 | 期望行为 | 验证方法 |
|--------|---------|---------|
| `bar.sync N, count` 在 half-active 时 | arrive += popc(half_mask)，**非** 32 | 单元测试：setup_pred(0x0000FFFF) + bar.sync 0, 32 |
| `bar.warp.sync 0x03` 全 32 lane 执行 | UB；应仅 lanes 0,1 真正执行 | 测试用 `@%p` 把 lanes 2-31 跳过 |
| 两半分别在 PC 屏障 | Wbar 不释放直到双方都 arrive | 集成测试：两半 PC 不同，先 step half_A → 检查未释放 |
| `bar.sync` 谓词关闭一半 | UB（Volta+）；模拟器应警告 | e2e 测试：if(threadIdx.x<16) bar.sync; else skip |

---

## 核心结论

1. **`arrive()` 调用对象 = 实际执行该屏障指令的 lane 集合**
   - 对 `bar.sync N, count`：即 active_mask 中的所有 lane（谓词开启的）
   - 对 `bar.warp.sync mask`：即 membermask ∩ active_mask

2. **到达计数**：
   - CTA 级 `bar.sync N, b`：跨 CTA 累加 `popc(per-warp active mask)`，目标 = `b`
   - Warp 级 `bar.warp.sync mask`：在 mask 内的 lane 都抵达后释放；mask 是固定值

3. **半 A / 半 B 不同时抵达**：屏障**不**在第一半到达时释放；必须等所有 mask/active lane 都抵达

4. **Volta+ UB 情形**：
   - 非退出线程跳过 `bar.sync` → UB / 死锁
   - `bar.warp.sync` 的 mask 外 lane 执行指令 → UB
   - 模拟器可以选择：(a) 严格 UB 检查并断言失败；(b) 宽松实现（只累计实际到达的 lane）

5. **`bar.warp.sync` 是收敛屏障**（convergent）：调度器会"拉"mask 内但尚未到达的 lane 抵达屏障 — 对 PTX-EMU 的 SIMT stack 与 Wbar 释放时机非常关键。

---

## 🔑 关键源参考

1. **MLIR NVVM Dialect** — `bar.warp.sync` 与 `barrier` 操作规范：https://mlir.llvm.org/docs/Dialects/NVVMDialect/#nvvmbarwarpsync-nvvmsyncwarpop
2. **LLVM NVPTX PR #140615** — `bar.sync` ↔ `barrier.cta.sync.aligned` 映射：https://github.com/llvm/llvm-project/pull/140615
3. **PTX ISA 9.3 Introduction** — ITS 与 divergence-of-threads：https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#divergence-of-threads-in-control-constructs
4. **Volta Tuning Guide §1.4.1.2** — "all non-exited threads must reach the barrier"：https://docs.nvidia.com/cuda/volta-tuning-guide/
5. **NVIDIA dev forum #282509** — ITS 调度器对分歧屏障的"组合同步"行为：https://forums.developer.nvidia.com/t/requesting-clarification-cuda-warp-level-primitives-and-thread-divergence/282509
6. **Stack Overflow (tera, 2017)** — `bar.arrive` thread-count 含义：https://stackoverflow.com/questions/44487382/what-does-thread-count-mean-for-bar-arrive-ptx-barrier-synchronization-instructi
7. **CICC Reverse Engineering** — Pre-Volta `bar.sync` 隐式 membar + Volta+ 重排风险：https://gh.evko.io/nvopen-tools/cicc/builtins/barriers.html
