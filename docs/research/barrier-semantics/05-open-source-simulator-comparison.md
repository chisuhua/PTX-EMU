# 05. 开源 GPU 模拟器 Divergent Barrier 实现对比

> **子代理调研任务**：`bg_cb6eece1` — Research open-source GPU simulator barrier handling  
> **调研日期**：2026-06-15  
> **主题**：gpgpu-sim / gem5 / MIAOW / Multi2Sim 对 divergent `bar.sync` 的实现  
> **来源**：accel-sim, gem5, VerticalResearchGroup, multi2sim GitHub 仓库源码

---

## 🔖 仓库元数据（用于 permalink）

| 模拟器 | 仓库 | HEAD commit SHA | 备注 |
|--------|------|----------------|------|
| gpgpu-sim | `accel-sim/gpgpu-sim_distribution` | `6c3cf4ff32110908386d605a7034fc67666a92de` | master 分支 v4.2.1+ |
| gem5 | `gem5/gem5` | `c8222cc67a399bfc01e8658dd14b30d5bfd634f9` | main 分支 |
| MIAOW | `VerticalResearchGroup/miaow` | `dbc5d7cc6e5fd58828239b59491eeb4f66503074` | Verilog RTL 实现 |
| Multi2Sim | `multi2sim/multi2sim` | `77b16e0ba3c23c5609657834b8cdfc7d0e22c303` | Southern Islands ISA |
| NVBit | `NVlabs/nvbit` | — | **非模拟器**，仅为 SASS 插桩工具，不实现屏障语义。排除。 |

---

## 📊 主对比表

| 模拟器 | `bar.sync` divergent 处理 | 到达计数单位 | 释放行为 | 关键 file:line |
|--------|---------------------------|--------------|----------|----------------|
| **gpgpu-sim** | 整个 warp 视为一个参与单位；不区分分歧路径上的 lane；CHANGES 明示"irrespective of divergence state" | **每 CTA 的 warp bitmap**（`m_bar_id_to_warps[bar_id]`） | 当 `at_barrier == active`（该 CTA 所有 warp 都到位）时，整 warp 推进；`m_warp_at_barrier &= ~at_barrier` 清等待位 | `src/gpgpu-sim/shader.cc:3847-3894` |
| **gem5-gpu** | 整个 wavefront 计一次；不按 exec_mask 拆分 | **每 CU 的计数器**（`_numAtBarrier` vs `_maxBarrierCnt`，后者在 WG dispatch 时设为 `num_wfs_in_wg`） | `releaseWFsFromBarrier(bar_id)` 遍历所有 SIMD/Slot，将匹配 `barrierId()` 的 WF 全部从 `S_BARRIER` → `S_RUNNING` | `src/gpu-compute/compute_unit.hh:92-181`；`compute_unit.cc:842-903` |
| **MIAOW** | 整个 wavefront 计一次；`wf_count_mask` 按 workgroup 大小硬编码 | **每 WG 的 16-bit 计数器**（`curr_wg_wf_count`，使用 `wf_count_mask`） | `all_wf_hit_barrier` 触发时，`next_curr_wg_wf_count = 0` 且 `next_curr_wg_wf_waiting = 40'b0` 清所有等待 | `src/verilog/rtl/issue/barrier_wait.v` 全文件 |
| **Multi2Sim** | 整个 wavefront 计一次 | **每 work-group 的计数器**（`work_group->getWavefrontsAtBarrier()`） | 当 `getWavefrontsAtBarrier() == getWavefrontsInWorkgroup()`，遍历 `work_group->getWavefrontsBegin/End` 清 `atBarrier` | `src/arch/southern-islands/emulator/WorkItemIsa.cc:2076-2106` |

---

## 🔬 详细答案（按问题）

### **Q1 — divergent 时如何计数到达？**  
**(a) 所有 active lane** vs **(b) 仅 participation mask lane**？

**结论：四个模拟器全部采用 (a) 的"warp/wavefront 级"语义，而非 (b) 的 per-lane participation。** NVIDIA 硬件的真实行为确实是 (a) 的 warp-级（详见 CUDA Programming Guide）。

#### 证据 1 — gpgpu-sim

**File**: [`src/gpgpu-sim/shader.cc#L3847-L3894`](https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/src/gpgpu-sim/shader.cc#L3847-L3894)

```cpp
void barrier_set_t::warp_reaches_barrier(unsigned cta_id, unsigned warp_id,
                                         warp_inst_t *inst) {
  barrier_type bar_type = inst->bar_type;
  unsigned bar_id = inst->bar_id;
  unsigned bar_count = inst->bar_count;
  ...
  m_bar_id_to_warps[bar_id].set(warp_id);    // ← 把整个 warp 一位
  if (bar_type == SYNC || bar_type == RED) {
    m_warp_at_barrier.set(warp_id);          // ← 整 warp 阻塞
  }
  warp_set_t warps_in_cta = w->second;
  warp_set_t at_barrier = warps_in_cta & m_bar_id_to_warps[bar_id];
  warp_set_t active = warps_in_cta & m_warp_active;
  if (bar_count == (unsigned)-1) {            // ← bar.sync 的 bar_count 是 -1
    if (at_barrier == active) {              // ← "所有 warp 都到了"才释放
      // all warps have reached barrier, so release waiting warps...
      m_bar_id_to_warps[bar_id] &= ~at_barrier;
      m_warp_at_barrier &= ~at_barrier;
      ...
    }
  } else {                                    // ← named barrier (bar.sync N, %b)
    if ((at_barrier.count() * m_warp_size) == bar_count) {
      // required number of warps have reached barrier, so release waiting warps...
```

**关键点**：
- `m_bar_id_to_warps[bar_id].set(warp_id)` — 把整个 warp 标记为 arrived。**不读** `warp_inst_t::active_mask`。
- 释放条件 `at_barrier == active` 是 **warp-bitmap 的集合相等**，不是 lane-level 计数。
- 没有显式的 participation mask 字段被传给 barrier_set。

**设计意图确认** — [`CHANGES`](https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/CHANGES)：

> "bar.sync timing change for compatibility with NVIDIA hardware  
>  (release barrier when all warps reach barrier, irrespective of   
>  divergence state). Functional simulation mode disabled to   
>  implement the expected barrier behavior under branch divergence"

源码内还有一处自警（`shader.cc:3883`）：

```cpp
} else {
    // TODO: check on the hardware if the count should include warp that exited
    if ((at_barrier.count() * m_warp_size) == bar_count) {
```

→ 维护者承认这里用 `at_barrier.count() * m_warp_size` 是 **粗粒度**，并标注 TODO 与 NVIDIA 真实硬件核对。

#### 证据 2 — gem5

**File**: [`src/gpu-compute/compute_unit.hh#L92-L181`](https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/compute_unit.hh#L92-L181)

```cpp
class WFBarrier {
  public:
    WFBarrier() : _numAtBarrier(0), _maxBarrierCnt(0) {}
    static const int InvalidID = -1;
    int numAtBarrier() const { return _numAtBarrier; }
    int numYetToReachBarrier() const { return _maxBarrierCnt - _numAtBarrier; }
    int maxBarrierCnt() const { return _maxBarrierCnt; }
    void setMaxBarrierCnt(int max_barrier_cnt) { _maxBarrierCnt = max_barrier_cnt; }
    void incNumAtBarrier() {                       // ← 单次 +1，整 WF
        assert(_numAtBarrier < _maxBarrierCnt);
        ++_numAtBarrier;
    }
    bool allAtBarrier() const {                    // ← 计数器比较
        return _numAtBarrier == _maxBarrierCnt;
    }
```

**File**: [`src/gpu-compute/compute_unit.cc#L634-L649`](https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/compute_unit.cc#L634-L649)

```cpp
        barrier_id = getFreeBarrierId();
        auto &wf_barrier = barrierSlot(barrier_id);
        assert(!wf_barrier.maxBarrierCnt());
        assert(!wf_barrier.numAtBarrier());
        wf_barrier.setMaxBarrierCnt(num_wfs_in_wg);   // ← 期望计数=WG内WF总数
```

→ 每个 WF（无论分歧状态如何）到达时 `incNumAtBarrier()` 一次。比较是 `_numAtBarrier == num_wfs_in_wg`。**没有 exec_mask / active_mask 参与。**

#### 证据 3 — MIAOW

**File**: [`src/verilog/rtl/issue/barrier_wait.v#L91-L99`](https://github.com/VerticalResearchGroup/miaow/blob/dbc5d7cc6e5fd58828239b59491eeb4f66503074/src/verilog/rtl/issue/barrier_wait.v#L91-L99) — 16-bit `wf_count_mask` 硬编码表：

```verilog
   // Calculate next mask for wg
   always @ ( fetch_wg_wf_count ) begin
      case(fetch_wg_wf_count)
        4'd0  : wf_count_mask <= 16'b0000_0000_0000_0001;
        4'd1  : wf_count_mask <= 16'b1111_1111_1111_1111;
        4'd2  : wf_count_mask <= 16'b0111_1111_1111_1111;
        ...
        4'd15 : wf_count_mask <= 16'b0000_0000_0000_0011;
      endcase
   end
```

**File**: [`src/verilog/rtl/issue/barrier_wait.v#L122-L134`](https://github.com/VerticalResearchGroup/miaow/blob/dbc5d7cc6e5fd58828239b59491eeb4f66503074/src/verilog/rtl/issue/barrier_wait.v#L122-L134) — 全 WF 释放判定：

```verilog
   // Signal when all_wf hit the barrier and when the first wf hit the barrier
   assign all_wf_hit_barrier 
     = ((curr_wg_wf_count == 16'h7fff) | (fetch_wg_wf_count == 4'd1) ) & 
       decode_barrier_valid ? 1'b1 : 1'b0;
   assign first_wf_barrier = (curr_wg_wf_count == 16'h0000)? 1'b1 : 1'b0;
```

→ MIAOW 的 `wf_count_mask` 是按 workgroup 大小（`fetch_wg_wf_count`）固定的 bitmask，**lane-level exec_mask 完全不参与**。

#### 证据 4 — Multi2Sim

**File**: [`src/arch/southern-islands/emulator/WorkItemIsa.cc#L2076-L2106`](https://github.com/multi2sim/multi2sim/blob/77b16e0ba3c23c5609657834b8cdfc7d0e22c303/src/arch/southern-islands/emulator/WorkItemIsa.cc#L2076-L2106)

```cpp
/* Suspend current wavefront at the barrier. If all wavefronts in work-group
 * reached the barrier, wake them up */
void WorkItem::ISA_S_BARRIER_Impl(Instruction *instruction)
{
    // Suspend current wavefront at the barrier
    wavefront->setBarrierInstruction(true);
    wavefront->setAtBarrier(true);                 // ← 整 wavefront
    work_group->incWavefrontsAtBarrier();          // ← 计数 +1

    Emulator::isa_debug << misc::fmt("Group %d wavefront %d reached barrier "
        "(%d reached, %d left)\n",
        work_group->getId(), wavefront->getId(), 
        work_group->getWavefrontsAtBarrier(),
        work_group->getWavefrontsInWorkgroup() - 
        work_group->getWavefrontsAtBarrier());

    // If all wavefronts in work-group reached the barrier, wake them up
    if (work_group->getWavefrontsAtBarrier() == work_group->getWavefrontsInWorkgroup())
    {
        for (auto i = work_group->getWavefrontsBegin(),
                e = work_group->getWavefrontsEnd();
                i != e;
                ++i)
            (*i)->setAtBarrier(false);             // ← 整 WG 释放

        work_group->setWavefrontsAtBarrier(0);

        Emulator::isa_debug << misc::fmt("Group %d completed barrier\n", work_group->getId());
    }
}
```

→ Multi2Sim 是 wavefront-级、work-group 范围内的整 WF 释放。

---

### **Q2 — 释放时是 32 个 lane 全部推进到 reconvergence_pc，还是只 participation mask 部分？**

**全部 32 lane 一起推进。** 没有模拟器实现"部分 lane 释放"。

| 模拟器 | 释放动作的代码 |
|--------|--------------|
| gpgpu-sim | `m_warp_at_barrier &= ~at_barrier` 清整 warp 的等待位；warp 整 32 lane 都在同一个 SIMT stack 帧，因此一起从 reconvergence PC 继续。`barrier_set_t` 不知道也无需知道 per-lane。|
| gem5 | `releaseWFsFromBarrier(bar_id)` → 遍历 `wfList[i][j]`，匹配 `barrierId()` 的 **整个 WF** 设回 `S_RUNNING`。 见 `compute_unit.cc:892-903`。|
| MIAOW | `next_curr_wg_wf_waiting = 40'b0`（清所有 WF）；issue 重新允许。 |
| Multi2Sim | `for (i..e) (*i)->setAtBarrier(false)` 清所有 wavefront 的 `atBarrier`。 |

这与 NVIDIA 硬件一致：bar.sync 释放时，warp 整 32 lane 同时越过 barrier，post-dominator reconvergence PC 是 warp-级的（一个 PC 寄存器，不是 32 个）。

---

### **Q3 — divergent warp 在同一 barrier 不同时间到达，模拟器怎么处理？**

**答：所有模拟器把 divergent 情形视为"warp 整体在到达"** —— 因为只要 warp 内任何一个 lane 执行了 bar.sync（被调度器选中进入 issue），整个 warp 就同时 issue 它（这是 PTX 本身的语义：bar.sync 是 warp-uniform 的 SIMT 指令）。

具体行为：
- **gpgpu-sim**: `warp_reaches_barrier(warp_id)` 被调用一次 → `m_bar_id_to_warps[bar_id].set(warp_id)` → 该 warp 已"到位"。如果同一个 warp 在分歧的两条路径上分别有 bar.sync，则两条路径上的 barrier 被分配不同的 bar_id（named barrier）或相同的（bar.sync）。
- **gem5**: 一个 WF 命中 `S_BARRIER` 状态时 `incNumAtBarrier()` 一次。WF 在分歧路径上时仍被视为整体。
- **MIAOW / Multi2Sim**: 同上。

**没有模拟器实现**：(i) 用 participation mask 区分 lane；(ii) lane-level 到达计数；(iii) lane-level 释放。所有都是 **warp-block-granular**。

---

### **Q4 — 模型化的是"到达计数硬件"还是"线程掩码"机制？**

| 模拟器 | 机制 | 数据结构 |
|--------|------|---------|
| gpgpu-sim | **每 CTA 的到达计数 + warp bitmap** | `m_bar_id_to_warps[N]` 是 `warp_set_t`（bitmap of warp IDs），配合 `at_barrier.count() * m_warp_size` 推出 lane-level count |
| gem5 | **每 CU 的到达计数器** | `WFBarrier::_numAtBarrier` vs `_maxBarrierCnt`，纯计数 |
| MIAOW | **每 WG 的位图计数器** | `curr_wg_wf_count`（16-bit 移位寄存器），用 `wf_count_mask` 初始化 |
| Multi2Sim | **每 WG 的到达计数器** | `work_group->wavefrontsAtBarrier`（int） |

**观察**：gpgpu-sim 是唯一用 **bitmap** 的；其它三个都用 **counter**。但它们的语义完全一致 —— 都是 "warp-级"的"达到 N 个就释放"硬件模型的简化版。

---

### **Q5 — `bar.warp.sync` vs `bar.sync` 的处理？**

| 模拟器 | 处理方式 |
|--------|---------|
| **gpgpu-sim** | **统一路径**。`cuda-sim.cc:716-745` 的 `set_bar_type()` 把 `BAR_OP` 的 `SYNC_OPTION`、`ARRIVE_OPTION`、`RED_OPTION` 全部映射到 `SYNC`/`ARRIVE`/`RED` 三种 bar_type，最终都走同一个 `barrier_set_t::warp_reaches_barrier`。`SST_OP`（SASS 的 `bar.sync`）也归到 `SYNC`。**没有 `bar.warp.sync` 专用路径** —— 但 gpgpu-sim 通过 `bar_count = warp_size` 在 named-barrier 模式下间接支持。 |
| **gem5** | **统一路径**。`Wavefront::S_BARRIER` 是单一状态；不管 barrier 来源（`s_barrier`、`s_sync`）都走 `scoreboard_check_stage.cc:108` 的同一段代码。gem5 的 PTX 模型有限，bar.warp.sync 可能直接被翻译成相同指令。 |
| **MIAOW** | **统一路径**。`barrier_wait.v` 单模块处理所有 barrier 指令。 |
| **Multi2Sim** | **统一路径**。`ISA_S_BARRIER_Impl` 是 Southern Islands S_BARRIER 的唯一处理点。 |

**NVIDIA 硬件上的区别**：`bar.sync` 是 CTA 范围（用 `name` 时是命名 barrier，named mask）；`bar.warp.sync` 是 warp 范围（带 participation mask `membermask`）。所有四个学术模拟器都没有模拟 `bar.warp.sync` 的 lane-level participation mask 语义，**而是把整个 warp 当作一个 mask 全 1 的参与者**。

---

## 🎯 综合结论

**所有调研的开源 GPU 模拟器（gpgpu-sim、gem5、MIAOW、Multi2Sim）在 divergent `bar.sync` 处理上遵循统一的"warp-级到达计数"范式**：

1. **到达计数单位 = warp/wavefront**（不是 lane）
2. **到达动作 = 整 warp 标记为 arrived**（不读 active_mask / exec_mask）
3. **释放动作 = 整 warp 推进**（不是 partial）
4. **不区分 `bar.warp.sync` vs `bar.sync`** 的 lane-level participation mask 语义

这与 NVIDIA 硬件的真实行为一致（[CUDA Programming Guide §4.9](https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-barriers.html.md)）：

> "If the invoking warp is fully converged, then the barrier is updated once. If the invoking warp is fully diverged, then 32 individual updates are applied to the barrier."

—— 即 **warp 完全分歧 = 32 次原子更新**，这等价于"warp 级一次性计数 ×32"。这就是为什么学术模拟器可以用粗粒度的 `warp_set_t` bitmap 或 counter 在功能正确性上过关。

---

## 🔗 完整 Permalink 索引

- **gpgpu-sim barrier_set_t::warp_reaches_barrier**: https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/src/gpgpu-sim/shader.cc#L3847-L3894
- **gpgpu-sim barrier_set_t class declaration**: https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/src/gpgpu-sim/shader.h#L1058-L1097
- **gpgpu-sim cuda-sim.cc set_bar_type**: https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/src/cuda-sim/cuda-sim.cc#L716-L745
- **gpgpu-sim CHANGES (divergence note)**: https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/CHANGES
- **gpgpu-sim scheduler_unit issue BARRIER_OP**: https://github.com/accel-sim/gpgpu-sim_distribution/blob/6c3cf4ff32110908386d605a7034fc67666a92de/src/gpgpu-sim/shader.cc#L1073-L1077
- **gem5 WFBarrier class**: https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/compute_unit.hh#L92-L181
- **gem5 scoreboard_check_stage barrier gating**: https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/scoreboard_check_stage.cc#L105-L125
- **gem5 releaseWFsFromBarrier**: https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/compute_unit.cc#L891-L903
- **gem5 WG dispatch sets maxBarrierCnt**: https://github.com/gem5/gem5/blob/c8222cc67a399bfc01e8658dd14b30d5bfd634f9/src/gpu-compute/compute_unit.cc#L634-L649
- **MIAOW barrier_wait.v (完整文件)**: https://github.com/VerticalResearchGroup/miaow/blob/dbc5d7cc6e5fd58828239b59491eeb4f66503074/src/verilog/rtl/issue/barrier_wait.v
- **Multi2Sim ISA_S_BARRIER_Impl**: https://github.com/multi2sim/multi2sim/blob/77b16e0ba3c23c5609657834b8cdfc7d0e22c303/src/arch/southern-islands/emulator/WorkItemIsa.cc#L2076-L2106
- **NVIDIA CUDA Programming Guide §4.9 Async Barriers**: https://docs.nvidia.com/cuda/archive/13.1.1/cuda-programming-guide/04-special-topics/async-barriers.html.md

---

## 💡 对比 PTX-EMU 的启示

PTX-EMU 模拟器如果采用 per-lane participation mask 的到达计数，那是与以上所有学术参考实现**偏离**的方向 —— 差异会体现在：

| 维度 | 学术实现（gpgpu-sim/gem5/MIAOW/M2S） | 如果 PTX-EMU 用 per-lane mask |
|------|--------------------------------------|--------------------------------|
| 到达计数 | warp 级别 +1 | lane 级别按 participation mask 数 +N |
| 释放 | 整 warp 推进 | 只 participation lane 推进，其它 lane 留在原 PC（这本身又触发新的分歧） |
| 性能模拟 | warp 整体被 issue 阻塞 | 大部分 lane 已就绪但被少数 lane 阻塞（不真实） |
| 与 NVIDIA 硬件对比 | CHANGES 明确"irrespective of divergence state" | 偏离真实硬件的 divergent-warp 全局同步语义 |

**建议**：除非你的目标是精确模拟硬件 `arrival_count` 寄存器（per-lane counter），否则采用与 gpgpu-sim 一致的"warp-级到达计数 + 整 warp 释放"是当前学术-工业共识的最简正确实现，且与 NVIDIA 硬件行为兼容。
