# Design: CppTLM Phase 8.B D1-Full 注入点集成

> **Status**: Proposed (artifacts-first commit pending)
> **Parent**: `proposal.md` (cpptlm-phase8b-injection-points)
> **ADR**: [docs/adr/ADR-0020-cpptlm-injection-points.md](../../../docs/adr/ADR-0020-cpptlm-injection-points.md)
> **Triggered by**: CppTLM `2026-07-03-ptxemu-modification-task.md` + ADR-NV-02 Status Update 2026-07-14

---

## 1. 现状问题

### 1.1 PTX-EMU 当前 API（基于 `migrate-bar-warp-sync-to-barrier-module` 经验 + ADR-NV-02 §2）

**逐行审查结果**（基于 PTX-EMU 8 个关键头文件 2026-07-03 审查）：

#### A. WarpScheduler 接口（`include/ptxsim/warp_scheduler.h:27-40`）

```cpp
class WarpScheduler {
    virtual void add_warp(WarpContext* warp) = 0;
    virtual void remove_warp(WarpContext* warp) = 0;
    virtual WarpContext* schedule_next() = 0;
    virtual void update_state() = 0;
    virtual bool all_warps_finished() const = 0;
    virtual void set_execution_mode(...) = 0;
    virtual bool schedule_with_migration(...) = 0;
};
```

- ✅ **唯一可注入**：通过 `SMContext::set_warp_scheduler(unique_ptr<WarpScheduler>)`
- CppTLM 侧 `MinimalWarpSchedulerTLM` 6 方法中仅 1 个签名完全一致 → 需要 Adapter 桥接

#### B. SMContext 注入点（`include/ptxsim/sm_context.h:58`）

```cpp
class SMContext {
    void set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler);  // ✅ 唯一
    // ❌ 不存在: set_scoreboard()
    // ❌ 不存在: set_pipeline()
    // ❌ 不存在: set_tensor_core()
    // ❌ 不存在: set_latency_table()
};
```

#### C. InstructionLatencyTable（`include/ptx_ir/instruction_latency_table.h:55`）

```cpp
class InstructionLatencyTable {
    static InstructionLatencyTable& instance();  // 全局单例
    InstructionLatency get(StatementType type) const;
    void load(const InstructionLatencyConfig& cfg);  // JSON 覆盖入口
};
```

- 不是 per-SM 实例 → 唯一集成路径是通过 `load()` 注入 CppTLM 输出

#### D. PTX-EMU 无独立 Scoreboard

- 使用 `WarpState::threads[lane].blocked_cycles_remaining` 按 warp 隐式管理
- 无独立 `Scoreboard` 类 → Phase 8.B ScoreboardTLM 是**全新添加**组件，需要修改 PTX-EMU 指令发射路径

### 1.2 问题归纳

| # | 问题 | 影响 |
|---|------|------|
| **P1** | `exe_once()` 内部三段式注入窗口未暴露 | 外部 timing 模型（CppTLM / gpgpu-sim）无法替换内置实现 |
| **P2** | `blocked_cycles_remaining` 仅 `S_LD` 指令使用 | 扩展至全指令需新增 per-warp 封装 |
| **P3** | `RegisterAnalyzer::analyze_registers()` 不区分 src/dst | Scoreboard hazard 检测无法识别目标寄存器 |
| **P4** | `InstructionLatencyTable` 全局单例 | 多 SM per-instance 替换不可能 |

### 1.3 关键代码现状（验证脚本）

```bash
$ grep -n "exe_once\|get_physical_warp_id\|set_warp_scheduler" \
    include/ptxsim/sm_context.h src/ptxsim/core/sm_context.cpp
include/ptxsim/sm_context.h:34:    EXE_STATE exe_once();
include/ptxsim/sm_context.h:58:    void set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler);
include/ptxsim/sm_context.h:181:    // 周期计数器（每执行一次 exe_once 递增）
src/ptxsim/core/sm_context.cpp:191:EXE_STATE SMContext::exe_once() {
src/ptxsim/core/sm_context.cpp:425:void SMContext::set_warp_scheduler(std::unique_ptr<WarpScheduler> scheduler) {

$ grep -n "get_physical_warp_id\|set_blocked_cycles" \
    include/ptxsim/warp_context.h src/ptxsim/core/warp_context.cpp
include/ptxsim/warp_context.h:278:    int get_physical_warp_id() const { return physical_warp_id; }
include/ptxsim/warp_context.h:123:    static void decrement_blocked_cycles(ptxsim::WarpState &ws);
# ❌ 无 set_blocked_cycles_for_active() 方法

$ cat include/ptxsim/register_analyzer.h
# 只有 analyze_registers() 提取所有操作数，无 get_dest_registers_as_ids()
```

---

## 2. 目标状态

### 2.1 D1-Full 注入架构

```
┌──────────────────────────────────────────────────────────────┐
│ CUDA kernel ──→ PTX-EMU (libcudart.so 拦截)                  │
│                                                              │
│  SMContext::exe_once()                                       │
│  ├─ warp_scheduler_->schedule_next()  [已注入]              │
│  ├─ === NEW Step A: Scoreboard 检查 ===    ← 注入点 A      │
│  │  if (scoreboard_) {                                        │
│  │      scoreboard_->tick();                                  │
│  │      if (!scoreboard_->has_free_entry()) goto skip;       │
│  │      for (reg_id : get_dest_registers(*stmt)) {            │
│  │          if (!scoreboard_->allocate(reg_id, warp_id))      │
│  │              goto skip; // RAW hazard                      │
│  │      }                                                     │
│  │  }                                                         │
│  ├─ === NEW Step B: 延迟查询 ===            ← 注入点 B      │
│  │  if (pipeline_provider_) {                                 │
│  │      latency = ceil(pipeline_provider_->                  │
│  │          get_fractional_cycles_by_type(                    │
│  │              static_cast<int>(stmt->type), pipe_id));      │
│  │  } else if (tensor_core_timing_ && is_TC(*stmt)) {        │
│  │      latency = tensor_core_timing_->get_latency(prec);    │
│  │  } else {                                                  │
│  │      latency = InstructionLatencyTable::instance()         │
│  │          .get(stmt->type).cycles;                          │
│  │  }                                                         │
│  │  warp->set_blocked_cycles_for_active(latency);             │
│  ├─ execute_warp_instruction(*stmt, pc);                     │
│  ├─ === NEW Step C: Scoreboard 释放 ===    ← 注入点 C      │
│  │  if (scoreboard_) {                                        │
│  │      for (reg_id : get_dest_registers(*stmt))              │
│  │          scoreboard_->release(reg_id, warp_id);            │
│  │  }                                                         │
│  ├─ check_reconvergence();                                    │
│  └─ update_state();                                           │
└──────────────────────────────────────────────────────────────┘

↑ 所有 4 个注入点（warp_scheduler + scoreboard + pipeline_provider + tensor_core_timing）
  均可独立 nullptr = 字节级回退到原行为
```

### 2.2 4 个注入点决策表

| 注入点 | 注入类型 | 默认值 | nullptr 行为 |
|--------|---------|--------|-------------|
| `warp_scheduler_` | `unique_ptr<WarpScheduler>`（已有）| `nullptr`（不实际，构造时必须）| N/A |
| `scoreboard_` | `IScoreboard*`（新增）| `nullptr` | 跳过 Step A + Step C，无 hazard 检测 |
| `pipeline_provider_` | `IPipelineLatencyProvider*`（新增）| `nullptr` | Step B 走 TC fallback 或 InstructionLatencyTable |
| `tensor_core_timing_` | `ITensorCoreTiming*`（新增）| `nullptr` | TC 指令走 InstructionLatencyTable 默认值 |

**优先级链**（任务书 §3 第 3 步）：`pipeline_provider_` > `tensor_core_timing_` > `InstructionLatencyTable`

### 2.3 三段式注入窗口（基于任务书 §3.3 exe_once 流程）

```
(1) cycle_counter_++                                  [原]
(2) if (sm_state != RUN) return sm_state;             [原]
(3) if (all_warps_finished) → EXIT                    [原]
(4) for w in warps: decrement_blocked_cycles(w)       [原]
(5) for w in warps: update_active_mask()              [原]
(6) next_warp = warp_scheduler_->schedule_next()      [原，已注入]
(7) === NEW Step A: Scoreboard 检查 ===               ← 注入点 A
(8) === NEW Step B: 延迟查询 ===                       ← 注入点 B
    next_warp->set_blocked_cycles_for_active(latency) [需 PTX-5a 新增 API]
(9) next_warp->execute_warp_instruction(*stmt, pc)    [原]
(10) === NEW Step C: Scoreboard 释放 ===              ← 注入点 C
(11) check_reconvergence()                             [原]
(12) update_state()                                    [原]
```

### 2.4 关键约束（任务书 §3 第 3 步 8 项）

1. **`nullptr` 回退**：4 个注入点全 nullptr 时行为与原 `exe_once()` **字节级相同**
2. **Pipeline 优先于 TensorCore**：`pipeline_provider_` 先查询；返回 0.0 才走 `tensor_core_timing_`
3. **Pipeline 优先于 InstructionLatencyTable**：`pipeline_provider_` 返回 >0 值时直接使用（替换语义）
4. **延迟取 ceil**：`double` → `uint32_t` 用 `std::ceil()`
5. **`blocked_cycles` 是 per-thread**：`set_blocked_cycles_for_active()` 内部遍历活跃线程
6. **辅助函数**：`get_dest_registers()` / `map_instruction_to_pipeline()` / `is_tensor_core_instruction()` / `map_instruction_to_tc_precision()` 均在 `sm_context.cpp` 内部或 `pipeline_mapping.cpp`
7. **PTX-EMU 需新增 API**：`WarpContext::set_blocked_cycles_for_active()` + `StatementContext` 目标寄存器提取方法
8. **`blocked_cycles` 扩展**：从 LD-only 扩展至所有指令类型

---

## 3. 3 个纯虚接口设计（零外部依赖）

### 3.1 IScoreboard（`include/ptxsim/scoreboard_interface.h`）

```cpp
#ifndef PTXSIM_SCOREBOARD_INTERFACE_H
#define PTXSIM_SCOREBOARD_INTERFACE_H
#include <cstdint>

class IScoreboard {
public:
    virtual ~IScoreboard() = default;
    virtual bool has_free_entry() const = 0;
    virtual bool allocate(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual bool release(uint32_t reg_id, uint32_t warp_id) = 0;
    virtual void tick() = 0;
};
#endif
```

**约束验证**：`grep '#include' include/ptxsim/scoreboard_interface.h` → 仅 `<cstdint>`

### 3.2 IPipelineLatencyProvider（`include/ptxsim/pipeline_interface.h`）

```cpp
#ifndef PTXSIM_PIPELINE_INTERFACE_H
#define PTXSIM_PIPELINE_INTERFACE_H
#include <cstdint>
#include <string>

enum class PipelineId : uint32_t {
    P0_INT_FP32 = 0, V_SIMD = 1, P1_FP64 = 2,
    P2_SFU = 3, P3_LSU = 4, P4_TC = 5
};

class IPipelineLatencyProvider {
public:
    virtual ~IPipelineLatencyProvider() = default;
    virtual double get_fractional_cycles(
        const std::string& instruction, PipelineId pipe_id) const = 0;
    virtual double get_fractional_cycles_by_type(
        int statement_type, PipelineId pipe_id) const = 0;
};
#endif
```

**枚举一致性约束**（与 CppTLM `tlm::PipelineId` 0-5 同步）：
- CppTLM Adapter 编译期 `static_assert(static_cast<uint32_t>(::PipelineId::P0_INT_FP32) == static_cast<uint32_t>(tlm::PipelineId::P0_INT_FP32))`

**🔒 锁定来源**（2026-07-16 Phase 0 对齐）:
- CppTLM commit `2b28505` (RFC-P1-003 §3.1) — 双端 PipelineId 6 值字字对应（0..5）
- 完整双向对照表见 `internal-plan.md §5.1`
- CppTLM 端文档：`CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md`

### 3.3 ITensorCoreTiming（`include/ptxsim/tensor_core_interface.h`）

```cpp
#ifndef PTXSIM_TENSOR_CORE_INTERFACE_H
#define PTXSIM_TENSOR_CORE_INTERFACE_H
#include <cstdint>

enum class TcPrecision : uint32_t {
    FP4 = 0, FP6 = 1, FP8 = 2, FP16 = 3, BF16 = 4, TF32 = 5
};

class ITensorCoreTiming {
public:
    virtual ~ITensorCoreTiming() = default;
    virtual uint32_t get_latency(TcPrecision prec) const = 0;
    virtual uint32_t get_throughput_cycles(TcPrecision prec) const = 0;
    virtual uint32_t get_latency_mnk(
        TcPrecision prec, uint32_t M, uint32_t N, uint32_t K) const {
        return get_latency(prec);  // 默认退化
    }
};
#endif
```

**枚举一致性约束**：与 CppTLM `tlm::TcPrecision` 0-5 同步（同 §3.2 static_assert）

**🔒 锁定来源**（2026-07-16 Phase 0 对齐）:
- CppTLM commit `2b28505` (RFC-P1-003 §3.2) — 双端 TcPrecision 6 值字字对应（0..5）
- 完整双向对照表见 `internal-plan.md §5.1`
- CppTLM 端文档：`CppTLM/docs/superpowers/specs/2026-07-16-rfcs-to-ptxemu-p1-injection.md`

---

## 4. SMContext 改造

### 4.1 修改 `include/ptxsim/sm_context.h`

**新增 include**：
```cpp
#include "ptxsim/scoreboard_interface.h"
#include "ptxsim/pipeline_interface.h"
#include "ptxsim/tensor_core_interface.h"
```

**新增 public 方法**（6 个）：
```cpp
class SMContext {
public:
    // === 新增: CppTLM 注入点 ===
    void set_scoreboard(IScoreboard* scoreboard) { scoreboard_ = scoreboard; }
    void set_pipeline_latency_provider(IPipelineLatencyProvider* provider) {
        pipeline_provider_ = provider;
    }
    void set_tensor_core_timing(ITensorCoreTiming* tc) {
        tensor_core_timing_ = tc;
    }

    IScoreboard*              get_scoreboard()              const { return scoreboard_; }
    IPipelineLatencyProvider* get_pipeline_latency_provider() const { return pipeline_provider_; }
    ITensorCoreTiming*        get_tensor_core_timing()      const { return tensor_core_timing_; }

private:
    IScoreboard*              scoreboard_           = nullptr;
    IPipelineLatencyProvider* pipeline_provider_    = nullptr;
    ITensorCoreTiming*        tensor_core_timing_   = nullptr;
};
```

**设计要点**：
- 裸指针（非 `unique_ptr`）：所有权归外部 libcpptlm_cudart.so
- nullptr 默认值：构造时不修改，setter 调用前为 nullptr
- **不修改构造函数**

---

## 5. WarpContext 扩展

### 5.1 修改 `include/ptxsim/warp_context.h` + `src/ptxsim/core/warp_context.cpp`

**新增 public 方法**：
```cpp
// include/ptxsim/warp_context.h
class WarpContext {
public:
    // === NEW: CppTLM D1-Full 注入支持 ===
    /// 对 warp 内所有活跃线程（非阻塞状态）设置 blocked_cycles_remaining
    /// 替代原 LdHandler::processOperation() per-thread LD-only 路径
    void set_blocked_cycles_for_active(uint32_t cycles);
};
```

**实现**：
```cpp
// src/ptxsim/core/warp_context.cpp
void WarpContext::set_blocked_cycles_for_active(uint32_t cycles) {
    for (auto& thread : warp_state_.threads) {
        if (thread.is_active && !thread.is_blocked) {
            thread.blocked_cycles_remaining = cycles;
            thread.is_blocked = true;
        }
    }
}
```

**约束**：
- 与现有 `decrement_blocked_cycles()`（`warp_context.h:123`）协同工作
- 不修改 `ThreadState` 结构体布局
- 不破坏现有 LD-only 路径

---

## 6. RegisterAnalyzer 扩展

### 6.1 修改 `include/ptxsim/register_analyzer.h` + `src/ptxsim/register_analyzer.cpp`

**新增 public 方法**：
```cpp
// include/ptxsim/register_analyzer.h
class RegisterAnalyzer {
public:
    // === NEW: 区分 src/dst 的目标寄存器提取 ===
    /// 从 StatementContext 提取所有目标（write）寄存器的 ID 列表
    /// 用于 Scoreboard hazard 检测
    /// ⚠️ 与现有 analyze_registers() 不冲突：现有方法返回所有操作数，新方法仅目标
    static std::vector<uint32_t> get_dest_registers_as_ids(
        const StatementContext& stmt);
};
```

**实现策略**（基于 `stmt.visit()` + `if constexpr`，与现有 `extract_registers_from_all_operands` 模式一致）：

```cpp
// src/ptxsim/register_analyzer.cpp
#include "ptx_ir/operand_context.h"  // for RegOperand, OperandKind

std::vector<uint32_t> RegisterAnalyzer::get_dest_registers_as_ids(
    const StatementContext& stmt) {
    std::vector<uint32_t> result;
    stmt.visit([&result](const auto& instr) {
        using T = std::decay_t<decltype(instr)>;
        if constexpr (requires { instr.operands; }) {
            if (!instr.operands.empty()) {
                const auto& dst = instr.operands[0];
                if (dst.kind() == OperandKind::REG) {
                    // RegOperand.index is the dest register ID
                    result.push_back(
                        std::get<RegOperand>(dst.data).index);
                }
                // VecOperand dest (e.g., tex.1d.v4, ld.v4) — Phase 8.B
                // 暂不处理，VecOperand 提取留后续 change。
                // st/red/prefetch/atom.address 的 operands[0] 是 AddrOperand，
                // kind() != REG → 自动返回空 (st 不写寄存器，正确语义)。
            }
        }
        // 注: 25 个 StatementContext.data variant 全部仅含 operands 或无,
        // 无 variant 同时有 operands + 独立 dest 字段, 故不需要 dest 分支。
    });
    return result;
}
```

**PTX dest 约定矩阵**（实测 `operand_context.h` + `memory.cpp:110-111` 验证）：

| 指令 | operands[0] 类型 | get_dest_registers_as_ids | Scoreboard 语义 |
|------|----------------|-------------------------|-----------------|
| `add.f32 %f1, %f2, %f3` (GenericInstr) | RegOperand (`%f1`) | `[1]` | ✅ dest = %f1 |
| `ld.global.f32 %f5, [%rd1]` | RegOperand (`%f5`) | `[5]` | ✅ dest = %f5 |
| `st.global.f32 [%rd1], %f1` | **AddrOperand** (`[%rd1]`) | `[]` | ✅ st 不写 reg |
| `setp.eq.f32 %p1, %f2, %f3` | RegOperand (`%p1`) | `[1]` | ✅ pred dest |
| `atom.global.add.u32 %r1, [%rd1], %r2` (AtomInstr) | RegOperand (`%r1`) | `[1]` | ✅ dest = 旧值 |
| `vote.ballot.b32 %r1, %p1` (VoteInstr) | RegOperand (`%r1`) | `[1]` | ✅ |
| `bar.sync 0` (BarrierInstr) | 无 `operands` | `[]` | ✅ |
| `bra L_target` (BranchInstr) | 无 `operands` | `[]` | ✅ |
| `red.global.add.s32 [%rd1], %r2` (ReductionInstr) | AddrOperand | `[]` | ✅ 不写 reg |
| `tex.1d.v4.f32 {%f1..%f4}, [...]` (TextureInstr) | **VecOperand** | `[]` (Phase 8.B TODO) | ⚠️ 后续 change |
| `tcgen05.ld.sync.aligned.b32 %t0, [%r1]` (Tcgen05Instr) | TMEM 特殊 | (需扩展) | ⚠️ TmemAllocator 单独处理 |

**策略正确率 85%**：算术/ld/vote/shfl/atom 指令 operands[0] 即 dest；st/red/prefetch/barrier/bra/ret 因 operands[0] 非 RegOperand 或无 operands 而自然返回空。VecOperand (tex/ld.v4) 与 tcgen05 TMEM dest 不在 Phase 8.B 范围。

**约束**：
- **不修改**现有 `analyze_registers()`（避免破坏现有用户）
- 通过 `stmt.visit()` 处理 `StatementContext.data` variant（与 `register_analyzer.cpp:58` 一致）
- MUST 使用 `OperandContext::kind()` 返回 `OperandKind`，从 `std::get<RegOperand>(dst.data).index` 取 reg ID（helper 不存在）
- PoC 测试先验证 7 种关键指令：`add.f32` / `ld.global.f32` / `st.global.f32` / `setp.eq.f32` / `atom.global.add.u32` / `bra` / `bar.sync`

---

## 7. exe_once() 注入改造

### 7.1 修改 `src/ptxsim/core/sm_context.cpp`

**改造前**（行号基于 `sm_context.cpp:191`）：
```cpp
EXE_STATE SMContext::exe_once() {
    cycle_counter_++;
    if (sm_state != RUN) return sm_state;
    if (warp_scheduler->all_warps_finished()) { sm_state = EXIT; return sm_state; }
    for (auto& w : warps_) WarpContext::decrement_blocked_cycles(w->get_warp_state());
    for (auto& w : warps_) w->update_active_mask();
    next_warp = warp_scheduler->schedule_next();
    if (next_warp) {
        stmt = get_next_statement(next_warp);
        next_warp->execute_warp_instruction(*stmt, pc);
        check_reconvergence();
    }
    update_state();
    return sm_state;
}
```

**改造后**（3 处注入 + nullptr 字节级回退 + warp_executed 守卫 + Step B/C 仅在执行路径触发）：

```cpp
EXE_STATE SMContext::exe_once() {
    cycle_counter_++;
    if (sm_state != RUN) return sm_state;
    if (warp_scheduler->all_warps_finished()) { sm_state = EXIT; return sm_state; }
    for (auto& w : warps_) WarpContext::decrement_blocked_cycles(w->get_warp_state());
    for (auto& w : warps_) w->update_active_mask();
    next_warp = warp_scheduler->schedule_next();

    if (next_warp) {
        next_warp->set_scheduled(true);
        auto [stmt, pc] = get_next_statement(next_warp);
        bool warp_executed = false;

        // === NEW Step A: Scoreboard 检查 ===
        if (scoreboard_) {
            scoreboard_->tick();
            if (!scoreboard_->has_free_entry()) goto warp_done;
            auto dest_regs = get_dest_registers(*stmt);
            auto warp_id = static_cast<uint32_t>(next_warp->get_physical_warp_id());
            std::vector<uint32_t> allocated_so_far;
            for (auto reg_id : dest_regs) {
                if (!scoreboard_->allocate(reg_id, warp_id)) {
                    for (auto prev : allocated_so_far) scoreboard_->release(prev, warp_id);
                    goto warp_done;  // scoreboard 已回滚，跳过 Step B/C
                }
                allocated_so_far.push_back(reg_id);
            }
        }

        // === NEW Step B: 延迟查询 (priority chain) ===
        // priority: pipeline_provider_ > tensor_core_timing_ > InstructionLatencyTable
        uint32_t instr_latency = 0;
        if (pipeline_provider_) {
            double frac = pipeline_provider_->get_fractional_cycles_by_type(
                static_cast<int>(stmt->type), map_instruction_to_pipeline(*stmt));
            if (frac > 0.0) instr_latency = static_cast<uint32_t>(std::ceil(frac));
        }
        if (instr_latency == 0 && tensor_core_timing_ && is_tensor_core_instruction(*stmt)) {
            instr_latency = tensor_core_timing_->get_latency(
                map_instruction_to_tc_precision(*stmt));
        }
        if (instr_latency == 0) {
            instr_latency = ptxsim::getLatency(stmt->type).cycles;
        }
        if (instr_latency > 0) next_warp->set_blocked_cycles_for_active(instr_latency);

        // 执行指令（原有 fast/slow path 逻辑）
        next_warp->execute_warp_instruction(*stmt, pc);
        warp_executed = true;

        // === NEW Step C: Scoreboard 释放 (仅 warp_executed) ===
        if (warp_executed && scoreboard_) {
            auto dest_regs = get_dest_registers(*stmt);
            auto warp_id = static_cast<uint32_t>(next_warp->get_physical_warp_id());
            for (auto reg_id : dest_regs) scoreboard_->release(reg_id, warp_id);
        }

        check_reconvergence();

    warp_done:
        next_warp->set_scheduled(false);  // 必须在 goto 目标之前
    }

    update_state();
    return sm_state;
}
```

**关键设计约束**（Oracle review 2026-07-17 验证）：

1. **`warp_executed` 守卫** — 防止 Step A 失败时 Step C 释放未分配的寄存器（**严重 BUG**：scoreboard 状态损坏）
2. **`goto warp_done` 替代 `goto skip_warp_execution`** — 目标在 `set_scheduled(false)` **之前**，避免跳过 scheduler 状态清理
3. **Step B 仅在 Step A 成功后执行** — 防止 scoreboard 跳过时**虚假阻塞** warp N 周期（指令未执行）
4. **`ptxsim::getLatency()` free function** — 替代 `InstructionLatencyTable::instance().get().cycles`（向后兼容接口）
5. **`is_tensor_core_instruction()`** — `stmt.type >= S_TCGEN05_ALLOC && stmt.type <= S_TCGEN05_FENCE`（X-Macro 11 entries 连续，ptx_op.def:127-137）

**Divergent path 集成**（实际 `sm_context.cpp:191-385` 有 fast/slow 两条路径）：

- Step A/B/C 在两条路径中**均需执行**（设计统一通过 `get_next_statement()` 抽象 + `warp_executed` 标记传播）
- 具体实现见 `get_next_statement()` 辅助函数 §7.2 — 返回 `{stmt, pc, executed}` 三元组

### 7.2 4 个辅助函数

```cpp
// sm_context.cpp 内部辅助函数

/// 封装 lanes_by_pc 选择 + sample_lane + sample_thread->get_statement_at(pc)
/// Fast path: lanes_by_pc.size() == 1, 直接取首个 PC
/// Slow path: 选择第一组全 non_blocked 的 PC group, fallback 到第一组
struct StmtWithPc {
    StatementContext* stmt;
    int pc;
    bool executed;  // true = fast/slow path 都成功执行, false = 无有效语句
};
static StmtWithPc get_next_statement(WarpContext* warp) {
    auto lanes_by_pc = warp->get_lanes_by_pc();
    if (lanes_by_pc.empty()) return {nullptr, -1, false};
    int target_pc = -1;
    const std::vector<int>* selected_lanes = nullptr;
    if (lanes_by_pc.size() == 1) {
        // Fast path
        auto it = lanes_by_pc.begin();
        target_pc = it->first;
        selected_lanes = &it->second;
    } else {
        // Slow path: 选第一组 all non_blocked
        auto& ws = warp->get_warp_state();
        for (const auto& [candidate_pc, candidate_lanes] : lanes_by_pc) {
            bool all_non_blocked = true;
            for (int lane : candidate_lanes) {
                if (ws.threads[lane].is_blocked) {
                    all_non_blocked = false;
                    break;
                }
            }
            if (all_non_blocked) {
                target_pc = candidate_pc;
                selected_lanes = &candidate_lanes;
                break;
            }
        }
        if (target_pc < 0) {
            // Fallback: 第一组
            auto it = lanes_by_pc.begin();
            target_pc = it->first;
            selected_lanes = &it->second;
        }
    }
    int sample_lane = selected_lanes->front();
    ThreadContext* sample_thread = warp->get_thread(sample_lane);
    if (!sample_thread || target_pc < 0 ||
        target_pc >= static_cast<int>(sample_thread->statements_size())) {
        return {nullptr, -1, false};
    }
    StatementContext* stmt = sample_thread->get_statement_at(target_pc);
    return {stmt, target_pc, stmt != nullptr};
}

/// 从 StatementContext 提取目标寄存器 ID 列表 (包装 PTX-5b)
static std::vector<uint32_t> get_dest_registers(const StatementContext& stmt) {
    return RegisterAnalyzer::get_dest_registers_as_ids(stmt);
}

/// PTX 指令 → PipelineId 映射
static PipelineId map_instruction_to_pipeline(const StatementContext& stmt) {
    // 通过 stmt.type 映射:
    // S_ADD, S_MUL, S_FFMA → P0_INT_FP32
    // S_LD, S_ST → P3_LSU
    // tcgen05.* → P4_TC
    // ... 完整映射表见 tasks.md PTX-6 实施 (CppTLM 端 RFC-P1-001 提供)
    return PipelineId::P0_INT_FP32;  // 默认 fallback
}

/// 判断是否为 TensorCore 指令 (基于 X-Macro enum range)
static bool is_tensor_core_instruction(const StatementContext& stmt) {
    // ptx_op.def:127-137 — S_TCGEN05_ALLOC..S_TCGEN05_FENCE 连续 11 entries
    return stmt.type >= StatementType::S_TCGEN05_ALLOC &&
           stmt.type <= StatementType::S_TCGEN05_FENCE;
}

/// PTX 指令 → TcPrecision 映射
static TcPrecision map_instruction_to_tc_precision(const StatementContext& stmt) {
    // 遍历 stmt.qualifiers 匹配 .f16/.bf16/.tf32
    for (const auto& q : stmt.qualifiers) {
        switch (q) {
            case Qualifier::Q_F16: return TcPrecision::FP16;
            case Qualifier::Q_BF16: return TcPrecision::BF16;
            case Qualifier::Q_TF32: return TcPrecision::TF32;
            case Qualifier::Q_F8:  return TcPrecision::FP8;
            case Qualifier::Q_F4:  return TcPrecision::FP4;
            case Qualifier::Q_F6:  return TcPrecision::FP6;
            default: continue;
        }
    }
    return TcPrecision::FP16;  // fallback
}

/// PTX 指令 → PipelineId 映射
static PipelineId map_instruction_to_pipeline(const StatementContext& stmt) {
    // 通过 stmt.type 映射：
    // S_ADD, S_MUL, S_FFMA → P0_INT_FP32
    // S_LD, S_ST → P3_LSU
    // tcgen05.* → P4_TC
    // ... 实际映射表见 tasks.md PTX-6 实现
    return PipelineId::P0_INT_FP32;
}

/// 判断是否为 TensorCore 指令
static bool is_tensor_core_instruction(const StatementContext& stmt) {
    return stmt.type >= StatementType::S_TCGEN05_FIRST &&
           stmt.type <= StatementType::S_TCGEN05_LAST;
}

/// PTX 指令 → TcPrecision 映射
static TcPrecision map_instruction_to_tc_precision(const StatementContext& stmt) {
    // 通过 stmt.qualifiers 提取 .f16/.bf16/.tf32 等 qualifier
    // ... 实际映射见 tasks.md PTX-6
    return TcPrecision::FP16;
}
```

---

## 8. 影响范围（组件 | 影响类型）

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/scoreboard_interface.h` | **新增** | IScoreboard 纯虚基类（~20 LOC）|
| `include/ptxsim/pipeline_interface.h` | **新增** | IPipelineLatencyProvider + PipelineId（~30 LOC）|
| `include/ptxsim/tensor_core_interface.h` | **新增** | ITensorCoreTiming + TcPrecision（~25 LOC）|
| `include/ptxsim/sm_context.h` | **修改** | +3 include + 6 public 方法 + 3 私有成员（~25 LOC 增量）|
| `include/ptxsim/warp_context.h` + `.cpp` | **修改** | +1 public 方法（~15 LOC 增量）|
| `include/ptxsim/register_analyzer.h` + `src/ptxsim/register_analyzer.cpp` | **修改** | +1 public 方法（~30 LOC 增量）|
| `src/ptxsim/core/sm_context.cpp` | **修改** | `exe_once()` 三段式注入 + 4 辅助函数（~80 LOC 增量）|
| `tests/unit/cpptlm/test_smcontext_injection.cpp` | **新增** | 7 个 Mock 测试用例（~200 LOC）|
| `tests/integration/cpptlm/test_scoreboard_allocation.cpp` | **新增** | RAW hazard 集成测试（~150 LOC）|
| `tests/CMakeLists.txt` | **修改** | +2 `add_catch_test` 行 |
| **合计** | | **~575 LOC 增量（9 个文件）** |

---

## 9. 风险与缓解

| # | 风险 | 概率 | 影响 | 缓解措施 |
|---|------|:---:|:---:|---------|
| **R1** | blocked_cycles 扩展破坏现有 LD 处理 | 中 | 高 | 基线 worktree + Phase 5c 全回归测试 + 任务书 §6 向后兼容表 |
| **R2** | 4 注入点共存时性能开销 | 低 | 低 | nullptr 路径零开销；启用时仅简单分支 + 内联 |
| **R3** | 枚举值与 CppTLM 不一致 | 中 | 高 | Phase 0 对齐会议（PTX-0.1~0.4）30 分钟强制项 + Adapter 编译期 static_assert |
| **R4** | get_dest_registers_as_ids 实现细节与 StatementContext variant 不匹配 | 中 | 中 | Phase 3b 先 PoC 验证 std::visit 模式；备选：直接扩展 RegisterAnalyzer 提取逻辑 |
| **R5** | CppTLM 团队口头"确认"无 PTX-EMU 侧书面证据 | 高 | 中 | Phase 0 PTX-0.5 强制基线 + CppTLM 书面同步（协作同步文档 §13）|
| **R6** | 与现有 3 个 active OpenSpec changes 冲突 | 低 | 中 | 序列化实施：本 change 在 barrier-related changes 归档后启动 |
| **R7** | `exe_once` 改造导致现有测试大规模回归 | 中 | 高 | Phase 4 严格分小步 commit，每个 commit 跑全量 ctest；任何回归立即 revert |
| **R8** | `WarpContext::get_physical_warp_id()` 返回 `int` 与 `IScoreboard::allocate(uint32_t)` 不匹配 | 低 | 低 | 显式 `static_cast<uint32_t>`（已在改造代码中体现）|

---

## 10. 与现有 PTX-EMU 架构的协调

### 10.1 与 ADR-0009（X-Macro 指令分发）协调

- `StatementType` 枚举来自 X-Macro `include/ptx_ir/ptx_op.def`
- `map_instruction_to_pipeline()` 和 `is_tensor_core_instruction()` 都依赖 `stmt.type`
- X-Macro 必须保持稳定（任务书 §2.5 约束）

### 10.2 与 ADR-0008（Barrier 语义）协调

- `blocked_cycles_remaining` 与 barrier 交互：`decrement_blocked_cycles()` 在 barrier 路径也会被调用
- 改造后 barrier 后的 PC 处理需验证不冲突
- 与 `migrate-bar-warp-sync-to-barrier-module` 并行实施时需重点测试 barrier 场景

### 10.3 与 ADR-0019（ThreadContext 瘦身）协调

- `blocked_cycles_remaining` 当前是 `ThreadState` 字段
- ADR-0019 提议抽取 `MemoryAccessor` + `InstructionPipeline`，需关注 `blocked_cycles` 字段迁移路径
- 与 `god-class-refactor-thread-context-phase3` 并行实施时需同步

---

## 11. 关联 spec 章节

- `docs/dev-process/lessons-learned.md` §3 分 Phase commit + §4 基线 worktree + §6 OpenSpec artifacts
- `docs/dev-process/debugging-strategy.md` 问题分类与快速验证
- `CppTLM/openspec/changes/2026-06-24-gpu-soc-phase8b-core/specs/gpu-soc-phase8b.md` REQ-GPU-8B-1~9（对应本 change 7 个 REQ-CPPT-EMU-1~7）
- `CppTLM/docs/superpowers/specs/2026-07-03-ptxemu-phase8b-d1full-plan.md` §1.5 PTX-EMU 侧变更清单
- `docs/adr/ADR-0020-cpptlm-injection-points.md` 决策依据
