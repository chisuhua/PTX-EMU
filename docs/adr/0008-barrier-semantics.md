# ADR-0008: Barrier 语义增强 - Convergence + Memory Fence

| 属性 | 值 |
|------|-----|
| **状态** | Active |
| **日期** | 2026-05-05 |
| **关联任务** | Phase 4 (Barrier 增强) |
| **作者** | PTX-EMU Team |

## 上下文

PTX 的 barrier 指令（`bar.sync`、`bar.warp.sync`）具有两个关键语义：

1. **同步语义**：所有参与的线程必须到达 barrier 后才能继续
2. **内存语义**：barrier 隐含 memory fence，barrier 前的所有内存写入对 barrier 后的所有线程可见

早期的简单 counting barrier 只实现了同步语义，缺少：

- 参与线程的精确跟踪（participation mask）
- Memory fence 验证
- 收敛点关联（barrier 后线程应恢复到哪个 PC）

## 决策驱动因素

1. **PTX ISA 规范要求**：PTX ISA 9.1 Section 9.7.13.1 明确 barrier 隐含 memory fence
2. **调试需求**：需要验证 barrier 语义是否被正确遵守
3. **SIMT 集成**：barrier 必须在 reconvergence point 之后，需要与 SIMT 栈协调

## 考虑的替代方案

### 方案 A: Counting Barrier（简单计数器）

**描述**: 维护一个计数器，每个到达的线程计数+1，计满后释放所有线程

**优点**:
- 实现简单
- 性能好

**缺点**:
- 无法验证参与线程是否正确
- 无法关联 reconvergence point
- 无法验证 memory fence

### 方案 B: Always-On Memory Fence 验证

**描述**: 每次 barrier 都验证所有前置内存操作已完成

**优点**:
- 完整保护
- 能检测所有语义错误

**缺点**:
- 性能开销 5-10%
- 需要跟踪所有内存操作

### 方案 C: Debug-Only Convergence Barrier (✅ 选中)

**描述**: Wbar 实现 convergence barrier + memory fence 验证，但验证仅在 PTX_DEBUG 模式下启用

**优点**:
- 开发时能检测语义错误
- 生产模式无性能影响
- 与 SIMT 栈集成良好

**缺点**:
- 生产模式无法检测 barrier 误用

**选择理由**: PTX-EMU 主要用于研究和教学，开发阶段的正确性验证比运行时性能更重要。Debug-only 验证平衡了两者。

## 决策内容

### 设计原则

1. **Participation Mask**：明确哪些线程必须参与 barrier
2. **Arrived Mask 跟踪**：记录哪些线程已到达
3. **Reconvergence PC 关联**：barrier 完成后线程恢复到的 PC
4. **Debug-Only 验证**：memory fence 验证仅在 PTX_DEBUG 模式下启用

### 实现要点

```cpp
class Wbar {
public:
    int barrier_id;
    int reconvergence_pc;           // barrier 后线程恢复的 PC
    uint32_t participation_mask;    // 必须参与的线程
    uint32_t arrived_mask;          // 已到达的线程
    
    // Memory fence 状态（仅调试用）
    bool is_initialized;
    bool memory_fence_verification_enabled;
    
    #ifdef PTX_DEBUG
    std::vector<std::pair<int, uint64_t>> pre_barrier_stores;  // barrier 前的内存操作
    #endif
    
    void init(int _reconvergence_pc, uint32_t _participation_mask) {
        reconvergence_pc = _reconvergence_pc;
        participation_mask = _participation_mask;
        arrived_mask = 0;
        is_initialized = true;
    }
    
    void arrive(int lane_id) {
        arrived_mask |= (1u << lane_id);
        #ifdef PTX_DEBUG
        // 记录 barrier 前的内存操作
        pre_barrier_stores.push_back({lane_id, get_last_store_addr(lane_id)});
        #endif
    }
    
    bool is_complete() const {
        return (arrived_mask & participation_mask) == participation_mask;
    }
    
    #ifdef PTX_DEBUG
    void verify_memory_fence() const {
        if (!memory_fence_verification_enabled) return;
        
        // 验证所有 barrier 前的 store 对参与线程可见
        for (const auto& store : pre_barrier_stores) {
            int lane_id = store.first;
            uint64_t addr = store.second;
            for (int i = 0; i < 32; i++) {
                if (participation_mask & (1u << i)) {
                    if (!is_store_visible(i, addr)) {
                        PTX_ERROR("Lane %d store to 0x%lx not visible to lane %d!",
                                  lane_id, addr, i);
                    }
                }
            }
        }
    }
    #endif
};
```

### Barrier 执行流程

```cpp
// barrier.cpp
void BarWarpSyncHandler::execute(ThreadContext* context, const BarrierInstr& instr) {
    WarpContext* warp_ctx = context->warp_context_;
    int lane_id = context->lane_id_;
    
    // 1. 当前线程到达 barrier
    warp_ctx->wbar.arrive(lane_id);
    
    // 2. 检查 barrier 是否完成
    if (!warp_ctx->wbar.is_complete()) {
        // 未完成，线程阻塞
        context->state = ThreadStatus::Blocked;
        return;
    }
    
    // 3. Barrier 完成，验证 memory fence（Debug 模式）
    #ifdef PTX_DEBUG
    warp_ctx->wbar.verify_memory_fence();
    #endif
    
    // 4. 所有线程恢复到 reconvergence PC
    for (int i = 0; i < WARP_SIZE; i++) {
        if (warp_ctx->wbar.participation_mask & (1u << i)) {
            if (i == lane_id) {
                // 当前线程使用 force_set_pc
                context->force_set_pc(warp_ctx->wbar.reconvergence_pc);
                context->set_next_pc(warp_ctx->wbar.reconvergence_pc);
            } else {
                // 其他线程通过 WarpContext 设置
                warp_ctx->set_thread_pc(i, warp_ctx->wbar.reconvergence_pc);
            }
            // 重置线程状态
            warp_ctx->set_thread_status(i, ThreadStatus::Active);
        }
    }
    
    // 5. 重置 barrier
    warp_ctx->wbar.reset();
    
    // 6. 检查 SIMT 栈收敛（barrier 可能在 reconvergence point 后）
    warp_ctx->check_reconvergence();
}
```

### PC 保护机制：pc_overridden_ vs force_set_pc()

在 barrier 场景中，有两种 PC 保护机制协同工作：

1. **pc_overridden_ 标志**（PipelineHandler 层）：
   - 当线程阻塞在 barrier 时，`set_pc_overridden(true)` 阻止 `ExecPipe` 调用 `set_next_pc(saved_pc + 1)`
   - 这确保阻塞线程的 `next_pc` 不会被意外修改
   - 线程保持 `pc_overridden_ = true` 直到 barrier 完成并解除阻塞

2. **force_set_pc()**（WarpContext 层）：
   - barrier 完成后，当前线程使用 `force_set_pc(reconvergence_pc)` 设置到聚合点
   - 其他线程通过 `WarpContext::set_thread_pc(i, reconvergence_pc)` 批量设置
   - 这是 ADR-0003 定义的 warp-level 强制设置模式

**为什么需要两种机制**：
- `pc_overridden_` 保护的是 **指令执行阶段** 的 PC 不被默认逻辑覆盖
- `force_set_pc()` 是 **barrier 完成后的显式设置**，属于 warp-level 操作
- 两者在不同时机生效，互不冲突

```cpp
// instruction_base.cpp - pc_overridden_ 保护机制
void PipelineHandler::ExecPipe(ThreadContext *context, StatementContext &stmt) {
    int saved_pc = context->get_pc();
    // ... prepare, execute, commit ...
    
    // 默认设置 next_pc = saved_pc + 1
    if (!pc_overridden_) {
        context->set_next_pc(saved_pc + 1);
    }
    // 只有在非阻塞状态下才重置 pc_overridden_
    bool is_blocked = context->warp_context_ &&
        context->warp_context_->get_warp_state().threads[context->lane_id_].is_blocked;
    if (!is_blocked) {
        pc_overridden_ = false;
    }
}

// barrier.cpp - barrier 完成后的 PC 设置
void BarWarpSyncHandler::processOperation(ThreadContext* context, ...) {
    if (wbar.is_complete()) {
        // barrier 完成，设置所有线程到 reconvergence PC
        if (i == context->lane_id_) {
            context->force_set_pc(reconvergence_pc);  // 当前线程
            context->set_next_pc(reconvergence_pc);
        } else {
            warp_ctx->set_thread_pc(i, reconvergence_pc);  // 其他线程
        }
        warp_ctx->set_thread_status(i, ThreadStatus::Active);
    } else {
        // 线程阻塞，设置 pc_overridden_ 保护 PC
        set_pc_overridden(true);
        warp_state.threads[lane_id].is_blocked = true;
    }
}
```

### 影响范围

| 组件 | 影响类型 | 说明 |
|------|---------|------|
| `include/ptxsim/wbar.h` | 修改 | Wbar 结构增强 |
| `src/ptxsim/instructions/barrier.cpp` | 修改 | barrier 执行流程 + pc_overridden_ 保护 |
| `src/ptxsim/core/sm_context.cpp` | 修改 | barrier 后调用 check_reconvergence |
| `src/ptxsim/instruction_base.cpp` | 修改 | ExecPipe 中条件重置 pc_overridden_ |

## 后果

### 正面影响

- 精确验证 barrier 语义
- Debug 模式能检测 memory fence 违规
- 与 SIMT 栈正确集成

### 负面影响

- Debug 模式下 memory fence 验证有性能开销
- 需要跟踪 barrier 前的内存操作（调试用）

### 风险与缓解

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| participation_mask 计算错误 | 中 | 高 | 单元测试覆盖发散场景的 barrier |
| Memory fence 验证遗漏 | 低 | 高 | Debug 模式下充分测试 |
| Barrier 完成后线程状态泄漏 | 低 | 高 | 确保 barrier 后重置 status = Active |

## 合规检查

后续相关开发应检查：

- [ ] barrier 指令正确初始化 participation_mask
- [ ] 所有参与线程都调用 arrive()
- [ ] barrier 完成后验证 is_complete()
- [ ] barrier 后线程恢复到正确的 reconvergence_pc
- [ ] Debug 模式下 memory fence 验证启用
- [ ] barrier 阻塞线程设置 pc_overridden_ 保护 PC
- [ ] barrier 完成后不重置 exec_mask/active_mask（保留 divergence 状态）
- [ ] barrier 后调用 check_reconvergence 检查 SIMT 栈收敛

## 更新记录

| 日期 | 更新内容 | 作者 |
|------|---------|------|
| 2026-05-05 | 初始版本 | PTX-EMU Team |
| 2026-05-06 | 添加 pc_overridden_ 机制说明、更新合规检查项 | PTX-EMU Team |
| 2026-06-15 | 追加：Warp-级到达计数正式决策（基于 [docs/research/barrier-semantics/06-synthesis-and-recommendations.md](../research/barrier-semantics/06-synthesis-and-recommendations.md)） | Sisyphus |

### 2026-06-15 追加：Warp-级到达计数

经过对 NVIDIA PTX ISA 9.3、Volta Tuning Guide、Hopper Whitepaper 以及 4 个学术 GPU 模拟器
（gpgpu-sim、gem5-gpu、MIAOW、Multi2Sim）的系统调研（详见
[`docs/research/barrier-semantics/`](../research/barrier-semantics/)），本项目正式确认 barrier
到达计数采用 **warp-级粒度**：

| 决策项 | 决策 | 证据 |
|--------|------|------|
| **到达计数单位** | **warp（32 lanes）**—— `arrive(lane_id)` 对所有 32 个 active lanes 调用，`arrived_mask` 累计为全 1，`count_arrived()` = 32 | NVIDIA 硬件 + 4 个学术模拟器一致 |
| **释放范围** | **整 32 lane 一同推进到同一 reconvergence_pc** | gpgpu-sim CHANGES: "irrespective of divergence state" |
| **`participation_mask` 的角色** | **静态声明**（PTX 指令的 static_mask）—— 仅用于 `is_complete()` 的位运算判定，**不限制** `arrive()` 调用对象 | PTX 9.3 §9.7.14.1; MLIR NVVM Dialect |
| **divergent 两半时序** | **不释放直到所有 active lane 都到达** | Volta+ Independent Thread Scheduling |
| **16 named barrier 槽** | **未来工作**—— 当前实现只用 `wbars[0]`，需扩展到全部 16 个 | CUTLASS `HardwareMaxNumNamedBarriers = 16` |

**已修复的不一致**：之前 `tests/integration/barrier/test_warp_barrier integrat ed.cpp` 的部分断言
（`count_arrived() == popcount(mask)`、`get_pc() == 1` for lanes 8-15 等）是基于单 lane 执行 API
（`execute_warp_instruction(stmt, pc)`）的期望值，未对齐 commit `ca2140f` 引入的 32-lane warp 驱动
API（`step_warp`）。本次更新将这些断言的期望值改为与 warp-级语义一致：

- `count_arrived() == 4/2/8` → `count_arrived() == 32`（所有 32 active lanes 都 arrive）
- `count_participants() == 2` → 保持 2（mask 0x03）/ 改为 4（mask 0x0F）/ 改为 8（mask 0xFF）
- `is_complete() == false` → `is_complete() == true`（BUG-POSTBARRIER 修复后 wbar 持续 complete）
- `get_pc() == 1` for lanes 8-15 → `get_pc() >= 4`（lanes 不再卡在 PC=1）

**未来工作**（不属本 ADR 范围）：
- 16 named barrier 槽完整实现
- Cluster barrier (sm_90+)
- mbarrier 完整实现（64-bit shared mem 对象 + phase parity）
- 显式 membar/fence 完整实现（Volta+ 不再隐式 membar.cta）
- `bar.warp.sync` membermask 的 UB 检测

**调研依据**（详见 [docs/research/barrier-semantics/](../research/barrier-semantics/)）:
1. [01-ptx-isa-official-semantics.md](../research/barrier-semantics/01-ptx-isa-official-semantics.md) — NVIDIA 官方 ISA 语义
2. [02-divergent-warp-hardware-behavior.md](../research/barrier-semantics/02-divergent-warp-hardware-behavior.md) — divergent 行为
3. [03-hopper-blackwell-new-features.md](../research/barrier-semantics/03-hopper-blackwell-new-features.md) — sm_90+ 新特性
4. [04-ptx-emu-current-implementation.md](../research/barrier-semantics/04-ptx-emu-current-implementation.md) — 本项目代码地图
5. [05-open-source-simulator-comparison.md](../research/barrier-semantics/05-open-source-simulator-comparison.md) — 4 个开源模拟器对比
6. [06-synthesis-and-recommendations.md](../research/barrier-semantics/06-synthesis-and-recommendations.md) — 综合分析与决策

### Barrier 测试场景覆盖（2026-05-06 添加）

以下 barrier 场景必须有回归测试覆盖：

| 场景 | 测试文件 | 验证要点 |
|------|---------|---------|
| CTA barrier 保留 exec_mask | `test_barrier_active_mask_preserved.cpp` | barrier 后 exec_mask 不变 |
| single-warp bar.warp.sync | barrier 单元测试 | 32 线程正确到达和释放 |
| barrier 后 SIMT 栈收敛 | `sm_context.cpp` while 循环 | 嵌套分支收敛正确 |
| barrier 阻塞 PC 保护 | `instruction_base.cpp` | pc_overridden_ 阻止 next_pc 修改 |
| 发散分支后 barrier | 集成测试 | participation_mask 正确计算 |

## 参考

- [PTX ISA 9.1 Section 9.7.13.1 - Barrier Synchronization](../archive/ptx-instruction-reference/9.7.13_sync_comm.md)
- [SIMT 架构文档](../architecture/SIMT-ARCHITECTURE-V2.md#34-barrier-机制)
- [ADR-0006: SIMT Stack 显式控制流管理](./0006-simt-stack-management.md)

## 2026-06-18 追加：BarrierModule 集成与状态机扩展（commit 13b6b36 ~ 83be5f7）

### 决策

- `BarrierModule` 由 `CTAContext` 独占持有（每个 CTA 一个实例，per-CTA lifecycle），替代 `SMContext` 全局 mutex + map
- `WarpContext` 添加反向链接 `cta_context_`（`set_cta_context` / `get_cta_context`），由 `CTAContext::init` 设置
- `release_cta_barrier(cta_barrier_id, cta_ctx, post_barrier_pc)` 新增 `CTAContext*` 和 `post_barrier_pc` 参数；遍历 `arrived_threads_`，对每个 thread 调用 `set_state(RUN)` + `warp->advance_thread_pc(lane, post_barrier_pc)` —— **修复 BUG-HANDLER-PC-ADVANCE**
- `release_warp_barrier` 实施 `set_active_mask(get_active_mask() | arrived_mask)`（OR 逻辑），由 `BarrierModule` 自身负责（**Caller 层 OR**，不可改 `set_active_mask` 全局语义 —— ret handler 依赖覆写语义清零）
- `WarpBarrier::init` re-init 时**只更新** metadata（`participation_mask` / `reconvergence_pc` / `expected_count` / `state`），**保留** `arrived_mask` / `arrived_count`（force_reconvergence 路径需求 —— BUG-RECONVERGENCE-SIMPLEGEMM）

### 状态机扩展

- `WarpBarrier::State` 5 态：`Uninitialized → Initializing → Waiting → Complete → Released`
- `CTABarrier` 沿用相同语义（通过 `is_initialized_` + `arrived_count >= expected_threads` 表达）

### 实现状态

| 状态机迁移 | 状态 | 证据 |
|----------|------|------|
| `BarHandler::executeBarrier` 走 BarrierModule | ✅ 完成 | `barrier.cpp:230-273` (commit b04cdb2) |
| `BarWarpSyncHandler::processOperation` 走 BarrierModule | ✅ 完成 | `barrier.cpp:110-225` (commit 36dbb9a) |
| `CTAContext` 持有 `BarrierModule` | ✅ 完成 | `cta_context.h:96` + `cta_context.cpp:25` (commit 13b6b36) |
| `WarpContext` 反向链接 CTA | ✅ 完成 | `warp_context.h:201-202` (commit b04cdb2) |
| `release_cta_barrier` 真正推进 PC | ✅ 完成 | `barrier_module.cpp:159-201` (commit 13b6b36) |
| `release_warp_barrier` OR active_mask | ✅ 完成 | `barrier_module.cpp:108-110` (commit c48b1cc) |
| `WarpBarrier::init` re-init 保留 arrived_mask | ✅ 完成 | `warp_barrier.cpp:14-31` (commit 6212624) |

### 废弃 / 暂留项

| 项 | 状态 | 计划 |
|---|------|------|
| `include/ptxsim/wbar.h` 旧 `Wbar` 结构体 | ⚠️ `[[deprecated]]` 标记 | Phase 5 独立 change 处理完整迁移 |
| `warp_state.h::wbars[]` + `current_wbar_id` 字段 | ⚠️ 保留（向后兼容）| 同上 |
| `SMContext::synchronize_barrier` 方法 | ✅ **已删除**(2026-06-20, `cleanup-deprecated-barrier-apis` commit `7914764`) | — |
| `SMContext::barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` | ✅ **已删除**(同上) | — |
| `sm_context.cpp:204-242` periodic barrier check | ✅ **已删除**(同上;保留 lines 244+ warp 调度维护) | — |

### 合规检查项

- [x] `release_warp_barrier` 实施 `set_active_mask(get_active_mask() \| arrived_mask)` —— `src/ptxsim/barrier/barrier_module.cpp:108-110` ✓
- [x] `WarpBarrier::init` re-init 分支**保留** `arrived_mask` / `arrived_count` —— `src/ptxsim/barrier/warp_barrier.cpp:14-31` ✓
- [x] `set_active_mask` 全局实现**未修改** —— 仅 caller 层 OR（ret handler 安全）✓
- [x] `CTAContext` 通过 `std::make_unique<BarrierModule>` 持有 —— `src/ptxsim/core/cta_context.cpp:25` ✓
- [x] `WarpContext::get_cta_context()` 公开访问器 —— `include/ptxsim/warp_context.h:201` ✓
- [x] `BarHandler` 不再调用 `synchronize_barrier` —— `barrier.cpp:230-273` 全部走 `bm.arrive_at_cta_barrier` + `release_cta_barrier` ✓
- [x] `BarWarpSyncHandler` 不再调用 `bsync_manager_` —— `barrier.cpp:110-225` grep `bsync_manager` 应为 0 ✓
- [x] `release_cta_barrier` 调用 `warp->advance_thread_pc(lane, pc)` 真正推进 PC —— `barrier_module.cpp:185-187` ✓
- [x] 集成测试 work-around (`test_cta_barrier_memory_visibility.cpp:138-184`) 已删除 —— commit ad7a46f ✓

### 调研依据

- [`docs/research/barrier-semantics/04-ptx-emu-current-implementation.md`](../research/barrier-semantics/04-ptx-emu-current-implementation.md) — 完整代码地图（Phase 7a 重写）
- [`docs/technical_design/barrier_module_design.md`](../technical_design/barrier_module_design.md) — 落地 v1 状态（Phase 7b 更新）
- [openspec change: integrate-barrier-module-cta-warp](../../openspec/changes/integrate-barrier-module-cta-warp/) — 完整 proposal/design/specs/tasks

### OpenSpec Change

```text
openspec/changes/integrate-barrier-module-cta-warp/
├── proposal.md              (Why/What/Capabilities/Impact)
├── design.md                (Context/Goals/Decisions/Risks/Migration)
├── specs/
│   ├── cta-barrier-module/spec.md          (4 Requirements)
│   ├── warp-barrier-unification/spec.md    (5 Requirements)
│   └── barrier-handler-bugfix/spec.md      (4 Requirements)
└── tasks.md                 (8 phases × 35 atomic tasks)
```

---

## 2026-06-20 追加：Phase 6 Partial Cleanup (`cleanup-deprecated-barrier-apis`)

本变更（commit `8a5573d` + `7914764`，3 commits）完成 Phase 6 partial cleanup：

### 删除项

- `BsyncManager` 类 + `BsyncState` 结构体（`include/ptxsim/bsync_state.h` + `src/ptxsim/core/bsync_state.cpp`）
- 3 个单元测试（`test_bsync_state.cpp` / `test_barrier_scenarios.cpp` / `test_barrier_active_mask_preserved.cpp`）
- `SMContext::synchronize_barrier()` 方法体（`sm_context.cpp:605-705`，约 100 行）
- SM 级全局 barrier 状态字段（`barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_`）
- `sm_context.cpp:204-242` 周期 barrier 检查块（**保留** lines 244+ 的 `decrement_blocked_cycles` + `update_active_mask` warp 调度维护）

### 迁移项

- `warp_context.cpp:283-296` BAR_SYNC fallback 从 `sm_context_->synchronize_barrier()` 迁移到 `cta_context_->get_barrier_module().arrive_at_cta_barrier()`（保留 lessons-learned §1 BAR_SYNC 翻译链：`ThreadState::Blocked` → `state = BAR_SYNC` → `is_blocked = true`）
- `DivergenceExecutionMode` 枚举从 `bsync_state.h` 迁移到 `warp_scheduler.h`（`bsync_state.h` 删除后保留此类型，因 `WarpScheduler` 类直接使用）

### 保留项

- `Wbar` struct（`include/ptxsim/wbar.h`）仍 `[[deprecated]]`（**2026-07-03 已删除** — see ADR §2026-07-03 postmortem）
- `warp_state.h::wbars[]` + `current_wbar_id` 字段（**2026-07-03 已删除**）
- `BarWarpSyncHandler` 仍用 `warp_state.wbars[0]`（Phase 5 推迟到独立 change `migrate-bar-warp-sync-to-barrier-module`）（**2026-07-03 已完成迁移**）
- 19 个 include `ptxsim/wbar.h` 的测试文件全部保留（**2026-07-03 已迁移/重写到 BarrierModule API**）
- `tests/integration/divergence/test_post_barrier_divergence.cpp` 已知 BUG 回归测试保留

### 决策

**`thread->bar_id` 语义验证**（Phase 6 BLOCKING gate）：实施前 grep 验证 `thread->bar_id` 字段在生产代码中**永远为 0**（无任何代码主动设置，仅 `thread_context.cpp:49` 初始化为 0）。这与 `BarrierModule::arrive_at_cta_barrier(0, thread)` 语义一致，因此 fallback 替换是安全的。

**`bsync_state.h` 双重职责问题**（实施中发现）：原 `bsync_state.h` 不仅包含 `BsyncManager` 类（要删除），还包含 `DivergenceExecutionMode` 枚举（要保留）。解决方案：把 `DivergenceExecutionMode` + `divergence_execution_mode_to_string` 移到 `warp_scheduler.h`（`WarpScheduler` 类直接使用）。

**`sm_context.cpp:200-260` 行范围澄清**（ADR-0008 之前错误标注）：实际仅 lines 204-242 是周期 barrier 检查；lines 244-260 是 `decrement_blocked_cycles` + `update_active_mask`（warp 调度维护，必须保留）。本变更的"废弃/暂留项"表已修正。

### 验证

8 个关键测试全部 PASS：
- `unit_barrier_module` ✓
- `unit_post_barrier_two_halves`（BUG-POSTBARRIER-TWOHALVES 回归）✓
- `unit_barrier_reconvergence` ✓
- `unit_barrier_pc_overwrite` ✓
- `unit_barrier_divergence_reconvergence_simplegemm`（BUG-RECONVERGENCE-SIMPLEGEMM 回归）✓
- `integration_barrier_module` ✓
- `integration_barrier_full_lifecycle` ✓
- `integration_post_barrier_divergence`（已知 BUG 回归）✓

### 后续工作

- ~~`migrate-bar-warp-sync-to-barrier-module`（独立 change）：将 `BarWarpSyncHandler` 完整迁移到 `BarrierModule` API，删除 `Wbar` struct + `warp_state.wbars[]` 字段~~ — **2026-07-03 已完成**（commits `0e311566` + `f5640042` + `0bab6487`），详见下一节 §2026-07-03

---

## 2026-07-03 追加：BarWarpSyncHandler 迁移到 BarrierModule API + Wbar 完全删除

### 目标

完成两个独立但相关的清理：
1. **Phase 3**: 将 `BarWarpSyncHandler::processOperation`（commit `36dbb9a` 失败后被 `f033312` revert 的工作）从直接操作 `warp_state.wbars[0]` + `sm_ctx->bsync_manager_` 迁移到 `BarrierModule` API
2. **Phase 7 (P0-A5)**: 完整删除 `Wbar` struct + `warp_state.wbars[]` + `current_wbar_id` + `get_wbar()` compat shim

### 实施

#### Phase 3: BarWarpSyncHandler 迁移（commit `0e311566`）

**修改路径**:
- 路径 A（`force_reconvergence` 分歧场景）：`warp_state.wbars[0]` → `cta_context->get_barrier_module().get_warp_barrier(0)`
- 路径 B（正常 barrier）：同样替换为 `BarrierModule` API

**关键不变式**（**verified** by `tasks.md` / lessons-learned §1）:
1. `WarpBarrier::init` 的 `is_initialized_` 分支保留 `arrived_mask` / `arrived_count` —— 满足 `BUG-RECONVERGENCE-SIMPLEGEMM`
2. `release_warp_barrier` 调用方**必须**在返回值后调用 `context->set_pc_overridden(true)` —— 防止 `commit_pc()` 二次推进跳过 reconvergence point
3. `release_warp_barrier` **必须**完整更新线程状态字段：`is_blocked=false` + `status=Active` + `is_active=true` —— 这是 lesson §1 "跨模块间接状态翻译" 的核心案例
4. 守卫条件替换：`warp_state.current_wbar_id < 0` → `!init_wbar->is_initialized()`；`current_wbar_id >= 0` → `init_wbar->is_initialized()`

**移除残留**: `BsyncManager` 调用残留在路径 A/B 全部删除（`BsyncManager` 类本身已于 `cleanup-deprecated-barrier-apis` commit `7914764` 删除）。

#### Phase 7: Wbar 完全删除（commit `f5640042`, P0-A5）

**删除**:
- `include/ptxsim/wbar.h`（121 行）
- `warp_state.h::wbars[]` 字段 + `current_wbar_id` 字段
- `warp_state::reset()` 中相关循环
- `warp_context.h::get_wbar()` compat shim 声明
- `warp_context.cpp::get_wbar()` 实现
- `warp_state.h` 中 `#include "ptxsim/wbar.h"`

#### Phase 7b: 测试修复（commit `0bab6487`）

5 个被 Phase 7 破坏的测试重写：从直接验证 `Wbar` 内部字段改为通过 `execute_warp_instruction` 约束的 `BarrierModule` API 行为验证。

### 关键决策落实（验证 lessons-learned §1 应用）

**跨模块状态翻译（Cross-module State Translation）** — 三大教训在本次迁移中正确应用:

1. **不要迁移"主逻辑" — 行级 diff 是底线**: 旧 `barrier.cpp:139-141` 三行看似次要的 `set_active_mask` / `is_blocked` / `status` 翻译，迁移到 `release_warp_barrier` 内 (`barrier_module.cpp:119-134`) 一字不差。Comparison: 行 1 = 行 1.

2. **递归锁 = 死锁**: `release_warp_barrier` 已持有 `cta_context->barrier_module_.mutex_`，由同锁 guard 的其他公共方法（如 `arrive_at_warp_barrier`）不能再被它的 caller 路径递归调用（本次迁移中确认所有迁移调用路径均正确）。

3. **Phase 隔离 = 可回退**: Phase 3 和 Phase 7 在不同 commits，Phase 7 失败可单独 revert 不影响 Phase 3 逻辑。

### 验证

| 测试 | 状态 |
|------|------|
| 23/23 barrier 测试 | ✅ ALL PASS |
| `unit_barrier_module` | ✅ |
| `unit_post_barrier_two_halves` (BUG-POSTBARRIER-TWOHALVES) | ✅ |
| `unit_barrier_reconvergence_simplegemm` (BUG-RECONVERGENCE-SIMPLEGEMM) | ✅ |
| `e2e_barrier_warp_sync` | ✅ |
| `e2e_test3_cfg_full` | ✅ |
| `ctest -R "barrier"` (23 tests) | ✅ ALL PASS |

**Wbar 残留检查**: `grep -rn "wbar\.h\|warp_state\.wbars\|current_wbar_id\|get_wbar(" include/ src/ tests/` → **代码层面零残留**，仅剩测试注释中的引用（已逐步清理）。

### 已知未完成 / 后续工作

| 类别 | 描述 | 出处 |
|------|------|------|
| **lifecycle 单元测试** | 删除的 `test_syncthreads_test3_repro.cpp` (-190 行) + `test_exec_layer_e1_e3.cpp` (-59 行) 覆盖了 init→complete→reset→re-init + participation_mask 边界。已规划 follow-up change `barrier-module-lifecycle-tests` 重建这些单元测试 | Code Review Issue I1 |
| **barrier.cpp 注释** | `barrier.cpp:105-228` 124 行单函数 Path A/B 注释不足，文档化 "force_reconvergence path" vs "normal sync path" 边界 | Code Review Issue M1 |
| **exec_mask vs active_mask 文档** | `warp_context.h` 中参数命名歧义（注释，非代码） | Code Review Issue M2 |

### OpenSpec Change

```text
openspec/changes/migrate-bar-warp-sync-to-barrier-module/
├── proposal.md              (Why/What: completed via commits 0e311566+f5640042+0bab6487)
├── design.md                (Context/Decisions/Risks)
├── specs/
│   └── warp-barrier-unification/spec.md    (modified)
└── tasks.md                 (8 phases, 3,5,7 fully landed; doc sync finalised 2026-07-03)
```
