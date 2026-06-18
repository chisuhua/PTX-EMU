# 04. PTX-EMU 当前 Barrier 实现 — 代码地图

> **最后更新**：2026-06-18（integrate-barrier-module-cta-warp Phase 7）
> **来源**：项目源码 + git history
> **本次更新要点**：从 `BarWarpSyncHandler`/`BarHandler` 旧实现（直接操作 `Wbar` + `SMContext::synchronize_barrier`）迁移到统一 `BarrierModule` API

---

## 1. 生产路径：所有 barrier 操作走 `BarrierModule`

`include/ptxsim/barrier/barrier_module.h` 暴露**唯一**的统一接口：

| API | 用途 |
|-----|------|
| `BarrierModule::init_warp_barrier(id, mask, reconv_pc, barrier_pc)` | 初始化 warp 级屏障 |
| `BarrierModule::arrive_at_warp_barrier(id, lane_id)` | 单 lane 到达；返回 `bool complete` |
| `BarrierModule::is_warp_barrier_complete(id)` | 检查是否所有参与 lane 已到达 |
| `BarrierModule::release_warp_barrier(id, warp_ctx)` | 释放 warp 到 reconvergence_pc（含 BUG-POSTBARRIER-TWOHALVES 修复：OR `active_mask`）|
| `BarrierModule::init_cta_barrier(id, total_threads, warp_count)` | 初始化 CTA 级屏障 |
| `BarrierModule::arrive_at_cta_barrier(id, thread)` | 单 thread 到达；返回 `bool complete` |
| `BarrierModule::is_cta_barrier_complete(id)` | 检查是否所有 CTA 内线程已到达 |
| `BarrierModule::release_cta_barrier(id, cta_ctx, post_barrier_pc)` | **推进每个到达 thread 的 per-thread PC**（修复 BUG-HANDLER-PC-ADVANCE）|

`BarrierModule` 实例由每个 `CTAContext` 独占持有（per-CTA lifecycle 与 barrier 作用域对齐），通过 `CTAContext::get_barrier_module()` 访问。

---

## 2. `bar.warp.sync` 实现 — `BarWarpSyncHandler::processOperation()`

**文件**：`src/ptxsim/instructions/barrier.cpp:110-225`（commit `36dbb9a` 重写）

### 2.1 完整状态机

| 阶段 | 行为 |
|------|------|
| 取操作数 | `static_mask = *operands[0]`, `reconvergence_pc = *operands[1]` |
| 获取 `BarrierModule` | `cta_ctx->get_barrier_module()` |
| 计算 dynamic_mask | 仅当 wbar 未初始化时，遍历 32 lane 匹配 `pc == current_pc` |
| **force_reconvergence 路径** | `unique_pcs().size() > 1` 时调用 `bm.init_warp_barrier(static_mask)` 然后 `arrive_at_warp_barrier`；`is_complete()` → `bm.release_warp_barrier(WBAR_ID, warp_ctx)` |
| **正常路径** | `bm.init_warp_barrier(static_mask, reconv_pc, barrier_pc)` 然后 `arrive_at_warp_barrier`；`complete && is_initialized()` → `bm.release_warp_barrier` |
| 未完成 | 设置 `is_blocked=true`, `status=Blocked`, `set_pc_overridden(true)` |

### 2.2 关键修复（保留自旧实现，已迁移到新模块）

| Bug | 修复位置 | 行为 |
|-----|---------|------|
| **BUG-RECONVERGENCE-SIMPLEGEMM** | `warp_barrier.cpp:14-31` | `WarpBarrier::init` 在已初始化时**只更新 metadata**，保留 `arrived_mask` —— force_reconvergence 路径第二个 half 累积在第一个 half 上 |
| **BUG-POSTBARRIER-TWOHALVES** | `barrier_module.cpp:108-110` | `release_warp_barrier` 调用 `set_active_mask(get_active_mask() \| arrived_mask)` —— 第二次 release 不会丢失第一次释放的 lane |
| **BUG-CUTE-RMSNORM-BROADCAST-SKIP** | `barrier.cpp` 行内检查 | `complete && wbar->is_initialized()` —— 避免已释放的 wbar 被重复 release |

### 2.3 不变量

- `WBAR_ID = 0` 始终是 `bar.warp.sync` 使用的唯一槽
- `force_reconvergence_at_barrier` 仍是**空操作**（设计原则：不主动推进 PC，留给调度器自然选择）
- `set_pc_overridden(true)` 在 release 后调用，阻止 `ExecPipe` 默认的 `set_next_pc(saved_pc + 1)` 覆盖释放后的 PC

---

## 3. `bar.sync` (CTA 级) 实现 — `BarHandler::executeBarrier()`

**文件**：`src/ptxsim/instructions/barrier.cpp:230-273`（commit `b04cdb2` 重写）

### 3.1 流程

```cpp
void BarHandler::executeBarrier(ThreadContext* context, const BarrierInstr& instr) {
    int barId = instr.barId.value_or(0);
    WarpContext* warp_ctx = context->get_warp_context();
    CTAContext* cta_ctx = warp_ctx->get_cta_context();
    BarrierModule& bm = cta_ctx->get_barrier_module();

    if (warp_ctx->get_unique_pcs().size() > 1) {
        warp_ctx->force_reconvergence_at_barrier(context->get_pc());
    }

    bool sync_complete = bm.arrive_at_cta_barrier(barId, context);
    if (sync_complete) {
        int post_barrier_pc = context->get_pc() + 1;
        bm.release_cta_barrier(barId, cta_ctx, post_barrier_pc);
        context->set_next_pc(post_barrier_pc);
    } else {
        context->set_next_pc(context->get_pc());
    }
}
```

### 3.2 修复的 Handler Bug（BUG-HANDLER-PC-ADVANCE）

**Pre-fix（commit `b04cdb2` 之前）**：
```cpp
// sm_context.cpp:synchronize_barrier 旧逻辑
thread->set_state(RUN);
thread->set_next_pc(thread->get_pc() + 1);  // 只更新 next_pc，不更新 pc
```

**症状**：`warp_state.threads[lane].pc` 停留在 barrier PC，调度器下一周期从同一 PC 重新执行 `bar.sync`，**无限循环**。

**Post-fix**：`BarrierModule::release_cta_barrier` 调用 `warp->advance_thread_pc(lane, post_barrier_pc)`，真正推进 `warp_state.threads[lane].pc`（`set_pc + set_next_pc` 都被更新）。

---

## 4. `WarpBarrier` 数据结构 — `include/ptxsim/barrier/warp_barrier.h`

### 4.1 字段语义

| 字段 | 类型 | 语义 |
|------|------|------|
| `state_` | `State` enum | Uninitialized → Initializing → Waiting → Complete → Released |
| `participation_mask_` | `uint32_t` | 静态：PTX 指令指定的应参与 lane 位图 |
| `arrived_mask_` | `uint32_t` | 动态：已调用 `arrive()` 的 lane 位图 |
| `reconvergence_pc_` | `int` | 屏障完成后所有 lane 跳转到的 PC |
| `barrier_pc_` | `uint32_t` | 屏障指令本身的 PC（用于调试 + init 记录）|
| `expected_count_` | `uint32_t` | `popcount(participation_mask)` |
| `arrived_count_` | `uint32_t` | `popcount(arrived_mask)` |
| `is_initialized_` | `bool` | 是否已调用 `init()` |

### 4.2 关键方法

| 方法 | 行号 | 行为 |
|------|------|------|
| `init(mask, reconv_pc, barrier_pc)` | `warp_barrier.cpp:13-37` | 首次 init 设置所有字段；**re-init 保留 arrived_mask**（BUG-RECONVERGENCE 修复）|
| `arrive(lane_id)` | `warp_barrier.cpp:39-60` | `arrived_mask \|= 1<<lane_id`；状态转移 `Initializing/Waiting → Complete` |
| `is_complete()` | `warp_barrier.cpp:62-66` | `(arrived_mask & participation_mask) == participation_mask` |
| `needs_to_wait(lane_id)` | `warp_barrier.cpp:68-79` | 当前 lane 是否需要继续等待 |
| `reset()` | `warp_barrier.cpp:81-90` | 清零所有字段 |

### 4.3 与旧 `Wbar` 结构的对比

| 维度 | 旧 `Wbar` (`include/ptxsim/wbar.h`, **deprecated**) | 新 `WarpBarrier` |
|------|-----------------------------------------------------|------------------|
| 字段类型 | struct | class |
| 状态机 | 隐式（通过 `is_initialized`）| 显式 `State` enum |
| 内存 fence 验证字段 | `memory_fence_verification_enabled` + `pre_barrier_stores` | 无（仅运行时调试 dump）|
| re-init 语义 | 调用方需手动处理（BUG-RECONVERGENCE 风险） | `init` 内部处理 |
| 拥有者 | `WarpState::wbars[4]`（每 warp）| `BarrierModule::warp_barriers_[4]`（每 CTA）|

---

## 5. `CTABarrier` 数据结构 — `include/ptxsim/barrier/cta_barrier.h`

### 5.1 字段语义

| 字段 | 类型 | 语义 |
|------|------|------|
| `barrier_id_` | `int` | Named barrier ID（0-15）|
| `expected_threads_` | `int` | 该 CTA 应到达的线程数 |
| `warp_count_` | `int` | CTA 中的 warp 数 |
| `arrived_threads_` | `std::set<ThreadContext*>` | 已到达 thread 集合 |
| `mutex_` | `std::mutex` | 保护 arrived_threads_ 的并发访问 |
| `is_initialized_` | `bool` | 是否已调用 `init()` |

### 5.2 关键方法

| 方法 | 行为 |
|------|------|
| `init(barrier_id, total_threads, warp_count)` | 清空 arrived_threads_，设置元数据 |
| `arrive(thread)` | 插入到 arrived_threads_；返回 `arrived_count >= expected_threads` |
| `is_complete()` | 锁内检查 `arrived_count >= expected_threads` |
| `reset()` | 清空所有状态 |

### 5.3 与 NVIDIA 硬件对齐

- `MAX_BARRIERS_PER_CTA = 16`（`include/ptxsim/barrier/barrier_types.h:10`）—— 与 NVIDIA `HardwareMaxNumNamedBarriers = 16` 一致（CUTLASS `include/cutlass/arch/barrier.h:2810`）
- 单 CTA 内可同时使用最多 16 个不同 ID 的 `bar.sync N` / `bar.arrive N` / `bar.red.* N`

---

## 6. `CTAContext` 持有 `BarrierModule`

**文件**：`include/ptxsim/cta_context.h` + `src/ptxsim/core/cta_context.cpp`（commit `13b6b36`）

```cpp
class CTAContext {
    ...
private:
    std::unique_ptr<BarrierModule> barrier_module_;
public:
    BarrierModule& get_barrier_module() { return *barrier_module_; }
};
```

**初始化时机**：`CTAContext::init()` 开头 `barrier_module_ = std::make_unique<BarrierModule>()`。

**生命周期**：与 CTA 同生同死（`unique_ptr` 自动管理）。

---

## 7. `WarpContext` 反向链接到 `CTAContext`

**文件**：`include/ptxsim/warp_context.h` + `src/ptxsim/core/cta_context.cpp`（commit `b04cdb2`）

```cpp
class WarpContext {
    ...
public:
    void set_cta_context(CTAContext *cta_ctx) { cta_context_ = cta_ctx; }
    CTAContext *get_cta_context() const { return cta_context_; }
private:
    CTAContext *cta_context_ = nullptr;
};
```

**初始化时机**：`CTAContext::init()` 中 `warp->set_cta_context(this)` 在每个 warp 创建时调用。

**用途**：barrier handler 通过 `warp_ctx->get_cta_context()->get_barrier_module()` 获取 barrier 状态机，无需通过 SM/线程遍历查询。

---

## 8. 已知已修复的 Bug 总结（迁移到新模块）

| Bug | 修复 commit | 修复位置 | 触发场景 | 症状 | 修复 |
|-----|-----------|---------|---------|------|------|
| **BUG-RETHANG** | (2026-06) | `control.cpp:RetHandler` | `ret` 指令发散时 | `WarpContext::is_finished()` 永远 false，warp 调度死循环 | 标记**所有 32 lane** 为 exited（`is_exited=true`, `state=EXIT`），再 `update_active_mask()` |
| **BUG-POSTBARRIER-TWOHALVES** | `c48b1cc` | `barrier_module.cpp:108-110` | divergent warp 两半在不同 cycle 到达同一 `bar.warp.sync` | 第二次释放覆写 `active_mask`，丢失第一次释放的 lane | 释放前 `set_active_mask(get_active_mask() \| arrived_mask)`（caller 层 OR，不改 `set_active_mask` 全局语义） |
| **BUG-RECONVERGENCE-SIMPLEGEMM** | `6212624` | `warp_barrier.cpp:14-31` | simpleGEMM 风格的 `bar.sync` 翻译为 `bar.warp.sync`，第一半 lane 16-31 释放后被 wbar 重新 init 抹除到达记录 | 后续到达永远无法凑齐 `participation_mask` → barrier 永不完成 | wbar 已初始化时**只更新** `participation_mask` / `reconvergence_pc`，**保留** `arrived_mask` |
| **BUG-HANDLER-PC-ADVANCE** | `b04cdb2` | `barrier_module.cpp::release_cta_barrier` + `cta_context.cpp::init` | `BarHandler::executeBarrier` 旧实现只 `set_next_pc(pc+1)`，不更新 `warp_state.threads[].pc` | 释放后线程停在 barrier PC，调度器下一周期重执行 `bar.sync`，无限循环 | `release_cta_barrier` 调用 `warp->advance_thread_pc(lane, post_pc)` 真正推进 PC |

---

## 9. 最近 Barrier 相关 Commits

| Commit | 描述 |
|--------|------|
| `83be5f7` | chore(barrier): deprecate legacy Wbar struct in favor of WarpBarrier |
| `36dbb9a` | feat(barrier): migrate BarWarpSyncHandler to BarrierModule API |
| `ad7a46f` | test(barrier): remove CTA handler bug work-around |
| `b04cdb2` | feat(barrier): migrate BarHandler to BarrierModule + fix handler PC advance bug |
| `acb2311` | test(barrier): CTA barrier release advances per-thread PC |
| `13b6b36` | feat(barrier): extend BarrierModule::release_cta_barrier + CTAContext holds BarrierModule |
| `6212624` | fix(barrier): preserve arrived_mask on WarpBarrier re-init |
| `c48b1cc` | fix(barrier): OR active_mask in BarrierModule::release_warp_barrier |
| `09de279` | fix(barrier): OR new arrived_mask with existing active_mask（BUG-POSTBARRIER 原修复，barrier.cpp 内）|
| `5820f7e` | fix(barrier): preserve arrived_mask across force_reconvergence re-init（BUG-RECONVERGENCE 原修复，barrier.cpp 内）|
| `03bf0c5` | fix(barrier): guard against repeat-release in BarWarpSyncHandler |
| `25002c9` | fix(barrier): fall back to current_pc+1 when reconvergence_pc unset |
| `e087e4f` | ptxsim: implement warp-level divergence reconvergence at barrier |
| `c0e67ae` | feat: Phase 1.3 - integrate BsyncManager into bar.warp.sync handler |

---

## 10. 屏障相关测试覆盖

### 单元测试 (`tests/unit/barrier/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_barrier_module.cpp` | `WarpBarrier` / `CTABarrier` / `BarrierModule` 基础数据结构 |
| `test_warp_barrier.cpp` | 旧 `Wbar` API 直接调用（**待迁移到 WarpBarrier**）|
| `test_barrier_reconvergence.cpp` | barrier 与 reconvergence 交互 |
| `test_barrier_scenarios.cpp` | 各种 barrier 场景 |
| `test_barrier_scenarios_integrated.cpp` | 同上，集成版 |
| `test_barrier_interaction_integrated.cpp` | barrier 与其他指令的交互 |
| `test_barrier_verification.cpp` | 屏障验证逻辑 |
| `test_post_barrier_two_halves.cpp` | BUG-POSTBARRIER-TWOHALVES 直接测试 |
| `test_barrier_divergence_reconvergence_simplegemm.cpp` | BUG-RECONVERGENCE-SIMPLEGEMM 场景重现 |

### 集成测试 (`tests/integration/barrier/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_cta_barrier_memory_visibility.cpp` | 2-warp CTA barrier（**work-around 已删除**，验证 handler 修复后路径独立可用）|
| `test_warp_barrier_memory_visibility.cpp` | warp-level sister test |
| `test_barrier_full_lifecycle.cpp` | 屏障完整生命周期（init → arrive → release → reset）|
| `test_barrier_divergence_scheduling.cpp` | 屏障在分歧调度器下的行为 |
| `test_barrier_module_integrated.cpp` | **新增**：`BarrierModule::release_cta_barrier` 推进 per-thread PC（BUG-HANDLER-PC-ADVANCE）|
| `test_barrier_verification_integrated.cpp` | 屏障状态验证 |

### 分歧 + 屏障集成测试 (`tests/integration/divergence/`)

| 文件 | 覆盖内容 |
|------|---------|
| `test_post_barrier_two_halves.cpp` | BUG-POSTBARRIER-TWOHALVES smoke test |
| `test_post_barrier_reconvergence_simplegemm.cpp` | BUG-RECONVERGENCE-SIMPLEGEMM 端到端 |
| `test_post_barrier_divergence.cpp` | barrier 后分歧行为 |
| `test_nested_divergence.cpp` | 嵌套分歧 |
| `test_divergence_sync_convergence.cpp` | 分歧同步收敛（基础）|

---

## 11. 关键架构洞察

### 11.1 Dual State Mechanism（来自 `src/ptxsim/core/AGENTS.md`）

`active_mask[32]` 与 `warp_state.threads[i].is_active` 的 self-healing 机制在 `BarrierModule` 迁移后**保持不变**：
- `BarrierModule::release_warp_barrier` 设置 `active_mask = existing \| arrived_mask`（OR 语义）
- `BarrierModule::release_cta_barrier` 不直接修改 `active_mask`（per-thread `is_active` 由 `advance_thread_pc` + 调度器 `update_active_mask` 重建）

### 11.2 SCOPE-OF-EFFECT 原则

`BarrierModule::release_*` 方法考虑**所有 32 lane**：
- `release_warp_barrier`：遍历 32 lane，`arrived_mask` 中置位的全部推进
- `release_cta_barrier`：遍历 `arrived_threads_` set（不一定 32 个），全部推进

### 11.3 已知遗留技术债

1. **`SMContext::synchronize_barrier` 仍存在但 handler 不再调用** —— 死代码，sm_context.cpp:200-260 的 periodic check 仍用 `barrier_waiting_threads` map（永远空）。完整删除需重写 periodic check 逻辑（未来 sprint）。
2. **`warp_state.wbars[4]` + `current_wbar_id` 字段保留** —— 与 `Wbar` deprecation 一起保留向后兼容，约 50+ 测试仍使用旧 API。
3. **`include/ptxsim/wbar.h` 标记 deprecated** —— 新代码必须用 `WarpBarrier`；测试可在 deprecation warning 下继续使用直至迁移完成。

---

## 🎯 核心架构总结

1. **所有 barrier 操作通过 `BarrierModule` API**（CTA + Warp 两种 scope 由同一类管理）
2. **`CTAContext` 持有 `BarrierModule`**（per-CTA 生命周期对齐）
3. **`WarpContext` 通过 `cta_context_` 反向链接**（handler 直接访问 barrier 状态机）
4. **`release_cta_barrier` 真正推进 per-thread PC**（修复 BUG-HANDLER-PC-ADVANCE）
5. **`release_warp_barrier` OR `active_mask`**（保留 BUG-POSTBARRIER 修复）
6. **`WarpBarrier::init` re-init 保留 `arrived_mask`**（保留 BUG-RECONVERGENCE 修复）
7. **`force_reconvergence_at_barrier` 仍是空操作**（设计哲学：依赖调用方立即 `is_blocked=true`）
