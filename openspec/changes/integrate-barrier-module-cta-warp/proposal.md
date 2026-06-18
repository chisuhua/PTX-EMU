## Why

PTX-EMU 当前存在**双轨 barrier 实现并存**的技术债：旧 `Wbar` 结构体 + `SMContext::synchronize_barrier` 全局 map 是生产路径，而 `BarrierModule` / `CTABarrier` / `WarpBarrier` 整套新模块已实现、已加入 `ptxsim.so` 构建，**却从未被任何生产代码调用**。这导致 413 行死代码 + 文档/测试/AGENTS 多处错位（`04-ptx-emu-current-implementation.md` 描述旧路径、`src/ptxsim/AGENTS.md` 无 `barrier/` 子目录、`barrier_module_design.md` 状态仍是"草稿"）。同时 `BarHandler::executeBarrier` 存在已知 bug（释放线程时未调用 `commit_pc()`，被集成测试用 `advance_thread_pc` work-around 掩盖）。本次修复同步集成新模块、修复 handler bug、清理备份文件、统一文档。

## What Changes

- **集成 BarrierModule 到生产路径**：`BarWarpSyncHandler::processOperation` 调用 `BarrierModule::init_warp_barrier` / `arrive_at_warp_barrier` / `release_warp_barrier`；`BarHandler::executeBarrier` 调用 `BarrierModule::arrive_at_cta_barrier`
- **修复 `BarHandler` 已知 bug**：补齐 `commit_pc()` 调用（实际推进 `warp_state.threads[].pc`），删除 `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184` 的测试 work-around
- **删除旧 `Wbar` 数据结构**：`include/ptxsim/wbar.h` 移除；`warp_state.h` 移除 `std::array<Wbar, 4> wbars;` 和 `current_wbar_id`；`barrier.cpp` 移除 `bsync_manager_` 桥接层
- **让 CTAContext 持有 BarrierModule 实例**（每个 CTA 一个，生命周期与 CTA 一致），替代 SMContext 的全局 mutex + map
- **删除遗留备份**：`src/ptxsim/instructions/barrier.cpp.bak`、`barrier.cpp.orig`
- **同步文档**：`docs/research/barrier-semantics/04-ptx-emu-current-implementation.md` 描述新模块为生产路径；`barrier_module_design.md` 状态改为"已落地";`src/ptxsim/AGENTS.md` + `src/ptxsim/instructions/AGENTS.md` 描述 `barrier/` 子目录
- **实现 `BarrierModule::release_cta_barrier` 真正释放线程**：遍历 `arrived_threads_`，调用 `set_state(RUN)` + `advance_thread_pc`，对齐 `release_warp_barrier` 行为
- **新增 unit 测试覆盖 `CTABarrier::arrive` / `is_complete` / `release` 完整流程**（当前测试只覆盖 `init`/`get_*`）
- **更新 ADR-0008** 描述新 `BarrierModule` 架构决策

## Capabilities

### New Capabilities

- `cta-barrier-module`: CTA 级屏障统一管理模块（16 个 named barrier 槽，与 NVIDIA 硬件对齐）—— 集成 `BarrierModule` + `CTABarrier` 到生产 handler 路径，让 `BarHandler::executeBarrier` 走新 API
- `warp-barrier-unification`: Warp 级屏障从旧 `Wbar` 结构体迁移到新 `WarpBarrier` 类，统一通过 `BarrierModule` 调度
- `barrier-handler-bugfix`: 修复 `BarHandler::executeBarrier` 释放线程未调用 `commit_pc()` 的已知 bug，删除测试 work-around

### Modified Capabilities

无（首次引入 barrier 相关 specs）

## Impact

| 类别 | 影响 |
|------|------|
| `src/ptxsim/instructions/barrier.cpp` | **修改**：`BarWarpSyncHandler` + `BarHandler` 调用 `BarrierModule` API；删除旧 `bsync_manager_` 桥接 |
| `src/ptxsim/core/sm_context.cpp` | **修改**：移除 `synchronize_barrier` 和 `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` 全局状态 |
| `src/ptxsim/core/warp_context.cpp` | **修改**：移除 `warp_state.wbars[]` / `current_wbar_id` 引用；改为持有 `BarrierModule*` |
| `src/ptxsim/core/warp_state.h` | **修改**：移除 `std::array<Wbar, 4> wbars;` + `current_wbar_id` 字段 |
| `src/ptxsim/core/cta_context.cpp` + `cta_context.h` | **修改**：持有 `BarrierModule` 实例（每个 CTA 一个），生命周期管理 |
| `include/ptxsim/wbar.h` | **删除**：旧 `Wbar` 结构体 |
| `src/ptxsim/instructions/barrier.cpp.bak` / `.orig` | **删除**：遗留备份文件 |
| `src/ptxsim/barrier/{barrier_module,warp_barrier,cta_barrier}.cpp` | **扩展**：`release_cta_barrier` 实现真正的线程释放 |
| `include/ptxsim/barrier/barrier_module.h` | **扩展**：`release_cta_barrier` 签名包含 `CTAContext*` 用于遍历线程 |
| `tests/unit/barrier/test_barrier_module.cpp` | **扩展**：新增 `CTABarrier::arrive` / `release` 完整流程覆盖 |
| `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp` | **修改**：删除 L138-184 work-around，验证 `BarHandler` bug 修复后路径独立可用 |
| `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md` | **重写**：描述新 `BarrierModule` 生产路径，移除旧实现描述 |
| `docs/technical_design/barrier_module_design.md` | **更新**：状态从"草稿"改为"已落地 v1"；补迁移路径章节 |
| `docs/adr/0008-barrier-semantics.md` | **追加**：描述 `BarrierModule` 集成决策与状态机映射（State enum: Uninitialized/Initializing/Waiting/Complete/Released）|
| `src/ptxsim/AGENTS.md` + `src/ptxsim/instructions/AGENTS.md` | **更新**：描述 `src/ptxsim/barrier/` 子目录为生产路径 |

## References

- Skill: `ptx-barrier-mechanism`（屏障机制全解）
- Skill: `ptx-instruction-pipeline`（指令执行流水线）
- Skill: `regression-bisect`（重构后回归定位）
- Skill: `state-modification-audit`（状态修改交叉引用）
- ADR-0008（barrier 语义增强 + 2026-06-15 追加 warp-级到达计数决策）
- ADR-0006（SIMT Stack 显式控制流管理）
- 调研：`docs/research/barrier-semantics/`（01-06 全部 6 份调研文档）
- 设计：`docs/technical_design/barrier_module_design.md`
- 已知 bug：`tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184`（测试 work-around 注释）
