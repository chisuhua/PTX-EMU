## Why

PTX-EMU 在 commit `12390b7`(merge `fix/barrier-architecture-migration`)后已成功将 `BarHandler`(CTA 路径)迁移到 `BarrierModule` API,但**遗留了三套过时的 barrier 实现并存**:

1. **`include/ptxsim/bsync_state.h`** — `BsyncManager` 类(已 `[[deprecated]]` 标识,但仍被 `barrier.cpp:189,240,249` 调用)
2. **`src/ptxsim/core/sm_context.{h,cpp}`** — `synchronize_barrier()` 方法 + 周期检查逻辑(被 `warp_context.cpp:292` BAR_SYNC fallback 使用)
3. **SM 级全局 barrier 状态**:`barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_`

实际生产代码(`barrier.cpp::BarWarpSyncHandler`)仍直接操作 `warp_state.wbars[0]` 和 `sm_ctx->bsync_manager_.bsync/release`,意味着 `Wbar` 字段是**仍在使用**的状态。本 change 完成 Phase 6 partial cleanup(仅删除 `BsyncManager` + `SMContext::synchronize_barrier` + SM 级全局 barrier 状态;保留 `Wbar` struct 留给 Phase 5 独立 change)。

**重要边界**:本 change **不**触碰 `BarWarpSyncHandler`(warp 路径)的旧实现。warp 路径迁移是高风险独立工作(Phase 5,被 commit `36dbb9a` 实施但 commit `f033312` revert),由独立 change `migrate-bar-warp-sync-to-barrier-module` 处理。

## What Changes

> **边界修订(2026-06-20)**:原 proposal 计划删除 `Wbar` struct + `warp_state.wbars[]` 字段,与 design.md Decision 1(保留 BarWarpSyncHandler 操作 wbars[0])冲突。**修订后边界**:仅删除 `BsyncManager` + `SMContext::synchronize_barrier` + SM 级全局 barrier 状态;**保留** `Wbar` struct + `warp_state.wbars[]` 字段(由 Phase 5 独立 change 处理)。

- **删除 `BsyncManager`**:`rm include/ptxsim/bsync_state.h src/ptxsim/core/bsync_state.cpp`;从 `include/ptxsim/sm_context.h` 移除 `BsyncManager bsync_manager_` 字段(line 195)+ `#include "ptxsim/bsync_state.h"` (line 6);从 `src/ptxsim/core/warp_scheduler.cpp` 移除 `#include "ptxsim/bsync_state.h"` (line 2);从 `src/CMakeLists.txt` 移除对应条目
- **删除 SM 级 barrier 状态**:从 `include/ptxsim/sm_context.h` 移除 `synchronize_barrier()` 声明(line 114) + `barrier_waiting_threads` (line 189) / `barrier_thread_counts` (line 190) / `barrier_mutex_` (line 192) 字段;从 `src/ptxsim/core/sm_context.cpp` 删除 `synchronize_barrier()` 方法体(lines 605-705)+ 周期 barrier 检查代码块(lines 204-242,**保留** lines 244+ 的 `decrement_blocked_cycles` + `update_active_mask` warp 调度维护)
- **替换 `warp_context.cpp` BAR_SYNC fallback**:`src/ptxsim/core/warp_context.cpp:283-296` 的 `sm_context_->synchronize_barrier(...)` (line 292) 替换为 `cta_context_->get_barrier_module().arrive_at_cta_barrier(thread->bar_id, thread)`(commit b04cdb2 引入的 `get_cta_context()` 访问器;lessons-learned §1 BAR_SYNC 翻译链保留)
- **同步 `barrier.cpp` 调用点**:删除 lines 189, 240, 249 的 `sm_ctx->bsync_manager_.bsync/release` 调用(无替代,删除即可 - `Wbar.arrived_mask` 已记录到达,`BsyncManager` 重复);删除 line 23 `#include "ptxsim/sm_context.h"`(若仅用于 `synchronize_barrier`);删除 line 385 过期注释(行号引用 :703 应改为 :605);清理 lines 188 + 238 的 `SMContext* sm_ctx` 死代码
- **删除单元测试**:`rm tests/unit/sync/test_bsync_state.cpp` + `tests/unit/sync/test_barrier_active_mask_preserved.cpp` + `tests/unit/barrier/test_barrier_scenarios.cpp`(因 BsyncManager 类 + synchronize_barrier 方法被删除)+ 对应 `tests/unit/CMakeLists.txt` 条目
- **不修改 `BarWarpSyncHandler::processOperation` 主逻辑**:`barrier.cpp::BarWarpSyncHandler` 仍操作 `warp_state.wbars[0]`,Phase 5 独立 change `migrate-bar-warp-sync-to-barrier-module` 处理完整迁移
- **不删除 `Wbar` struct**:`include/ptxsim/wbar.h` + `include/ptxsim/warp_state.h`(`std::array<Wbar, 4> wbars` + `int current_wbar_id = -1`)**保留**;19 个 include `ptxsim/wbar.h` 的测试文件**保留**;`tests/integration/divergence/test_post_barrier_divergence.cpp` 已知 BUG 回归测试**保留**

**前置验证(已完成)**:
- ✅ `grep -rn "bsync_manager_\|synchronize_barrier" src/ include/ tests/` 验证:仅历史注释 + 设计文档保留
- ✅ BsyncManager 9 个 getter(`is_waiting` / `get_waiting_mask` / `get_state` / `bssy` / `bsync` / `check_release` / `release` / `cleanup` / `reset`)在生产代码无消费者
- ✅ `barrier_mutex_` 唯一消费者是 `sm_context.cpp:204`(周期检查)+ `sm_context.cpp:608`(synchronize_barrier)
- ✅ `warp_context.cpp:283-296` BAR_SYNC fallback 的 `cta_context_->get_barrier_module()` 替换路径通过 `get_cta_context()` 访问器可达
- ✅ `thread->bar_id` 字段在生产代码中**永远为 0**(`thread_context.cpp:49` 初始化为 0,无任何代码主动设置;与 `arrive_at_cta_barrier(0, thread)` 语义一致)
- ✅ 当前 main 分支 `cmake --build build --target ptxsim` 编译通过
- ✅ 既有 `.worktrees/fix-pre-p0-baseline` 已存在,可复用为 baseline worktree

## Capabilities

### New Capabilities
<!-- 无新能力。本 change 是纯清理。 -->

### Modified Capabilities
<!-- 无 spec 级别行为变更。已存在的 capability 未变。 -->

## Impact

### 代码修改(8 文件)

| 类别 | 影响 | 备注 |
|------|------|------|
| `include/ptxsim/bsync_state.h` | **删除** | 与 `.cpp` 一并删除 |
| `src/ptxsim/core/bsync_state.cpp` | **删除** | 唯一生产调用点是 `barrier.cpp:189,240,249` |
| `include/ptxsim/sm_context.h` | **修改** | 删除 `BsyncManager bsync_manager_` (line 195) + `#include "ptxsim/bsync_state.h"` (line 6) + `synchronize_barrier()` 声明 (line 114) + `barrier_waiting_threads` (line 189) / `barrier_thread_counts` (line 190) / `barrier_mutex_` (line 192) 字段 |
| `src/ptxsim/core/sm_context.cpp` | **修改** | 删除 lines 605-705 `synchronize_barrier()` 方法体 + lines 204-242 周期 barrier 检查代码块 |
| `src/ptxsim/core/warp_context.cpp` | **修改** | **关键**:`warp_context.cpp:283-296` BAR_SYNC fallback 替换 - 删除 line 292 `sm_context_->synchronize_barrier(thread->bar_id, thread)`,改为 `cta_context_->get_barrier_module().arrive_at_cta_barrier(thread->bar_id, thread)`(commit b04cdb2 引入的 `get_cta_context()` 访问器) |
| `src/ptxsim/core/warp_scheduler.cpp` | **修改** | 删除 line 2 `#include "ptxsim/bsync_state.h"`(grep 确认无 `BsyncManager` / `bsync_manager` 使用) |
| `include/ptxsim/warp_scheduler.h` | **修改** | 添加 `DivergenceExecutionMode` 枚举 + `divergence_execution_mode_to_string` 函数(从 `bsync_state.h` 迁移);删除 `#include "ptxsim/bsync_state.h"` (line 5) |
| `src/ptxsim/instructions/barrier.cpp` | **修改** | 删除 lines 189, 240, 249 的 `sm_ctx->bsync_manager_.bsync/release` 调用 + 删除 lines 188 + 238 死代码 `SMContext* sm_ctx` + 删除 line 23 `#include "ptxsim/sm_context.h"` + 删除 line 385 过期注释;**保留** `warp_state.wbars[0]` 操作(用户方向) |
| `src/ptxsim/core/thread_context.cpp:774` | **修改** | 更新注释(`synchronize_barrier` 引用改为 `BarrierModule::arrive_at_cta_barrier`) |
| `src/CMakeLists.txt` | **修改** | 移除 `ptxsim/core/bsync_state.cpp` 条目 |
| `tests/unit/CMakeLists.txt` | **修改** | 移除 `add_catch_test(unit_bsync_state ...)` + `set_tests_properties` 条目(lines 291-294);移除 `test_barrier_scenarios.cpp` 条目;移除 `test_barrier_active_mask_preserved.cpp` 条目 |

### 代码保留(关键边界澄清)

| 类别 | 状态 | 备注 |
|------|------|------|
| `include/ptxsim/wbar.h` | **保留** | Wbar struct 仍存在,Phase 5 独立 change 处理完整迁移 |
| `include/ptxsim/warp_state.h` | **保留** | `std::array<Wbar, 4> wbars;` + `int current_wbar_id = -1;` 不动 |
| `src/ptxsim/barrier/` (warp_barrier / cta_barrier / barrier_module) | **保留** | 生产路径,无任何修改 |

### 测试修改

| 类别 | 影响 | 备注 |
|------|------|------|
| `tests/unit/sync/test_bsync_state.cpp` | **删除** | BsyncManager 类删除后该测试必须删除(非回归) |
| `tests/unit/sync/test_barrier_active_mask_preserved.cpp` | **删除** | synchronize_barrier 删除后失去测试目标 |
| `tests/unit/barrier/test_barrier_scenarios.cpp` | **删除** | synchronize_barrier 删除后失去测试目标 |
| 19 个 include `ptxsim/wbar.h` 的测试文件 | **保留** | Wbar struct 保留,全部编译通过,无需迁移 |
| `tests/integration/divergence/test_post_barrier_divergence.cpp` | **保留** | 已知 BUG 测试,作为回归保护(Wbar 仍存在) |

### 文档同步(必须)

| 文件 | 影响 |
|------|------|
| `docs/adr/0008-barrier-semantics.md` | 追加 "2026-06-20 Phase 6 partial cleanup" 段落;更新"废弃/暂留项"表 |
| `src/ptxsim/core/AGENTS.md` | 删除 line 22 `synchronize_barrier()` 行 + line 85 KNOWN ISSUE 注释(改为指向 `test_post_barrier_divergence.cpp`);line 42 注释更新 |
| `src/ptxsim/AGENTS.md` | line 42 注释更新("`BarWarpSyncHandler` still uses `Wbar` (Phase 5 deferred)");保留 line 48 "DO NOT add new uses of Wbar" 警告 |
| `tests/AGENTS.md` | line 21 描述更新 |

### 实施结果(2026-06-20,merge 到 main)

- **Commit 1** `8a5573d` `refactor(barrier): remove BsyncManager dead code`(11 文件,+19/-776)
- **Commit 2** `7914764` `refactor(barrier): remove SM-level barrier state and migrate fallback`(5 文件,+4/-160)
- **Commit 3** `6ec8efd` `docs(barrier): sync ADR/AGENTS for Phase 6 partial cleanup`(4 文件,+63/-9)

**回归测试结果**:
- 全量 `cmake --build build -j$(nproc)`: ✅ 100% PASS
- 全量 `ctest`: 3 个 pre-existing failures(lessons-learned §15 标注:`unit_simt_stack_stale_entry_blocks_lane0` + `integration_cute_rmsnorm_bar_sync_pattern` + `cute_rmsnorm`),**无新回归**
- PTX 语法测试 `tests/ptx/test_all_ptx.sh`: ✅ 33/33 PASS
- 8 个关键 barrier 测试: ✅ 8/8 PASS(`unit_barrier_module` / `unit_post_barrier_two_halves` / `unit_barrier_reconvergence` / `unit_barrier_pc_overwrite` / `unit_barrier_divergence_reconvergence_simplegemm` / `integration_barrier_module` / `integration_barrier_full_lifecycle` / `integration_post_barrier_divergence`)

## References

- 前置 change(已归档):`openspec/changes/archive/2026-06-19-integrate-barrier-module-cta-warp/`(Phase 5 DEFERRED、Phase 6 待办)
- 后续 change:`migrate-bar-warp-sync-to-barrier-module`(Phase 5 工作,独立调查)
- 审查报告:`.opencode/notes/cleanup-barrier-review.md`(2026-06-20 Sisyphus agent 完整审查 + Oracle 二轮验证)
- Skill:`ptx-barrier-mechanism`(屏障机制全解)
- Skill:`ptx-lessons-learned`(项目经验沉淀,含 lessons-learned §1, §5, §6, §14 Checklist)
- Skill:`regression-bisect`(重构后回归定位)
- Skill:`state-modification-audit`(状态修改交叉引用)
- Skill:`adr-compliance-check`(ADR 合规检查)
- ADR-0008(barrier 语义增强,含 2026-06-18 Postmortem + 2026-06-20 Phase 6 partial cleanup 追加)
- 调研:`docs/research/barrier-semantics/`(01-06 全部 6 份调研文档)
- 经验沉淀:`docs/dev-process/lessons-learned.md`(含 BAR_SYNC 翻译链、递归锁、分 Phase commit、基线 worktree)
- 当前已 deprecated 状态:`include/ptxsim/wbar.h`(`[[deprecated]]` 注解,**保留**)
