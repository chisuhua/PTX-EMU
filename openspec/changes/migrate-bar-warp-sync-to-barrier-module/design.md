## Context

PTX-EMU 在 commit `12390b7` 合并 `fix/barrier-architecture-migration` 后，`BarHandler`（CTA 路径）已切到 `BarrierModule` API。但 `BarWarpSyncHandler::processOperation`（warp 路径）仍直接操作 `warp_state.wbars[0]` 和 `sm_ctx->bsync_manager_`，处于半迁移状态。

**失败历史（commit `36dbb9a` → `f033312` revert）**：

commit `36dbb9a` 实施过完整迁移，commit `f033312` revert。revert 原因：

> **Phase 5 推迟（2026-06-18）**：
> 分歧 warp 的两半 lanes（0-15 vs 16-31）在 post-barrier PC 处卡住，无法到达 MAIN_LOOP_PC。`force_reconvergence` 路径与正常 barrier 释放路径的交互存在问题。

postmortem 详见 `docs/dev-process/lessons-learned.md`。

### 现状代码路径（`barrier.cpp:116-286`）

```cpp
void BarWarpSyncHandler::processOperation(...) {
    // 1. 解析 mask + reconvergence_pc
    // 2. 获取 WarpContext + WarpState
    // 3. 路径 A: force_reconvergence (分歧场景)
    if (unique_pcs.size() > 1 && warp_state.current_wbar_id < 0) {
        warp_ctx->force_reconvergence_at_barrier(current_pc);
        warp_state.current_wbar_id = 0;
        ptxsim::Wbar& init_wbar = warp_state.wbars[0];

        // BUG-RECONVERGENCE-SIMPLEGEMM fix
        if (!init_wbar.is_initialized) {
            init_wbar.init(participation_mask, reconvergence_pc);
        } else {
            init_wbar.participation_mask = participation_mask;
            init_wbar.reconvergence_pc = reconvergence_pc;
            // ... 保留 arrived_mask
        }
        init_wbar.arrive(lane_id);

        if (init_wbar.is_complete()) {
            sm_ctx->bsync_manager_.release(0);  // ← 旧 BsyncManager 调用
            warp_ctx->set_exec_mask(init_wbar.arrived_mask);
            // ... 释放线程
            warp_ctx->set_active_mask(
                warp_ctx->get_active_mask() | init_wbar.arrived_mask);  // BUG-POSTBARRIER-TWOHALVES fix
            warp_state.current_wbar_id = -1;
        } else {
            // 标记阻塞
        }
        return;
    }

    // 4. 路径 B: 正常 barrier 路径（无分歧）
    int wbar_id = 0;
    ptxsim::Wbar& wbar = warp_state.wbars[wbar_id];
    if (!wbar.is_initialized) {
        wbar.init(participation_mask, reconvergence_pc);
        warp_state.current_wbar_id = wbar_id;
    }
    wbar.arrive(lane_id);
    sm_ctx->bsync_manager_.bsync(wbar_id, lane_id, current_pc);

    if (wbar.is_complete() && warp_state.current_wbar_id >= 0) {
        sm_ctx->bsync_manager_.release(wbar_id);
        // ... 释放线程 + BUG-POSTBARRIER-TWOHALVES fix
    }
}
```

**失败根因分析（推测）**：
1. **路径 A 的双重 arrived**：`force_reconvergence_at_barrier` 重新进入时调用 `init_wbar.init(...)`；如果 init_wbar 已初始化（BUG-RECONVERGENCE-SIMPLEGEMM 场景），需要保留 arrived_mask。**这是关键不变性**
2. **BarrierModule 抽象不持有"已初始化"语义**：`init_warp_barrier` 当前直接覆盖（包括 arrived_mask）→ 破坏 BUG-RECONVERGENCE-SIMPLEGEMM 修复逻辑
3. **force_reconvergence 与 WarpBarrier::init 的交互未设计**：commit `36dbb9a` 直接迁移 API，未解决"已初始化时怎么办"的问题

### 关键文件

| 文件 | 当前职责 | 目标职责 |
|------|---------|---------|
| `src/ptxsim/instructions/barrier.cpp` | `BarWarpSyncHandler::processOperation` 操作 `warp_state.wbars[0]` | 调用 `BarrierModule::init_warp_barrier / arrive_at_warp_barrier / release_warp_barrier` |
| `src/ptxsim/barrier/warp_barrier.cpp` | `WarpBarrier::init` 直接覆盖（无 is_initialized_ 分支） | 增加 `is_initialized_` 分支：保留 arrived_mask |
| `src/ptxsim/barrier/barrier_module.cpp` | `init_warp_barrier` 直接调用 `wbar->init(...)` | 调用更新后的 `wbar->init(...)`（带 is_initialized_ 分支） |

## Goals / Non-Goals

**Goals:**
- 完整统一：`BarWarpSyncHandler` 通过 `BarrierModule` API（与其他 handler 一致）
- 修复 commit `36dbb9a` 失败案例：分歧 warp 两半 barrier 完成
- 修复 BUG-RECONVERGENCE-SIMPLEGEMM 不被破坏：`WarpBarrier::init` 已初始化时保留 arrived_mask
- 删除 `BsyncManager` 间接层（与 `cleanup-deprecated-barrier-apis` 协调）

**Non-Goals:**
- **不实现 `bar.warp.sync` membermask 的 UB 检测**（已记录在 ADR-0008 未来工作）
- **不实现 Cluster barrier (sm_90+) / mbarrier**（Hopper/Blackwell 路线图）
- **不修改 `force_reconvergence_at_barrier` 主逻辑**：本次只解决 `barrier.cpp` 与 `BarrierModule` 的交互，不重写 `force_reconvergence`
- **不删除 `WarpBarrier::needs_to_wait()` 两个重载**：审计后决定

## Decisions

### Decision 1: `WarpBarrier::init` 增加 `is_initialized_` 分支

**选择**: 修改 `WarpBarrier::init(participation_mask, reconvergence_pc, barrier_pc)`：

```cpp
void WarpBarrier::init(uint32_t participation_mask, int reconvergence_pc, uint32_t barrier_pc) {
    if (is_initialized_) {
        // 已初始化：仅更新 metadata，保留 arrived_mask_/arrived_count_
        participation_mask_ = participation_mask;
        reconvergence_pc_ = reconvergence_pc;
        barrier_pc_ = barrier_pc;
        expected_count_ = __builtin_popcount(participation_mask);
        state_ = BarrierState::Waiting;
        // NOT reset arrived_mask_ / arrived_count_
    } else {
        // 首次初始化：完全 reset
        arrived_mask_ = 0;
        arrived_count_ = 0;
        participation_mask_ = participation_mask;
        reconvergence_pc_ = reconvergence_pc;
        barrier_pc_ = barrier_pc;
        expected_count_ = __builtin_popcount(participation_mask);
        state_ = BarrierState::Waiting;
        is_initialized_ = true;
    }
}
```

**理由**:
- `is_initialized_` 检查是 force_reconvergence 重新进入的关键不变性
- 保留 arrived_mask 是 BUG-RECONVERGENCE-SIMPLEGEMM 修复的核心
- 这是 commit `36dbb9a` 失败的关键修复点

**风险**: 如果其他调用方依赖"init 总是完全 reset"语义，会破坏。需审计 `WarpBarrier::init` 所有调用点。

### Decision 2: `BarrierModule::init_warp_barrier` 调用更新后的 `WarpBarrier::init`

**选择**: `BarrierModule::init_warp_barrier` 不变（仍调用 `wbar->init(...)`），但通过 Decision 1 的 `WarpBarrier::init` 实现 `is_initialized_` 分支

**理由**: 不变量集中在 `WarpBarrier::init`（决策一致性）；`BarrierModule` 保持简单转发。

### Decision 3: `BarWarpSyncHandler::processOperation` 路径 A 的 arrived_mask 处理

**选择**: 路径 A 中：

```cpp
// 旧代码：
if (!init_wbar.is_initialized) {
    init_wbar.init(participation_mask, reconvergence_pc);
} else {
    init_wbar.participation_mask = participation_mask;
    init_wbar.reconvergence_pc = reconvergence_pc;
    // ...
}
init_wbar.arrive(lane_id);

// 新代码：
// 直接调用 barrier_module.init_warp_barrier（带 is_initialized_ 分支）
auto* bm = warp_ctx->get_cta_context()->get_barrier_module().get_warp_barrier(0);
bm->init(participation_mask, reconvergence_pc, current_pc);  // 内部处理 is_initialized_
bm->arrive(lane_id);  // 调用 WarpBarrier::arrive
```

**理由**: 简化逻辑 — `is_initialized_` 分支集中在 `WarpBarrier::init`；handler 不需要重复此检查。

### Decision 4: `BarrierModule::release_warp_barrier` 已包含 BUG-POSTBARRIER-TWOHALVES 修复

**选择**: 保留 `barrier_module.cpp:111-113` 的 `set_active_mask(get_active_mask() | arrived_mask)` OR 逻辑

**理由**: 已 main commit `b04cdb2` 实施；handler 调用 `release_warp_barrier` 后无需在 handler 重复此逻辑。

### Decision 5: 不重写 `force_reconvergence_at_barrier`

**选择**: 保留 `force_reconvergence_at_barrier` 主逻辑；只迁移 barrier 状态管理 API

**理由**: `force_reconvergence_at_barrier` 是 SIMT Stack 操作，与 barrier 状态管理正交。Phase 5 工作只关心 barrier 状态如何被初始化/到达/释放。

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| commit `36dbb9a` 失败原因未完整理解，再次实施踩同一坑 | **高** | **高** | **实施前必须做 root cause 分析**（task 1.0）：git blame `36dbb9a` + `f033312` commit message + 当时失败测试 log |
| `WarpBarrier::init` is_initialized_ 分支破坏其他调用方 | 中 | 高 | task 1.1 全项目 grep `WarpBarrier::init` 调用点；逐个审计 |
| `force_reconvergence` 重新进入时 arrived_mask 累积错乱 | 中 | 高 | 新增 task 3.4 单元测试覆盖 `force_reconvergence + barrier.init twice` 场景 |
| 路径 A（分歧）与路径 B（正常）BarrierModule API 调用不一致 | 中 | 中 | 新增 task 3.5 集成测试覆盖两条路径都使用 BarrierModule API |
| `BsyncManager` 删除（`cleanup-deprecated-barrier-apis`）与本 change 冲突 | 中 | 中 | **强制顺序**：先 `cleanup-deprecated-barrier-apis`，后本 change（或同时进行同一 worktree） |
| `tests/integration/divergence/test_post_barrier_divergence.cpp` 已知 bug 测试失效 | 低 | 低 | 该测试已记录 BUG（`is_blocked` 不更新），由独立 change 处理 |

## Migration Plan

### Phase 0: 强制 Root Cause 分析（必做！）

> **来自 lessons-learned.md §1**：跨模块间接状态翻译是常见失败模式。commit `36dbb9a` 失败根因未完整记录。

- [ ] 0.1 阅读 commit `36dbb9a`（实施）与 `f033312`（revert）的 commit message + diff
- [ ] 0.2 阅读 `docs/dev-process/lessons-learned.md` 与 `docs/adr/0008-barrier-semantics.md` §2026-06-18 Postmortem
- [ ] 0.3 列出 commit `36dbb9a` 引入的具体代码变更（vs revert 后）
- [ ] 0.4 输出 `failure_root_cause.md` 文档：本 change 必须解决的具体 bug 列表
- [ ] 0.5 **如果 root cause 未明确，STOP 实施，等待更多调查**（不重蹈覆辙）

### Phase 1: 审计（半天）

- [ ] 1.0 完成 Phase 0 root cause 分析（MUST 在 1.1 之前）
- [ ] 1.1 全项目 grep `WarpBarrier::init` + `init_warp_barrier` + `arrive_at_warp_barrier` + `release_warp_barrier` 所有调用点
- [ ] 1.2 阅读 `force_reconvergence_at_barrier` 实现，确认其与 barrier 状态管理的交互点
- [ ] 1.3 阅读 `barrier.cpp::BarWarpSyncHandler::processOperation` 完整路径 A + 路径 B
- [ ] 1.4 输出 `warp_sync_audit.md`：列出所有需要修改的代码位置 + 风险评估
- [ ] 1.5 创建 worktree：`git worktree add ../ptx-emu-warp-sync -b feat/migrate-bar-warp-sync`
- [ ] 1.6 建立基线：`./scripts/sanity.sh --quick > baseline.txt`

### Phase 2: WarpBarrier::init 增强（半天）

- [ ] 2.1 修改 `src/ptxsim/barrier/warp_barrier.cpp::WarpBarrier::init`：增加 `is_initialized_` 分支处理（Decision 1）
- [ ] 2.2 验证：`cmake --build build --target ptxsim` 编译通过
- [ ] 2.3 新增 `tests/unit/barrier/test_warp_barrier.cpp` 测试 `WarpBarrier::init preserves arrived_mask when re-init`：init → arrive(0) → init again → 验证 arrived_mask 仍含 lane 0
- [ ] 2.4 `ctest -R "WarpBarrier::init preserves" -V` PASS
- [ ] 2.5 `ctest -R "post_barrier_reconvergence_simplegemm" -V` 不回归

### Phase 3: BarWarpSyncHandler 迁移（1 天）

- [ ] 3.1 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 路径 A（force_reconvergence）：
  - 替换 `warp_state.wbars[0]` 为 `warp_ctx->get_cta_context()->get_barrier_module().get_warp_barrier(0)`
  - 替换 `init_wbar.init(...)` 为 `bm->init(...)`（自动处理 is_initialized_ 分支）
  - 替换 `init_wbar.arrive(lane_id)` 为 `bm->arrive(lane_id)`
  - 替换 `init_wbar.is_complete()` 为 `bm->is_complete()`
  - 替换 `init_wbar.arrived_mask` 为 `bm->get_arrived_mask()`
  - 替换 `init_wbar.count_*` 为 `bm->get_*_count()`
- [ ] 3.2 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 路径 B（正常）：
  - 同样替换为 `BarrierModule` API
- [ ] 3.3 移除 `sm_ctx->bsync_manager_.bsync/release` 调用（依赖 `cleanup-deprecated-barrier-apis` 已完成）
- [ ] 3.4 验证：`cmake --build build && ctest -R "barrier" -V` 全部 PASS
- [ ] 3.5 新增 `tests/integration/divergence/test_post_barrier_two_halves_barrier_module.cpp`：覆盖分歧 warp 两半分别到达 barrier 场景，验证 BarrierModule API 路径下 barrier 正常完成（commit `36dbb9a` 失败案例的复现 + 修复验证）
- [ ] 3.6 `ctest -R "integration_post_barrier_two_halves" -V` PASS

### Phase 4: 全量回归（半天）

- [ ] 4.1 `./scripts/sanity.sh --quick` 全部 PASS；与 baseline.txt 对比，MUST NOT 新增 FAIL
- [ ] 4.2 `./scripts/sanity.sh` 完整回归 PASS；e2e 测试 `e2e_barrier_warp_sync` / `e2e_test3_cfg_full` 全部 PASS
- [ ] 4.3 `./tests/ptx/test_all_ptx.sh` 全部 PTX 语法测试通过

### Phase 5: 文档 + 发布（半天）

- [ ] 5.1 更新 `docs/adr/0008-barrier-semantics.md`：追加 §"2026-06-19 追加：BarWarpSyncHandler 迁移 + WarpBarrier::init 不变性"
- [ ] 5.2 更新 `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md`：描述 `BarrierModule` 统一管理 CTA + Warp barrier
- [ ] 5.3 在 worktree 中创建最终 commit：`git add . && git commit -m "feat(barrier): migrate BarWarpSyncHandler to BarrierModule API + fix WarpBarrier::init re-init semantics"`
- [ ] 5.4 合并到主分支：`git checkout main && git merge --no-ff feat/migrate-bar-warp-sync -m "Merge..."`
- [ ] 5.5 清理 worktree

### Rollback Strategy

> **来自 lessons-learned.md §4**：任何已有测试回归 → 立即 revert 该 Phase，不混入后续 commit

- **Phase 2 失败**（WarpBarrier::init 破坏）：git revert 到 Phase 1（仅审计）
- **Phase 3 失败**（BarWarpSyncHandler 破坏）：git revert 到 Phase 2 完成点（WarpBarrier::init 已增强但未使用）
- **任何 Phase 严重问题**：`git stash` + 报告用户 + 询问是否回滚
- **如 commit `36dbb9a` 失败模式重现**：立即停止，参考 postmortem 重做 root cause 分析

## Open Questions

1. **commit `36dbb9a` 失败的具体 6 个回归测试是哪些**？需 git log + ctest -V 输出复现
2. **`force_reconvergence_at_barrier` 是否在 BarrierModule 迁移后还需要独立管理 `current_wbar_id`/`barrier_active` 状态**？还是统一由 `is_initialized_` 替代？
3. **`WarpBarrier::needs_to_wait()` 两个重载（无参 vs 有参）是否都被新 handler 路径需要**？task 1.1 审计
4. **路径 A 的 arrived_mask 累积是否需要持久化到 BarrierModule 跨 `release` 调用**？当前决策是不需要，但需验证 `tests/integration/divergence/test_post_barrier_two_halves` 不依赖此
5. **`cleanup-deprecated-barrier-apis` 与本 change 的 worktree 是否合并**？推荐同步进行以减少集成风险