## Context

PTX-EMU 在 commit `12390b7`（merge `fix/barrier-architecture-migration`）后：
- ✅ `BarHandler::executeBarrier`（CTA 路径）已切到 `BarrierModule` API（`barrier.cpp:344-392`）
- ✅ `CTAContext` 持有 `BarrierModule` 实例（`cta_context.cpp:31`）
- ✅ `BarrierModule::release_cta_barrier(barId, CTAContext*, post_pc)` 真正释放线程（`barrier_module.cpp:161-200`）
- ✅ `Wbar` 标记 `[[deprecated]]`（commit `83be5f7`）但**未删除**

**遗留问题（Phase 6 待办）**：

1. **`Wbar` 字段仍存在于生产路径**
   - `warp_state.h:17`: `std::array<Wbar, 4> wbars;`
   - `warp_state.h:18`: `int current_wbar_id = -1;`
   - `barrier.cpp:162, 221` 等处仍在直接操作 `warp_state.wbars[0]`
   - `[[deprecated]]` 注解已发出警告，但无新代码使用 → 字段实际上**只是历史包袱**

2. **`BsyncManager` 仍被调用**
   - `include/ptxsim/bsync_state.h:36-57` 定义 `BsyncManager` 类
   - `src/ptxsim/core/bsync_state.cpp` 实现 `bsync/release/check_release/get_state` 等方法
   - `sm_context.h` 含 `BsyncManager bsync_manager_` 字段
   - `barrier.cpp:189, 240, 249` 调用 `sm_ctx->bsync_manager_.release(0)` / `.bsync(...)`

3. **`SMContext::synchronize_barrier` 已死代码**
   - `sm_context.cpp:605-706` 实现 `synchronize_barrier(int barId, ThreadContext* thread)`
   - 但 `BarHandler::executeBarrier`（commit `b04cdb2` 后）已不再调用它
   - 全局状态 `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_` 占用资源但无功能

4. **Wbar/BsyncManager 关系**：当前 `Wbar` 的 `arrive/is_complete` 由 `barrier.cpp` 直接调用；`BsyncManager` 维护一个**并行**的 `waiting_threads_mask` map（`bsync_state.cpp:14-24`）。两套状态本质重复 — `Wbar.arrived_mask` 已经记录到达，但 `BsyncManager.waiting_threads_mask` 再次记录。

### 关键文件

| 文件 | 当前职责 | 目标职责 |
|------|---------|---------|
| `include/ptxsim/wbar.h` | `Wbar` 结构体（`[[deprecated]]`） | **删除** |
| `src/ptxsim/core/bsync_state.{h,cpp}` | `BsyncManager` 类（封装 wbar 全局 map） | **删除** |
| `include/ptxsim/warp_state.h` | 含 `wbars[]` + `current_wbar_id` | 移除两字段 + `#include "ptxsim/wbar.h"` |
| `src/ptxsim/core/sm_context.h` + `.cpp` | 含 `BsyncManager bsync_manager_` + `synchronize_barrier()` + 全局 map | 移除全部 |
| `src/ptxsim/instructions/barrier.cpp` | `BarWarpSyncHandler` 操作 `warp_state.wbars[0]` + `sm_ctx->bsync_manager_` | 移除 `bsync_manager_` 调用；**保留** `warp_state.wbars[0]` 操作（Phase 5 工作） |

## Goals / Non-Goals

**Goals:**
- 单一来源：CTA 级 barrier 状态全部由 `CTAContext::barrier_module_` 持有（已实现）；Warp 级 barrier 状态由 `WarpContext` 直接持有 `WarpBarrier`（Phase 5 工作）
- 死代码归零：删除 `Wbar` + `BsyncManager` + `synchronize_barrier` 后无残留
- 全局 mutex 移除：`barrier_mutex_` 删除后 SM 级无 mutex
- 测试可移植：单元测试不再需要 `BsyncManager` 间接层

**Non-Goals:**
- **不修改 `BarWarpSyncHandler::processOperation` 主逻辑**：本次只删除 `BsyncManager` 类，**保留** `warp_state.wbars[0]` + `warp_state.current_wbar_id` 的直接操作。Phase 5 工作（`migrate-bar-warp-sync-to-barrier-module`）处理完整迁移
- **不实现 `bar.warp.sync` membermask 的 UB 检测**（已记录在 ADR-0008 未来工作）
- **不实现 Cluster barrier (sm_90+) / mbarrier**（Hopper/Blackwell 路线图）
- **不重写 warp barrier 行为**（仅删除 `BsyncManager` 间接层，warp barrier 行为不变）

## Decisions

### Decision 1: 删除 `BsyncManager` 类，但 `BarWarpSyncHandler` 仍用 `Wbar`

**选择**: 删除 `BsyncManager` + `bsync_state.{h,cpp}`，但 `BarWarpSyncHandler` 中调用 `sm_ctx->bsync_manager_.bsync/release` 的地方改为**空操作**或**直接读取 `warp_state.wbars[0]` 状态**

**理由**:
- `BsyncManager` 当前职责与 `Wbar` 重复：`Wbar.arrived_mask` 已经记录到达，`BsyncManager.waiting_threads_mask` 是重复记录
- 删除 `BsyncManager` 不影响 `BarWarpSyncHandler` 主流程（仅失去 `bsync_manager_.release()` 的"清理释放"副作用，但 `Wbar` 本身会被 reset）
- 把 `BarWarpSyncHandler` 的真实 `BarrierModule` 迁移留给 Phase 5 单独 change

**风险**: 如果 `bsync_manager_.release()` 有未记录的副作用（例如清理 `barriers_` map），删除后可能漏掉清理。需在实施前 read `bsync_state.cpp` 全部方法确认。

### Decision 2: 删除 SM 级全局 barrier 状态

**选择**: 删除 `barrier_waiting_threads` + `barrier_thread_counts` + `barrier_mutex_` 字段及其在 `sm_context.cpp:204-242` 的周期检查逻辑

**理由**:
- `SMContext::synchronize_barrier` 已无生产调用方（commit `b04cdb2` 后 `BarHandler` 走 `CTAContext`）
- `barrier_waiting_threads` map 是"绕路实现"：本应存于 CTA 级，现在存于 SM 级全局
- 删除后 `SMContext` 职责更聚焦（仅 warp 调度 + SM 级状态），CTA 同步完全由 `CTAContext::barrier_module_` 接管

**风险**: 如果 `exe_once` 中的周期 barrier 检查逻辑（`sm_context.cpp:204-242`）还有其他用途，需保留；本次完整审计后再删除。

### Decision 3 (revised): Wbar struct 与 `warp_state.wbars[]` 字段保留

**选择**: **保留** `Wbar` struct (`include/ptxsim/wbar.h`) 与 `warp_state.wbars[]` 字段 (`include/ptxsim/warp_state.h:17-18`)。

**理由**:
- `Wbar` 完整 API 由 `BarWarpSyncHandler` 在 9+ 处使用(`barrier.cpp:162, 178, 180-183, 221, 224, 227, 230, 236, 247, 256-263`)
- `WarpBarrier` 是 `class`(private 字段,getter API),与 `Wbar`(public 字段 struct) API 不兼容;无法做"字段类型替换"
- `WarpBarrier` 应由 `BarrierModule` 拥有(`include/ptxsim/barrier/barrier_module.h:74` 的 `std::array<WarpBarrier, MAX_WARP_BARRIERS=4>`),不能嵌入 `WarpState`
- ADR-0008 §"2026-06-18 Postmortem" 已明确 Phase 5 推迟决策;Phase 5 工作(`migrate-bar-warp-sync-to-barrier-module`)将完整迁移 `BarWarpSyncHandler`,届时再删 `Wbar`
- **本 change 边界**:仅删除 `BsyncManager` + `SMContext::synchronize_barrier` + SM 级全局 barrier 状态

**风险**: `Wbar` `[[deprecated]]` 警告持续存在,但实际使用无法消除。Phase 5 是唯一出路。

**实施影响**:
- `include/ptxsim/wbar.h` 不删除
- `include/ptxsim/warp_state.h` 不修改
- 19 个测试文件 include `ptxsim/wbar.h`,**全部保留**(无需迁移)
- `tests/integration/divergence/test_post_barrier_divergence.cpp` 已知 BUG 测试保留(Wbar 仍存在)

### Decision 4: 测试 work-around 不修改

**选择**: 不修改 `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp`（已被 `ad7a46f` 删除 work-around）

**理由**: 该 work-around 已在 main 删除（`ad7a46f` commit），当前 `BarHandler` 已独立完成 PC 推进。

### Decision 5: `warp_context.cpp:283-296` BAR_SYNC fallback 替换为 BarrierModule 调用

**选择**: 替换 `synchronize_barrier` 调用为 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(...)`。

**理由**:
- BAR_SYNC 状态仍有 2 个生产 setter(`barrier.cpp:386` + `sm_context.cpp:703`)+ 1 个翻译器(`thread_context.cpp:749` 的 `Blocked → BAR_SYNC` 翻译)
- `warp_context.cpp:283` 的 fallback 不是 dead code,必须保留
- 删除 `synchronize_barrier` 后,fallback 必须有替代实现
- `cta_context_` 已通过 `warp_ctx->get_cta_context()` 可访问(commit b04cdb2 引入)

**风险**: 替换后 CTA 同步路径与 BarHandler 路径走相同 API,理论上行为一致;但需要类型二/三测试覆盖(集成测试驱动调度器,验证 divergent 两半场景)

### Decision 6: CMakeLists 清理

**选择**: 从 `src/CMakeLists.txt` 移除 `ptxsim/core/bsync_state.cpp` 条目；**保留** `ptxsim/barrier/{warp_barrier,cta_barrier,barrier_module}.cpp`（已加入）

**理由**: barrier_module 系列是新生产路径，必须保留。

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| `warp_context.cpp:283-296` BAR_SYNC fallback 替换后漏掉 BAR_SYNC 翻译链(lessons-learned §1) | 中 | 高 | 复用既有 `fix-pre-p0-baseline` worktree 做 baseline 对比;Phase 2 实施后立即跑 e2e barrier 测试 + 已知 BUG regression 测试 |
| `BsyncManager` 有未记录的副作用被忽略 | 低 | 低 | 已 grep 验证:`is_waiting`/`get_waiting_mask`/`get_state` 在生产代码无消费者;`bssync`/`release` 是单向写入 |
| 删除 `bsync_manager_.release(0)` 后 `Wbar` 状态未 reset | 低 | 高 | `Wbar` 在 `BarWarpSyncHandler` 释放路径中显式调用 `wbar.reset()`（`barrier.cpp:224`、重构后需保留此逻辑） |
| 删除 `sm_context.cpp:204-242` 周期 barrier 检查逻辑时破坏其他功能 | 低 | 高 | 已 grep 验证:`barrier_waiting_threads` map 唯一写入方是 `synchronize_barrier`(将被删除);周期检查失去唯一数据源后是 dead code |
| `warp_context.cpp:292` 替换 `synchronize_barrier` 为 `BarrierModule::arrive_at_cta_barrier` 路径走通性 | 中 | 中 | `cta_context_` 已通过 `warp_ctx->get_cta_context()` 暴露(commit b04cdb2);`BarrierModule::arrive_at_cta_barrier` 已实现(`barrier_module.cpp:127`);类型二/三集成测试覆盖 |
| 测试目录大量引用 `Wbar`(`include/ptxsim/wbar.h`)导致需大量迁移 | 低 | 低 | **Wbar struct 保留**:19 个测试文件 include `ptxsim/wbar.h`,全部保留,无需迁移 |
| 删除 `BsyncManager` 后 `tests/unit/sync/test_bsync_state.cpp` 单元测试编译失败 | 中 | 低 | 删除整个测试文件(BsyncManager 类删除后该测试必须删除);非回归 |

## Migration Plan

**重要**:本 change 仅删除 `BsyncManager` + `SMContext::synchronize_barrier` + SM 级全局 barrier 状态。`Wbar` struct + `warp_state.wbars[]` 字段**保留**到 Phase 5 独立 change。

### Phase 1: 审计与准备（半天）
1. `grep -rn "Wbar\b\|\bwbar\.\|wbars\[\|bsync_manager_\|synchronize_barrier\|barrier_waiting_threads\|barrier_mutex_\|barrier_thread_counts" src/ include/ tests/ > /tmp/cleanup_audit.txt`
2. 阅读 `src/ptxsim/core/bsync_state.cpp` 全部方法(`bssy/bsync/check_release/release/get_state/is_waiting/get_waiting_mask/cleanup/reset`),确认每个 API 的引用点
3. 复用既有 `.worktrees/fix-pre-p0-baseline` 作为 baseline worktree(避免 15-20 分钟基线 build)
4. 在 baseline 中建立基线:`./scripts/sanity.sh > baseline.txt`,保存用于对比

### Phase 2 (Commit 1): 删除 BsyncManager + 同步 barrier.cpp 调用点（半天，独立可 revert）
5. `rm include/ptxsim/bsync_state.h src/ptxsim/core/bsync_state.cpp`
6. 从 `src/CMakeLists.txt` 移除 `ptxsim/core/bsync_state.cpp` 条目
7. 从 `include/ptxsim/sm_context.h` 移除 `BsyncManager bsync_manager_` 字段、`#include "ptxsim/bsync_state.h"`
8. 从 `src/ptxsim/core/warp_scheduler.cpp` 移除 `#include "ptxsim/bsync_state.h"`(验证无 BsyncManager 使用后)
9. 从 `src/ptxsim/instructions/barrier.cpp` 删除 lines 189, 240, 249 的 `sm_ctx->bsync_manager_.bsync/release` 调用
10. 删除 `tests/unit/sync/test_bsync_state.cpp`(BsyncManager 类删除后该测试必须删除)
11. 验证:`cmake --build build --target ptxsim` 编译通过;`ctest -L "barrier;warp"` 全部 PASS;`./scripts/sanity.sh --quick` 全部 PASS

### Phase 3 (Commit 2): 删除 SM 级 barrier 状态 + 替换 warp_context.cpp BAR_SYNC fallback（半天，独立可 revert）
12. 从 `include/ptxsim/sm_context.h` 移除 `barrier_waiting_threads` (line 189) / `barrier_thread_counts` (line 190) / `barrier_mutex_` (line 192) 字段 + `synchronize_barrier()` 声明 (line 114)
13. 从 `src/ptxsim/core/sm_context.cpp` 删除 lines 200-260 周期 barrier 检查代码块(含 `barrier_mutex_` lock at line 204)
14. 从 `src/ptxsim/core/sm_context.cpp` 删除 lines 605-706 `synchronize_barrier()` 方法体
15. **关键**:`src/ptxsim/core/warp_context.cpp:283-296` BAR_SYNC fallback 替换:
    - 删除 `sm_context_->synchronize_barrier(thread->bar_id, thread);` (line 292)
    - 改为 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(thread->bar_id, thread);`
    - `cta_context_` 通过 `warp_ctx->get_cta_context()` 获取(commit b04cdb2 引入)
    - **必须添加注释**:"替换 synchronize_barrier fallback (lessons-learned §1 BAR_SYNC 翻译链)"
16. 验证:`cmake --build build --target ptxsim` 编译通过
17. `ctest -L "barrier;warp"` 全部 PASS
18. `./scripts/sanity.sh --quick` 全部 PASS
19. `./scripts/sanity.sh` 完整回归 PASS
20. 已知 BUG 回归测试验证:`tests/integration/divergence/test_post_barrier_divergence.cpp` 仍 PASS

### Phase 4 (Commit 3): 文档同步（半天，独立可 revert）
21. `docs/adr/0008-barrier-semantics.md`: 追加 "2026-06-20 Phase 6 partial cleanup" 段落,记录:
    - `BsyncManager` 与 SM 级 barrier 状态删除
    - `Wbar` struct 保留到 Phase 5
    - `warp_context.cpp:283-296` BAR_SYNC fallback 替换为 `BarrierModule::arrive_at_cta_barrier`
    - 引用 commit `f033312` lessons-learned §1 BAR_SYNC 翻译链
22. `src/ptxsim/core/AGENTS.md`:
    - 删除 "Barrier sync \| `sm_context.cpp` \| `synchronize_barrier()`" 行(line 22)
    - 删除 KNOWN ISSUES 中 `synchronize_barrier() may not update active_mask` 注释(line 85)
    - 改为指向 `tests/integration/divergence/test_post_barrier_divergence.cpp` 作为 BUG 文档(known issue, 仍存在)
23. `src/ptxsim/AGENTS.md`:
    - line 42 注释更新:`BarWarpSyncHandler` still uses `Wbar` (Phase 5 deferred)
    - 保留 line 48 "DO NOT add new uses of Wbar struct" 警告(Wbar 仍存在)
24. `tests/AGENTS.md`: 更新 line 15 "barrier/Wbar 数据结构" 描述(保留);`bsync` 描述移到 `archive/` 或删除

### Phase 5: 最终验证（半天）
25. 验证:`grep -rn "bsync_manager_\|bsync_state\.h\|synchronize_barrier\|barrier_waiting_threads\|barrier_mutex_" src/ include/ tests/` 输出为空(MUST 零匹配)
26. 验证:`grep -rn "Wbar\b\|\bwbar\.\|wbars\[" src/ tests/` 仍有匹配(19 个测试文件保留 Wbar 引用)
27. `./scripts/sanity.sh --quick` 全部 PASS;与 baseline.txt 对比,MUST NOT 新增 FAIL
28. `./scripts/sanity.sh` 完整回归 PASS
29. `./tests/ptx/test_all_ptx.sh` 全部 PASS
30. 对比 baseline.txt,MUST NOT 新增 FAIL

### Rollback Strategy

- **Phase 2 (Commit 1) 失败**:`git revert HEAD` (Phase 2 是独立 commit,revert 后 Phase 3-5 仍可继续)
- **Phase 3 (Commit 2) 失败**:`git revert HEAD` (Phase 3 是独立 commit,revert 后 BsyncManager 仍为删除态,但 synchronize_barrier 恢复 → 编译仍通过)
- **Phase 4 (Commit 3) 失败**:`git revert HEAD` (纯文档 revert,无代码影响)
- **任何阶段严重问题**:`git stash` + 报告用户 + 询问是否回滚

## Open Questions

1. **`warp_context.cpp:292` 替换为 `BarrierModule::arrive_at_cta_barrier` 后,`bar_id` 的语义是否一致**?原 `synchronize_barrier(thread->bar_id, thread)` 的 `bar_id` 来自 thread 的 `bar_id` 字段;`BarrierModule::arrive_at_cta_barrier` 接受 `(cta_barrier_id, thread)`,语义是 CTA 级 barrier ID(0-15)。需要验证 `thread->bar_id` 是否就是 cta_barrier_id。
2. **`barrier.cpp:386` 的 `context->set_state(BAR_SYNC)` 是否会被 `warp_context.cpp:267` 的 `blocked_at_barrier` 检查正确识别**?已 grep 确认(`thread->get_state() == BAR_SYNC` 在 line 267 和 283 两处使用),但替换 fallback 后需 e2e 测试验证 divergent 两半场景。
3. **`tests/unit/sync/test_bsync_state.cpp` 删除后,是否有 e2e 测试覆盖 BsyncManager 行为**?e2e 测试 `e2e_barrier_warp_sync` 应已覆盖,但需验证。
4. **Phase 5 (`migrate-bar-warp-sync-to-barrier-module`) 计划**:`Wbar` struct 保留为本 change 的最终状态;Phase 5 独立 change 负责完整迁移。