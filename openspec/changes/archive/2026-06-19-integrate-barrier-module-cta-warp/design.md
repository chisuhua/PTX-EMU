## Context

PTX-EMU 当前有两条 barrier 实现路径并存：

1. **生产路径（active）**：
   - `bar.warp.sync` → `src/ptxsim/instructions/barrier.cpp:108-278 BarWarpSyncHandler::processOperation` → 操作 `warp_state.wbars[0]`（旧 `Wbar` 结构体）+ `sm_context->bsync_manager_.bsync/release()`
   - `bar.sync` (CTA 级) → `src/ptxsim/instructions/barrier.cpp:333-384 BarHandler::executeBarrier` → `src/ptxsim/core/sm_context.cpp:605-706 SMContext::synchronize_barrier` → 全局 `barrier_waiting_threads[barId]` map（mutex 保护）

2. **孤儿路径（designed, never wired）**：
   - `include/ptxsim/barrier/{barrier_module,warp_barrier,cta_barrier,barrier_types}.h` + `src/ptxsim/barrier/{barrier_module,warp_barrier,cta_barrier}.cpp`（413 行）
   - `BarrierModule` 统一管理 4 个 `WarpBarrier` + 16 个 `CTABarrier`，与 NVIDIA 硬件 16-named-barrier 对齐
   - 已加入 `ptxsim.so` 构建（`src/CMakeLists.txt:110-112`），但 `BarHandler` / `BarWarpSyncHandler` 都不调用

### 现状问题

1. **413 行死代码**：`BarrierModule` 系列代码 0% 被生产路径调用，但编译进 `ptxsim.so`
2. **handler 已知 bug**：`BarHandler::executeBarrier` 释放线程时设置 `state=RUN` + `next_pc=pc+1` 但**未调用 `commit_pc()`** 推进 `warp_state.threads[].pc`；`tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184` 用 `advance_thread_pc` work-around 掩盖
3. **遗留备份**：`src/ptxsim/instructions/barrier.cpp.bak` + `barrier.cpp.orig`（Apr 11 重构残留）
4. **文档错位**：
   - `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md` 详述旧路径，完全忽略新模块
   - `src/ptxsim/AGENTS.md` 不含 `barrier/` 子目录说明
   - `docs/technical_design/barrier_module_design.md` 状态"草稿"，但已实现

### 关键文件

| 文件 | 当前职责 | 目标职责 |
|------|---------|---------|
| `src/ptxsim/instructions/barrier.cpp` | 调用旧 `Wbar` + `SMContext::synchronize_barrier` | 调用 `BarrierModule` API |
| `src/ptxsim/core/sm_context.cpp` | 含 `synchronize_barrier` + 全局 barrier map | 移除 barrier 状态；仅保留 SM 级调度 |
| `src/ptxsim/core/cta_context.cpp` + `.h` | CTA 生命周期 + shared mem | 增加持有 `BarrierModule` 实例 |
| `src/ptxsim/core/warp_context.cpp` | 操作 `warp_state.wbars[]` + `bsync_manager` | 调用 `BarrierModule::init/arrive/release_warp_barrier` |
| `src/ptxsim/core/warp_state.h` | 含 `std::array<Wbar, 4> wbars;` + `current_wbar_id` | 移除这两个字段 |
| `include/ptxsim/wbar.h` | 旧 `Wbar` 结构体 | **删除** |
| `src/ptxsim/barrier/*.cpp` | 已实现但无调用者 | 扩展 `release_cta_barrier` 真正释放线程；成为生产路径 |

## Goals / Non-Goals

**Goals:**
- 单一来源：所有 barrier 操作通过 `BarrierModule` API（包括 CTA 和 Warp 两种 scope）
- handler bug 修复：删除测试 work-around，证明 `BarHandler` 在没有测试代码辅助下也能正确释放线程
- 代码-文档对齐：所有描述当前 barrier 实现的文档必须与代码一致
- 死代码归零：删除 `Wbar` + `bsync_manager_` + 备份文件后无残留
- 集成测试可移植：测试不再依赖"测试代码手动调用 advance_thread_pc"的反模式

**Non-Goals:**
- 不实现 `bar.warp.sync` membermask 的 UB 检测（已记录在 ADR-0008 未来工作）
- 不实现 Cluster barrier (sm_90+) / mbarrier / 显式 membar.fence（Hopper/Blackwell 路线图）
- 不重写 `bsync_manager_` 行为本身（如果其行为正确，只是迁移调用方）
- 不扩展 `MAX_WARP_BARRIERS` 从 4 → 16（`bar.warp.sync` 是单次使用屏障，4 槽足够；只有 CTA `bar.sync` 需要 16 named 槽）

## Decisions

### Decision 1: BarrierModule 由 CTAContext 持有（非 SMContext）

**选择**: 每个 `CTAContext` 持有一个 `BarrierModule` 实例

**理由**:
- CTA 是 barrier 的自然作用域（bar.sync 在 CTA 内同步）
- 当前 `SMContext::synchronize_barrier` 用全局 mutex + map 模拟 CTA 作用域，**是绕路实现**
- `CTAContext` 生命周期与 launch 的 block 一致，barrier 状态随 CTA 销毁/重置
- 不需要改动 SM 调度逻辑

**替代方案**:
- ❌ 由 `SMContext` 持有：需要 key-by-CTA 的复杂索引，违反 SM 与 CTA 的职责划分
- ❌ 由 `WarpContext` 持有：barrier 是 CTA 级，warp 看不到其他 warp 状态

### Decision 2: 旧 `bsync_manager_` 行为映射到新 `BarrierModule`

**选择**: 让 `BarrierModule::release_warp_barrier` 承担原 `bsync_manager_.release` 的角色

**理由**:
- `bsync_manager_.release(wbar_id)` 当前负责：清空 wbar 状态 + 推进 PC；新 `release_warp_barrier` 已经实现 `advance_thread_pc + set_exec_mask`，职责一致
- 删除 `bsync_manager_` 减少一个间接层（之前路径：`BarWarpSyncHandler` → `bsync_manager_.release` → 内部释放逻辑，新路径：`BarWarpSyncHandler` → `BarrierModule::release_warp_barrier` → 直接释放）

**风险**: `bsync_manager_` 是否承担了 `BarrierModule` 没承担的副作用？需要审计 `src/ptxsim/core/bsync_state.cpp` 的所有方法

### Decision 3: `release_cta_barrier` 需要 CTAContext 参数

**选择**: 修改 `BarrierModule::release_cta_barrier(int cta_barrier_id)` → `release_cta_barrier(int cta_barrier_id, CTAContext* cta_ctx)`

**理由**:
- 当前 `release_cta_barrier` 只 reset barrier 状态（不操作线程）—— 这就是它从未被生产调用的根本原因
- 真正的 release 需要遍历 `arrived_threads_` set，对每个 `ThreadContext*` 调用 `set_state(RUN)` + `advance_thread_pc(lane, post_barrier_pc)`
- 这与 `SMContext::synchronize_barrier` 当前做的事一致（行 663-682），但改为通过 `CTAContext` 访问

**API 签名变更**:
```cpp
// Before
void release_cta_barrier(int cta_barrier_id);
// After
void release_cta_barrier(int cta_barrier_id, CTAContext* cta_ctx);
```

### Decision 4: `warp_state.h` 移除 `wbars` 和 `current_wbar_id` 字段

**选择**: 完整迁移到 `BarrierModule`，不保留双轨

**理由**:
- `Wbar` 字段全部迁移到 `WarpBarrier` 类，行为等价
- `current_wbar_id` 是 `warp_state` 层的"现在是第几号 wbar 在用"状态，新 `BarrierModule` 用 `is_initialized()` 查询，无需独立字段

**回退策略**: 若集成测试发现未覆盖的 `wbars` 访问点，**回退决策 4**（保留字段但标记 deprecated），不阻断主流程

### Decision 5: 测试 work-around 直接删除，不"软化"

**选择**: 删除 `tests/integration/barrier/test_cta_barrier_memory_visibility.cpp:138-184` 的 47 行手动 `advance_thread_pc` 代码

**理由**:
- 修复后的 `BarHandler` 应该独立完成 PC 推进
- 如果删除后测试失败，证明 handler 修复不完整（这是好事，暴露问题）
- 工作绕过代码掩盖了真实问题，与"测试驱动开发"原则冲突

**回退策略**: 如果有未知的并发竞态使得修复后仍需 work-around，改为**修复 test setup** 而非保留 work-around

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| 删除 `Wbar` 后某处代码仍引用导致编译错误 | 中 | 高 | 实施前用 `grep -rn "Wbar\b\|wbar\.\|wbars\["` 全项目搜索；逐步迁移而非一次性删除 |
| `bsync_manager_` 有未记录的副作用被忽略 | 中 | 中 | 实施前读 `src/ptxsim/core/bsync_state.cpp` 全部方法；如果有 `BarHandler` 用不到的副作用，直接删除 |
| `release_cta_barrier` 集成后线程释放时机不对导致 deadlock | 中 | 高 | 实施后立即跑 `tests/integration/barrier/test_barrier_full_lifecycle.cpp` + `test_cta_barrier_memory_visibility.cpp`；失败则回滚到选项 C（仅文档+清理）|
| 单 warp CTA 路径不经过 `bar.sync` (PTX parser 优化) 掩盖新实现 bug | 中 | 中 | 增加 `tests/integration/barrier/test_cta_barrier_multi_warp_only.cpp` 强制 ≥2 warp CTA 走 `bar.sync` 而非 `bar.warp.sync` 优化路径 |
| 删除 `barrier.cpp.bak` 后 git history 丢失 | 低 | 低 | 验证 `.git/objects` 中仍可恢复；如果担心，提前 `git mv barrier.cpp.bak archive/` |
| `warp_state.h` 字段删除破坏其他 module 的 ABI | 低 | 中 | 用 `git grep -n "current_wbar_id\|warp_state.wbars"` 全项目审计；如果有外部依赖，保留字段但 deprecated 警告 |
| 集成测试 `test_post_barrier_divergence.cpp` 仍在记录 active_mask 更新问题（已知 BUG）| 低 | 低 | 在设计文档备注该测试由其他 change 处理；本 change 不主动修复 |
| `CTAContext` 新增成员影响 CTA 创建/销毁的对称性 | 低 | 中 | 在 `CTAContext` 构造函数初始化 `BarrierModule`，析构时依赖 unique_ptr 自动清理；编写 `tests/unit/cta/test_cta_context_lifecycle.cpp` 验证 |
| `sm_context.cpp:200-260` 周期 barrier 检查逻辑依赖 `barrier_mutex_` 与全局 `barrier_waiting_threads` map | 中 | 高 | 完整 audit `sm_context.cpp` 中所有 `barrier_mutex_` 引用；删除时同时移除 `barrier_mutex_` 字段与周期检查代码块；CTA 同步由 `CTAContext::barrier_module_` 完全接管 |

## 已实现的 Phase 5 推迟（2026-06-18）

> 详细 postmortem 见 [`docs/adr/0008-barrier-semantics.md` §2026-06-18 Postmortem](../../../docs/adr/0008-barrier-semantics.md#2026-06-18-postmortemplase-5-推迟决策) 和 [`docs/dev-process/lessons-learned.md`](../../../docs/dev-process/lessons-learned.md)。

**实际状态**：
- Phase 5 实施 commit `36dbb9a` 引入 6 个分歧/集成测试回归（基线 `00f698f` 通过）
- 已在 commit `f033312` 中 revert
- `Wbar` 标记 `[[deprecated]]`，旧 `BarWarpSyncHandler` 路径仍为生产路径
- 旧 `bsync_manager_` 保留（Phase 5.4 推迟）

**推迟原因**：
- `BarWarpSyncHandler` 涉及 `force_reconvergence` 路径与正常 barrier 释放路径的交互
- 单 PC（非分歧）barrier 路径通过
- 分歧场景（lanes 0-15 vs 16-31 走不同路径后到达同一 barrier）的 `force_reconvergence` 重置 + arrived 计数交互有问题
- 未能在本次 change 中彻底解决，强行合并会引入更多 bug

**未来实施指引**：
- 单独建 change：`migrate-bar-warp-sync-to-barrier-module`
- 优先解决分歧场景的 `force_reconvergence` + barrier 计数交互
- 实施前必读 `lessons-learned.md` §1（跨模块间接状态翻译）和 §4（分 Phase commit 纪律）

**本次 change 完成的实际范围**：
- ✅ Phase 1-4：BarHandler（CTA 路径）切换到 BarrierModule
- ✅ Phase 6：保留 `Wbar` 但标记 `[[deprecated]]`（部分完成）
- ❌ Phase 5：BarWarpSyncHandler 切换推迟
- ❌ `bsync_manager_` 删除推迟
- ❌ `warp_state.wbars[]` 字段删除推迟

---

## Migration Plan

### Phase 1: 准备工作（半天）
1. 全项目 grep 旧 API 使用点
2. 阅读 `bsync_state.cpp` 全部方法做行为清单
3. 备份关键文件到 `.bak2026XXXX`（.git 提交后立即恢复）
4. 在 worktree 中工作（避免污染主分支）

### Phase 2: BarrierModule 扩展（半天）
5. 修改 `BarrierModule::release_cta_barrier` 签名 + 实现
6. 修改 `CTAContext` 持有 `BarrierModule`
7. 新增 `tests/unit/barrier/test_barrier_module.cpp` 的完整流程覆盖

### Phase 3: BarHandler 切换（1 天）
8. 修改 `BarHandler::executeBarrier` 调用 `cta_ctx->barrier_module_.arrive_at_cta_barrier(barId, context)`
9. **关键验证**：删除 `test_cta_barrier_memory_visibility.cpp:138-184` work-around，跑测试
10. 修复任何因 handler bug 修复而暴露的问题

### Phase 4: BarWarpSyncHandler 切换（1 天）
11. 修改 `BarWarpSyncHandler::processOperation` 调用 `warp_ctx->get_cta_context()->get_barrier_module().arrive_at_warp_barrier`
12. 保留 force_reconvergence 路径（其行为正确，只是迁移调用）
13. 测试 `tests/integration/barrier/test_warp_barrier_integrated.cpp` + `test_post_barrier_two_halves.cpp`

### Phase 5: 旧代码删除（半天）
14. 删除 `include/ptxsim/wbar.h`
15. 从 `warp_state.h` 移除 `wbars[]` + `current_wbar_id`
16. 从 `sm_context.cpp` 移除 `synchronize_barrier` + barrier 状态
17. 从 `CMakeLists.txt` 移除旧引用（如有）
18. 删除 `barrier.cpp.bak` + `barrier.cpp.orig`

### Phase 6: 文档同步（半天）
19. 重写 `04-ptx-emu-current-implementation.md`
20. 更新 `barrier_module_design.md` 状态
21. 更新 ADR-0008 追加 `BarrierModule` 集成
22. 更新 `src/ptxsim/AGENTS.md` + `src/ptxsim/instructions/AGENTS.md`

### Phase 7: 验证（半天）
23. `./scripts/sanity.sh --quick` → 关键 bug 检查
24. `./scripts/sanity.sh` → 完整回归
25. `cmake --build build && ctest --output-on-failure` → 全测试通过

### Rollback Strategy

如果 Phase 7 失败：
- **Phase 3-5 失败**：git revert 到 Phase 2 完成点（BarrierModule 扩展 + 测试通过，但未切换 handler）
- **Phase 2 失败**：git revert 到 Phase 1（仅文档+审计）
- **任何阶段严重问题**：`git stash` + 报告用户 + 询问是否回滚

## Open Questions

1. **`bsync_manager_` 在 `bsync_state.cpp:14-24` 的 `bsync(thread_id, lane_id, pc)` 是否有 BarrierModule 不承担的副作用？** 需在 Phase 1 完整审计
2. **`tests/integration/divergence/test_post_barrier_divergence.cpp` 标注的 2 个 TEST_CASE（active_mask 更新问题）是否与本次修复相关？** 决策：本次不主动修复，但需要在 tasks.md 中标注后续工作
3. **`force_reconvergence_at_barrier` 空操作的"设计哲学"在新架构中是否需要保持？** 当前依赖副作用 `is_blocked=true` 让调度器跳过，需要验证新路径是否仍满足此不变量
4. **`include/ptxsim/barrier/warp_barrier.h` 中 `WarpBarrier::needs_to_wait()` 重载（无参 vs 有参）两个 API 是否都被新调用者需要？** 需审计
