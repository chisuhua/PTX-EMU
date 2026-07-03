## Context

`migrate-bar-warp-sync-to-barrier-module` change（archived 2026-07-03）在删除 `Wbar` struct 时连带删除了覆盖 `BarrierModule` API 单元级语义的测试。归档后存在覆盖度缺口：

**已删除的覆盖**（commit `f5640042` Phase 7 中）：
- `tests/unit/sync/test_syncthreads_test3_repro.cpp`（-190 行）：覆盖 `init → is_complete → reset → re-init → is_complete` 生命周期 + 16-thread partial mask 完整周期
- `tests/unit/exec/test_exec_layer_e1_e3.cpp`（-59 行）：覆盖 32-thread full mask 中 31-arrival 不完整语义
- `tests/unit/barrier/test_barrier_verification.cpp`（-162 行部分）：覆盖 `arrive_at_warp_barrier()` 返回的 complete 标志

**现存的间接覆盖**：
- `tests/integration/divergence/test_post_barrier_two_halves.cpp`：通过 `ptxsim::testing::step_warp` 间接验证完整调度路径
- `tests/unit/barrier/test_barrier_module.cpp::WarpBarrier::init preserves arrived_mask`：单测 `WarpBarrier::init` 的 `is_initialized_` 分支

**覆盖率缺口**：
1. **`BarrierModule::release_warp_barrier` 的 5 项状态翻译**（`barrier_module.cpp:85-138`）：OR-on-active_mask + `is_blocked=false` + `status=Active` + `is_active=true` + `set_pc_overridden(true)` —— 这些是 `lessons-learned.md §1` "跨模块间接状态翻译" 在本次迁移中的关键不变式，目前**无单元级验证**
2. **`WarpBarrier::init → arrive → is_complete → release → reset → re-init` 完整生命周期** —— 单元级验证缺失
3. **`participation_mask` 边界条件**（partial-mask vs full-mask）—— 单元级验证缺失

### 关键文件

| 文件 | 现有职责 | 本 change 目标 |
|------|---------|-------------|
| `src/ptxsim/barrier/barrier_module.cpp:85-138` | `release_warp_barrier` 实现 OR + 5 项状态翻译 | **不修改**，新增单元测试直接断言每项翻译 |
| `src/ptxsim/barrier/warp_barrier.cpp` | `WarpBarrier::init/arrive/is_complete` | **不修改**，新增 lifecycle 测试覆盖 `init → arrive → complete → reset → re-init` |
| `tests/unit/barrier/AGENTS.md`（如果存在） | 已存在的 barrier 单元测试目录约定 | **不修改**，按现有模式追加 |
| `tests/CMakeLists.txt` | ctest 目标注册 | **修改**，注册 3 个新 `unit_*` 测试 |

## Goals / Non-Goals

**Goals:**
- 直接单元级验证 `BarrierModule::release_warp_barrier` 的 5 项状态翻译（防止未来误删其中任何一项）
- 直接单元级验证 `WarpBarrier` 的 lifecycle（init → complete → reset → re-init）
- 直接单元级验证 `participation_mask` 边界（16-lane partial、32-lane 中 31-arrival）
- 与现有 `WarpBarrier::init preserves arrived_mask` 测试互补 —— 形成 barrier module API 完整测试矩阵

**Non-Goals:**
- **不修改生产代码** —— 这是纯测试补完 change
- **不实现新功能** —— 不新增 public API、不改 behaviour
- **不重构现有测试** —— 现有 `test_barrier_module.cpp` 等保持原状
- **不补 e2e 测试** —— 现有 `e2e_barrier_warp_sync` / `e2e_test3_cfg_full` 已覆盖端到端
- **不验证 post-commit 仅在 change 内有 n 个测试的功能** —— 测试数量非目标，覆盖质量才是

## Decisions

### Decision 1: 3 个独立测试文件（而非一个大文件）

**选择**: 分别建 3 个测试文件 `test_barrier_module_release.cpp` + `test_warp_barrier_lifecycle.cpp` + `test_participation_mask_boundaries.cpp`，每个独立 `add_catch_test` 目标。

**理由**:
- **清楚划分测试域**：`release_warp_barrier` vs `WarpBarrier` lifecycle vs `participation_mask` 是 3 个独立的 invariant 轴
- **隔离失败定位**：未来某 invariant 破坏时，只有一个 `TEST_CASE` 失败，而非整个文件失败
- **符合 `tests/README.md` 测试分类规范**：单元测试按子领域分目录/分文件存放
- **避免"mega test file"** 阻塞 case 增多后维护成本上升

**替代方案考虑**:
- 单个 `test_barrier_module_full.cpp` 大文件 → 但单文件 > 300 行违反项目测试组织约定
- `TEST_CASE_METHOD` 共用 setup → 但 lifecycle 测试是顺序状态机，每个 case 需要独立 setup，共享 base class 反而复杂

### Decision 2: 直接调用 BarrierModule API（不通过 step_warp）

**选择**: 测试用例直接 `BarrierModule bm; bm.init_warp_barrier(...); bm.arrive_at_warp_barrier(...); bm.release_warp_barrier(...);` 构造，不用 `ptxsim::testing::step_warp` 间接驱动。

**理由**:
- **明确的"单元 vs 集成"分层**：本 change 目标是补**单元级**覆盖率，集成测试已经存在（`test_post_barrier_two_halves` 集成）
- **避免 SM/Warp/Thread setup 噪声**：直接构造 `BarrierModule` 对象 + 最小化 `WarpContext` mock（仅 `get_active_mask` / `set_active_mask` 等需要的接口）
- **精确断言每个状态字段**：`ts.is_blocked`、`ts.status`、`ts.is_active` 必须逐一断言，正是 lessons §1 "行级 diff" 的实践

**替代方案考虑**:
- 通过 `execute_warp_instruction` → `step_warp` 间接验证 → 不符合单元测试规范（参考 `tests/README.md` "类型一：直接单元测试"）
- mock 整个 `CTAContext` → 过度 mock，违反"测真实行为"原则

### Decision 3: set_pc_overridden(true) 是 release 调用方契约，不在 BarrierModule 内断言

**选择**: 测试用例**不**断言 `BarrierModule::release_warp_barrier` 内部调用了 `set_pc_overridden(true)`（因为该调用本就不在 BarrierModule 内，而在调用方 `BarWarpSyncHandler`）。

**理由**:
- **`set_pc_overridden(true)` 是 handler 调用方的契约**（`barrier.cpp` 在调用 `release_warp_barrier` 后立刻调用），不在 `BarrierModule` 内
- **BarrierModule 单元测试只断言 BarrierModule 自己的输出**：state 字段更新 + active_mask OR 语义
- **handler 层契约验证**：留到 `tests/integration/barrier/` 集成测试（`bar_warp_sync_handler_release_pc_overridden` 之类的 case 已有 "test_barrier_full_lifecycle.cpp" 部分覆盖）

**替代方案考虑**:
- 在 `BarrierModule::release_warp_barrier` 内部调用 `set_pc_overridden(true)` → 反 lessons §2 "可重入安全：public 方法不应该再锁"。`set_pc_overridden` 涉及 ThreadContext，跟 BarrierModule 的关注点分离
- 测试 mock ThreadContext → 过度 mock

### Decision 4: WarpContext mock 最小化

**选择**: 测试用例构造的 `WarpContext` 使用真实的 mock 或简化 fake（`warp_state.threads[0..31]` 仅设置必要字段），不依赖完整 `SMContext` / `CTAContext` 栈。

**理由**:
- **隔离 BarrierModule 测试**：本 change 是 BarrierModule 单元测试，不是 WarpContext 测试
- **`BarrierModule::release_warp_barrier` 调用 WarpContext 的 5 个接口**：`get_active_mask()`、`set_active_mask(uint32_t)`、`get_warp_state().threads[i]`、`advance_thread_pc`（如果使用）。最小 mock 只暴露这些接口
- **现有 `test_barrier_module.cpp` 模式一致**：参考 `WarpBarrier::init preserves arrived_mask` 测试如何构造场景

**替代方案考虑**:
- 构造完整 `SMContext` + `CTAContext` + 真实 `WarpContext` → 违反"单元测试"定义，需要 GPU/SM/CTA/Warp/Thread 全部装配，重过
- 把 BarrierModule 改为可注入 warp_context 接口 → 改 API，超出本 change 范围

## Risks / Trade-offs

| 风险 | 概率 | 影响 | 缓解 |
|------|------|------|------|
| WarpContext mock 设计复杂 → 测试失败 | 中 | 中 | 参考 `test_barrier_module.cpp` 现有模式；若失败降级为"需要构造完整 WarpContext"（变成集成测试而非单元测试） |
| `release_warp_barrier` 未来重构（如改为多 barrier 同步）→ 测试断言失效 | 低 | 低 | 用 `get_arrived_mask` / `get_expected_count` 等公共 API 断言，避免断言私有字段 |
| `participation_mask` 边界测试揭示 production bug（例如 mask 全 0 时 `is_complete()` 永不 true）| 低 | 中 | 该 bug 应作为新紧急 change 修复；本 change 只补测试，不修 production |
| 测试通过后 production 行为仍回归（如 `BarrierModule::release_warp_barrier` 被误改） | 低 | 高 | 单元测试 + 集成测试双层保护；任何 release_warp_barrier 改动必须同时通过 `test_barrier_module_release` 和 `test_post_barrier_two_halves` |
| tag 命名违反项目约定（`[unit;barrier]` vs 其他） | 极低 | 低 | 参考 `tests/unit/barrier/CMakeLists.txt` 现有 `add_catch_test` 调用，确保 label 一致 |

## Migration Plan

### Phase 1: 测试文件创建（单 commit）

```bash
# 1. 创建 3 个测试文件
git add tests/unit/barrier/test_barrier_module_release.cpp \
        tests/unit/barrier/test_warp_barrier_lifecycle.cpp \
        tests/unit/barrier/test_participation_mask_boundaries.cpp \
        tests/unit/barrier/CMakeLists.txt
git commit -m "test(barrier): add direct unit coverage for BarrierModule release + lifecycle + mask boundaries

Closing coverage gap identified in migrate-bar-warp-sync-to-barrier-module
review (Issue I1). 8 unit tests across 3 files:

1. test_barrier_module_release.cpp (3 cases):
   - release_warp_barrier_OR_active_mask
   - release_warp_barrier_resets_is_blocked_status_is_active
   - release_warp_barrier_idempotency_within_cycle

2. test_warp_barrier_lifecycle.cpp (3 cases):
   - init_arrive_complete_reset_reinit_full_cycle
   - re_init_preserves_arrived_mask_for_force_reconvergence (BUG-RECONVERGENCE-SIMPLEGEMM)
   - multiple_completion_cycles_no_state_leak

3. test_participation_mask_boundaries.cpp (2 cases):
   - full_mask_32_arrive_31_is_incomplete
   - partial_mask_16_all_arrive_completes_at_16

Refs: migrate-bar-warp-sync-to-barrier-module review I1;
      lessons-learned §1, §19"
```

### Phase 2: 注册 + 验证（单 commit）

```bash
# 1. 注册 3 个 add_catch_test 目标
# 2. cmake --build build --target unit_barrier_module_release 验证编译
# 3. ctest -R "unit_barrier_module_release|unit_warp_barrier_lifecycle|unit_participation_mask" -V 验证 PASS
# 4. ./scripts/sanity.sh --quick 验证无回归
git add tests/CMakeLists.txt  # if needed
git commit -m "test(barrier): register BarrierModule lifecycle + mask boundary tests"
```

### Phase 3: 文档同步（如有）

- `docs/adr/0008-barrier-semantics.md` 可能追加一行注释指向新测试（不强制）
- `tests/unit/barrier/AGENTS.md`（如存在）列出新测试文件

### Rollback Strategy

单个 commit。如发现测试设计错误或揭示 production bug：

```bash
git revert HEAD --no-edit
# 或
git reset --hard HEAD~1   # 仅当未 push
```

不涉及生产代码，回滚成本极低。

## Open Questions

1. **`BarrierModule::release_warp_barrier` 内部多 barrier ID 的语义？** 当前 1 个 BarrierModule 持有 4 个 WarpBarrier slot，release 哪个由 `warp_barrier_id` 参数决定。测试是否覆盖多 slot 场景，还是只测 slot 0？—— **建议**：本 change 只测 slot 0（最常见路径），多 slot 场景留给后续独立 change

2. **`WarpContext::advance_thread_pc` 是否在 release_warp_barrier 内部调用？** 当前代码（`barrier_module.cpp:128`）只设置 state 字段，不调用 `advance_thread_pc`（PC 翻转由 `set_pc_overridden(true)` + handler 后续 `commit_pc()` 完成）。测试是否需要 mock `advance_thread_pc`？—— **建议**：不 mock；测试只断言 release 后的 state，不追踪 PC 翻转

3. **CTA barrier（`release_cta_barrier`）的对称测试是否在范围内？** 当前 change 标题是 "barrier-module-lifecycle"，可包含 CTA barrier。 但 CTA barrier 测试复杂（涉及 thread mutex），单独 follow-up 更合适。 —— **建议**：本 change 只覆盖 warp barrier，CTA barrier 留给独立 change

## References

- Proprosal: [`proposal.md`](./proposal.md)
- 前置 change（已归档）：`openspec/changes/archive/2026-07-03-migrate-bar-warp-sync-to-barrier-module/`
- ADR-0008：`docs/adr/0008-barrier-semantics.md` §2026-06-18 + §2026-07-03
- Skill：`ptx-barrier-mechanism`、`ptx-lessons-learned`（§1/§18/§19）、`test-coverage-enforcer`
- Code Review Issue I1：`migrate-bar-warp-sync-to-barrier-module` review issues I1：lifecycle 单元测试缺失
