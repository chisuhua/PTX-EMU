## Why

`migrate-bar-warp-sync-to-barrier-module`（archived 2026-07-03, commits `0e311566`+`f5640042`+`0bab6487`）在删除 `Wbar` struct 时一并删除了覆盖 BarrierModule / WarpBarrier API 单元级语义的测试文件：

- `tests/unit/sync/test_syncthreads_test3_repro.cpp`（-190 行）覆盖 `init → is_complete → reset → re-init → is_complete` 生命周期
- `tests/unit/exec/test_exec_layer_e1_e3.cpp`（-59 行）覆盖 32-thread full mask 中 31-arrival 不完整语义
- `tests/unit/barrier/test_barrier_verification.cpp`（-162 行部分）覆盖 16-thread participation mask 边界

虽然 `tests/integration/divergence/test_post_barrier_two_halves.cpp` 通过 `step_warp` 间接验证了完整调度路径，且 `tests/unit/barrier/test_barrier_module.cpp` 包含 `WarpBarrier::init preserves arrived_mask` 单测，但**`BarrierModule::release_warp_barrier` 的 OR 语义、状态字段翻译 (`is_blocked=false`、`status=Active`、`is_active=true`)、`set_pc_overridden(true)` 调用方契约**目前**没有直接的单元测试**——只能依赖集成路径间接覆盖。

代码审查（`migrate-bar-warp-sync-to-barrier-module` review I1）明确指出该覆盖率缺口：`BarWarpSyncHandler` 迁移后的核心不变式（`BarrierModule::release_warp_barrier` 封装的 5 项状态翻译）没有 atomic-level 测试，未来若有人修改 `release_warp_barrier` 删减这些翻译，集成测试可能会晚一拍才反应过来。

## What Changes

- 新增 `tests/unit/barrier/test_barrier_module_release.cpp`：3 个单元测试覆盖 `BarrierModule::release_warp_barrier` 的 5 项状态翻译 (`is_blocked=false`、`status=Active`、`is_active=true`、OR-on-active_mask、`set_pc_overridden(true)`)
- 新增 `tests/unit/barrier/test_warp_barrier_lifecycle.cpp`：3 个单元测试覆盖 `WarpBarrier::init → is_complete → reset → re-init → is_complete` 生命周期
- 新增 `tests/unit/barrier/test_participation_mask_boundaries.cpp`：2 个测试覆盖 participation_mask 边界条件（partial-mask 16-lane、full-mask 32-lane 中 31-arrival 不完整）
- 不修改生产代码（这是 follow-up 测试补完，非性能/重构）
- 不破坏任何现有通过测试（净增加覆盖）

## Capabilities

### New Capabilities

- `barrier-module-unit-tests`: 提供 `BarrierModule` API（init/arrive/release）的单元级 lifecycle + state-translation 直接测试，与现有 `WarpBarrier::init preserves arrived_mask` 测试互补

### Modified Capabilities

- 无（不修改现有 spec-level 行为）

## Impact

| 类别 | 影响 |
|------|------|
| `tests/unit/barrier/` | **新增**：`test_barrier_module_release.cpp`（约 100 行）+ `test_warp_barrier_lifecycle.cpp`（约 80 行）+ `test_participation_mask_boundaries.cpp`（约 60 行） |
| `tests/CMakeLists.txt` | **修改**：3 个新 `add_catch_test` 目标，按 `unit_` 前缀 + `[unit;barrier]` 标签 |
| 生产代码 | **无修改** —— 这是测试补完 change，不是实现 change |
| `docs/adr/0008-barrier-semantics.md` | **可能追加** §2026-XX 注释："direct unit tests now cover release_warp_barrier state translation" |
| `docs/dev-process/lessons-learned.md` | **不追加** —— §19 已经覆盖了本次迁移的成功证据 |

## ⚠️ 风险与历史教训

来自 `docs/dev-process/lessons-learned.md` §18（OpenSpec artifacts 提交遗漏）和 `tests/README.md`：

1. **避免"测试覆盖度盲区"** —— 移除旧 API 时连带删了相关测试，没意识到这些测试是 BarrierModule API 层级的"独占验证"。这是 lessons §18 的变体 —— 删除代码前**先 grep "谁依赖这个 API"**。
2. **避免"集成测试掩盖单元缺口"** —— `tests/integration/divergence/test_post_barrier_two_halves.cpp` 通过 `step_warp` 间接覆盖，但若 `BarrierModule::release_warp_barrier` 修改后未触发集成测试场景，单元缺口就会沉默。
3. **每个测试 case 独立可还原** —— 与 §4 "复杂迁移必须分 Phase commit" 一致：每个 `TEST_CASE` 内只 setup 一个 barrier state，不跨 case 共享状态。
4. **tag 命名约定** —— 所有新测试必须 `[unit;barrier]` 双重标签（commit `ab55e06` 后的命名约束），ctest 目标名带 `unit_` 前缀。

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性

- [x] 不涉及代码迁移 —— 这是纯测试补完 change
- [x] 不涉及状态机修改
- [x] 不涉及 mutex 修改

### 多 Phase 推进

- [x] 不涉及多 Phase 推进 —— 单 change 单次提交
- [x] 无需基线 worktree（直接基于 main 即可）
- [x] 失败处理：任何单元测试失败 → 立即修复测试或（如果揭示 production bug）单独建紧急 change revert

### 文档同步

- [ ] AGENTS.md 同步：可能追加 `tests/unit/barrier/AGENTS.md`（如果已存在）
- [ ] ADR 追加段落：可能 §2026-XX 在 ADR-0008 提及"direct unit tests now cover release_warp_barrier"
- [ ] tasks.md：5 个原子 task（3 个测试文件 + CMakeLists + 验证）

## References

- 前置 change（已归档 2026-07-03）：`openspec/changes/archive/2026-07-03-migrate-bar-warp-sync-to-barrier-module/`
  - §2026-07-03 postmortem 提到"已知未完成 / lifecycle 单元测试" follow-up
  - Code Review Issue I1 列举了 3 个测试缺口
- ADR-0008（barrier 语义增强）：`docs/adr/0008-barrier-semantics.md` §2026-06-18 决策记录 OR logic 单点拥有者
- Skill：`ptx-barrier-mechanism`（屏障机制全解）—— 测试断言 point 必读
- Skill：`ptx-lessons-learned`（§18 OpenSpec artifacts 教训 + §19 跨模块状态翻译成功案例）
- Skill：`test-coverage-enforcer`（Wbar API 测试覆盖率保证）
