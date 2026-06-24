# Plans Archive

> 已完成的实施计划归档。`docs/superpowers/plans/` 仅保留当前活跃计划。

## 归档原则

满足以下任一条件即归档:

- ✅ 所有任务 commit 已落地（git log 验证）
- ✅ 对应的 OpenSpec change 已 `chore(openspec): archive`（见 `openspec/changes/archive/`）
- ✅ 计划文件无活跃 TODO（仅保留决策记录）

## 归档索引

| 文件 | 完成日期 | 验证 |
|------|---------|------|
| `2026-05-04-simt-architecture-fix.md` | 2026-06-23 | T1-1..T1-5 commits (c3d8fd3, eeab07d, bcadf32, 95ffc23, 04a62c4) |
| `2026-05-05-barrier-ptx-integration-fix.md` | 2026-06-23 | BarrierModule 集成 (eb2195e) |
| `2026-05-05-simt-architecture-alignment.md` | 2026-05-26 | 计划内显式标记 "✅ 完成" |
| `2026-05-07-active-mask-consistency.md` | 2026-06-23 | T2-1 (8b1d23b, 5e0e315) |
| `2026-05-07-atomic-add-implementation.md` | 2026-06-13 | 4b73fd7 `feat(atomic): implement atomicAdd/And/Or/Xor/etc` |
| `2026-05-07-simt-architecture-cleanup.md` | 2026-06-23 | Phase 1/2/3 commits |
| `2026-05-07-test3-reproduction-cleanup.md` | 2026-06-23 | 45295cc `refactor(tests): remove tests/archive/` |
| `2026-06-06-ptx-emu-tier3-ptx-tests.md` | 2026-06-06 | 5 commits (058510c, c80e95a, e9e368a, 7e7fa2a, de5d83c) |
| `2026-06-07-ptx-emu-p2-enable-commented-tests.md` | 2026-06-07 | tests/unit/CMakeLists.txt P2-1/P2-3 re-enabled |
| `2026-06-07-ptx-emu-tier8.md` | 2026-06-07 | `integration_barrier_full_lifecycle` ctest entry |
| `2026-06-07-ptxsim-testing-helper-extraction.md` | 2026-06-07 | `memory_test_utils.h` (b27d218, 2bbb65e, 565e447, b656dbd) |
| `2026-06-22-phase2-critical-debt.md` | 2026-06-23 | T1-1..T1-5 commits + `phase-1-foundation` archived |
| `2026-06-23-phase3-critical-debt.md` | 2026-06-24 | T2-1/3/4/5/6/7 + Phase 3 changes archived; `c9256b5` synced to delivered state |

## 归档操作

- **方式**：`git mv`（保留完整 git 历史）
- **触发条件**：
  1. 计划所有任务已 commit
  2. 对应 OpenSpec change 已归档到 `openspec/changes/archive/`
  3. 计划文件无活跃 checkbox（`grep -cE '^- \[[ x]\]'` 全为 0 或已确认过时）
- **归档时机**：在下一个 Phase 计划创建前、或月度 housekeeping
- **不归档**：报告类文件（非计划），参考 `docs/audits/` 取代策略直接删除或移至 `docs/archive/`

## 已被取代/移除的文件

| 原文件 | 移除原因 | 取代物 |
|--------|---------|--------|
| `2026-05-07-code-evaluation-report.md` | 一次性代码评估报告（数据已 1.5 月陈旧） | `docs/audits/HEALTH-AUDIT-2026-06-21.md` |
