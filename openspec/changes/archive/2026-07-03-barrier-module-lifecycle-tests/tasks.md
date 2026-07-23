## 1. 测试设计与基线

- [x] 1.1 **MUST** 阅读现有 `tests/unit/barrier/test_barrier_module.cpp::WarpBarrier::init preserves arrived_mask` —— 摸清现有单测对 WarpBarrier / BarrierModule 的构造模式（mock WarpContext 程度、setup 方式）
- [x] 1.2 阅读 `src/ptxsim/barrier/barrier_module.cpp::release_warp_barrier`（L85-138）—— 列出 5 项状态翻译断言点
- [x] 1.3 阅读 `src/ptxsim/barrier/warp_barrier.cpp` —— 列出完整 public API 用于 lifecycle 测试断言
- [x] 1.4 **MUST** 建立基线：`cd build && ctest -R "barrier" -V > /tmp/baseline_barrier.txt`（应 23/23 PASS）

## 2. WarpBarrier lifecycle 单元测试

> **设计决策（来自 review）**：
> - Task 2.3 移除 —— BUG-RECONVERGENCE-SIMPLEGEMM re-init 语义已被 `test_barrier_module.cpp::WarpBarrier::init preserves arrived_mask on re-init` 完整覆盖，无需重复。
> - 最终保留 2 个 case：`init_arrive_complete_reset_reinit_full_cycle`（单次完整周期）+ `multiple_completion_cycles_no_state_leak`（3 次连续周期，验证 reset 不漏状态）。

- [x] 2.1 创建 `tests/unit/barrier/test_warp_barrier_lifecycle.cpp` 头文件与 TEST_CASE 框架
- [x] 2.2 实现 `init_arrive_complete_reset_reinit_full_cycle`：构造 WarpBarrier → init(mask=0xFFFFFFFF) → arrive(0..31) → is_complete() → reset() → re-init → arrive(0..31) → is_complete()
- [x] 2.3 ~~**已删除**：re-init preserves arrived_mask 不再重复实现 —— `test_barrier_module.cpp:167-201` 已完整覆盖~~
- [x] 2.4 实现 `multiple_completion_cycles_no_state_leak`：连续 3 次完整 cycle，每次都用 is_complete() 后立即 reset()，验证每次 reset 后 is_initialized()=false 且 arrived_count_=0
- [x] 2.5 编译验证：`cmake --build build --target unit_warp_barrier_lifecycle` 通过
- [x] 2.6 运行验证：`ctest -R "unit_warp_barrier_lifecycle" -V` 全部 PASS

## 3. BarrierModule::release_warp_barrier 状态翻译测试

- [x] 3.1 创建 `tests/unit/barrier/test_barrier_module_release.cpp` 头文件与 TEST_CASE 框架
- [x] 3.2 实现 `release_warp_barrier_OR_active_mask`：mock WarpContext 含 active_mask=0xFFFF0000，调用 release(mask=0x0000FFFF) → 验证 active_mask 变为 0xFFFFFFFF（OR 语义）
- [x] 3.3 实现 `release_warp_barrier_resets_is_blocked_status_is_active`：mock warp_state.threads[i].is_blocked=true 等 → release → 验证 is_blocked=false + status=Active + is_active=true
- [x] 3.4 实现 `release_warp_barrier_two_cycle_OR_preserves_first_half_lanes`：**模拟 BUG-POSTBARRIER-TWOHALVES 的两个独立周期** —— Cycle 1: init(lowerHalf mask) + arrive(lowerHalf) + release → active_mask=lowerHalf；Cycle 2: init(upperHalf mask) + arrive(upperHalf) + release → 验证 active_mask=0xFFFFFFFF（OR 保留第一半已释放的 lane）。**注意**：每次 release 都会调用 wbar.reset()，所以两个 cycle 必须是顺序独立的 init/arrive/complete 序列。
- [x] 3.5 编译验证：`cmake --build build --target unit_barrier_module_release` 通过
- [x] 3.6 运行验证：`ctest -R "unit_barrier_module_release" -V` 全部 PASS

## 4. participation_mask 边界条件测试

- [x] 4.1 创建 `tests/unit/barrier/test_participation_mask_boundaries.cpp` 头文件与 TEST_CASE 框架
- [x] 4.2 实现 `full_mask_32_arrive_31_is_incomplete`：mask=0xFFFFFFFF → arrive(0..30) → 验证 is_complete() == false
- [x] 4.3 实现 `partial_mask_16_all_arrive_completes_at_16`：mask=0x0000FFFF → arrive(0..15) → 验证 is_complete() == true 且 arrived_count=16
- [x] 4.4 编译验证：`cmake --build build --target unit_participation_mask_boundaries` 通过
- [x] 4.5 运行验证：`ctest -R "unit_participation_mask_boundaries" -V` 全部 PASS

## 5. CMake 注册

- [x] 5.1 在 `tests/unit/CMakeLists.txt`（**注意：不是 `tests/unit/barrier/CMakeLists.txt` —— barrier 子目录无独立 CMakeLists.txt**）添加 3 个 `add_catch_test` 目标（按 commit `ab55e06` 后的命名约定：`unit_*` 前缀）
- [x] 5.2 设置 `set_tests_properties` 标签为 `[unit;barrier]`
- [x] 5.3 完整构建：`cmake --build build --target unit_barrier_module_release unit_warp_barrier_lifecycle unit_participation_mask_boundaries` 通过

## 6. 全量回归验证

- [x] 6.1 `ctest -R "barrier" -V` 全部 PASS（包括 23 个旧测试 + 7 个新测试 = 30/30）
- [x] 6.2 `./scripts/sanity.sh --quick` 全部 PASS（无回归）
- [x] 6.3 **MUST** 对比 baseline.txt：MUST NOT 新增 FAIL；新增 PASS = 7（新测试全部通过）
- [x] 6.4 grep 验证：`grep -rn "TODO\|FIXME" tests/unit/barrier/test_barrier_module_release.cpp tests/unit/barrier/test_warp_barrier_lifecycle.cpp tests/unit/barrier/test_participation_mask_boundaries.cpp` → 应为空

## 7. 文档同步

- [x] 7.1 在 `docs/adr/ADR-0008-barrier-semantics.md` §2026-07-03 "已知未完成 / lifecycle 单元测试" 一项追加："已通过本 change (`barrier-module-lifecycle-tests`) 补完，覆盖 `BarrierModule::release_warp_barrier` + `WarpBarrier` lifecycle + `participation_mask` 边界 7 个测试（test_barrier_module_release.cpp: 3 cases / test_warp_barrier_lifecycle.cpp: 2 cases / test_participation_mask_boundaries.cpp: 2 cases）。注：BUG-RECONVERGENCE-SIMPLEGEMM re-init 不变量仍由 test_barrier_module.cpp 覆盖。"
- [x] 7.2 **可选**：若 `tests/unit/barrier/AGENTS.md` 存在，在其中列出 3 个新测试文件
- [x] 7.3 **可选**：更新 `docs/dev-process/lessons-learned.md` 顶部"来源"行加本次 change 引用（本 change 不是新教训，是 §19 的实操落地，故非强制）

## 8. 验证与发布

- [x] 8.1 创建 commit：`git add . && git commit -m "test(barrier): add direct unit coverage for BarrierModule release + lifecycle + mask boundaries

7 unit tests across 3 files (BUG-RECONVERGENCE-SIMPLEGEMM re-init already
covered in test_barrier_module.cpp — not duplicated):

1. test_barrier_module_release.cpp (3 cases):
   - release_warp_barrier_OR_active_mask
   - release_warp_barrier_resets_is_blocked_status_is_active
   - release_warp_barrier_two_cycle_OR_preserves_first_half_lanes (BUG-POSTBARRIER-TWOHALVES)

2. test_warp_barrier_lifecycle.cpp (2 cases):
   - init_arrive_complete_reset_reinit_full_cycle
   - multiple_completion_cycles_no_state_leak

3. test_participation_mask_boundaries.cpp (2 cases):
   - full_mask_32_arrive_31_is_incomplete
   - partial_mask_16_all_arrive_completes_at_16

Refs: migrate-bar-warp-sync-to-barrier-module review I1;
      lessons-learned §1, §19"`（commit message 引用 I1 + §19）
- [x] 8.2 **可选**：合并到主分支（如果从 worktree 操作） 或 push（如果从 main 操作）
- [x] 8.3 **可选**：当所有 task 完成且 sanity.sh 全通过后，运行 `openspec archive change "barrier-module-lifecycle-tests"` 完成归档

---

## ⚠️ 紧急停止条件

实施过程中如出现以下任一情况，立即 STOP 并回滚到上一稳定状态：

1. **新单元测试触发 production 编译错误** —— 立即 revert 该文件（说明 BarrierModule API 使用方式与当前实现不兼容）
2. **新单元测试全部失败** —— 说明 BarrierModule 实际语义与假定不同，应立即停止并 review `BarrierModule::release_warp_barrier` 实现
3. **任何已有测试回归** —— `./scripts/sanity.sh --quick` 任意 FAIL → 立即 revert（违反 lessons-learned §4）
4. **揭示 production bug** （如 `participation_mask=0` 时 `release_warp_barrier` 行为异常）—— **不要在测试代码里"修补"以让测试通过**，应单独建紧急 change 修复 production，本 change revert

参考：`docs/dev-process/lessons-learned.md` §4（任何已有测试回归 → 立即 revert 该 Phase）
