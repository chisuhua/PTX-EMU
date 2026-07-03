## 1. 测试设计与基线

- [ ] 1.1 **MUST** 阅读现有 `tests/unit/barrier/test_barrier_module.cpp::WarpBarrier::init preserves arrived_mask` —— 摸清现有单测对 WarpBarrier / BarrierModule 的构造模式（mock WarpContext 程度、setup 方式）
- [ ] 1.2 阅读 `src/ptxsim/barrier/barrier_module.cpp::release_warp_barrier`（L85-138）—— 列出 5 项状态翻译断言点
- [ ] 1.3 阅读 `src/ptxsim/barrier/warp_barrier.cpp` —— 列出完整 public API 用于 lifecycle 测试断言
- [ ] 1.4 **MUST** 建立基线：`cd build && ctest -R "barrier" -V > /tmp/baseline_barrier.txt`（应 23/23 PASS）

## 2. WarpBarrier lifecycle 单元测试

- [ ] 2.1 创建 `tests/unit/barrier/test_warp_barrier_lifecycle.cpp` 头文件与 TEST_CASE 框架
- [ ] 2.2 实现 `init_arrive_complete_reset_reinit_full_cycle`：构造 WarpBarrier → init(mask=0xFFFFFFFF) → arrive(0..31) → is_complete() → reset() → re-init → arrive(0..31) → is_complete()
- [ ] 2.3 实现 `re_init_preserves_arrived_mask_for_force_reconvergence`：模拟 `BUG-RECONVERGENCE-SIMPLEGEMM` —— init → arrive(0..15) → 再次 init（force_reconvergence 重新进入） → 验证 arrived_mask 仍含 0..15
- [ ] 2.4 实现 `multiple_completion_cycles_no_state_leak`：连续 3 次完整 cycle，每次都用 is_complete() 后立即 reset()
- [ ] 2.5 编译验证：`cmake --build build --target unit_warp_barrier_lifecycle` 通过
- [ ] 2.6 运行验证：`ctest -R "unit_warp_barrier_lifecycle" -V` 全部 PASS

## 3. BarrierModule::release_warp_barrier 状态翻译测试

- [ ] 3.1 创建 `tests/unit/barrier/test_barrier_module_release.cpp` 头文件与 TEST_CASE 框架
- [ ] 3.2 实现 `release_warp_barrier_OR_active_mask`：mock WarpContext 含 active_mask=0xFFFF0000，调用 release(mask=0x0000FFFF) → 验证 active_mask 变为 0xFFFFFFFF（OR 语义）
- [ ] 3.3 实现 `release_warp_barrier_resets_is_blocked_status_is_active`：mock warp_state.threads[i].is_blocked=true 等 → release → 验证 is_blocked=false + status=Active + is_active=true
- [ ] 3.4 实现 `release_warp_barrier_idempotency_within_cycle`：模拟 BUG-POSTBARRIER-TWOHALVES —— 释放一半，状态字段正确更新后，再释放另一半（不应该丢失）
- [ ] 3.5 编译验证：`cmake --build build --target unit_barrier_module_release` 通过
- [ ] 3.6 运行验证：`ctest -R "unit_barrier_module_release" -V` 全部 PASS

## 4. participation_mask 边界条件测试

- [ ] 4.1 创建 `tests/unit/barrier/test_participation_mask_boundaries.cpp` 头文件与 TEST_CASE 框架
- [ ] 4.2 实现 `full_mask_32_arrive_31_is_incomplete`：mask=0xFFFFFFFF → arrive(0..30) → 验证 is_complete() == false
- [ ] 4.3 实现 `partial_mask_16_all_arrive_completes_at_16`：mask=0x0000FFFF → arrive(0..15) → 验证 is_complete() == true 且 arrived_count=16
- [ ] 4.4 编译验证：`cmake --build build --target unit_participation_mask_boundaries` 通过
- [ ] 4.5 运行验证：`ctest -R "unit_participation_mask_boundaries" -V` 全部 PASS

## 5. CMake 注册

- [ ] 5.1 在 `tests/unit/barrier/CMakeLists.txt` 添加 3 个 `add_catch_test` 目标（按 commit `ab55e06` 后的命名约定：`unit_*` 前缀）
- [ ] 5.2 设置 `set_tests_properties` 标签为 `[unit;barrier]`
- [ ] 5.3 完整构建：`cmake --build build --target unit_barrier_module_release unit_warp_barrier_lifecycle unit_participation_mask_boundaries` 通过

## 6. 全量回归验证

- [ ] 6.1 `ctest -R "barrier" -V` 全部 PASS（包括 23 个旧测试 + 8 个新测试 = 31/31）
- [ ] 6.2 `./scripts/sanity.sh --quick` 全部 PASS（无回归）
- [ ] 6.3 **MUST** 对比 baseline.txt：MUST NOT 新增 FAIL；新增 PASS = 8（新测试全部通过）
- [ ] 6.4 grep 验证：`grep -rn "TODO\|FIXME" tests/unit/barrier/test_barrier_module_release.cpp tests/unit/barrier/test_warp_barrier_lifecycle.cpp tests/unit/barrier/test_participation_mask_boundaries.cpp` → 应为空

## 7. 文档同步

- [ ] 7.1 在 `docs/adr/0008-barrier-semantics.md` §2026-07-03 "已知未完成 / lifecycle 单元测试" 一项追加："已通过本 change (`barrier-module-lifecycle-tests`) 补完，覆盖 `BarrierModule::release_warp_barrier` + `WarpBarrier` lifecycle + `participation_mask` 边界 8 个测试"
- [ ] 7.2 **可选**：若 `tests/unit/barrier/AGENTS.md` 存在，在其中列出 3 个新测试文件
- [ ] 7.3 **可选**：更新 `docs/dev-process/lessons-learned.md` 顶部"来源"行加本次 change 引用（本 change 不是新教训，是 §19 的实操落地，故非强制）

## 8. 验证与发布

- [ ] 8.1 创建 commit：`git add . && git commit -m "test(barrier): add direct unit coverage for BarrierModule release + lifecycle + mask boundaries"`（commit message 引用 I1 + §19）
- [ ] 8.2 **可选**：合并到主分支（如果从 worktree 操作） 或 push（如果从 main 操作）
- [ ] 8.3 **可选**：当所有 task 完成且 sanity.sh 全通过后，运行 `openspec archive change "barrier-module-lifecycle-tests"` 完成归档

---

## ⚠️ 紧急停止条件

实施过程中如出现以下任一情况，立即 STOP 并回滚到上一稳定状态：

1. **新单元测试触发 production 编译错误** —— 立即 revert 该文件（说明 BarrierModule API 使用方式与当前实现不兼容）
2. **新单元测试全部失败** —— 说明 BarrierModule 实际语义与假定不同，应立即停止并 review `BarrierModule::release_warp_barrier` 实现
3. **任何已有测试回归** —— `./scripts/sanity.sh --quick` 任意 FAIL → 立即 revert（违反 lessons-learned §4）
4. **揭示 production bug** （如 `participation_mask=0` 时 `release_warp_barrier` 行为异常）—— **不要在测试代码里"修补"以让测试通过**，应单独建紧急 change 修复 production，本 change revert

参考：`docs/dev-process/lessons-learned.md` §4（任何已有测试回归 → 立即 revert 该 Phase）
