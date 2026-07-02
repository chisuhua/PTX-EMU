## 0. 强制 Root Cause 分析（必做！在 Phase 1 之前完成）

- [ ] 0.1 阅读 commit `36dbb9a`（实施）与 `f033312`（revert）的 commit message + diff
- [ ] 0.2 阅读 `docs/dev-process/lessons-learned.md` 与 `docs/adr/0008-barrier-semantics.md` §2026-06-18 Postmortem
- [ ] 0.3 列出 commit `36dbb9a` 引入的具体代码变更（vs revert 后）
- [ ] 0.4 输出 `failure_root_cause.md` 文档：本 change 必须解决的具体 bug 列表
- [ ] 0.5 **如果 root cause 未明确，STOP 实施，等待更多调查**（不重蹈覆辙）

## 1. 审计与准备

- [ ] 1.0 完成 Phase 0 root cause 分析
- [ ] 1.1 全项目 grep `WarpBarrier::init` + `init_warp_barrier` + `arrive_at_warp_barrier` + `release_warp_barrier` 所有调用点：`grep -rn "WarpBarrier\|init_warp_barrier\|arrive_at_warp_barrier\|release_warp_barrier" src/ include/ tests/`
- [ ] 1.2 阅读 `force_reconvergence_at_barrier` 实现（`src/ptxsim/core/warp_context.cpp`），确认其与 barrier 状态管理的交互点
- [ ] 1.3 阅读 `barrier.cpp::BarWarpSyncHandler::processOperation` 完整路径 A（force_reconvergence）+ 路径 B（正常）
- [ ] 1.4 输出 `warp_sync_audit.md`：列出所有需要修改的代码位置 + 风险评估
- [ ] 1.5 创建 worktree：`git worktree add ../ptx-emu-warp-sync -b feat/migrate-bar-warp-sync`
- [ ] 1.6 在 worktree 中建立基线：`./scripts/sanity.sh --quick > baseline.txt`

## 2. WarpBarrier::init 增强（解决 force_reconvergence 重新进入不变性）

- [ ] 2.1 修改 `src/ptxsim/barrier/warp_barrier.cpp::WarpBarrier::init`：增加 `is_initialized_` 分支处理——若已初始化，仅更新 metadata（participation_mask、reconvergence_pc、barrier_pc、expected_count、state=Waiting），**不重置** arrived_mask_/arrived_count_；保持首次 init 路径不变
- [ ] 2.2 验证：`cmake --build build --target ptxsim` 编译通过
- [ ] 2.3 新增 `tests/unit/barrier/test_warp_barrier.cpp` 测试 `WarpBarrier::init preserves arrived_mask when re-init`：init(mask=0xFF, pc=10) → arrive(0) → init(mask=0xFFFF, pc=20) → 验证 arrived_mask 仍含 lane 0（即 0x01），participation_mask=0xFFFF，reconvergence_pc=20
- [ ] 2.4 验证：`cd build && ctest -R "WarpBarrier::init preserves" -V` PASS
- [ ] 2.5 不回归：`ctest -R "post_barrier_reconvergence_simplegemm" -V` PASS
- [ ] 2.6 不回归：`ctest -R "unit_barrier_module" -V` 全部 PASS

## 3. BarWarpSyncHandler 迁移到 BarrierModule API

### 3a. 路径 A（force_reconvergence 分歧场景）

- [ ] 3.1 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 路径 A：
  - 替换 `ptxsim::Wbar& init_wbar = warp_state.wbars[0];` 为 `WarpBarrier* init_wbar = warp_ctx->get_cta_context()->get_barrier_module().get_warp_barrier(0);`
  - 替换 `init_wbar.init(participation_mask, reconvergence_pc);` 为 `init_wbar->init(participation_mask, reconvergence_pc, current_pc);`（自动处理 is_initialized_ 分支）
  - 替换 `init_wbar.arrive(lane_id);` 为 `init_wbar->arrive(lane_id);`
  - 替换 `init_wbar.is_complete()` 为 `init_wbar->is_complete()`
  - 替换 `init_wbar.arrived_mask` 为 `init_wbar->get_arrived_mask()`
  - 替换 `init_wbar.count_arrived()` 为 `init_wbar->get_arrived_count()`
  - 替换 `init_wbar.count_participants()` 为 `init_wbar->get_expected_count()`
- [ ] 3.2 移除 `warp_state.current_wbar_id = 0;` 与 `warp_state.current_wbar_id = -1;` 直接赋值，改为 `init_wbar->is_initialized()` 检查

### 3b. 路径 B（正常 barrier，无分歧）

- [ ] 3.3 修改路径 B：
  - 同样替换 `warp_state.wbars[wbar_id]` 为 `barrier_module.get_warp_barrier(wbar_id)`
  - 同样替换所有 `wbar.method()` 为 `wbar->method()`
- [ ] 3.4 移除 `sm_ctx->bsync_manager_.bsync(...)` 与 `sm_ctx->bsync_manager_.release(...)` 调用（依赖 `cleanup-deprecated-barrier-apis` 已完成删除 `BsyncManager` 类）；若 `cleanup-deprecated-barrier-apis` 未先实施，保留调用但标记 TODO
- [ ] 3.5 验证：`cmake --build build && ctest -R "barrier" -V` 全部 PASS

### 3c. 新增测试覆盖 commit `36dbb9a` 失败案例

- [ ] 3.6 新增 `tests/integration/divergence/test_post_barrier_two_halves_barrier_module.cpp`：复现分歧 warp 两半分别到达 barrier 场景，验证 BarrierModule API 路径下 barrier 正常完成（commit `36dbb9a` 失败案例的复现 + 修复验证）
- [ ] 3.7 验证：`cd build && ctest -R "integration_post_barrier_two_halves" -V` PASS

## 4. 全量回归

- [ ] 4.1 `./scripts/sanity.sh --quick` 全部 PASS；与 baseline.txt 对比，MUST NOT 新增 FAIL
- [ ] 4.2 `./scripts/sanity.sh` 完整回归 PASS
- [ ] 4.3 `./tests/ptx/test_all_ptx.sh` 全部 PTX 语法测试通过
- [ ] 4.4 `ctest -R "e2e_barrier_warp_sync|e2e_test3_cfg_full" -V` 全部 PASS
- [ ] 4.5 `ctest -R "unit_barrier|integration_barrier" -V` 全部 PASS（**关键**：所有 barrier 测试不能有回归）

## 5. 文档同步

- [ ] 5.1 更新 `docs/adr/0008-barrier-semantics.md`：追加 §"2026-06-19 追加：BarWarpSyncHandler 迁移 + WarpBarrier::init 不变性"
- [ ] 5.2 更新 `docs/research/barrier-semantics/04-ptx-emu-current-implementation.md`：描述 `BarrierModule` 统一管理 CTA + Warp barrier
- [ ] 5.3 更新 `src/ptxsim/instructions/AGENTS.md`：将"barrier.cpp (指令分发入口) → barrier/barrier_module.cpp (实际状态管理)"标注 BarWarpSyncHandler 也走 BarrierModule API

## 6. 验证与发布

- [ ] 6.1 在 worktree 中创建最终 commit：`git add . && git commit -m "feat(barrier): migrate BarWarpSyncHandler to BarrierModule API + fix WarpBarrier::init re-init semantics"`；commit message 引用 issue `BUG-POSTBARRIER-TWOHALVES` + `BUG-RECONVERGENCE-SIMPLEGEMM`
- [ ] 6.2 合并到主分支：`git checkout main && git merge --no-ff feat/migrate-bar-warp-sync -m "Merge branch..."`；清理 worktree
- [ ] 6.3 （可选）创建 PR：如不直接 merge 则 `gh pr create --title "feat(barrier): migrate BarWarpSyncHandler to BarrierModule API"`

## ⚠️ 紧急停止条件

实施过程中如出现以下任一情况，立即 STOP 并回滚到上一稳定 Phase：

1. `commit 36dbb9a` 失败模式重现（分歧 warp 两半在 post-barrier PC 卡住）
2. `post_barrier_reconvergence_simplegemm` 测试回归
3. `unit_post_barrier_two_halves` 测试回归
4. `e2e_barrier_warp_sync` 或 `e2e_test3_cfg_full` e2e 测试回归
5. `Phase 0` root cause 分析未明确

参考：`docs/dev-process/lessons-learned.md` §4（任何已有测试回归 → 立即 revert 该 Phase）