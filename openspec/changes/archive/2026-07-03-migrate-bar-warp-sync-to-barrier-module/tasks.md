## 0. 强制 Root Cause 分析（必做！在 Phase 1 之前完成）

- [ ] 0.1 阅读 commit `36dbb9a`（实施）与 `f033312`（revert）的 commit message + diff
- [ ] 0.2 阅读 `docs/dev-process/lessons-learned.md` 与 `docs/adr/ADR-0008-barrier-semantics.md` §2026-06-18 Postmortem
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

## 2. WarpBarrier::init 增强验证

> **⚠️ 本 Phase 已在 main 分支预先落地**（通过 `integrate-barrier-module-cta-warp-fix` plan，commit `b04cdb2`）。Phase 2 只需**验证**现有实现正确，无需重复实施。

- [x] 2.1 ~~修改 `src/ptxsim/barrier/warp_barrier.cpp::WarpBarrier::init`~~ → **已在 main 实现**：`warp_barrier.cpp:18-31` 含 `if (is_initialized_)` 分支，保留 `arrived_mask_/arrived_count_`，符合 Decision 1 语义
- [x] 2.2 ~~验证编译~~ → main 编译通过
- [x] 2.3 ~~新增 `test_warp_barrier.cpp` re-init 测试~~ → **已在 `tests/unit/barrier/test_barrier_module.cpp` 实现**：`WarpBarrier::init preserves arrived_mask` 测试用例覆盖
- [ ] 2.4 **验证**：`cd build && ctest -R "WarpBarrier::init preserves" -V` → **确认 PASS**
- [ ] 2.5 **不回归验证**：`ctest -R "post_barrier_reconvergence_simplegemm" -V` → **确认 PASS**
- [ ] 2.6 **不回归验证**：`ctest -R "unit_barrier_module" -V` → **确认全部 PASS**

## 3. BarWarpSyncHandler 迁移到 BarrierModule API

### 3a. 路径 A（force_reconvergence 分歧场景）

- [ ] 3.1 修改 `src/ptxsim/instructions/barrier.cpp::BarWarpSyncHandler::processOperation` 路径 A：
  - 替换 `ptxsim::Wbar& init_wbar = warp_state.wbars[0];` 为 `WarpBarrier* init_wbar = warp_ctx->get_cta_context()->get_barrier_module().get_warp_barrier(0);`
  - 替换 `init_wbar.init(participation_mask, reconvergence_pc);` 为 `init_wbar->init(participation_mask, reconvergence_pc, current_pc);`（自动处理 `is_initialized_` 分支）
  - 替换 `init_wbar.arrive(lane_id);` 为 `init_wbar->arrive(lane_id);`
  - 替换 `init_wbar.is_complete()` 为 `init_wbar->is_complete()`
  - 替换 `init_wbar.arrived_mask` 为 `init_wbar->get_arrived_mask()`
  - 替换 `init_wbar.count_arrived()` 为 `init_wbar->get_arrived_count()`
  - 替换 `init_wbar.count_participants()` 为 `init_wbar->get_expected_count()`
  - 移除 `sm_ctx->bsync_manager_.release(0)` 调用（`BsyncManager` 已于 `cleanup-deprecated-barrier-apis` 归档中删除）
  - 替换 `warp_ctx->set_exec_mask(init_wbar.arrived_mask)` + OR 逻辑为 `warp_ctx->get_cta_context()->get_barrier_module().release_warp_barrier(0, warp_ctx)`（已包含 BUG-POSTBARRIER-TWOHALVES OR 逻辑 + `is_blocked=false` / `status=Active` / `is_active=true` 状态翻译）
  - ⚠️ 在 `release_warp_barrier` 调用后，必须调用 `context->set_pc_overridden(true)` — `release_warp_barrier` 已推进 PC，不通知 ThreadContext 会导致 `commit_pc()` 二次推进（跳过 reconvergence point）
- [ ] 3.2 移除 `warp_state.current_wbar_id = 0;` 与 `warp_state.current_wbar_id = -1;` 直接赋值，替换守卫条件：
  - 替换 `warp_state.current_wbar_id < 0`（barrier.cpp:145,157 分裂检测守卫）为 `!init_wbar->is_initialized()`
  - 替换 `warp_state.current_wbar_id >= 0`（barrier.cpp:184 完成守卫）为 `init_wbar->is_initialized() && init_wbar->is_complete()`

### 3b. 路径 B（正常 barrier，无分歧）

- [ ] 3.3 修改路径 B：
  - 同样替换 `warp_state.wbars[wbar_id]` 为 `barrier_module.get_warp_barrier(wbar_id)`
  - 同样替换所有 `wbar.method()` 为 `wbar->method()`
  - 完成分支：替换 `set_exec_mask + advance_thread_pc + is_blocked + status + set_active_mask(OR)` 为 `barrier_module.release_warp_barrier(wbar_id, warp_ctx)` + 跟随 `context->set_pc_overridden(true)`
  - 替换守卫 `warp_state.current_wbar_id < 0 && wbar.is_initialized`（barrier.cpp:217）为 `!init_wbar->is_initialized()`
  - 替换守卫 `wbar.is_complete() && warp_state.current_wbar_id >= 0`（barrier.cpp:236）为 `init_wbar->is_complete() && init_wbar->is_initialized()`
- [ ] 3.4 移除 `sm_ctx->bsync_manager_.bsync(...)` 与 `sm_ctx->bsync_manager_.release(...)` 调用 — **注意**：`BsyncManager` / `synchronize_barrier` 已于 `cleanup-deprecated-barrier-apis`（2026-06-20 归档，commit `ded4f96` → `archive/2026-06-20-cleanup-deprecated-barrier-apis/`）中删除。执行本 task 时若编译器报 `bsync_manager_` 未定义，直接删除调用行即可（无需保留 TODO）
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

- [x] 5.1 更新 `docs/adr/ADR-0008-barrier-semantics.md`：追加 §"BarWarpSyncHandler 迁移 + Wbar 删除"（2026-07-03 完成，see §2026-07-03）
- [x] 5.2 更新 `src/ptxsim/instructions/AGENTS.md`：将"barrier.cpp (指令分发入口) → barrier/barrier_module.cpp (实际状态管理)"标注 BarWarpSyncHandler 也走 BarrierModule API（2026-07-03 完成 by docs(barrier): sync AGENTS.md）
- [x] 5.3 更新 `src/ptxsim/core/AGENTS.md`：移除 `Wbar` / `warp_state.wbars[]` 相关 ANTI-PATTERNS 条目；澄清 `BUG-POSTBARRIER-TWOHALVES` fix 现在由 `BarrierModule::release_warp_barrier` 封装（2026-07-03 完成 by docs(barrier): sync AGENTS.md）

## 6. 验证与发布

- [ ] 6.1 在 worktree 中创建最终 commit：`git add . && git commit -m "feat(barrier): migrate BarWarpSyncHandler to BarrierModule API + delete legacy Wbar"`；commit message 引用 issue `BUG-POSTBARRIER-TWOHALVES` + `BUG-RECONVERGENCE-SIMPLEGEMM` + `P0-A5`
- [ ] 6.2 合并到主分支：`git checkout main && git merge --no-ff feat/migrate-bar-warp-sync -m "Merge branch..."`；清理 worktree
- [ ] 6.3 （可选）创建 PR：如不直接 merge 则 `gh pr create --title "feat(barrier): migrate BarWarpSyncHandler to BarrierModule API + delete legacy Wbar"`

## 7. Wbar 最终删除（cleanup chain 收尾）

> **来源**：P0-A5（来自 `docs/audits/debt-audit-2026-07-02.md` §1.1）。前置 change `cleanup-deprecated-barrier-apis` 删除了 `BsyncManager` / `synchronize_barrier`，但 **未** 删除 `Wbar` struct 本身。本 Phase 完成 barrier cleanup chain 的最后一步。
>
> **前置条件**：Phase 3 完成（`BarWarpSyncHandler` 已迁移到 `BarrierModule` API，`barrier.cpp` 中零处 `warp_state.wbars[]` / `current_wbar_id` 引用）

### 7a. 删除旧代码

- [ ] 7.1 **删除 `include/ptxsim/wbar.h`**（全部 121 行）：`Wbar` struct 的所有生产调用点已迁移至 `BarrierModule` / `WarpBarrier` API，无保留价值
- [ ] 7.2 **删除 `include/ptxsim/warp_state.h` 中的 `wbars[]` 字段**（L23-26）：`std::array<Wbar, 4> wbars;` + `[[deprecated]]` 注释
- [ ] 7.3 **删除 `include/ptxsim/warp_state.h` 中的 `current_wbar_id` 字段**（L27-29）
- [ ] 7.4 **修改 `include/ptxsim/warp_state.h` 的 `reset()` 方法**：移除 `for (auto &wbar : wbars) { wbar.reset(); }` 循环和 `current_wbar_id = -1;` 赋值（L38-41）
- [ ] 7.5 **修改 `include/ptxsim/warp_state.h` 的 `#include`**：移除 `#include "ptxsim/wbar.h"`（L5）
- [ ] 7.6 **删除 `include/ptxsim/warp_context.h` 的 `get_wbar()` compat shim**（L222-227）：`[[deprecated]] ptxsim::Wbar &get_wbar(int wbar_id);` 声明
- [ ] 7.7 **删除 `src/ptxsim/core/warp_context.cpp` 的 `get_wbar()` 实现**（L540-556）：整个 compat shim 函数体（从 `// Legacy Wbar mirror` 注释到 `return warp_state.wbars[idx];`）
- [ ] 7.8 编译验证：`cmake --build build --target ptxsim` 通过，确认无 `wbar.h` / `wbars` / `current_wbar_id` / `get_wbar` 残留引用

### 7b. 更新文档

- [x] 7.9 **更新 `src/ptxsim/core/AGENTS.md`**：移除 ANTI-PATTERNS 中 "DO NOT add new uses of `Wbar` struct" 条目（`Wbar` 已不存在），移除 "DO NOT call methods from WarpContext" 中 `warp_state.wbars[]` 相关说明（**2026-07-03 完成**：`BUG-POSTBARRIER-TWOHALVES` fix 描述更新为 BarrierModule 封装 OR 语义）
- [x] 7.10 **更新 `src/ptxsim/AGENTS.md`** L42：删除 "`BarWarpSyncHandler` still uses `warp_state.wbars[]` (Phase 5 deferred)" 描述，改为 "`BarWarpSyncHandler` routes through `BarrierModule` API (migrated in `migrate-bar-warp-sync-to-barrier-module`)"（**2026-07-03 完成**：line 42, 48 都已更新）

### 7c. 验证与回归

- [ ] 7.11 `ctest -R "unit_barrier|integration_barrier|e2e_barrier" -V` 全部 PASS（**关键**：Wbar 删除后所有 barrier 测试不能有回归）
- [ ] 7.12 `./scripts/sanity.sh --quick` 全部 PASS
- [ ] 7.13 全项目 grep 确认零残留：`grep -rn "wbar\.h\|warp_state\.wbars\|current_wbar_id\|get_wbar(" include/ src/ tests/` → **预期输出为空**

### 7d. Commit

- [ ] 7.14 创建独立 commit：`git add . && git commit -m "chore(barrier): delete legacy Wbar struct, warp_state.wbars[], current_wbar_id, and get_wbar() compat shim

Closes P0-A5 from debt-audit-2026-07-02. All production handlers now
use BarrierModule API. WarpState reduced by 2 deprecated fields.

BREAKING: wbar.h removed. Tests must use BarrierModule APIs."`

---

## ⚠️ 紧急停止条件

实施过程中如出现以下任一情况，立即 STOP 并回滚到上一稳定 Phase：

1. `commit 36dbb9a` 失败模式重现（分歧 warp 两半在 post-barrier PC 卡住）
2. `post_barrier_reconvergence_simplegemm` 测试回归
3. `unit_post_barrier_two_halves` 测试回归
4. `e2e_barrier_warp_sync` 或 `e2e_test3_cfg_full` e2e 测试回归
5. `Phase 0` root cause 分析未明确
6. Phase 7 执行后 `wbar.h` 删除导致编译失败（说明仍有未迁移的调用方）

参考：`docs/dev-process/lessons-learned.md` §4（任何已有测试回归 → 立即 revert 该 Phase）