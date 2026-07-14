## 1. 审计与准备

- [ ] 1.1 全项目 grep 旧 API 使用点:`grep -rn "Wbar\b\|\bwbar\.\|wbars\[\|bsync_manager_\|synchronize_barrier\|barrier_waiting_threads\|barrier_mutex_\|barrier_thread_counts" src/ include/ tests/ > /tmp/cleanup_audit.txt`;统计每个 API 的引用次数
- [ ] 1.2 阅读 `src/ptxsim/core/bsync_state.cpp` 全部 9 个方法(`bssy/bsync/check_release/release/get_state/is_waiting/get_waiting_mask/cleanup/reset`);输出 `cleanup_audit.md` 列出每个 API 的引用点(已验证:无生产消费者)
- [ ] 1.3 阅读 `src/ptxsim/core/sm_context.cpp:204-242` 周期 barrier 检查代码块;确认 `barrier_mutex_` 和 `barrier_waiting_threads` 唯一消费者(已 grep 验证:仅 sm_context.cpp:204 + 608)
- [ ] 1.4 验证 `warp_context.cpp:283-296` BAR_SYNC fallback 的 `bar_id` 语义(是 thread 的 `bar_id` 字段,不是 `current_wbar_id`);确认替换为 `BarrierModule::arrive_at_cta_barrier(bar_id, thread)` 时 `bar_id` 直接可用。**BLOCKING**:若 `thread->bar_id` 语义不是 CTA barrier ID(0-15),则需 Open Question 1 解决方案(参考 `barrier.cpp:345-348` BarHandler 的 `instr.barId` 取值方式)
- [ ] 1.5 复用既有 `.worktrees/fix-pre-p0-baseline` 作为 baseline worktree(已存在,无需重建;节省 15-20 分钟基线 build)
- [ ] 1.6 在 baseline worktree 中建立基线:`./scripts/sanity.sh > baseline.txt`,保存用于对比
- [ ] 1.7 创建实施 worktree:`git worktree add ../ptx-emu-cleanup -b refactor/cleanup-deprecated-barrier-apis`

## 2. 删除 BsyncManager + 同步调用点(Commit 1,独立可 revert)

- [ ] 2.1 `rm include/ptxsim/bsync_state.h src/ptxsim/core/bsync_state.cpp`
- [ ] 2.2 从 `src/CMakeLists.txt` 移除 `ptxsim/core/bsync_state.cpp` 条目
- [ ] 2.3 从 `include/ptxsim/sm_context.h` 移除 `BsyncManager bsync_manager_` 字段(line 195)、`#include "ptxsim/bsync_state.h"` (line 6)
- [ ] 2.4 从 `src/ptxsim/core/warp_scheduler.cpp` 移除 `#include "ptxsim/bsync_state.h"` (line 2);验证无 `BsyncManager` / `bsync_manager` 使用
- [ ] 2.5 从 `src/ptxsim/instructions/barrier.cpp` 删除 lines 189, 240, 249 的 `sm_ctx->bsync_manager_.bsync/release` 调用;**同时清理死代码**:删除 lines 188 + 238 的 `SMContext* sm_ctx = warp_ctx->get_sm_context();` 和 `if (sm_ctx) { }` 空块(无其他 `bsync_manager_` 调用后,sm_ctx 变量无引用)
- [ ] 2.6 删除 `tests/unit/sync/test_bsync_state.cpp`(BsyncManager 类删除后该测试必须删除);**同时从 `tests/unit/CMakeLists.txt:291-294` 删除**:
      ```
      add_catch_test(unit_bsync_state
          sync/test_bsync_state.cpp
      )
      set_tests_properties(unit_bsync_state PROPERTIES LABELS "unit;bsync")
      ```
- [ ] 2.7 处理 `tests/unit/sync/test_barrier_active_mask_preserved.cpp`(4 处 `sm.synchronize_barrier(0, t)` 调用 lines 45,84,123,132)和 `tests/unit/barrier/test_barrier_scenarios.cpp`(2 处 `sm.synchronize_barrier(0, t)` 调用 lines 335,344);**决策**:删除这 2 个测试文件(synchronize_barrier 删除后失去测试目标,2 个测试仅验证 synchronize_barrier 行为);如需保留 barrier active_mask 测试语义,迁移到 `cta_ctx->get_barrier_module()->arrive_at_cta_barrier(0, thread)` API(参考 barrier.cpp:345-392 BarHandler 实现)
- [ ] 2.8 验证:`cmake --build build --target ptxsim` 编译通过(确认所有测试文件无 `synchronize_barrier` / `BsyncManager` / `bsync_manager_` 引用)
- [ ] 2.9 `ctest -L "barrier;warp"` 全部 PASS
- [ ] 2.10 `./scripts/sanity.sh --quick` 全部 PASS
- [ ] 2.11 commit:`git commit -m "refactor(barrier): remove BsyncManager dead code"`(commit message 按 lessons-learned §14 Checklist D 格式列出独立 fix 编号):
      - Fix #1: 删除 `include/ptxsim/bsync_state.h` + `src/ptxsim/core/bsync_state.cpp`(2 文件)
      - Fix #2: 删除 `tests/unit/sync/test_bsync_state.cpp` + `tests/unit/CMakeLists.txt` 对应条目(测试文件 + 构建配置)
      - Fix #3: 删除 `tests/unit/sync/test_barrier_active_mask_preserved.cpp` + `tests/unit/barrier/test_barrier_scenarios.cpp`(synchronize_barrier 单元测试)
      - Fix #4: 删除 `src/ptxsim/instructions/barrier.cpp` 3 处 `bsync_manager_` 调用 + 死代码清理

## 3. 删除 SM 级 barrier 状态 + 替换 warp_context.cpp BAR_SYNC fallback(Commit 2,独立可 revert)

- [ ] 3.1 从 `include/ptxsim/sm_context.h` 移除 `barrier_waiting_threads` (line 189) / `barrier_thread_counts` (line 190) / `barrier_mutex_` (line 192) 字段
- [ ] 3.2 从 `include/ptxsim/sm_context.h` 移除 `synchronize_barrier()` 声明(line 114)
- [ ] 3.3 从 `src/ptxsim/core/sm_context.cpp` 删除 lines 204-242 周期 barrier 检查代码块(含 `barrier_mutex_` lock at line 204);**保留** lines 244-260 的 `decrement_blocked_cycles` + `update_active_mask`(warp 调度维护,非 barrier 代码)
- [ ] 3.4 从 `src/ptxsim/core/sm_context.cpp` 删除 lines 605-706 `synchronize_barrier()` 方法体
- [ ] 3.5 从 `src/ptxsim/instructions/barrier.cpp` 删除 line 23 `#include "ptxsim/sm_context.h"`(若仅用于 `synchronize_barrier`);删除 line 385 过期注释(行号 :605 而非 :703)
- [ ] 3.6 **关键**:`src/ptxsim/core/warp_context.cpp:283-296` BAR_SYNC fallback 替换:
      - 删除 `sm_context_->synchronize_barrier(thread->bar_id, thread);` (line 292)
      - 改为 `cta_context_->get_barrier_module()->arrive_at_cta_barrier(thread->bar_id, thread);`
      - **添加 null 检查**:`if (cta_context_ != nullptr) { ... }`(参考 `barrier.cpp:357-362` BarHandler 模式)
      - `cta_context_` 通过 `warp_ctx->get_cta_context()` 获取(commit b04cdb2 引入)
      - **必须添加注释**:"替换 synchronize_barrier fallback (lessons-learned §1 BAR_SYNC 翻译链)"
- [ ] 3.7 从 `src/ptxsim/core/thread_context.cpp:774` 更新注释:`synchronize_barrier` 引用改为 `BarrierModule::arrive_at_cta_barrier`
- [ ] 3.8 验证:`cmake --build build --target ptxsim` 编译通过
- [ ] 3.9 `ctest -L "barrier;warp"` 全部 PASS
- [ ] 3.10 `ctest -R "unit_post_barrier_two_halves" -V` 全部 PASS(BUG-POSTBARRIER-TWOHALVES 修复未破坏)
- [ ] 3.11 `ctest -R "unit_barrier_divergence_reconvergence_simplegemm" -V` 全部 PASS(BUG-RECONVERGENCE-SIMPLEGEMM 未回归)
- [ ] 3.12 `tests/integration/divergence/test_post_barrier_divergence.cpp` 仍 PASS(已知 BUG 测试,作为回归保护)
- [ ] 3.13 `./scripts/sanity.sh --quick` 全部 PASS;与 baseline.txt 对比 MUST NOT 新增 FAIL
- [ ] 3.14 commit:`git commit -m "refactor(barrier): remove SM-level barrier state and migrate warp_context fallback"`(commit message 按 lessons-learned §14 Checklist D 格式列出独立 fix 编号):
      - Fix #1: 删除 `include/ptxsim/sm_context.h` 4 个 SM 级 barrier 字段(`BsyncManager` / `barrier_waiting_threads` / `barrier_thread_counts` / `barrier_mutex_`)
      - Fix #2: 删除 `sm_context.cpp:204-242` 周期 barrier 检查代码块(保留 lines 244-260 warp 调度维护)
      - Fix #3: 删除 `sm_context.cpp:605-706` `synchronize_barrier()` 方法体
      - Fix #4: 替换 `warp_context.cpp:292` BAR_SYNC fallback 为 `BarrierModule::arrive_at_cta_barrier`(含 null 检查 + lessons-learned §1 注释)
      - Fix #5: 删除 `barrier.cpp` line 23 `#include "ptxsim/sm_context.h"` + line 385 过期注释

## 4. 文档同步(Commit 3,独立可 revert)

- [ ] 4.1 `docs/adr/0008-barrier-semantics.md` 追加 "2026-06-20 Phase 6 partial cleanup" 段落,记录:
      - `BsyncManager` 与 SM 级 barrier 状态删除
      - `Wbar` struct 保留到 Phase 5
      - `warp_context.cpp:283-296` BAR_SYNC fallback 替换为 `BarrierModule::arrive_at_cta_barrier`
      - 引用 commit `f033312` lessons-learned §1 BAR_SYNC 翻译链
- [ ] 4.2 `src/ptxsim/core/AGENTS.md`:
      - 删除 "Barrier sync \| `sm_context.cpp` \| `synchronize_barrier()`" 行(line 22)
      - 删除 KNOWN ISSUES 中 `synchronize_barrier() may not update active_mask` 注释(line 85)
      - 改为指向 `tests/integration/divergence/test_post_barrier_divergence.cpp` 作为 BUG 文档(known issue 仍存在)
- [ ] 4.3 `src/ptxsim/AGENTS.md`:
      - line 42 注释更新:`BarWarpSyncHandler` still uses `Wbar` (Phase 5 deferred)
      - 保留 line 48 "DO NOT add new uses of Wbar struct" 警告(Wbar 仍存在)
- [ ] 4.4 `tests/AGENTS.md`: 更新 line 15 "barrier/Wbar 数据结构" 描述(保留);`bsync` 描述移到 `archive/` 或删除
- [ ] 4.5 验证:OpenSpec 5 个 artifacts(spec.md + design.md + tasks.md + proposal.md + README.md)内部一致
- [ ] 4.6 commit:`git commit -m "docs(barrier): update ADR/AGENTS for Phase 6 partial cleanup"`(commit message 列出 4 个独立文档 fix)

## 5. 最终验证

- [ ] 5.1 验证 grep 零匹配:`grep -rn "bsync_manager_\|bsync_state\.h\|synchronize_barrier\|barrier_waiting_threads\|barrier_mutex_" src/ include/ tests/` 输出为空(MUST 零匹配)
- [ ] 5.2 验证 Wbar 仍存在:`grep -rn "Wbar\b\|\bwbar\.\|wbars\[" src/ include/ tests/` 应有 ≥19 个匹配(测试文件保留)
- [ ] 5.3 `cmake --build build` 全量编译通过;MUST NOT 引入新编译错误
- [ ] 5.4 `./scripts/sanity.sh --quick` 全部通过
- [ ] 5.5 `./scripts/sanity.sh` 完整回归通过;e2e 测试 `e2e_barrier_warp_sync` / `e2e_test3_cfg_full` 全部 PASS
- [ ] 5.6 `./tests/ptx/test_all_ptx.sh` 全部 PTX 语法测试通过
- [ ] 5.7 对比 baseline.txt,MUST NOT 新增 FAIL

## 6. 合并与发布

- [ ] 6.1 验证 3 个 commit 各自独立可 revert:`git revert HEAD~2..HEAD` 应能干净 revert 而不破坏编译
- [ ] 6.2 合并到主分支:`git checkout main && git merge --no-ff refactor/cleanup-deprecated-barrier-apis -m "Merge branch..."`
- [ ] 6.3 清理 worktree:`git worktree remove ../ptx-emu-cleanup`
- [ ] 6.4 (可选)创建 PR:如不直接 merge 则 `gh pr create`,PR 描述引用 issue `BUG-POSTBARRIER-TWOHALVES` + `BUG-RECONVERGENCE-SIMPLEGEMM`(commit message 已引用)
- [ ] 6.5 归档 OpenSpec change:`openspec archive cleanup-deprecated-barrier-apis`(归档时强制询问 postmortem 写入 ADR-0008)
