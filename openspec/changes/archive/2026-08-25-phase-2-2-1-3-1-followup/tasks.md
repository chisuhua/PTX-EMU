# phase-2-2-1-3-1-followup — Tasks

> **REVISION HISTORY** (per Metis session `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`):
> - Initial tasks assumed `WarpStatus` 4 fields (std::array + active_mask + blocked_cycles_remaining + finished); Metis verified `WarpStatus` is 5 fields at `include/ptxemu/device_api.h:69-75`
> - Initial tasks assumed `thread->state` direct field access; **Corrected**: use `thread->get_state()` returning `EXE_STATE` (per `include/ptxsim/thread_context.h:205`)
> - Initial tasks assumed WarpContext internal `active_mask_` / `lanes_` / `blocked_cycles_remaining` fields; **Corrected**: use `warp->get_warp_state().threads[]` access path
> - 4 artifacts (proposal.md, design.md, specs/*, tasks.md) revised in lockstep

## 1. Pre-Implementation Setup (per ptx-lessons-learned §4 + §7)

- [ ] 1.1 MUST: 验证基线 worktree `.worktrees/phase-2-2-1-3-1-baseline` 编译通过 (per §4)
- [ ] 1.2 MUST: 跑基线 ctest 249/249 PASS (per device-api-delegation archive, post-2.2/2.3 state)
- [ ] 1.3 MUST: Metis pre-impl review 审计 4 OpenSpec artifacts (per §7 + §H in ptx-lessons-learned)
- [ ] 1.4 应用 Metis MUST-RESOLVE 列表 (3 个全部已修订: WarpStatus 5 fields + thread->get_state() + WarpState::threads[]), 重审直至 ⚠️→GO 或 ✅
- [ ] 1.5 NOTE: 4 artifacts 范围数字一致性按 Checklist I 校验
- [ ] 1.6 MUST: `git add openspec/changes/phase-2-2-1-3-1-followup/` + commit (per Checklist E, 防止 audit chain 断裂)
- [ ] 1.7 创建 `feat/phase-2-2-1-3-1-followup` 分支从 origin main

## 2. Phase 1 — `warp_exe_once` + `get_thread_state` 实施 (Commit 1 of 2)

- [ ] 2.1 实施 `warp_exe_once` (per design Decision 1) — 委托到 `g_gpu_context->get_sm(sm_id)->get_warp(warp_id)->execute_warp_instruction()`:
  ```cpp
  int warp_exe_once(uint32_t sm_id, uint32_t warp_id) override {
      if (!g_gpu_context) return -1;
      auto* sm = g_gpu_context->get_sm(sm_id);
      if (!sm) return -1;
      auto* warp = sm->get_warp(warp_id);
      if (!warp) return -1;
      warp->execute_warp_instruction();  // instance method, NOT global exe_once()
      return 0;
  }
  ```
  - **注释**: 此方法为 state-mutating hot path, 引用 `ptx-instruction-pipeline` skill + BUG-RETHANG guard (overwrite semantics in BarrierModule::release_warp_barrier)
- [ ] 2.2 实施 `get_thread_state` (per design Decision 5) — 委托到 `ThreadContext::get_state()` + map EXE_STATE → ptxemu::ThreadState:
  ```cpp
  ThreadState get_thread_state(uint32_t sm_id, uint32_t warp_id,
                               uint32_t lane_id) override {
      if (!g_gpu_context) return ThreadState::kIdle;
      auto* sm = g_gpu_context->get_sm(sm_id);
      if (!sm) return ThreadState::kIdle;
      auto* warp = sm->get_warp(warp_id);
      if (!warp) return ThreadState::kIdle;
      auto* thread = warp->get_thread(static_cast<int>(lane_id));
      if (!thread) return ThreadState::kIdle;
      return map_state(thread->get_state());  // reuse existing helper at device_api_impl.cc:45-53
  }
  ```
  - **CRITICAL**: 使用 `thread->get_state()` 而非 `thread->state` (后者不存在)
  - **注释**: 此方法为 READ-ONLY, 不修改任何 state (per `state-modification-audit` skill)
- [ ] 2.3 验证 Phase 1 build success: `cmake --build build` — expected 100% build
- [ ] 2.4 验证 Phase 1 ctest: `ctest --test-dir build --output-on-failure` — expected 249/249 PASS (无新测试)
- [ ] 2.5 commit "feat(ptxemu): phase 2.2.1 warp_exe_once + get_thread_state delegation"
  - 必须验证 `git diff --stat` 仅修改 `src/ptxemu/device_api_impl.cc` (2 方法实现, ~30 LOC)
  - 必须未修改 `include/ptxemu/device_api.h` (PTXEMU_API_VERSION=1 冻结 + HSK-8 spec §Decision 5 sizeof visibility)
  - 必须未新增 public 方法

## 3. Phase 2 — `get_warp_status` + `map_thread_status` helper + e2e test + drift_check exemption removal (Commit 2 of 2)

- [ ] 3.1 添加 `map_thread_status(ThreadStatus)` helper function (per design Decision 5) — 位于 `device_api_impl.cc` 匿名命名空间, parallel to `map_state` (L45-53):
  ```cpp
  namespace {
  // Map ptxsim::ThreadStatus (WarpState::threads[i].status) → ptxemu::ThreadState
  // Parallel to map_state(EXE_STATE) helper above.
  // Yielded maps to kIdle (conservative default; ThreadState enum frozen at 4 values
  // per HSK-8 spec §Decision 6 — new value = ABI break).
  ThreadState map_thread_status(ptxsim::ThreadStatus ts) {
      switch (ts) {
          case ptxsim::ThreadStatus::Active:  return ThreadState::kRun;
          case ptxsim::ThreadStatus::Blocked: return ThreadState::kBarSync;
          case ptxsim::ThreadStatus::Exited:  return ThreadState::kExit;
          case ptxsim::ThreadStatus::Yielded: return ThreadState::kIdle;
      }
      return ThreadState::kIdle;
  }
  }  // namespace
  ```
  - **NOTE**: 需要 `#include <ptxsim/thread_state.h>` (per `include/ptxsim/thread_state.h:24`)
- [ ] 3.2 实施 `get_warp_status` (per design Decision 2) — populate EXISTING 5-field WarpStatus struct (device_api.h:69-75), 严格不引入新字段:
  ```cpp
  WarpStatus get_warp_status(uint32_t sm_id, uint32_t warp_id) override {
      if (!g_gpu_context) return WarpStatus{};
      auto* sm = g_gpu_context->get_sm(sm_id);
      if (!sm) return WarpStatus{};
      auto* warp = sm->get_warp(warp_id);
      if (!warp) return WarpStatus{};

      WarpStatus s;
      s.warp_id = warp_id;
      s.sm_id = sm_id;

      const auto& ws = warp->get_warp_state();
      s.lanes.reserve(32);
      for (int i = 0; i < 32; ++i) {
          LaneStatus ls;
          ls.lane_id = static_cast<uint32_t>(i);
          ls.state = map_thread_status(ws.threads[i].status);
          ls.pc = ws.threads[i].pc;
          s.lanes.push_back(ls);
      }

      s.active_count = static_cast<uint32_t>(ws.count_active_lanes());

      // Sum blocked_cycles_remaining across threads, clamped to int32_t
      uint64_t total_blocked = 0;
      for (const auto& thread : ws.threads) {
          total_blocked += thread.blocked_cycles_remaining;
      }
      s.blocked_cycles = (total_blocked > static_cast<uint64_t>(INT32_MAX))
                         ? INT32_MAX
                         : static_cast<int32_t>(total_blocked);

      return s;
  }
  ```
  - **CRITICAL**: 必须经 `warp->get_warp_state()` 取数据, 不得直接访问 WarpContext internal fields
  - **CRITICAL**: 5 个字段必须全部填充 (warp_id / sm_id / lanes / active_count / blocked_cycles)
  - **NOTE**: 必须 `#include <ptxsim/warp_state.h>` (per `include/ptxsim/warp_state.h:14`)
  - **注释**: 此方法为 READ-ONLY, 不修改任何 state (per `state-modification-audit` skill)
- [ ] 3.3 创建 `tests/integration/warp/` 目录 + `tests/integration/warp/CMakeLists.txt` 注册 helper
- [ ] 3.4 创建 `tests/integration/warp/test_warp_status_snapshot.cpp` (per `e2e-delegation-validation/spec.md`):
  - 测试场景: all-active / all-finished / no-active / mixed / blocked-cycles 累加 / warp_id+sm_id fields
  - Catch2 测试 + RAII GpuContextScope fixture
  - 注册到 `tests/integration/CMakeLists.txt`
- [ ] 3.5 创建 `tests/integration/warp/test_device_api_delegation_e2e.cc` (per `e2e-delegation-validation/spec.md`):
  - 测试场景: set_next_pc + warp_exe_once + get_thread_state + get_warp_status
  - 测试场景: set_active_mask + warp_exe_once + get_warp_status.active_count 验证 overwrite
  - Catch2 测试 + RAII GpuContextScope fixture (与 test_set_active_mask_overwrite.cpp 共享 pattern)
  - 注册到 `tests/integration/CMakeLists.txt`
- [ ] 3.6 修改 `tests/integration/simt/test_set_active_mask_overwrite.cpp`:
  - 移除 `WARN + early-return when no warp exists` 防护
  - 替换为 proper warp setup (per `WarpExecutorTestFixture` 在 tests/integration/warp/ 共享)
  - 添加 BarrierModule interaction test
- [ ] 3.7 修改 `.github/workflows/drift_check.yml` Invariant 6:
  - 移除 exemption list 中 `warp_exe_once` / `get_thread_state` / `get_warp_status` 3 个 entries
  - exemption list 变为 EMPTY (0 methods)
  - 更新 Scenario "Deferred stub methods exemption list is EMPTY after this change" 文档
- [ ] 3.8 修改 `include/ptxemu/AGENTS.md`: 更新 `IPtxEmuDevice` method status table (12/12 methods implemented)
- [ ] 3.9 修改 `README.md` §已实现功能 IPtxEmuDevice bullet:
  - 从 "Phase 2.2/2.3 delegation 完成 (commits `4f6b5e1a` + `488fe75e`)" 
  - 改为 "Phase 2.2/2.3 + Phase 2.2.1/2.3.1 delegation 完成 (12/12 methods, commits `<commit-1-hash>` + `<commit-2-hash>`)"
- [ ] 3.10 验证 Phase 2 build + ctest: expected 251/251 PASS (新增 2 测试文件 — `unit_warp_status_snapshot` + `integration_device_api_delegation_e2e` = 249+2=251)
- [ ] 3.11 commit "feat(ptxemu): phase 2.3.1 get_warp_status + e2e delegation test + drift_check exemption removal"

## 4. drift_check + README sync + 验证 (per ptx-lessons-learned §21 + this plan Risk R3)

- [ ] 4.1 验证 drift_check invariants PASS:
  - Invariant 1-5: existing invariants unchanged
  - Invariant 6: 无 empty-body IPtxEmuDevice method stubs (exemption list 现在 EMPTY per Phase 2)
  - **NOTE**: Invariant 7 (per `antlr4-path-hardcoding-fix`) 假设该 change 已合并;若未合并,本 change 的 4.1 验证接受 6 invariants + 后续 antlr4 change 合并后补齐 7 invariants
- [ ] 4.2 验证 ctest 251/251 PASS (新增测试 + zero regressions)
- [ ] 4.3 验证 `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` Postmortem 包含 Phase 2.2.1/2.3.1 resolution note

## 5. PR submission

- [ ] 5.1 push `feat/phase-2-2-1-3-1-followup` 分支
- [ ] 5.2 PR to main (建议标题): "feat(ptxemu): Phase 2.2.1/2.3.1 — complete IPtxEmuDevice delegation (12/12 methods)"
- [ ] 5.3 merge (squash) — commits `<commit-1-hash>` + `<commit-2-hash>` 合并为 1 个
- [ ] 5.4 通知 CppTLM owner: PTX-EMU 端 12/12 IPtxEmuDevice methods 已实现, CppTLM 可以 bump submodule

## 6. Archive + Handoff

- [ ] 6.1 NOT archive change until CppTLM bumps (per HSK-8 follow-up plan 决策点 2)
- [ ] 6.2 archive 时 postmortem 段: 3 个 deferred stubs + 1 deferred e2e test 全部完成 + drift_check Invariant 6 exemption 移除
- [ ] 6.3 update HSK-PROTOCOL-NOTES.md §HSK-8 实践示例: 引用 Phase 2.2.1/2.3.1 作为 HSK-9 准入准备完成案例 (所有 12 methods 已 wired)
- [ ] 6.4 ac-verifier archive-time check: README 包含 "Phase 2.2.1/2.3.1 delegation 完成" 字样
- [ ] 6.5 archive: `mv openspec/changes/phase-2-2-1-3-1-followup openspec/changes/archive/$(date +%Y-%m-%d)-phase-2-2-1-3-1-followup`

## Reference

- **Parent HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` §Phase 3 Task 3.1-3.2
- **Archived change** (this change completes): `openspec/changes/archive/2026-08-25-device-api-delegation/`
- **HSK-8 spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` (Decision 5: sizeof visibility; Decision 6: ThreadState enum)
- **HSK-8 audit**: `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` Postmortem
- **Metis REVISE session**: `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`
- **Stub locations** (before this change):
  - `src/ptxemu/device_api_impl.cc:90-94` (warp_exe_once)
  - `:114-118` (get_thread_state)
  - `:156-160` (get_warp_status)
- **Public types (preserved, no layout change)**:
  - `include/ptxemu/device_api.h:62-66` (LaneStatus 3-field struct: lane_id + state + pc)
  - `:69-75` (WarpStatus 5-field struct: warp_id + sm_id + lanes[vector<LaneStatus>] + active_count + blocked_cycles)
- **Internal APIs**:
  - `SMContext::get_warp(uint32_t) → WarpContext*` (existing)
  - `WarpContext::execute_warp_instruction()` (instance method, `src/ptxsim/core/warp_context.cpp`)
  - `WarpContext::get_thread(int lane_id) → ThreadContext*` (existing)
  - `ThreadContext::get_state() const → EXE_STATE` (`include/ptxsim/thread_context.h:205`)
  - `WarpContext::get_warp_state() → WarpState&` (existing)
  - `WarpState::threads[32]` (`include/ptxsim/warp_state.h:14`)
  - `WarpState::count_active_lanes()` (`include/ptxsim/warp_state.h:40`)
  - `ThreadState::blocked_cycles_remaining` (`include/ptxsim/thread_state.h:40`)
- **Public API frozen**: `include/ptxemu/device_api.h:117` `static_assert(PTXEMU_API_VERSION == 1, ...)`
- **drift_check workflow**: `.github/workflows/drift_check.yml` (Invariant 6 modified — exemption list shrunk)
- **Skills referenced**:
  - `ptx-lessons-learned` §1 (跨模块状态翻译) + §3 (分 Phase commit — 2 Phase) + §4 (baseline worktree) + §7 (Metis pre-impl) + §21 (README Checklist I)
  - `ptx-barrier-mechanism` (set_active_mask overwrite semantics for BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES)
  - `ptx-instruction-pipeline` (warp_exe_once state mutation hot path)
  - `state-modification-audit` (get_thread_state / get_warp_status read-only verification)
  - `test-coverage-enforcer` (e2e test via execute_warp_instruction)
  - `oracle-prompting` (Decision 5 split rationale)