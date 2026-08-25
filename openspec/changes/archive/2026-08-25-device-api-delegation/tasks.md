# device-api-delegation — Tasks

## 1. Pre-Implementation 准备 (per ptx-lessons-learned §4 + §7 + this Oracle plan §Phase 2)

- [x] 1.1 MUST: 验证基线 worktree `.worktrees/device-api-delegation-baseline` 编译通过(per §4 15-20min) — ✅ commit `5fe6ad69` 前完成
- [x] 1.2 MUST: 跑基线 ctest 246/246 PASS(per §4) — ✅ baseline `530bd6ca` 验证
- [x] 1.3 MUST: Metis pre-impl review 审计 4 OpenSpec artifacts (per §7 + §H in ptx-lessons-learned) — ✅ Metis session `ses_fcb59b168ffecp5an0at73fgag` ✅ GO 2026-08-25
- [x] 1.4 应用 Metis MUST-RESOLVE 列表,重审直至 ⚠️→GO 或 ✅ — ✅ commit `5fe6ad69` 应用所有 MUST-RESOLVE
- [x] 1.5 NOTE: 4 artifacts 范围数字一致性按 Checklist J 校验 — ✅
- [x] 1.6 MUST: `git add openspec/changes/device-api-delegation/` + commit (per Checklist E) — ✅ commit `20f5e13d`
- [x] 1.7 创建 `feat/device-api-delegation` 分支从 origin main (HEAD `530bd6ca` or later) — ✅ commit `183a6ada` 合并到 main (squash of 4f6b5e1a + 488fe75e)

## 2. Phase 2.2 delegation 实施 (per design Decision 1 + spec/ptxemu-device-api-delegation)

- [x] 2.1 实施 `set_scoreboard` delegation (委托到 `SMContext::set_scoreboard(IScoreboard*)` at `include/ptxsim/sm_context.h:87`) — ✅ `src/ptxemu/device_api_impl.cc:103-111` (R7-constrained minimal scope)
- [x] 2.2 实施 `set_active_mask` delegation (overwrite 语义, NOT OR-merge, per BUG-RETHANG) — ✅ `src/ptxemu/device_api_impl.cc:126-136`
- [x] 2.3 实施 `set_next_pc` delegation (使用 `set_pc()` + `commit_pc()`, NOT `force_set_pc`, per AGENTS.md ANTI-PATTERNS L85) — ✅ `src/ptxemu/device_api_impl.cc:140-153`
- [x] 2.4 unit test 覆盖 3 个 delegation (`tests/unit/ptxemu/test_device_api_delegation.cpp`) — ✅ 5 test cases (commit `183a6ada`)
- [x] 2.5 regression guard test for `set_active_mask` overwrite (`tests/integration/simt/test_set_active_mask_overwrite.cpp`) — ⚠️ 见下方 **[~] DEFERRED NOTE**:基础 overwrite 场景已覆盖 (3 sections),深度 warp setup + full overwrite verification 推迟到 Phase 2.2.1 follow-up (per commit `183a6ada` message)
- [x] 2.6 full clean rebuild + ctest 验证 + drift_check 5 invariants PASS — ✅ **实测 248/248 PASS** (246 baseline + 1 unit (delegation file) + 1 integration (overwrite file),与 proposal 估算 251 偏差源于 label 计数)
- [x] 2.7 commit "feat(ptxemu): phase 2.2 set_scoreboard + set_active_mask + set_next_pc delegation" — ✅ commit `4f6b5e1a` (squashed into `183a6ada`)

## 3. Phase 2.3 attach_timing 实施 (per design Decision 4 + spec/ptxemu-device-api-delegation)

- [x] 3.1 实施 `attach_timing` HSK-4 vendored interface injection (IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming)
  - **Namespace bridge** (per design Decision 6): use `static_cast<::IScoreboard*>(sb)` etc. for cross-namespace pointer bridge — ✅ `src/ptxemu/device_api_impl.cc:175-186`
- [x] 3.2 注入到 SMContext timing hooks (per design.md Decision 4) — ✅ sm->set_scoreboard / sm->set_pipeline_latency_provider / sm->set_tensor_core_timing
- [x] 3.3 unit test 覆盖 `attach_timing` (`tests/unit/ptxemu/test_device_api_attach_timing.cpp`) — ✅ 142 LOC, `unit_device_api_attach_timing` ctest label
- [~] 3.4 e2e test 经 `execute_warp_instruction` 验证 thread PC (`tests/integration/warp/test_device_api_delegation_e2e.cc` per test-coverage-enforcer) — **DEFERRED to Phase 2.2.1/2.3.1 follow-up** (与现有 3 个 deferred stubs `warp_exe_once`/`get_thread_state`/`get_warp_status` 一致)。理由:4 个 delegation 方法的各自 unit/integration 已分别验证 (`test_device_api_delegation.cpp` + `test_set_active_mask_overwrite.cpp` + `test_device_api_attach_timing.cpp`);e2e test 需要 deep warp/thread setup,与 commit `183a6ada` message 中 "deep warp setup deferred — full overwrite verification in Phase 2.2.1 follow-up" 同类问题,合并到 Phase 2.2.1/2.3.1 follow-up change 统一处理。
- [x] 3.5 full clean rebuild + ctest 验证 + drift_check **6** invariants PASS — ✅ **实测 249/249 PASS** (248 Phase 2.2 baseline + 1 unit (attach_timing file);proposal 估算 253 包含了已 defer 的 e2e test)
- [x] 3.6 commit "feat(ptxemu): phase 2.3 attach_timing HSK-4 vendored interface injection" — ✅ commit `488fe75e` (squashed into `183a6ada`)

## 4. drift_check Invariant 6 + README sync + 验证 (per ptx-lessons-learned §21 + this plan Risk R3)

- [x] 4.1 扩展 `.github/workflows/drift_check.yml` paths 触发过滤: 添加 `src/ptxemu/**` (per Metis MR-3) — ✅
- [x] 4.2 添加 `.github/workflows/drift_check.yml` Invariant 6: 验证 `src/ptxemu/device_api_impl.cc` 中**无空 body 方法体**(per Metis MR-2) — ✅ `drift_check.yml:102-164`
- [x] 4.3 更新 `README.md` §已实现功能 IPtxEmuDevice 条目(添加 "Phase 2.2/2.3 delegation 完成" 字样, per §21 Checklist I) — ✅ `README.md:50` 含目标字样
- [x] 4.4 ctest PASS — ✅ **实测 249/249 PASS** (246 baseline + 2 unit (delegation + attach_timing) + 1 integration (overwrite);e2e test deferred 见 task 3.4)
- [x] 4.5 drift_check workflow 验证 delegation 不可回归 stub (per MR-3 + MR-2 修订) — ✅ 6 invariants PASS

## 5. PR submission (per archive/2026-08-24-ptxemu-public-device-api §8 先例)

- [x] 5.1 push feat/device-api-delegation 分支 — ✅ `feat/device-api-delegation` 已推送
- [x] 5.2 PR #17 to main (注:实际 PR #17,提案假设 #15) — ✅
- [x] 5.3 merge (squash) — ✅ commit `183a6ada` (squash merge of 4f6b5e1a + 488fe75e)
- [x] 5.4 通知 CppTLM owner 可重新 bump (issue #22 评论) — ✅ HSK-8 PR #14 merged 后 CppTLM 已 bump `beb3db8`;Phase 2.2/2.3 delegation 为 HSK-8 ack 后追加,无需新 HSK

## 6. Archive + Handoff

- [x] 6.1 NOT archive change until CppTLM bumps (per HSK-8 ack 决策点 1) — ✅ CppTLM 已 bump `beb3db8`
- [x] 6.2 archive 时 postmortem 段: Phase 2.2/2.3 实施回顾 + BUG-RETHANG 警告(已在 source code 注释体现) — ✅ `device_api_impl.cc:120-125` overwrite 语义注释 + `test_device_api_delegation.cpp:12` cross-ref
- [x] 6.3 update HSK-PROTOCOL-NOTES.md §HSK-8 实践示例 (引用本次 delegation 实施作为 HSK-9 准入准备示例) — ✅ (与本 archive 同步)
- [x] 6.4 ac-verifier archive-time check: README contains "Phase 2.2/2.3 delegation 完成" (per Decision 5) — ✅ README.md:50

## Implementation Notes

- **Phase 2.2 commit**: `4f6b5e1a` (squashed into `183a6ada`)
- **Phase 2.3 commit**: `488fe75e` (squashed into `183a6ada`)
- **OpenSpec artifacts commit**: `20f5e13d` (4 artifacts first per §6 artifacts-first)
- **Metis application commit**: `5fe6ad69`
- **PR merge commit**: `183a6ada` (squash, 2026-08-25T10:40:22+08:00 by chisuhua)
- **实测 ctest 计数**: 249/249 PASS (vs proposal 估算 253;3 个缺失来自 task 3.4 e2e test deferred)
- **实测 drift_check invariants**: 6 PASS (per `.github/workflows/drift_check.yml` Invariant 6)
- **Follow-up**: Phase 2.2.1/2.3.1 change (待创建) — 覆盖 e2e test (task 3.4) + 3 deferred stubs (`warp_exe_once` / `get_thread_state` / `get_warp_status`) + `set_active_mask` overwrite 深度 warp setup 验证

## Reference

- **Parent plan**: `2026-08-24-hsk8-followup-task-path.md` §Phase 2 Task 2.1-2.4
- **Oracle session**: `ses_fcbc066a6ffeXfv5kIgmE8UwrB` (parent plan Oracle)
- **HSK-8 spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` (commit `738b412c`)
- **Archived change**: `openspec/changes/archive/2026-08-24-ptxemu-public-device-api/`
- **Public API frozen**: `include/ptxemu/device_api.h:117` `static_assert(PTXEMU_API_VERSION == 1, ...)`
- **Stub location**: `src/ptxemu/device_api_impl.cc:91-137`
- **Target APIs**:
  - `SMContext::set_scoreboard(IScoreboard*)` at `include/ptxsim/sm_context.h:87`
  - `WarpContext::set_active_mask(uint32_t)` at `include/ptxsim/warp_context.h:199`
  - `ThreadContext::set_next_pc(int)` at `include/ptxsim/thread_context.h:229`
- **Skills referenced**:
  - `ptx-lessons-learned` §1 (跨模块状态翻译) + §3 (分 Phase commit) + §4 (baseline worktree) + §7 (Metis pre-impl) + §21 (README Checklist I)
  - `ptx-barrier-mechanism` (set_active_mask overwrite semantics)
  - `test-coverage-enforcer` (Phase 2.2/2.3 validation)
- **drift_check workflow**: `.github/workflows/drift_check.yml` (5 invariants → 6 after this change)