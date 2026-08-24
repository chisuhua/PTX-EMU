# device-api-delegation — Tasks

## 1. Pre-Implementation 准备 (per ptx-lessons-learned §4 + §7 + this Oracle plan §Phase 2)

- [ ] 1.1 MUST: 验证基线 worktree `.worktrees/device-api-delegation-baseline` 编译通过(per §4 15-20min)
- [ ] 1.2 MUST: 跑基线 ctest 246/246 PASS(per §4)
- [ ] 1.3 MUST: Metis pre-impl review 审计 4 OpenSpec artifacts (per §7 + §H in ptx-lessons-learned)
- [ ] 1.4 应用 Metis MUST-RESOLVE 列表,重审直至 ⚠️→GO 或 ✅
- [ ] 1.5 NOTE: 4 artifacts 范围数字一致性按 Checklist J 校验
- [ ] 1.6 MUST: `git add openspec/changes/2026-08-24-device-api-delegation/` + commit (per Checklist E)
- [ ] 1.7 创建 `feat/device-api-delegation` 分支从 origin main (HEAD `530bd6ca` or later)

## 2. Phase 2.2 delegation 实施 (per design Decision 1 + spec/ptxemu-device-api-delegation)

- [ ] 2.1 实施 `set_scoreboard` delegation (委托到 `SMContext::set_scoreboard(IScoreboard*)` at `include/ptxsim/sm_context.h:87`)
- [ ] 2.2 实施 `set_active_mask` delegation (overwrite 语义, NOT OR-merge, per BUG-RETHANG)
- [ ] 2.3 实施 `set_next_pc` delegation (使用 `set_pc()` + `commit_pc()`, NOT `force_set_pc`, per AGENTS.md ANTI-PATTERNS L85)
- [ ] 2.4 unit test 覆盖 3 个 delegation (`tests/unit/ptxemu/test_device_api_delegation.cpp`)
- [ ] 2.5 e2e test 经 `execute_warp_instruction` 验证 thread PC (`tests/integration/warp/test_device_api_delegation_e2e.cc` per test-coverage-enforcer)
- [ ] 2.6 regression guard test for `set_active_mask` overwrite (`tests/integration/warp/test_set_active_mask_overwrite.cpp`)
- [ ] 2.7 full clean rebuild + ctest 验证 252/252 (246 baseline + 6 new unit) + drift_check 5 invariants PASS
- [ ] 2.8 commit "feat(ptxemu): phase 2.2 set_scoreboard + set_active_mask + set_next_pc delegation"

## 3. Phase 2.3 attach_timing 实施 (per design Decision 4 + spec/ptxemu-device-api-delegation)

- [ ] 3.1 实施 `attach_timing` HSK-4 vendored interface injection (IScoreboard/IPipelineLatencyProvider/ITensorCoreTiming)
- [ ] 3.2 注入到 SMContext timing hooks (per design.md Decision 4)
- [ ] 3.3 unit test 覆盖 `attach_timing` (`tests/unit/ptxemu/test_device_api_attach_timing.cpp`)
- [ ] 3.4 full clean rebuild + ctest 验证 253/253 (252 baseline + 1 new e2e) + drift_check 5 invariants PASS
- [ ] 3.5 commit "feat(ptxemu): phase 2.3 attach_timing HSK-4 vendored interface injection"

## 4. drift_check Invariant 6 + README sync + 验证 (per ptx-lessons-learned §21 + this plan Risk R3)

- [ ] 4.1 扩展 `.github/workflows/drift_check.yml` Invariant 6: 验证 `src/ptxemu/device_api_impl.cc` 中无 `return false` (除 `attach_timing` 的 void 返回)
- [ ] 4.2 更新 `README.md` §已实现功能 IPtxEmuDevice 条目(添加 "Phase 2.2/2.3 delegation 完成" 字样, per §21 Checklist I)
- [ ] 4.3 drift_check workflow 验证 delegation 不可回归 stub (`grep 'return false' src/ptxemu/device_api_impl.cc` 应为 0 except `attach_timing`)
- [ ] 4.4 ctest PASS(253 tests: 246 baseline + 6 new unit + 1 e2e + 0 regression guard counted in 6)

## 5. PR submission (per archive/2026-08-24-ptxemu-public-device-api §8 先例)

- [ ] 5.1 push feat/device-api-delegation 分支
- [ ] 5.2 PR #15 to main
- [ ] 5.3 merge (squash)
- [ ] 5.4 通知 CppTLM owner 可重新 bump (issue #22 评论)

## 6. Archive + Handoff

- [ ] 6.1 NOT archive change until CppTLM bumps (per HSK-8 ack 决策点 1)
- [ ] 6.2 archive 时 postmortem 段: Phase 2.2/2.3 实施回顾 + BUG-RETHANG 警告(已在 source code 注释体现)
- [ ] 6.3 update HSK-PROTOCOL-NOTES.md §HSK-8 实践示例 (引用本次 delegation 实施作为 HSK-9 准入准备示例)
- [ ] 6.4 ac-verifier archive-time check: README contains "Phase 2.2/2.3 delegation 完成" (per Decision 5)

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