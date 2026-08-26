# Tasks: attach-timing-consumer-e2e-tests

> **Pre-implementation**: ✅ Metis pre-impl review completed (4 BLOCKERs identified). ✅ Oracle consultation pivoted to Path A (reuse existing `WarpExecutorTestFixture`). ✅ Falsification via static analysis confirmed (Q1+Q2 hypotheses verified, no build required).
>
> **Pivot summary**: Original design proposed `sm_test_access::add_warp` friend namespace. Path B rejected due to 4 BLOCKERs + lessons-learned §1 violation. New design reuses existing `WarpExecutorTestFixture` with backward-compatible optional `statements` parameter.

## 1. Pre-flight: Baseline worktree setup (per `ptx-lessons-learned` §4)

- [x] 1.1 建立 baseline worktree: `git worktree add .worktrees/attach-timing-baseline main`
- [x] 1.2 baseline 全量 build (with testing enabled): `cmake -S .worktrees/attach-timing-baseline -B .worktrees/attach-timing-baseline/build -DCMAKE_BUILD_TYPE=Release -DPTXEMU_BUILD_TESTING=ON && cmake --build .worktrees/attach-timing-baseline/build -j$(nproc)` (15-20 min)
- [x] 1.3 baseline ctest 验证: `cd .worktrees/attach-timing-baseline/build && ctest --output-on-failure` → 期望 251/251 PASS
- [x] 1.4 记录 baseline timestamp + commit hash 到 `.opencode/notes/attach-timing-baseline.txt`

## 2. Phase 1: Fixture parameterization (no tests yet)

- [x] 2.1 `include/ptxemu/testing/warp_executor_test_fixture.h:44`: 修改构造函数签名为 `WarpExecutorTestFixture(std::vector<StatementContext> statements = {})`
- [x] 2.2 `include/ptxemu/testing/warp_executor_test_fixture.h:57`: 删除局部空 `std::vector<StatementContext> statements;`，改用构造参数
- [x] 2.3 `include/ptxemu/testing/warp_executor_test_fixture.h:63`: 把构造参数 `statements` 传给 `block->init(..., statements, ...)`
- [x] 2.4 验证: 251/251 ctest PASS 不变 (3 fixture-using tests compile + run as before due to default param) + drift_check 7 invariants PASS
- [x] 2.5 单独跑 3 fixture-using tests: `ctest -R "test_set_active_mask_overwrite|test_warp_status_snapshot|test_device_api_delegation_e2e" --output-on-failure` → 期望全部 PASS
- [x] 2.6 Commit Phase 1: `git add include/ptxemu/testing/warp_executor_test_fixture.h && git commit -m "test(fixture): parameterize WarpExecutorTestFixture with optional statements (backward-compatible)"`
- [x] 2.7 [Revert safety check] 验证 Phase 1 commit 独立可 revert: `git revert HEAD` 后 build + ctest 仍 251/251 PASS

## 3. Phase 2: 4 integration tests

- [x] 3.1 创建 `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp` 骨架 (header includes + anonymous namespace + 4 TEST_CASE stubs)
- [x] 3.2 实现 mock class: `TrackingScoreboard` (复用 `tests/integration/cpptlm/test_scoreboard_allocation.cpp:73-87` 模式 + 加 `alloc_calls/release_calls` counters)
- [x] 3.3 实现 mock class: `FixedPipeline` (复用 `tests/integration/cpptlm/test_scoreboard_allocation.cpp:92-103` 模式 + 加 `cycles_calls` counter)
- [x] 3.4 实现 mock class: `CountingTC` (NEW ~15 行: `ITensorCoreTiming` 子类 + `get_latency_calls/get_throughput_calls` counters)
- [x] 3.5 实现 helper: `attach_through_device(dev, sb, pl, tc)` 完成 namespace bridge round-trip (`static_cast<void*>`) + 调用 `dev->attach_timing(...)`
- [x] 3.6 实现 G1 TEST_CASE: `attach_timing: scoreboard queried by exe_once step_a/c` — `WarpExecutorTestFixture` 默认空 statements + `attach_through_device(dev, &sb, nullptr, nullptr)` + `dev->sm_exe_once(0)` → 期望 `sb.alloc_calls > 0` AND `sb.release_calls > 0`
  - **NOTE**: G1 依赖 `warp_->is_finished()==false` + exe_once 调度器选择该 warp。fixture 提供的 warp（空 statements, threads 默认 active）应当 schedulable。若 alloc_calls==0 则需回退到 BLOCKER 4 路径（手动注入 S_FMA statements）。
- [x] 3.7 实现 G2 TEST_CASE: `attach_timing: pipeline queried by step_b (S_FMA)` — `WarpExecutorTestFixture` + `make_ffma("%f0", "%f1", "%f2", "%f3")` + 直接调 `SMContext::step_b_set_blocked_cycles(...)` → 期望 `pipeline.cycles_calls > 0`
- [x] 3.8 实现 G3 TEST_CASE: `attach_timing: tensor_core queried by step_b (S_TCGEN05_MMA)` — `make_stmt(S_TCGEN05_MMA)` + `GenericInstr{Q_F16}` + pipeline mock 返回 0.0 + 直接调 `step_b_set_blocked_cycles` → 期望 `tc.get_latency_calls > 0`
- [x] 3.9 实现 G4 TEST_CASE: `attach_timing: e2e — exe_once queries all 3 injected interfaces` — `WarpExecutorTestFixture({make_ffma(...)})` (传入 FFMA!) + 全 3 mock + `dev->sm_exe_once(0)` → 期望 `scoreboard alloc/release > 0` + `pipeline.cycles_calls > 0` + `tensor_core.get_latency_calls == 0` (S_FMA 非 TC, pipeline 路径优先级)
- [x] 3.10 `tests/integration/CMakeLists.txt`: 添加 `add_catch_test(integration_attach_timing_consumer_e2e cpptlm/test_attach_timing_consumer_e2e.cpp)` + LABELS "integration;cpptlm;attach_timing"
- [x] 3.11 验证: 252/252 ctest PASS (251 baseline + 1 new target with 4 TEST_CASE) + drift_check 7 invariants PASS
- [x] 3.12 单独运行新测试: `ctest -R integration_attach_timing_consumer_e2e --output-on-failure` → 期望 4/4 sub-tests PASS
- [x] 3.13 Commit Phase 2: `git add tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp tests/integration/CMakeLists.txt && git commit -m "test(cpptlm): add 4 attach_timing consumer e2e tests (G1-G4)"`
- [x] 3.14 [Revert safety check] 验证 Phase 2 commit 独立可 revert: `git revert HEAD` 后 build + ctest 仍 251/251 PASS

## 4. Phase 3: docs sync + archive

- [x] 4.1 验证无 AGENTS.md 改动需求 (drift_check 不需新 invariant; fixture 头改动在 `include/ptxemu/testing/`，不在 `device_api.h`)
- [x] 4.2 验证无 ADR 改动需求 (HSK-8 协议不变, 仅测试工具增强)
- [x] 4.3 更新 `openspec/changes/attach-timing-consumer-e2e-tests/tasks.md`: 所有 checkbox 标记为 [x]
- [x] 4.4 验证 OpenSpec artifacts git-tracked: `git ls-files openspec/changes/attach-timing-consumer-e2e-tests/` 不为空 (per ptx-lessons-learned Checklist E)
- [x] 4.5 Commit docs: `git add openspec/changes/attach-timing-consumer-e2e-tests/ && git commit -m "docs(openspec): mark all tasks complete for attach-timing-consumer-e2e-tests"`
- [x] 4.6 Archive change: `openspec archive attach-timing-consumer-e2e-tests --yes`
- [x] 4.7 验证 archive 后: `git log --all --oneline -- openspec/changes/attach-timing-consumer-e2e-tests/` 包含 archive commit
- [x] 4.8 最终验证: `ctest --output-on-failure` → 252/252 PASS + drift_check 7 invariants PASS

## 5. Post-change cleanup

- [x] 5.1 移除 baseline worktree: `git worktree remove .worktrees/attach-timing-baseline` (per ptx-lessons-learned §4 step 4)
- [x] 5.2 通知 CppTLM 维护者: PTX-EMU 可供 submodule bump (CppTLM `plans/ptxemu-followup-roadmap.md` 后续 Phase)
- [x] 5.3 [可选] 写 postmortem: `.opencode/notes/attach-timing-consumer-e2e-postmortem.md` 沉淀 fixture 参数化经验

## 关键约束 (CRITICAL)

- **MUST**: 每个 Phase 独立 commit, 失败立即 revert 该 Phase (per `ptx-lessons-learned` §3)
- **MUST**: Phase 1+2 之前分别建立/使用 baseline worktree (per §4)
- **MUST**: OpenSpec artifacts 在 archive 前必须 git-tracked (per Checklist E)
- **MUST NOT**: 不要 bump `PTXEMU_API_VERSION` 或修改 `include/ptxemu/device_api.h` (HSK-9 触发)
- **MUST NOT**: 不要 touch production code (SMContext, GPUContext, device_api_impl.cc) — 仅测试 + fixture header
- **MUST NOT**: 不要 touch `GpuContextScope` (existing unit test pattern) — blast radius 不可控 (per Oracle Path A rejection of Path B)
- **NOTE**: G1/G4 失败 → 检查 `make_ffma` 是否正确传入 fixture + warp 是否 schedulable (`is_finished()==false`)
- **NOTE**: G3 失败 → 检查 `stmt.data = GenericInstr{Q_F16}` 是否正确设置（`map_instruction_to_tc_precision` 需要 qualifier 才能非默认 FP16）