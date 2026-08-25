# phase-2-2-1-3-1-followup — Design

> **REVISION HISTORY** (per Metis session `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`):
> - Initial design assumed `WarpStatus` was empty/4-field; Metis verification confirmed `WarpStatus` is 5-field struct at `include/ptxemu/device_api.h:69-75`
> - Initial design assumed `thread->state` field existed; Metis confirmed correct API is `thread->get_state()` returning `EXE_STATE` (`include/ptxsim/thread_context.h:205`)
> - Initial design assumed `WarpContext::active_mask_` / `lanes_` / `blocked_cycles_remaining` were direct fields; Metis confirmed correct access is via `warp->get_warp_state().threads[]`

## Context

`device-api-delegation` (archived 2026-08-25, commit `183a6ada`) implemented 4 of 12 `IPtxEmuDevice` methods. **3 deferred stubs** + **1 deferred e2e test** + **1 incomplete overwrite verification** remain. The 3 deferred stubs prevent CppTLM-side (per HSK-8 spec §CppTLM 端接受条件 #1) from:

1. **Per-warp scheduling**: `warp_exe_once` returns `-1` instead of advancing one warp
2. **Per-thread diagnostics**: `get_thread_state` returns hardcoded `ThreadState::kIdle` instead of reading `ThreadContext::get_state()`
3. **Warp snapshot**: `get_warp_status` returns default-constructed `WarpStatus{}` instead of populating its 5 existing fields

**Public ABI surface FROZEN** at `PTXEMU_API_VERSION=1` (per HSK-8 spec §Decision 5: "sizeof visibility is mandatory, pure data only"). The `WarpStatus` 5-field struct at `include/ptxemu/device_api.h:69-75` MUST be preserved (no layout changes). The change populates existing fields only.

## Goals / Non-Goals

**Goals:**
- Implement `IPtxEmuDevice::warp_exe_once` delegating to `SMContext::get_warp(warp_id)->execute_warp_instruction()` (instance method)
- Implement `IPtxEmuDevice::get_thread_state` reading `ThreadContext::get_state()` + map `EXE_STATE` → `ptxemu::ThreadState` via existing `map_state` helper
- Implement `IPtxEmuDevice::get_warp_status` populating the existing 5-field `WarpStatus` struct:
  - `warp_id`, `sm_id` from input parameters
  - `lanes` (std::vector<LaneStatus>) from `warp->get_warp_state().threads[]`
  - `active_count` from `WarpState::count_active_lanes()`
  - `blocked_cycles` from sum of `threads[i].blocked_cycles_remaining`
- Add `tests/integration/warp/test_device_api_delegation_e2e.cc` e2e test
- Extend `tests/integration/simt/test_set_active_mask_overwrite.cpp` removing WARN+early-return guard
- Add `tests/integration/warp/test_warp_status_snapshot.cpp` unit test
- Reduce drift_check Invariant 6 deferred-stubs exemption from 3 to 0

**Non-Goals:**
- `cpp-tlm-consumes-ptxemu-device` reverse consumption (HSK-9 gated)
- New IPtxEmuDevice methods (would require HSK-9)
- `attach_timing` reverse-direction consumer wiring (separate concern)
- **Modifying `WarpStatus` / `LaneStatus` struct layouts** (would require HSK-9 ABI bump, frozen at v1)

## Decisions

### Decision 1: `warp_exe_once` calls instance method, not global `exe_once()`

**Choice**: Delegate to `g_gpu_context->get_sm(sm_id)->get_warp(warp_id)->execute_warp_instruction()` (instance method).

**Rationale**:
- HSK-8 spec §CppTLM 端接受条件 #1: 1:1 mapping to S1 facade. S1 facade calls `warp_exe_once(sm_id, warp_id)` to advance **one** warp (per-warp scheduler semantics)
- Current implementation `return -1` forces CppTLM scheduler to fall back to global `exe_once()`, which advances ALL warps in ALL SMs — defeats per-warp scheduling intent
- Per `ptx-instruction-pipeline` skill, `WarpContext::execute_warp_instruction()` is the canonical entry point for single-warp execution
- Validates `warp_exe_once` returns 0 on success and -1 on invalid sm_id/warp_id (mirroring `sm_exe_once` pattern at L82-88)

**Alternatives considered**:
- Add new instance-based scheduler state (instance of `GPUContext` per IPtxEmuDevice): rejected — bigger refactor, would require ABI bump
- Return success but silently skip execution: rejected — defeats 1:1 facade mapping

### Decision 2: `get_warp_status` populates EXISTING 5-field struct (no new types)

**Choice**: Populate the 5 existing fields of `WarpStatus` at `include/ptxemu/device_api.h:69-75`:
```cpp
struct WarpStatus {
    uint32_t warp_id = 0;
    uint32_t sm_id = 0;
    std::vector<LaneStatus> lanes;
    uint32_t active_count = 0;
    int32_t blocked_cycles = 0;
};
```

**Rationale**:
- **HSK-8 spec §Decision 5**: sizeof visibility is mandatory, pure data only. Any struct layout change = ABI break = requires HSK-9.
- The 5 existing fields already cover all needed semantics:
  - `warp_id` + `sm_id` — input parameters
  - `lanes` — per-lane `LaneStatus` (lane_id + state + pc)
  - `active_count` — number of active (non-exited) lanes
  - `blocked_cycles` — sum of per-thread blocked_cycles_remaining

**Alternatives considered**:
- Add new `bool finished` / `std::array<ThreadState, 32>` / `uint32_t active_mask` fields: rejected — would change struct sizeof, violates HSK-8 spec §Decision 5
- Use separate `WarpStatus` struct in a new header: rejected — would create ABI ambiguity

### Decision 3: 2-Phase commit structure (not 3+)

**Choice**: 2 atomic commits:
- **Commit 1**: `warp_exe_once` + `get_thread_state` (1 state-mutating + 1 read-only pair)
- **Commit 2**: `get_warp_status` + `map_thread_status` helper + e2e test + drift_check Invariant 6 exemption removal

**Rationale**:
- Per `ptx-lessons-learned` §3 multi-phase criterion: ≥3 commits OR independent rollback granularity
- Commit 1 and 2 are independent rollback units (Commit 1 fixes scheduler path, Commit 2 fixes diagnostics + tests)
- Splitting further creates artificial dependencies — `get_warp_status` test setup shares fixtures with `warp_exe_once` e2e
- Total ~150 LOC across 2 commits is manageable

**Alternatives considered**:
- Single commit: rejected — too large, harder to bisect if regression
- 3+ commits (one per method): rejected — Commit 3 (`get_warp_status`) depends on Commit 2 e2e infra for snapshot test

### Decision 4: e2e test 路径 — `tests/integration/warp/` 新建

**Choice**: New directory `tests/integration/warp/` for e2e tests driven via `WarpContext::execute_warp_instruction`.

**Rationale**:
- Per `test-coverage-enforcer` skill, e2e tests for delegation should be in `tests/integration/warp/` (not `tests/integration/simt/` or `tests/unit/ptxemu/`)
- The existing `tests/integration/simt/test_set_active_mask_overwrite.cpp` stays in `simt/` (SIMT-level regression guard)
- New directory `warp/` mirrors existing `simt/` and `barrier/` directories' testing patterns

**Alternatives considered**:
- Reuse `tests/integration/simt/`: rejected — different testing concern
- New `tests/e2e/ptxemu/`: rejected — reserved for full CUDA kernel e2e tests

### Decision 5: New `map_thread_status` helper for `ptxsim::ThreadStatus` → `ptxemu::ThreadState`

**Choice**: Add `map_thread_status(ThreadStatus ts)` parallel to existing `map_state(EXE_STATE s)` (device_api_impl.cc:45-53).

**Rationale**:
- `WarpState::threads[i].status` is `ptxsim::ThreadStatus` (Active/Blocked/Exited/Yielded), NOT `EXE_STATE`
- Existing `map_state(EXE_STATE)` helper maps 4 values: IDLE/RUN/EXIT/BAR_SYNC
- Required mapping for `get_warp_status`:
  - `ThreadStatus::Active` → `ThreadState::kRun`
  - `ThreadStatus::Blocked` → `ThreadState::kBarSync`
  - `ThreadStatus::Exited` → `ThreadState::kExit`
  - `ThreadStatus::Yielded` → `ThreadState::kIdle`
- Helper is pure function (no state mutation), easily testable

**Alternatives considered**:
- Inline conversion in `get_warp_status`: rejected — duplicates pattern of `map_state`, less readable
- Extend `map_state` to take a variant: rejected — breaks existing callers

## Risks / Trade-offs

| Risk | Severity | Mitigation |
|------|----------|------------|
| `warp_exe_once` 引入 state mutation hot path — BUG-RETHANG / BUG-POSTBARRIER-TWOHALVES regression vector | HIGH | (1) 复用 `BarrierModule::release_warp_barrier` overwrite semantics (per `ptx-barrier-mechanism`);(2) e2e test 真实 warp setup + BarrierModule interaction;(3) `set_active_mask` overwrite 回归测试扩展 |
| `get_thread_state` 误用 state-mutating accessor | MEDIUM | Per `state-modification-audit`: verify all accessors are const-correct (`get_state() const` per thread_context.h:205) |
| `WarpContext::get_warp_state()` accessor 不可用 | LOW | 已确认存在 (warp_context.h 调用 `get_warp_state()` 返回 `WarpState&`);验证后实施 |
| 3 个方法同时实施 — high LOC churn + regression surface | MEDIUM | (1) baseline worktree per `ptx-lessons-learned` §4;(2) 2-Phase commit 结构;(3) 每个 commit 跑全 ctest + drift_check |
| e2e test 需要 deep warp setup — 与 `set_active_mask` overwrite 测试相同问题 | MEDIUM | (1) 在新 `tests/integration/warp/` 目录建立共享 fixture helper `WarpExecutorTestFixture`;(2) `set_active_mask` overwrite 防护移除在该 fixture 上重新构建 |
| `cpp 不暴露` 约束保持 | LOW | 不引入新 public 方法或类型;仅填充 existing public `WarpStatus` 5 字段 |
| drift_check Invariant 6 exemption 移除 — 引入硬失败 | LOW | 3 个方法真实委托实现后才移除 exemption;如果任意方法 revert, exemption 必须恢复 |

## Migration Plan

**Pre-deployment** (per `ptx-lessons-learned` §4):
```bash
git worktree add .worktrees/phase-2-2-1-3-1-baseline HEAD
cd .worktrees/phase-2-2-1-3-1-baseline
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build --output-on-failure
# Expected: 249/249 PASS
```

**Deployment** (2-Phase):
```bash
# Phase 1: warp_exe_once + get_thread_state
git checkout -b feat/phase-2-2-1-3-1-followup
# Edit src/ptxemu/device_api_impl.cc:90-94 (warp_exe_once) + :114-118 (get_thread_state)
ctest --test-dir build --output-on-failure
git commit -m "feat(ptxemu): phase 2.2.1 warp_exe_once + get_thread_state delegation"

# Phase 2: get_warp_status + map_thread_status + e2e test + drift_check
# Add map_thread_status helper in device_api_impl.cc anonymous namespace
# Edit src/ptxemu/device_api_impl.cc:156-160 (get_warp_status, populate 5 fields)
# Add tests/integration/warp/ directory + 2 new test files
# Edit tests/integration/simt/test_set_active_mask_overwrite.cpp (remove guard)
# Edit .github/workflows/drift_check.yml (Invariant 6 exemption list: 3 → 0)
# Edit include/ptxemu/AGENTS.md + README.md + postmortem.md
ctest --test-dir build --output-on-failure
git commit -m "feat(ptxemu): phase 2.3.1 get_warp_status + e2e delegation test + drift_check exemption removal"
```

**Rollback**:
- Commit 1 failure: `git revert <commit-1-hash>`. No impact on Commit 2 (independently revertable).
- Commit 2 failure: `git revert <commit-2-hash>`. Reapply Commit 1. drift_check Invariant 6 exemption list restored.
- Full rollback: `git worktree remove .worktrees/phase-2-2-1-3-1-baseline && git checkout <pre-change-commit-hash>`.

## Open Questions

**Q1**: `map_thread_status(ThreadStatus::Yielded)` → `ThreadState::kIdle` 是合理映射吗? 或者应该新增 `ThreadState::kYielded`?
- **Recommendation**: `→ kIdle`。`ThreadState` 是 frozen enum (4 values, per HSK-8 spec §Decision 6);新 enum value = ABI break。Yielded 语义上等价于 Active 但让出 CPU — 映射到 `kIdle` 是保守方案。

**Q2**: `WarpContext::get_warp_state()` 是 public accessor 吗?
- **Recommendation**: 已确认(在 warp_context.h 调用)。实施时验证;若不可见,可能需要在 WarpContext 公共暴露该方法(非 ABI 破坏)。

**Q3**: `blocked_cycles` (int32_t) 在 `threads[i].blocked_cycles_remaining` (uint32_t) 求和时如何处理 overflow?
- **Recommendation**: 累积到 int32_t 边界时 clamp 到 `INT32_MAX`。极端场景(32 lane × 4 billion cycles)概率极低但保留防御逻辑。

## Reference

- **HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` §Phase 3 Task 3.1-3.2
- **HSK-8 spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` (Decision 5: sizeof visibility)
- **HSK-8 audit**: `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` Postmortem
- **Metis REVISE session**: `ses_fc8cd6f96ffeuhJjjA8RB7Y7pY`
- **Stub locations**:
  - `src/ptxemu/device_api_impl.cc:90-94` (warp_exe_once)
  - `:114-118` (get_thread_state)
  - `:156-160` (get_warp_status)
- **Public types preserved**:
  - `include/ptxemu/device_api.h:62-66` (LaneStatus: lane_id + state + pc)
  - `:69-75` (WarpStatus: warp_id + sm_id + lanes[vector] + active_count + blocked_cycles)
- **Internal APIs**:
  - `SMContext::get_warp(uint32_t) → WarpContext*`
  - `WarpContext::execute_warp_instruction()` (existing)
  - `WarpContext::get_thread(int lane_id) → ThreadContext*`
  - `ThreadContext::get_state() const → EXE_STATE` (`include/ptxsim/thread_context.h:205`)
  - `WarpContext::get_warp_state() → WarpState&` (existing)
  - `WarpState::threads[32]` (warp_state.h:14)
  - `WarpState::count_active_lanes()` (warp_state.h:40)
  - `ThreadState::blocked_cycles_remaining` (thread_state.h:40)
- **Public API frozen**: `include/ptxemu/device_api.h:117` `static_assert(PTXEMU_API_VERSION == 1, ...)`
- **drift_check workflow**: `.github/workflows/drift_check.yml` (Invariant 6 modified — exemption list shrunk)
- **Skills referenced**:
  - `ptx-lessons-learned` §1 + §3 + §4 + §21
  - `ptx-barrier-mechanism` (BUG-RETHANG guard)
  - `ptx-instruction-pipeline` (warp_exe_once hot path)
  - `state-modification-audit` (read-only verification)
  - `test-coverage-enforcer` (e2e test)
  - `oracle-prompting` (Decision 5 split rationale)