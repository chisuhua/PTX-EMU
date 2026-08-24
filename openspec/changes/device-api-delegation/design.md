# device-api-delegation — Design

## Context

HSK-8 Phase 2 shipped `IPtxEmuDevice` (12 pure virtual methods) as the cross-repo ABI surface between PTX-EMU and CppTLM. Of these 12 methods, 4 remain **stubbed** in `src/ptxemu/device_api_impl.cc:91-137`:

| Method | Lines | Current behavior |
|--------|-------|------------------|
| `bool set_scoreboard(uint32_t sm_id, uint32_t warp_id, uint64_t mask)` | 92-96 | `return false` |
| `bool set_active_mask(uint32_t sm_id, uint32_t warp_id, uint64_t mask)` | 105-110 | `return false` |
| `bool set_next_pc(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id, uint32_t pc)` | 113-118 | `return false` |
| `void attach_timing(IScoreboard*, IPipelineLatencyProvider*, ITensorCoreTiming*)` | 131-134 | empty `void{}` |

CppTLM-side `facade` (consumed via `beb3db8` submodule pin) calls these methods expecting real delegation. Today every call silently no-ops, meaning **CppTLM cannot actually modify PTX-EMU state through the public API**. This contradicts HSK-8 spec §"cpp 不暴露" constraint's intent — the API exists but doesn't work.

**Constraints**:
- `include/ptxemu/device_api.h` is FROZEN by `static_assert(PTXEMU_API_VERSION == 1)` at L117
- drift_check Invariant 1 hard-fails any change to `PTXEMU_API_VERSION`
- Any public signature change requires HSK-9 handshake (4-week protocol)

**Opportunity**: All 4 target delegation methods EXIST in PTX-EMU internal APIs (`SMContext`/`WarpContext`/`ThreadContext`). The work is purely to wire stubs to real implementations — no new APIs, no behavior changes, no ABI impact.

## Goals / Non-Goals

### Goals

1. **Implement `set_scoreboard`**: delegate to `SMContext::set_scoreboard(IScoreboard*)` (`include/ptxsim/sm_context.h:87`)
2. **Implement `set_active_mask`**: delegate to `WarpContext::set_active_mask(uint32_t)` (`include/ptxsim/warp_context.h:199`) with **overwrite semantics** (NOT OR-merge)
3. **Implement `set_next_pc`**: delegate to `ThreadContext::set_next_pc(int)` (`include/ptxsim/thread_context.h:229`) via `set_pc()` + `commit_pc()` pattern (NOT `force_set_pc`)
4. **Implement `attach_timing`**: store HSK-4 vendored 3 interfaces and inject into SMContext timing hooks
5. **Add regression tests**: unit + e2e coverage of all 4 methods
6. **Add drift_check Invariant 6**: verify zero `return false` stubs outside `attach_timing`'s legitimate void no-op

### Non-Goals

1. ❌ Modify `include/ptxemu/device_api.h` (HSK-9 territory)
2. ❌ Add new methods to `IPtxEmuDevice` interface
3. ❌ Bump `PTXEMU_API_VERSION`
4. ❌ Touch CppTLM-side code (cpp 不暴露 constraint)
5. ❌ Implement other 8 `IPtxEmuDevice` methods (out of scope; those are not stubs)
6. ❌ Refactor `SMContext`/`WarpContext`/`ThreadContext` (delegation-only, no underlying API changes)
7. ❌ Phase 1.5 namespace migration (per Doc1 §Phase 3 Task 3.2, `[~] DEFERRED`)

## Decisions

### Decision 1: Delegation order — `set_scoreboard` → `set_active_mask` → `set_next_pc` (Phase 2.2), `attach_timing` (Phase 2.3)

**Rationale** (per Doc1 §Phase 2 Task 2.1 Step 3 Decision 1 + ptx-lessons-learned §1):
- **Memory state first** (`set_scoreboard`) — establishes the IScoreboard reference for subsequent dependency chains
- **Mask state second** (`set_active_mask`) — lane activation precedes thread-level operations
- **PC last** (`set_next_pc`) — thread PC is the leaf state, all dependencies must be set first
- **attach_timing separated as Phase 2.3** — distinct concern (vendored interface injection), independent commit per ptx-lessons-learned §3

**Implementation sketch (Phase 2.2)**:
```cpp
// device_api_impl.cc
bool set_scoreboard(uint32_t sm_id, uint32_t warp_id, uint64_t mask) override {
    SMContext* sm = g_gpu_context->get_sm(sm_id);
    if (!sm) return false;
    // Phase 2.2 scope: SMContext::set_scoreboard(IScoreboard*) — accepts IScoreboard
    // We do not have a real IScoreboard object to pass; return false if sm doesn't own one
    IScoreboard* sb = sm->get_scoreboard();  // NEW accessor (Phase 2.2 scope decision)
    if (!sb) return false;
    // Apply mask: depends on IScoreboard interface contract
    // For Phase 2.2 minimum: just verify wiring works, mask propagation TBD
    return sb->apply_warp_mask(warp_id, mask);  // assumed IScoreboard method
}
```

> **NOTE**: The exact IScoreboard interface for mask application needs verification during implementation. Doc1 §Phase 2 Task 2.1 Oracle verification table confirmed `SMContext::set_scoreboard(IScoreboard*)` exists at `sm_context.h:87` but didn't specify the IScoreboard methods.

### Decision 2: `set_active_mask` overwrite semantics (NOT OR-merge)

**Rationale** (per `device_api_impl.cc:105-109` existing comment + ptx-barrier-mechanism skill):
- `set_active_mask` MUST overwrite the existing mask, NOT OR-merge
- OR-merge logic is encapsulated in `BarrierModule::release_warp_barrier` (per ptx-barrier-mechanism skill + ptx-lessons-learned §1 BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES)
- The `ret` handler relies on overwrite (per `src/ptxsim/instructions/AGENTS.md` if exists, else known SIMT semantics)

**Implementation sketch**:
```cpp
bool set_active_mask(uint32_t sm_id, uint32_t warp_id, uint64_t mask) override {
    SMContext* sm = g_gpu_context->get_sm(sm_id);
    if (!sm) return false;
    WarpContext* warp = sm->get_warp(warp_id);
    if (!warp) return false;
    // ptx-lessons-learned §1 + BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES:
    // set_active_mask is OVERWRITE (not OR-merge). OR logic lives in
    // BarrierModule::release_warp_barrier. The ret handler depends on
    // overwrite semantics to clear retired lanes correctly.
    warp->set_active_mask(static_cast<uint32_t>(mask));  // overwrite
    return true;
}
```

**Regression guard test** (per test-coverage-enforcer):
```cpp
// tests/integration/warp/test_set_active_mask_overwrite.cpp
// Verify: given warp.active_mask = 0xFF, calling set_active_mask(0x01)
// results in active_mask == 0x01 (overwrite), NOT 0xFF (no-op) or 0xFFFFFFFF
// (OR-merge). This guards against reintroduction of the BUG-RETHANG bug.
```

### Decision 3: `set_next_pc` uses `set_pc()` + `commit_pc()` (NOT `force_set_pc`)

**Rationale** (per AGENTS.md ANTI-PATTERNS line 85: "❌ `force_set_pc()` — 用 `set_pc()` + `commit_pc()`"):
- `force_set_pc` bypasses PC synchronization invariants
- `set_pc` + `commit_pc` is the canonical PC update path used by all internal handlers
- This avoids subtle bugs in branch reconvergence

**Implementation sketch**:
```cpp
bool set_next_pc(uint32_t sm_id, uint32_t warp_id, uint32_t lane_id, uint32_t pc) override {
    SMContext* sm = g_gpu_context->get_sm(sm_id);
    if (!sm) return false;
    WarpContext* warp = sm->get_warp(warp_id);
    if (!warp) return false;
    ThreadContext* thread = warp->get_thread(lane_id);
    if (!thread) return false;
    // AGENTS.md ANTI-PATTERNS L85: NEVER force_set_pc(). Use set_pc + commit_pc.
    thread->set_pc(static_cast<int>(pc));
    thread->commit_pc();
    return true;
}
```

### Decision 4: Phase commit discipline (Phase 2.2 + Phase 2.3 independent)

**Rationale** (per ptx-lessons-learned §3):
- Phase 2.2 (3 methods) and Phase 2.3 (1 method) MUST be separate commits
- Each commit independently revertable — failure in one doesn't poison the other
- Allows incremental CppTLM bump after each Phase if needed
- Matches Phase 5/6 split precedent in HSK-8 (`3678a0d7` / `d5600e89`)

**Commit structure**:
```
Phase 2.2: "feat(ptxemu): phase 2.2 set_scoreboard + set_active_mask + set_next_pc delegation"
Phase 2.3: "feat(ptxemu): phase 2.3 attach_timing HSK-4 vendored interface injection"
```

**CI gate between phases**:
- Phase 2.2 commit triggers CI: ctest 246/246 baseline + 6 new unit + 0 e2e = 252/252 expected
- Phase 2.3 commit triggers CI: ctest 252/252 baseline + 1 new e2e = 253/253 expected
- If Phase 2.3 fails → `git revert <phase-2.3-sha>` cleanly removes it; Phase 2.2 remains

### Decision 5: README §已实现功能 sync discipline (R3 prevention)

**Rationale** (per Doc1 §Risk R3 + ptx-lessons-learned §21 Checklist I):
- README drift caused by repeated "forgot to update README" pattern (Doc1 §21 违规史)
- 4-item discipline: 代码 + 单元测试 + e2e + README 同步
- Archive-time gate: `grep "Phase 2.2/2.3 delegation 完成"` MUST match before archive

**Implementation**:
- Add `tasks.md` Section 4.1 as explicit README sync task
- Add `ac-verifier` check at archive-time that README contains "Phase 2.2/2.3 delegation 完成"

## Risks / Trade-offs

| # | Risk | Likelihood | Impact | Mitigation |
|---|------|-----------|--------|------------|
| R1 | `device_api.h` accidental modification triggers CppTLM cross-repo breakage | Low | Critical | `git diff --name-only` pre-commit guard; drift_check Invariant 1 already enforces PTXEMU_API_VERSION==1 |
| R2 | `set_active_mask` reintroduction of OR-merge (BUG-RETHANG regression) | Medium | High | `tests/integration/warp/test_set_active_mask_overwrite.cpp` regression guard + test-coverage-enforcer 强制 |
| R3 | README drift reoccurrence (Doc1 §21 违规) | High | Medium | tasks.md §4.1 explicit checkbox + ac-verifier archive-time check |
| R4 | `attach_timing` injection breaks SMContext timing hooks (HSK-4 interfaces misaligned) | Low | High | Phase 2.3 separate commit allows revert; HSK-4 vendored interface contract validated by Phase 2.3 unit test |
| R5 | `g_gpu_context` race condition (set_next_pc concurrent with execute_warp_instruction) | Medium | Medium | Existing SMContext internal locking applies; if needed, add explicit `lock_guard` in delegation methods (mirrors ptx-lessons-learned §2 recursive lock pattern) |
| R6 | IScoreboard::apply_warp_mask method signature unverified (Decision 1 NOTE) | Medium | Medium | Phase 2.2 implementation verifies against IScoreboard header; if signature differs, fall back to a no-op return false with TODO for Phase 2.2.1 follow-up |

## Migration Plan

This change is NOT a function migration. Per ptx-lessons-learned §1 (跨模块状态翻译), no cross-module state translation needed because:

- `set_scoreboard` IScoreboard ownership: already managed by SMContext (no state translation)
- `set_active_mask` overwrite: state is `WarpContext::active_mask_` (direct assignment)
- `set_next_pc` thread PC: state is `ThreadContext::simt_pc_mgr_->next_pc_` (direct via set_pc+commit_pc)
- `attach_timing` vendored interfaces: state is `SMContext::timing_*_` (member fields, direct store)

**Migration discipline** (per ptx-lessons-learned §3 + Checklist B):

```
□ baseline worktree .worktrees/device-api-delegation-baseline (per §4 15-20 min)
□ baseline ctest 246/246 PASS confirmed
□ Phase 2.2 implementation in feat/device-api-delegation branch
□ Phase 2.2 commit (independent revertable)
□ Phase 2.2 CI green (ctest 252/252 + drift_check 5 invariants)
□ Phase 2.3 implementation in same branch
□ Phase 2.3 commit (independent revertable)
□ Phase 2.3 CI green (ctest 253/253 + drift_check 6 invariants)
□ README sync commit (per Decision 5)
□ PR #15 to main + merge squash
□ Notify CppTLM owner (issue #22 comment)
□ Archive OpenSpec change (NOT until CppTLM bumps per HSK-8 ack §Decision 1)
```

## Open Questions

None blocking implementation. Q1/Q2 from Doc1 §Open Questions (consumer_smoke attribution, CppTLM AGENTS.md sync) tracked separately in Doc1 Phase 3.

## Verification

1. **device_api.h unchanged**: `git diff origin/main..feat/device-api-delegation -- include/ptxemu/device_api.h` MUST be empty
2. **drift_check 6 invariants**: GitHub Actions UI or `act -j drift-check`
   - Invariant 1: PTXEMU_API_VERSION == 1 (existing)
   - Invariant 2: IPtxEmuDevice ≥ 12 pure virtual methods (existing)
   - Invariant 3: C++17 compatible (existing)
   - Invariant 4: 4 symbols present (existing)
   - Invariant 5: ptxemu_core STATIC target name (existing)
   - **Invariant 6 (NEW)**: zero `return false` stubs in `device_api_impl.cc` outside `attach_timing`'s legitimate void no-op
3. **ctest PASS**: `cd build && ctest --output-on-failure` expected 253/253 (246 baseline + 6 new unit + 1 e2e)
4. **set_active_mask overwrite test**: `tests/integration/warp/test_set_active_mask_overwrite.cpp` validates overwrite semantics
5. **e2e delegation test**: `tests/integration/warp/test_device_api_delegation_e2e.cc` validates thread PC updates via `execute_warp_instruction`