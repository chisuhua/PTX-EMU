# device-api-delegation

> **For agentic workers**: REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

## Why

HSK-8 Phase 2 (`PR #14 merged fcdad151` + CppTLM bump `beb3db8`) successfully shipped the public device API surface (`IPtxEmuDevice` 12-method interface + `ptxemu_core` STATIC library + `PTXEMU_API_VERSION=1` freeze per `include/ptxemu/device_api.h:117` static_assert). The `cpp 不暴露` constraint (CppTLM-side 0 PTX-EMU internal header includes, commit `09c27d5`) is honored, but **7 of the 12 pure virtual methods have stub bodies** in `src/ptxemu/device_api_impl.cc`:

**In-scope (this change, 4 methods)**:

| Method | Stub at `device_api_impl.cc` | Real target |
|--------|------------------------------|-------------|
| `set_scoreboard` (sm_id, warp_id, mask) | L92-96 (`return false`) | `SMContext::set_scoreboard(IScoreboard*)` at `include/ptxsim/sm_context.h:87` |
| `set_active_mask` (sm_id, warp_id, mask) | L105-110 (`return false`) | `WarpContext::set_active_mask(uint32_t)` at `include/ptxsim/warp_context.h:199` |
| `set_next_pc` (sm_id, warp_id, lane_id, pc) | L113-118 (`return false`) | `ThreadContext::set_pc(int)` + `commit_pc()` at `include/ptxsim/thread_context.h:227-232` |
| `attach_timing` (sb, pl, tc) | L131-134 (`void{}`) | HSK-4 vendored interfaces injection into SMContext timing hooks |

**Deferred (out-of-scope, 3 methods, follow-up change)**:

| Method | Stub at `device_api_impl.cc` | Why deferred |
|--------|------------------------------|--------------|
| `warp_exe_once` (sm_id, warp_id) | L85-88 (`return -1`) | Requires instance-based `g_gpu_context` migration (Phase 2.2) + warp scheduler integration testing |
| `get_thread_state` (sm_id, warp_id, lane_id) | L99-102 (`return ThreadState::kIdle`) | Requires ThreadContext read accessor + enum mapping validation |
| `get_warp_status` (sm_id, warp_id) | L121-126 (default `WarpStatus s{}; return s;`) | Requires WarpContext lane/active_mask/blocked_cycles snapshotter |

**Already implemented (5 methods, no change needed)**:
`initialize`, `shutdown`, `exe_once`, `sm_exe_once`, `is_finished` (Note: per empirical reading of `src/ptxemu/device_api_impl.cc:128-134`, `is_finished` IS implemented — calls `g_gpu_context->get_state() == EXE_STATE::IDLE` — contrary to the prior Oracle table claim of "5 stubs confirmed").

CppTLM-side `facade` (`beb3db8` state) consumes these methods today; the 4 in-scope stubs no-op, meaning **all delegated state modifications silently fail**; the 3 deferred stubs also silently fail but are out of scope. Until the 4 in-scope are wired, HSK-8 acceptance is incomplete for state-delegation semantics (read-only queries via `exe_once`/`sm_exe_once` work, but state mutations don't).

## What Changes

- **Modify** `src/ptxemu/device_api_impl.cc`: implement 4 delegated methods (`set_scoreboard` / `set_active_mask` / `set_next_pc` / `attach_timing`) by forwarding to existing `SMContext`/`WarpContext`/`ThreadContext` APIs
- **Add** `tests/unit/ptxemu/test_device_api_delegation.cpp`: unit tests covering overwrite semantics + delegation path (including `set_active_mask` overwrite vs OR-merge regression guard)
- **Add** `tests/integration/warp/test_device_api_delegation_e2e.cc`: integration test driven via `WarpContext::execute_warp_instruction` to verify thread PC reflects delegated state changes (per `test-coverage-enforcer`)
- **Modify** `.github/workflows/drift_check.yml`: extend invariant to verify `device_api_impl.cc` contains zero `return false` stubs outside `attach_timing`'s legitimate void no-op

**Constraints (hard, non-negotiable)**:
- ❌ **MUST NOT modify `include/ptxemu/device_api.h`** — public signature frozen by `PTXEMU_API_VERSION==1` static_assert (HSK-8 spec §Decision 3 + drift_check Invariant 1)
- ❌ **MUST NOT introduce new public methods** — any signature change requires HSK-9 handshake
- ❌ **MUST NOT touch `cpp` CppTLM-side** — `cpp 不暴露` constraint per HSK-8 ack `738b412c`

## Impact

| Component | Impact | Specifics |
|-----------|--------|-----------|
| `src/ptxemu/device_api_impl.cc` | Modify (~30 LOC across 4 methods) | Stub bodies replaced with `g_gpu_context->get_sm(sm_id)->...` delegation |
| `tests/unit/ptxemu/test_device_api_delegation.cpp` | New file (~150 LOC) | Catch2 unit tests, direct method invocation |
| `tests/integration/warp/test_device_api_delegation_e2e.cc` | New file (~100 LOC) | Catch2, driven via `execute_warp_instruction` |
| `.github/workflows/drift_check.yml` | Extend Invariant 6 | Verify no `return false` in `device_api_impl.cc` outside `attach_timing` |
| `include/ptxemu/device_api.h` | **NO CHANGE** | Static_assert at L117 + 4 method signatures unchanged |
| `tests/integration/warp/test_set_active_mask_overwrite.cpp` | New file (~80 LOC) | Regression guard for BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES |

- **Test impact**: ctest count grows ~6 new unit + ~1 e2e (~253 tests from current 246 baseline)
- **Build impact**: `ptxemu_core` library gains 4 functional methods (compile-time: ~30 LOC; runtime: zero cost when not called)
- **ABI surface**: unchanged (no new symbols, no changed signatures)
- **CI impact**: drift_check grows from 5 invariants to 6; runtime < 1 second
- **Documentation**: README §已实现功能 "Phase 2.2/2.3 delegation 完成" 字样 (per §21 Checklist I)

## Capabilities

### New Capabilities

- `ptxemu-device-api-delegation` — Phase 2.2 set_scoreboard + set_active_mask + set_next_pc delegation to SM/Warp/Thread contexts
- `delegation-thread-pc-invariants` — Thread PC update invariants under delegated set_next_pc (regression guards for BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES patterns)
- `ci-drift-check-extension` — drift_check Invariant 6 verifying zero `return false` stubs in `device_api_impl.cc` after Phase 2.2/2.3

### Modified Capabilities

- (none — pure delegation, no behavior change to existing specs)

## Design-Time Checklist (Lessons-Learned)

### 函数迁移完整性 (N/A)
- These are not function migrations; they are stub implementations that delegate to existing APIs.
- No `set_*`/`commit_*`/`force_*` cross-module state translation needed (delegation is direct method call)

### 状态修改 (CRITICAL — applies)
- `set_active_mask` MUST use overwrite semantics (NOT OR-merge) — see `device_api_impl.cc:105-109` comment + ptx-lessons-learned §1
- `set_next_pc` MUST use `ThreadContext::set_next_pc` (NOT `force_set_pc`) — see ptx-lessons-learned ANTI-PATTERNS
- `set_scoreboard` delegates to existing `SMContext::set_scoreboard(IScoreboard*)` — no state translation needed

### 多 Phase 推进 (CRITICAL — applies)
- Per ptx-lessons-learned §3: Phase 2.2 (3 methods) and Phase 2.3 (1 method) MUST be independent commits, each independently revertable
- Per ptx-lessons-learned §4: baseline worktree required before Phase 2.2 implementation
- Per ptx-lessons-learned Checklist B: failure → immediate revert that Phase, no mixing into next commit

### 文档同步 (Checklist I)
- README §已实现功能 IPtxEmuDevice bullet MUST update with "Phase 2.2/2.3 delegation 完成" before archive (R3 prevention per Doc1 §Risk)
- AGENTS.md ANTI-PATTERNS list MUST NOT add `set_active_mask OR-merge` (already there at L87)

## Reference

- **HSK-8 follow-up plan**: `2026-08-24-hsk8-followup-task-path.md` (parent plan, this change is Doc1 Phase 2 implementation)
- **HSK-8 spec**: `docs/superpowers/specs/2026-08-22-hsk-8-ptxemu-public-api-ack.md` (commit `738b412c`)
- **HSK-8 audit**: `docs/audits/2026-08-13-hsk8-ptxemu-public-api.md` (postmortem at commit `8aa72f1d`)
- **HSK-8 archived change**: `openspec/changes/archive/2026-08-24-ptxemu-public-device-api/`
- **Oracle consultation**: `ses_fcbc066a6ffeXfv5kIgmE8UwrB` (parent plan Oracle; re-consult recommended for delegation order validation)
- **Public API frozen**: `include/ptxemu/device_api.h:117` `static_assert(PTXEMU_API_VERSION == 1, ...)`
- **Stub location**: `src/ptxemu/device_api_impl.cc:91-137` (4 method stubs)
- **Target APIs**:
  - `SMContext::set_scoreboard(IScoreboard*)` at `include/ptxsim/sm_context.h:87`
  - `WarpContext::set_active_mask(uint32_t)` at `include/ptxsim/warp_context.h:199` (overwrite, NOT OR-merge)
  - `ThreadContext::set_next_pc(int)` at `include/ptxsim/thread_context.h:229` (NOT `force_set_pc`)
- **Skills referenced**:
  - `ptx-lessons-learned` §1 (跨模块状态翻译) + §3 (分 Phase commit) + §4 (baseline worktree) + §7 (Metis pre-impl) + §21 (README Checklist I)
  - `ptx-barrier-mechanism` (set_active_mask overwrite semantics for BUG-RETHANG/BUG-POSTBARRIER-TWOHALVES)
  - `test-coverage-enforcer` (Phase 2.2/2.3 validation: unit + e2e via `execute_warp_instruction`)
- **drift_check workflow**: `.github/workflows/drift_check.yml` (5 invariants → 6 after this change)