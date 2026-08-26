## Context

**Current state**: PTX-EMU side has 3 unit tests in `tests/unit/ptxemu/test_device_api_attach_timing.cpp` that verify `IPtxEmuDevice::attach_timing` round-trip identity via the namespace bridge (`static_cast<void*>` round-trip per HSK-8 spec Decision 6). These tests stop at the `sm->get_scoreboard() == mock` assertion — they verify the injection happened, but **do not verify the injected module is queried during `SMContext::exe_once()` execution**.

CppTLM side added 7 integration tests in `test/test_cuda_core_adapter_timing.cc` (commit `d909407` 2026-08-25) that verify the consumer-side flow (`facade.attach_timing` → `ScoreboardTLM.allocate/release` etc.). These tests run against CppTLM's own mocks (`ScoreboardTLM` / `PipelineTLM` / `TensorCoreTLM`) and assert behavior at the CppTLM module boundary, not the `SMContext` boundary.

**Gap**: Neither side verifies that PTX-EMU's `SMContext::exe_once()` step_a / step_b / step_c paths actually invoke the injected modules. A regression in `src/ptxsim/core/sm_context.cpp:273-309` (step_a_scoreboard_check, step_b_set_blocked_cycles, step_c_release_scoreboard) would not be caught by either side's existing tests.

**Existing tooling** (just discovered — pivot target):
- `include/ptxemu/testing/warp_executor_test_fixture.h:42-100` — `WarpExecutorTestFixture` already handles:
  - `g_gpu_context` creation + `init()` call (line 50) — solves BLOCKER 2
  - `sm_->add_block(CTAContext)` which internally does `warps.push_back + warp_scheduler->add_warp` (per `sm_block_dispatch.cpp:91-92`) — solves BLOCKER 1
  - Provides `dev()/sm()/warp()/gpu()` accessors
  - RAII cleanup
  - Already used by 3 existing tests: `tests/integration/{simt,warp}/test_*.cpp`
- Header comment line 15-17 documents the design intent: "delegation tests must drive actual warp setup, NOT just create empty GPUContext (which has no warps and triggers early-return guards)" — exactly the lessons-learned §1 principle this change embodies.

**Constraints**:
- `IPtxEmuDevice::attach_timing` is the only public entry — tests must go through it (per HSK-8 spec Decision 6)
- 251/251 ctest currently PASS; target 252/252 PASS after change
- `PTXEMU_API_VERSION=1` frozen; no public signature change allowed
- drift_check 7 invariants must remain green (Invariant 7 added by `antlr4-path-hardcoding-fix` commit `2148e15c`)

**Stakeholders**:
- PTX-EMU maintainer (this repo)
- CppTLM maintainer (consumes via `external/PTX-EMU` submodule at `0e0ba7ad`)
- HSK-8 author (drift_check CI gate)

## Goals / Non-Goals

**Goals**:
1. Add 4 PTX-EMU-side integration tests verifying `IPtxEmuDevice::attach_timing` → downstream query path participation
2. Reuse existing `WarpExecutorTestFixture` (parameterized for G4's S_FMA statements) instead of re-inventing friend access helpers
3. Maintain drift_check 7 invariants, PTXEMU_API_VERSION=1, public ABI frozen
4. Establish precedent for future integration tests in `tests/integration/cpptlm/`

**Non-Goals**:
- Modifying production code (`SMContext`, `device_api_impl.cc`, `exe_once` paths)
- Adding new `IPtxEmuDevice` methods (would trigger HSK-9)
- Adding new friend namespace (rejected per Oracle Path A)
- Replacing existing `tests/unit/cpptlm/test_smcontext_injection.cpp` or `tests/integration/cpptlm/test_mock_injection_*_path.cpp` (those use direct `sm->set_*` not `IPtxEmuDevice::attach_timing`; orthogonal coverage)

## Decisions

### Decision 1: Reuse `WarpExecutorTestFixture` (Path A pivot)

**Chosen**: Use existing `WarpExecutorTestFixture` (`include/ptxemu/testing/warp_executor_test_fixture.h:42-100`) with a backward-compatible optional `std::vector<StatementContext> statements = {}` parameter for G4's S_FMA injection. Constructor signature change from `WarpExecutorTestFixture()` to `WarpExecutorTestFixture(std::vector<StatementContext> statements = {})`; the 3 existing fixture-using tests compile unchanged because of the default value.

**Rationale**:
- Fixture already solves BLOCKER 1 (warp scheduler dual registration via `add_block`) and BLOCKER 2 (init() call)
- Matches existing pattern (3 fixture-using tests in `tests/integration/{simt,warp}/`)
- Eliminates the `sm_test_access` friend namespace entirely (which was originally proposed in Path B)
- Per `ptx-lessons-learned` §1: test helpers that bypass production registration paths create translation drift; reusing `add_block` avoids this

**Alternatives considered**:
- ❌ **Path B**: `sm_test_access::add_warp` + `set_state` friend namespace — rejected after Metis review found 4 BLOCKERs (warp_scheduler dual registration, missing init(), wrong CMakeLists path, empty operands). Per Oracle consultation, this approach was "re-invention" that violated `ptx-lessons-learned` §1.
- ❌ Modify `GpuContextScope` (used in `test_device_api_attach_timing.cpp`) to call `init()` — would change behavior of existing tests that "pass" via `WARN + early return` (`test_device_api_attach_timing.cpp:102-107`). Blast radius on existing unit tests is uncontrolled.
- ❌ Construct second `CTAContext` block with FFMA statements inside test (don't modify fixture) — works but inconsistent with 3 existing tests' pattern of "one fixture per test".

### Decision 2: 1 file, 4 TEST_CASE (not 4 files)

**Chosen**: Single `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp` containing 4 `TEST_CASE` cases tagged `[integration][cpptlm][attach_timing][g1-g4]`.

**Rationale**:
- All 4 tests share the same `WarpExecutorTestFixture` setup, namespace bridge helper, and mock infrastructure (DRY)
- `tests/AGENTS.md` Type 2 (integration) convention allows multi-case files
- ctest registration is single `add_catch_test` entry — simpler than 4 separate registrations
- Matches existing `test_mock_injection_slow_path.cpp` pattern (3 TEST_CASE in 1 file)

**Alternatives considered**:
- ❌ 4 separate files (`test_attach_timing_g1.cpp` ... `g4.cpp`) — duplicated helpers, 4 ctest entries, more CMake boilerplate
- ❌ Merging with existing `test_mock_injection_*_path.cpp` — violates single-responsibility; existing tests use direct `sm->set_*` not public API

### Decision 3: G4 e2e uses S_FMA via `make_ffma` (not `make_stmt(S_FMA)`)

**Chosen**: G4 uses `StatementType::S_FMA` constructed via `instruction_helpers.h:460 make_ffma(dst, src1, src2, src3)` — sets up `GenericInstr.operands[0] = RegOperand{dst}` correctly so `RegisterAnalyzer::get_dest_registers_as_ids` returns non-empty dest register list (avoids BLOCKER 4).

**Rationale**:
- S_FMA is non-TC (`is_tensor_core_instruction` returns false → `step_b` calls `pipeline.get_fractional_cycles_by_type` with `PipelineId::P0_INT_FP32`, not TC path)
- S_FMA is non-LD/ST/ATOM (`map_instruction_to_pipeline` returns `P0_INT_FP32` for default branch)
- S_FMA is non-SFU (per `sm_context.cpp:166-179` SFU switch)
- Result: G4 can verify both `scoreboard.allocate/release > 0` (step_a/c) AND `pipeline.get_fractional_cycles_by_type > 0` (step_b) in single e2e run, but NOT `tc.get_latency > 0` (correctly — FMA is not a TC instruction)
- `make_ffma` properly fills `GenericInstr.operands[]` with RegOperand — solves BLOCKER 4 (`make_stmt(S_FMA)` only sets `s.type`, leaving `s.data` empty)

**Alternatives considered**:
- ❌ S_ADD — too simple (no FMA path coverage)
- ❌ S_TCGEN05_MMA — would force G4 into TC path; loses ability to verify pipeline path
- ❌ S_LD/ST — exercises LD/ST-specific paths (memory.cpp:47,71,139), distracts from timing injection focus

### Decision 4: Fixture parameterization is backward-compatible

**Chosen**: Add `std::vector<StatementContext> statements = {}` parameter to `WarpExecutorTestFixture` constructor. Default empty vector preserves current behavior — 3 existing fixture-using tests compile and run without source changes.

**Rationale**:
- Source-compatible change (default parameter)
- 3 existing tests unaffected (CI immediately verifies after change)
- G4 passes `{make_ffma("...", ...)}` to enable statement scheduling for `exe_once()`
- G1/G2/G3 use default `{}` (don't need statements — they don't drive `exe_once()` execution paths)

**Alternatives considered**:
- ❌ New `WarpExecutorTestFixture2` class with statements parameter — code duplication, 6 fixture-using tests need migration
- ❌ Two separate fixtures (one for statements, one for empty) — same issue
- ❌ Helper function to inject statements post-construction — modifies `WarpContext::statements` from outside, brittle

### Decision 5: Mock counting strategy — direct member counters

**Chosen**: Each mock has `mutable int call_count = 0;` member, increments inside each pure virtual method.

**Rationale**:
- Simplest possible verification (no state machines, no signal handlers)
- `mutable` allows `const` method overrides to increment counters (`get_fractional_cycles` is `const`)
- Test assertion: `REQUIRE(mock.call_count > 0)` directly proves the path was queried
- Matches existing `MockSlowScoreboard` pattern in `test_mock_injection_slow_path.cpp:18-25`

**Alternatives considered**:
- ❌ Side-effect verification (e.g., verify `warp->is_blocked` changed after `step_b`) — coupled to `warp->set_blocked_cycles_for_active` internals; brittle
- ❌ Callback functions — over-engineering for test mocks

## Risks / Trade-offs

| # | Risk | Mitigation |
|---|------|-----------|
| R1 | Fixture parameter default value breaks some compile configuration | Default `{}` is source-compatible — verify by building all 3 existing fixture-using tests after change |
| R2 | G1/G4 e2e brittleness — `sm.exe_once()` may evolve | Test mocks are minimal; assertions only check call_counts, not specific execution order; if exe_once changes call sites, tests still pass as long as injection works |
| R3 | drift_check regression — fixture header change might trigger new invariant | Phase 1 commits fixture header change in isolation; if drift_check fails, revert Phase 1 commit only (per `ptx-lessons-learned` §3) |
| R4 | CppTLM PTX-EMU submodule bump needed after merge | Out of scope for THIS change; CppTLM maintainer bumps submodule per their `plans/ptxemu-followup-roadmap.md` |
| R5 | S_FMA may not trigger step_b in future refactor | Spec contract states pipeline is queried for FMA (per `sm_context.cpp:166`); if refactored, test will fail and force spec update |
| R6 | G3 needs `stmt.data = GenericInstr{Q_F16}` to verify precision | Documented in test comments; if precision mapping refactored, G3 still verifies `get_latency_calls > 0` regardless of precision arg |

## Migration Plan

**2 phases, each independently revertible** (per `ptx-lessons-learned` §3):

### Phase 1: Fixture parameterization (no tests yet)
- `include/ptxemu/testing/warp_executor_test_fixture.h`: add optional `std::vector<StatementContext> statements = {}` parameter to constructor (line 44-75)
- Pass `statements` to `block->init(...)` instead of empty `std::vector<StatementContext> statements` (line 57)
- Verify: 251/251 ctest PASS unchanged (3 fixture-using tests compile + run as before) + drift_check 7 invariants PASS

### Phase 2: 4 integration tests
- `tests/integration/cpptlm/test_attach_timing_consumer_e2e.cpp`: NEW, 4 TEST_CASE
- `tests/integration/CMakeLists.txt`: `add_catch_test(integration_attach_timing_consumer_e2e ...)`
- Verify: 252/252 ctest PASS (251 baseline + 1 new target with 4 TEST_CASE) + drift_check PASS

### Phase 3: docs sync + archive
- Update `openspec/changes/attach-timing-consumer-e2e-tests/tasks.md` checkbox progress
- Verify no AGENTS.md / ADR change needed (no architectural shift)
- Archive via `openspec archive attach-timing-consumer-e2e-tests`

**Rollback strategy**: Each phase is single commit; `git revert HEAD` per phase if regression detected.

**Baseline worktree** (per `ptx-lessons-learned` §4):
```bash
# Pre-Phase-1
git worktree add .worktrees/attach-timing-baseline main
cd .worktrees/attach-timing-baseline && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DPTXEMU_BUILD_TESTING=ON && cmake --build build -j$(nproc)
cd build && ctest --output-on-failure  # 251/251 baseline
```

## Open Questions

**Q1**: Should `WarpExecutorTestFixture` use a different parameter name to avoid shadowing the local `statements` variable?
- **Current decision**: `std::vector<StatementContext> statements` (matches local variable in current fixture).
- **Owner input needed if alternative naming desired**.

**Q2**: G3 test verification scope — should we assert specific `TcPrecision` argument (e.g., `REQUIRE(tc.last_precision == TcPrecision::FP16)`) or just call_count > 0?
- **Current decision**: call_count > 0 only (looser coupling); precision mapping tested separately in `test_mock_injection_slow_path.cpp:99-101` (`is_tensor_core_instruction` assertion).
- **Owner input needed if stricter assertion desired**.