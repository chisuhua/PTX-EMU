# god-class-refactor-thread-context

**ThreadContext god-class refactor (Phases 1+2)**: Extract `SimtPcManager` (PC + execution state) and `RegisterAccessLayer` (register lookup + bank manager) from the 884-line `ThreadContext`.

## Scope (Implemented: Phases 1+2)

- **Phase 1**: Extract `SimtPcManager` (70 + 161 lines) — 17 PC/state methods, bidirectional `warp_state` sync with `already_blocked` guard
- **Phase 2**: Extract `RegisterAccessLayer` (50 + 56 lines) — `acquire_register()`, `register_bank_manager_` ownership with special register resolution (`tid.x`, `ctaid.x`, etc.)
- **AGENTS.md sync** (Phase 3.3): Updated `src/ptxsim/core/AGENTS.md` `WHERE TO LOOK` + `KEY FILES` tables

## NOT in Scope (Deferred to `god-class-refactor-thread-context-phase3`)

- **Phase 3.1 MemoryAccessor** — Cancelled during implementation: `shared_mem_space`/`local_mem_space` are public members on `ThreadContext` settable from external code; MemoryAccessor copies diverged, causing 9 test regressions. Requires first converting these to non-public or `shared_ptr` references.
- **Phase 3.2 InstructionPipeline** — Fundamentally incompatible: `handler->ExecPipe(this, statement)` requires `this` to be `ThreadContext*`. Extracting to `InstructionPipeline` would break all 40+ handler interfaces. Requires first refactoring handler signature to accept both `ThreadContext*` and `InstructionPipeline*`.
- Phase 3.3 ADR-0017 — Deferred to Phase 3 completion.

## Implementation Plan (Phases 1+2)

5 commits, each independently revert-able:

1. **Commit 1** (`9ce8e93`): Add `SimtPcManager` class (header + impl + CMakeLists.txt)
2. **Commit 2** (`3082e80`): Migrate `ThreadContext` to delegate through `SimtPcManager` (17 methods → inline forwarders)
3. **Commit 3** (`bc1a0aa`): Add `RegisterAccessLayer` class (header + impl + CMakeLists.txt)
4. **Commit 4** (`3fa5e5f`): Migrate `ThreadContext` register access to `RegisterAccessLayer` delegation
5. **Commit 5** (`a2d7ab0`): Sync `AGENTS.md` for new class structure

## Reviews Applied

- **Metis pre-implementation review** (session `ses_0ca442a15ffeTGnLE1vdxlOw45`): 5 MUST-RESOLVE items (MR-1 through MR-5) all resolved before implementation
- **Oracle pre-implementation review** (session `ses_0ca36539effe2432w8IEjv4OQS`): 3 additional gaps (G-1 through G-3) all resolved before implementation

## Verification Results

- **Full build** (`cmake --build build -j$(nproc)`): **100% PASS**
- **`ctest`**: **174/174 PASS** (baseline: 172/174, net +2 improvement)
- **`./scripts/sanity.sh --quick`**: All tests passed
- **`git ls-files openspec/changes/`**: artifacts are tracked (no missing tracked files)

## Implementation Discovered Issues (handled in-flight)

1. **`call.cpp:15`** had bare field access `context->state = EXIT` (138-byte MR-1 risk). Changed to `context->set_state(EXIT)` to match the Phase 1 delegation pattern. Same handler line 26 already used `set_state(EXIT)` correctly, so line 15 was the only inconsistency.

2. **`exec_state_.state` POD consistency** (MR-2 + G-1): `init()` and `reset()` must backfill `exec_state_.state` from `simt_pc_mgr_->get_state()` after migration. `exec_state_` has zero readers anywhere in `src/`/`include/`/`tests/` (verified via grep), so risk is zero, but backfill maintained for future-proofing.

3. **`set_warp_context()` fanout** (MR-4, originally mis-rated as "low risk"): `warp_context.cpp:236` calls `threads[lane_id]->set_warp_context(this)` in production. Phase 1 makes `simt_pc_mgr_->set_warp_context()` fanout mandatory — `ThreadContext::set_warp_context()` now updates both fields atomically.

4. **`SimtPcManager` construction ordering** (MR-5): Must be constructed **after** `warp_id_`/`lane_id_` calculation in `init()` (lines 57-58). Constructing earlier would give `SimtPcManager` an uninitialized `lane_id_` leading to wrong `warp_state.threads[]` access.

5. **Phase 3.1 MemoryAccessor divergence**: The `get_memory_addr()` callback approach (using `std::function<void*(...)>`) worked for compilation but `MemoryAccessor::shared_mem_space_` and `MemoryAccessor::local_mem_space_` diverged from `ThreadContext::shared_mem_space` / `ThreadContext::local_mem_space` because those are public members settable from external code. The resulting 9 test regressions (including SEGFAULT in `e2e_shared_memory_dynamic`) caused Phase 3.1 to be cancelled. Phase 3 of the refactor must first make these non-public or use shared state references.

## References

- OpenSpec artifacts: `proposal.md`, `design.md`, `tasks.md`, `specs/simt-pc-state/spec.md`
- Debt audit: `docs/audits/debt-audit-2026-07-02.md` §2.2 C-1 (P1, 10h → partially addressed by Phase 1+2)
- Roadmap: `docs/roadmap/post-phase3-debt-roadmap.md` §1.2 C-1
- Lessons-learned integration: §1 (cross-module state translation), §3 (multi-Phase commit), §7 (pre-impl Metis review)
- Follow-up change: `god-class-refactor-thread-context-phase3` (memory + control flow extraction)
