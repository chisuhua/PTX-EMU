# Phase 3 Structural Debt — T2-4 Scope Document

> **Purpose**: Detail T2-4 Steps 2-7 scope for Phase 4 planning.
> **Status**: Step 1 complete; Steps 2-7 deferred to Phase 4 (multi-day ANTLR regeneration).
> **Ref**: T2-4 Step 1 commit `2e339ea` ("refactor(kernel-context): remove unused usesAsyncStore/usesRedAsync")

## Context

PTX 8.7+ instructions were added as **placeholders** during initial grammar bootstrapping (see `include/ptx_ir/ptx_op.def` §13-15). They were parsed by ANTLR grammar but never had handler implementations. T2-4 removes these placeholders systematically.

## Step 1 ✅ (commit `2e339ea`)

Remove `KernelContext::usesAsyncStore()` and `KernelContext::usesRedAsync()` — zero callers, removed from `include/ptx_ir/kernel_context.h`.

## Steps 2-7 ⏸ Phase 4 (multi-day)

### Step 2: Remove X-macro entries from `include/ptx_ir/ptx_op.def`

Lines 166-167, 172-179 (10 entries):
- Section 13 (async mem): `S_ST_ASYNC`, `S_RED_ASYNC`
- Section 14 (tcgen): `S_TCGEN_ALLOC`, `S_TCGEN_DEALLOC`, `S_TCGEN_RELINQUISH`, `S_TCGEN_CP`, `S_TCGEN_SHIFT`, `S_TCGEN_MMA`, `S_TCGEN_COMMIT`
- Section 15 (tensor map): `S_TENSORMAP_REPLACE`

### Step 3: Remove `statement_factory.h` dispatch entries

Lines 380, 388 — `make_stAsyncInstr`, `make_redAsyncInstr` factory functions.

### Step 4: Update grammar (`src/grammar/ptxInstructions.g4`)

Lines 461-499:
- Remove `stAsyncInst` rule (line 461)
- Remove `redAsyncInst` rule (line 462)
- Remove `tcgenAllocInst`, `tcgenDeallocInst`, `tcgenRelinquishInst`, `tcgenCpInst`, `tcgenShiftInst`, `tcgenMmaInst`, `tcgenCommitInst` (lines 463-468)
- Remove `tcgenMma` from `incomplete_inst` fallback (line 481)
- Remove `stAsyncQualifiers`, `redAsyncQualifiers` rules (lines 487-499)
- Update `incomplete_inst` to remove these (lines 474-481)

### Step 5: Update lexer (`src/grammar/ptxLexer.g4`)

Lines 387-388:
- Remove `ST_ASYNC : 'st.async';`
- Remove `RED_ASYNC : 'red.async';`
- Remove corresponding TCGEN_* and TENSORMAP_REPLACE lexer rules

### Step 6: Regenerate ANTLR parser

```bash
cmake --build build --target GenerateParser
```

**Estimated time**: 4-8 hours (depends on Java + ANTLR 4.11.1 runtime version + first-time generation friction). Multiple validation iterations needed.

### Step 7: Update tests + verify

- No existing tests reference PTX 8.7+ placeholders (verified 2026-06-24 via `grep -rn "S_ST_ASYNC\|S_RED_ASYNC\|S_TCGEN\|S_TENSORMAP_REPLACE" tests/` returned empty)
- Add negative tests confirming removed tokens fail parsing
- `./scripts/sanity.sh` full regression
- `./tests/ptx/test_all_ptx.sh` to confirm no regression on existing PTX
- `./scripts/sanity.sh --ptx` (lightweight PTX-only sanity)

## Verification Snapshot (2026-06-24)

| Artifact | Placeholder references | Status |
|----------|----------------------|--------|
| `tests/` | 0 matches for S_ST_ASYNC/S_RED_ASYNC/S_TCGEN_*/S_TENSORMAP_REPLACE | ✅ safe to remove |
| `src/ptxsim/instructions/*.cpp` | 0 `process_stAsync\|process_redAsync\|process_tcgen` handlers | ✅ no handler impact |
| `src/ptxsim/InstructionFactory` | Handlers not registered (no `process_*` funcs exist) | ✅ no dispatch impact |
| `src/cudart/` | No consumer of these instruction types | ✅ no runtime impact |
| `include/ptx_ir/statement_context.h:223,231` | `AsyncStoreInstr`, `AsyncReduceInstr` struct definitions | ⚠️ ORPHAN — become unused if X-macro entries removed |
| `src/ptxir/ptxir_writer.cpp:371,385` | `if constexpr (std::is_same_v<T, AsyncStoreInstr/AsyncReduceInstr>)` template specializations | ⚠️ ORPHAN — become dead code if X-macro entries removed |

**Orphan cleanup scope (additional Step 8)**:
- Remove `struct AsyncStoreInstr { ... }` and `struct AsyncReduceInstr { ... }` from `include/ptx_ir/statement_context.h` (lines 223-237 approx)
- Remove template specializations in `src/ptxir/ptxir_writer.cpp` lines 371 and 385
- Note: TCGEN and TENSORMAP structs may exist similarly — need full audit before Step 2

**⚠️ S_ST_BULK is OUT OF SCOPE**:
- `X(S_ST_BULK, stBulk, StBulk, 3, GENERIC_INSTR, tcgen)` at line 188 of ptx_op.def is NOT a PTX 8.7+ placeholder
- Uses `GENERIC_INSTR` (not `ASYNC_STORE`/`TCGEN_INSTR`)
- Has `stBulkInst` rule in ptxInstructions.g4:471 + `ST_BULK` token in ptxLexer.g4:397
- Has its own qualifier rule `stBulkQualifiers`
- KEEP all S_ST_BULK related entries (not part of T2-4 scope)

## Risk Assessment

**Low overall risk** because:
- Zero test coverage means no test regression
- Zero handlers means no dispatch regression
- Zero consumers means no runtime regression
- Only risk is **parser regeneration** introducing subtle grammar conflicts

**⚠️ CRITICAL: Steps 2-7 are ATOMIC — cannot be done incrementally**

**Experimental validation (2026-06-24)**:
Tried removing only `X(S_ST_ASYNC, ...)` from `include/ptx_ir/ptx_op.def` to test if Steps 2-3 could be done in isolation. **Build immediately broke** with:
```
src/CMakeFiles/ptxsim.dir/utils/ptx_lane_verification.cpp.o: Error 1
src/CMakeFiles/ptxsim.dir/all: Error 2
```

**Root cause**: X-macro is consumed by `include/ptx_parser/ptx_visiter.h:95` to generate `visit*Inst` method declarations:
```cpp
#define X(enum_val, type_name, opstr, op_count, struct_kind, instr_kind) \
    std::any visit##opstr##Inst(ptxparser::ptxParser::opstr##InstContext *pCtx) override;
#include "ptx_ir/ptx_op.def"
```

Removing the X-macro entry removes the visitor override, leaving the ANTLR-generated `ptxParser` base class's pure virtual method unimplemented → compile error. **Reverted immediately, no commit.**

**Implication**: Must do Steps 2-7 (or at minimum Steps 2 + 4-6) atomically in single commit/branch. Any partial implementation creates broken state.

**Mitigation**:
- Apply Steps 2-7 as single atomic commit (NOT incremental)
- Use feature branch `refactor/t2-4-ptx-87-cleanup` for isolation
- After commit, verify build + sanity + PTX tests in one shot
- If anything breaks, revert entire commit (not partial)
- Keep ANTLR-generated files in `build/antlr4_generated_src/` regenerable from .g4 sources
- Diff generated parser before/after to verify minimal change

## Estimated Effort

- **Atomic Steps 2-7 sequence**: 1-2 working days
  - Step 2 (C++ X-macro removal): 30 min
  - Step 3 (statement_factory.h): 30 min
  - Step 4 (grammar edits): 2-3 hours
  - Step 5 (lexer edits): 30 min
  - Step 6 (ANTLR regen + iterative debugging): 4-8 hours ← **largest risk**
  - Step 7 (test verification): 2-3 hours
- Cannot parallelize any steps (each depends on prior)
- Cannot incrementally commit (atomic only)

## Phase 4 Handoff Notes

When picking up T2-4 Steps 2-7 in Phase 4:
1. Read this document + T2-4 Step 1 commit message + this experimental evidence section
2. Use `git worktree add ../ptx-emu-t2-4 -b refactor/t2-4-ptx-87-cleanup` (note: user 2026-06-23 chose direct push to main — adapt as needed)
3. **CRITICAL**: Apply Steps 2-7 atomically in single feature branch — DO NOT attempt incremental commits
4. Verify in one shot: build + sanity.sh + test_all_ptx.sh
5. Single commit for atomicity (commit message: `refactor(ptx): remove PTX 8.7+ placeholder instructions (T2-4 Steps 2-7)`)
6. If any step fails, revert entire commit and debug

## Why This Document Exists

T2-4 Step 1 commit (`2e339ea`) message references this roadmap file but it was never created. This document fills that gap with:
- Complete Steps 2-7 scope breakdown
- Critical experimental evidence that incremental work breaks build (2026-06-24)
- Atomicity requirement based on actual build verification
- Concrete handoff notes for Phase 4 pickup

## Cross-References

- Master plan: `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` (line 37, 187, 221)
- Health audit: `docs/audits/HEALTH-AUDIT-2026-06-21.md` line 33, `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` line 118
- T2-4 Step 1: commit `2e339ea`
- PTX op definitions: `include/ptx_ir/ptx_op.def` §13-15 (lines 164-180)
- Grammar: `src/grammar/ptxInstructions.g4` lines 461-499
- Lexer: `src/grammar/ptxLexer.g4` lines 387-388

## Pre-Phase 4 Prerequisite: Missing Infrastructure

**⚠️ `docs/ptx/` directory does NOT exist (verified 2026-06-24)**

The `ptx-grammar-modification` skill (`.opencode/skills/ptx-grammar-modification/SKILL.md`) mandates:

> "□ 2. 阅读 docs/ptx/ 对应章节"

Before any grammar modification, the corresponding PTX documentation section MUST be read. For T2-4 Steps 2-7, this would mean:
- `docs/ptx/st-async.md` (PTX 8.7+ st.async instruction spec)
- `docs/ptx/red-async.md` (PTX 8.7+ red.async instruction spec)
- `docs/ptx/tcgen05.md` (PTX 9.0+ tcgen05.mma family spec)
- `docs/ptx/tensormap.md` (PTX 8.7+ tensormap.replace instruction spec)

**None of these docs exist** — `docs/ptx/` is completely missing. This means the proper ptx-grammar-modification TDD workflow **cannot be followed** without first creating this infrastructure.

**Implication for Phase 4 pickup**:
1. Phase 4 must begin by creating `docs/ptx/` directory + extracting relevant PTX 8.7+ spec sections (from NVIDIA PTX ISA reference at https://docs.nvidia.com/cuda/parallel-thread-execution/)
2. THEN follow TDD workflow per ptx-grammar-modification skill: add negative test case → modify grammar → regenerate → verify
3. THEN proceed with Steps 2-7 atomic removal

**Estimated additional Phase 4 effort for docs/ptx/**:
- Create directory structure: 30 min
- Extract PTX 8.7+ st.async/red.async sections: 2-3 hours
- Extract PTX 9.0+ tcgen05.* sections: 4-6 hours (most complex)
- Extract PTX 8.7+ tensormap.replace section: 1-2 hours
- Total: ~1 working day BEFORE T2-4 Steps 2-7 can begin

This finding further validates the original "multi-day" assessment and confirms T2-4 Steps 2-7 should be deferred to a dedicated Phase 4 session with proper preparation.

## Summary: Why This Document Has 4 Commits of Evidence

This roadmap file has been incrementally expanded as skeptical re-examination revealed new facts:

| Commit | Finding | Implication |
|--------|---------|-------------|
| `b2a0a75` | Initial scope breakdown (10 X-macros, 4 files) | Baseline scope |
| `21a3187` | Experimental build break (Step 2 alone) | Atomicity proven |
| `c811b6b` | Orphan types in statement_context.h:223,231 + ptxir_writer.cpp:371,385 | Scope +1 step (Step 8 orphan cleanup) |
| (pending) | `docs/ptx/` directory missing | Phase 4 prerequisite ~1 day extra effort |

**Total Phase 4 effort estimate** (revised):
- docs/ptx/ infrastructure: 1 day
- Steps 2-7 atomic removal: 1-2 days (master plan)
- Step 8 orphan cleanup: 0.5 day
- Verification + iterative debugging: 1 day
- **Total: 3.5-4.5 working days**

This revised estimate (4 days vs original 1-2 days) strengthens the case for dedicated Phase 4 session rather than attempting within Phase 3 closeout.