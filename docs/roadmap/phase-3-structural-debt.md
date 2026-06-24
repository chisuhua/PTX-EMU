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

## Risk Assessment

**Low risk** because:
- Zero test coverage means no test regression
- Zero handlers means no dispatch regression
- Zero consumers means no runtime regression
- Only risk is **parser regeneration** introducing subtle grammar conflicts

**Mitigation**:
- Do Steps 4-6 in isolation (single feature branch)
- Run full PTX syntax test suite after each .g4 edit
- Keep ANTLR-generated files in `build/antlr4_generated_src/` regenerable from .g4 sources
- Diff generated parser before/after to verify minimal change

## Estimated Effort

- Steps 2-3: 1 hour (mechanical removal from C++ files)
- Step 4: 2-3 hours (grammar understanding + careful edits)
- Step 5: 30 minutes (lexer rule removal)
- Step 6: 4-8 hours (ANTLR regen + iterative debugging)
- Step 7: 2-3 hours (test verification)

**Total: 1-2 working days** (matches "multi-day" assessment in master plan)

## Phase 4 Handoff Notes

When picking up T2-4 Steps 2-7 in Phase 4:
1. Read this document + T2-4 Step 1 commit message
2. Use `git worktree add ../ptx-emu-t2-4 -b refactor/t2-4-ptx-87-cleanup` (per master plan §worktree guidance, but note user 2026-06-23 chose direct push to main — adapt as needed)
3. Apply Steps 2-3 first (C++ cleanup, safe), verify build
4. Apply Step 4 (grammar), regenerate parser, verify sanity
5. Apply Step 5 (lexer), regenerate, verify sanity
6. Apply Step 6 (regen), run full regression
7. Apply Step 7 (tests + verify)
8. Single commit or split per step (recommend single commit for atomicity)

## Cross-References

- Master plan: `docs/superpowers/plans/2026-06-23-phase3-critical-debt.md` (line 37, 187, 221)
- Health audit: `docs/audits/HEALTH-AUDIT-2026-06-21.md` line 33, `docs/audits/HEALTH-AUDIT-2026-06-21-ERRATA.md` line 118
- T2-4 Step 1: commit `2e339ea`
- PTX op definitions: `include/ptx_ir/ptx_op.def` §13-15 (lines 164-180)
- Grammar: `src/grammar/ptxInstructions.g4` lines 461-499
- Lexer: `src/grammar/ptxLexer.g4` lines 387-388