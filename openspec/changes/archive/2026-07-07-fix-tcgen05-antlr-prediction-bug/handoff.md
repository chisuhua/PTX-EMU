# Handoff: fix-tcgen05-antlr-prediction-bug → future changes

Generated on 2026-07-07 for change `fix-tcgen05-antlr-prediction-bug`.

## Scope Status (post-implementation)

| Item | Status |
|------|--------|
| ANTLR Kleene star prediction fix | ✅ Resolved (recursive `tcgen05QualList` rule) |
| Qualifier ordering permutations | ✅ Supported per PTX ISA §9.7.16 |
| `tcgen05_permutations.ptx` regression coverage | ✅ Added (8 permutation cases) |
| lessons-learned §23 ANTLR Kleene Star 预测陷阱 | ✅ Documented |
| Change-3a D1 workaround reordering | ✅ Resolved (4 fixtures can restore natural qualifier order) |

## Deferred Items (for future changes)

| # | Item | File(s) | Gate condition |
|---|------|---------|----------------|
| 1 | Restore natural qualifier order in 4 workaround fixtures | `tests/ptx/tcgen05_ld.ptx`, `tcgen05_st.ptx`, `tcgen05_cp.ptx`, `tcgen05_cp_multicast.ptx` | All 4 must continue to PASS after reordering (no reordering workaround needed post-fix) |
| 2 | Apply same Kleene star fix pattern to other grammar rules with similar `(X? Y)*` patterns | TBD (audit) | If similar LL(*) bugs are discovered in wmma/other instruction families |
| 3 | Consider Option B (lexer mode) for complex future qualifiers | TBD | If Option A recursive rewrite proves insufficient for future qualifier expansions |

## Reverse Cross-Reference

Add to any future `fix-*/refactor-*` changes touching ANTLR grammar:

```bash
# Verify no regression of fixed Kleene star bug
./tests/ptx/test_all_ptx.sh  # 50/50 PASS expected post-fix
./tests/ptx/tcgen05_permutations.ptx  # 8/8 PASS expected

# Verify lessons-learned §23 cited
grep -c "Kleene Star\|ANTLR.*预测" docs/dev-process/lessons-learned.md  # ≥1 match
```

## Risk Note

If a future change modifies `src/grammar/ptxInstructions.g4` and the
`tcgen05QualList` recursive rule is reverted to `(X Y)*` Kleene star form,
the LL(*) prediction bug may resurface. Mitigation:
1. `tcgen05_permutations.ptx` serves as the regression guard
2. Pre-implementation Metis review must include "grammar LL(*) prediction" check
3. lessons-learned §23 documents the failure mode

## Status Tracking

- 2026-07-07: Created (proposal-only artifact, implementation pending)
